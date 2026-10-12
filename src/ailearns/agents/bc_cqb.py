"""Imitation for CQB (ep9 3bb): show the brain the scripted flash-and-enter, then let it practise.

Eight tries to find the flashbang from wins alone failed (PPO 3x-3ab, evolution 3az-3ba). "Shown the ropes" (the
owner's words): record the scripted walker_flash's winning rounds and train the brain to do what it did in them
(behaviour cloning), starting from a brain that already fights well. Then PPO from there (the usual trainer,
--init-from this run) to see whether practice keeps the flashbang or unlearns it.

- Demonstrations: walker_flash as Assault against the training opponents, on training buildings, a share of them
  entry drills (where its flashbang pays most). Kept: every decision of the rounds it WON, as (what Assault saw,
  what the script did). The script steers by the floor plan, which the brain can't see, so the copy can't be exact.
- Training: the brain's move means towards the script's moves (squared error), its on/off switches towards the
  script's (cross-entropy); a throw is rare (a few decisions in a round), so those decisions count `throw_weight`
  times. Only the decision layers and outputs change; the value head is left for PPO to fit.
- Saved like a PPO run (checkpoints/uNNNNN.pt = after epoch N, u00000 = the start), so cqb_eval's report and
  `cloud train --init-from runs/<this>:N` work as they are.
"""

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import ppo_cqb as A
from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_jax import adam_init, adam_step, to_torch
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S
from ailearns.games.cqb import ASSAULT, DEFENSE, TRAIN_SEEDS, CQBConfig, CQBGame, make_buildings

TRAINED = ("body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias", "policy.weight", "policy.bias")


@dataclass
class CQBBCConfig:
    side: str = "assault"
    demo: str = "walker_flash"   # the script whose winning rounds are copied
    init_from: str = ""          # the brain it starts from (runs/x:update)
    opponent: str = "camper"     # who the script plays in the demonstrations (rounds shared out among them)
    time_cost: float = 0.0       # the game's rules, as in CQBPPOConfig
    flashbangs: tuple = (0, 0)
    magazine: int = 0
    health: float = 0.0
    react: float = 0.0
    demo_rounds: int = 6144      # rounds the script plays (only its wins are kept)
    stacked: float = 0.5         # this share of them are entry drills
    epochs: int = 20
    batch: int = 4096
    lr: float = 3e-4
    throw_weight: float = 20.0   # a throw decision counts this much in the switch loss
    throw_only: bool = False     # copy only the throw decision: train just the throw output's row, so the brain's
                                 # moving, shooting and doors stay exactly as they were (bc1 showed that copying the
                                 # script's walking wrecks it: ep9 3bb)
    buildings: int = 1024
    seed: int = 0
    method: str = "bc"


def record(game: CQBGame, me: int, script, opp, bis, keys, drill):
    """Rounds in buildings bis (keys, drill starts) with `script` as `me` against `opp`. -> per round and decision:
    what `me` saw, what the script did, whether that decision was before the round ended; and per round its result
    (for `me`: +1 won)."""
    other = 1 - me

    def one(key, bi, d):
        st = jax.tree.map(lambda a, b: jnp.where(d, a, b), game.init_stacked(key, bi), game.init(key, bi))

        def tick(st, k):
            ka, kb = jax.random.split(k)
            obs = game.observe(st, me)
            mine = cqb.full_action(script(game, st, me, ka))
            theirs = cqb.full_action(opp(game, st, other, kb))
            new, _ = game.step(st, jnp.stack([mine, theirs]) if me == ASSAULT else jnp.stack([theirs, mine]))
            return new, (obs, mine, ~st.done)

        last, (obs, act, live) = jax.lax.scan(tick, st, jax.random.split(jax.random.fold_in(key, 1), game.T))
        return obs, act, live, last.outcome * (1 if me == ASSAULT else -1)

    return jax.vmap(one)(keys, bis, drill)


def demonstrations(cfg: CQBBCConfig, game: CQBGame, me: int, chunk: int = 512) -> dict:
    """The script's winning rounds, every decision: obs (N, obs), act (N, 8), and some numbers about them."""
    opponents = A.pool_members(cfg.opponent, 1 - me)
    script = S.SCRIPTS[cfg.demo]
    play = jax.jit(lambda opp_i, bis, keys, drill: record(game, me, script, opponents[opp_i], bis, keys, drill),
                   static_argnums=0)
    obs, act, n_rounds, n_won, n_throws, per_opp = [], [], 0, 0, 0, {}
    per = cfg.demo_rounds // len(opponents)
    key = jax.random.PRNGKey(cfg.seed + 3000)
    for i, name in enumerate(cfg.opponent.split(",")):
        won_i = 0
        for start in range(0, per, chunk):
            n = min(chunk, per - start)
            k = jax.random.fold_in(key, i * 100_000 + start)
            kb, kk = jax.random.split(k)
            bis = jax.random.randint(kb, (n,), 0, cfg.buildings)
            drill = jnp.arange(n) < int(round(cfg.stacked * n))
            o, a, live, result = jax.device_get(play(i, bis, jax.random.split(kk, n), drill))
            won = result > 0
            keep = live & won[:, None]
            obs.append(o[keep])
            act.append(a[keep])
            n_rounds += n
            n_won += int(won.sum())
            won_i += int(won.sum())
            n_throws += int(((a[..., cqb.THROW] > 0) & keep).sum())
        per_opp[name.strip()] = won_i / per
    obs, act = np.concatenate(obs), np.concatenate(act)
    return {"obs": obs, "act": act, "rounds": n_rounds, "won": n_won, "decisions": len(obs),
            "throws_per_won_round": n_throws / max(n_won, 1), "won_by_opponent": per_opp}


def make_step(cfg: CQBBCConfig, n_switch: int):
    """One minibatch of behaviour cloning: (params, opt, obs, act) -> (params, opt, losses). throw_only: the loss is
    the throw decision's alone, and only the throw output's row and bias move (everything else is held fixed)."""
    throw = A.SWITCHES.index("throw")
    row = A.N_MOVE + throw

    def loss_fn(trained, rest, obs, act):
        mean, logits, _, _ = A.forward({**rest, **trained}, obs)
        moves = jnp.mean((mean - act[:, :A.N_MOVE]) ** 2)
        target = (act[:, A.N_MOVE:A.N_MOVE + n_switch] > 0).astype(jnp.float32)
        bce = -(target * jax.nn.log_sigmoid(logits) + (1 - target) * jax.nn.log_sigmoid(-logits))
        w = jnp.ones_like(bce).at[:, throw].set(jnp.where(target[:, throw] > 0, cfg.throw_weight, 1.0))
        if cfg.throw_only:
            w = jnp.zeros_like(bce).at[:, throw].set(w[:, throw])
            moves = 0.0 * moves
        switches = jnp.sum(bce * w) / jnp.sum(w)
        agree = ((logits > 0) == (target > 0)).astype(jnp.float32)
        throw_recall = jnp.sum(agree[:, throw] * target[:, throw]) / jnp.maximum(target[:, throw].sum(), 1)
        return moves + switches, (moves, switches, throw_recall)

    def step(params, opt, obs, act):
        trained = {n: params[n] for n in TRAINED}
        rest = {k: v for k, v in params.items() if k not in TRAINED}
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(trained, rest, obs, act)
        if cfg.throw_only:   # nothing but the throw output's row moves
            keep = {n: jnp.zeros_like(g) for n, g in grads.items()}
            keep["policy.weight"] = keep["policy.weight"].at[row].set(grads["policy.weight"][row])
            keep["policy.bias"] = keep["policy.bias"].at[row].set(grads["policy.bias"][row])
            grads = keep
        trained, opt = adam_step(trained, opt, grads, cfg.lr)
        return {**params, **trained}, opt, (loss, *aux)
    return step


def train(cfg: CQBBCConfig, run_dir: str | Path, max_hours: float | None = None,
          game_cfg: CQBConfig = CQBConfig(), resume: bool = False) -> None:
    import torch
    from ailearns.agents import tracking
    from ailearns.checkpoints import pick_checkpoint
    game_cfg = replace(game_cfg, time_cost=cfg.time_cost, flashbangs=tuple(cfg.flashbangs), magazine=cfg.magazine,
                       health=cfg.health, react=cfg.react)
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if resume:   # (cheap enough to do again from the start after a preemption)
        for f in (run_dir / "checkpoints").glob("*.pt"):
            f.unlink()
        (run_dir / "metrics.jsonl").unlink(missing_ok=True)
    elif (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already has a run")
    dev = jax.devices()[0]
    (run_dir / "config.json").write_text(json.dumps(
        {**asdict(cfg), "game": "cqb", "cqb": asdict(game_cfg), "git": _git_version(), "backend": "jax",
         "device": f"1 x {dev.device_kind}"}, indent=2))
    me = ASSAULT if cfg.side == "assault" else DEFENSE
    print(f"BC on {dev.device_kind}; building {cfg.buildings} buildings …", flush=True)
    game = CQBGame(make_buildings(list(TRAIN_SEEDS)[: cfg.buildings], game_cfg), game_cfg)
    t0 = time.time()
    d = demonstrations(cfg, game, me)
    info = {k: v for k, v in d.items() if k not in ("obs", "act")}
    print(f"demonstrations: {cfg.demo} won {d['won']} of {d['rounds']} rounds "
          f"({', '.join(f'{k}: {v:.0%}' for k, v in d['won_by_opponent'].items())}); kept {d['decisions']:,} decisions, "
          f"{d['throws_per_won_round']:.2f} throws per won round ({time.time() - t0:.0f} s)", flush=True)
    (run_dir / "demos.json").write_text(json.dumps(info, indent=2))

    n_switch = A.switches_for(game_cfg, me)
    run, _, ck = cfg.init_from.partition(":")
    params = A.widen(A.load(pick_checkpoint(run, ck or "last"))[0], game_cfg.obs_size, n_switch)
    n_in = params["body.0.weight"].shape[1]
    meta = {"config": asdict(cfg), "cqb": asdict(game_cfg), "backend": "jax", "game": "cqb"}

    def save(epoch: int, stats: dict) -> None:
        torch.save({"model": to_torch(params), **meta, "update": epoch, "stats": stats},
                   run_dir / "checkpoints" / f"u{epoch:05d}.pt")

    save(0, {})
    opt = adam_init({n: params[n] for n in TRAINED})
    step = jax.jit(make_step(cfg, n_switch))
    obs = jnp.asarray(d["obs"][:, :n_in])
    act = jnp.asarray(d["act"])
    N = obs.shape[0]
    log = open(run_dir / "metrics.jsonl", "a")
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(N)
        sums = np.zeros(4)
        n_b = 0
        for i in range(0, N - cfg.batch + 1, cfg.batch):
            idx = jnp.asarray(order[i:i + cfg.batch])
            params, opt, out = step(params, opt, obs[idx], act[idx])
            sums += np.asarray(jax.device_get(jnp.stack(out)))
            n_b += 1
        loss, moves, switches, recall = sums / max(n_b, 1)
        stats = {"update": epoch, "loss": loss, "moves_loss": moves, "switch_loss": switches, "throw_recall": recall,
                 "time": round(time.time() - t0, 1)}
        log.write(json.dumps(stats) + "\n")
        log.flush()
        tracking.log(tracker, stats)
        save(epoch, stats)
        print(f"epoch {epoch:3d}/{cfg.epochs}  loss {loss:.3f} (moves {moves:.3f}, switches {switches:.3f})  "
              f"throws copied {recall:.0%}  {stats['time']:.0f} s", flush=True)
        if max_hours is not None and time.time() - t0 > max_hours * 3600:
            break
    log.close()
    tracking.finish(tracker)
