"""PPO for tag: the car's brain (two outputs plus how unsure it is, ppo_car.py) learning to
chase (or, later, to run) against an opponent that plays inside the same compiled step.

Episode 6: the chaser learns, the runner is a script. Episode 7 (self-play): the opponent can
also be a frozen brain, or a pool of them ("runs/a:175,runs/b:200"): round i of the 2048
always plays pool member i % K, so every past generation gets the same share of practice.

Everything (all rounds playing T decisions, advantages, learning) is one compiled
function, data-parallel over every device. Checkpoints are torch files of the
weights, like the car's, so the usual tooling picks them up.
"""

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_car import forward, init_params, load, log_prob
from ailearns.agents.ppo_jax import adam_init, adam_step, to_torch
from ailearns.games import tag
from ailearns.games.tag import CHASER, RUNNER, TRAIN_SEEDS, TagConfig, TagGame, make_arenas

OPPONENTS = {"flee": tag.runner_flee, "smart": tag.runner_smart, "still": tag.runner_still,
             "naive": tag.chaser_naive}


@dataclass
class TagPPOConfig:
    total_steps: int = 50_000_000
    side: str = "chaser"         # who learns
    opponent: str = "flee"       # who it plays against: a script (OPPONENTS) or brains "runs/a:175,runs/b:200"
    closer_reward: float = 0.0   # reward per metre closer (TagConfig.closer_reward); 0 = catches only
    init_from: str = ""          # start from another run's brain, e.g. "runs/tag-v2-closer:190"
    remember_heading: bool = False  # TagConfig.remember_heading
    reset_std: bool = False      # with init_from: experiment again (std back to exp(init_log_std))
    n_envs: int = 2048
    n_steps: int = 128
    epochs: int = 4
    minibatch: int = 8192
    lr: float = 3e-4
    gamma: float = 0.99          # values ~10 s ahead
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    init_log_std: float = -0.5
    seed: int = 0
    arenas: int = len(TRAIN_SEEDS)
    early_checkpoints: tuple = (0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75)
    checkpoint_every: int = 25


def brain(params: dict, who: int, sample: bool = False):
    """A brain as a player (game, state, key) -> action: its best guess, or (sample) a try
    around it like in training (footage of untrained brains: a best guess near zero just stands still)."""
    def play(game: TagGame, st, key=None):
        mean, log_std, _ = forward(params, game.observe(st, who)[None])
        a = mean[0]
        if sample:
            a = a + jnp.exp(log_std) * jax.random.normal(key, a.shape)
        return jnp.clip(a, -1, 1)
    return play


def is_script(spec: str) -> bool:
    return spec in OPPONENTS


def load_pool(spec: str) -> dict:
    """"runs/a:175,runs/b:200" -> their weights stacked (K, ...); all must be the same size."""
    from ailearns.checkpoints import pick_checkpoint
    members = []
    for one in spec.split(","):
        run, _, ck = one.strip().partition(":")
        members.append(load(pick_checkpoint(run, ck or "last"))[0])
    return jax.tree.map(lambda *xs: jnp.stack(xs), *members)


def pool_player(pool: dict, who: int):
    """A pool of frozen brains as one player (game, state, key, slot) -> action: member slot % K
    plays (best guess, as in the tests). Runs every member and keeps one: cheap for a few members."""
    K = jax.tree.leaves(pool)[0].shape[0]

    def play(game: TagGame, st, key, slot):
        obs = game.observe(st, who)[None]
        means = jax.vmap(lambda p: forward(p, obs)[0][0])(pool)  # (K, 2)
        return jnp.clip(means[slot % K], -1, 1)
    return play


def player(spec: str, who: int):
    """A script name or brain(s) -> player (game, state, key) -> action (a pool plays its first member)."""
    if is_script(spec):
        return OPPONENTS[spec]
    pool = pool_player(load_pool(spec), who)
    return lambda game, st, key=None: pool(game, st, key, 0)


def load_player(spec: str, who: int, sample: bool = False):
    """A script name or "runs/x:ckpt" -> (player (game, state, key) -> action, the game settings its
    brain was trained with, or None for a script)."""
    if is_script(spec):
        return OPPONENTS[spec], None
    from ailearns.checkpoints import pick_checkpoint
    run, _, ck = spec.partition(":")
    params, meta = load(pick_checkpoint(run, ck or "last"))
    return brain(params, who, sample), meta.get("tag")


def game_config(*cfgs) -> TagConfig:
    """The game settings for a round: the first brain's (scripts have none), without the training hint."""
    c = next((c for c in cfgs if c), None)
    return replace(TagConfig(**c) if c else TagConfig(), closer_reward=0.0)


def tries(run: str | Path, update: int) -> int:
    """Rounds played to reach `update` of `run`, counting the run it started from (--init-from)."""
    rounds = 0
    for line in open(Path(run) / "metrics.jsonl"):
        r = json.loads(line)
        if r["update"] > update:
            break
        rounds = r["rounds"]
    meta = json.loads((Path(run) / "config.json").read_text())
    if meta.get("init_from"):
        src, _, ck = meta["init_from"].partition(":")
        root = Path(run).parent.parent  # runs/<name> -> repo
        rounds += tries(root / src, int(ck) if ck else 10 ** 9)
    return rounds


# ── training ────────────────────────────────────────────────────────────────

def make_update(cfg: TagPPOConfig, game: TagGame, n_local: int, n_dev: int):
    T = cfg.n_steps
    mb = cfg.minibatch // n_dev
    n_mb = (T * n_local) // mb
    me = CHASER if cfg.side == "chaser" else RUNNER
    sign = 1.0 if me == CHASER else -1.0  # rewards are the chaser's
    if is_script(cfg.opponent):
        script = OPPONENTS[cfg.opponent]
        opponent = lambda game, st, k, slot: script(game, st, k)  # noqa: E731
    else:
        opponent = pool_player(load_pool(cfg.opponent), 1 - me)
    vobs = jax.vmap(lambda s: game.observe(s, me))
    vreset = jax.vmap(game.reset_if)

    def play(st, mine, k, slot):
        theirs = opponent(game, st, k, slot)
        acts = jnp.stack([mine, theirs]) if me == CHASER else jnp.stack([theirs, mine])
        return game.step(st, acts)
    vplay = jax.vmap(play)
    slots = jnp.arange(n_local)  # which pool member each round plays (the same on every device)

    def loss_fn(params, b):
        mean, log_std, value = forward(params, b["obs"])
        logp = log_prob(mean, log_std, b["act"])
        ratio = jnp.exp(logp - b["logp"])
        a = b["adv"]
        a = (a - a.mean()) / (a.std(ddof=1) + 1e-8)
        pg = jnp.maximum(-a * ratio, -a * jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
        vl = 0.5 * ((value - b["ret"]) ** 2).mean()
        ent = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
        loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
        return loss, {"pg_loss": pg, "v_loss": vl, "entropy": ent, "std": jnp.exp(log_std).mean(),
                      "clipfrac": (jnp.abs(ratio - 1) > cfg.clip).mean()}

    def update(params, opt, env, obs, key, lr):
        key, k_roll, k_learn = jax.random.split(key, 3)
        dev = lax.axis_index("d")
        k_roll, k_learn = jax.random.fold_in(k_roll, dev), jax.random.fold_in(k_learn, dev)

        def tick(carry, k):
            env, obs = carry
            mean, log_std, value = forward(params, obs)
            ka, ko, kr = jax.random.split(k, 3)
            action = mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
            logp = log_prob(mean, log_std, action)
            env, reward = vplay(env, action, jax.random.split(ko, n_local), slots)
            done, caught, seconds = env.done, env.caught, env.t * game.cfg.dt  # read before the reset
            env = vreset(env, jax.random.split(kr, n_local))
            return (env, vobs(env)), {"obs": obs, "act": action, "logp": logp, "val": value, "rew": sign * reward,
                                      "done": done, "caught": caught, "seconds": seconds}
        (env, obs_next), tr = lax.scan(tick, (env, obs), jax.random.split(k_roll, T))

        _, _, next_value = forward(params, obs_next)

        def gae(carry, x):
            last, nv = carry
            nonterminal = 1.0 - x["done"].astype(jnp.float32)
            delta = x["rew"] + cfg.gamma * nv * nonterminal - x["val"]
            last = delta + cfg.gamma * cfg.lam * nonterminal * last
            return (last, x["val"]), last
        _, adv = lax.scan(gae, (jnp.zeros_like(next_value), next_value), tr, reverse=True)
        batch = {"obs": tr["obs"], "act": tr["act"], "logp": tr["logp"], "adv": adv, "ret": adv + tr["val"]}
        batch = jax.tree.map(lambda x: x.reshape(T * n_local, *x.shape[2:]), batch)

        def epoch(carry, k):
            def step(carry, idx):
                params, opt = carry
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params,
                                                                            jax.tree.map(lambda x: x[idx], batch))
                grads = lax.pmean(grads, "d")
                norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
                scale = jnp.minimum(1.0, cfg.max_grad_norm / (norm + 1e-6))
                params, opt = adam_step(params, opt, jax.tree.map(lambda g: g * scale, grads), lr)
                return (params, opt), aux
            perm = jax.random.permutation(k, T * n_local)[: n_mb * mb].reshape(n_mb, mb)
            return lax.scan(step, carry, perm)
        (params, opt), aux = lax.scan(epoch, (params, opt), jax.random.split(k_learn, cfg.epochs))

        ended, caught = tr["done"], tr["caught"]
        m = {"ended": ended.sum(), "caught": caught.sum(),
             "catch_seconds": jnp.where(caught, tr["seconds"], 0).sum()}
        m = jax.tree.map(lambda x: lax.psum(x, "d"), m)
        m.update(jax.tree.map(lambda x: lax.pmean(x.mean(), "d"), aux))
        return params, opt, env, obs_next, key, m

    return update


def train(cfg: TagPPOConfig, run_dir: str | Path, resume: bool = False, max_hours: float | None = None,
          game_cfg: TagConfig = TagConfig()) -> None:
    import torch
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    n_dev = len(devices)
    if cfg.n_envs % n_dev or cfg.minibatch % n_dev:
        raise SystemExit(f"--envs and minibatch must divide by the {n_dev} devices")
    game_cfg = replace(game_cfg, closer_reward=cfg.closer_reward, remember_heading=cfg.remember_heading)
    sha = _git_version()
    state = None
    if resume:
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        meta = json.loads((run_dir / "config.json").read_text())
        meta["total_steps"] = cfg.total_steps
        meta.setdefault("resumed", []).append({"update": state["update"], "git": sha, "device": str(devices[0])})
        (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    else:
        if (run_dir / "metrics.jsonl").exists():
            raise SystemExit(f"{run_dir} already has a run; use --resume or a new directory")
        (run_dir / "config.json").write_text(json.dumps(
            {**asdict(cfg), "game": "tag", "tag": asdict(game_cfg), "git": sha, "backend": "jax",
             "device": f"{n_dev} x {devices[0].device_kind}"}, indent=2))
    first = state["update"] + 1 if state else 1
    print(f"JAX on {n_dev} x {devices[0].device_kind}; building {cfg.arenas} arenas …", flush=True)

    game = TagGame(make_arenas(list(TRAIN_SEEDS)[: cfg.arenas], game_cfg), game_cfg)
    me = CHASER if cfg.side == "chaser" else RUNNER
    if state:
        params = {k: jnp.asarray(v.numpy()) for k, v in state["model"].items()}
        opt = jax.tree.map(jnp.asarray, state["optimizer"])
    elif cfg.init_from:
        from ailearns.checkpoints import pick_checkpoint
        run, _, ck = cfg.init_from.partition(":")
        params, _ = load(pick_checkpoint(run, ck or "last"))
        if cfg.reset_std:
            params["log_std"] = jnp.full_like(params["log_std"], cfg.init_log_std)
        opt = adam_init(params)
    else:
        params = init_params(jax.random.key(cfg.seed), game_cfg.obs_size, cfg.hidden, cfg.init_log_std)
        opt = adam_init(params)

    mesh = Mesh(np.array(devices), ("d",))
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    T, N = cfg.n_steps, cfg.n_envs
    update = make_update(cfg, game, N // n_dev, n_dev)
    update = jax.jit(jax.shard_map(update, mesh=mesh, in_specs=(P(), P(), P("d"), P("d"), P(), P()),
                                   out_specs=(P(), P(), P("d"), P("d"), P(), P())),
                     donate_argnums=(0, 1, 2, 3))
    key = jax.random.key(cfg.seed + 1000 + first)
    key, k_env = jax.random.split(key)
    env = jax.jit(jax.vmap(game.init), out_shardings=shard)(jax.random.split(k_env, N))
    obs = jax.jit(jax.vmap(lambda s: game.observe(s, me)), out_shardings=shard)(env)
    params, opt, key = jax.device_put((params, opt, key), rep)

    n_updates = cfg.total_steps // (T * N)
    rounds = state["rounds"] if state else 0
    start = time.time() - (state["elapsed"] if state else 0.0)
    session_start = time.time()
    log = open(run_dir / "metrics.jsonl", "a")
    from ailearns.agents import tracking
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))
    meta = {"config": asdict(cfg), "tag": asdict(game_cfg), "backend": "jax", "game": "tag"}

    def save(update_no: int, stats: dict) -> None:
        torch.save({"model": to_torch(params), **meta, "update": update_no, "steps": update_no * T * N,
                    "rounds": rounds, "stats": stats}, run_dir / "checkpoints" / f"u{update_no:05d}.pt")

    def save_latest(update_no: int) -> None:
        torch.save({"model": to_torch(params), "optimizer": jax.tree.map(np.asarray, opt), **meta,
                    "update": update_no, "rounds": rounds, "elapsed": time.time() - start},
                   run_dir / "latest.pt.tmp")
        (run_dir / "latest.pt.tmp").replace(run_dir / "latest.pt")

    if not state:
        save(0, {})
    for update_no in range(first, n_updates + 1):
        lr = jnp.float32(cfg.lr * (1 - (update_no - 1) / n_updates))
        params, opt, env, obs, key, m = update(params, opt, env, obs, key, lr)
        m = jax.device_get(m)
        ended, caught = int(m["ended"]), int(m["caught"])
        rounds += ended
        stats = {
            "update": update_no, "steps": update_no * T * N, "rounds": rounds, "time": round(time.time() - start, 1),
            "catch_rate": caught / ended if ended else None,
            "catch_seconds": float(m["catch_seconds"]) / caught if caught else None,
            "pg_loss": float(m["pg_loss"]), "v_loss": float(m["v_loss"]), "entropy": float(m["entropy"]),
            "std": float(m["std"]), "clipfrac": float(m["clipfrac"]),
        }
        log.write(json.dumps(stats) + "\n")
        log.flush()
        tracking.log(tracker, stats)
        out_of_time = max_hours is not None and time.time() - session_start > max_hours * 3600
        if update_no in cfg.early_checkpoints or update_no % cfg.checkpoint_every == 0 or update_no == n_updates \
                or out_of_time:
            save(update_no, stats)
            save_latest(update_no)
        if update_no % 10 == 0 or update_no == first:
            sps = (update_no - first + 1) * T * N / (time.time() - session_start)
            cr, cs = stats["catch_rate"], stats["catch_seconds"]
            print(f"update {update_no:5d}/{n_updates}  steps {stats['steps']:>11,}  rounds {rounds:>8,}  "
                  f"caught {cr if cr is None else round(cr, 2)!s:>5}  in {cs if cs is None else round(cs, 1)!s:>5} s  "
                  f"std {stats['std']:.2f}  {sps:,.0f} steps/s", flush=True)
        if out_of_time:
            print(f"stopped after {max_hours} h at update {update_no}; carry on with --resume", flush=True)
            break
    log.close()
    tracking.finish(tracker)


# ── evaluation ──────────────────────────────────────────────────────────────

def evaluate(run: str, checkpoint: str = "last", rounds: int = 256, first: int = 0, seed: int = 5000,
             opponent: str | None = None) -> dict:
    """Standard tag test: best-guess play, one round in each of the test arenas first..first+rounds
    (never trained in). Choose checkpoints on arenas 0-63, report from 64-255 (playbook: never report
    a number from the rounds you picked with)."""
    from ailearns.checkpoints import pick_checkpoint
    path = pick_checkpoint(run, str(checkpoint))
    params, ck = load(path)
    cfg = ck["config"]
    game_cfg = TagConfig(**ck["tag"])
    game = TagGame(make_arenas(list(tag.TEST_SEEDS)[first:first + rounds], game_cfg), game_cfg)
    me = CHASER if cfg["side"] == "chaser" else RUNNER
    other = player(opponent or cfg["opponent"], 1 - me)
    players = (brain(params, me), other) if me == CHASER else (other, brain(params, me))
    res = tag.play_rounds(game, *players, rounds, seed=seed)
    return {"run": str(run), "update": ck["update"], "moves": ck["steps"], "rounds": rounds,
            "catch_rate": res["catch_rate"], "catch_time": res["catch_time"], "result": res}


def match(chaser: str, runner: str, rounds: int = 192, first: int = 64, seed: int = 5000,
          game_cfg: TagConfig | None = None) -> dict:
    """Any chaser vs any runner (scripts or "runs/x:ckpt"), one best-guess round per test arena:
    one cell of the tournament table. Game settings: the first brain's (they must agree)."""
    if game_cfg is None:
        from ailearns.checkpoints import pick_checkpoint
        cfgs = []
        for spec in (chaser, runner):
            if not is_script(spec):
                run, _, ck = spec.split(",")[0].partition(":")
                cfgs.append(load(pick_checkpoint(run, ck or "last"))[1]["tag"])
        game_cfg = TagConfig(**cfgs[0]) if cfgs else TagConfig()
    game = TagGame(make_arenas(list(tag.TEST_SEEDS)[first:first + rounds], game_cfg), game_cfg)
    res = tag.play_rounds(game, player(chaser, CHASER), player(runner, RUNNER), rounds, seed=seed)
    return {"catch_rate": res["catch_rate"], "catch_time": res["catch_time"], "result": res}
