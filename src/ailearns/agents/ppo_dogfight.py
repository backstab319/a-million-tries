"""PPO for the dogfight (ep21), in JAX, compiled end to end. Step 2: the chase.

Plane 0 learns; plane 1 flies a scripted manoeuvre (straight, level turn, weave, a break turn when threatened) and
never shoots. Same algorithm and network as `ppo_rocket.py` (Gaussian policy over the four controls, separate
policy and value MLPs, GAE, minibatch epochs, gradients averaged over devices).

The ladder: at level 0 a fight starts with the learner ~600 m behind the target; each time the smoothed kill rate
reaches `level_up_at` the start box widens a step, up to level 1 (anywhere within ~4 km, any heading).

    python src/ailearns/agents/ppo_dogfight.py --out runs/dogfight-chase1 --steps 300_000_000 --hours 11
"""

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_jax import adam_init, adam_step
from ailearns.agents.ppo_rocket import forward, log_prob, std_of
from ailearns.games.dogfight import ALIVE, CRASHED, N_ACTIONS, SHOT, FighterConfig, Fighters, n_obs


@dataclass
class DogfightPPOConfig:
    total_steps: int = 300_000_000
    n_envs: int = 2048
    n_steps: int = 64
    epochs: int = 4
    minibatch: int = 16384
    lr: float = 3e-4
    gamma: float = 0.995   # ~10 s ahead at 20 steps a second
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: tuple = (256, 256)
    log_std_init: float = -0.5
    max_std: float = 1.0
    seed: int = 0
    level_start: float = 0.0
    level_step: float = 0.05
    level_up_at: float = 0.5
    # self-play (step 3): both seats fly this brain in `self_share` of the fights; the rest fight a frozen past
    # version from the league (a new snapshot every `snapshot_every` updates, up to `league_size`; the league
    # opponent changes every `swap_every` updates). Only this brain's seats are learned from.
    # (2g) "bernoulli": the trigger is its own on/off choice with a learned firing probability (the network's 4th
    # output is its logit). "gauss" (runs before 2g): a threshold on a fourth noisy control, which only fires through
    # the noise, so shooting dies out as the noise shrinks.
    trigger: str = "bernoulli"
    # (3e) a floor under the stick noise: in every run so far it shrank to 0.007 and fights drifted into draws
    min_std: float = 0.0
    mode: str = "chase"
    self_share: float = 0.5
    league_size: int = 16
    snapshot_every: int = 25
    swap_every: int = 10
    early_checkpoints: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75])
    checkpoint_every: int = 50


def init_params(cfg: DogfightPPOConfig, key) -> dict:
    orth = jax.nn.initializers.orthogonal
    params, sizes = {}, (n_obs(), *cfg.hidden)
    for trunk, out, gain in (("pi", N_ACTIONS, 0.01), ("v", 1, 1.0)):
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            key, k = jax.random.split(key)
            params[f"{trunk}{i}.w"] = orth(np.sqrt(2))(k, (a, b))
            params[f"{trunk}{i}.b"] = jnp.zeros(b)
        key, k = jax.random.split(key)
        params[f"{trunk}.out.w"] = orth(gain)(k, (sizes[-1], out))
        params[f"{trunk}.out.b"] = jnp.zeros(out)
    params["log_std"] = jnp.full(N_ACTIONS, cfg.log_std_init)
    return params


# ── the policy: three Gaussian controls (g, roll, throttle) and the trigger ───

def log_std(params, cfg: "DogfightPPOConfig"):
    ls = std_of(params, cfg.max_std)
    return ls if cfg.min_std <= 0 else jnp.maximum(ls, float(np.log(cfg.min_std)))


def sample_action(params, mean, key, cfg: "DogfightPPOConfig"):
    """mean (..., 4) -> an action for the sim (trigger as +1 fire / -1 hold)."""
    k1, k2 = jax.random.split(key)
    std = jnp.exp(log_std(params, cfg))
    if cfg.trigger == "gauss":
        return mean + std * jax.random.normal(k1, mean.shape)
    cont = mean[..., :3] + std[:3] * jax.random.normal(k1, mean[..., :3].shape)
    fire = jax.random.uniform(k2, mean.shape[:-1]) < jax.nn.sigmoid(mean[..., 3])
    return jnp.concatenate([cont, jnp.where(fire, 1.0, -1.0)[..., None]], -1)


def action_logp(params, mean, act, cfg: "DogfightPPOConfig"):
    ls = log_std(params, cfg)
    if cfg.trigger == "gauss":
        return log_prob(mean, ls, act)
    z = mean[..., 3]
    return log_prob(mean[..., :3], ls[:3], act[..., :3]) + jnp.where(act[..., 3] > 0, -jax.nn.softplus(-z),
                                                                     -jax.nn.softplus(z))


def action_entropy(params, mean, cfg: "DogfightPPOConfig"):
    """Per-row entropy: the Gaussian part plus the trigger's coin."""
    ls = log_std(params, cfg)
    if cfg.trigger == "gauss":
        return jnp.broadcast_to((ls + 0.5 * np.log(2 * np.pi * np.e)).sum(), mean.shape[:-1])
    z = mean[..., 3]
    return (ls[:3] + 0.5 * np.log(2 * np.pi * np.e)).sum() + jax.nn.softplus(z) - z * jax.nn.sigmoid(z)


def chase_step(game: Fighters):
    """(state, learner action) -> (state, learner reward, done): plane 1 flies its script."""
    def step(st, act):
        st, r, done = game.step(st, jnp.stack([act, game.scripted_action(st, 1)]))
        return st, r[0], done
    return step


def make_update(cfg: DogfightPPOConfig, game: Fighters, n_local: int, n_dev: int):
    T, L = cfg.n_steps, len(cfg.hidden)
    mb = cfg.minibatch // n_dev
    n_mb = (T * n_local) // mb
    vstep = jax.vmap(chase_step(game))
    vobs = jax.vmap(lambda st: game.observe(st, 0))

    def reset_if(st, done, key, level):
        return jax.tree.map(lambda a, b: jnp.where(done, a, b), game.init(key, level), st)
    vreset = jax.vmap(reset_if, in_axes=(0, 0, 0, None))

    def loss_fn(params, b):
        mean, value = forward(params, b["obs"], L)
        logp = action_logp(params, mean, b["act"], cfg)
        ratio = jnp.exp(logp - b["logp"])
        a = b["adv"]
        a = (a - a.mean()) / (a.std(ddof=1) + 1e-8)
        pg = jnp.maximum(-a * ratio, -a * jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
        vl = 0.5 * ((value - b["ret"]) ** 2).mean()
        ent = action_entropy(params, mean, cfg).mean()
        return pg + cfg.vf_coef * vl - cfg.ent_coef * ent, {
            "pg_loss": pg, "v_loss": vl, "entropy": ent, "clipfrac": (jnp.abs(ratio - 1) > cfg.clip).mean()}

    def update(params, opt, env, obs, key, lr, level):
        key, k_roll, k_learn = jax.random.split(key, 3)
        dev = lax.axis_index("d")
        k_roll, k_learn = jax.random.fold_in(k_roll, dev), jax.random.fold_in(k_learn, dev)

        def play(carry, k):
            env, obs = carry
            mean, value = forward(params, obs, L)
            ka, kr = jax.random.split(k)
            action = sample_action(params, mean, ka, cfg)
            logp = action_logp(params, mean, action, cfg)
            env, reward, done = vstep(env, action)
            end = {"cause0": env.cause[:, 0], "cause1": env.cause[:, 1], "time": env.time,
                   "dealt": 1.0 - env.health[:, 1], "fired": game.cfg.rounds - env.ammo[:, 0]}
            env = vreset(env, done, jax.random.split(kr, n_local), level)
            return (env, vobs(env)), {"obs": obs, "act": action, "logp": logp, "val": value, "rew": reward,
                                      "done": done, **end}
        (env, obs_next), tr = lax.scan(play, (env, obs), jax.random.split(k_roll, T))

        _, next_value = forward(params, obs_next, L)

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

        d = tr["done"]
        m = {"ended": d.sum(), "kills": (d & (tr["cause1"] == SHOT)).sum(),
             "target_crashed": (d & (tr["cause1"] == CRASHED)).sum(),
             "crashes": (d & (tr["cause0"] == CRASHED)).sum(),
             "timeouts": (d & (tr["cause0"] == ALIVE) & (tr["cause1"] == ALIVE)).sum(),
             "len_sum": jnp.where(d, tr["time"], 0).sum(), "rew_sum": tr["rew"].sum(),
             "dealt_sum": jnp.where(d, tr["dealt"], 0).sum(), "fired_sum": jnp.where(d, tr["fired"], 0).sum()}
        m = jax.tree.map(lambda x: lax.psum(x, "d"), m)
        m.update(jax.tree.map(lambda x: lax.pmean(x.mean(), "d"), aux))
        return params, opt, env, obs_next, key, m

    return update


def make_update_self(cfg: DogfightPPOConfig, game: Fighters, n_local: int, n_dev: int):
    """Self-play: obs (n_local, 2, n_obs). Seat 0 always flies `params`; seat 1 flies `params` in the first
    self_share of the fights and the league brain `opp` (its mean action) in the rest, whose seat-1 moves are
    masked out of the learning."""
    T, L = cfg.n_steps, len(cfg.hidden)
    rows = T * n_local * 2
    mb = cfg.minibatch // n_dev
    n_mb = rows // mb
    vstep = jax.vmap(game.step)
    vobs = jax.vmap(lambda st: jnp.stack([game.observe(st, 0), game.observe(st, 1)]))
    is_self = jnp.arange(n_local) < int(round(cfg.self_share * n_local))
    mask = jnp.stack([jnp.ones(n_local), is_self.astype(jnp.float32)], 1)  # (n_local, 2)

    def reset_if(st, done, key, level):
        return jax.tree.map(lambda a, b: jnp.where(done, a, b), game.init(key, level), st)
    vreset = jax.vmap(reset_if, in_axes=(0, 0, 0, None))

    def loss_fn(params, b):
        mean, value = forward(params, b["obs"], L)
        logp = action_logp(params, mean, b["act"], cfg)
        ratio = jnp.exp(logp - b["logp"])
        w = b["mask"]
        n = jnp.maximum(w.sum(), 1.0)
        a = b["adv"]
        mu = (a * w).sum() / n
        a = (a - mu) / (jnp.sqrt(((a - mu) ** 2 * w).sum() / n) + 1e-8)
        pg = (jnp.maximum(-a * ratio, -a * jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)) * w).sum() / n
        vl = 0.5 * (((value - b["ret"]) ** 2) * w).sum() / n
        ent = (action_entropy(params, mean, cfg) * w).sum() / n
        return pg + cfg.vf_coef * vl - cfg.ent_coef * ent, {
            "pg_loss": pg, "v_loss": vl, "entropy": ent,
            "clipfrac": ((jnp.abs(ratio - 1) > cfg.clip) * w).sum() / n}

    def update(params, opt, env, obs, key, lr, level, opp):
        key, k_roll, k_learn = jax.random.split(key, 3)
        dev = lax.axis_index("d")
        k_roll, k_learn = jax.random.fold_in(k_roll, dev), jax.random.fold_in(k_learn, dev)

        def play(carry, k):
            env, obs = carry
            flat = obs.reshape(n_local * 2, -1)
            mean, value = forward(params, flat, L)
            mean, value = mean.reshape(n_local, 2, -1), value.reshape(n_local, 2)
            ka, kr = jax.random.split(k)
            ka, kn = jax.random.split(ka)
            action = sample_action(params, mean, kn, cfg)
            logp = action_logp(params, mean, action, cfg)
            # the league brain flies as it was trained, with its noise: its mean almost never fires (step 2e)
            ko, ka = jax.random.split(ka)
            opp_mean, _ = forward(opp, obs[:, 1], L)
            opp_act = sample_action(opp, opp_mean, ko, cfg)
            played = action.at[:, 1].set(jnp.where(is_self[:, None], action[:, 1], opp_act))
            env, reward, done = vstep(env, played)
            end = {"cause0": env.cause[:, 0], "cause1": env.cause[:, 1], "time": env.time,
                   "fired": game.cfg.rounds - env.ammo[:, 0]}
            env = vreset(env, done, jax.random.split(kr, n_local), level)
            return (env, vobs(env)), {"obs": obs, "act": action, "logp": logp, "val": value, "rew": reward,
                                      "done": done, **end}
        (env, obs_next), tr = lax.scan(play, (env, obs), jax.random.split(k_roll, T))

        _, next_value = forward(params, obs_next.reshape(n_local * 2, -1), L)
        next_value = next_value.reshape(n_local, 2)

        def gae(carry, x):
            last, nv = carry
            nonterminal = 1.0 - x["done"].astype(jnp.float32)[:, None]
            delta = x["rew"] + cfg.gamma * nv * nonterminal - x["val"]
            last = delta + cfg.gamma * cfg.lam * nonterminal * last
            return (last, x["val"]), last
        _, adv = lax.scan(gae, (jnp.zeros_like(next_value), next_value), tr, reverse=True)
        batch = {"obs": tr["obs"], "act": tr["act"], "logp": tr["logp"], "adv": adv, "ret": adv + tr["val"],
                 "mask": jnp.broadcast_to(mask, adv.shape)}
        batch = jax.tree.map(lambda x: x.reshape(rows, *x.shape[3:]), batch)

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
            perm = jax.random.permutation(k, rows)[: n_mb * mb].reshape(n_mb, mb)
            return lax.scan(step, carry, perm)
        (params, opt), aux = lax.scan(epoch, (params, opt), jax.random.split(k_learn, cfg.epochs))

        d = tr["done"]
        lg = d & ~is_self[None]          # fights against the league, seen from seat 0 (this brain)
        sf = d & is_self[None]
        m = {"ended": d.sum(), "league_ended": lg.sum(),
             "league_wins": (lg & (tr["cause1"] != ALIVE) & (tr["cause0"] == ALIVE)).sum(),
             "league_losses": (lg & (tr["cause0"] != ALIVE) & (tr["cause1"] == ALIVE)).sum(),
             "kills": (d & ((tr["cause0"] == SHOT) | (tr["cause1"] == SHOT))).sum(),
             "crashes": (d & ((tr["cause0"] == CRASHED) | (tr["cause1"] == CRASHED))).sum(),
             "timeouts": (d & (tr["cause0"] == ALIVE) & (tr["cause1"] == ALIVE)).sum(),
             "self_ended": sf.sum(),
             "len_sum": jnp.where(d, tr["time"], 0).sum(), "fired_sum": jnp.where(d, tr["fired"], 0).sum()}
        m = jax.tree.map(lambda x: lax.psum(x, "d"), m)
        m.update(jax.tree.map(lambda x: lax.pmean(x.mean(), "d"), aux))
        return params, opt, env, obs_next, key, m

    return update


def train(cfg: DogfightPPOConfig, run_dir: str | Path, game_cfg: FighterConfig | None = None, resume: bool = False,
          max_hours: float | None = None, init_from: str | None = None, reset_std: bool = False) -> None:
    """init_from: a checkpoint whose network starts the run (self-play from the chase pilot); fresh optimizer."""
    selfplay = cfg.mode == "self"
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    n_dev = len(devices)
    if cfg.n_envs % n_dev or cfg.minibatch % n_dev:
        raise SystemExit(f"--envs and minibatch must divide by the {n_dev} devices")
    game_cfg = game_cfg or FighterConfig()
    state = None
    if resume:
        state = pickle.loads((run_dir / "latest.pkl").read_bytes())
    elif (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already has a run; use --resume or a new directory")
    else:
        (run_dir / "config.json").write_text(json.dumps(
            {**asdict(cfg), "game": asdict(game_cfg), "git": _git_version(), "backend": "jax", "init_from": init_from, "reset_std": reset_std,
             "device": f"{n_dev} x {devices[0].device_kind}"}, indent=2, default=float))
    first = state["update"] + 1 if state else 1
    print(f"JAX on {n_dev} x {devices[0].device_kind}", flush=True)

    key = jax.random.key(cfg.seed + first - 1)
    key, k_init, k_env = jax.random.split(key, 3)
    params = jax.tree.map(jnp.asarray, state["params"]) if state else init_params(cfg, k_init)
    if init_from and not state:
        if "*" in init_from:   # a glob (scripts/kaggle_with_brain.py: the brain is a dataset under /kaggle/input)
            import glob
            found = sorted(glob.glob(init_from, recursive=True))
            if not found:
                raise SystemExit(f"--init-from {init_from}: nothing there")
            init_from = found[0]
        start = pickle.loads(Path(init_from).read_bytes())
        if start["config"].get("trigger", "gauss") != cfg.trigger:
            raise SystemExit(f"{init_from}: its trigger is {start['config'].get('trigger', 'gauss')}, this run's {cfg.trigger}")
        start_params = start["params"]
        if any(np.shape(start_params[k]) != np.shape(params[k]) for k in params):
            raise SystemExit(f"{init_from}: a different network than this run's")
        params = jax.tree.map(jnp.asarray, start_params)
        if reset_std:   # (3c) the chase pilot's noise had shrunk to 0.007: re-open exploration for the new task
            params["log_std"] = jnp.full_like(params["log_std"], cfg.log_std_init)
        print(f"network from {init_from}" + (f", exploration reset to {np.exp(cfg.log_std_init):.2f}" if reset_std else ""),
              flush=True)
    opt = jax.tree.map(jnp.asarray, state["optimizer"]) if state else adam_init(params)
    league = state.get("league", []) if state else []
    if selfplay and not league:
        league = [jax.tree.map(np.asarray, params)]
    rng = np.random.default_rng(cfg.seed + first)
    opp_i = len(league) - 1

    mesh = Mesh(np.array(devices), ("d",))
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    game = Fighters(game_cfg)
    T, N = cfg.n_steps, cfg.n_envs
    if selfplay:
        update = make_update_self(cfg, game, N // n_dev, n_dev)
        specs = (P(), P(), P("d"), P("d"), P(), P(), P(), P())
    else:
        update = make_update(cfg, game, N // n_dev, n_dev)
        specs = (P(), P(), P("d"), P("d"), P(), P(), P())
    update = jax.jit(jax.shard_map(update, mesh=mesh, in_specs=specs,
                                   out_specs=(P(), P(), P("d"), P("d"), P(), P())),
                     donate_argnums=(0, 1, 2, 3))
    level = state.get("level", cfg.level_start) if state else cfg.level_start
    kill_ema = state.get("kill_ema", 0.0) if state else 0.0
    env = jax.jit(jax.vmap(game.init, in_axes=(0, None)), out_shardings=shard)(jax.random.split(k_env, N),
                                                                              jnp.float32(level))
    see = (lambda st: jnp.stack([game.observe(st, 0), game.observe(st, 1)])) if selfplay else \
        (lambda st: game.observe(st, 0))
    obs = jax.jit(jax.vmap(see), out_shardings=shard)(env)
    params, opt, key = jax.device_put((params, opt, key), rep)

    n_updates = cfg.total_steps // (T * N)
    fights = state["fights"] if state else 0
    start = time.time() - (state["elapsed"] if state else 0.0)
    session_start = time.time()
    log = open(run_dir / "metrics.jsonl", "a")
    from ailearns.agents import tracking
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))
    np_tree = lambda t: jax.tree.map(np.asarray, t)  # noqa: E731

    def save(update_no: int, stats: dict) -> None:
        (run_dir / "checkpoints" / f"u{update_no:05d}.pkl").write_bytes(pickle.dumps(
            {"params": np_tree(params), "config": asdict(cfg), "game": asdict(game_cfg), "update": update_no,
             "steps": update_no * T * N, "fights": fights, "level": level, "stats": stats}))

    def save_latest(update_no: int) -> None:
        (run_dir / "latest.pkl.tmp").write_bytes(pickle.dumps(
            {"params": np_tree(params), "optimizer": np_tree(opt), "update": update_no, "fights": fights,
             "elapsed": time.time() - start, "level": level, "kill_ema": kill_ema,
             "league": league}))
        (run_dir / "latest.pkl.tmp").replace(run_dir / "latest.pkl")

    if not state:
        save(0, {})
    for update_no in range(first, n_updates + 1):
        lr = jnp.float32(cfg.lr * (1 - (update_no - 1) / n_updates))
        if selfplay:
            if update_no % cfg.snapshot_every == 0:
                league = (league + [np_tree(params)])[-cfg.league_size:]
            if update_no % cfg.swap_every == 0 or update_no == first:
                opp_i = int(rng.integers(len(league)))
            params, opt, env, obs, key, m = update(params, opt, env, obs, key, lr, jnp.float32(level),
                                                   jax.device_put(jax.tree.map(jnp.asarray, league[opp_i]), rep))
        else:
            params, opt, env, obs, key, m = update(params, opt, env, obs, key, lr, jnp.float32(level))
        m = jax.device_get(m)
        ended = int(m["ended"])
        fights += ended
        per = lambda k: float(m[k]) / ended if ended else None  # noqa: E731
        fight_level = level
        if ended and not selfplay:
            kill_ema = 0.9 * kill_ema + 0.1 * per("kills")
        if level < 1.0 and kill_ema >= cfg.level_up_at:
            level, kill_ema = min(1.0, round(level + cfg.level_step, 4)), 0.0
        stats = {
            "update": update_no, "steps": update_no * T * N, "fights": fights, "time": round(time.time() - start, 1),
            "level": fight_level, "kill_rate": per("kills"), "crash_rate": per("crashes"),
            "timeout_rate": per("timeouts"), "fight_s": per("len_sum"), "rounds_fired": per("fired_sum"),
        }
        if selfplay:
            le = int(m["league_ended"])
            stats.update({"league_size": len(league), "opponent": opp_i, "league_fights": le,
                          "league_win": float(m["league_wins"]) / le if le else None,
                          "league_loss": float(m["league_losses"]) / le if le else None})
            stats["dealt"] = None
        else:
            stats.update({"target_crash_rate": per("target_crashed"), "dealt": per("dealt_sum"),
                          "reward_per_fight": per("rew_sum")})
        stats.update({
            "pg_loss": float(m["pg_loss"]), "v_loss": float(m["v_loss"]),
            "entropy": float(m["entropy"]), "clipfrac": float(m["clipfrac"]),
            "std": [round(float(s), 3) for s in np.exp(np.asarray(log_std(params, cfg)))],
        })
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
            f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
            print(f"update {update_no:5d}/{n_updates}  level {fight_level:.2f}  steps {stats['steps']:>12,}  "
                  f"fights {fights:>8,}  kill {f(stats['kill_rate'], 3)}  crash {f(stats['crash_rate'], 3)}  "
                  f"draw {f(stats['timeout_rate'], 3)}  dealt {f(stats['dealt'], 2)}  "
                  f"fired {f(stats['rounds_fired'], 0)}  fight {f(stats['fight_s'], 1)} s  "
                  + (f"league win {f(stats['league_win'], 3)} loss {f(stats['league_loss'], 3)} "
                     f"(v #{opp_i} of {len(league)})  " if selfplay else "") + f"std {stats['std']}  "
                  f"{sps:,.0f} steps/s", flush=True)
        if out_of_time:
            print(f"stopped after {max_hours} h at update {update_no}; carry on with --resume", flush=True)
            break
    log.close()
    tracking.finish(tracker)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="run folder, under runs/")
    p.add_argument("--steps", type=float, default=DogfightPPOConfig.total_steps)
    p.add_argument("--envs", type=int, default=DogfightPPOConfig.n_envs)
    p.add_argument("--n-steps", type=int, default=DogfightPPOConfig.n_steps)
    p.add_argument("--minibatch", type=int, default=DogfightPPOConfig.minibatch)
    p.add_argument("--lr", type=float, default=DogfightPPOConfig.lr)
    p.add_argument("--gamma", type=float, default=DogfightPPOConfig.gamma)
    p.add_argument("--ent-coef", type=float, default=DogfightPPOConfig.ent_coef)
    p.add_argument("--level-start", type=float, default=DogfightPPOConfig.level_start)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--level-up-at", type=float, default=DogfightPPOConfig.level_up_at,
                   help="smoothed kill rate that widens the start box a step")
    p.add_argument("--level-step", type=float, default=DogfightPPOConfig.level_step)
    p.add_argument("--log-std-init", type=float, default=DogfightPPOConfig.log_std_init,
                   help="starting exploration spread (log); -0.5 = 0.6 of each control's range")
    p.add_argument("--trigger", choices=["bernoulli", "gauss"], default=DogfightPPOConfig.trigger)
    p.add_argument("--mode", choices=["chase", "self"], default="chase")
    p.add_argument("--self-share", type=float, default=DogfightPPOConfig.self_share)
    p.add_argument("--init-from", default=None, help="start from this checkpoint's network (fresh optimizer)")
    p.add_argument("--min-std", type=float, default=DogfightPPOConfig.min_std, help="floor under the stick noise")
    p.add_argument("--reset-std", action="store_true", help="with --init-from: exploration back to --log-std-init")
    a = p.parse_args()
    cfg = DogfightPPOConfig(total_steps=int(a.steps), n_envs=a.envs, n_steps=a.n_steps, minibatch=a.minibatch,
                            lr=a.lr, gamma=a.gamma, ent_coef=a.ent_coef, level_start=a.level_start, seed=a.seed,
                            mode=a.mode, self_share=a.self_share, trigger=a.trigger, min_std=a.min_std, log_std_init=a.log_std_init,
                            level_up_at=a.level_up_at, level_step=a.level_step)
    train(cfg, a.out, resume=a.resume, max_hours=a.hours, init_from=a.init_from, reset_std=a.reset_std)


if __name__ == "__main__":
    main()
