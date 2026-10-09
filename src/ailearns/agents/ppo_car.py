"""PPO for the race car: the same algorithm as the Snake trainer (ppo_jax.py), with a
brain that outputs numbers instead of picking a button.

Snake chose between 3 moves (a categorical policy). The car's brain outputs a
steering and a throttle value plus how unsure it is about each: a Gaussian
policy. While training it tries values near its best guess (wider when unsure);
when it's tested, or racing a human, it uses its best guess.

Everything (all cars driving T decisions, advantages, learning) is one compiled
function, data-parallel over every device (both T4s on Kaggle). Checkpoints are
torch files of the weights, like Snake's, so the usual tooling picks them up.
"""

import json
import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_jax import adam_init, adam_step, to_torch
from ailearns.games.car import TRAIN_SEEDS, CarConfig, CarGame, make_tracks


@dataclass
class CarPPOConfig:
    total_steps: int = 200_000_000
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
    messy_starts: bool = False   # see CarConfig.messy_starts
    ray_range: float = 60.0      # metres the lasers reach (CarConfig.ray_range)
    init_log_std: float = -0.5   # starts trying values about +-0.6 around its guess
    seed: int = 0
    tracks: int = len(TRAIN_SEEDS)
    early_checkpoints: tuple = (0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75)
    checkpoint_every: int = 25


# ── the brain ────────────────────────────────────────────────────────────────

def init_params(key, obs_size: int, hidden: int, init_log_std: float) -> dict:
    """Two tanh layers shared by the policy (mean steer/throttle) and the value head."""
    def dense(k, n_in, n_out, scale):
        w = jax.nn.initializers.orthogonal(scale)(k, (n_out, n_in))
        return w, jnp.zeros(n_out)
    k = jax.random.split(key, 4)
    p = {}
    p["body.0.weight"], p["body.0.bias"] = dense(k[0], obs_size, hidden, np.sqrt(2))
    p["body.2.weight"], p["body.2.bias"] = dense(k[1], hidden, hidden, np.sqrt(2))
    p["policy.weight"], p["policy.bias"] = dense(k[2], hidden, 2, 0.01)  # start near "do nothing"
    p["value.weight"], p["value.bias"] = dense(k[3], hidden, 1, 1.0)
    p["log_std"] = jnp.full(2, init_log_std)
    return p


def forward(params: dict, obs: jax.Array):
    """obs (batch, obs_size) -> mean action (batch, 2), log std (2,), value (batch,)."""
    h = jnp.tanh(obs @ params["body.0.weight"].T + params["body.0.bias"])
    h = jnp.tanh(h @ params["body.2.weight"].T + params["body.2.bias"])
    mean = h @ params["policy.weight"].T + params["policy.bias"]
    value = (h @ params["value.weight"].T + params["value.bias"])[:, 0]
    return mean, params["log_std"], value


def activity(params: dict, obs: jax.Array):
    """The two hidden layers' firing (batch, hidden) each: the same numbers forward() computes."""
    h1 = jnp.tanh(obs @ params["body.0.weight"].T + params["body.0.bias"])
    return h1, jnp.tanh(h1 @ params["body.2.weight"].T + params["body.2.bias"])


def log_prob(mean, log_std, a):
    return (-0.5 * ((a - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)


def act(params: dict, obs) -> np.ndarray:
    """Best-guess action(s) for footage and tests (no exploration noise)."""
    mean, _, _ = forward(params, jnp.atleast_2d(jnp.asarray(obs, jnp.float32)))
    return np.clip(np.asarray(mean), -1, 1)


def load(path) -> tuple[dict, dict]:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: jnp.asarray(v.numpy()) for k, v in ck["model"].items()}, ck


# ── training ────────────────────────────────────────────────────────────────

def make_update(cfg: CarPPOConfig, game: CarGame, n_local: int, n_dev: int):
    T = cfg.n_steps
    mb = cfg.minibatch // n_dev
    n_mb = (T * n_local) // mb
    vstep, vobs, vreset = jax.vmap(game.step), jax.vmap(game.observe), jax.vmap(game.reset_if)

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

        def drive(carry, k):
            env, obs = carry
            mean, log_std, value = forward(params, obs)
            ka, kr = jax.random.split(k)
            action = mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
            logp = log_prob(mean, log_std, action)
            env, reward = vstep(env, action)
            laps = env.progress / game.tracks.length[env.track]  # read before the reset
            seconds = env.t * game.cfg.dt
            done, crashed = env.done, env.crashed
            env = vreset(env, jax.random.split(kr, n_local))
            return (env, vobs(env)), {"obs": obs, "act": action, "logp": logp, "val": value, "rew": reward,
                                      "done": done, "crashed": crashed, "laps": laps, "seconds": seconds}
        (env, obs_next), tr = lax.scan(drive, (env, obs), jax.random.split(k_roll, T))

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

        ended = tr["done"]
        m = {"ended": ended.sum(), "crashes": tr["crashed"].sum(),
             "laps_sum": jnp.where(ended, tr["laps"], 0).sum(),
             "seconds_sum": jnp.where(ended, tr["seconds"], 0).sum(),
             "finished": (ended & ~tr["crashed"]).sum()}
        m = jax.tree.map(lambda x: lax.psum(x, "d"), m)
        m["laps_max"] = lax.pmax(jnp.where(ended, tr["laps"], 0).max(), "d")
        m.update(jax.tree.map(lambda x: lax.pmean(x.mean(), "d"), aux))
        return params, opt, env, obs_next, key, m

    return update


def train(cfg: CarPPOConfig, run_dir: str | Path, resume: bool = False, max_hours: float | None = None,
          game_cfg: CarConfig = CarConfig()) -> None:
    import torch
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    n_dev = len(devices)
    if cfg.n_envs % n_dev or cfg.minibatch % n_dev:
        raise SystemExit(f"--envs and minibatch must divide by the {n_dev} devices")
    game_cfg = replace(game_cfg, messy_starts=cfg.messy_starts, ray_range=cfg.ray_range)
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
            {**asdict(cfg), "game": "car", "car": asdict(game_cfg), "git": sha, "backend": "jax",
             "device": f"{n_dev} x {devices[0].device_kind}"}, indent=2))
    first = state["update"] + 1 if state else 1
    print(f"JAX on {n_dev} x {devices[0].device_kind}; building {cfg.tracks} tracks …", flush=True)

    tracks = make_tracks(list(TRAIN_SEEDS)[: cfg.tracks], game_cfg)
    game = CarGame(tracks, game_cfg)
    if state:
        params = {k: jnp.asarray(v.numpy()) for k, v in state["model"].items()}
        opt = jax.tree.map(jnp.asarray, state["optimizer"])
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
    obs = jax.jit(jax.vmap(game.observe), out_shardings=shard)(env)
    params, opt, key = jax.device_put((params, opt, key), rep)

    n_updates = cfg.total_steps // (T * N)
    runs = state["runs"] if state else 0
    best_laps = state["best_laps"] if state else 0.0
    start = time.time() - (state["elapsed"] if state else 0.0)
    session_start = time.time()
    log = open(run_dir / "metrics.jsonl", "a")
    from ailearns.agents import tracking
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))

    def save(update_no: int, stats: dict) -> None:
        torch.save({"model": to_torch(params), "config": asdict(cfg), "car": asdict(game_cfg), "update": update_no,
                    "steps": update_no * T * N, "runs": runs, "stats": stats, "backend": "jax", "game": "car"},
                   run_dir / "checkpoints" / f"u{update_no:05d}.pt")

    def save_latest(update_no: int) -> None:
        torch.save({"model": to_torch(params), "optimizer": jax.tree.map(np.asarray, opt), "config": asdict(cfg),
                    "car": asdict(game_cfg), "update": update_no, "runs": runs, "best_laps": best_laps,
                    "elapsed": time.time() - start, "backend": "jax", "game": "car"}, run_dir / "latest.pt.tmp")
        (run_dir / "latest.pt.tmp").replace(run_dir / "latest.pt")

    if not state:
        save(0, {})
    for update_no in range(first, n_updates + 1):
        lr = jnp.float32(cfg.lr * (1 - (update_no - 1) / n_updates))
        params, opt, env, obs, key, m = update(params, opt, env, obs, key, lr)
        m = jax.device_get(m)
        ended = int(m["ended"])
        runs += ended
        if ended:
            best_laps = max(best_laps, float(m["laps_max"]))
        stats = {
            "update": update_no, "steps": update_no * T * N, "runs": runs, "time": round(time.time() - start, 1),
            "laps_mean": float(m["laps_sum"]) / ended if ended else None,
            "laps_max": float(m["laps_max"]) if ended else None, "best_laps": best_laps,
            "crash_rate": int(m["crashes"]) / ended if ended else None,
            "seconds_mean": float(m["seconds_sum"]) / ended if ended else None,
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
            lm, cr = stats["laps_mean"], stats["crash_rate"]
            print(f"update {update_no:5d}/{n_updates}  steps {stats['steps']:>11,}  runs {runs:>8,}  "
                  f"laps {lm if lm is None else round(lm, 2)!s:>5}  crash {cr if cr is None else round(cr, 2)!s:>5}  "
                  f"std {stats['std']:.2f}  {sps:,.0f} steps/s", flush=True)
        if out_of_time:
            print(f"stopped after {max_hours} h at update {update_no}; carry on with --resume", flush=True)
            break
    log.close()
    tracking.finish(tracker)


# ── evaluation ──────────────────────────────────────────────────────────────

def drive_lap(params: dict, game: CarGame, track: int, max_seconds: float = 180.0, record: bool = False) -> dict:
    """One run with the best-guess policy from the start line (standing start, no noise).
    Returns crash flag, every lap time, and (optionally) the whole drive for footage."""
    st = game.init(jax.random.PRNGKey(0), track=track, random_start=False)
    L = float(game.tracks.length[track])
    laps, frames, last_cross = [], [], 0.0
    n = int(max_seconds / game.cfg.dt)
    step, observe = _jit(game)
    for _ in range(n):
        a = act(params, observe(st))[0]
        if record:
            frames.append({"pos": np.asarray(st.pos), "heading": float(st.heading), "speed": float(st.speed),
                           "lasers": np.asarray(observe(st))[:len(game.cfg.ray_angles)] * game.cfg.ray_range,
                           "action": a, "progress": float(st.progress)})
        st, _ = step(st, jnp.asarray(a))
        if float(st.progress) >= (len(laps) + 1) * L:
            t = int(st.t) * game.cfg.dt
            laps.append(round(t - last_cross, 2))
            last_cross = t
        if bool(st.done) or bool(st.crashed):
            break
    out = {"track": track, "crashed": bool(st.crashed), "laps": laps, "seconds": int(st.t) * game.cfg.dt,
           "distance": float(st.progress)}
    if record:
        out["frames"] = frames
    return out


_STEPS = {}


def _jit(game: CarGame):
    if id(game) not in _STEPS:
        _STEPS[id(game)] = jax.jit(game.step), jax.jit(game.observe)
    return _STEPS[id(game)]


def evaluate(run: str, checkpoint: str = "last", n_tracks: int = 16, first: int = 0) -> dict:
    """Standard car test: test tracks first..first+n_tracks (never trained on), one run each,
    best-guess driving from a standing start, up to 3 minutes. Flying lap = 2nd lap onwards.
    Choose checkpoints on tracks 0-15, report numbers from tracks 16-63 (playbook: never
    report a number from the games you picked with)."""
    from dataclasses import replace
    from ailearns.games.car import TEST_SEEDS
    from ailearns.checkpoints import pick_checkpoint
    path = pick_checkpoint(run, str(checkpoint))
    params, ck = load(path)
    game_cfg = replace(CarConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["car"].items()}),
                       time_limit=180.0)
    seeds = list(TEST_SEEDS)[first:first + n_tracks]
    game = CarGame(make_tracks(seeds, game_cfg), game_cfg)
    res = [drive_lap(params, game, i) for i in range(n_tracks)]
    flying = {seeds[r["track"]]: min(r["laps"][1:]) for r in res if len(r["laps"]) > 1}
    return {"run": str(run), "update": ck["update"], "moves": ck["steps"], "tracks": n_tracks,
            "crashes": sum(r["crashed"] for r in res), "completed_lap": sum(bool(r["laps"]) for r in res),
            "flying_laps": flying, "per_track": res}
