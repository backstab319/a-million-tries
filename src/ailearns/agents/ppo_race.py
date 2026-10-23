"""Driver school (episode 14): PPO for the racing sandbox's driver, on cars of every design.

The same algorithm and brain as episode 4's car (`ppo_car`: a Gaussian policy for steering and pedal, two tanh layers,
one compiled update over every device). What's new is the school: every run restarts on a random circuit, a random
stretch of it, at a random speed, **in a car of random design** (all `race_car.DESIGN` numbers drawn across their
ranges), and the driver is told the design. So it learns to drive anything the engineer might build. A run ends at
the wall or after `time_limit` seconds; the score is metres along the track, so faster is better.

Checkpoints are torch files like the other trainers', with `"game": "race"`.
"""

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_car import act, init_params, load, make_update
from ailearns.agents.ppo_jax import adam_init, to_torch
from ailearns.games import race
from ailearns.games import race_car as rc


@dataclass
class RacePPOConfig:
    total_steps: int = 300_000_000
    n_envs: int = 2048
    n_steps: int = 128
    epochs: int = 4
    minibatch: int = 8192
    lr: float = 3e-4
    gamma: float = 0.995         # values ~10 s ahead: braking from 300 km/h starts well before the corner
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    init_log_std: float = -0.5
    start_speed: float = 40.0    # runs start at a random speed up to this (m/s): it meets fast corners early
    time_limit: float = 60.0     # seconds per run
    design_center: tuple = ()    # empty: every design in the parts box; else cars near this one (specialising)
    design_spread: float = 0.15
    init_from: str = ""          # start from this brain ("runs/x:<update>") instead of from scratch
    reset_std: bool = True       # with init_from: exploration back to init_log_std (a trained brain has ~none left)
    seed: int = 0
    tracks: int = len(race.TRAIN_SEEDS)
    early_checkpoints: tuple = (0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75)
    checkpoint_every: int = 25


def game_config(cfg: RacePPOConfig) -> race.RaceConfig:
    return replace(race.RaceConfig(), start_speed=cfg.start_speed, time_limit=cfg.time_limit,
                   design_center=tuple(cfg.design_center), design_spread=cfg.design_spread)


def train(cfg: RacePPOConfig, run_dir: str | Path, resume: bool = False, max_hours: float | None = None) -> None:
    import torch
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    n_dev = len(devices)
    if cfg.n_envs % n_dev or cfg.minibatch % n_dev:
        raise SystemExit(f"--envs and minibatch must divide by the {n_dev} devices")
    game_cfg = game_config(cfg)
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
            {**asdict(cfg), "game": "race", "race": asdict(game_cfg), "design": rc.DESIGN_NAMES, "git": sha,
             "backend": "jax", "device": f"{n_dev} x {devices[0].device_kind}"}, indent=2))
    first = state["update"] + 1 if state else 1
    print(f"JAX on {n_dev} x {devices[0].device_kind}; building {cfg.tracks} circuits …", flush=True)

    game = race.RaceGame(race.make_tracks(list(race.TRAIN_SEEDS)[: cfg.tracks], game_cfg), game_cfg)
    if state:
        params = {k: jnp.asarray(v.numpy()) for k, v in state["model"].items()}
        opt = jax.tree.map(jnp.asarray, state["optimizer"])
    elif cfg.init_from:
        from ailearns.checkpoints import pick_checkpoint
        path, _, ck = cfg.init_from.partition(":")
        params, _ = load(pick_checkpoint(path, ck or "last"))
        if cfg.reset_std:
            params["log_std"] = jnp.full(2, cfg.init_log_std)
        opt = adam_init(params)
        print(f"starting from {cfg.init_from}", flush=True)
    else:
        params = init_params(jax.random.key(cfg.seed), game_cfg.obs_size, cfg.hidden, cfg.init_log_std)
        opt = adam_init(params)

    mesh = Mesh(np.array(devices), ("d",))
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    T, N = cfg.n_steps, cfg.n_envs
    update = make_update(cfg, game, N // n_dev, n_dev, per_lap=False)
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
    best = state["best_metres"] if state else 0.0
    start = time.time() - (state["elapsed"] if state else 0.0)
    session_start = time.time()
    log = open(run_dir / "metrics.jsonl", "a")
    from ailearns.agents import tracking
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))
    common = {"config": asdict(cfg), "race": asdict(game_cfg), "backend": "jax", "game": "race"}

    def save(update_no: int, stats: dict) -> None:
        torch.save({"model": to_torch(params), **common, "update": update_no, "steps": update_no * T * N,
                    "runs": runs, "stats": stats}, run_dir / "checkpoints" / f"u{update_no:05d}.pt")

    def save_latest(update_no: int) -> None:
        torch.save({"model": to_torch(params), "optimizer": jax.tree.map(np.asarray, opt), **common,
                    "update": update_no, "runs": runs, "best_metres": best, "elapsed": time.time() - start},
                   run_dir / "latest.pt.tmp")
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
            best = max(best, float(m["laps_max"]))
        metres = float(m["laps_sum"]) / ended if ended else None
        secs = float(m["seconds_sum"]) / ended if ended else None
        stats = {
            "update": update_no, "steps": update_no * T * N, "runs": runs, "time": round(time.time() - start, 1),
            "metres_mean": metres, "metres_max": float(m["laps_max"]) if ended else None, "best_metres": best,
            "kmh_mean": 3.6 * metres / secs if ended and secs else None,
            "crash_rate": int(m["crashes"]) / ended if ended else None, "seconds_mean": secs,
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
            show = lambda v, k=0: "-" if v is None else f"{v:.{k}f}"  # noqa: E731
            print(f"update {update_no:5d}/{n_updates}  steps {stats['steps']:>11,}  runs {runs:>8,}  "
                  f"m/run {show(metres):>6}  km/h {show(stats['kmh_mean']):>4}  crash {show(stats['crash_rate'], 2)}"
                  f"  std {stats['std']:.2f}  {sps:,.0f} steps/s", flush=True)
        if out_of_time:
            print(f"stopped after {max_hours} h at update {update_no}; carry on with --resume", flush=True)
            break
    log.close()
    tracking.finish(tracker)


# ── evaluation ──────────────────────────────────────────────────────────────

def drive(params: dict, game: race.RaceGame, track: int, design, seconds: float = 90.0) -> dict:
    """One run with the best-guess policy (no noise) from a standing start on the line: crash flag, distance, speed,
    and every lap time."""
    st = game.init(jax.random.PRNGKey(0), design=jnp.asarray(design), track=track, random_start=False)
    step, observe = _jit(game)
    L = float(game.tracks.length[track])
    laps, last = [], 0.0
    for _ in range(int(seconds / game.cfg.dt)):
        st, _ = step(st, jnp.asarray(act(params, observe(st))[0]))
        if float(st.progress) >= (len(laps) + 1) * L:
            t = int(st.t) * game.cfg.dt
            laps.append(round(t - last, 2))
            last = t
        if bool(st.done):
            break
    secs = int(st.t) * game.cfg.dt
    return {"track": track, "crashed": bool(st.crashed), "metres": round(float(st.progress), 1),
            "kmh": round(3.6 * float(st.progress) / max(secs, 1e-6), 1), "seconds": secs, "laps": laps}


_STEPS: dict = {}


def _jit(game):
    if id(game) not in _STEPS:
        _STEPS[id(game)] = jax.jit(game.step), jax.jit(game.observe)
    return _STEPS[id(game)]


def evaluate(run: str, checkpoint: str = "last", n_tracks: int = 8, first: int = 0, designs: int = 4,
             seconds: float = 90.0) -> dict:
    """Driver test on circuits it never trained on (test seeds first..first+n_tracks): the F1-like reference car
    plus `designs` random designs (fixed seed), one standing-start run each per circuit. Choose checkpoints on
    the first circuits, report numbers from others (playbook: never report from the games you picked with)."""
    from ailearns.checkpoints import pick_checkpoint
    params, ck = load(pick_checkpoint(run, str(checkpoint)))
    gcfg = replace(race.RaceConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["race"].items()}),
                   time_limit=seconds)
    seeds = list(race.TEST_SEEDS)[first:first + n_tracks]
    game = race.RaceGame(race.make_tracks(seeds, gcfg), gcfg)
    rng = np.random.default_rng(7)
    span = rc.DESIGN_HIGH - rc.DESIGN_LOW
    cars = {"f1": rc.F1_LIKE[0]} | {f"random{i}": rc.DESIGN_LOW + rng.random(len(rc.DESIGN)) * span
                                     for i in range(designs)}
    res = {name: [drive(params, game, i, d, seconds) for i in range(n_tracks)] for name, d in cars.items()}
    summary = {name: {"crashes": sum(r["crashed"] for r in rs),
                      "metres": round(float(np.mean([r["metres"] for r in rs]))),
                      "kmh": round(float(np.mean([r["kmh"] for r in rs])), 1)} for name, rs in res.items()}
    return {"run": str(run), "update": ck["update"], "steps": ck["steps"], "tracks": seeds, "summary": summary,
            "per_track": res}


if __name__ == "__main__":  # local runs; on Kaggle: `ailearns cloud train runs/<name> --game race ...`
    import argparse
    ap = argparse.ArgumentParser(description="driver school (episode 14)")
    ap.add_argument("run")
    ap.add_argument("--steps", type=float, default=300e6)
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval", default=None, help="evaluate this checkpoint instead of training (e.g. last, 50)")
    a = ap.parse_args()
    if a.eval:
        out = evaluate(a.run, a.eval)
        print(json.dumps(out["summary"], indent=1))
    else:
        train(RacePPOConfig(total_steps=int(a.steps), n_envs=a.envs), a.run, resume=a.resume, max_hours=a.hours)
