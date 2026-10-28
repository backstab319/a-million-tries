"""Race day (episode 15), step 8: PPO for a driver among other cars, in the race school (`games/race_school.py`).

The same algorithm and brain as driver school (`ppo_car`: a Gaussian policy, two tanh layers, one compiled update over
every device), now with three inputs (steering, pedal, reverse) and the race-day senses (115 numbers: the cars around
it, its damage, tyres, the race left, the pits).

**It starts from the ep14 driver** (`init_from`, default `runs/race-v4-1`): its weights for the 82 senses it had, zero
for the new ones (so on day one it drives exactly like the ep14 driver, a car ahead looking like a wall to its
lasers), and a reverse input that starts off (bias -1).

**Self-play with a pool.** Each field of `n_cars` has `n_learn` cars driven by the learner (they learn from all of
them), the rest by one past version from a pool of `pool_size`: slot 0 is always the starting brain, the others are
refreshed with the learner every `pool_every` updates (oldest first). So it races itself, and versions of itself it
has to keep beating (the plan's guard against self-play collapse: two drivers that only know each other).

The reward is driver school's (metres along the track, -3 at the wall) plus the team's stop payment; damage is physics
only. A car's run ends at the wall, wrecked, finished, or at the stretch's end; the steps after that, while its field
runs on, don't count.
"""

import json
import time
from dataclasses import asdict, dataclass
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
from ailearns.games import race, race_field, race_school
from ailearns.games import race_car as rc

N_ACT = 3  # steering, pedal, reverse


@dataclass
class FieldPPOConfig:
    total_steps: int = 300_000_000  # learner car-steps
    n_fields: int = 256          # races at once (each n_cars cars)
    n_cars: int = 8
    n_learn: int = 4             # of them driven by the learner; the rest by a pool driver
    laps: int = 25
    n_steps: int = 128
    epochs: int = 4
    minibatch: int = 8192
    lr: float = 1e-4             # a trained brain to start from: smaller steps than from scratch
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    init_log_std: float = -1.0   # exploration on every input (the ep14 driver had almost none left)
    init_from: str = "runs/race-v4-1"  # "" for a brain from scratch
    pool_size: int = 8
    pool_every: int = 10
    stretch_s: float = 60.0
    grid_share: float = 0.2
    call_rate_s: float = 120.0
    coach: float = 0.0           # the pit coach's weight (race_school.SchoolConfig.coach; 0: off)
    pit_share: float = 0.5       # race_school.SchoolConfig's pit lessons
    pit_car: float = 0.5
    pit_school_steps: int = 0    # first this many learner steps in a pit school (every stretch mid-race and a lesson,
                                 # every car in it): learn the stop first, then race with lessons (step 8g)
    seed: int = 0
    tracks: int = len(race.TRAIN_SEEDS)
    early_checkpoints: tuple = (0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75)
    checkpoint_every: int = 25


def make_school(cfg: FieldPPOConfig, seeds, pit_school: bool = False,
                field: race_field.RaceField | None = None) -> race_school.RaceSchool:
    """The race school for `cfg` (or the pit school: every stretch mid-race and a lesson, every car in it)."""
    if field is None:
        rcfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
        field = race_field.RaceField(race.make_tracks(list(seeds), rcfg), n_cars=cfg.n_cars, laps=cfg.laps, cfg=rcfg)
    shares = dict(grid_share=0.0, pit_share=1.0, pit_car=1.0) if pit_school else dict(
        grid_share=cfg.grid_share, pit_share=cfg.pit_share, pit_car=cfg.pit_car)
    return race_school.RaceSchool(field, race_school.SchoolConfig(stretch_s=cfg.stretch_s, call_rate_s=cfg.call_rate_s,
                                                                  coach=cfg.coach, coach_gamma=cfg.gamma, **shares))


def widen(params: dict, obs_size: int, init_log_std: float) -> dict:
    """A single-car driver's brain for the race-day senses and three inputs: zero weights for the new senses, a
    reverse input that starts off (mean -1, needs > 0.5 to select), exploration reset."""
    p = dict(params)
    w = p["body.0.weight"]
    p["body.0.weight"] = jnp.concatenate([w, jnp.zeros((w.shape[0], obs_size - w.shape[1]))], 1)
    p["policy.weight"] = jnp.concatenate([p["policy.weight"], jnp.zeros((1, p["policy.weight"].shape[1]))], 0)
    p["policy.bias"] = jnp.concatenate([p["policy.bias"], jnp.array([-1.0])])
    p["log_std"] = jnp.full(N_ACT, init_log_std)
    return p


def make_update(cfg: FieldPPOConfig, school: race_school.RaceSchool, n_local: int, n_dev: int):
    """One PPO update: every field drives T decisions, then the learner learns from its own cars' steps."""
    T, K, n = cfg.n_steps, cfg.n_learn, cfg.n_cars
    G = cfg.pool_size
    if n_local % G:
        raise SystemExit(f"fields per device ({n_local}) must divide by the pool ({G})")
    B = T * n_local * K
    mb = cfg.minibatch // n_dev
    n_mb = B // mb
    dt = school.field.cfg.dt
    vstep, vobs, vreset = jax.vmap(school.step), jax.vmap(school.observe), jax.vmap(school.reset_if)
    pool_forward = jax.vmap(forward)

    def loss_fn(params, b):
        mean, log_std, value = forward(params, b["obs"])
        logp = log_prob(mean, log_std, b["act"])
        ratio = jnp.exp(logp - b["logp"])
        m = b["valid"].astype(jnp.float32)
        cnt = jnp.maximum(m.sum(), 1.0)
        avg = lambda x: (x * m).sum() / cnt  # noqa: E731
        a = b["adv"]
        mu = avg(a)
        a = (a - mu) / (jnp.sqrt(avg((a - mu) ** 2)) + 1e-8)
        pg = avg(jnp.maximum(-a * ratio, -a * jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)))
        vl = 0.5 * avg((value - b["ret"]) ** 2)
        ent = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
        loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
        return loss, {"pg_loss": pg, "v_loss": vl, "entropy": ent, "std": jnp.exp(log_std).mean(),
                      "clipfrac": avg((jnp.abs(ratio - 1) > cfg.clip).astype(jnp.float32))}

    def update(params, opt, pool, les, obs, key, lr):
        key, k_roll, k_learn = jax.random.split(key, 3)
        dev = lax.axis_index("d")
        k_roll, k_learn = jax.random.fold_in(k_roll, dev), jax.random.fold_in(k_learn, dev)

        def drive(carry, k):
            les, obs = carry  # obs (n_local, n, O)
            ka, ko, ks, kr = jax.random.split(k, 4)
            mine = obs[:, :K].reshape(n_local * K, -1)
            mean, log_std, value = forward(params, mine)
            act = mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
            logp = log_prob(mean, log_std, act)
            theirs = obs[:, K:].reshape(G, (n_local // G) * (n - K), -1)  # field group g races pool driver g
            om, ostd, _ = pool_forward(pool, theirs)
            oact = om + jnp.exp(ostd)[:, None, :] * jax.random.normal(ko, om.shape)
            actions = jnp.concatenate([act.reshape(n_local, K, N_ACT), oact.reshape(n_local, n - K, N_ACT)], 1)
            before = les.field
            les, rew, racing, ended, over = vstep(les, actions, jax.random.split(ks, n_local))
            f = les.field
            over = over | ~jax.vmap(school.racing)(f)[:, :K].any(1)  # all the learner's cars out: a new stretch
            ended = ended | (racing & over[:, None])
            sl = lambda x: x[:, :K].reshape(-1)  # noqa: E731
            out = {"obs": mine, "act": act, "logp": logp, "val": value, "rew": sl(rew), "done": sl(ended),
                   "valid": sl(racing),
                   "metres": sl(f.cars.progress - les.start), "seconds": jnp.repeat((f.t - les.t0) * dt, K),
                   "wall": sl(f.cars.crashed & ~f.wrecked), "wrecked": sl(f.wrecked), "finished": sl(f.finished),
                   "contact": sl(racing & (f.bump > 0)), "stops": sl(racing & (f.stops > before.stops)),
                   "calls": sl(racing & f.box_call & ~before.box_call),
                   "reverse": sl(racing & (f.cars.car.gear == 0))}
            les = vreset(les, over, jax.random.split(kr, n_local))
            return (les, vobs(les)), out
        (les, obs_next), tr = lax.scan(drive, (les, obs), jax.random.split(k_roll, T))

        _, _, next_value = forward(params, obs_next[:, :K].reshape(n_local * K, -1))

        def gae(carry, x):
            last, nv = carry
            nonterminal = 1.0 - x["done"].astype(jnp.float32)
            delta = x["rew"] + cfg.gamma * nv * nonterminal - x["val"]
            last = delta + cfg.gamma * cfg.lam * nonterminal * last
            return (last, x["val"]), last
        _, adv = lax.scan(gae, (jnp.zeros_like(next_value), next_value), tr, reverse=True)
        batch = {"obs": tr["obs"], "act": tr["act"], "logp": tr["logp"], "adv": adv, "ret": adv + tr["val"],
                 "valid": tr["valid"]}
        batch = jax.tree.map(lambda x: x.reshape(B, *x.shape[2:]), batch)

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
            perm = jax.random.permutation(k, B)[: n_mb * mb].reshape(n_mb, mb)
            return lax.scan(step, carry, perm)
        (params, opt), aux = lax.scan(epoch, (params, opt), jax.random.split(k_learn, cfg.epochs))

        e = tr["done"]
        m = {"ended": e.sum(), "valid": tr["valid"].sum(),
             "metres_sum": jnp.where(e, tr["metres"], 0).sum(), "seconds_sum": jnp.where(e, tr["seconds"], 0).sum(),
             "wall": (e & tr["wall"]).sum(), "wrecked": (e & tr["wrecked"]).sum(),
             "finished": (e & tr["finished"]).sum(), "contact": tr["contact"].sum(), "stops": tr["stops"].sum(),
             "calls": tr["calls"].sum(), "reverse": tr["reverse"].sum()}
        m = jax.tree.map(lambda x: lax.psum(x, "d"), m)
        m["metres_max"] = lax.pmax(jnp.where(e, tr["metres"], 0).max(), "d")
        m.update(jax.tree.map(lambda x: lax.pmean(x.mean(), "d"), aux))
        return params, opt, les, obs_next, key, m

    return update


def train(cfg: FieldPPOConfig, run_dir: str | Path, resume: bool = False, max_hours: float | None = None) -> None:
    import torch
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    n_dev = len(devices)
    if cfg.n_fields % n_dev or cfg.minibatch % n_dev:
        raise SystemExit(f"fields and minibatch must divide by the {n_dev} devices")
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
    print(f"JAX on {n_dev} x {devices[0].device_kind}; building {cfg.tracks} circuits …", flush=True)
    school = make_school(cfg, list(race.TRAIN_SEEDS)[: cfg.tracks])
    F = school.field
    if not resume:
        (run_dir / "config.json").write_text(json.dumps(
            {**asdict(cfg), "game": "race_field", "race": asdict(F.cfg), "school": asdict(school.cfg),
             "design": rc.DESIGN_NAMES, "obs_size": F.obs_size, "git": sha, "backend": "jax",
             "device": f"{n_dev} x {devices[0].device_kind}"}, indent=2))
    first = state["update"] + 1 if state else 1

    if state:
        params = {k: jnp.asarray(v.numpy()) for k, v in state["model"].items()}
        opt = jax.tree.map(jnp.asarray, state["optimizer"])
        pool = jax.tree.map(jnp.asarray, state["pool"])
    else:
        if cfg.init_from:
            from ailearns.checkpoints import pick_checkpoint
            path, _, ck = cfg.init_from.partition(":")
            params, _ = load(pick_checkpoint(path, ck or "last"))
            if params["body.0.weight"].shape[1] != F.obs_size:
                params = widen(params, F.obs_size, cfg.init_log_std)
            print(f"starting from {cfg.init_from}", flush=True)
        else:
            params = init_params(jax.random.key(cfg.seed), F.obs_size, cfg.hidden, cfg.init_log_std)
            params = widen(params, F.obs_size, cfg.init_log_std)
        opt = adam_init(params)
        pool = jax.tree.map(lambda x: jnp.stack([x] * cfg.pool_size), params)

    mesh = Mesh(np.array(devices), ("d",))
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    T, N, K = cfg.n_steps, cfg.n_fields, cfg.n_learn
    per_update = T * N * K
    def compiled(s):
        return jax.jit(jax.shard_map(make_update(cfg, s, N // n_dev, n_dev), mesh=mesh,
                                     in_specs=(P(), P(), P(), P("d"), P("d"), P(), P()),
                                     out_specs=(P(), P(), P("d"), P("d"), P(), P())), donate_argnums=(0, 1, 3, 4))
    update = compiled(school)
    pit_updates = cfg.pit_school_steps // per_update
    pit_update = compiled(make_school(cfg, (), pit_school=True, field=F)) if pit_updates else None
    key = jax.random.key(cfg.seed + 1000 + first)
    key, k_env = jax.random.split(key)
    les = jax.jit(jax.vmap(school.start), out_shardings=shard)(jax.random.split(k_env, N))
    obs = jax.jit(jax.vmap(school.observe), out_shardings=shard)(les)
    params, opt, pool, key = jax.device_put((params, opt, pool, key), rep)

    n_updates = cfg.total_steps // per_update
    runs = state["runs"] if state else 0
    start = time.time() - (state["elapsed"] if state else 0.0)
    session_start = time.time()
    log = open(run_dir / "metrics.jsonl", "a")
    from ailearns.agents import tracking
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))
    common = {"config": asdict(cfg), "race": asdict(F.cfg), "backend": "jax", "game": "race_field"}

    def save(update_no: int, stats: dict) -> None:
        torch.save({"model": to_torch(params), **common, "update": update_no, "steps": update_no * per_update,
                    "runs": runs, "stats": stats}, run_dir / "checkpoints" / f"u{update_no:05d}.pt")

    def save_latest(update_no: int) -> None:
        torch.save({"model": to_torch(params), "optimizer": jax.tree.map(np.asarray, opt),
                    "pool": jax.tree.map(np.asarray, pool), **common, "update": update_no, "runs": runs,
                    "elapsed": time.time() - start}, run_dir / "latest.pt.tmp")
        (run_dir / "latest.pt.tmp").replace(run_dir / "latest.pt")

    if not state:
        save(0, {})
    for update_no in range(first, n_updates + 1):
        lr = jnp.float32(cfg.lr * (1 - (update_no - 1) / n_updates))
        step = pit_update if update_no <= pit_updates else update  # same field and state: only the starts differ
        params, opt, les, obs, key, m = step(params, opt, pool, les, obs, key, lr)
        if update_no % cfg.pool_every == 0 and cfg.pool_size > 1:  # slot 0 keeps the starting brain
            slot = 1 + (update_no // cfg.pool_every - 1) % (cfg.pool_size - 1)
            pool = jax.tree.map(lambda q, p: q.at[slot].set(p), pool, params)
        m = jax.device_get(m)
        ended, valid = int(m["ended"]), max(int(m["valid"]), 1)
        runs += ended
        per = lambda k: int(m[k]) / ended if ended else None  # noqa: E731
        metres = float(m["metres_sum"]) / ended if ended else None
        secs = float(m["seconds_sum"]) / ended if ended else None
        stats = {
            "update": update_no, "steps": update_no * per_update, "runs": runs, "time": round(time.time() - start, 1),
            "pit_school": update_no <= pit_updates,
            "metres_mean": metres, "metres_max": float(m["metres_max"]) if ended else None,
            "kmh_mean": 3.6 * metres / secs if ended and secs else None, "seconds_mean": secs,
            "wall_rate": per("wall"), "wreck_rate": per("wrecked"), "finish_rate": per("finished"),
            "contact_share": int(m["contact"]) / valid, "stops": int(m["stops"]), "calls": int(m["calls"]),
            "reverse_share": int(m["reverse"]) / valid,
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
        if update_no == pit_updates and pit_updates:
            print(f"pit school done at update {update_no}: the race school from here", flush=True)
        if update_no % 10 == 0 or update_no == first:
            sps = (update_no - first + 1) * per_update / (time.time() - session_start)
            show = lambda v, k=0: "-" if v is None else f"{v:.{k}f}"  # noqa: E731
            print(f"update {update_no:5d}/{n_updates}  steps {stats['steps']:>11,}  runs {runs:>8,}  "
                  f"m/run {show(metres):>6}  km/h {show(stats['kmh_mean']):>4}  wall {show(stats['wall_rate'], 2)}"
                  f"  wreck {show(stats['wreck_rate'], 2)}  contact {stats['contact_share']:.3f}"
                  f"  stops {stats['stops']}/{stats['calls']}  std {stats['std']:.2f}  {sps:,.0f} steps/s", flush=True)
        if out_of_time:
            print(f"stopped after {max_hours} h at update {update_no}; carry on with --resume", flush=True)
            break
    log.close()
    tracking.finish(tracker)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="race school (episode 15, step 8)")
    ap.add_argument("run")
    ap.add_argument("--steps", type=float, default=300e6)
    ap.add_argument("--fields", type=int, default=256)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--tracks", type=int, default=len(race.TRAIN_SEEDS))
    ap.add_argument("--init-from", default="runs/race-v4-1")
    ap.add_argument("--coach", type=float, default=0.0)
    ap.add_argument("--pit-school", type=float, default=0.0, help="learner steps in the pit school first")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train(FieldPPOConfig(total_steps=int(a.steps), n_fields=a.fields, n_steps=a.n_steps, minibatch=a.minibatch,
                         tracks=a.tracks, init_from=a.init_from, coach=a.coach,
                         pit_school_steps=int(a.pit_school)), a.run, resume=a.resume,
          max_hours=a.hours)
