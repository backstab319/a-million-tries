"""Race day (episode 15), step 9: the strategist. A second, small network that runs each car's race strategy, as a
team's pit wall does; the driver (a frozen finish-mode brain) drives every car.

- Decides once a lap, as its car crosses a line `decide_m` before the pit entry: pit this lap or not, and if so which
  compound. And once on the grid: the starting compound.
- Sees ~25 numbers per car (`senses`): the race left, its tyres (compound, the four wears, the worst, laps on the set),
  fuel left, stops, damage, its position, the gaps to the cars just ahead and behind and their tyres and stops, the
  circuit's length, and whether this is the grid decision.
- Scored on the race result (owner, 2026-10-25): its cars' finishing positions (out cars at the back, F.order). Each
  race is 8 cars of random designs on a random training circuit from the grid, GP tyre wear; cars 1-4 run by the
  strategist, 5-8 by the team's fixed rule (pit when the worst tyre reaches `rule_wear`, fit hards; start on
  mediums), so beating the rule is what pays.
- It starts as the rule (`init_rule`: a direct path from the worst-tyre sense to the pit decision, hards preferred,
  mediums at the start), so evolution improves on today's strategy instead of starting from chaos. Evolution as
  `es_field` (mirrored pairs, centred ranks, Adam).

    uv run --group jax python -m ailearns.agents.es_strategist runs/strat-1 [--gens 20] [--hours 1] [--init runs/x:N]
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.es_cqb import centered_ranks
from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_car import forward, load
from ailearns.agents.ppo_jax import adam_init, adam_step, to_torch
from ailearns.games import race, race_field
from ailearns.games import race_car as rc

N_SENSES = 25
WORST = 9  # where the worst tyre's wear sits in `senses`
GRID = 24  # ... and the grid-decision flag


@dataclass
class StratConfig:
    driver: str = "runs/race-fin-7:50"  # the frozen driver of every car
    pop: int = 64
    rounds: int = 16             # races each member runs a generation (the same races for all members)
    sigma: float = 0.5           # (strat-1: 0.05 moved the pit point ~1% of wear and never flipped a tyre choice: no signal)
    lr_rel: float = 0.1
    hidden: int = 32
    generations: int = 100
    n_cars: int = 8
    n_learn: int = 4
    laps: int = 25
    cap_s: float = 3600.0        # a race's time cap (25 laps with stops run up to ~47 min; step 9h)
    gp_km: float = 305.0
    tyre_cal: float = 1.9
    rule_wear: float = 0.6       # the fixed rule the rival cars use (and the strategist starts as)
    rule_plan: int = 2           # ... fitting hards
    decide_m: float = 600.0      # the strategist decides this far before the pit entry, once a lap
    tracks: int = len(race.TRAIN_SEEDS)
    seed: int = 0
    bench: int = 16              # the brain alone on these fixed races every generation (0: off)
    checkpoint_every: int = 10
    init: str = ""               # carry on from a strategist checkpoint (runs/x:N); "": start as the rule
    method: str = "es-strategist"


def init_params(cfg: StratConfig, key) -> dict:
    """A small net plus a direct (linear) path; the direct path starts as the team's rule."""
    k1, k2 = jax.random.split(key)
    h = cfg.hidden
    p = {"w1": 0.1 * jax.random.normal(k1, (N_SENSES, h)) / np.sqrt(N_SENSES), "b1": jnp.zeros(h),
         "w2": 0.1 * jax.random.normal(k2, (h, 4)) / np.sqrt(h), "b2": jnp.zeros(4),
         "lin": jnp.zeros((N_SENSES, 4))}
    # out = [pit, soft, medium, hard]; pit when 20 x (worst - rule_wear) > 0; hards in the race, mediums on the grid
    p["lin"] = p["lin"].at[WORST, 0].set(20.0).at[GRID, 1:].set(jnp.array([0.0, 4.0, -4.0]))
    p["b2"] = jnp.array([-20.0 * cfg.rule_wear, 0.0, 0.0, 2.0])
    return p


def decide(p: dict, x: jax.Array):
    """senses (n, N_SENSES) -> (pit now?, compound)."""
    out = jnp.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"] + x @ p["lin"]
    return out[:, 0] > 0, jnp.argmax(out[:, 1:], -1)


def senses(F: race_field.RaceField, f: race_field.Field, set_from, grid_flag):
    """What the strategist sees, (N, N_SENSES): see the module docstring. set_from: where each car's tyres were fitted."""
    N = F.n
    c = f.cars
    L = F.tracks.length[c.track]
    D = F.laps * L
    left = jnp.clip(1 - c.progress / D, 0.0, 1.0)
    wear = c.car.wear
    pos = F.order(f)
    by_pos = jnp.argsort(pos)  # car at each position
    ahead = by_pos[jnp.clip(pos - 1, 0, N - 1)]
    behind = by_pos[jnp.clip(pos + 1, 0, N - 1)]
    gap_a = jnp.where(pos > 0, c.progress[ahead] - c.progress, 0.0)
    gap_b = jnp.where(pos < N - 1, c.progress - c.progress[behind], 0.0)
    d = f.damage
    return jnp.concatenate([
        left[:, None], (left * F.laps / 25.0)[:, None],
        jax.nn.one_hot(f.setup.compound, 3), wear, wear.max(-1, keepdims=True),
        (jnp.clip(c.progress - set_from, 0.0, None) / L / 25.0)[:, None],
        (1 - c.car.fuel_used / f.setup.fuel)[:, None], (f.stops / 3.0)[:, None],
        jnp.stack([d.floor, d.wing_front, d.puncture.any(-1).astype(jnp.float32)], -1),
        (pos / (N - 1.0))[:, None], jnp.clip(jnp.stack([gap_a, gap_b], -1) / 1000.0, 0.0, 3.0),
        jnp.stack([wear.max(-1)[ahead], wear.max(-1)[behind]], -1),
        jnp.stack([f.stops[ahead] - f.stops, f.stops[behind] - f.stops], -1) / 3.0,
        (L / 5000.0)[:, None], jnp.full((N, 1), grid_flag)], -1)


def make_race(cfg: StratConfig, F: race_field.RaceField, driver: dict):
    """(strategist, race key) -> each car's finishing position and what we watch."""
    K, N = cfg.n_learn, F.n
    T = int(round(cfg.cap_s / F.cfg.dt))
    g = F.game
    lane = jax.vmap(g.in_pit_lane)
    lo, span = jnp.asarray(rc.DESIGN_LOW), jnp.asarray(rc.DESIGN_HIGH - rc.DESIGN_LOW)
    mine = jnp.arange(N) < K

    def one(strat, key):
        kt, kd, kg, ks = jax.random.split(key, 4)
        track = jax.random.randint(kt, (), 0, F.tracks.length.shape[0])
        designs = lo + jax.random.uniform(kd, (N, len(rc.DESIGN))) * span
        f = F.grid(kg, track, designs)
        _, start = decide(strat, senses(F, f, f.cars.progress, 1.0))  # the starting tyre
        comp = jnp.where(mine, start, rc.COMPOUNDS.index("medium"))
        f = F.grid(kg, track, designs, comp)
        # grid slots shuffled every race (as race_school does): cars 1-4 mustn't always start at the front
        cars, perm = f.cars, jax.random.permutation(ks, N)
        o = jax.tree.map(lambda a: a[perm], cars)
        f = f._replace(cars=cars._replace(car=cars.car._replace(pos=o.car.pos, yaw=o.car.yaw), idx=o.idx, lat=o.lat,
                                          lap_pos=o.lap_pos, progress=o.progress))
        L = F.tracks.length[track]

        def to_entry(c):  # metres to the pit entry ahead
            return (-F.cfg.pit_entry - g.from_pit(c.track, c.idx)) % L

        def tick(carry, _):
            f, set_from, prev = carry
            racing = ~(f.cars.crashed | f.finished | f.wrecked)
            now = to_entry(f.cars)
            at_line = racing & ~f.box_call & ~lane(f.cars) & (prev > cfg.decide_m) & (now <= cfg.decide_m)
            pit, plan = decide(strat, senses(F, f, set_from, 0.0))
            rule = f.cars.car.wear.max(-1) >= cfg.rule_wear
            call = jnp.where(mine, at_line & pit, racing & ~f.box_call & rule)
            f = f._replace(box_call=f.box_call | call,
                           plan=jnp.where(call, jnp.where(mine, plan, cfg.rule_plan), f.plan))
            new, _ = F.step(f, jnp.clip(forward(driver, F.observe(f))[0], -1, 1))
            new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), f, new)
            set_from = jnp.where(new.stops > f.stops, new.cars.progress, set_from)
            return (new, set_from, now), None

        (f, _, _), _ = jax.lax.scan(tick, (f, f.cars.progress, to_entry(f.cars)), None, length=T)
        pos = F.order(f)
        out = f.cars.crashed | f.wrecked
        return {"pos": pos[:K].mean(), "rule_pos": pos[K:].mean(), "finished": f.finished[:K].sum(),
                "rule_finished": f.finished[K:].sum(), "out": out[:K].sum(), "stops": f.stops[:K].sum(),
                "rule_stops": f.stops[K:].sum(), "soft_start": (comp[:K] == 0).sum(), "hard_start": (comp[:K] == 2).sum(),
                "dry": ((f.cars.car.fuel_used >= f.setup.fuel) & ~f.finished)[:K].sum()}

    return lambda strat, keys: jax.vmap(lambda k: one(strat, k))(keys)


def train(cfg: StratConfig, run_dir: str | Path, max_hours: float | None = None) -> None:
    import torch
    from ailearns.agents import tracking
    from ailearns.checkpoints import pick_checkpoint
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    resume = (run_dir / "latest.pt").exists()
    dev = jax.devices()[0]
    print(f"strategist ES on {dev.device_kind}; building {cfg.tracks} circuits …", flush=True)
    rcfg = race.RaceConfig(time_limit=cfg.cap_s, pit_lane=True)
    F = race_field.RaceField(race.make_tracks(list(race.TRAIN_SEEDS)[: cfg.tracks], rcfg), n_cars=cfg.n_cars,
                             laps=cfg.laps, cfg=rcfg, gp_km=cfg.gp_km, tyre_cal=cfg.tyre_cal)
    dp, _, dk = cfg.driver.partition(":")
    driver, _ = load(pick_checkpoint(dp, dk or "last"))
    if resume:
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        params = {k: jnp.asarray(v.numpy()) for k, v in state["model"].items()}
        opt, first = jax.tree.map(jnp.asarray, state["optimizer"]), state["update"] + 1
    else:
        if cfg.init:
            ip, _, ik = cfg.init.partition(":")
            params = {k: jnp.asarray(v.numpy()) for k, v in torch.load(pick_checkpoint(ip, ik or "last"),
                                                                       map_location="cpu", weights_only=False)["model"].items()}
        else:
            params = init_params(cfg, jax.random.PRNGKey(cfg.seed))
        opt, first = adam_init(params), 1
        (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "git": _git_version(),
                                                         "device": f"1 x {dev.device_kind}"}, indent=2))
    play = make_race(cfg, F, driver)
    names, pairs = tuple(params), cfg.pop // 2

    def noise(k):
        return {n: jax.random.normal(kk, params[n].shape) for n, kk in zip(names, jax.random.split(k, len(names)))}

    @jax.jit
    def generation(center, key, keys):
        ks = jax.random.split(key, pairs)
        signs = jnp.concatenate([jnp.ones(pairs), -jnp.ones(pairs)])
        score = jax.lax.map(lambda x: -play({n: center[n] + x[1] * cfg.sigma * noise(x[0])[n] for n in names},
                                            keys)["pos"].mean(), (jnp.concatenate([ks, ks]), signs),
                            batch_size=cfg.pop)  # every member at once (one by one left the GPU mostly idle)
        ranks = centered_ranks(score)
        w = ranks[:pairs] - ranks[pairs:]
        g, _ = jax.lax.scan(lambda acc, kw: ({n: acc[n] + kw[1] * noise(kw[0])[n] for n in names}, None),
                            {n: jnp.zeros_like(center[n]) for n in names}, (ks, w))
        return {n: -g[n] / (pairs * cfg.sigma) for n in names}, score.mean(), score.max()

    bench_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 3000), max(cfg.bench, 1))
    bench = jax.jit(lambda p: jax.tree.map(lambda v: v.mean(), play(p, bench_keys)))
    meta = {"config": asdict(cfg), "game": "race_field", "method": cfg.method}

    def save(gn: int, stats: dict) -> None:
        torch.save({"model": to_torch(params), **meta, "update": gn, "stats": stats},
                   run_dir / "checkpoints" / f"u{gn:05d}.pt")
        torch.save({"model": to_torch(params), "optimizer": jax.tree.map(np.asarray, opt), **meta, "update": gn},
                   run_dir / "latest.pt")

    if not resume:
        save(0, {})
    log = open(run_dir / "metrics.jsonl", "a")
    try:
        tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))
    except Exception as e:  # no W&B (e.g. no key on a Kaggle script job): the metrics file is the record
        print("no W&B:", e, flush=True)
        tracker = None
    start = time.time()
    for gn in range(first, cfg.generations + 1):
        k_gen, k_rounds = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(cfg.seed + 2000), gn))
        grad, s_mean, s_best = generation(params, k_gen, jax.random.split(k_rounds, cfg.rounds))
        upd, opt = adam_step({n: params[n] for n in names}, opt, grad, cfg.lr_rel * cfg.sigma)
        params = dict(params, **upd)
        stats = {"update": gn, "time": time.time() - start, "members_pos": -float(s_mean), "best_member_pos": -float(s_best)}
        if cfg.bench:
            b = jax.device_get(bench(params))
            stats.update({f"bench_{k}": float(v) for k, v in b.items()})
        log.write(json.dumps(stats) + "\n")
        log.flush()
        if tracker:
            tracker.log(stats, step=gn)
        print(f"gen {gn:4d}/{cfg.generations}  members: mean position {stats['members_pos'] + 1:.2f}  "
              + (f"bench: strategist P{stats['bench_pos'] + 1:.2f} vs rule P{stats['bench_rule_pos'] + 1:.2f}, "
                 f"finished {stats['bench_finished']:.2f}/{cfg.n_learn} (rule {stats['bench_rule_finished']:.2f}), "
                 f"stops {stats['bench_stops'] / cfg.n_learn:.2f} a car (rule {stats['bench_rule_stops'] / cfg.n_learn:.2f}), "
                 f"soft starts {stats['bench_soft_start']:.2f}, hard {stats['bench_hard_start']:.2f}  " if cfg.bench else "")
              + f"{stats['time']:.0f} s", flush=True)
        if gn % cfg.checkpoint_every == 0 or gn == cfg.generations:
            save(gn, stats)
        if max_hours and time.time() - start > max_hours * 3600:
            save(gn, stats)
            print(f"stopping after {max_hours} h", flush=True)
            break
    if tracker:
        tracker.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--init", default="")
    ap.add_argument("--driver", default=StratConfig.driver)
    ap.add_argument("--pop", type=int, default=StratConfig.pop)
    ap.add_argument("--rounds", type=int, default=StratConfig.rounds)
    ap.add_argument("--tracks", type=int, default=StratConfig.tracks)
    ap.add_argument("--cap-s", type=float, default=StratConfig.cap_s)
    ap.add_argument("--bench", type=int, default=StratConfig.bench)
    a = ap.parse_args()
    train(StratConfig(generations=a.gens, init=a.init, driver=a.driver, pop=a.pop, rounds=a.rounds, tracks=a.tracks,
                      cap_s=a.cap_s, bench=a.bench), a.run, max_hours=a.hours)
