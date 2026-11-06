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
- (episode 16, `race_control`) Race control on: flags, the VSC, the Safety Car, red flags, the stewards. The
  strategist sees 8 more numbers (`race_control.STRATEGIST_SENSES`: 33 in all; a 25-sense brain is widened with zero
  weights, so it decides as before until it learns them), and has a second decision moment: when the VSC or the
  Safety Car is deployed, for every car that can still make this lap's pit entry (`reach_m` before it or more), as
  real pit walls decide then. Its cars are placed after the stewards (`RaceControl.classify`), and the rule cars add
  the obvious human heuristic (plan section 4): under the VSC or the Safety Car, box if the worst tyre is over
  `sc_wear` worn or it hasn't raced on a second compound yet.
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
from ailearns.agents.ppo_field import widen_senses
from ailearns.games import race, race_control, race_field
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
    race_control: bool = False   # (ep16 step 8c) see the module docstring
    sc_wear: float = 0.3         # the rule cars box under the VSC / Safety Car past this wear (or on one compound)
    reach_m: float = 250.0       # a car can still make the pit entry from this far (braking from ~80 m/s: our estimate)
    # episode 15's pit lane (one 7 m lane, boxes 12 m apart, a stop anywhere across it), as `race_full.py --ep15-lane`:
    # ep15's driver never saw the working lane and doesn't stop on its box (ep16 step 8c: no car stopped, ~0 finished)
    ep15_lane: bool = False
    # (ep16 step 14b) the incident school: this many breakdowns a race (any car, as `bc_control.breakdown`: dead on the
    # spot with nobody 100 m behind, or a dead engine), each at a random second in `incident_s`, so the pit wall meets
    # Safety Cars and VSCs at every point of a stint (D1-8 alone: a Safety Car in ~1 race in 5)
    incidents: int = 0
    incident_s: tuple = (120.0, 2400.0)


def make_field(cfg: StratConfig, seeds=None, **race_kw) -> race_field.RaceField:
    """race_kw: more `race.RaceConfig` settings (footage switches, ep16 16e)."""
    lane = dict(pit_outer=-14.0, pit_fast=-7.0) if cfg.ep15_lane else {}
    boxes = dict(box_gap=12.0, box_side=10.0) if cfg.ep15_lane else {}
    rcfg = race.RaceConfig(time_limit=cfg.cap_s, pit_lane=True, **lane, **race_kw)
    seeds = list(race.TRAIN_SEEDS)[: cfg.tracks] if seeds is None else seeds
    return race_field.RaceField(race.make_tracks(seeds, rcfg), n_cars=cfg.n_cars, laps=cfg.laps, cfg=rcfg,
                                gp_km=cfg.gp_km, tyre_cal=cfg.tyre_cal, **boxes)


def n_senses(cfg: StratConfig) -> int:
    return N_SENSES + (len(race_control.STRATEGIST_SENSES) if cfg.race_control else 0)


def widen(p: dict, n: int) -> dict:
    """A strategist for more senses (race control's): zero weights on the new ones, appended after the old."""
    extra = n - p["w1"].shape[0]
    return dict(p, w1=jnp.concatenate([p["w1"], jnp.zeros((extra, p["w1"].shape[1]))]),
                lin=jnp.concatenate([p["lin"], jnp.zeros((extra, p["lin"].shape[1]))]))


def init_params(cfg: StratConfig, key) -> dict:
    """A small net plus a direct (linear) path; the direct path starts as the team's rule."""
    k1, k2 = jax.random.split(key)
    h, n = cfg.hidden, n_senses(cfg)
    p = {"w1": 0.1 * jax.random.normal(k1, (n, h)) / np.sqrt(n), "b1": jnp.zeros(h),
         "w2": 0.1 * jax.random.normal(k2, (h, 4)) / np.sqrt(h), "b2": jnp.zeros(4),
         "lin": jnp.zeros((n, 4))}
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


def make_race(cfg: StratConfig, F: race_field.RaceField, driver: dict, learner_rule: str | None = None,
              trace: bool = False):
    """(strategist, race key) -> each car's finishing position and what we watch.
    learner_rule (ep16 step 14f, the Safety Car check): cars 1-K drive a scripted rule instead of the strategist
    ("rule": the team's rule, a control; "no_sc": the rule without its stops under the VSC/Safety Car).
    trace (ep16 step 14h): also return every tick's status and each car's speed, pit lane flag, progress, box work."""
    K, N = cfg.n_learn, F.n
    T = int(round(cfg.cap_s / F.cfg.dt))
    g = F.game
    lane = jax.vmap(g.in_pit_lane)
    lo, span = jnp.asarray(rc.DESIGN_LOW), jnp.asarray(rc.DESIGN_HIGH - rc.DESIGN_LOW)
    mine = jnp.arange(N) < K
    ctl = race_control.RaceControl(F) if cfg.race_control else None
    if ctl and driver["body.0.weight"].shape[1] == F.obs_size:
        driver = widen_senses(driver, ctl.obs_size)

    if cfg.incidents:
        from ailearns.agents.bc_control import breakdown
    lo_i, hi_i = (int(round(s / F.cfg.dt)) for s in cfg.incident_s)

    def one(strat, key):
        kt, kd, kg, ks = jax.random.split(key, 4)
        at = jax.random.randint(jax.random.fold_in(key, 11), (max(cfg.incidents, 1),), lo_i, hi_i)
        track = jax.random.randint(kt, (), 0, F.tracks.length.shape[0])
        designs = lo + jax.random.uniform(kd, (N, len(rc.DESIGN))) * span
        f = F.grid(kg, track, designs)
        x = senses(F, f, f.cars.progress, 1.0)
        if ctl:  # (race control's senses on the grid: all green, nothing used, the pit entry where it is)
            x = jnp.concatenate([x, ctl.strategist_senses(f, ctl.init(f))], -1)
        _, start = decide(strat, x)  # the starting tyre
        comp = jnp.where(mine & (learner_rule is None), start, rc.COMPOUNDS.index("medium"))
        f = F.grid(kg, track, designs, comp)
        # grid slots shuffled every race (as race_school does): cars 1-4 mustn't always start at the front
        cars, perm = f.cars, jax.random.permutation(ks, N)
        o = jax.tree.map(lambda a: a[perm], cars)
        f = f._replace(cars=cars._replace(car=cars.car._replace(pos=o.car.pos, yaw=o.car.yaw), idx=o.idx, lat=o.lat,
                                          lap_pos=o.lap_pos, progress=o.progress))
        L = F.tracks.length[track]
        c0 = ctl.init(f) if ctl else None

        def see(f, c, set_from, grid_flag):
            x = senses(F, f, set_from, grid_flag)
            return jnp.concatenate([x, ctl.strategist_senses(f, c)], -1) if ctl else x

        def to_entry(c):  # metres to the pit entry ahead
            return (-F.cfg.pit_entry - g.from_pit(c.track, c.idx)) % L

        def tick(carry, _):
            f, c, set_from, prev, prev_st, sc_calls, sc_moments, pl = carry
            st_in = c.status if ctl else jnp.asarray(0)  # (14g) the status as a car enters the pit lane
            if cfg.incidents:
                f = jax.lax.cond(jnp.any(f.t == at), lambda f: breakdown(F, f, jax.random.fold_in(key, f.t))[0],
                                 lambda f: f, f)
            racing = ~(f.cars.crashed | f.finished | f.wrecked)
            now = to_entry(f.cars)
            free = racing & ~f.box_call & ~lane(f.cars)
            at_line = free & (prev > cfg.decide_m) & (now <= cfg.decide_m)
            pit, plan = decide(strat, see(f, c, set_from, 0.0))
            rule = f.cars.car.wear.max(-1) >= cfg.rule_wear
            if ctl:  # deployed (VSC, Safety Car, or the VSC turned Safety Car; not the restart after a red flag)
                st = c.status
                neutral = (st == race_control.VSC) | (st == race_control.SC)
                deployed = neutral & (st != prev_st) & (prev_st != race_control.RED)
                moment = free & deployed & (now >= cfg.reach_m)
                at_line = at_line | moment
                one = c.used.sum(-1) < 2
                rule = rule | (neutral & (now >= cfg.reach_m) & ((f.cars.car.wear.max(-1) > cfg.sc_wear) | one))
                sc_calls = sc_calls + (mine & moment & pit)
                sc_moments = sc_moments + (mine & moment)
                prev_st = st
            if learner_rule:
                wear_only = racing & ~f.box_call & (f.cars.car.wear.max(-1) >= cfg.rule_wear)
                pit = wear_only if learner_rule == "no_sc" else racing & ~f.box_call & rule
                at_line, plan = jnp.ones_like(at_line), jnp.full_like(plan, cfg.rule_plan)
            call = jnp.where(mine, at_line & pit, racing & ~f.box_call & rule)
            f = f._replace(box_call=f.box_call | call,
                           plan=jnp.where(call, jnp.where(mine, plan, cfg.rule_plan), f.plan))
            if ctl:
                acts = jnp.clip(forward(driver, ctl.observe(f, c))[0], -1, 1)
                new, c_new, _ = ctl.step(f, c, acts)
                c = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), c, c_new)
            else:
                new, _ = F.step(f, jnp.clip(forward(driver, F.observe(f))[0], -1, 1))
            new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), f, new)
            # new tyres (a stop, or a red flag's free set): laps on the set start again
            fresh = (new.stops > f.stops) | (new.cars.car.wear.max(-1) < f.cars.car.wear.max(-1) - 1e-6)
            set_from = jnp.where(fresh, new.cars.progress, set_from)
            # (ep16 step 14g) each pass through the pit lane: seconds lost to the field = its time in the lane x the share
            # of the field's metres (median of the racing cars on track) it didn't cover; by the status at entry
            was, now_in = lane(f.cars), lane(new.cars)
            alive = ~(new.cars.crashed | new.finished | new.wrecked)
            on_track = alive & ~now_in
            med = jnp.nanmedian(jnp.where(on_track, new.cars.progress, jnp.nan))
            t_s = new.t * F.cfg.dt
            enter, leave = ~was & now_in & alive, was & ~now_in & alive
            cat = jnp.where(st_in == race_control.VSC, 1, jnp.where(st_in == race_control.SC, 2, jnp.where(
                st_in == race_control.RED, 3, 0))) if ctl else jnp.asarray(0)
            t0, p0, m0, k0 = (jnp.where(enter, v, pl[n]) for n, v in
                              (("t0", t_s), ("p0", new.cars.progress), ("m0", med), ("k0", cat)))
            dur, fadv = t_s - pl["t0"], med - pl["m0"]
            loss = dur * (1.0 - (new.cars.progress - pl["p0"]) / jnp.maximum(fadv, 1.0))
            ok = leave & (fadv > 1.0) & (dur > 0) & jnp.isfinite(loss)
            hot = (pl["k0"][:, None] == jnp.arange(4)) & ok[:, None]  # (N, 4) green, VSC, SC, red
            slot = (jnp.arange(8)[None] == pl["i"][:, None]) & ok[:, None]  # each car's first 8 passes, for medians
            pl = dict(t0=t0, p0=p0, m0=m0, k0=k0, n=pl["n"] + hot.sum(0), s=pl["s"] + jnp.where(hot, loss[:, None], 0).sum(0),
                      s2=pl["s2"] + jnp.where(hot, loss[:, None] ** 2, 0).sum(0), i=pl["i"] + ok,
                      loss=jnp.where(slot, loss[:, None], pl["loss"]), cat=jnp.where(slot, pl["k0"][:, None], pl["cat"]))
            ys = dict(st=st_in, v=jnp.hypot(new.cars.car.vx, new.cars.car.vy), lane=now_in, prog=new.cars.progress,
                      service=new.service, alive=alive, pit_s=F.pit_s(new.cars), to_box=F.to_box(new.cars),
                      lat=new.cars.lat, served=new.served) if trace else None
            return (new, c, set_from, now, prev_st, sc_calls, sc_moments, pl), ys

        st0 = c0.status if ctl else jnp.asarray(0)
        z = jnp.zeros(N)
        pl0 = dict(t0=z, p0=z, m0=z, k0=jnp.zeros(N, jnp.int32), n=jnp.zeros(4), s=jnp.zeros(4), s2=jnp.zeros(4),
                   i=jnp.zeros(N, jnp.int32), loss=jnp.zeros((N, 8)), cat=jnp.full((N, 8), -1, jnp.int32))
        (f, c, _, _, _, sc_calls, sc_moments, pl), ys = jax.lax.scan(
            tick, (f, c0, f.cars.progress, to_entry(f.cars), st0, jnp.zeros(N, jnp.int32), jnp.zeros(N, jnp.int32), pl0),
            None, length=T)
        pos = ctl.classify(f, c) if ctl else F.order(f)
        out = f.cars.crashed | f.wrecked
        extra = {}
        if ctl:
            dsq, pen = ctl.disqualified(f, c), ctl.penalty_s(c)
            extra = {"dsq": dsq[:K].sum(), "rule_dsq": dsq[K:].sum(), "penalty_s": pen[:K].sum(),
                     "rule_penalty_s": pen[K:].sum(), "sc_calls": sc_calls[:K].sum(), "sc_moments": sc_moments[:K].sum(), "vsc": c.vsc_n, "sc": c.sc_n,
                     "red": c.red_n}
        extra.update({f"lane_{w}_{k}": pl[k][i] for i, w in enumerate(("green", "vsc", "sc", "red")) for k in ("n", "s", "s2")}, lane_loss=pl["loss"], lane_cat=pl["cat"])
        if trace:
            extra["trace"] = ys
        return {**extra, "pos": pos[:K].mean(), "rule_pos": pos[K:].mean(), "finished": f.finished[:K].sum(),
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
    F = make_field(cfg)
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
            if params["w1"].shape[0] < n_senses(cfg):
                params = widen(params, n_senses(cfg))
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
              + (f"DSQ {stats['bench_dsq']:.2f} (rule {stats['bench_rule_dsq']:.2f}), VSC/SC/red {stats['bench_vsc']:.2f}/"
                 f"{stats['bench_sc']:.2f}/{stats['bench_red']:.2f} a race, boxed at {stats['bench_sc_calls']:.2f} of "
                 f"{stats['bench_sc_moments']:.2f} deployments  " if cfg.bench and cfg.race_control else "")
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
    ap.add_argument("--race-control", action="store_true")
    ap.add_argument("--ep15-lane", action="store_true")
    ap.add_argument("--incidents", type=int, default=0)
    a = ap.parse_args()
    train(StratConfig(generations=a.gens, init=a.init, driver=a.driver, pop=a.pop, rounds=a.rounds, tracks=a.tracks,
                      cap_s=a.cap_s, bench=a.bench, race_control=a.race_control,
                      ep15_lane=a.ep15_lane, incidents=a.incidents), a.run, max_hours=a.hours)
