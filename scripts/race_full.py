"""Race day (episode 15), step 9: the full race. Eight copies of one trained driver on a race grid, F1-like cars on
mediums with the fuel allowance, 25 laps on each test circuit (never trained on), the team calling a car in when its
tyres reach `--call-wear` (hards fitted), as in finish-mode training. Who finishes, in what order, the stops, the
wrecks, who runs dry; and every car's state every 0.25 s for the footage.

    uv run --group jax python scripts/race_full.py [--driver runs/race-fin-4:100] [--n 8] [--save]
    --save -> out/ep15/race-<driver>.npz (scripts/race_tv.py draws it)
    --strategist runs/strat-2:43 -> cars 1-4 run by the strategist (grid tyre + the once-a-lap call), 5-8 by the team's
    rule (--call-worst --call-wear 0.6 --gp-km 305 --random-designs: as strat_compare); grid slots shuffled
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo_car import forward, load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_field
from ailearns.games import race_car as rc

ap = argparse.ArgumentParser()
ap.add_argument("--driver", default="runs/race-fin-4:100")
ap.add_argument("--n", type=int, default=8, help="test circuits")
ap.add_argument("--laps", type=int, default=25)
ap.add_argument("--minutes", type=float, default=45.0, help="time cap")
ap.add_argument("--call-wear", type=float, default=0.7)
ap.add_argument("--call-worst", action="store_true", help="the team calls on the most worn tyre, not the mean of four")
ap.add_argument("--plan", type=int, default=2, help="compound fitted at a stop (rc.COMPOUNDS; 2: hard)")
ap.add_argument("--start", type=int, default=1, help="compound every car starts on (rc.COMPOUNDS: 0 soft, 1 medium)")
ap.add_argument("--gp-km", type=float, default=0.0, help="scale tyre wear to a Grand Prix of this many km (305; 0: off)")
ap.add_argument("--tyre-cal", type=float, default=1.9, help="with --gp-km: x wear, to real tyre lifespans (step 9e)")
ap.add_argument("--random-designs", action="store_true", help="each car a random design, as in training (default: "
                "8 identical F1-like cars)")
ap.add_argument("--compare", nargs="*", default=[], help="instead of one race: these drivers (runs/x:N), each over "
                "--grids mixed fields (random designs, a new grid draw each), totals only")
ap.add_argument("--grids", type=int, default=4)
ap.add_argument("--strategist", default="", help="runs/x:N: cars 1-4 run by this strategist, 5-8 by the rule")
ap.add_argument("--save", action="store_true")
ap.add_argument("--full", type=int, default=None, help="with --save: also every car's whole motion on this circuit "
                "(index, 0-7) every 0.1 s, for the 3D race (scripts/race_field_replay.py)")
a = ap.parse_args()

cfg = race.RaceConfig(time_limit=a.minutes * 60, pit_lane=True)
seeds = list(race.TEST_SEEDS)[:a.n]
F = race_field.RaceField(race.make_tracks(seeds, cfg), laps=a.laps, cfg=cfg, gp_km=a.gp_km, tyre_cal=a.tyre_cal)
path, _, ck = a.driver.partition(":")
params = None if a.compare else load(pick_checkpoint(path, ck or "last"))[0]  # (compare loads its own)
K, N, steps = len(seeds), F.n, int(cfg.time_limit / cfg.dt)
designs = jnp.tile(jnp.asarray(rc.F1_LIKE[0])[None], (N, 1))
if a.random_designs:  # race_school's draw: every dial uniform over its range
    lo, span = jnp.asarray(rc.DESIGN_LOW), jnp.asarray(rc.DESIGN_HIGH - rc.DESIGN_LOW)
    designs = lo + jax.random.uniform(jax.random.PRNGKey(7), (N, len(rc.DESIGN))) * span
comp = jnp.full(N, a.start)
strat = None
if a.strategist:  # (step 9j) the trained pit wall against the team's rule
    import torch

    from ailearns.agents.es_strategist import StratConfig, decide, senses
    sp, _, sk = a.strategist.partition(":")
    ck_s = torch.load(pick_checkpoint(sp, sk or "last"), map_location="cpu", weights_only=False)
    strat = {k: jnp.asarray(v.numpy()) for k, v in ck_s["model"].items()}
    DECIDE = StratConfig().decide_m
mine = (jnp.arange(N) < 4) if strat is not None else jnp.zeros(N, bool)
EVERY = 5  # saved frames: every 0.25 s
lane = jax.vmap(F.game.in_pit_lane)


def to_entry(c):  # metres to the pit entry ahead (where the strategist decides from)
    return (-cfg.pit_entry - F.game.from_pit(c.track, c.idx)) % F.tracks.length[c.track]


def body(carry, _):
    f, set_from, prev, held = carry   # held: last step's strategist decision (the replay keeps every 2nd step)
    racing = ~(f.cars.crashed | f.finished | f.wrecked)
    call = racing & ~f.box_call & ((f.cars.car.wear.max(-1) if a.call_worst else f.cars.car.wear.mean(-1)) >= a.call_wear)  # the team
    plan, now = jnp.full(N, a.plan), prev
    if strat is not None:  # its cars: called only as they pass the decision line, if it says pit
        now = to_entry(f.cars)
        at_line = racing & ~f.box_call & ~lane(f.cars) & (prev > DECIDE) & (now <= DECIDE)
        x = senses(F, f, set_from, 0.0)
        pit, its_plan = decide(strat, x)
        call, plan = jnp.where(mine, at_line & pit, call), jnp.where(mine, its_plan, plan)
        line_now, call_now = at_line | held[0], (at_line & pit) | held[1]
    f = f._replace(box_call=f.box_call | call, plan=jnp.where(call, plan, f.plan))
    obs = F.observe(f)
    new, _ = F.step(f, jnp.clip(forward(params, obs)[0], -1, 1))
    new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), f, new)  # the race is over: hold it
    set_from = jnp.where(new.stops > f.stops, new.cars.progress, set_from)
    c = new.cars
    dry = c.car.fuel_used >= new.setup.fuel
    motion = {} if a.full is None else dict(  # what scripts/race_replay.py draws a car from
        vx=c.car.vx, vy=c.car.vy, steer=c.car.steer, gear=c.car.gear, ax=c.car.ax, ay=c.car.ay, lat=c.lat, idx=c.idx,
        sliding=c.sliding, plank=c.plank.sum(-1), surface=c.surface, pedal=c.last[:, 1], wrecked_now=new.wrecked,
        impulse=new.bump, obs=obs.astype(jnp.float16),   # (step 10d) contact strength for the sound; the driver's senses
        **({} if strat is None else dict(strat_x=x.astype(jnp.float16), strat_line=line_now, strat_call=call_now)))
    held = (at_line, at_line & pit) if strat is not None else held
    return (new, set_from, now, held), ((c.car.pos, c.car.yaw, c.progress, F.order(new), new.finished, c.crashed & ~new.wrecked, new.wrecked,
                 dry, lane(c), new.stops, c.car.wear.mean(-1), 1 - c.car.fuel_used / new.setup.fuel, new.bump > 0,
                 new.setup.compound, F.gone(new)), motion)


def one(track, key):
    f = F.grid(key, track, designs, comp)
    if strat is not None:  # its starting tyre, then grid slots shuffled (as strat_compare: cars 1-4 not always in front)
        _, start = decide(strat, senses(F, f, f.cars.progress, 1.0))
        f = F.grid(key, track, designs, jnp.where(mine, start, comp))
        cars, perm = f.cars, jax.random.permutation(jax.random.fold_in(key, 1), N)
        o = jax.tree.map(lambda x: x[perm], cars)
        f = f._replace(cars=cars._replace(car=cars.car._replace(pos=o.car.pos, yaw=o.car.yaw), idx=o.idx, lat=o.lat,
                                          lap_pos=o.lap_pos, progress=o.progress))
    held = (jnp.zeros(N, bool), jnp.zeros(N, bool))
    (f, _, _, _), out = jax.lax.scan(body, (f, f.cars.progress, to_entry(f.cars), held), None, length=steps)
    return f, out


if a.compare:  # (step 9d) one race per circuit is too noisy to rank drivers (one pile-up takes out several cars)
    def race_all(p, designs, key):
        def body2(f, _):
            racing = ~(f.cars.crashed | f.finished | f.wrecked)
            call = racing & ~f.box_call & ((f.cars.car.wear.max(-1) if a.call_worst else f.cars.car.wear.mean(-1)) >= a.call_wear)
            f = f._replace(box_call=f.box_call | call, plan=jnp.where(call, a.plan, f.plan))
            new, _ = F.step(f, jnp.clip(forward(p, F.observe(f))[0], -1, 1))
            return jax.tree.map(lambda x, y: jnp.where(f.done, x, y), f, new), None
        f = jax.vmap(lambda t, k: jax.lax.scan(body2, F.grid(k, t, designs, comp), None, length=steps)[0])(
            jnp.arange(K), jax.random.split(key, K))
        dry = f.cars.car.fuel_used >= f.setup.fuel
        return {"finished": f.finished.sum(), "wrecked": f.wrecked.sum(), "wall": (f.cars.crashed & ~f.wrecked).sum(),
                "dry": (dry & ~f.finished).sum(), "stops": f.stops.sum()}
    race_all = jax.jit(race_all)
    lo, span = jnp.asarray(rc.DESIGN_LOW), jnp.asarray(rc.DESIGN_HIGH - rc.DESIGN_LOW)
    fields = [lo + jax.random.uniform(jax.random.PRNGKey(7 + g), (N, len(rc.DESIGN))) * span for g in range(a.grids)]
    cars = a.grids * K * N
    print(f"{a.grids} mixed grids x {K} test circuits x {N} cars = {cars} car-races per driver, {a.laps} laps; "
          f"devices {jax.devices()}", flush=True)
    print("| driver | finished | wrecked | wall | stranded dry | stops |\n|---|---|---|---|---|---|")
    for d in a.compare:
        dp, _, dk = d.partition(":")
        p, _ = load(pick_checkpoint(dp, dk or "last"))
        t = {}
        for g, des in enumerate(fields):
            t0 = time.time()
            for n, v in jax.device_get(race_all(p, des, jax.random.PRNGKey(100 + g))).items():
                t[n] = t.get(n, 0) + int(v)
            print(f"  {d} grid {g + 1}/{a.grids}: {time.time() - t0:.0f} s", flush=True)   # a stalled job shows
        print(f"| {d} | {t['finished'] / cars:.0%} | {t['wrecked'] / cars:.0%} | {t['wall'] / cars:.0%} | "
              f"{t['dry'] / cars:.0%} | {t['stops'] / cars:.2f} |", flush=True)
    raise SystemExit
f, (out, motion) = jax.jit(jax.vmap(one))(jnp.arange(K), jax.random.split(jax.random.PRNGKey(0), K))
(pos, yaw, prog, order, finished, walled, wrecked, dry, pit, stops, wear, fuel, bump, compound, gone) = [np.asarray(x) for x in out]
finish_t = np.asarray(f.finish_t)
print(f"{a.driver}: {N} cars, {a.laps} laps, {K} test circuits, {'random designs' if a.random_designs else 'F1-like'} on {rc.COMPOUNDS[a.start]}s, team calls at wear {a.call_wear}")
for k in range(K):
    L = float(F.tracks.length[k])
    end = -1
    fin = finished[k, end]
    how = np.where(fin, "finished", np.where(wrecked[k, end], "wrecked", np.where(walled[k, end], "wall",
                   np.where(dry[k, end], "dry", "running"))))
    t_win = np.nanmin(np.where(fin, finish_t[k], np.nan)) if fin.any() else np.nan
    print(f"circuit {seeds[k]} ({L / 1000:.1f} km): finished {fin.sum()}/{N}, winner {t_win / 60:.1f} min, "
          f"stops {stops[k, end].sum()}, out: wall {walled[k, end].sum()}, wrecked {wrecked[k, end].sum()}, "
          f"dry {(dry[k, end] & ~fin).sum()}  |  "
          + " ".join(f"P{int(order[k, end, i]) + 1}:{how[i][:3]}/{stops[k, end, i]}st" for i in np.argsort(order[k, end])))
o = finished[:, -1]
print(f"all: {o.mean():.0%} finished, {wrecked[:, -1].mean():.0%} wrecked, {walled[:, -1].mean():.0%} wall, "
      f"{(dry[:, -1] & ~o).mean():.0%} stranded dry, {stops[:, -1].mean():.2f} stops per car")
if strat is not None:
    m = np.asarray(mine)
    for who, sel in ((f"strategist {a.strategist} (cars 1-4)", m), ("the rule (cars 5-8)", ~m)):
        print(f"{who}: P{order[:, -1][:, sel].mean() + 1:.2f}, finished {o[:, sel].mean():.0%}, "
              f"{stops[:, -1][:, sel].mean():.2f} stops a car, started {[rc.COMPOUNDS[x] for x in compound[:, 0][:, sel][0]]}")
if a.save:
    dest = Path("out/ep15")
    dest.mkdir(parents=True, exist_ok=True)
    s = slice(EVERY - 1, None, EVERY)
    name = path.split("/")[-1] + (f"-{ck}" if ck else "") + ("-soft" if a.start == 0 else "") + ("-gp" if a.gp_km else "")
    if a.strategist:
        name += "-" + a.strategist.split("/")[-1].replace(":", "-")
    np.savez_compressed(dest / f"race-{name}.npz", seeds=np.array(seeds), dt=cfg.dt * EVERY, laps=a.laps, team=np.asarray(mine),
                        length=np.asarray(F.tracks.length), finish_t=finish_t,
                        **{n: v[:, s] for n, v in dict(pos=pos, yaw=yaw, progress=prog, order=order, finished=finished,
                                                       walled=walled, wrecked=wrecked, dry=dry, pit=pit, stops=stops,
                                                       wear=wear, fuel=fuel, bump=bump, compound=compound,
                                                       gone=gone).items()})
    print(f"-> {dest / f'race-{name}.npz'}")
    if a.full is not None:  # one circuit, every 2nd step (0.1 s), up to 5 s after the race ends
        k, done = a.full, np.asarray(f.done)[a.full]
        end = steps if not np.all(finished[k, -1] | gone[k, -1]) else min(
            int(np.argmax((finished[k] | gone[k]).all(-1))) + 100, steps)
        sl = slice(1, end, 2)
        full = dest / f"race-{name}-c{k}-full.npz"
        np.savez_compressed(full, seed=seeds[k], dt=cfg.dt * 2, laps=a.laps, minutes=a.minutes, team=np.asarray(mine),
                            designs=np.asarray(designs), finish_t=finish_t[k], pos=pos[k, sl], yaw=yaw[k, sl],
                            progress=prog[k, sl], order=order[k, sl], finished=finished[k, sl], walled=walled[k, sl],
                            wrecked=wrecked[k, sl], dry=dry[k, sl], pit=pit[k, sl], stops=stops[k, sl], wear=wear[k, sl],
                            fuel=fuel[k, sl], bump=bump[k, sl], compound=compound[k, sl], gone=gone[k, sl],
                            **{n: np.asarray(v)[k, sl] for n, v in motion.items()})
        print(f"-> {full} ({(end - 1) // 2 + 1} frames)")
