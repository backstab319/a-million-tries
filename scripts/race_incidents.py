"""Race day 2 (episode 16), step 9a: how often race control neutralises a race. Eight copies of the ep15 driver
(race-fin-7:50, blind to race control: zero weights on its senses) on the 8 test circuits, 25 laps, GP tyres, random
designs, every car on the team's rule (box at 60% worn, hards; under the VSC / Safety Car at 30% or on one compound),
race control on, no injected breakdowns. Per race: VSCs, Safety Cars and red flags, how long each lasted (the
leader's laps), wrecks, and the cars that went out within `--near` m of a car already stopped (ep15: 5 of 23 wrecks
were cars hitting an earlier wreck). ep15's pit lane by default (its driver doesn't stop on the new box, step 8c).

    uv run --group jax python scripts/race_incidents.py [--grids 2] [--driver runs/race-fin-7:50] [--new-lane]
    --save runs/<name>: also the first grid's races, every 0.25 s, for the TV view (scripts/race_tv.py): the cars as
    race_full --save has them, plus the race's status, each sector's flag, the Safety Car, penalties, the closed pit
    entry (a folder under runs/ because a Kaggle script job only brings those home)
    --save runs/<name> --full: also every circuit's whole motion every 0.1 s (race-full-c<K>.npz, race_full --save
    --full's format plus race control and the Safety Car's motion) for the 3D race (scripts/race_field_replay.py);
    --obs adds the drivers' senses (the onboard brain view; ~4x the size)
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.es_strategist import StratConfig, decide, make_field, n_senses, senses, widen
from ailearns.agents.ppo_car import forward, load
from ailearns.agents.ppo_field import widen_senses
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_control
from ailearns.games import race_car as rc

ap = argparse.ArgumentParser()
ap.add_argument("--driver", default="runs/race-fin-7:50")
ap.add_argument("--grids", type=int, default=2, help="races per test circuit (a new mixed grid each)")
ap.add_argument("--near", type=float, default=10.0, help="metres: out next to a stopped car (our estimate of a hit)")
ap.add_argument("--off", action="store_true", help="race control off (the baseline: the same races, no flags)")
ap.add_argument("--grid-only", type=int, default=None, help="only this many circuits (a quick look)")
ap.add_argument("--new-lane", action="store_true", help="the working lane and box point (default: ep15's lane)")
ap.add_argument("--cap", type=float, default=0.0, help="seconds: cut the races short (a smoke test; 0: 3600)")
ap.add_argument("--full", action="store_true", help="with --save: every circuit's whole motion too (the 3D race)")
ap.add_argument("--obs", action="store_true", help="with --full: the drivers' senses too (the onboard brain view)")
ap.add_argument("--key", type=int, default=200, help="the races' random base key (race_accept.py's are 100: --key 100 "
                "--new-lane replays its grid 1)")
ap.add_argument("--strategist", default="", help="runs/x:N: cars 1-4's stops called by this pit wall (as es_strategist's "
                "races: at its decision line, or when the VSC/Safety Car is deployed), 5-8 by the rule (ep16 16d)")
ap.add_argument("--old-crew", action="store_true", help="the pit crew before 14j (1 m across, 10 m pull-back, no "
                "wave-on): footage of the missed boxes")
ap.add_argument("--incidents", type=int, default=0, help="breakdowns injected a race, any car, at a random second in "
                "120-2400 s (es_strategist's --incidents: how the pit wall trained, and the 14i jam trace's races)")
ap.add_argument("--wall-line", action="store_true", help="footage only: the scripts drive the pit wall's line on "
                "episode 15's lane, the bug step 9c fixed (cars into the wall's end under a red flag)")
ap.add_argument("--save-grids", type=int, default=1, help="with --save: this many grids (grid g > 0 -> <save>-g<g>)")
ap.add_argument("--save", default="", help="a folder (runs/<name>): the first grid's races for scripts/race_tv.py")
a = ap.parse_args()

cfg = StratConfig(ep15_lane=not a.new_lane, race_control=True, **({"cap_s": a.cap} if a.cap else {}))
seeds = list(race.TEST_SEEDS)[:a.grid_only or 8]
F = make_field(cfg, seeds, **({"wall_line_scripts": True} if a.wall_line else {}))
if a.old_crew:  # (16d) set before anything is traced: the crew's rule is read from these when the step compiles
    F.box_side, F.pull_max, F.wave_on = 1.0, 10.0, False
ctl = race_control.RaceControl(F)
dp, _, dk = a.driver.partition(":")
driver = widen_senses(load(pick_checkpoint(dp, dk or "last"))[0], ctl.obs_size)
K, N, T = len(seeds), F.n, int(round(cfg.cap_s / F.cfg.dt))
lane = jax.vmap(F.game.in_pit_lane)
lo, span = jnp.asarray(rc.DESIGN_LOW), jnp.asarray(rc.DESIGN_HIGH - rc.DESIGN_LOW)
mine = jnp.arange(N) < (cfg.n_learn if a.strategist else 0)  # the pit wall's cars (the rest: the team's rule)
strat = None
if a.strategist:
    import torch
    sp, _, sk = a.strategist.partition(":")
    strat = {k: jnp.asarray(v.numpy()) for k, v in torch.load(pick_checkpoint(sp, sk or "last"), map_location="cpu",
                                                              weights_only=False)["model"].items()}
    if strat["w1"].shape[0] < n_senses(cfg):
        strat = widen(strat, n_senses(cfg))


def one(track, key):
    kd, kg = jax.random.split(key)
    lo_i, hi_i = (int(round(s_ / F.cfg.dt)) for s_ in cfg.incident_s)
    at = jax.random.randint(jax.random.fold_in(key, 11), (max(a.incidents, 1),), lo_i, hi_i)
    designs = lo + jax.random.uniform(kd, (N, len(rc.DESIGN))) * span
    medium = jnp.full(N, rc.COMPOUNDS.index("medium"))
    f = F.grid(kg, track, designs, medium)
    if strat is not None:  # the pit wall's starting tyre, from its senses on the grid (as es_strategist.make_race)
        _, start = decide(strat, jnp.concatenate([senses(F, f, f.cars.progress, 1.0),
                                                  ctl.strategist_senses(f, ctl.init(f))], -1))
        f = F.grid(kg, track, designs, jnp.where(mine, start, medium))
    L = F.tracks.length[track]

    def to_entry(c):
        return (-F.cfg.pit_entry - F.game.from_pit(c.track, c.idx)) % L

    def tick(carry, _):
        f, c, green_s, set_from, prev, prev_st = carry  # green_s: seconds since green came back after a
        # neutralisation (inf: none yet, or not green); the rest the pit wall's: laps on the set, metres to the entry
        if a.incidents:  # (16d) as es_strategist.make_race
            from ailearns.agents.bc_control import breakdown
            f = jax.lax.cond(jnp.any(f.t == at), lambda f: breakdown(F, f, jax.random.fold_in(key, f.t))[0],
                             lambda f: f, f)
        racing = ~(f.cars.crashed | f.finished | f.wrecked)
        st = c.status
        neutral = (st == race_control.VSC) | (st == race_control.SC)
        worst = f.cars.car.wear.max(-1)
        rule = (worst >= cfg.rule_wear) | (neutral & (to_entry(f.cars) >= cfg.reach_m)
                                           & ((worst > cfg.sc_wear) | (c.used.sum(-1) < 2)))
        call = racing & ~f.box_call & ~lane(f.cars) & rule
        plan = jnp.full(N, cfg.rule_plan)
        now = to_entry(f.cars)
        if strat is not None:  # (16d) es_strategist.make_race's calls for cars 1-4
            free = racing & ~f.box_call & ~lane(f.cars)
            deployed = neutral & (st != prev_st) & (prev_st != race_control.RED)
            at_line = (free & (prev > cfg.decide_m) & (now <= cfg.decide_m)) | (free & deployed & (now >= cfg.reach_m))
            pit, choice = decide(strat, jnp.concatenate([senses(F, f, set_from, 0.0), ctl.strategist_senses(f, c)], -1))
            call = jnp.where(mine, at_line & pit, call)
            plan = jnp.where(mine, choice, plan)
        f = f._replace(box_call=f.box_call | call, plan=jnp.where(call, plan, f.plan))
        out_before = (f.cars.crashed | f.wrecked | F.dry_out(f)) & ~f.finished & ~F.gone(f)
        if a.off:
            new, _ = F.step(f, jnp.clip(forward(driver, ctl.observe(f, c))[0], -1, 1))
            c_new = c
        else:
            new, c_new, _ = ctl.step(f, c, jnp.clip(forward(driver, ctl.observe(f, c))[0], -1, 1))
        new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), f, new)
        c = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), c, c_new)
        went_out = (new.cars.crashed | new.wrecked) & ~(f.cars.crashed | f.wrecked) & ~new.finished
        d = jnp.linalg.norm(new.cars.car.pos[:, None] - new.cars.car.pos[None], axis=-1)
        near = (out_before[None, :] & (d < a.near) & ~jnp.eye(N, dtype=bool)).any(1)
        running = ~(new.cars.crashed | new.wrecked | new.finished)
        lead = jnp.max(jnp.where(running | new.finished, F.laps_done(new), 0.0))
        # where it came to rest (stopped: `crashed`; a wreck once it has slid to a stop): across the track, in |lat|
        # bands: the tarmac's middle, its edge, the verge (not yet "off the track"), off the track
        rest = new.cars.crashed & ~f.cars.crashed & ~new.finished
        lat = jnp.abs(new.cars.lat)
        band = jnp.digitize(lat, jnp.array([3.0, F.cfg.width / 2, F.cfg.width / 2 + F.off_clear + 3.0]))
        bands = (jax.nn.one_hot(band, 4, dtype=jnp.int8) * rest[:, None]).sum(0)
        # when they went out: the race's status then (RACING, VSC, SC, SC_IN, RED), and green within 30 s of a
        # neutralisation ending (a restart)
        green_s = jnp.where(c.status == race_control.RACING,
                            jnp.where(st == race_control.RACING, green_s + F.cfg.dt, 0.0), jnp.inf)
        when = jnp.where(green_s < 30.0, 5, c.status)
        whens = (jax.nn.one_hot(when, 6, dtype=jnp.int8) * went_out[:, None]).sum(0)
        # (after the race is over its status is frozen: counted as racing, so a spell ends with the race)
        status = jnp.where(new.done, race_control.RACING, c.status)
        # the pit lane (step 9f): cars held by the crew (unsafe release: `service` parked at 1e-3, checked again),
        # contacts in the lane, cars out in it
        held = lambda x: jnp.abs(x.service - 1e-3) < 1e-6  # noqa: E731
        in_lane = lane(new.cars)
        pit = jnp.stack([(held(new) & ~held(f)).sum(), held(new).sum(),
                         (in_lane & (new.bump > 0) & (f.bump <= 0)).sum(), (in_lane & went_out).sum()])
        tv = {}
        if a.save:  # (step 9h) what the TV view draws: race_full --save's cars, and race control
            cars = new.cars
            tv = dict(pos=cars.car.pos, yaw=cars.car.yaw, progress=cars.progress, order=F.order(new),
                      finished=new.finished, walled=cars.crashed & ~new.wrecked, wrecked=new.wrecked,
                      dry=cars.car.fuel_used >= new.setup.fuel, pit=in_lane, stops=new.stops,
                      wear=cars.car.wear.mean(-1), fuel=1 - cars.car.fuel_used / new.setup.fuel, bump=new.bump > 0,
                      compound=new.setup.compound, gone=F.gone(new), status=status.astype(jnp.int8),
                      flags=ctl.flags(new).astype(jnp.int8), sc_pos=c.sc.car.pos[0], sc_yaw=c.sc.car.yaw[0],
                      sc_on=c.sc_on[0], penalty=ctl.penalty_s(c), pit_closed=c.pit_closed, breaks=c.breaks,
                      passes=c.passes, dbl=c.dbl, delta=c.delta, armed=c.armed,
                      dsq=ctl.disqualified(new, c), black=ctl.black_flagged(c),
                      classified=ctl.classify(new, c))
        if a.full:  # (step 9h) what scripts/race_replay.py draws a car from (as race_full's `motion`)
            def motion(st, pedal):
                return dict(vx=st.car.vx, vy=st.car.vy, steer=st.car.steer, gear=st.car.gear, ax=st.car.ax,
                            ay=st.car.ay, lat=st.lat, idx=st.idx, sliding=st.sliding, plank=st.plank.sum(-1),
                            surface=st.surface, pedal=pedal)
            tv.update(motion(cars, cars.last[:, 1]), wrecked_now=new.wrecked, impulse=new.bump,
                      **({"obs": ctl.observe(new, c).astype(jnp.float16)} if a.obs else {}),
                      **{f"sc_{n}": v[0] for n, v in motion(c.sc, c.sc.last[:, 1]).items()},
                      sc_progress=c.sc.progress[0])
        fresh = (new.stops > f.stops) | (new.cars.car.wear.max(-1) < f.cars.car.wear.max(-1) - 1e-6)  # new tyres
        set_from = jnp.where(fresh, new.cars.progress, set_from)
        return (new, c, green_s, set_from, now, st), (tv, status.astype(jnp.int8), lead.astype(jnp.float16), went_out.sum().astype(jnp.int8),
                          (went_out & near).sum().astype(jnp.int8), (went_out & new.wrecked).sum().astype(jnp.int8),
                          bands, whens, pit.astype(jnp.int8))

    c0 = ctl.init(f)
    (f, c, *_), rec = jax.lax.scan(tick, (f, c0, jnp.asarray(jnp.inf), f.cars.progress, to_entry(f.cars), c0.status),
                                   None, length=T)
    return {"designs": designs, "finish_t": f.finish_t, "finished": f.finished.sum(), "dsq": ctl.disqualified(f, c).sum(), "pen": ctl.penalty_s(c).sum(),
            "stops": f.stops.sum(), "vsc": c.vsc_n, "sc": c.sc_n, "red": c.red_n,
            # (step 9h) the penalties by kind (counts): sector breaks (VSC/SC minimum time), passes, double yellow,
            # closed pit entry
            "breaks": c.breaks.sum(), "passes": c.passes.sum(), "dbl": c.dbl.sum(), "closed": c.closed_in.sum()}, rec


play = jax.jit(jax.vmap(one))
print(f"{a.driver}, {'new' if a.new_lane else 'ep15'} pit lane, race control {'OFF' if a.off else 'on'}, stops: {a.strategist + ' (cars 1-4) v the rule' if a.strategist else 'the rule'}, {'old' if a.old_crew else '14j'} crew, {a.incidents} breakdowns a race: {a.grids} grids x {K} test "
      f"circuits, {cfg.laps} laps; devices {jax.devices()}", flush=True)
tot, spans, races = {}, {race_control.VSC: [], race_control.SC: [], race_control.RED: []}, 0
neutralised, outs, near_n, wrecks, where, when = 0, 0, 0, 0, np.zeros(4, int), np.zeros(6, int)
pits = np.zeros(4, int)
for g in range(a.grids):
    t0 = time.time()
    res, (tv, st, lead, out_n, near_k, wreck_n, bands, whens, pit) = jax.device_get(
        play(jnp.arange(K), jax.random.split(jax.random.PRNGKey(a.key + g), K)))
    designs, finish_t = res.pop("designs"), res.pop("finish_t")
    if a.save and g < a.save_grids:  # every 5th step (0.25 s), as race_full --save
        from pathlib import Path
        dest = Path(a.save if g == 0 else f"{a.save}-g{g}")
        dest.mkdir(parents=True, exist_ok=True)
        EVERY = 5
        np.savez_compressed(dest / "race.npz", seeds=np.array(seeds), dt=F.cfg.dt * EVERY, laps=cfg.laps,
                            team=np.asarray(mine), length=np.asarray(F.tracks.length),
                            box_gap=float(F.box_s[1] - F.box_s[0]), pit_outer=F.cfg.pit_outer,
                            pit_fast=F.cfg.pit_fast, sectors=np.asarray(ctl.sectors),
                            **{n: np.asarray(v)[:, EVERY - 1::EVERY] for n, v in tv.items()})
        print(f"-> {dest / 'race.npz'}", flush=True)
        for k in range(K) if a.full else ():  # every 2nd step (0.1 s), to 5 s after the race is over
            done = (tv["finished"][k] | tv["gone"][k]).all(-1)
            end = min(int(np.argmax(done)) + 100, T) if done.any() else T
            sl = slice(1, end, 2)
            np.savez_compressed(dest / f"race-full-c{k}.npz", seed=seeds[k], dt=F.cfg.dt * 2, laps=cfg.laps,
                                minutes=cfg.cap_s / 60, team=np.asarray(mine), designs=np.asarray(designs[k]),
                                finish_t=np.asarray(finish_t[k]), box_gap=float(F.box_s[1] - F.box_s[0]),
                                pit_outer=F.cfg.pit_outer, pit_fast=F.cfg.pit_fast, sectors=int(ctl.sectors[k]),
                                **{n: np.asarray(v)[k, sl] for n, v in tv.items()})
            print(f"-> {dest / f'race-full-c{k}.npz'}", flush=True)
        del tv
    for n, v in res.items():
        tot[n] = tot.get(n, 0) + float(np.sum(v))
    for k in range(K):
        s, ld = np.asarray(st[k]), np.asarray(lead[k], np.float32)
        neutral = np.isin(s, [race_control.VSC, race_control.SC, race_control.SC_IN, race_control.RED])
        neutralised += bool(neutral.any())
        # each spell: a run of one status (SC_IN counts with the Safety Car); its length in the leader's laps
        kind = np.where(s == race_control.SC_IN, race_control.SC, s)
        edges = np.flatnonzero(np.diff(np.r_[-1, kind, -1]))
        for i0, i1 in zip(edges[:-1], edges[1:]):
            if kind[i0] in spans:
                spans[kind[i0]].append((float(ld[i1 - 1] - ld[i0]), (i1 - i0) * F.cfg.dt))
        line = f"  circuit {seeds[k]}: " + " ".join(
            f"{n} {int(v[k])}" for n, v in res.items()) + f", out {int(out_n[k].sum())} ({int(near_k[k].sum())} by a stopped car)"
        print(line, flush=True)
    where += bands.sum((0, 1))
    when += whens.sum((0, 1))
    pits += pit.astype(int).sum((0, 1))
    outs, near_n, wrecks, races = outs + int(out_n.sum()), near_n + int(near_k.sum()), wrecks + int(wreck_n.sum()), races + K
    print(f"grid {g + 1}/{a.grids}: {time.time() - t0:.0f} s", flush=True)

cars = races * N
print(f"\n{races} races, {cars} cars: finished {tot['finished'] / cars:.0%}, stops {tot['stops'] / cars:.2f} a car, "
      f"penalties {tot['pen'] / cars:.1f} s a car, DSQ {tot['dsq'] / cars:.0%}")
print(f"penalties by kind, a car: sector breaks {tot['breaks'] / cars:.1f}, passes {tot['passes'] / cars:.1f}, "
      f"double yellow {tot['dbl'] / cars:.1f}, closed pit entry {tot['closed'] / cars:.2f}")
print(f"races neutralised (VSC, Safety Car or red): {neutralised}/{races}; VSC {tot['vsc']:.0f}, Safety Car "
      f"{tot['sc']:.0f}, red {tot['red']:.0f}")
for name, s in (("VSC", race_control.VSC), ("Safety Car (incl. in this lap)", race_control.SC), ("red flag", race_control.RED)):
    x = np.array(spans[s]).reshape(-1, 2)
    if len(x):  # (a spell can be cut short by a Safety Car or red flag: the VSC's then ends early)
        print(f"  {name}: {len(x)} spells, the leader's laps: median {np.median(x[:, 0]):.1f}, range "
              f"{x[:, 0].min():.1f}-{x[:, 0].max():.1f}; seconds: median {np.median(x[:, 1]):.0f}, range "
              f"{x[:, 1].min():.0f}-{x[:, 1].max():.0f}")
print(f"where they came to rest (|lat|): tarmac's middle (< 3 m) {where[0]}, tarmac's edge {where[1]}, verge (to "
      f"{F.cfg.width / 2 + F.off_clear + 3:.0f} m) {where[2]}, off the track {where[3]}")
print(f"when they went out: green {when[0]}, VSC {when[1]}, Safety Car {when[2]}, Safety Car in this lap {when[3]}, "
      f"red flag {when[4]}, the first 30 s of green after one {when[5]}")
print(f"cars out: {outs} ({wrecks} wrecked, {outs - wrecks} wall/stopped); out within {a.near:.0f} m of a car already "
      f"stopped: {near_n} ({near_n / max(outs, 1):.0%})")
print(f"pit lane ({tot['stops']:.0f} stops): unsafe-release holds {pits[0]} ({pits[1] * F.cfg.dt:.0f} s held in all), "
      f"contacts in the lane {pits[2]}, cars out in the lane {pits[3]}")
