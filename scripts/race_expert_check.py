"""Race day 2 (episode 16), step 9d: the scripted experts' own test (plan section 5): each does its job in at least
95% of its cases, alone and in traffic, with no more cars out than the ep15 driver alone in the same races.

Races as `race_incidents.py` (8 copies of race-fin-7:50, the 8 test circuits, 25 laps, the team's rule) with injected
breakdowns so there are enough cases: at `--at` seconds (two a race) a random car still racing on the track stops,
half of them dead where they are (a car that stops on the racing line: the Safety Car; only one with no car within
100 m behind), half with a dead engine (it
coasts off to the run-off: a VSC or a yellow). The same races are run with the experts on (`race_experts.experts`)
and off (the driver alone). The broken-down car's own cases aren't scored. 25 laps cut off at 1800 s (the cases
come from the incidents, not the distance; first try: a breakdown every 500 s left one car of 8 running; a 12-lap
race wore the tyres out: the GP tyres' wear is set by the race distance).

A case is a car's spell in one situation, and its job:
- yellow: in a yellow or double yellow sector (racing, no VSC or Safety Car): no overtaking, no double yellow
  penalty, not out;
- VSC: no overtaking, no sector under the minimum time, not out;
- Safety Car (to green): no overtaking (the Safety Car too), no sector under the minimum, not out, and in the queue
  (within 10 car lengths of the car ahead) when it's called in, if it was;
- restart: the first 30 s of green after a Safety Car: not out;
- box: called in (the team's rule), until the call ends: stops, and not out.
A case still open when the race is cut off isn't scored. Each failure is broken down by its reason.
"In traffic": another racing car within 50 m ahead as the case began.

    uv run --group jax python scripts/race_expert_check.py [--grids 2] [--off | --both] [--ep15-lane]
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import race_experts as rx
from ailearns.agents.es_strategist import StratConfig, make_field
from ailearns.agents.ppo_car import forward, load
from ailearns.agents.ppo_field import widen_senses
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_control
from ailearns.games import race_car as rc

ap = argparse.ArgumentParser()
ap.add_argument("--driver", default="runs/race-fin-7:50")
ap.add_argument("--grids", type=int, default=2)
ap.add_argument("--at", default="300,900", help="seconds into the race of the injected breakdowns")
ap.add_argument("--laps", type=int, default=25)
ap.add_argument("--cap-s", type=float, default=1800.0)
ap.add_argument("--off", action="store_true", help="the experts off: the driver alone")
ap.add_argument("--both", action="store_true", help="on, then off (one cloud job)")
ap.add_argument("--ep15-lane", action="store_true", help="episode 15's pit lane (default: the new one, run 4 on)")
ap.add_argument("--grid-only", type=int, default=None)
a = ap.parse_args()

cfg = StratConfig(ep15_lane=a.ep15_lane, race_control=True, laps=a.laps, cap_s=a.cap_s)
seeds = list(race.TEST_SEEDS)[:a.grid_only or 8]
F = make_field(cfg, seeds)
ctl = race_control.RaceControl(F)
dp, _, dk = a.driver.partition(":")
driver = widen_senses(load(pick_checkpoint(dp, dk or "last"))[0], ctl.obs_size)
K, N, T, dt = len(seeds), F.n, int(round(cfg.cap_s / F.cfg.dt)), F.cfg.dt
AT = jnp.asarray([round(float(x) / dt) for x in a.at.split(',')])
lane = jax.vmap(F.game.in_pit_lane)
lo, span = jnp.asarray(rc.DESIGN_LOW), jnp.asarray(rc.DESIGN_HIGH - rc.DESIGN_LOW)
SCX = ("called in", "the box expert's zone", "going round a stopped car", "contact", "within 300 m of the pits",
       "first 10 s of the Safety Car", "wrecked", "in the pit lane")
KINDS = ("yellow", "VSC", "Safety Car", "restart", "box")
K5 = len(KINDS)
STOP = jnp.arange(K5) == 4  # the box's job needs a stop
# each kind's job, of (out, passed, under the minimum / double yellow, not queued); the box's is out and the stop (a
# called car under a VSC or Safety Car is also in that case, judged there)
JOB = jnp.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], bool)
R = race_control


def inject(f, key):
    """One car still racing on the track (not in the pit lane) breaks down: dead (stops where it is) or its engine."""
    ku, kk = jax.random.split(key)
    ok = ~(f.cars.crashed | f.wrecked | f.finished | F.dry_out(f)) & ~lane(f.cars)
    kind = jax.random.bernoulli(kk)
    # a dead stop only with no car within 100 m behind (first try: one stopped dead from 36 m/s 5 m ahead of another:
    # nobody avoids that, and real cars don't stop dead)
    L = F.tracks.length[f.cars.track[0]]
    behind = (f.cars.lap_pos[:, None] - f.cars.lap_pos[None, :]) % L  # (i, j) metres j is behind i
    clear = ~((ok[None, :] & ~jnp.eye(N, dtype=bool) & (behind < 100.0)).any(1))
    ok = ok & (clear | ~kind)
    k = jnp.argmax(jnp.where(ok, jax.random.uniform(ku, (N,)), -1.0))
    one = (jnp.arange(N) == k) & ok.any()
    dead = one & kind
    car = f.cars.car._replace(vx=jnp.where(dead, 0.0, f.cars.car.vx), vy=jnp.where(dead, 0.0, f.cars.car.vy),
                              fuel_used=jnp.where(one & ~dead, f.setup.fuel, f.cars.car.fuel_used))
    return f._replace(cars=f.cars._replace(car=car, crashed=f.cars.crashed | dead, done=f.cars.done | dead)), one


def one(track, key, off):
    kd, kg, ki = jax.random.split(key, 3)
    designs = lo + jax.random.uniform(kd, (N, len(rc.DESIGN))) * span
    f = F.grid(kg, track, designs, jnp.full(N, rc.COMPOUNDS.index("medium")))
    L = F.tracks.length[track]

    def to_entry(c):
        return (-F.cfg.pit_entry - F.game.from_pit(c.track, c.idx)) % L

    def tick(carry, k):
        f, c, green_s, was_sc, case, why, busy, got, hit, n, ok, out_n, whys, sc_t, scx, pitc = carry
        t = f.t
        f, now_hit = jax.lax.cond(jnp.any(t == AT), inject, lambda f, k: (f, jnp.zeros(N, bool)), f, k)
        hit = hit | now_hit
        racing = ~(f.cars.crashed | f.finished | f.wrecked)
        st = c.status
        neutral = (st == R.VSC) | (st == R.SC)
        worst = f.cars.car.wear.max(-1)
        rule = (worst >= cfg.rule_wear) | (neutral & (to_entry(f.cars) >= cfg.reach_m)
                                           & ((worst > cfg.sc_wear) | (c.used.sum(-1) < 2)))
        call = racing & ~f.box_call & ~lane(f.cars) & rule
        f = f._replace(box_call=f.box_call | call, plan=jnp.where(call, cfg.rule_plan, f.plan))
        own = jnp.clip(forward(driver, ctl.observe(f, c))[0], -1, 1)
        ex, zones = rx.experts(ctl, f, c, own)
        act = own if off else ex
        new, c_new, _ = ctl.step(f, c, act)
        new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), f, new)
        c_new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), c, c_new)

        # each car's situation now (its cases)
        running = ctl.running(new) & ~lane(new.cars)
        flag = ctl.flags(new)[ctl.sector(new.cars)] != R.GREEN
        s2 = c_new.status
        green_s = jnp.where(s2 == R.RACING, jnp.where(st == R.RACING, green_s + dt, 0.0), jnp.inf)
        was_sc = jnp.where(s2 == R.RACING, was_sc | (st == R.SC_IN), False)
        each = jnp.ones(N, bool)
        now = jnp.stack([flag & (s2 == R.RACING), each & (s2 == R.VSC), each & ((s2 == R.SC) | (s2 == R.SC_IN)),
                         each & (s2 == R.RACING) & was_sc & (green_s < 30.0), new.box_call], 1)
        now = now & (running[:, None] | case)  # an open case stays open while the car is out (it closes when the spell ends)
        went_out = (new.cars.crashed | new.wrecked) & ~(f.cars.crashed | f.wrecked) & ~new.finished
        # every car out under the Safety Car (status SC or in this lap, not the broken-down one), described
        sc_now = (c.status == R.SC) | (c.status == R.SC_IN)
        sc_t = jnp.where(sc_now, sc_t + dt, 0.0)
        sp = jax.vmap(F.game.from_pit)(f.cars.track, f.cars.idx)
        desc = jnp.stack([f.box_call, zones[:, rx.BOX], zones[:, rx.DODGE], new.bump > 0, jnp.abs(sp) < 300.0,
                          jnp.broadcast_to(sc_t < 10.0, (N,)), new.wrecked, lane(f.cars)], 1)
        o = went_out & sc_now & ~hit
        scx = scx + jnp.concatenate([(o[:, None] & desc).sum(0), o.sum()[None]])
        passed = (c_new.owe > c.owe).any(1) | (c_new.owe_sc > c.owe_sc)
        broke = (c_new.breaks > c.breaks) | (c_new.dbl > c.dbl)
        gap = (new.cars.lap_pos[None, :] - new.cars.lap_pos[:, None]) % L
        ahead = (running[None, :] & ~jnp.eye(N, dtype=bool) & (gap < 50.0)).any(1)
        Lq = (c_new.sc.lap_pos[0] - new.cars.lap_pos) % L
        d = jnp.where(running, Lq, jnp.inf)
        nxt = jnp.where(d[None, :] < d[:, None], d[:, None] - d[None, :], d[:, None]).min(1)  # to the car ahead or SC
        called_in = (s2 == R.SC_IN) & (st == R.SC)
        unqueued = called_in & running & (nxt > ctl.queue_m)

        start = now & ~case
        busy = jnp.where(start, ahead[:, None], busy)
        why = jnp.where(start[..., None], False, why) | ((case | start)[..., None]
                                                         & jnp.stack([went_out, passed, broke, unqueued], -1)[:, None])
        outk = jnp.where(start, False, out_n[1]) | (case | start) & went_out[:, None]
        got = jnp.where(start, False, got) | (new.stops > f.stops)[:, None]
        # (a case still open when the race is cut off isn't scored: a call open at 1800 s isn't a missed stop)
        end = (case | start) & ~now & ~new.done & ~hit[:, None]
        bad = (why & JOB).any(-1)
        whys = whys + (end[..., None] & jnp.concatenate([why & JOB, (STOP & ~got)[..., None]], -1)).sum(0)
        idx = busy.astype(jnp.int32)
        n = n + jax.vmap(lambda e, i: jax.nn.one_hot(i, 2) * e[:, None], 1, 0)(end, idx).sum(1)
        ok = ok + jax.vmap(lambda e, i: jax.nn.one_hot(i, 2) * e[:, None], 1, 0)(end & ~bad & (got | ~STOP), idx).sum(1)
        outs = out_n[0] + jax.vmap(lambda e, i: jax.nn.one_hot(i, 2) * e[:, None], 1, 0)(end & outk, idx).sum(1)
        case = now & ~new.done
        # the pit lane (step 9f): unsafe-release holds (`service` parked at exactly 1e-3) and the time held, contacts
        # in the lane (of them under the red flag: its queue), cars out in the lane
        held = lambda x: jnp.abs(x.service - 1e-3) < 1e-6  # noqa: E731
        touch = lane(new.cars) & (new.bump > 0) & (f.bump <= 0)
        pitc = pitc + jnp.stack([(held(new) & ~held(f)).sum(), held(new).sum(), touch.sum(),
                                 (touch & (st == R.RED)).sum(), (lane(new.cars) & went_out).sum()])
        return (new, c_new, green_s, was_sc, case, why, busy, got, hit, n, ok, (outs, outk), whys, sc_t, scx, pitc), None

    z, zb = jnp.zeros((K5, 2)), jnp.zeros((N, K5), bool)
    carry = (f, ctl.init(f), jnp.asarray(jnp.inf), jnp.asarray(False), zb, jnp.zeros((N, K5, 4), bool), zb, zb,
             jnp.zeros(N, bool), z, z, (z, zb), jnp.zeros((K5, 5)), jnp.asarray(0.0),
             jnp.zeros(len(SCX) + 1), jnp.zeros(5, jnp.int32))
    carry, _ = jax.lax.scan(tick, carry, jax.random.split(ki, T))
    f, c, *_, n, ok, (outs, _), whys, _, scx, pitc = carry
    return {"finished": f.finished.sum(), "pen": ctl.penalty_s(c).sum(), "vsc": c.vsc_n, "sc": c.sc_n,
            "red": c.red_n, "out": (f.cars.crashed | f.wrecked).sum(), "stops": f.stops.sum()}, n, ok, outs, whys, scx, pitc




def report(off):
    play = jax.jit(jax.vmap(lambda t, k: one(t, k, off)))
    print(f"{a.driver}, experts {'OFF' if off else 'on'}, {'ep15' if a.ep15_lane else 'new'} pit lane, {a.laps} laps, "
          f"breakdowns at {a.at} s: {a.grids} grids x {K} test circuits; devices {jax.devices()}", flush=True)
    tot, n, ok, outs, whys = {}, np.zeros((K5, 2)), np.zeros((K5, 2)), np.zeros((K5, 2)), np.zeros((K5, 5))
    scx, pits = np.zeros(len(SCX) + 1), np.zeros(5)
    for g in range(a.grids):
        t0 = time.time()
        res, nn, kk, oo, ww, xx, pp = jax.device_get(play(jnp.arange(K), jax.random.split(jax.random.PRNGKey(300 + g), K)))
        for k_, v in res.items():
            tot[k_] = tot.get(k_, 0) + float(np.sum(v))
        n, ok, outs, whys, scx = n + nn.sum(0), ok + kk.sum(0), outs + oo.sum(0), whys + ww.sum(0), scx + xx.sum(0)
        pits = pits + pp.sum(0)
        print(f"grid {g + 1}/{a.grids}: {time.time() - t0:.0f} s", flush=True)
    cars = a.grids * K * N
    print(f"\n{a.grids * K} races: finished {tot['finished'] / cars:.0%}, out {tot['out']:.0f}, penalties "
          f"{tot['pen'] / cars:.1f} s a car; VSC {tot['vsc']:.0f}, Safety Car {tot['sc']:.0f}, red {tot['red']:.0f}")
    for i, name in enumerate(KINDS):
        cells = []
        for j, where in enumerate(("alone", "in traffic")):
            m = max(n[i, j], 1)
            cells.append(f"{where} {ok[i, j] / m:4.0%} of {n[i, j]:4.0f} (out {outs[i, j] / m:4.0%})")
        print(f"  {name:11s} job done: " + ", ".join(cells))
        print("              failed on: " + ", ".join(f"{w} {int(x)}" for w, x in zip(
            ("out", "passed", "under the minimum / double yellow", "not queued", "no stop"), whys[i]) if x))

    print(f"  out under the Safety Car: {int(scx[-1])}; of them: " + ", ".join(
        f"{w} {int(x)}" for w, x in zip(SCX, scx[:-1])), flush=True)
    print(f"  pit lane ({tot['stops']:.0f} stops): unsafe-release holds {int(pits[0])} ({pits[1] * dt:.0f} s held), "
          f"contacts in the lane {int(pits[2])} ({int(pits[3])} under the red flag), out in the lane {int(pits[4])}",
          flush=True)

for off in ((False, True) if a.both else (a.off,)):
    report(off)
