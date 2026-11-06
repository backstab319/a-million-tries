"""Race day 2 (episode 16), step 15: the acceptance test (plan section 6). Whole mixed races with race control on and
natural incidents only (no injected breakdowns): 4 mixed grids x 8 test circuits x 8 cars = 256 car-races a driver, 25
laps, GP tyres, the team's rule calling the stops. Each driver is scored against the baseline, episode 15's driver
with the hard-coded experts (`race_experts`) doing the flags for it.

    uv run --group jax python scripts/race_accept.py --drivers runs/race-d1-2:12 [--grids 4]
    (Kaggle CPU: ailearns cloud script "scripts/race_accept.py --baseline runs/race-fin-7:50 --drivers runs/race-d1-2:12"
    --on kaggle; the baseline named too: only checkpoints in the command are shipped)

The driver passes if: penalties per car-race under 0.2; cars hitting a stopped car down by at least half; finished
share and race time no worse than the baseline; no lap-1 or restart pile-up worse than the baseline's.
(`race_full.py --compare` is episode 15's version, without race control; this is its race-control twin.)
"""

import argparse
import time

import jax
import jax.numpy as jnp

from ailearns.agents import race_experts as rx
from ailearns.agents.es_strategist import StratConfig, make_field
from ailearns.agents.ppo_car import forward, load
from ailearns.agents.ppo_field import widen_senses
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_control
from ailearns.games import race_car as rc

ap = argparse.ArgumentParser()
ap.add_argument("--drivers", nargs="+", default=["runs/race-d1-2:12"])
ap.add_argument("--baseline", default="runs/race-fin-7:50", help="driven with the experts on")
ap.add_argument("--grids", type=int, default=4)
ap.add_argument("--laps", type=int, default=25)
ap.add_argument("--cap-s", type=float, default=3600.0)
ap.add_argument("--n", type=int, default=8, help="test circuits")
ap.add_argument("--trace", action="store_true", help="the hits on stopped cars per race, as episodes (a new one after "
                "2 s without one) and by speed")
a = ap.parse_args()

cfg = StratConfig(race_control=True, laps=a.laps, cap_s=a.cap_s)
seeds = list(race.TEST_SEEDS)[:a.n]
F = make_field(cfg, seeds)
ctl = race_control.RaceControl(F)
R = race_control
K, N, T, dt = len(seeds), F.n, int(round(cfg.cap_s / F.cfg.dt)), F.cfg.dt
lane = jax.vmap(F.game.in_pit_lane)
lo, span = jnp.asarray(rc.DESIGN_LOW), jnp.asarray(rc.DESIGN_HIGH - rc.DESIGN_LOW)
HIT_M = 8.0  # a car touching another within this of a stopped car's centre hit the stopped car


def one(params, experts, track, key):
    kd, kg = jax.random.split(key)
    designs = lo + jax.random.uniform(kd, (N, len(rc.DESIGN))) * span
    f = F.grid(kg, track, designs, jnp.full(N, rc.COMPOUNDS.index("medium")))
    L = F.tracks.length[track]

    def to_entry(cars):
        return (-F.cfg.pit_entry - F.game.from_pit(cars.track, cars.idx)) % L

    def tick(carry, _):
        f, c, green_s, was_sc, n, since = carry
        racing = ~(f.cars.crashed | f.finished | f.wrecked)
        st = c.status
        worst = f.cars.car.wear.max(-1)
        rule = (worst >= cfg.rule_wear) | (((st == R.VSC) | (st == R.SC)) & (to_entry(f.cars) >= cfg.reach_m)
                                           & ((worst > cfg.sc_wear) | (c.used.sum(-1) < 2)))
        call = racing & ~f.box_call & ~lane(f.cars) & rule
        f = f._replace(box_call=f.box_call | call, plan=jnp.where(call, cfg.rule_plan, f.plan))
        own = jnp.clip(forward(params, ctl.observe(f, c))[0], -1, 1)
        act = jnp.where(experts, rx.experts(ctl, f, c, own)[0], own)
        stopped, _ = ctl.stopped_on_track(f)
        new, c_new, _ = ctl.step(f, c, act)
        new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), f, new)
        c_new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), c, c_new)

        s2 = c_new.status
        green_s = jnp.where(s2 == R.RACING, jnp.where(st == R.RACING, green_s + dt, 0.0), jnp.inf)
        was_sc = jnp.where(s2 == R.RACING, was_sc | (st == R.SC_IN), False)
        went_out = (new.cars.crashed | new.wrecked) & ~(f.cars.crashed | f.wrecked) & ~new.finished
        # a moving car that touches something next to a car stopped on the track
        d = jnp.linalg.norm(new.cars.car.pos[:, None] - f.cars.car.pos[None, :], axis=-1)
        near = (stopped[None, :] & ~jnp.eye(N, dtype=bool) & (d < HIT_M)).any(1)
        hit = racing & ~stopped & (new.bump > 0) & (f.bump <= 0) & near
        lap1 = f.cars.progress.max() < L
        restart = was_sc & (green_s < 30.0)
        v = jnp.hypot(f.cars.car.vx, f.cars.car.vy)
        n = n + jnp.stack([hit.sum(), (went_out & lap1).sum(), (went_out & restart).sum(), (hit & (since > 2.0)).sum(),
                           (hit & (v > 10.0)).sum(), (hit & (since > 2.0) & (v > 10.0)).sum()])
        since = jnp.where(hit, 0.0, since + dt)
        return (new, c_new, green_s, was_sc, n, since), None

    carry = (f, ctl.init(f), jnp.asarray(jnp.inf), jnp.asarray(False), jnp.zeros(6, jnp.int32), jnp.full(N, jnp.inf))
    (f, c, _, _, n, _), _ = jax.lax.scan(tick, carry, None, length=T)
    pen = ctl.penalty_s(c)
    fin = f.finished
    return {"finished": fin.sum(), "time": jnp.where(fin, f.finish_t + pen, 0.0).sum(),
            "pens": ctl.infringements(c).sum(), "pen_s": pen.sum(), "dsq": ctl.disqualified(f, c).sum(),
            "out": (f.cars.crashed | f.wrecked).sum(), "stops": f.stops.sum(), "hit_stopped": n[0],
            "lap1_out": n[1], "restart_out": n[2], "hit_ep": n[3], "hit_fast": n[4], "hit_ep_fast": n[5], "vsc": c.vsc_n, "sc": c.sc_n, "red": c.red_n}


run = jax.jit(lambda p, e, keys: jax.vmap(lambda t, k: one(p, e, t, k), (0, 0))(jnp.arange(K), keys))
cars = a.grids * K * N
print(f"{a.grids} mixed grids x {K} test circuits x {N} cars = {cars} car-races a driver, {a.laps} laps, race control "
      f"on, natural incidents; devices {jax.devices()}", flush=True)
rows = []
for d, experts in [(a.baseline, True)] + [(d, False) for d in a.drivers]:
    dp, _, dk = d.partition(":")
    p = widen_senses(load(pick_checkpoint(dp, dk or "last"))[0], ctl.obs_size)
    t = {}
    for g in range(a.grids):
        t0 = time.time()
        out = jax.device_get(run(p, jnp.asarray(experts), jax.random.split(jax.random.PRNGKey(100 + g), K)))
        if a.trace:
            print(f"    hits per race {out['hit_stopped'].tolist()}, episodes {out['hit_ep'].tolist()}, "
                  f"of them over 10 m/s {out['hit_ep_fast'].tolist()}", flush=True)
        for k, v in out.items():
            t[k] = t.get(k, 0) + float(v.sum())
        print(f"  {d} grid {g + 1}/{a.grids}: {time.time() - t0:.0f} s", flush=True)
    t["name"] = d + (" + experts" if experts else "")
    rows.append(t)
    print(f"  {t}", flush=True)

print(f"\n| driver | finished | race time (min, finishers, + penalties) | penalties a car-race | pen s a car | DSQ | "
      f"out | hit a stopped car | out on lap 1 | out at restarts | stops a car | VSC / SC / red |\n|" + "---|" * 12)
for t in rows:
    print(f"| {t['name']} | {t['finished'] / cars:.0%} | {t['time'] / max(t['finished'], 1) / 60:.2f} | "
          f"{t['pens'] / cars:.2f} | {t['pen_s'] / cars:.1f} | {t['dsq']:.0f} | {t['out']:.0f} | {t['hit_stopped']:.0f} | "
          f"{t['lap1_out']:.0f} | {t['restart_out']:.0f} | {t['stops'] / cars:.2f} | "
          f"{t['vsc']:.0f} / {t['sc']:.0f} / {t['red']:.0f} |")
if a.trace:
    for t in rows:
        print(f"{t['name']}: hits {t['hit_stopped']:.0f} (over 10 m/s {t['hit_fast']:.0f}) in {t['hit_ep']:.0f} episodes "
              f"(over 10 m/s {t['hit_ep_fast']:.0f})")
b = rows[0]
for t in rows[1:]:
    checks = {"penalties a car-race < 0.2": t["pens"] / cars < 0.2,
              "hit a stopped car at most half the baseline's": t["hit_stopped"] <= b["hit_stopped"] / 2,
              "finished no worse": t["finished"] >= b["finished"],
              "race time no worse": t["time"] / max(t["finished"], 1) <= b["time"] / max(b["finished"], 1),
              "lap-1 pile-ups no worse": t["lap1_out"] <= b["lap1_out"],
              "restart pile-ups no worse": t["restart_out"] <= b["restart_out"]}
    print(f"\n{t['name']}: {'PASS' if all(checks.values()) else 'FAIL'}")
    for k, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {k}")
