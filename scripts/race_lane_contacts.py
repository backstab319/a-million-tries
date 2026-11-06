"""Race day 2 (episode 16), step 13h: what the contacts in the pit lane are. D2-4 touched other cars 1,207 times in 234
stops on the experts' test (D1-2: 28 in 295), with few penalties: the stewards count one collision per incident (2 s
without contact), so a car leaning on another all the way down the lane is one. Traced here, per driver.

Races as `race_expert_check.py` (the 8 test circuits, every car one brain, 25 laps cut at `--cap-s`, the team's rule
calling stops, the new pit lane), no injected breakdowns. Every decision a car in the pit lane is in contact
(`Field.bump` > 0) is recorded with: its speed, the other car (the nearest centre), that car's speed, where (the
fast lane or the working lane, beside its box or not, either car stopped in its box), the damage that contact did
(joules), whether it was blamed (`Field.fault`) and counted (`Control.lane_hits`). Contact steps are joined into
episodes (a car's contact with no gap of more than `--gap-s`), and summarised per driver.

    uv run --group jax python scripts/race_lane_contacts.py --drivers runs/race-d2-4:100 runs/race-d1-2:12
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.bc_control import breakdown
from ailearns.agents.es_strategist import StratConfig, make_field
from ailearns.agents.ppo_car import forward, load
from ailearns.agents.ppo_field import widen_senses
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_control
from ailearns.games import race_car as rc

ap = argparse.ArgumentParser()
ap.add_argument("--drivers", nargs="+", default=["runs/race-d2-4:100", "runs/race-d1-2:12"])
ap.add_argument("--cap-s", type=float, default=1800.0)
ap.add_argument("--gap-s", type=float, default=0.5, help="contact steps this close are one episode")
ap.add_argument("--at", default="", help="seconds into each race of injected breakdowns, as race_expert_check (e.g. "
                "300,900: Safety Cars, crowded pit lanes)")
ap.add_argument("--grids", type=int, default=1, help="races per circuit (a new grid draw each)")
a = ap.parse_args()

cfg = StratConfig(race_control=True, laps=25, cap_s=a.cap_s)
seeds = list(race.TEST_SEEDS)[:8]
F = make_field(cfg, seeds)
ctl = race_control.RaceControl(F)
K, N, T, dt = len(seeds), F.n, int(round(cfg.cap_s / F.cfg.dt)), F.cfg.dt
AT = jnp.asarray([round(float(x) / dt) for x in a.at.split(',')]) if a.at else None
lane = jax.vmap(F.game.in_pit_lane)
lo, span = jnp.asarray(rc.DESIGN_LOW), jnp.asarray(rc.DESIGN_HIGH - rc.DESIGN_LOW)
R = race_control


def race_one(driver, track, key):
    kd, kg = jax.random.split(key)
    designs = lo + jax.random.uniform(kd, (N, len(rc.DESIGN))) * span
    f = F.grid(kg, track, designs, jnp.full(N, rc.COMPOUNDS.index("medium")))
    L = F.tracks.length[track]

    def to_entry(c):
        return (-F.cfg.pit_entry - F.game.from_pit(c.track, c.idx)) % L

    def tick(carry, k):
        f, c = carry
        if AT is not None:
            f, _ = jax.lax.cond(jnp.any(f.t == AT), lambda f: breakdown(F, f, k), lambda f: (f, jnp.zeros(N, bool)), f)
        racing = ~(f.cars.crashed | f.finished | f.wrecked)
        st = c.status
        neutral = (st == R.VSC) | (st == R.SC)
        worst = f.cars.car.wear.max(-1)
        rule = (worst >= cfg.rule_wear) | (neutral & (to_entry(f.cars) >= cfg.reach_m)
                                           & ((worst > cfg.sc_wear) | (c.used.sum(-1) < 2)))
        call = racing & ~f.box_call & ~lane(f.cars) & rule
        f = f._replace(box_call=f.box_call | call, plan=jnp.where(call, cfg.rule_plan, f.plan))
        obs = ctl.observe(f, c)
        act = jnp.clip(forward(driver, obs)[0], -1, 1)
        new, c_new, _ = ctl.step(f, c, act)
        new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), f, new)
        c_new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), c, c_new)
        cars = new.cars
        v = jnp.hypot(cars.car.vx, cars.car.vy)
        d = jnp.linalg.norm(cars.car.pos[:, None] - cars.car.pos[None, :], axis=-1) + jnp.eye(N) * 1e9
        other = jnp.argmin(d, 1)
        in_lane = lane(cars)
        fast = cars.lat > F.cfg.box_lat + (F.cfg.pit_fast - F.cfg.box_lat) / 2  # nearer the fast lane's line
        boxed = new.service > 0
        rec = dict(touch=in_lane & (new.bump > 0), v=v, v_other=v[other], other=other, fast=fast,
                   boxed=boxed, other_boxed=boxed[other], joules=new.hit.sum(-1), fault=new.fault,
                   counted=c_new.lane_hits > c.lane_hits, bump=new.bump, stop=new.stops > f.stops,
                   sense=obs[:, ctl.obs_size - 3] > 0, status=jnp.broadcast_to(c.status, (N,)),
                   # (15q) going out in the lane: how
                   out_now=in_lane & (cars.crashed | new.wrecked) & ~(f.cars.crashed | f.wrecked) & ~new.finished,
                   walled=cars.crashed & ~new.wrecked, wrecked=new.wrecked, dry=F.dry_out(new), lat=cars.lat,
                   v_before=jnp.hypot(f.cars.car.vx, f.cars.car.vy),
                   from_pit=jax.vmap(F.game.from_pit)(cars.track, cars.idx), called=new.box_call,
                   # (15r) into the lane without a call
                   uncalled_in=in_lane & ~lane(f.cars) & ~f.box_call & ~(cars.crashed | new.wrecked))
        return (new, c_new), rec

    (f, c), rec = jax.lax.scan(tick, (f, ctl.init(f)), jax.random.split(jax.random.fold_in(key, 3), T))
    return rec, f.stops.sum(), ctl.penalty_s(c).sum(), (f.cars.crashed | f.wrecked).sum()


run = jax.jit(jax.vmap(race_one, in_axes=(None, 0, 0)))


def episodes(touch, gap):
    """(T,) bool -> [(start, end)] runs of contact, joining gaps of up to `gap` steps."""
    idx = np.flatnonzero(touch)
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap + 1:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return out


for name in a.drivers:
    dp, _, dk = name.partition(":")
    driver = widen_senses(load(pick_checkpoint(dp, dk or "last"))[0], ctl.obs_size)
    t0 = time.time()
    gap = int(round(a.gap_s / dt))
    eps, tot, lane_out, uncalled = [], np.zeros(3), [], []
    for g in range(a.grids):
        rec, stops, pen, out = jax.device_get(run(driver, jnp.arange(K), jax.random.split(jax.random.PRNGKey(g), K)))
        tot += [stops.sum(), pen.sum(), out.sum()]
        for k, t, i in zip(*np.nonzero(rec["uncalled_in"])):
            w = slice(max(t - int(round(3.0 / dt)), 0), t + 1)
            later = rec["out_now"][k, t:t + int(round(30.0 / dt)), i].any()
            uncalled.append(dict(status=int(rec["status"][k, t, i]), sense=bool(rec["sense"][k, w, i].any()),
                                 at=float(rec["from_pit"][k, t, i]), v=float(rec["v"][k, t, i]), out=bool(later)))
        for k, t, i in zip(*np.nonzero(rec["out_now"])):
            b = max(t - int(round(1.0 / dt)), 0)
            lane_out.append(dict(how="wrecked" if rec["wrecked"][k, t, i] else "dry" if rec["dry"][k, t, i] else "wall",
                                 v=float(rec["v"][k, b, i]), lat=float(rec["lat"][k, t, i]),
                                 status=int(rec["status"][k, t, i]), touched=bool(rec["touch"][k, b:t + 1, i].any()),
                                   sense=bool(rec["sense"][k, max(t - int(round(3.0 / dt)), 0):t + 1, i].any()),
                                   at=float(rec["from_pit"][k, t, i]), called=bool(rec["called"][k, b, i])))
        for k in range(K):
            for i in range(N):
                for s, e in episodes(rec["touch"][k, :, i], gap):
                    sl = slice(s, e + 1)
                    on = rec["touch"][k, sl, i]
                    eps.append(dict(
                        dur=(e - s + 1) * dt, touches=int((on[1:] & ~on[:-1]).sum() + on[0]),
                        v=float(rec["v"][k, sl, i][on].mean()), dv=float(np.abs(rec["v"][k, sl, i] - rec["v_other"][k, sl, i])[on].mean()),
                        fast=float(rec["fast"][k, sl, i][on].mean()), boxed=bool(rec["boxed"][k, sl, i].any()),
                        other_boxed=bool(rec["other_boxed"][k, sl, i].any()), joules=float(rec["joules"][k, sl, i].sum()),
                        blamed=int(rec["fault"][k, sl, i].sum()), counted=int(rec["counted"][k, sl, i].sum()),
                        sense=bool(rec["sense"][k, sl, i].any()), neutral=bool((rec["status"][k, sl, i] != R.RACING).any())))
    print(f"\n{name}: {K * a.grids} races, {time.time() - t0:.0f} s; stops {int(tot[0])}, penalties {tot[1]:.0f} s, "
          f"out {int(tot[2])}", flush=True)
    if uncalled:
        from collections import Counter
        U = uncalled
        print(f"  into the lane uncalled: {len(U)}; then out in the lane within 30 s {np.mean([x['out'] for x in U]):.0%}; the "
              f"stopped-car sense on in the 3 s before {np.mean([x['sense'] for x in U]):.0%}; speed median "
              f"{np.median([x['v'] for x in U]):.1f} m/s; metres from the pit's middle: median "
              f"{np.median([x['at'] for x in U]):.0f}; race status " + ", ".join(
                  f"{['green', 'VSC', 'SC', 'SC in', 'red'][k]} {v}" for k, v in Counter(x["status"] for x in U).items()))
    if lane_out:
        from collections import Counter
        print(f"  out in the pit lane: {len(lane_out)}: " + ", ".join(f"{k} {v}" for k, v in Counter(
            x["how"] for x in lane_out).items()) + f"; speed 1 s before: median {np.median([x['v'] for x in lane_out]):.1f} "
              f"m/s; across the lane (lat): " + " ".join(f"{x['lat']:.1f}" for x in lane_out[:12])
              + f"; touching another just before {np.mean([x['touched'] for x in lane_out]):.0%}; the stopped-car sense on in the "
              f"3 s before {np.mean([x['sense'] for x in lane_out]):.0%}; metres from the pit's middle (the wall runs "
              f"-150..150): " + " ".join(f"{x['at']:.0f}" for x in lane_out[:14]) + f"; called in {np.mean([x['called'] for x in lane_out]):.0%}; race status "
              + ", ".join(f"{['green', 'VSC', 'SC', 'SC in', 'red'][k]} {v}" for k, v in Counter(
                  x["status"] for x in lane_out).items()))
    if not eps:
        print("  no contact in the pit lane")
        continue
    E = {k: np.asarray([x[k] for x in eps]) for k in eps[0]}
    n = len(eps)
    print(f"  {n} contact episodes (car-sides; a two-car touch is two), {int(E['touches'].sum())} touches "
          f"(the experts' test's count), {int(E['counted'].sum())} counted by the stewards")
    print(f"  length: median {np.median(E['dur']):.2f} s, 90% {np.percentile(E['dur'], 90):.2f} s, longest "
          f"{E['dur'].max():.1f} s; touches an episode: median {np.median(E['touches']):.0f}, max {E['touches'].max()}")
    print(f"  speed: median {np.median(E['v']):.1f} m/s (closing on the other: median {np.median(E['dv']):.2f} m/s); "
          f"in the fast lane {np.mean(E['fast'] > 0.5):.0%}")
    print(f"  with a car stopped in its box: this one {np.mean(E['boxed']):.0%}, the other {np.mean(E['other_boxed']):.0%}")
    print(f"  damage: median {np.median(E['joules']):.0f} J an episode, {np.mean(E['joules'] > 100):.0%} over 100 J; "
          f"blamed at least once {np.mean(E['blamed'] > 0):.0%}")
    print(f"  the stopped-car-ahead sense on during it {np.mean(E['sense']):.0%}; under a VSC / Safety Car / red "
          f"{np.mean(E['neutral']):.0%}")
    long = np.argsort(-E["dur"])[:5]
    for q in long:
        x = eps[q]
        print(f"    longest: {x['dur']:.1f} s, {x['touches']} touches, {x['v']:.1f} m/s (dv {x['dv']:.2f}), "
              f"fast lane {x['fast']:.0%}, boxed {x['boxed']}/{x['other_boxed']}, {x['joules']:.0f} J, "
              f"counted {x['counted']}")
