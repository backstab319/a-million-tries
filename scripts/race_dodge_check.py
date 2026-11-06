"""Race day 2 (episode 16), step 15f: does the teacher get round a wreck from a standstill? The acceptance test's long
leans (15c: a car parked 5 m behind a wreck for 104 s) and D1-3's failed top-up (15e) point at the experts' dodge,
planned for arriving at speed. Set up exactly that: one car dead on the track, another stopped `--gaps` m behind it
on its line, the rest of the field driving away ahead; race control on. Each driver (the experts in charge, then each
brain alone) gets `--secs` s: did the car behind get past, how long it took, how many touches on the wreck.

    uv run --group jax python scripts/race_dodge_check.py --drivers runs/race-d1-2:12 runs/race-d1-3:8
"""

import argparse
import itertools

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
ap.add_argument("--drivers", nargs="+", default=["runs/race-d1-2:12"])
ap.add_argument("--teacher-brain", default="", help="the brain the experts cap (default: the first driver)")
ap.add_argument("--gaps", default="6,10,20", help="metres from the car behind's centre to the wreck's (a car is 5.6 m long: 6 is nose to tail)")
ap.add_argument("--lats", default="-1.5,0,1.5", help="the wreck's metres left of the centreline (the car behind on "
                "the same line)")
ap.add_argument("--spots", type=int, default=4, help="places round each lap")
ap.add_argument("--speed", type=float, default=0.0, help="the car behind's speed at the start (m/s)")
ap.add_argument("--secs", type=float, default=40.0)
ap.add_argument("--n", type=int, default=8, help="test circuits")
ap.add_argument("--experts-only", action="store_true")
ap.add_argument("--blind", action="store_true", help="also the experts judging 'out' only by speed (what a driver "
                "senses), not race control's record")
a = ap.parse_args()

cfg = StratConfig(race_control=True)
seeds = list(race.TEST_SEEDS)[:a.n]
F = make_field(cfg, seeds)
ctl = race_control.RaceControl(F)
N, T, dt = F.n, int(round(a.secs / F.cfg.dt)), F.cfg.dt
design = jnp.asarray(rc.F1_LIKE[0])
gaps = [float(x) for x in a.gaps.split(",")]
lats = [float(x) for x in a.lats.split(",")]
cases = list(itertools.product(range(len(seeds)), range(a.spots), gaps, lats))
C = jnp.asarray([[k, s, g, w] for k, s, g, w in cases], jnp.float32)
WRECK, ME = 0, 1
BANDS = (10.0, 25.0, 50.0, 100.0)  # metres to the wreck: the steering error is reported in these bands


def setup(case, key):
    k = case[0].astype(jnp.int32)
    L = F.tracks.length[k]
    m = 400.0 + case[1] / a.spots * (L - 800.0)   # clear of the start line, so no lap wrap
    f = F.grid(key, k, jnp.tile(design[None], (N, 1)))
    metres = jnp.concatenate([jnp.stack([m, m - case[2]]), m + 250.0 + 40.0 * jnp.arange(N - 2)])
    side = jnp.concatenate([jnp.stack([case[3], case[3]]), jnp.where(jnp.arange(N - 2) % 2 == 0, 2.0, -2.0)])
    speed = jnp.zeros(N).at[ME].set(a.speed)
    cars = jax.vmap(lambda kk, mm, ss, vv: F.place(kk, k, design, mm, ss, vv))(jax.random.split(key, N), metres, side,
                                                                               speed)
    dead = jnp.arange(N) == WRECK
    cars = cars._replace(crashed=cars.crashed | dead, done=cars.done | dead)
    return f._replace(cars=cars)


def drive(params, teacher, experts, case, key):
    f = setup(case, key)

    def tick(carry, _):
        f, c, touches, passed_t, rev, sc_at, side, serr = carry
        obs = ctl.observe(f, c)
        own = jnp.clip(forward(params, obs)[0], -1, 1)
        base = jnp.clip(forward(teacher, obs)[0], -1, 1)
        ex, zones = jax.lax.cond(experts == 2, lambda: rx.experts(ctl, f, c, base, rx.ExpertConfig(blind=True)),
                                 lambda: rx.experts(ctl, f, c, base))
        # the brain's steering against the teacher's, while the teacher would be going round (by metres to the wreck)
        to_w = f.cars.progress[WRECK] - f.cars.progress[ME]
        band = jnp.digitize(to_w, jnp.asarray(BANDS))
        on = zones[ME, rx.DODGE] & (to_w > 0)
        serr = serr + jnp.where(on, jax.nn.one_hot(band, len(BANDS) + 1) * jnp.stack([jnp.abs(own[ME, 0] - ex[ME, 0]),
                                                                                       1.0])[:, None], 0.0)
        act = jnp.where(experts > 0, ex, own)
        new, c_new, _ = ctl.step(f, c, act)
        d = jnp.linalg.norm(new.cars.car.pos[ME] - new.cars.car.pos[WRECK])
        touch = (new.bump[ME] > 0) & (f.bump[ME] <= 0) & (d < 8.0)
        past = (new.cars.progress[ME] > new.cars.progress[WRECK] + 8.0) & ~(new.cars.crashed[ME] | new.wrecked[ME])
        passed_t = jnp.where(past & ~jnp.isfinite(passed_t), new.t * dt, passed_t)
        there = (d < 30.0) & (sc_at < 0)   # the race's status when it first got within 30 m of the wreck
        sc_at = jnp.where(there, c_new.status, sc_at)
        # what it was trying: its offset from the wreck (left positive) the first time it's within 12 m behind it
        level = (new.cars.progress[WRECK] - new.cars.progress[ME] < 12.0) & ~jnp.isfinite(side)
        side = jnp.where(level, new.cars.lat[ME] - new.cars.lat[WRECK], side)
        return (new, c_new, touches + touch, passed_t, rev | (new.cars.car.gear[ME] == 0), sc_at, side, serr), None

    (f, c, touches, passed_t, rev, sc_at, side, serr), _ = jax.lax.scan(tick, (f, ctl.init(f), jnp.asarray(0),
        jnp.asarray(jnp.inf), jnp.asarray(False), jnp.asarray(-1), jnp.asarray(jnp.inf),
        jnp.zeros((2, len(BANDS) + 1))), None,
                                                length=T)
    return {"passed": jnp.isfinite(passed_t), "t": passed_t, "touches": touches,
            "out": f.cars.crashed[ME] | f.wrecked[ME], "rev": rev, "end_rev": f.cars.car.gear[ME] == 0, "sc_at": sc_at, "side": side, "serr": serr,
            "end_behind": f.cars.progress[WRECK] - f.cars.progress[ME]}


run = jax.jit(jax.vmap(drive, (None, None, None, 0, 0)))
load_p = lambda d: widen_senses(load(pick_checkpoint(*(lambda p, _, k: (p, k or "last"))(*d.partition(":"))))[0],  # noqa: E731
                                ctl.obs_size)
teacher = load_p(a.teacher_brain or a.drivers[0])
print(f"{len(cases)} cases ({len(seeds)} test circuits x {a.spots} spots x gaps {gaps} m x wreck at {lats} m), the car "
      f"behind at {a.speed} m/s, {a.secs:.0f} s each; devices {jax.devices()}", flush=True)
keys = jax.random.split(jax.random.PRNGKey(0), len(cases))
cs = np.asarray(C)
print("| driver | got past | median s to pass | touches on the wreck (cases with any) | out |\n|---|---|---|---|---|")
for name, p, ex in [(f"experts (on {a.teacher_brain or a.drivers[0]})", teacher, 1)] + (
        [("experts, blind (speed only)", teacher, 2)] if a.blind else []) + [(d, load_p(d), 0)
                                                                                            for d in ([] if a.experts_only else a.drivers)]:
    r = {k: np.asarray(v) for k, v in jax.device_get(run(p, teacher, jnp.asarray(ex, jnp.int32), C, keys)).items()}
    t = r["t"][r["passed"]]
    print(f"| {name} | {r['passed'].mean():.0%} | {np.median(t) if len(t) else float('nan'):.1f} | "
          f"{r['touches'].sum()} ({(r['touches'] > 0).mean():.0%}) | {r['out'].sum()} | reversed in {r['rev'].mean():.0%} |",
          flush=True)
    sc = (r["sc_at"] == race_control.SC) | (r["sc_at"] == race_control.SC_IN)
    print(f"    Safety Car out on arrival in {sc.mean():.0%} of cases")
    if sc.any() and (~sc).any():
        print(f"    got past with the Safety Car out on arrival {r['passed'][sc].mean():.0%} of {sc.sum()}, without it "
              f"{r['passed'][~sc].mean():.0%} of {(~sc).sum()}")
    if not ex:
        e = r["serr"].sum(0)
        names = [f"<{BANDS[0]:.0f}"] + [f"{a:.0f}-{b:.0f}" for a, b in zip(BANDS, BANDS[1:])] + [f">{BANDS[-1]:.0f}"]
        print("    steering off the teacher's while it would go round (mean |steer difference|, by metres to the wreck): "
              + ", ".join(f"{n} m {e[0, i] / max(e[1, i], 1):.2f}" for i, n in enumerate(names)))
    stuck = ~r["passed"] & ~r["out"]
    for what, sel in (("got past", r["passed"]), ("not past", stuck), ("went out", r["out"])):
        x = r["side"][sel & np.isfinite(r["side"])]
        if len(x):
            print(f"    {what}: offset from the wreck 12 m short (m, left +): median {np.median(x):+.1f}, |offset| under 1 m "
                  f"{(np.abs(x) < 1.0).mean():.0%}")
    if stuck.any():
        print(f"    not past ({stuck.sum()}): still in reverse at the end {r['end_rev'][stuck].mean():.0%}, metres behind "
              f"the wreck at the end: median {np.median(r['end_behind'][stuck]):.1f} (10-90%: "
              f"{np.percentile(r['end_behind'][stuck], 10):.1f} to {np.percentile(r['end_behind'][stuck], 90):.1f})")
    for gi, g in enumerate(gaps):   # by gap and side: where it fails
        row = []
        for w in lats:
            sel = (cs[:, 2] == g) & (cs[:, 3] == w)
            row.append(f"wreck at {w:+.1f} m: {r['passed'][sel].mean():.0%} (touched {(r['touches'][sel] > 0).mean():.0%}, "
                       f"12 m short: left {(r['side'][sel] > 1.0).mean():.0%} in line {(np.abs(r['side'][sel]) <= 1.0).mean():.0%} right "
                       f"{(r['side'][sel] < -1.0).mean():.0%})")
        print(f"    gap {g:.0f} m: " + ", ".join(row))
