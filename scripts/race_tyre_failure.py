"""Race day (episode 15), step 8l check: does a worn-through tyre fail, and what does the failure do to a driver?

One car alone on each test circuit, every tyre already worn to `--wear`, plenty of fuel, driven by a brain for
`--seconds`: once with tyre failures (the field's default) and once without (failure points out of reach). Per
circuit: when the first tyre failed and which, whether the car went off or was wrecked, the distance covered both ways,
and its speed in the 20 s before and after the failure.

    uv run --group jax python scripts/race_tyre_failure.py runs/race-field-5:last [--tracks 8] [--wear 0.88]
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo_car import forward, load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_field
from ailearns.games import race_car as rc

ap = argparse.ArgumentParser()
ap.add_argument("brain")
ap.add_argument("--tracks", type=int, default=8)
ap.add_argument("--wear", type=float, default=0.88)
ap.add_argument("--seconds", type=float, default=300.0)
ap.add_argument("--compound", default="soft", choices=rc.COMPOUNDS)
a = ap.parse_args()

cfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
seeds = list(race.TEST_SEEDS)[:a.tracks]
tracks = race.make_tracks(seeds, cfg)
kw = dict(n_cars=2, laps=25, cfg=cfg, contact=False, slipstream=False)  # the 2nd a ghost (the senses want 2 cars)
fields = {"failures": race_field.RaceField(tracks, **kw), "none": race_field.RaceField(tracks, burst_wear=(2.0, 2.0), **kw)}
params, meta = load(pick_checkpoint(*(lambda p: (p[0], p[2] or "last"))(a.brain.partition(":"))))
design = jnp.tile(jnp.asarray(rc.F1_LIKE[0])[None], (2, 1))
compound = rc.COMPOUNDS.index(a.compound)
steps = int(a.seconds / cfg.dt)


def make_run(F):
    def run(track):
        f = F.grid(jax.random.key(0), track, design, jnp.array([compound, compound]), fuel=jnp.array([200.0, 200.0]))
        f = f._replace(finished=jnp.array([False, True]),  # the ghost: frozen and unseen
                       cars=f.cars._replace(car=f.cars.car._replace(wear=jnp.full((2, 4), a.wear))))

        def tick(f, _):
            mean, _, _ = forward(params, F.observe(f))
            f2, _ = F.step(f, jnp.clip(mean, -1, 1))
            return f2, (f2.damage.puncture[0], jnp.hypot(f2.cars.car.vx[0], f2.cars.car.vy[0]), f2.cars.progress[0],
                        f2.cars.crashed[0], f2.wrecked[0], f2.cars.car.wear[0])
        return jax.lax.scan(tick, f, None, steps)[1]
    return jax.jit(jax.vmap(run))


out = {k: [np.asarray(x) for x in make_run(F)(jnp.arange(a.tracks))] for k, F in fields.items()}
print(f"{a.brain} (update {meta['update']}), F1-like car alone, {a.compound}s worn to {a.wear}, {a.seconds:.0f} s\n")
names = ("front left", "front right", "rear left", "rear right")
for k in range(a.tracks):
    punct, v, prog, crashed, wrecked, wear = (x[k] for x in out["failures"])
    _, v0, prog0, crashed0, wrecked0, _ = (x[k] for x in out["none"])
    start = prog[0]
    first = np.flatnonzero(punct.any(-1))
    line = f"circuit {seeds[k]}: "
    if len(first):
        i = first[0]
        w = 20 * int(1 / cfg.dt)
        line += (f"failed at {i * cfg.dt:.0f} s ({', '.join(n for n, p in zip(names, punct[i]) if p)}; wear "
                 f"{wear[i].max():.2f}), {punct[-1].sum()} flat by the end; speed {3.6 * v[max(i - w, 0):i].mean():.0f}"
                 f" -> {3.6 * v[i:i + w].mean():.0f} km/h")
    else:
        line += f"no failure (wear {wear[-1].max():.2f})"
    out_at = np.flatnonzero(wrecked)
    off = f"out at {out_at[0] * cfg.dt:.0f} s" if len(out_at) else "off" if crashed[-1] else ""
    off0 = "wrecked" if wrecked0[-1] else "off" if crashed0[-1] else ""
    line += (f"; distance {(prog[-1] - start) / 1000:.1f} km{' (' + off + ')' if off else ''} vs "
             f"{(prog0[-1] - start) / 1000:.1f} km without failures{' (' + off0 + ')' if off0 else ''}")
    print(line)
