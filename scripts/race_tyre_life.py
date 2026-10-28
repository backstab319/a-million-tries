"""Race day (episode 15), step 8h: do the tyres force a stop? (before the strategist, step 9)

One car alone (an F1-like design, no contact, no calls) on each test circuit, one run per compound, a full race
distance with no stop, the race-trained driver (best guess, no noise). Per lap: lap time and tread used. Then, from
those curves: does a one-stop race beat a no-stop one, given what a stop costs (`--stop-s`, measured ~19 s)?

    uv run --group jax python scripts/race_tyre_life.py runs/race-field-5:last [--tracks 8] [--laps 25]
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
ap.add_argument("--laps", type=int, default=25)
ap.add_argument("--stop-s", type=float, default=19.0, help="seconds a stop costs (scripts/race_pitstop.py)")
a = ap.parse_args()

cfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
seeds = list(race.TEST_SEEDS)[:a.tracks]
F = race_field.RaceField(race.make_tracks(seeds, cfg), n_cars=2,  # the 2nd a ghost (the senses want 4 slots' worth)
                         laps=a.laps, cfg=cfg, contact=False,
                         slipstream=False)
path, _, ck = a.brain.partition(":")
params, meta = load(pick_checkpoint(path, ck or "last"))
steps = int(cfg.time_limit / cfg.dt)
design = jnp.tile(jnp.asarray(rc.F1_LIKE[0])[None], (2, 1))


def run(track, compound):
    f = F.grid(jax.random.key(0), track, design, jnp.array([compound, compound]))
    f = f._replace(finished=jnp.array([False, True]))  # the ghost: frozen and unseen

    def tick(f, _):
        mean, _, _ = forward(params, F.observe(f))
        f2, _ = F.step(f, jnp.clip(mean, -1, 1))
        f2 = f2._replace(done=f2.finished[0] | f2.cars.crashed[0] | f2.wrecked[0] | f2.done)
        f2 = jax.tree.map(lambda new, old: jnp.where(f.done, old, new), f2, f)  # hold a finished race
        return f2, (F.laps_done(f2)[0], f2.t, f2.cars.car.wear[0].mean(), f2.cars.crashed[0] | f2.wrecked[0],
                    f2.setup.fuel[0] - f2.cars.car.fuel_used[0])
    f, (laps, t, wear, out, fuel) = jax.lax.scan(tick, f, None, steps)
    return laps, t, wear, out, fuel


laps, t, wear, out, fuel = jax.jit(jax.vmap(jax.vmap(run, in_axes=(None, 0)), in_axes=(0, None)))(
    jnp.arange(a.tracks), jnp.arange(len(rc.COMPOUNDS)))
laps, t, wear, out, fuel = map(np.asarray, (laps, t, wear, out, fuel))
print(f"{a.brain} (update {meta['update']}), F1-like car alone, {a.laps} laps, no stop\n")
for c, name in enumerate(rc.COMPOUNDS):
    print(f"{name}:")
    for k in range(a.tracks):
        lp, tt, ww, oo = laps[k, c], t[k, c], wear[k, c], out[k, c]
        cross = np.flatnonzero(np.diff(lp) > 0) + 1  # the step each lap ended (the first: the grid's run to the line)
        times = np.diff(tt[cross]) * cfg.dt  # full laps only
        w = ww[cross[1:]]
        crashed = bool(oo[-1])
        cliff = next((i + 1 for i, x in enumerate(w) if x >= 0.8), None)
        dry = fuel[k, c, -1] <= 0.01
        line = f"  circuit {seeds[k]}: {len(times):2d} laps{' (out)' if crashed else ''}{' (no fuel)' if dry else ''}"
        if len(times) >= 3:
            best = times.min()
            line += f", best {best:.1f} s, last {times[-1]:.1f} s (+{times[-1] / best - 1:.0%}), tread {w[-1]:.2f}, " \
                    f"cliff lap {cliff or '-'}"
            n = len(times)  # one stop after lap i: the second stint runs the first stint's laps again (fresh tyres)
            if n >= a.laps - 1:
                one = min(times[:i].sum() + a.stop_s + times[:n - i].sum() for i in range(1, n))
                line += f"; no stop {times.sum():.0f} s, best one stop {one:.0f} s: " \
                        f"{'STOP' if one < times.sum() else 'stay out'}"
        print(line)
