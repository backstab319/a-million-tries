"""Race day (episode 15), step 8d: where does a called-in car go? (step 8c: 0 stops from ~60k calls)

Mid-race stretches from the race school on test circuits, every car driven by one brain (best guess, no noise) and
every car called in at the start. Per car over the stretch: did it reach the pit zone (the stretch of lap with the
lane beside it), how far right it got there (the lane is past the pit wall's line), did it enter the lane, how close it
got to its box and how slow, did it stop; and how its run ended.

    uv run --group jax python scripts/race_box_calls.py [runs/field-speed-256:0 runs/field-speed-256:last] [--races 16]
"""

import argparse
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo_car import forward, load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_field, race_school

ap = argparse.ArgumentParser()
ap.add_argument("brains", nargs="*", default=["runs/field-speed-256:0", "runs/field-speed-256:last"])
ap.add_argument("--races", type=int, default=16)
ap.add_argument("--tracks", type=int, default=8)
ap.add_argument("--seconds", type=float, default=60.0)
ap.add_argument("--lessons", type=float, default=0.0, help="share of stretches that are pit lessons (0: calls mid-race only)")
a = ap.parse_args()

cfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
F = race_field.RaceField(race.make_tracks(list(race.TEST_SEEDS)[:a.tracks], cfg), cfg=cfg)
S = race_school.RaceSchool(F, race_school.SchoolConfig(stretch_s=a.seconds, grid_share=0.0, call_rate_s=1e9,
                                                                    pit_share=a.lessons))
g, n, steps = F.game, F.n, int(a.seconds / cfg.dt)


def measure(params):
    def one(key):
        les = S.start(key)
        les = les._replace(field=les.field._replace(box_call=jnp.ones(n, bool)))
        z = jnp.zeros(n)
        acc = dict(zone=z > 0, entered=z > 0, right=jnp.full(n, 99.0), box=jnp.full(n, 999.0), slow=jnp.full(n, 999.0))

        def tick(carry, k):
            les, acc = carry
            f = les.field
            mean, _, _ = forward(params, S.observe(les))
            les, *_ = S.step(les, jnp.clip(mean, -1, 1), k)
            f = les.field
            live = S.racing(f) & ~les.field.done
            cars = f.cars
            zone = jax.vmap(g.in_pit_zone)(cars.track, cars.idx) & live
            lane = jax.vmap(g.in_pit_lane)(cars) & live
            s = g.from_pit(cars.track, cars.idx)
            v = jnp.hypot(cars.car.vx, cars.car.vy)
            acc = dict(zone=acc["zone"] | zone, entered=acc["entered"] | lane,
                       right=jnp.where(zone, jnp.minimum(acc["right"], cars.lat), acc["right"]),
                       box=jnp.where(lane, jnp.minimum(acc["box"], jnp.abs(s - F.box_s)), acc["box"]),
                       slow=jnp.where(lane, jnp.minimum(acc["slow"], v), acc["slow"]))
            return (les, acc), None
        (les, acc), _ = jax.lax.scan(tick, (les, acc), jax.random.split(key, steps))
        f = les.field
        return acc | dict(stopped=f.stops > 0, wall=f.cars.crashed & ~f.wrecked, wrecked=f.wrecked)
    return jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(5), a.races))


for brain in a.brains:
    path, _, ck = brain.partition(":")
    params, meta = load(pick_checkpoint(path, ck or "last"))
    r = jax.tree.map(lambda x: np.asarray(x).reshape(-1), measure(params))
    cars = len(r["zone"])
    z = r["zone"]
    print(f"\n{brain} (update {meta['update']}): {cars} called cars, {a.races} stretches of {a.seconds:.0f} s, "
          f"pit lessons {a.lessons:.0%}")
    print(f"  reached the pit zone   {z.sum():4d}")
    if z.any():
        print(f"  furthest right there   median {np.median(r['right'][z]):.1f} m (lane past {cfg.pit_wall_lat} m)")
    print(f"  entered the lane       {r['entered'].sum():4d}")
    e = r["entered"]
    if e.any():
        print(f"  closest to its box     median {np.median(r['box'][e]):.1f} m (counts within {F.box_half} m)")
        print(f"  slowest in the lane    median {np.median(r['slow'][e]) * 3.6:.0f} km/h")
    print(f"  stopped                {r['stopped'].sum():4d}")
    print(f"  wall {r['wall'].sum()}  wrecked {r['wrecked'].sum()}")
