"""Race day (episode 15), step 8u: is the fuel allowance savable by driving? (race-fin-3: ~60 of 128 cars run dry)

The finish-mode benchmark (es_field's 32 fixed rounds, the last 2 laps, 4 learner cars each) with one brain, every
car's throttle capped at each level in turn. A cap is the crudest saving (real drivers lift and coast before the
braking zones, which costs less time), so what it saves is a lower bound on what driving can save. If no cap gets
the dry tanks down without losing more time than it saves, the allowance is too tight for driving to fix.

    uv run --group jax python scripts/race_fuel_save.py runs/race-fin-3:60 [--caps 1 0.95 0.9 0.85 0.8] [--bench 32]
"""

import argparse
from dataclasses import fields

import jax
import jax.numpy as jnp

from ailearns.agents import es_field
from ailearns.agents.ppo_car import forward, load
from ailearns.checkpoints import pick_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("brain", nargs="?", default="runs/race-fin-3:60")
ap.add_argument("--caps", type=float, nargs="*", default=[1.0, 0.95, 0.9, 0.85, 0.8])
ap.add_argument("--bench", type=int, default=32)
ap.add_argument("--need", action="store_true", help="also run with tanks that can't run dry: the fuel the brain "
                "needs vs what the allowance gives")
ap.add_argument("--tracks", type=int, default=0, help="circuits to build (0: the run's; fewer = other rounds)")
a = ap.parse_args()

path, _, ck = a.brain.partition(":")
params, ck = load(pick_checkpoint(path, ck or "last"))
saved = ck["config"]  # the run's config travels in the checkpoint (cloud scripts get checkpoints only)
names = {f.name for f in fields(es_field.FieldESConfig)}
cfg = es_field.FieldESConfig(**{k: v for k, v in saved.items() if k in names}
                              | ({"tracks": a.tracks} if a.tracks else {}))
school = es_field.make_school(cfg, list(es_field.race.TRAIN_SEEDS)[: cfg.tracks])
keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 3000), a.bench)  # es_field's benchmark rounds
per_car = a.bench * cfg.n_learn
print(f"{a.brain} on {jax.devices()[0].device_kind}, {a.bench} rounds x {cfg.n_learn} cars, {cfg.tracks} circuits",
      flush=True)
print("| throttle cap | finished | ran dry (of them, didn't finish) | wrecked | stops | race time per car |\n|---|---|---|---|---|---|")
def bench(cap):
    def capped(p, obs):  # every car, the brain's own action with the pedal held under the cap
        mean, *rest = forward(p, obs)
        return (mean.at[:, 1].set(jnp.minimum(mean[:, 1], cap)), *rest)
    es_field.forward = capped
    play = es_field.make_play_finish(cfg, school, params)
    return jax.device_get(jax.jit(lambda p: {k: v.sum() for k, v in play(p, keys).items()})(params))


share = school.field.fuel_share
for cap in a.caps:
    b = bench(cap)
    need = ""
    if a.need:  # same rounds, 3x the tanks: what it burns when it never runs dry
        school.field.fuel_share = 3 * share
        n = bench(cap)  # kg per metre it drove, over the metres it had to go, vs the kg it had
        need = (f" needs {float(n['fuel_burned']) / float(n['metres']) * float(b['to_go']) / float(b['fuel_had']):.2f}"
                " x the fuel it had |")
        school.field.fuel_share = share
    print(f"| {cap:.2f} | {b['finished']:.0f}/{per_car} | {b['dry']:.0f} ({b['dry_out']:.0f}) | {b['wrecked']:.0f} | "
          f"{b['stops']:.0f} | {float(b['race_s']) / per_car:.1f} s |{need}", flush=True)
