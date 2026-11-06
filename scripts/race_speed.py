"""Race day 2 (episode 16), step 9g: how fast training runs, and where the time goes. The ES generation's own pieces
(`es_field.make_school`, finish mode: 8 cars, 25 laps, GP tyres) on `--races` races at once (a generation runs pop x
rounds = 128 x 24 = 3072 side by side), `--steps` decisions, timed after compiling:

- full: observe, the brain for all 8 cars, `school.step` (race control on / off)
- step: `school.step` alone with fixed inputs (so: observe + brain = full - step)

and from each, a generation's estimated seconds (129 x 24 races x 48 000 decisions, the judge and bench left out).

    uv run --group jax python scripts/race_speed.py [--driver runs/race-fin-7:50] [--races 3072] [--steps 200]
"""

import argparse
import time

import jax
import jax.numpy as jnp

from ailearns.agents import es_field
from ailearns.agents.ppo_car import forward, load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race
from ailearns.profiling import profile

ap = argparse.ArgumentParser()
ap.add_argument("--driver", default="runs/race-fin-7:50")
ap.add_argument("--races", type=int, default=3072)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--tracks", type=int, default=64)
ap.add_argument("--scale", default="", help="races at once, comma-separated: only the full step, race control on, each size")
ap.add_argument("--profile", action="store_true", help="the full step (race control on) under the JAX profiler: GPU "
                "time per kernel, each traced back to its line of code")
a = ap.parse_args()

GEN_RACES, GEN_STEPS = 129 * 24, int(2400.0 / 0.05)
print(f"{jax.devices()[0].device_kind}: {a.races} races x {a.steps} decisions, 8 cars", flush=True)


def timed(fn, *args):
    jax.block_until_ready(fn(*args))  # compile
    t0 = time.time()
    jax.block_until_ready(fn(*args))
    return time.time() - t0


ap_finish = a.steps >= 40000  # (whole races: when does each one end? the generation runs every race to the cap)
if a.scale or ap_finish:  # does a bigger batch use the GPU better? (race-decisions per second at each size)
    cfg = es_field.FieldESConfig(mode="finish", n_cars=8, n_learn=4, laps=25, gp_km=305.0, tyre_cal=1.9,
                                 call_worst=True, race_control=True)
    school = es_field.make_school(cfg, list(race.TRAIN_SEEDS)[:a.tracks])
    path, _, ck = a.driver.partition(":")
    brain = es_field._fit(load(pick_checkpoint(path, ck or "last"))[0], a.driver, school)

    def race_one(les, key):
        def tick(les, k):
            return school.step(les, forward(brain, school.observe(les))[0], k)[0], None
        if ap_finish:
            def tick_t(carry, k):
                les, end = carry
                les, _, _, _, over = school.step(les, forward(brain, school.observe(les))[0], k)
                return (les, jnp.where(over & (end < 0), les.field.t - les.t0, end)), None
            (les, end), _ = jax.lax.scan(tick_t, (les, jnp.asarray(-1)), jax.random.split(key, a.steps))
            return jnp.where(end < 0, a.steps, end) * school.field.cfg.dt
        out = jax.lax.scan(tick, les, jax.random.split(key, a.steps))[0]
        return sum(jnp.sum(x.astype(jnp.float32)) for x in jax.tree.leaves(out))
    play = jax.jit(jax.vmap(race_one))
    if ap_finish:
        import numpy as np
        keys = jax.random.split(jax.random.PRNGKey(0), int(a.scale or 3072))
        end = np.asarray(play(jax.jit(jax.vmap(school.start))(keys), keys))
        print(f"{len(end)} races to {a.steps * school.field.cfg.dt:.0f} s: over at (s) p10 {np.percentile(end, 10):.0f}, "
              f"median {np.median(end):.0f}, p90 {np.percentile(end, 90):.0f}, max {end.max():.0f}; at the cap "
              f"{(end >= a.steps * school.field.cfg.dt).mean():.0%}", flush=True)
        raise SystemExit
    for n in map(int, a.scale.split(",")):
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        t = timed(play, jax.jit(jax.vmap(school.start))(keys), keys)
        print(f"{n:6d} races: {t:6.2f} s, {n * a.steps / t / 1e6:.2f} M race-decisions/s, a generation "
              f"~{t / (n * a.steps) * GEN_RACES * GEN_STEPS:.0f} s", flush=True)
    raise SystemExit

for control in (True, False):
    cfg = es_field.FieldESConfig(mode="finish", n_cars=8, n_learn=4, laps=25, gp_km=305.0, tyre_cal=1.9,
                                 call_worst=True, race_control=control)
    school = es_field.make_school(cfg, list(race.TRAIN_SEEDS)[:a.tracks])
    path, _, ck = a.driver.partition(":")
    brain = es_field._fit(load(pick_checkpoint(path, ck or "last"))[0], a.driver, school)
    keys = jax.random.split(jax.random.PRNGKey(0), a.races)
    starts = jax.jit(jax.vmap(school.start))(keys)
    fixed = jnp.tile(jnp.array([[0.0, 0.5, -1.0]]), (8, 1))

    def run(full):
        def race_one(les, key):
            def tick(les, k):
                acts = forward(brain, school.observe(les))[0] if full else fixed
                return school.step(les, acts, k)[0], None
            out = jax.lax.scan(tick, les, jax.random.split(key, a.steps))[0]
            # (everything the step touches, so the compiler can't drop any of it: the step count alone timed 0 s)
            return sum(jnp.sum(x.astype(jnp.float32)) for x in jax.tree.leaves(out))
        return jax.jit(jax.vmap(race_one))

    if a.profile:
        profile(run(True), starts, keys)
        raise SystemExit
    t_full, t_step = timed(run(True), starts, keys), timed(run(False), starts, keys)
    # each piece of the senses on top of the step (fixed inputs), summed into the carry so it's kept
    F, ctl = school.field, school.control
    bodies = (lambda les: ctl.bodies(les.control)) if ctl else (lambda les: None)  # noqa: E731
    obs0 = school.observe(jax.tree.map(lambda x: x[0], starts))
    pieces = {
        "observe (all)": lambda les: school.observe(les).sum(),
        "  own view": lambda les: jax.vmap(F.game.observe)(les.field.cars, les.field.setup).sum(),
        "    wall lasers": lambda les: jax.vmap(F.game.lasers)(les.field.cars).sum(),
        "  car lasers": lambda les: F.car_lasers(les.field, bodies(les)).sum(),
        "  neighbours": lambda les: F.neighbours(les.field, bodies(les)).sum(),
        "brain (fixed senses)": lambda les: forward(brain, obs0 + les.field.cars.lat[:, None] * 0.0)[0].sum(),
    }
    if ctl:
        pieces["  race control senses"] = lambda les: ctl.driver_senses(les.field, les.control).sum()
        leaves = lambda x: sum(jnp.sum(v.astype(jnp.float32)) for v in jax.tree.leaves(x))  # noqa: E731
        pieces["control: drive_sc"] = lambda les: leaves(ctl.drive_sc(les.field, les.control))
        pieces["control: red_action"] = lambda les: ctl.red_action(les.field, les.control).sum()
        pieces["control: direct"] = lambda les: leaves(ctl.direct(les.field, les.control))
        pieces["control: red_needed"] = lambda les: ctl.red_needed(les.field).sum()
        pieces["field: step_with SC"] = lambda les: leaves(F.step_with(les.field, fixed, ctl.bodies(les.control))[0])
        pieces["field: step alone"] = lambda les: leaves(F.step(les.field, fixed)[0])

    def piece(fn):
        def race_one(les, key):
            def tick(carry, k):
                les, acc = carry
                return (school.step(les, fixed, k)[0], acc + fn(les)), None
            (out, acc), _ = jax.lax.scan(tick, (les, 0.0), jax.random.split(key, a.steps))
            return acc + sum(jnp.sum(x.astype(jnp.float32)) for x in jax.tree.leaves(out))
        return jax.jit(jax.vmap(race_one))
    for name, fn in pieces.items():
        print(f"    {name:24s} {timed(piece(fn), starts, keys) - t_step:6.2f} s", flush=True)
    per = lambda t: t / (a.races * a.steps)  # noqa: E731  seconds per race-decision
    name = "race control ON " if control else "race control OFF"
    print(f"{name}: full {t_full:6.2f} s ({a.races * a.steps / t_full / 1e6:.2f} M race-decisions/s, a generation "
          f"~{per(t_full) * GEN_RACES * GEN_STEPS:.0f} s); step alone {t_step:6.2f} s; observe + brain "
          f"{t_full - t_step:6.2f} s ({(t_full - t_step) / t_full:.0%})", flush=True)
