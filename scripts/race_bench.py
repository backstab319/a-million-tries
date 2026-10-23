"""How fast the racing sim runs: car-steps per second (step + observe, as training uses it), for a few batch sizes.

    uv run --group jax python scripts/race_bench.py [--cars 1024 4096] [--steps 200]
"""

import argparse
import time

import jax
import jax.numpy as jnp

from ailearns.games import race

ap = argparse.ArgumentParser()
ap.add_argument("--cars", type=int, nargs="+", default=[256, 1024, 4096])
ap.add_argument("--steps", type=int, default=200)
args = ap.parse_args()

cfg = race.RaceConfig(start_speed=40.0)
game = race.RaceGame(race.make_tracks(range(64), cfg), cfg)
print("devices:", jax.devices())

for n in args.cars:
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    st = jax.vmap(game.init)(keys)

    @jax.jit
    def roll(st, key):
        def body(carry, k):
            s = carry
            obs = jax.vmap(game.observe)(s)
            action = jnp.tanh(obs[:, :2] * 0.0 + jax.random.normal(k, (n, 2)) * 0.3) + jnp.array([0.0, 0.4])
            s, r = jax.vmap(game.step)(s, action)
            s = jax.vmap(game.reset_if)(s, jax.random.split(k, n))
            return s, r.sum()
        return jax.lax.scan(body, st, jax.random.split(key, args.steps))

    jax.block_until_ready(roll(st, jax.random.PRNGKey(1)))  # compile
    t0 = time.perf_counter()
    out = jax.block_until_ready(roll(st, jax.random.PRNGKey(2)))
    dt = time.perf_counter() - t0
    print(f"{n:6d} cars: {n * args.steps / dt:10,.0f} car-steps/s ({n * args.steps * cfg.dt / dt:,.0f} sim-seconds/s)")
