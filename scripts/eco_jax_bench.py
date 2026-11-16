"""Benchmark the JAX ecosystem world (GPU port, plan section 4) against numpy: world-days a second.

    uv run --group jax python scripts/eco_jax_bench.py                 # here (CPU)
    uv run --group jax --group modal --env-file .env ailearns cloud script "scripts/eco_jax_bench.py" --on modal

Port step 1 has only the plants (no creatures yet), so this times the world alone; the numbers that decide the port
come with creatures (steps 2-4).
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.games import ecosystem_jax as ej
from ailearns.games.ecosystem import World, WorldConfig

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=128)
ap.add_argument("--worlds", type=int, nargs="+", default=[1, 16, 128, 512, 2048])
ap.add_argument("--days", type=float, default=30.0)
a = ap.parse_args()

cfg = WorldConfig(size=a.size, rain=True)
steps = int(a.days / cfg.dt_days)
print("devices:", jax.devices(), flush=True)

w = World.make(0, cfg)
t = time.perf_counter()
for _ in range(steps):
    w.step()
np_rate = a.days / (time.perf_counter() - t)
print(f"numpy, 1 world {a.size}x{a.size}: {np_rate:.1f} world-days/s", flush=True)

land0, p0 = ej.from_numpy(World.make(0, cfg))
for n in a.worlds:
    land = jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *x.shape)), land0)
    p = jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *x.shape)), p0)
    run = jax.jit(jax.vmap(lambda l, q, k: ej.run(cfg, l, q, k, steps)))
    keys = jax.random.split(jax.random.key(0), n)
    try:
        jax.block_until_ready(run(land, p, keys))   # compile
        t = time.perf_counter()
        out = jax.block_until_ready(run(land, p, keys))
        dt = time.perf_counter() - t
    except Exception as e:   # out of memory: say so and stop growing
        print(f"jax, {n} worlds: failed ({type(e).__name__})", flush=True)
        break
    rate = n * a.days / dt
    err = float(jnp.abs(jax.vmap(ej.error)(out)).max())
    print(f"jax, {n} worlds: {rate:.0f} world-days/s ({rate / np_rate:.0f}x numpy), "
          f"{dt:.2f} s for {a.days:g} days, worst book error {err:.1e}, storms lost {int(out.storms_lost.sum())}",
          flush=True)
