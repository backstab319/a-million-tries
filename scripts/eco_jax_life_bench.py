"""Benchmark the JAX ecosystem with creatures (GPU port, plan section 4): world-days a second against numpy.

The standard world (--rain --thirst, 1000 plant-eaters with diet 0-0.1, as eco_kills.py), one seed per world.

    uv run --group jax python scripts/eco_jax_life_bench.py --worlds 1 4 --days 1
    uv run --group jax --group modal --env-file .env ailearns cloud script "scripts/eco_jax_life_bench.py" --on modal
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.games import ecosystem_jax_life as jl
from ailearns.games.ecosystem import World, WorldConfig
from ailearns.games.ecosystem_life import Life, LifeConfig

ap = argparse.ArgumentParser()
ap.add_argument("--worlds", type=int, nargs="+", default=[1, 8, 32, 128])
ap.add_argument("--days", type=float, default=2.0)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--cap", type=int, default=4096)
ap.add_argument("--numpy-days", type=float, default=1.0)
a = ap.parse_args()
print("devices:", jax.devices(), flush=True)


def make(seed):
    world = World.make(seed, WorldConfig(size=128, rain=True))
    life = Life(world, 0, seed, LifeConfig(thirst=True))
    life.add(a.n, diet=(0.0, 0.1))
    return life


life = make(0)
per_day = round(life.cfg.day_s / life.cfg.dt)
t = time.perf_counter()
for _ in range(int(a.numpy_days * per_day)):
    life.step()
np_s = (time.perf_counter() - t) / a.numpy_days
print(f"numpy: {np_s:.2f} s a world day ({int(life.alive.sum())} alive)", flush=True)

wcfg, cfg = life.world.cfg, life.cfg
steps = int(a.days * per_day / jl.plant_every(wcfg, cfg))
for n in a.worlds:
    pairs = [jl.from_numpy(make(s), a.cap) for s in range(n)]
    st = jax.tree.map(lambda *x: jnp.stack(x), *(p[0] for p in pairs))
    wd = jax.tree.map(lambda *x: jnp.stack(x), *(p[1] for p in pairs))
    del pairs
    run = jax.jit(jax.vmap(lambda s, w, k: jl.run(wcfg, cfg, s, w, k, steps)))
    keys = jax.random.split(jax.random.key(0), n)
    try:
        t = time.perf_counter()
        jax.block_until_ready(run(st, wd, keys))
        comp = time.perf_counter() - t
        t = time.perf_counter()
        out = jax.block_until_ready(run(st, wd, keys))
        dt = time.perf_counter() - t
    except Exception as e:
        print(f"jax, {n} worlds: failed ({type(e).__name__}: {str(e)[:200]})", flush=True)
        break
    rate = n * a.days / dt
    err = np.abs(np.asarray(jax.vmap(lambda w: jl.error(w, cfg))(out))).max()
    alive = np.asarray(out.c.alive.sum(1))
    print(f"jax, {n} worlds x {a.cap} slots: {rate:.2f} world-days/s = {rate * np_s:.1f}x numpy "
          f"({dt:.1f} s for {a.days:g} days, compile {comp - dt:.0f} s); alive {alive.min()}-{alive.max()}, "
          f"book error {err:.1e}, crowded {int(out.books.crowded.sum())}, full {int(out.books.full.sum())}, "
          f"births {int(out.books.births.sum())}", flush=True)
