"""Race day 2 (episode 16), step 9g: the brains in 16-bit on the GPU's tensor cores? The brains were ~25% of a T4
generation (`gen_speed.py --profile`), in 32-bit. Here, on a T4: the time of 200 decisions of a generation's brains
(129 brains on 96 cars each, the rivals' brain on 12384), 32-bit v 16-bit (multiplies in 16-bit, sums in 32-bit);
and how far the 16-bit actions are from the 32-bit ones on senses from real races.

    uv run --group jax python scripts/brain_bench.py [--run runs/race-fin-7:50]
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import es_field
from ailearns.agents.ppo_car import forward, load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/race-fin-7:50")
ap.add_argument("--races", type=int, default=3072)
ap.add_argument("--steps", type=int, default=200)
a = ap.parse_args()


def forward16(params: dict, obs):
    """forward() with the multiplies in 16-bit (tensor cores), each sum kept in 32-bit; biases and tanh 32-bit."""
    def mm(x, w):
        return jnp.matmul(x.astype(jnp.float16), w.T.astype(jnp.float16), preferred_element_type=jnp.float32)
    h = jnp.tanh(mm(obs, params["body.0.weight"]) + params["body.0.bias"])
    h = jnp.tanh(mm(h, params["body.2.weight"]) + params["body.2.bias"])
    return mm(h, params["policy.weight"]) + params["policy.bias"]


cfg = es_field.FieldESConfig(mode="finish", n_cars=8, n_learn=4, laps=25, gp_km=305.0, tyre_cal=1.9,
                             call_worst=True, race_control=True)
school = es_field.make_school(cfg, list(race.TRAIN_SEEDS)[:64])
path, _, ck = a.run.partition(":")
brain = es_field._fit(load(pick_checkpoint(path, ck or "last"))[0], a.run, school)
print(f"{jax.devices()[0].device_kind}: {a.run}, {brain['body.0.weight'].shape[1]} senses, hidden "
      f"{brain['body.0.weight'].shape[0]}", flush=True)

# senses from real races: 3072 races x 8 cars, driven 10 s by the 32-bit brain, a snapshot every 2.5 s
keys = jax.random.split(jax.random.PRNGKey(0), a.races)
les = jax.jit(jax.vmap(school.start))(keys)


@jax.jit
def drive(les, k):
    def tick(les, kk):
        return jax.vmap(lambda s, q: school.step(s, forward(brain, school.observe(s))[0], q)[0])(les, kk), None
    return jax.lax.scan(tick, les, jax.random.split(k, (50, a.races)))[0]


obs = []
for i in range(4):
    les = drive(les, jax.random.PRNGKey(10 + i))
    obs.append(jax.jit(jax.vmap(school.observe))(les).reshape(-1, school.obs_size))
obs = jnp.concatenate(obs)
a32 = np.clip(np.asarray(jax.jit(lambda o: forward(brain, o)[0])(obs)), -1, 1)
a16 = np.clip(np.asarray(jax.jit(lambda o: forward16(brain, o))(obs)), -1, 1)
d = np.abs(a16 - a32)
print(f"actions, 16-bit v 32-bit, {len(obs)} cars on real senses: mean {d.mean():.5f}, p99 {np.percentile(d, 99):.5f}, "
      f"max {d.max():.5f} (actions in -1..1; per output: max {d.max(0).round(5).tolist()})", flush=True)

# time: a generation's brains, 200 decisions (129 noisy brains on 96 cars each + the rivals' one brain on 12384)
members = jax.tree.map(lambda v: v + 0.01 * jax.random.normal(jax.random.PRNGKey(1), (129,) + v.shape), brain)
mo = jnp.resize(obs, (129 * 96, obs.shape[1])).reshape(129, 96, -1)  # (repeated if there are too few: a smoke test)
ro = jnp.resize(obs, (12384, obs.shape[1]))


def timed(fn):
    def run():
        def tick(acc, i):
            o = mo + i * 1e-4
            m = jax.vmap(fn)(members, o).sum() + fn(brain, ro + i * 1e-4).sum()
            return acc + m, None
        return jax.lax.scan(tick, 0.0, jnp.arange(a.steps, dtype=jnp.float32))[0]
    f = jax.jit(run)
    jax.block_until_ready(f())
    t0 = time.time()
    jax.block_until_ready(f())
    return time.time() - t0


t32 = timed(lambda p, o: forward(p, o)[0])
t16 = timed(forward16)
print(f"brains, {a.steps} decisions: 32-bit {t32:.3f} s, 16-bit {t16:.3f} s ({t32 / t16:.1f}x)", flush=True)
