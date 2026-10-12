"""How fast is the CQB game? `uv run --group jax python scripts/cqb_speed.py [rounds ...]`

Plays N rounds at once with random actions for 128 decisions, computing both players'
observations every decision and restarting finished rounds, all in one compiled function
(what training does, minus the brain). Tag runs in the same harness for comparison.
Prints decisions per second (one decision = one step of one round, both players acting).
"""

import sys
import time

import jax
import jax.numpy as jnp

from ailearns.games import cqb, tag


def rate(game, n_actions, observe, n, steps=128, repeats=3):
    def tick(carry, k):
        st, = carry
        k1, k2 = jax.random.split(k)
        acts = jax.vmap(lambda kk: jax.random.uniform(kk, n_actions, minval=-1, maxval=1))(jax.random.split(k1, n))
        st, _ = jax.vmap(game.step)(st, acts)
        obs = jax.vmap(observe)(st)
        st = jax.vmap(game.reset_if)(st, jax.random.split(k2, n))
        return (st,), obs.sum()

    @jax.jit
    def run(st, key):
        (st,), o = jax.lax.scan(tick, (st,), jax.random.split(key, steps))
        return st, o.sum()

    st = jax.vmap(game.init)(jax.random.split(jax.random.PRNGKey(0), n))
    st, o = run(st, jax.random.PRNGKey(1))                 # compile
    o.block_until_ready()
    best = 0.0
    for r in range(repeats):
        t = time.perf_counter()
        st, o = run(st, jax.random.PRNGKey(2 + r))
        o.block_until_ready()
        best = max(best, n * steps / (time.perf_counter() - t))
    return best


def main(sizes):
    print(f"devices: {jax.devices()}")
    cg = cqb.CQBGame(cqb.make_buildings(cqb.TRAIN_SEEDS))
    tg = tag.TagGame(tag.make_arenas(tag.TRAIN_SEEDS))
    for n in sizes:
        c = rate(cg, (2, 6), lambda s: jnp.stack([cg.observe(s, 0), cg.observe(s, 1)]), n)
        t = rate(tg, (2, 2), lambda s: jnp.stack([tg.observe(s, 0), tg.observe(s, 1)]), n)
        print(f"{n:>6} rounds at once: cqb {c / 1e3:7.1f}k decisions/s   tag {t / 1e3:7.1f}k   (tag / cqb = {t / c:.1f}x)")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1024, 4096])
