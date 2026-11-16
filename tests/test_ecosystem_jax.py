"""The JAX world against the numpy reference (plan section 4, port step 1)."""

import copy

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ailearns.games import ecosystem_jax as ej  # noqa: E402
from ailearns.games.ecosystem import World, WorldConfig  # noqa: E402

CFG = WorldConfig(size=48, rain=True)
STEPS_A_YEAR = int(CFG.year_days / CFG.dt_days)


def numpy_storms(world):
    """The storms numpy's next `weather()` will draw, from a copy of its stream, padded for JAX."""
    cfg, rng = world.cfg, copy.deepcopy(world.rng)
    L = cfg.size * cfg.cell_m
    k = rng.poisson(cfg.storms_ha_day * L * L / 10_000 * cfg.dt_days)
    m = ej.max_storms(cfg)
    x, y = np.zeros(m), np.zeros(m)
    if k:
        bx, by = world.belt()
        front = rng.random(k) < cfg.belt_share
        x[:k] = np.where(front, rng.normal(bx, cfg.belt_m, k), rng.uniform(0, L, k))
        y[:k] = np.where(front, rng.normal(by, cfg.belt_m, k), rng.uniform(0, L, k))
    assert k <= m
    return ej.Storms(jnp.asarray(x), jnp.asarray(y), jnp.arange(m) < k, jnp.asarray(0, jnp.int32))


@pytest.mark.parametrize("rain", [True, False])
def test_a_year_matches_numpy_to_rounding(rain):
    """Same land, same storms, same grazing and fruit-eating: every pool equal to float64 rounding."""
    cfg = WorldConfig(size=48, rain=rain)
    with jax.enable_x64(True):
        world = World.make(2, cfg)
        assert len(world.trees) > 0
        land, p = ej.from_numpy(world)
        step = jax.jit(lambda p, s: ej.step(cfg, land, p, s))
        graze = jax.jit(lambda p, c, w: ej.graze(land, p, c, w))
        fruit = jax.jit(ej.eat_fruit)
        rng = np.random.default_rng(5)
        storms = ej.no_storms(cfg)
        for t in range(STEPS_A_YEAR):
            if rain:
                storms = numpy_storms(world)
            world.step()
            p = step(p, storms)
            cells = rng.integers(0, cfg.size ** 2, 200)
            cells[:20] = cells[0]          # a crowd on one cell, to test the pro-rata share
            want = rng.random(200) * 300
            g_np, d_np = world.graze(cells, want, split=True)
            p, g_j, d_j = graze(p, jnp.asarray(cells), jnp.asarray(want))
            f_np = world.eat_fruit(cells, want / 10)
            p, f_j = fruit(p, jnp.asarray(cells), jnp.asarray(want / 10))
            if t % 120 == 0 or t == STEPS_A_YEAR - 1:
                for a, b in ((world.plants, p.plants), (world.roots, p.roots), (world.dead, p.dead),
                             (world.fruit, p.fruit), (world.wet, p.wet)):
                    np.testing.assert_allclose(np.asarray(b), a, rtol=1e-9, atol=1e-6)
                for a, b in ((g_np, g_j), (d_np, d_j), (f_np, f_j)):
                    np.testing.assert_allclose(np.asarray(b), a, rtol=1e-9, atol=1e-6)
        led = world.ledger
        for a, b in ((led.sun_in, p.sun_in), (led.eaten, p.eaten), (led.dieback, p.dieback)):
            assert float(b) == pytest.approx(a, rel=1e-9)
        assert float(p.day) == world.day
        assert abs(float(ej.error(p))) < 1e-9 and abs(world.error()) < 1e-9


def test_many_worlds_at_once_in_float32_keep_their_books():
    """Three seeds stacked and run a year in one compiled call (fresh JAX storms): the books hold to float32 rounding,
    every world grows and gets rained on, and no storm is lost to the cap."""
    worlds = [ej.from_numpy(World.make(s, CFG)) for s in range(3)]
    land = jax.tree.map(lambda *a: jnp.stack(a), *(w[0] for w in worlds))
    p = jax.tree.map(lambda *a: jnp.stack(a), *(w[1] for w in worlds))
    assert p.plants.dtype == jnp.float32
    run = jax.jit(jax.vmap(lambda l, q, k: ej.run(CFG, l, q, k, STEPS_A_YEAR)))
    out = run(land, p, jax.random.split(jax.random.key(0), 3))
    assert np.all(np.abs(np.asarray(jax.vmap(ej.error)(out))) < 1e-5)
    assert np.all(np.asarray(out.sun_in) > 0) and np.all(np.asarray(out.storms_lost) == 0)
    assert np.all(np.asarray(out.day) == CFG.year_days)
    wet = np.asarray(out.wet)
    assert np.all(wet.max(axis=(1, 2)) - wet.min(axis=(1, 2)) > 0.1)   # storms fell in places, not evenly


def test_jax_storms_follow_numpys_odds():
    """Many steps of JAX storms: the same mean count a step, and ~belt_share of them near the front."""
    cfg = WorldConfig(size=128, rain=True)
    L = cfg.size * cfg.cell_m
    lam = cfg.storms_ha_day * L * L / 10_000 * cfg.dt_days
    s = jax.vmap(lambda k: ej.draw_storms(cfg, k, 0.0))(jax.random.split(jax.random.key(1), 50_000))
    real = np.asarray(s.real)
    assert real.sum(1).mean() == pytest.approx(lam, rel=0.03)
    bx, by = (float(v) for v in ej.belt(cfg, 0.0))
    near = np.hypot(np.asarray(s.x) - bx, np.asarray(s.y) - by)[real] < 3 * cfg.belt_m
    # front storms are within 3 sd 1 - e^-4.5 of the time (2D gaussian); uniform ones by the area they cover
    u = np.random.default_rng(0).uniform(0, L, (100_000, 2))
    area = (np.hypot(u[:, 0] - bx, u[:, 1] - by) < 3 * cfg.belt_m).mean()
    want = cfg.belt_share * (1 - np.exp(-4.5)) + (1 - cfg.belt_share) * area
    assert near.mean() == pytest.approx(want, abs=0.01)
