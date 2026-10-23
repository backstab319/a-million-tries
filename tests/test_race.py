"""The racing sandbox's circuits and game loop (episode 14)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ailearns.games import race
from ailearns.games import race_car as rc
from ailearns.games.race_checks import test_driver as scripted_driver

CFG = race.RaceConfig()


@pytest.fixture(scope="module")
def game():
    return race.RaceGame(race.make_tracks(range(4), CFG), CFG)


def run(game, st, policy, seconds):
    step = jax.jit(lambda s: game.step(s, policy(game, s))[0])
    for _ in range(int(seconds / CFG.dt)):
        st = step(st)
        if bool(st.done):
            break
    return st


def test_circuits_are_real_scale_and_drivable():
    for seed in range(6):
        t = race.make_track(seed, CFG)
        assert 1500 < t["length"] < 6000, t["length"]
        assert 1 / np.abs(t["curvature"]).max() >= CFG.min_corner * 0.95
    a, b = race.make_track(3, CFG), race.make_track(3, CFG)
    assert np.allclose(a["center"], b["center"])  # same seed, same circuit


def test_observation_shape_and_values(game):
    st = game.init(jax.random.PRNGKey(0), design=jnp.asarray(rc.F1_LIKE[0]))
    obs = game.observe(st)
    assert obs.shape == (CFG.obs_size,)
    assert bool(jnp.all(jnp.isfinite(obs)))
    lasers = game.lasers(st)
    assert float(lasers[0]) == pytest.approx(CFG.edge, abs=1.0)    # 90 degrees left: the wall beside the car
    assert float(lasers[-1]) == pytest.approx(CFG.edge, abs=1.0)


def test_scripted_driver_laps_cleanly(game):
    st = game.init(jax.random.PRNGKey(1), design=jnp.asarray(rc.F1_LIKE[0]), track=0, random_start=False)
    st = run(game, st, scripted_driver, 90)
    assert not bool(st.crashed)
    assert float(st.progress) > 0.6 * float(game.tracks.length[0])  # well round the lap in 90 s
    assert abs(float(st.lat)) < CFG.width / 2


def test_steering_into_the_wall_crashes(game):
    st = game.init(jax.random.PRNGKey(2), design=jnp.asarray(rc.F1_LIKE[0]), track=1, random_start=False)
    st = run(game, st, lambda g, s: jnp.array([1.0, 0.6]), 30)
    assert bool(st.crashed) and bool(st.done)


def test_grass_is_slow(game):
    d = jnp.asarray(rc.F1_LIKE[0])
    st = game.init(jax.random.PRNGKey(3), design=d, track=2, start=0)
    p = rc.derive(d)
    on_tarmac = on_grass = rc.car_at([0, 0], 0.0, 40.0)
    grass = rc.Surface(*(jnp.full(4, v) for v in CFG.grass_surface))
    for _ in range(40):  # two seconds coasting
        on_tarmac, _ = rc.drive(p, on_tarmac, jnp.array([0.0, 0.0]))
        on_grass, _ = rc.drive(p, on_grass, jnp.array([0.0, 0.0]), grass)
    assert float(on_grass.vx) < float(on_tarmac.vx) - 1.0
    assert st.surface.shape == (4,)


def test_grass_dirties_tyres(game):
    d = jnp.asarray(rc.F1_LIKE[0])
    st = game.init(jax.random.PRNGKey(3), design=d, track=2, start=0)
    h = game.tracks.heading[2, 0] + jnp.pi / 2
    out = st.car.pos + 9.0 * jnp.stack([jnp.cos(h), jnp.sin(h)])  # 3 m past the road edge, on the grass
    on_grass, _ = game.step(st._replace(car=st.car._replace(pos=out)), jnp.array([0.0, 0.0]))
    assert np.allclose(on_grass.dirt, 1.0)
    back = st._replace(dirt=on_grass.dirt)  # back on the tarmac with dirty tyres: it wears off over dirt_clear
    for _ in range(int(1.0 / CFG.dt)):
        back, _ = game.step(back, jnp.array([0.0, 0.0]))
    assert np.allclose(back.dirt, 1 - 1.0 / CFG.dirt_clear, atol=1e-3)


def test_many_cars_many_designs(game):
    keys = jax.random.split(jax.random.PRNGKey(4), 64)
    st = jax.vmap(game.init)(keys)
    step = jax.jit(jax.vmap(lambda s: game.step(s, jnp.array([0.0, 0.3]))))
    for _ in range(5):
        st, r = step(st)
    obs = jax.vmap(game.observe)(st)
    assert obs.shape == (64, CFG.obs_size) and bool(jnp.all(jnp.isfinite(obs)))
    assert bool(jnp.all(st.progress > 0))
    assert len(np.unique(np.asarray(st.design[:, 0]))) == 64  # every car its own design


def test_wake_tows_the_car_behind_only():
    pos = jnp.array([[0.0, 0.0], [-8.0, 0.0], [-8.0, 6.0], [10.0, 0.0], [-200.0, 0.0]])
    air = race.wake(pos, jnp.zeros(5))
    # car 0 is 10 m behind car 3: towed; car 1 is right behind car 0: towed hardest; car 2 is off to the side;
    # car 3 leads (clean air); car 4 is too far back to feel anything
    assert float(air.drag[1]) < float(air.drag[0]) < 1.0
    assert float(air.drag[2]) > 0.95 and float(air.drag[3]) == 1.0 and float(air.drag[4]) == 1.0
    assert float(air.down_front[1]) < float(air.down_rear[1])  # dirty air hits the front wing hardest


def test_designs_near_a_center_stay_near_it():
    from dataclasses import replace
    center = rc.F1_LIKE[0]
    cfg = replace(CFG, design_center=tuple(float(x) for x in center), design_spread=0.1)
    g = race.RaceGame(race.make_tracks(range(1), cfg), cfg)
    st = jax.vmap(g.init)(jax.random.split(jax.random.PRNGKey(5), 256))
    off = np.abs(np.asarray(st.design) - center) / (rc.DESIGN_HIGH - rc.DESIGN_LOW)
    assert off.max() <= 0.1 + 1e-5 and off.mean() > 0.02


def test_circuits_have_hills_and_banking_and_keep_their_layout():
    t = race.make_track(5, CFG)
    assert 1.0 < np.ptp(t["z"]) <= CFG.hill_range[1] + 1e-6
    assert np.abs(t["grade"]).max() <= CFG.max_grade + 1e-6
    corner = np.abs(t["curvature"]) > 1 / 60
    assert (np.sign(t["side"][corner]) != np.sign(t["curvature"][corner])).mean() > 0.9 or np.abs(t["side"]).max() < 1e-3
    assert np.allclose(race.make_track(5, CFG)["center"], t["center"])
