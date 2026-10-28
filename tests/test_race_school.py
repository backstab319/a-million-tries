"""Race day (episode 15), step 8: the race school's stretches and the widened ep14 brain."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ailearns.agents.ppo_car import forward, init_params
from ailearns.agents.ppo_field import widen
from ailearns.games import race, race_field, race_school

N = 4


@pytest.fixture(scope="module")
def school():
    cfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
    F = race_field.RaceField(race.make_tracks(range(2), cfg), n_cars=N, laps=5, cfg=cfg)
    return race_school.RaceSchool(F, race_school.SchoolConfig(stretch_s=1.0))


def test_midrace_starts_are_a_spread_out_pack_late_in_the_race(school):
    F, c = school.field, school.cfg
    for i in range(4):
        grid = F.grid(jax.random.key(i), 0, jnp.tile(jnp.asarray(race_field.rc.F1_LIKE[0])[None], (N, 1)))
        f = school.midrace(jax.random.key(10 + i), grid, 0)
        prog = np.sort(np.asarray(f.cars.progress))
        assert np.all(np.diff(prog) >= c.min_gap - 0.5)  # no two boxes spawned on top of each other
        L = float(F.tracks.length[0])
        assert -L < prog.min() and prog.max() < F.laps * L
        wear = np.asarray(f.cars.car.wear)
        assert np.all((wear >= 0) & (wear <= c.worn))
        left = np.asarray(f.setup.fuel - f.cars.car.fuel_used)
        assert np.all(left > 0)
        pace = prog.max() / max(int(f.t) * F.cfg.dt, 1e-6)
        assert prog.max() <= 0 or c.pace[0] - 1 < pace < c.pace[1] + 1


def test_grid_slots_are_shuffled(school):
    poles = set()
    for i in range(12):
        f = school.field.grid(jax.random.key(0), 0, jnp.tile(jnp.asarray(race_field.rc.F1_LIKE[0])[None], (N, 1)))
        cars = school.shuffle(jax.random.key(i), f.cars)
        poles.add(int(np.argmax(np.asarray(cars.progress))))
    assert len(poles) > 1  # the learner's cars (the first few) don't always start at the front


def test_a_stretch_ends_on_time_and_ends_every_run(school):
    s = school.start(jax.random.key(3))
    step = jax.jit(school.step)
    a = jnp.zeros((N, 3))  # coasting: nobody reaches the wall in a second
    for k in range(int(school.cfg.stretch_s / school.field.cfg.dt)):
        racing_before = school.racing(s.field)
        s, r, racing, ended, over = step(s, a, jax.random.key(k))
        assert np.array_equal(np.asarray(racing), np.asarray(racing_before))
    assert bool(over)
    assert np.array_equal(np.asarray(ended), np.asarray(racing))
    fresh = school.reset_if(s, over, jax.random.key(99))
    assert int(fresh.field.t) == int(fresh.t0)


def test_the_widened_brain_drives_like_the_old_one(school):
    old = init_params(jax.random.key(0), race.RaceConfig().obs_size, 32, -0.5)
    new = widen(old, school.field.obs_size, -1.0)
    obs = jax.random.normal(jax.random.key(1), (5, school.field.obs_size))
    m_old, _, v_old = forward(old, obs[:, :race.RaceConfig().obs_size])
    m_new, log_std, v_new = forward(new, obs)
    np.testing.assert_allclose(m_new[:, :2], m_old, atol=1e-5)
    np.testing.assert_allclose(v_new, v_old, atol=1e-5)
    assert np.all(np.asarray(m_new[:, 2]) == -1.0) and log_std.shape == (3,)  # reverse starts off


def test_pit_lessons_start_called_cars_before_their_boxes_apart():
    cfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
    F = race_field.RaceField(race.make_tracks(range(1), cfg), n_cars=N, laps=5, cfg=cfg)
    S = race_school.RaceSchool(F, race_school.SchoolConfig(pit_share=1.0))
    grid = F.grid(jax.random.key(0), 0, jnp.tile(jnp.asarray(race_field.rc.F1_LIKE[0])[None], (N, 1)))
    for i in range(4):
        f = S.midrace(jax.random.key(20 + i), grid, 0)
        called = np.asarray(f.box_call)
        lesson = np.flatnonzero(np.asarray(jax.vmap(F.game.in_pit_zone)(f.cars.track, f.cars.idx)) & called)
        s = np.asarray(F.game.from_pit(f.cars.track, f.cars.idx))
        lane = np.asarray(jax.vmap(F.game.in_pit_lane)(f.cars))
        pos = np.asarray(f.cars.car.pos)[lane]
        gaps = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + 99 * np.eye(len(pos))
        assert len(pos) < 2 or gaps.min() >= 7.0  # spaced: a car is ~5.6 m long
        for k in np.flatnonzero(lane):
            assert called[k] and s[k] <= float(F.box_s[k]) + 3.0  # in the lane: called, its box still ahead
            assert float(jnp.hypot(f.cars.car.vx[k], f.cars.car.vy[k])) <= cfg.pit_limit + 1e-3
        assert len(lesson) <= N


def test_the_pit_coach_points_a_called_car_at_its_mark(school):
    F = school.field
    S = race_school.RaceSchool(F, race_school.SchoolConfig(coach=1.0))
    f = F.grid(jax.random.key(0), 0, jnp.tile(jnp.asarray(race_field.rc.F1_LIKE[0])[None], (N, 1)))
    k, box = 0, float(F.box_s[0])
    lane = (F.cfg.pit_wall_lat + F.cfg.pit_outer) / 2

    def phi(s, lat, v, called=True):
        car = F.place(jax.random.key(1), 0, f.cars.design[k], float(F.game.pit_at[0]) + box + s, lat, v)
        cars = jax.tree.map(lambda a, b: a.at[k].set(b), f.cars, car)
        return float(S.potential(f._replace(cars=cars, box_call=f.box_call.at[k].set(called)))[k])

    assert abs(phi(0.0, lane, 0.0)) < 0.1                    # stopped on its mark: the top
    assert phi(-30.0, lane, 10.0) < phi(-10.0, lane, 2.0)    # closer and slower is better
    assert phi(-150.0, 0.0, 20.0) < phi(-150.0, lane, 20.0)  # in the lane beats on the track beside it
    assert phi(10.0, lane, 10.0) < phi(-10.0, lane, 10.0)    # past its mark: everything the approach earned goes back
    assert phi(-10.0, lane, 2.0, called=False) == 0.0


def test_the_box_sense_is_sharp_near_the_mark():
    sense = race_field.RaceField.box_sense
    np.testing.assert_allclose(sense(jnp.array([1.0, 4.0, 25.0, 100.0, 400.0, -4.0])),
                               [0.1, 0.2, 0.5, 1.0, 1.0, -0.2], atol=1e-6)
