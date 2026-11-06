"""Race day (episode 15), step 8: the race school's stretches and the widened ep14 brain."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ailearns.agents.ppo_car import forward, init_params
from ailearns.agents.ppo_field import widen
from ailearns.games import race, race_control, race_field, race_school

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


def test_race_control_in_the_school():
    """(ep16 step 8b) a stretch under race control: its senses, its state carried, mid-race cars on two compounds."""
    from ailearns.games import race_control
    cfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
    F = race_field.RaceField(race.make_tracks(range(2), cfg), n_cars=N, laps=5, cfg=cfg)
    ctl = race_control.RaceControl(F)
    for grid_share, used in ((0.0, 2), (1.0, 1)):
        sch = race_school.RaceSchool(F, race_school.SchoolConfig(stretch_s=0.5, grid_share=grid_share), ctl)
        s = sch.start(jax.random.key(4))
        assert np.all(np.asarray(s.control.used).sum(-1) == used)  # mid-race: its stop was before the stretch
        assert sch.obs_size == ctl.obs_size and sch.observe(s).shape == (N, ctl.obs_size)
    step = jax.jit(sch.step)
    for k in range(int(0.5 / F.cfg.dt)):
        s, r, racing, ended, over = step(s, jnp.zeros((N, 3)), jax.random.key(k))
    assert bool(over) and int(s.control.status) == race_control.RACING and r.shape == (N,)
    assert int(s.control.prog.shape[0]) == N and float(jnp.abs(s.control.prog - s.field.cars.progress).max()) == 0.0
    fresh = sch.reset_if(s, over, jax.random.key(5))
    assert int(fresh.field.t) == int(fresh.t0) == 0  # a grid start, race control fresh too
    assert np.all(np.asarray(fresh.control.used).sum(-1) == 1)


def test_es_finish_rounds_under_race_control():
    """(ep16 step 8b) es_field's finish rounds with race control: a 115-sense brain widened to 130, scored after the
    stewards."""
    from ailearns.agents import es_field
    from ailearns.agents.ppo_field import widen
    cfg = es_field.FieldESConfig(mode="finish", n_cars=N, n_learn=2, tracks=2, laps=3, finish_cap_s=1.0, race_control=True,
                                   sigma_rc=0.02)
    school = es_field.make_school(cfg, list(race.TRAIN_SEEDS)[:2])
    F = school.field
    brain = widen(init_params(jax.random.key(0), race.RaceConfig().obs_size, 32, -0.5), F.obs_size, -1.0)
    brain = es_field._fit(brain, "test", school)
    assert brain["body.0.weight"].shape[1] == school.obs_size == F.obs_size + len(race_control.DRIVER_SENSES)
    sig = es_field._sigmas(cfg, brain, F.obs_size)["body.0.weight"]
    np.testing.assert_allclose(np.asarray(sig[0, [0, F.obs_size - 1, F.obs_size]]), [cfg.sigma, cfg.sigma_pit,
                                                                                     cfg.sigma_rc])
    out = jax.jit(es_field.make_play_finish(cfg, school, brain))(brain, jax.random.split(jax.random.key(1), 2))
    assert out["race_s"].shape == (2,) and np.all(np.isfinite(np.asarray(out["race_s"])))
    assert int(out["dsq"].sum()) == 0 and float(out["penalty_s"].sum()) == 0.0  # 1 s of green: nothing to penalise


def test_es_generation_with_the_centre_folded_in():
    """(ep16 step 9g) all members at once with the centre's own rounds as one more member (no noise) give the same
    gradient and stats as members in chunks and the centre's rounds after them."""
    from dataclasses import replace

    from ailearns.agents import es_field
    # (big noise: members ranked by score, and members a hair apart could swap places on float rounding alone, which
    # changes the whole gradient; seen with the default sigma: two members 3e-5 apart)
    cfg = es_field.FieldESConfig(mode="finish", n_cars=N, n_learn=2, tracks=2, laps=3, finish_cap_s=1.0, pop=4,
                                 rounds=2, race_control=True, sigma=1.0, sigma_pit=1.0, sigma_rc=1.0)
    school = es_field.make_school(cfg, list(race.TRAIN_SEEDS)[:2])
    F = school.field
    brain = es_field._fit(init_params(jax.random.key(0), school.obs_size, 32, -0.5), "test", school)
    sig = es_field._sigmas(cfg, brain, F.obs_size)
    play = es_field.make_play_finish(cfg, school, brain)
    keys = jax.random.split(jax.random.key(1), cfg.rounds)
    out = {}
    for chunk in (4, 2):  # 4 = pop: folded; 2: in chunks, the centre after
        gen = jax.jit(es_field.make_generation(replace(cfg, chunk=chunk), play, sig))
        out[chunk] = jax.device_get(gen(brain, jax.random.key(2), keys))
    for n in out[4][0]:
        np.testing.assert_allclose(out[4][0][n], out[2][0][n], rtol=1e-4, atol=1e-6)
    assert out[4][1].keys() == out[2][1].keys()
    for k in out[4][1]:
        np.testing.assert_allclose(out[4][1][k], out[2][1][k], rtol=1e-4, atol=1e-4, err_msg=k)


def test_es_brains_in_16_bit_drive_like_32_bit():
    """(ep16 step 9g) es_field brain16 (the brains' multiplies in 16-bit while training): rounds played by the same
    brain in 16-bit and 32-bit come out nearly the same (1 s races: little time for the tiny differences to grow)."""
    from dataclasses import replace

    from ailearns.agents import es_field
    cfg = es_field.FieldESConfig(mode="finish", n_cars=N, n_learn=2, tracks=2, laps=3, finish_cap_s=1.0,
                                 race_control=True)
    school = es_field.make_school(cfg, list(race.TRAIN_SEEDS)[:2])
    brain = es_field._fit(init_params(jax.random.key(0), school.obs_size, 32, -0.5), "test", school)
    keys = jax.random.split(jax.random.key(1), 2)
    out = {b: jax.device_get(jax.jit(es_field.make_play_finish(replace(cfg, brain16=b), school, brain))(brain, keys))
           for b in (False, True)}
    np.testing.assert_allclose(out[True]["metres"], out[False]["metres"], rtol=1e-3)
    np.testing.assert_allclose(out[True]["race_s"], out[False]["race_s"], rtol=1e-3)
