"""The racing car's physics: sanity, then reality checks against real cars (episode 14).

Reference figures (see the episode 14 devlog for sources): Porsche 911 Carrera (992): 283 kW, 0-100 km/h 4.0-4.2 s,
293 km/h; road-tyre braking 100-0 ~32-34 m and ~1 g steady cornering. F1 car: 798 kg minimum, ~1,000 hp, 0-100
~2.2-2.6 s, ~5 g in fast corners, 100-0 ~15-20 m. Bands are about +-10%: real equations checked against real
numbers, not sim-grade precision.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ailearns.games import race_car as rc
from ailearns.games import race_checks as chk


def test_design_defaults_mid_range_and_rejects_unknown_names():
    d = rc.design(power_kw=300)
    assert d[rc.D["power_kw"]] == 300
    assert d[rc.D["wing_rear"]] == pytest.approx(0.5)
    assert np.allclose(rc.normalise(rc.design()), 0.0, atol=1e-6)
    with pytest.raises(ValueError):
        rc.design(turbo=1)


def test_every_part_weighs_something():
    base = rc.mass_kg(rc.design())
    for name in ("power_kw", "wing_front", "wing_rear", "tyre_front_mm", "tyre_rear_mm"):
        bigger = rc.design(**{name: rc.DESIGN_HIGH[rc.D[name]]})
        assert rc.mass_kg(bigger) > base, name


def test_car_at_rest_stays_put_and_straight_line_stays_straight():
    p = rc.derive(*rc.F1_LIKE)
    s, _ = rc.drive(p, rc.car_at([0, 0], 0.0), jnp.array([0.0, 0.0]))
    assert float(jnp.abs(s.pos).max()) < 1e-4
    s = rc.car_at([0, 0], 0.0, speed=50.0)
    for _ in range(40):
        s, _ = rc.drive(p, s, jnp.array([0.0, 1.0]))
    assert abs(float(s.pos[1])) < 1e-3 and abs(float(s.yaw)) < 1e-4


def test_left_and_right_turns_mirror():
    p = rc.derive(*rc.PORSCHE_911)
    ends = []
    for steer in (0.3, -0.3):
        s = rc.car_at([0, 0], 0.0, speed=15.0)
        for _ in range(40):
            s, _ = rc.drive(p, s, jnp.array([steer, 0.2]))
        ends.append(s)
    assert float(ends[0].pos[1]) > 1.0
    assert float(ends[0].pos[1]) == pytest.approx(-float(ends[1].pos[1]), rel=1e-3)
    assert float(ends[0].vx) == pytest.approx(float(ends[1].vx), rel=1e-4)


def test_many_designs_at_once():
    lo, hi = rc.DESIGN_LOW, rc.DESIGN_HIGH
    designs = jnp.asarray(lo + (hi - lo) * np.random.default_rng(0).random((16, len(lo))), jnp.float32)
    params = jax.vmap(lambda d: rc.derive(d))(designs)
    states = jax.vmap(lambda _: rc.car_at([0, 0], 0.0, speed=20.0))(jnp.arange(16))
    run = jax.jit(jax.vmap(lambda p, s: rc.drive(p, s, jnp.array([0.1, 0.5]))[0]))
    out = run(params, states)
    assert out.vx.shape == (16,) and bool(jnp.all(jnp.isfinite(out.vx)))


def test_weight_balance_follows_the_engine():
    front_engine = rc.design(engine_pos=0.1)
    rear_engine = rc.design(engine_pos=1.0)
    assert rc.weight_front(front_engine) > rc.weight_front(rear_engine) + 0.08
    assert 0.37 <= float(rc.weight_front(*rc.PORSCHE_911)) <= 0.40  # the real 911: ~38-39% front


def test_mid_engine_turns_more_eagerly_than_rear_engine():
    mid, rear = rc.design(engine_pos=0.6), rc.design(engine_pos=1.0)
    assert float(rc.derive(mid).yaw_inertia) < float(rc.derive(rear).yaw_inertia)


def test_four_wheel_drive_costs_weight_and_launches_better():
    rwd, awd = rc.design(power_kw=600, drive_front=0.0), rc.design(power_kw=600, drive_front=0.4)
    assert rc.mass_kg(awd) > rc.mass_kg(rwd) + 50
    assert chk.zero_to(rc.derive(awd)) < chk.zero_to(rc.derive(rwd))


def test_too_low_and_soft_bottoms_out_at_speed():
    low_soft = rc.design(wing_front=1, wing_rear=1, ride_height_mm=15, spring_n_mm=30)
    high_stiff = rc.design(wing_front=1, wing_rear=1, ride_height_mm=60, spring_n_mm=300)
    for d, bottoms in ((low_soft, True), (high_stiff, False)):
        p = rc.derive(d)
        s = rc.car_at([0, 0], 0.0, speed=80.0, ride_height=float(p.ride_height))
        for _ in range(10):
            s, info = rc.drive(p, s, jnp.array([0.0, 0.0]))
        assert bool(info.plank.sum() > 0) == bottoms


def test_lower_floor_more_downforce():
    p_low, p_high = rc.derive(rc.design(ride_height_mm=25, spring_n_mm=300)), rc.derive(rc.design(ride_height_mm=100))
    assert chk.downforce(p_low, 200) > chk.downforce(p_high, 200) * 1.2


def test_slipstream_hook_cuts_drag():
    p = rc.derive(*rc.F1_LIKE)
    s = rc.car_at([0, 0], 0.0, speed=80.0, ride_height=0.03)
    tow = rc.Air(jnp.asarray(0.7), jnp.asarray(0.8), jnp.asarray(0.9))
    clean, _ = rc.drive(p, s, jnp.array([0.0, 1.0]))
    towed, _ = rc.drive(p, s, jnp.array([0.0, 1.0]), air=tow)
    assert float(towed.vx) > float(clean.vx)


def test_downforce_buys_grip_and_costs_top_speed():
    d, spec = rc.F1_LIKE
    low = d.copy()
    low[rc.D["wing_front"]] = low[rc.D["wing_rear"]] = 0.3
    hi_p, lo_p = rc.derive(d, spec), rc.derive(low, spec)
    assert chk.cornering_g(hi_p, 250) > chk.cornering_g(lo_p, 250) + 0.5
    assert chk.top_speed(lo_p) > chk.top_speed(hi_p) + 15
    assert chk.top_speed(lo_p) > 340  # a low-drag F1 trim reaches Monza speeds


@pytest.fixture(scope="module")
def road_car():
    return rc.derive(*rc.PORSCHE_911)


@pytest.fixture(scope="module")
def f1_car():
    return rc.derive(*rc.F1_LIKE)


def test_911_road_test(road_car):
    assert 3.8 <= chk.zero_to(road_car) <= 4.5
    assert 8.8 <= chk.zero_to(road_car, 160) <= 10.0               # Porsche: 9.3 s
    assert 13.8 <= chk.zero_to(road_car, 200, limit_s=30) <= 15.5  # Porsche: 14.5 s
    assert 275 <= chk.top_speed(road_car) <= 305
    assert 30 <= chk.braking(road_car) <= 37
    assert 0.95 <= chk.cornering_g(road_car, 100) <= 1.12


def test_f1_like_road_test(f1_car):
    assert 1.9 <= chk.zero_to(f1_car) <= 2.8
    assert 300 <= chk.top_speed(f1_car) <= 340  # full wing: a high-downforce trim
    assert 14 <= chk.braking(f1_car) <= 20
    assert 2.0 <= chk.cornering_g(f1_car, 100) <= 2.8
    assert 4.5 <= chk.cornering_g(f1_car, 250) <= 6.5  # F1: 4-6.5 g in fast corners (Wikipedia, cited)


def test_cost_cap_forces_a_choice():
    everything = rc.DESIGN_HIGH.copy()
    everything[rc.D["drive_front"]] = 0.5
    cheapest = rc.DESIGN_LOW.copy()
    assert float(rc.cost_k(everything)) > 1.4 * rc.BUDGET_K      # can't have it all
    assert float(rc.cost_k(cheapest)) < 0.5 * rc.BUDGET_K
    big_engine = rc.design(power_kw=750, wing_front=1, wing_rear=1, tyre_front_mm=185, tyre_rear_mm=185)
    assert float(rc.cost_k(big_engine)) > rc.BUDGET_K             # the biggest engine and full wings: over
    setup_only = rc.design(brake_front=0.4), rc.design(brake_front=0.8)
    assert float(rc.cost_k(setup_only[0])) == float(rc.cost_k(setup_only[1]))  # setup is free


def test_spinning_parts_slow_the_launch_most_in_low_gears():
    d, spec = rc.PORSCHE_911
    still = rc.derive(d, replace(spec, wheel_spin_share=0.0, engine_inertia=0.0))
    real = rc.derive(d, spec)
    assert chk.zero_to(real) > chk.zero_to(still) + 0.1
    assert float(real.spin_engine / real.gear_top[0] ** 2) > 5 * float(real.spin_engine / real.gear_top[-1] ** 2)


def test_engine_braking_slows_a_coasting_car():
    d, spec = rc.PORSCHE_911
    ps = [rc.derive(d, replace(spec, engine_brake=0.0)), rc.derive(d, spec)]
    ends = []
    for p in ps:
        s = rc.car_at([0, 0], 0.0, speed=100 / 3.6)
        for _ in range(40):  # two seconds off the throttle and off the brakes
            s, _ = rc.drive(p, s, jnp.array([0.0, 0.0]))
        ends.append(float(s.vx))
    assert ends[1] < ends[0] - 0.3


def test_sliding_sideways_costs_downforce_and_adds_drag(f1_car):
    straight = rc.car_at([0, 0], 0.0, speed=60.0, ride_height=float(f1_car.ride_height))
    sideways = straight._replace(vy=jnp.asarray(60.0 * np.tan(np.radians(10)), jnp.float32))
    _, info_s = rc.drive(f1_car, straight, jnp.array([0.0, 0.0]), substeps=1)
    _, info_y = rc.drive(f1_car, sideways, jnp.array([0.0, 0.0]), substeps=1)
    assert 0.6 < float(info_y.downforce) / float(info_s.downforce) < 0.85  # ~25% gone at 10 degrees


def test_tyres_grip_best_in_their_window(f1_car):
    g = rc.temp_grip(f1_car, jnp.array([40.0, 70.0, 100.0, 130.0, 170.0]))
    assert float(g[2]) == pytest.approx(1.0)
    assert float(g[0]) < float(g[1]) < 1.0 and float(g[4]) < float(g[3]) < 1.0


def test_sliding_heats_tyres_and_wide_tyres_heat_slower():
    def heat(p, pedal):
        s = rc.car_at([0, 0], 0.0, speed=40.0, ride_height=float(p.ride_height))
        for _ in range(10):  # half a second
            s, _ = rc.drive(p, s, jnp.array([0.0, pedal]))
        return float(s.temp.mean() - 90.0)
    narrow = rc.derive(rc.design(tyre_front_mm=185, tyre_rear_mm=185))
    wide = rc.derive(rc.design(tyre_front_mm=405, tyre_rear_mm=405))
    assert heat(narrow, -1.0) > 5 * max(heat(narrow, 0.0), 0.1)  # locked wheels cook the tyres
    assert heat(narrow, -1.0) > 1.5 * heat(wide, -1.0)


def test_a_long_fast_corner_cooks_the_outside_tyres(f1_car):
    v = 250 / 3.6

    def body(s, _):  # turning left near the limit, the pedal holding the speed: the right-hand tyres carry the load
        outs = jax.vmap(lambda pd: rc.drive(f1_car, s, jnp.stack([0.13, pd]), dt=chk.DT))(chk.PEDALS)
        return jax.tree.map(lambda x: x[jnp.argmin(jnp.abs(outs[0].vx - v))], outs[0]), None

    s0 = rc.car_at([0, 0], 0.0, speed=v)
    s, _ = jax.jit(lambda s: jax.lax.scan(body, s, None, length=160))(s0)  # eight seconds
    assert float(s.temp[1]) > float(s.temp[0]) + 15 and float(s.temp[1]) > 115


def test_tyres_wear_from_sliding_and_fall_off_a_cliff(f1_car):
    g = rc.wear_grip(f1_car, jnp.array([0.0, 0.5, 0.8, 1.0]))
    assert float(g[0]) == 1.0 and 0.9 < float(g[1]) < 1.0
    assert float(g[2] - g[3]) > 3 * float(g[1] - g[2]) / 1.5  # steeper past the cliff
    rolling = locked = rc.car_at([0, 0], 0.0, speed=40.0, ride_height=float(f1_car.ride_height))
    for _ in range(10):
        rolling, _ = rc.drive(f1_car, rolling, jnp.array([0.0, 0.0]))
        locked, _ = rc.drive(f1_car, locked, jnp.array([0.0, -1.0]))
    assert float(locked.wear.sum()) > 10 * float(rolling.wear.sum())


def test_fuel_burns_with_the_power_used_and_big_engines_start_heavier():
    small, big = rc.design(power_kw=300), rc.design(power_kw=700)
    assert float(rc.fuel_start_kg(big)) > float(rc.fuel_start_kg(small)) + 1.0
    p = rc.derive(big)
    s = rc.car_at([0, 0], 0.0, speed=50.0, ride_height=float(p.ride_height))
    for _ in range(100):  # five seconds flat out
        s, _ = rc.drive(p, s, jnp.array([0.0, 1.0]))
    full_power_kg = 5.0 * 700e3 / (rc.RACE.thermal_eff * rc.PETROL_J_PER_KG)
    assert 0.5 * full_power_kg < float(s.fuel_used) <= 1.01 * full_power_kg


def test_hills_slow_climbs_crests_lighten_and_banking_helps(f1_car):
    def run(slope, steer=0.0, secs=2.0):
        s = rc.car_at([0, 0], 0.0, speed=30.0, ride_height=float(f1_car.ride_height))
        for _ in range(int(secs / 0.05)):
            s, info = rc.drive(f1_car, s, jnp.array([steer, 0.0]), slope=slope)
        return s, info
    g10 = rc.G * 0.1 / np.sqrt(1.01)  # a 10% grade
    # half a second: short enough that gears, drag and engine braking stay the same, so the hill's pull shows
    flat, _ = run(rc.FLAT, secs=0.5)
    up, _ = run(rc.Slope(jnp.asarray(-g10), jnp.asarray(0.0), jnp.asarray(1.0)), secs=0.5)
    down, _ = run(rc.Slope(jnp.asarray(g10), jnp.asarray(0.0), jnp.asarray(1.0)), secs=0.5)
    assert float(up.vx) < float(flat.vx) - 0.35 and float(down.vx) > float(flat.vx) + 0.35  # ~0.45 m/s each way
    _, crest = run(rc.Slope(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(0.5)), secs=0.5)
    _, level = run(rc.FLAT, secs=0.5)
    assert float(crest.load.sum()) < 0.8 * float(level.load.sum())
    # full lock to the left, past the grip limit, on a road banked down to the left: gravity pulls the car's path
    # round (in a gentle turn the tyres just carry less of the load), so it travels in a tighter curve
    banked, _ = run(rc.Slope(jnp.asarray(0.0), jnp.asarray(rc.G * 0.1), jnp.asarray(1.0)), steer=1.0, secs=1.0)
    plain, _ = run(rc.FLAT, steer=1.0, secs=1.0)
    travel = lambda s: float(s.yaw + jnp.arctan2(s.vy, s.vx))  # noqa: E731 (direction of travel)
    assert travel(banked) > travel(plain) + 0.01
