"""Race day (episode 15): several cars on one circuit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ailearns.games import race, race_field
from ailearns.games import race_car as rc
from ailearns.games.race_checks import cornering_g, test_driver as scripted_driver

N = 4


@pytest.fixture(scope="module")
def field():
    cfg = race.RaceConfig(time_limit=400.0)
    # the race-rule tests run without contact or wakes: these scripted drivers are blind, run into each other and
    # run wide in dirty air
    return race_field.RaceField(race.make_tracks(range(2), cfg), n_cars=N, laps=1, cfg=cfg, contact=False,
                                slipstream=False)


def designs():
    slow = rc.F1_LIKE[0].copy()
    slow[rc.D["power_kw"]] = 250.0
    return jnp.asarray(np.stack([rc.F1_LIKE[0], slow] * (N // 2)))


def test_grid_is_behind_the_line_and_staggered(field):
    f = field.grid(jax.random.PRNGKey(0), 0, designs())
    prog, lat = np.asarray(f.cars.progress), np.asarray(f.cars.lat)
    assert np.all(prog < 0) and np.all(np.diff(prog) < -5)  # pole closest to the line, each slot further back
    assert np.all(np.abs(lat) < field.cfg.width / 2 - 1) and np.all(lat[::2] > 0) and np.all(lat[1::2] < 0)
    assert np.all(np.asarray(field.laps_done(f)) == -1)
    assert list(np.asarray(field.order(f))) == list(range(N))


def run(field, f, policy, steps):
    def body(f, _):
        acts = jax.vmap(lambda i, s: policy(field.game, i, s))(jnp.arange(N), f.cars)
        return field.step(f, acts)[0], None
    return jax.jit(lambda f: jax.lax.scan(body, f, None, steps)[0])(f)


def test_one_lap_race_finishes_in_order(field):
    f = field.grid(jax.random.PRNGKey(0), 0, designs())
    f = run(field, f, lambda g, i, s: scripted_driver(g, s), int(field.cfg.time_limit / field.cfg.dt))
    assert bool(f.flag) and bool(f.done)
    assert np.all(np.asarray(f.finished)) and not np.any(np.asarray(f.cars.crashed))
    order, ft = np.asarray(field.order(f)), np.asarray(f.finish_t)
    assert np.all(np.diff(ft[np.argsort(order)]) >= 0)  # positions follow the finish times
    assert ft[0] < ft[1]  # the 740 kW car beats the 250 kW one


def test_a_crashed_car_is_out_and_stops(field):
    f = field.grid(jax.random.PRNGKey(0), 0, designs())

    def policy(game, i, s):  # car 0 floors it into the wall
        return jnp.where(i == 0, jnp.array([1.0, 1.0]), scripted_driver(game, s))

    f = run(field, f, policy, 200)
    crashed = np.asarray(f.cars.crashed)
    assert crashed[0] and not crashed[1]
    pos0 = np.asarray(f.cars.car.pos[0])
    f2 = run(field, f, policy, 20)
    assert np.allclose(np.asarray(f2.cars.car.pos[0]), pos0)  # frozen where it hit
    assert np.asarray(field.order(f2))[0] == N - 1  # classified behind every running car


def two_cars(field, gap, side=0.0, v_back=40.0, v_front=30.0, yaw_front=0.0):
    """Car 1 `gap` metres (centre to centre) behind car 0 on a straight in open space, both facing +x."""
    f = field.grid(jax.random.PRNGKey(0), 0, designs())
    c = f.cars.car
    pos = jnp.array([[gap, side], [0.0, 0.0]] + [[0.0, 100.0 * k] for k in range(1, N - 1)])
    car = c._replace(pos=pos, yaw=jnp.zeros(N).at[0].set(yaw_front),
                     vx=jnp.zeros(N).at[0].set(v_front).at[1].set(v_back), vy=jnp.zeros(N), r=jnp.zeros(N))
    return f.cars._replace(car=car)


def momentum(field, cars):
    p, *_ = field.boxes(cars)
    m = p.mass - cars.car.fuel_used
    v = jnp.stack([cars.car.vx * jnp.cos(cars.car.yaw) - cars.car.vy * jnp.sin(cars.car.yaw),
                   cars.car.vx * jnp.sin(cars.car.yaw) + cars.car.vy * jnp.cos(cars.car.yaw)], -1)
    return np.asarray((m[:, None] * v)[:2].sum(0))


def test_rear_end_keeps_momentum_and_bounces_a_little(field):
    _, front, rear, _ = field.boxes(two_cars(field, 0.0))
    cars = two_cars(field, float(rear[0] + front[1]) - 0.2)  # overlapping by 20 cm
    every = jnp.ones(N, bool)
    new, bump, _ = field.collide(cars, every, every)
    assert np.allclose(momentum(field, new), momentum(field, cars), atol=1.0)
    closing = float(new.car.vx[1] - new.car.vx[0])
    assert -0.25 * 10 < closing < 0  # 10 m/s closing becomes a small opening: restitution 0.2
    assert float(bump[0]) > 0 and float(bump[0]) == float(bump[1]) and float(bump[2]) == 0
    assert abs(float(new.car.r[0])) < 1e-3  # square hit: no spin


def test_a_tap_on_the_rear_corner_spins_the_car(field):
    _, front, rear, half_w = field.boxes(two_cars(field, 0.0))
    cars = two_cars(field, float(rear[0] + front[1]) - 0.15, side=float(half_w[0] + half_w[1]) * 0.8)
    every = jnp.ones(N, bool)
    new, _, _ = field.collide(cars, every, every)
    # car 0 sits to the left: its right rear is shoved forward (nose turns left) and car 1's left front is held back
    # (its nose turns left too)
    assert float(new.car.r[0]) > 0.05 and float(new.car.r[1]) > 0.05


def test_an_out_car_does_not_move(field):
    _, front, rear, _ = field.boxes(two_cars(field, 0.0))
    cars = two_cars(field, float(rear[0] + front[1]) - 0.2, v_front=0.0)
    moving = jnp.ones(N, bool).at[0].set(False)
    new, _, _ = field.collide(cars, jnp.ones(N, bool), moving)
    assert np.allclose(np.asarray(new.car.pos[0]), np.asarray(cars.car.pos[0]))
    assert float(new.car.vx[0]) == 0.0 and -10 < float(new.car.vx[1]) < 0  # car 1 bounces off it (40 -> -8)


def hit_parts(field, cars):
    every = jnp.ones(N, bool)
    _, _, hit = field.collide(cars, every, every)
    return np.asarray(hit), field.break_parts(jax.tree.map(lambda a: a[:N], field.grid(
        jax.random.PRNGKey(0), 0, designs()).damage), jnp.asarray(hit))


def test_a_slow_tap_breaks_wings_only(field):
    _, front, rear, _ = field.boxes(two_cars(field, 0.0))
    cars = two_cars(field, float(rear[0] + front[1]) - 0.1, v_back=33.0)  # 3 m/s into the back of car 0
    hit, (dmg, wrecked) = hit_parts(field, cars)
    P = race_field.PARTS
    assert hit[1, P.index("wing_front")] > 0 and hit[0, P.index("wing_rear")] > 0
    assert 0.05 < float(dmg.wing_front[1]) < 1 and 0 < float(dmg.wing_rear[0]) < 1
    assert not np.any(np.asarray(dmg.puncture)) and not np.any(np.asarray(wrecked))


def test_a_big_hit_wrecks_both_and_the_wrecks_slide_to_a_stop(field):
    _, front, rear, _ = field.boxes(two_cars(field, 0.0))
    cars = two_cars(field, float(rear[0] + front[1]) - 0.3, v_back=60.0, v_front=35.0)  # 25 m/s closing
    _, (_, wrecked) = hit_parts(field, cars)
    assert np.all(np.asarray(wrecked)[:2])
    f = field.grid(jax.random.PRNGKey(0), 0, designs())
    f = f._replace(wrecked=f.wrecked.at[0].set(True),
                   cars=f.cars._replace(car=f.cars.car._replace(vx=f.cars.car.vx.at[0].set(40.0))))
    seen = []
    for _ in range(3):
        f, reward = field.step(f, jnp.ones((N, 2)) * jnp.array([0.0, 1.0]))
        seen.append(float(f.cars.car.vx[0]))
    assert float(reward[0]) == 0 and not bool(f.cars.crashed[0])  # out of the race, still moving ...
    assert 0 < seen[-1] < seen[0] < 40  # ... braking
    f = f._replace(cars=f.cars._replace(car=f.cars.car._replace(vx=f.cars.car.vx.at[0].set(0.3))))
    f, _ = field.step(f, jnp.zeros((N, 2)))
    assert bool(f.cars.crashed[0]) and np.asarray(field.order(f))[0] == N - 1  # stopped: parked, classified last


def drive_straight(field, damage, steps=60, air=rc.CLEAN_AIR, throttle=0.3, v=50.0):
    g = field.game
    st = g.init(jax.random.PRNGKey(1), design=rc.F1_LIKE[0], track=0, start=0)
    st = st._replace(car=st.car._replace(vx=jnp.asarray(v)))

    def body(st, _):
        return g.step(st, jnp.array([0.0, throttle]), damage, air)[0], None
    st = jax.lax.scan(body, st, None, steps)[0]
    return float(st.car.vx), float(st.car.yaw)


def test_a_bent_steering_arm_pulls_and_a_puncture_slows(field):
    v0, yaw0 = drive_straight(field, race.NO_DAMAGE)
    _, yaw_bent = drive_straight(field, race.NO_DAMAGE._replace(toe=jnp.deg2rad(2.0)))
    v_flat, _ = drive_straight(field, race.NO_DAMAGE._replace(puncture=jnp.array([True, False, False, False])))
    assert yaw_bent > yaw0 + 0.1  # 2 degrees of toe-left: the car turns left with the wheel straight
    assert v_flat < v0 - 0.5


def test_the_tow_pulls_the_car_behind_and_a_parked_car_leaves_no_wake(field):
    v = jnp.full(N, 60.0)
    towed = race.wake(jnp.array([[12.0, 0.0], [0.0, 0.0], [0.0, 50.0], [0.0, 100.0]]), jnp.zeros(N), v)
    parked = race.wake(jnp.array([[12.0, 0.0], [0.0, 0.0], [0.0, 50.0], [0.0, 100.0]]), jnp.zeros(N),
                       v.at[0].set(0.0))
    assert float(towed.drag[1]) < 0.8 and float(parked.drag[1]) == 1.0
    one = jax.tree.map(lambda a: a[1], towed)
    v_clean, _ = drive_straight(field, race.NO_DAMAGE, steps=40, throttle=1.0, v=60.0)
    v_tow, _ = drive_straight(field, race.NO_DAMAGE, steps=40, throttle=1.0, v=60.0, air=one)
    assert v_tow > v_clean + 1.0  # 2 s flat out in the tow: over 1 m/s more


def test_the_field_puts_the_car_behind_in_the_wake(field):
    f = field.grid(jax.random.PRNGKey(0), 0, designs())
    f = f._replace(cars=two_cars(field, 12.0, v_back=60.0, v_front=60.0))
    air = field.air(f, jnp.ones(N, bool))
    assert float(air.drag[1]) < 0.8 and float(air.drag[0]) == 1.0
    assert float(air.down_front[1]) < float(air.down_rear[1]) < float(air.floor[1]) < 1.0  # floors barely notice


def test_drivers_see_the_cars_around_them(field):
    f = field.grid(jax.random.PRNGKey(0), 0, designs())
    obs = field.observe(f)
    assert obs.shape == (N, field.obs_size) and bool(jnp.all(jnp.isfinite(obs)))
    ahead = list(field.cfg.ray_angles).index(0)
    straight = f._replace(cars=two_cars(field, 20.0))
    rays = np.asarray(field.car_lasers(straight))[:, ahead]
    _, _, rear, _ = field.boxes(straight.cars)
    assert abs(rays[1] - (20.0 - float(rear[0]))) < 0.05  # car 1's forward laser stops on car 0's tail
    assert rays[0] == field.cfg.ray_range  # nothing ahead of car 0
    gone = straight._replace(finished=straight.finished.at[0].set(True))
    assert np.asarray(field.car_lasers(gone))[1, ahead] == field.cfg.ray_range  # a finished car is a ghost
    nb = np.asarray(field.neighbours(f))
    assert nb[1, 0, 0] == 1 and abs(nb[1, 0, 1] * field.sense_range - 8.0) < 0.5  # car 1: pole 8 m ahead ...
    assert nb[1, 0, 2] > 0                                                        # ... on its left
    assert nb[0, 0, 0] == 0 and nb[0, 2, 0] == 1 and nb[0, 2, 1] < 0  # pole: nobody ahead, car 1 just behind
    assert nb[N - 1, 2, 0] == 0  # the last car: nobody behind


def test_drivers_feel_what_is_broken(field):
    f = field.grid(jax.random.PRNGKey(0), 0, designs())
    f = f._replace(damage=f.damage._replace(wing_front=f.damage.wing_front.at[1].set(1.0),
                                            puncture=f.damage.puncture.at[2, 3].set(True)))
    broken = np.asarray(field.observe(f))[:, -17:-9]
    assert broken[1, 0] == 1 and broken[2, 7] == 1 and broken[0].sum() == 0


def test_softs_grip_more_and_wear_faster(field):
    g = field.game
    grip = [cornering_g(g.params(jnp.asarray(rc.F1_LIKE[0]), race.Setup(jnp.asarray(c))), 150.0) for c in range(3)]
    assert grip[0] > grip[1] > grip[2]

    def wear(c):  # 20 s of the scripted driver on each compound
        setup = race.Setup(jnp.asarray(c))
        st = g.init(jax.random.PRNGKey(1), design=rc.F1_LIKE[0], track=0, start=0)
        body = lambda st, _: (g.step(st, scripted_driver(g, st), None, rc.CLEAN_AIR, setup)[0], None)
        return float(jax.lax.scan(body, st, None, 400)[0].car.wear.mean())
    w = [wear(c) for c in range(3)]
    assert w[0] > w[1] > w[2] > 0
    f = field.grid(jax.random.PRNGKey(0), 0, designs(), compounds=jnp.array([0, 1, 2, 0]))
    assert list(np.asarray(field.observe(f))[:, -9:-6].argmax(1)) == [0, 1, 2, 0]  # the driver knows its tyres


def test_an_empty_tank_stops_the_engine(field):
    g = field.game

    def flat_out(fuel):
        setup = race.Setup(jnp.asarray(1), jnp.asarray(fuel))
        st = g.init(jax.random.PRNGKey(1), design=rc.F1_LIKE[0], track=0, start=0)
        st = st._replace(car=st.car._replace(vx=jnp.asarray(40.0)))
        body = lambda st, _: (g.step(st, jnp.array([0.0, 1.0]), None, rc.CLEAN_AIR, setup)[0], None)
        return jax.lax.scan(body, st, None, 40)[0].car  # 2 s flat out
    full, dry = flat_out(20.0), flat_out(0.0)
    assert float(full.vx) > 45.0 and float(dry.vx) < 40.0  # no fuel: the car coasts and the engine drags it
    assert float(dry.fuel_used) == 0.0


def test_a_tyre_worn_through_fails(field):
    f = field.grid(jax.random.PRNGKey(0), 0, designs())
    assert np.all((np.asarray(f.burst_at) >= 0.9) & (np.asarray(f.burst_at) <= 1.0))
    wear = f.cars.car.wear.at[0].set(1.0).at[1].set(0.85)  # car 0 worn through; car 1 past the cliff, tread left
    f = f._replace(cars=f.cars._replace(car=f.cars.car._replace(wear=wear)))
    f, _ = jax.jit(field.step)(f, jnp.tile(jnp.array([0.0, 0.5]), (N, 1)))
    assert np.all(np.asarray(f.damage.puncture[0])) and not np.any(np.asarray(f.damage.puncture[1:]))


def test_a_flat_shreds_then_the_rim_breaks(field):
    f = field.grid(jax.random.PRNGKey(0), 0, designs())
    f = f._replace(cars=f.cars._replace(car=f.cars.car._replace(vx=jnp.full(N, 30.0))),
                   damage=f.damage._replace(puncture=f.damage.puncture.at[0, 0].set(True).at[1, 0].set(True)),
                   flat_m=f.flat_m.at[0, 0].set(field.shred_m - 0.5).at[1, 0].set(field.shred_m + field.rim_m - 0.5))
    f, _ = jax.jit(field.step)(f, jnp.tile(jnp.array([0.0, 0.5]), (N, 1)))
    assert float(f.damage.floor[0]) == pytest.approx(field.shred_floor) and not bool(f.wrecked[0])  # shredded
    assert bool(f.wrecked[1])  # on the rim too long: out
    assert float(f.flat_m[2, 0]) == 0.0 and not bool(f.wrecked[2])  # no flat, nothing counted


def test_each_car_gets_a_fuel_allowance(field):
    d = designs()
    f = field.grid(jax.random.PRNGKey(0), 0, d)
    fuel = np.asarray(f.setup.fuel)
    assert fuel[0] > fuel[1] > 1.0  # 740 kW burns more than 250 kW: a bigger allowance
    p = jax.vmap(field.game.params)(f.cars.design, f.setup)
    p_light = jax.vmap(field.game.params)(f.cars.design, f.setup._replace(fuel=f.setup.fuel - 1.0))
    assert np.allclose(np.asarray(p.mass - p_light.mass), 1.0, atol=1e-3)  # the fuel weighs what it weighs
    assert np.allclose(np.asarray(field.observe(f))[:, -6], 1.0)  # on the grid, the whole race is left


@pytest.fixture(scope="module")
def pits():
    cfg = race.RaceConfig(time_limit=400.0, pit_lane=True)
    return race_field.RaceField(race.make_tracks(range(1), cfg), n_cars=N, laps=2, cfg=cfg, contact=False,
                                slipstream=False)


def place(field, f, k, s, lat, v=0.0):
    """Car k at `s` metres from the pit lane's middle, `lat` metres off the centreline, going `v` m/s along the track."""
    g, P = field.game, field.cfg.points
    L = float(field.tracks.length[0])
    i = int(round((float(g.pit_at[0]) + s) / L * P)) % P
    yaw = field.tracks.heading[0, i]
    pos = field.tracks.center[0, i] + lat * jnp.stack([-jnp.sin(yaw), jnp.cos(yaw)])
    car = jax.tree.map(lambda a: a[k], f.cars.car)._replace(pos=pos, yaw=yaw, vx=jnp.asarray(v))
    idx, la, lap_pos = g._locate(jnp.asarray(0), jnp.asarray(i), pos)
    cars = jax.tree.map(lambda a, b: a.at[k].set(b), f.cars,
                        jax.tree.map(lambda a: a[k], f.cars)._replace(car=car, idx=idx, lat=la, lap_pos=lap_pos))
    return f._replace(cars=cars)


def test_the_pit_lane_is_tarmac_behind_a_wall(pits):
    f = pits.grid(jax.random.PRNGKey(0), 0, designs())
    f = place(pits, f, 0, 0.0, -10.5)   # in the lane, in its middle
    f = place(pits, f, 1, 0.0, -7.0)    # on the pit wall
    f = place(pits, f, 2, 200.0, -7.0)  # where the lane rejoins the track: no wall
    f, _ = jax.jit(pits.step)(f, jnp.zeros((N, 2)))
    assert np.allclose(np.asarray(f.cars.surface[0]), 1.0)
    assert list(np.asarray(f.cars.crashed[:3])) == [False, True, False]
    st = jax.tree.map(lambda a: a[3], place(pits, f, 3, 0.0, -3.0).cars)  # on the track beside the wall
    right = float(pits.game.lasers(st)[0])  # the -90 degree laser
    assert 3.0 < right < 5.0  # it sees the pit wall 4 m away, not the outer wall at 13 m


def test_the_pit_limiter_holds_80(pits):
    f = place(pits, pits.grid(jax.random.PRNGKey(0), 0, designs()), 0, -140.0, -10.5, v=40.0)
    step = jax.jit(pits.step)
    def keep_lane(f):  # the scripted driver's steering, aimed at the middle of the lane (it bends with the track);
        # 40% throttle, plenty to go over 80 km/h (flat out at 80 km/h spins a 740 kW car's rear wheels)
        steer = jax.vmap(lambda st: scripted_driver(pits.game, st, offset=-10.5)[0])(f.cars)
        return jnp.stack([steer, jnp.full(N, 0.4)], -1)
    for _ in range(60):  # 3 s
        f, _ = step(f, keep_lane(f))
    assert float(f.cars.car.vx[0]) < pits.cfg.pit_limit + 0.5 and not bool(f.cars.crashed[0])


def test_a_stop_in_the_box_fits_new_tyres_and_a_nose(pits):
    f = pits.grid(jax.random.PRNGKey(0), 0, designs())
    f = place(pits, f, 1, float(pits.box_s[1]), -10.5)
    worn = f.cars.car._replace(wear=f.cars.car.wear.at[1].set(0.6))
    f = f._replace(cars=f.cars._replace(car=worn), plan=f.plan.at[1].set(0),
                   damage=f.damage._replace(wing_front=f.damage.wing_front.at[1].set(0.5),
                                            puncture=f.damage.puncture.at[1, 2].set(True)))
    step = jax.jit(pits.step)
    go = jnp.tile(jnp.array([0.0, 0.5]), (N, 1))
    f, _ = step(f, go)
    assert int(f.stops[1]) == 1 and int(f.setup.compound[1]) == 0 and float(f.damage.wing_front[1]) == 0
    assert np.all(np.asarray(f.cars.car.wear[1]) == 0)
    assert not np.any(np.asarray(f.damage.puncture[1]))
    for _ in range(int((pits.tyre_s + pits.nose_s) / pits.cfg.dt) - 2):  # held while the crew works
        f, _ = step(f, go)
    assert float(f.cars.car.vx[1]) < 0.1
    for _ in range(40):
        f, _ = step(f, go)
    assert float(f.cars.car.vx[1]) > 3.0 and int(f.stops[1]) == 1  # away, and no second stop


def test_the_team_pays_for_a_stop_it_called(pits):
    """A called-in car that stops on its mark is paid back `stop_pay_s` seconds at its own race pace; a stop nobody
    called earns nothing."""
    f = pits.grid(jax.random.PRNGKey(0), 0, designs())
    for k in (1, 2):
        f = place(pits, f, k, float(pits.box_s[k]), -10.5)
    pace, t = 50.0, 1000
    f = f._replace(t=jnp.asarray(t), box_call=f.box_call.at[1].set(True),
                   cars=f.cars._replace(progress=jnp.full(N, pace * t * pits.cfg.dt)))
    f2, reward = jax.jit(pits.step)(f, jnp.tile(jnp.array([0.0, -1.0]), (N, 1)))
    assert list(np.asarray(f2.stops[1:3])) == [1, 1] and not bool(f2.box_call[1])
    paid = pits.stop_pay_s * pace * pits.cfg.progress_reward
    assert abs(float(reward[1]) - paid) < 0.05 * paid and abs(float(reward[2])) < 0.05 * paid


def test_tyre_wear_scales_with_race_distance():
    """(step 9e) gp_km: a set wears x gp_km / race distance x tyre_cal; off by default."""
    cfg = race.RaceConfig(time_limit=60.0, pit_lane=True)
    tracks = race.make_tracks([0], cfg)
    d = jnp.tile(jnp.asarray(rc.F1_LIKE[0])[None], (8, 1))
    plain = race_field.RaceField(tracks, laps=10, cfg=cfg).grid(jax.random.PRNGKey(0), 0, d)
    gp = race_field.RaceField(tracks, laps=10, cfg=cfg, gp_km=305.0, tyre_cal=2.0).grid(jax.random.PRNGKey(0), 0, d)
    assert np.allclose(plain.setup.wear, 1.0)
    race_km = 10 * float(tracks.length[0]) / 1000.0
    assert np.allclose(gp.setup.wear, 2.0 * 305.0 / race_km, rtol=1e-5)
    g = race.RaceGame(tracks, cfg)
    one = jax.tree.map(lambda x: x[0], gp.setup)
    base = jax.tree.map(lambda x: x[0], plain.setup)
    assert np.isclose(g.params(d[0], one).wear_rate, g.params(d[0], base).wear_rate * float(one.wear), rtol=1e-5)


def test_marshals_recover_an_out_car():
    """(step 9l) A stopped car stays an obstacle the cars behind see, until the marshals clear it after recover_s."""
    cfg = race.RaceConfig(time_limit=400.0)
    F = race_field.RaceField(race.make_tracks(range(1), cfg), n_cars=N, laps=1, cfg=cfg, recover_s=2.0)
    f = F.grid(jax.random.PRNGKey(0), 0, designs())
    f = f._replace(cars=f.cars._replace(crashed=f.cars.crashed.at[0].set(True)))  # pole, out on the grid
    brake = jnp.zeros((N, 2)).at[:, 1].set(-1.0)
    step = jax.jit(F.step)
    seen = lambda f: float(F.neighbours(f)[1, :, 0].sum())  # noqa: E731  cars car 1 notices
    before = seen(f)
    for _ in range(int(1.0 / cfg.dt)):
        f, _ = step(f, brake)
    assert not bool(F.gone(f)[0]) and seen(f) == before  # 1 s: still there
    for _ in range(int(1.5 / cfg.dt)):
        f, _ = step(f, brake)
    assert bool(F.gone(f)[0]) and not bool(F.gone(f)[1:].any())
    assert seen(f) == before - 1  # gone from car 1's senses (and from contact: solid excludes it)


def test_a_dry_car_retires_and_leaves_the_circuit():
    """(step 9l) An empty tank retires the car: it pulls off (a ghost at once), never finishes, ranks with the out cars."""
    cfg = race.RaceConfig(time_limit=400.0)
    F = race_field.RaceField(race.make_tracks(range(1), cfg), n_cars=N, laps=1, cfg=cfg)
    fuel = F.fuel_allowance(0, designs()).at[0].set(0.05)  # car 0: a few metres of fuel
    f = F.grid(jax.random.PRNGKey(0), 0, designs(), fuel=fuel)
    go = jnp.zeros((N, 2)).at[:, 1].set(1.0)
    step = jax.jit(F.step)
    for _ in range(int(3.0 / cfg.dt)):
        f, _ = step(f, go)
    assert bool(F.dry_out(f)[0]) and bool(F.gone(f)[0]) and not bool(F.gone(f)[1:].any())
    at = np.asarray(f.cars.car.pos[0])
    f, _ = step(f, go)
    assert np.allclose(np.asarray(f.cars.car.pos[0]), at)  # retired: it doesn't move on
    order, running = np.asarray(F.order(f)), ~np.asarray(f.cars.crashed | f.wrecked | F.dry_out(f))
    assert running.any() and (order[running] < order[0]).all()  # behind every car still racing: it's out
