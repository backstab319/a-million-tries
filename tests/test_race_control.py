"""Race control (episode 16): marshalling sectors and flags."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ailearns.games import race, race_control, race_field
from ailearns.games import race_car as rc
from ailearns.games.race_checks import test_driver as scripted_driver

N = 4


@pytest.fixture(scope="module")
def ctl():
    cfg = race.RaceConfig(time_limit=400.0, pit_lane=True)
    F = race_field.RaceField(race.make_tracks(range(1), cfg), n_cars=N, laps=2, cfg=cfg, contact=False,
                             slipstream=False)
    return race_control.RaceControl(F)


def racing(ctl, spots, speed=40.0):
    """A field on its 60th second: car k at spots[k] = (metres along the lap, metres left of the centreline)."""
    F = ctl.field
    designs = jnp.asarray(np.stack([rc.F1_LIKE[0]] * N))
    f = F.grid(jax.random.PRNGKey(0), 0, designs)
    cars = [F.place(jax.random.PRNGKey(k), 0, designs[k], m, side, speed) for k, (m, side) in enumerate(spots)]
    return f._replace(cars=jax.tree.map(lambda *a: jnp.stack(a), *cars), t=jnp.asarray(int(60.0 / F.cfg.dt)))


def stop(f, k, out=False, out_s=0.0):
    cars = f.cars._replace(car=f.cars.car._replace(vx=f.cars.car.vx.at[k].set(0.0)))
    if out:
        cars = cars._replace(crashed=cars.crashed.at[k].set(True))
    return f._replace(cars=cars, out_s=f.out_s.at[k].set(out_s))


def spread(ctl):
    """Four cars in the middle of sectors 3, 5, 7, 9, on the centreline."""
    w = float(ctl.sector_len(0))
    return [((s + 0.5) * w, 0.0) for s in (3, 5, 7, 9)]


def test_sectors_are_about_250_m(ctl):
    L, n = float(ctl.field.tracks.length[0]), int(ctl.sectors[0])
    assert n >= 4 and abs(L / n - ctl.sector_m) <= ctl.sector_m / 2 / n + 1e-3
    f = racing(ctl, spread(ctl))
    assert list(np.asarray(ctl.sector(f.cars))) == [3, 5, 7, 9]


def test_all_green_while_racing(ctl):
    f = racing(ctl, spread(ctl))
    assert not np.asarray(ctl.flags(f)).any()
    assert not np.asarray(ctl.senses(f))[:, :4].any()


def test_a_car_stopped_on_the_track_double_yellow_and_a_warning_before(ctl):
    f = stop(racing(ctl, spread(ctl)), 1, out=True, out_s=30.0)  # car 1, sector 5
    flags = np.asarray(ctl.flags(f))
    assert flags[5] == race_control.DOUBLE and flags[4] == race_control.YELLOW
    assert (np.delete(flags, [4, 5]) == race_control.GREEN).all()
    sense = np.asarray(ctl.senses(f))
    assert list(sense[1, :4]) == [0, 1, 0, 0]  # car 1 is in the double yellow
    assert list(sense[0, :4]) == [0, 0, 1, 0]  # car 0 (sector 3): the warning in sector 4 is next
    f = racing(ctl, [(4.5 * float(ctl.sector_len(0)), 0.0)] + spread(ctl)[1:])  # car 0 now in sector 4
    sense = np.asarray(ctl.senses(stop(f, 1, out=True, out_s=30.0)))
    assert list(sense[0, :4]) == [1, 0, 0, 1] and 0.4 < sense[0, 4] < 0.6  # yellow here, double ahead, half a sector


def test_a_crawling_car_is_an_incident_and_clears_when_it_goes(ctl):
    f = stop(racing(ctl, spread(ctl)), 2)  # car 2 stopped (a spin), not out
    assert np.asarray(ctl.flags(f))[7] == race_control.DOUBLE
    assert not np.asarray(ctl.flags(racing(ctl, spread(ctl)))).any()  # on its way again: green


def test_off_the_track_single_yellow_until_the_post_clears_it(ctl):
    spots = spread(ctl)
    spots[2] = (spots[2][0], 12.0)  # car 2 in the run-off
    f = racing(ctl, spots)
    for out_s, want in ((30.0, race_control.YELLOW), (ctl.post_clear_s + 1.0, race_control.GREEN)):
        flags = np.asarray(ctl.flags(stop(f, 2, out=True, out_s=out_s)))
        assert flags[7] == want and flags[6] == race_control.GREEN


def test_no_flags_for_a_recovered_car_the_start_or_the_pit_lane(ctl):
    F = ctl.field
    f = stop(racing(ctl, spread(ctl)), 1, out=True, out_s=F.recover_s + 1.0)  # the marshals have taken it
    assert not np.asarray(ctl.flags(f)).any()
    designs = jnp.asarray(np.stack([rc.F1_LIKE[0]] * N))
    assert not np.asarray(ctl.flags(F.grid(jax.random.PRNGKey(0), 0, designs))).any()  # a standing start
    f = racing(ctl, spread(ctl))
    lane_car = F.place(jax.random.PRNGKey(9), 0, designs[0], float(F.game.pit_at[0]), F.cfg.fast_lat, 0.0)
    cars = jax.tree.map(lambda a, b: a.at[0].set(b), f.cars, lane_car)
    assert bool(F.game.in_pit_lane(lane_car))
    assert not np.asarray(ctl.flags(f._replace(cars=cars))).any()  # stopped in the lane (in its box): no flag


def test_flags_jit(ctl):
    f = stop(racing(ctl, spread(ctl)), 1, out=True, out_s=30.0)
    assert np.array_equal(np.asarray(jax.jit(ctl.senses)(f)), np.asarray(ctl.senses(f)))


def test_the_reference_car_laps_like_the_test_driver(ctl):
    """The speed profile is the test driver's lap without its crashes; the VSC minimum is ~1.8x it."""
    v = race_control.reference_speeds(ctl.field.tracks)
    lap = float((float(ctl.field.tracks.length[0]) / v.shape[1] / v[0]).sum())
    assert 40.0 < lap < 150.0
    assert np.allclose(np.asarray(ctl.min_pace) * v, 1.4 * 1.3, rtol=1e-5)


def test_the_director_vsc_at_the_side_sc_on_the_line_or_for_two(ctl):
    F = ctl.field
    beside = F.cfg.width / 2 + 1.5  # its centre past the edge, but not clear of the track
    edge = F.cfg.width / 2 - 1.5    # on the tarmac, at its edge (step 9a: a VSC, not the Safety Car)
    c0 = ctl.init(racing(ctl, spread(ctl)))
    for spots, out, want in (([(m, beside) if k == 1 else (m, s) for k, (m, s) in enumerate(spread(ctl))], [1], race_control.VSC),
                             ([(m, edge) if k == 1 else (m, s) for k, (m, s) in enumerate(spread(ctl))], [1], race_control.VSC),
                             (spread(ctl), [1], race_control.SC),
                             ([(m, beside) for m, _ in spread(ctl)], [1, 2], race_control.SC)):
        f = racing(ctl, spots)
        for k in out:
            f = stop(f, k, out=True, out_s=10.0)
        assert int(ctl.direct(f, c0).status) == want
    assert int(ctl.direct(racing(ctl, spread(ctl)), c0).status) == race_control.RACING


def test_vsc_ends_after_the_track_is_clear(ctl):
    F = ctl.field
    spots = spread(ctl)
    spots[1] = (spots[1][0], F.cfg.width / 2 + 1.5)
    f = stop(racing(ctl, spots), 1, out=True, out_s=10.0)
    c = ctl.direct(f, ctl.init(f))
    assert int(c.status) == race_control.VSC and int(c.vsc_n) == 1
    f = f._replace(out_s=f.out_s.at[1].set(F.recover_s + 1.0))  # the marshals have taken it
    direct = jax.jit(ctl.direct)
    for _ in range(int((ctl.vsc_end_s - 1.0) / F.cfg.dt)):
        c = direct(f, c)
    assert int(c.status) == race_control.VSC  # "VSC ENDING"
    for _ in range(int(2.0 / F.cfg.dt)):
        c = direct(f, c)
    assert int(c.status) == race_control.RACING and int(c.vsc_n) == 1


def follower(F, ctl, f, c):
    """A scripted field under race control: the test driver, at the minimum pace's speed (a bit more with time banked)
    under the VSC or the Safety Car, and easing off behind the car ahead (the Safety Car too)."""
    act = jax.vmap(lambda st: scripted_driver(F.game, st))(f.cars)
    v = jnp.hypot(f.cars.car.vx, f.cars.car.vy)
    near = F.neighbours(f, ctl.bodies(c))[:, 0]  # the nearest ahead: there, gap / range, side, faster by / 20
    gap = jnp.where(near[:, 0] > 0, near[:, 1] * F.sense_range, 999.0)
    target = jnp.where(c.status != race_control.RACING,
                       (1.0 + 0.2 * jnp.clip(c.delta, 0.0, 2.0)) / ctl.min_pace[0, f.cars.idx], 999.0)
    behind = (gap < 40.0) & (jnp.abs(near[:, 2]) < 0.4)
    target = jnp.where(behind, jnp.minimum(target, v + near[:, 3] * 20.0 + (gap - 20.0) * 0.3), target)
    return act.at[:, 1].set(jnp.minimum(act[:, 1], jnp.clip(0.5 * (target - v), -0.5, 1.0)))


@pytest.fixture(scope="module")
def long_race(ctl):
    F = race_field.RaceField(ctl.field.tracks, n_cars=N, laps=10, cfg=ctl.field.cfg, contact=False, slipstream=False)
    return race_control.RaceControl(F)


def test_a_safety_car_period(long_race):
    """Car 0 out on the grid: the Safety Car waits at the end of the pit lane for the leader, leads the queue, "in
    this lap" once the field is queued behind it, into the pit lane, green at the leader's next line."""
    ctl = long_race
    F, dt = ctl.field, ctl.field.cfg.dt
    f = F.grid(jax.random.PRNGKey(0), 0, jnp.asarray(np.stack([rc.F1_LIKE[0]] * N)))
    f = f._replace(cars=f.cars._replace(crashed=f.cars.crashed.at[0].set(True)))

    def body(carry, _):
        f, c = carry
        f, c, _ = ctl.step(f, c, follower(F, ctl, f, c))
        sc = jax.tree.map(lambda a: a[0], c.sc)
        return (f, c), (c.status, c.sc_on[0], c.sc_wait, sc.lat, F.game.from_pit(sc.track, sc.idx),
                        F.laps_done(f)[1], ctl.queued(f, c))
    (f, c), out = jax.jit(lambda f: jax.lax.scan(body, (f, ctl.init(f)), None, int(500.0 / dt)))(f)
    status, on, wait, lat, s, laps, queued = map(np.asarray, out)
    assert status[0] == race_control.SC and int(c.sc_n) == 1 and on[0] and wait[0]
    go = np.argmax(~wait)
    assert go > 0 and on[go]  # it waited for the leader, then led
    call = np.argmax(status == race_control.SC_IN)
    assert call > go and queued[call] and call * dt >= F.recover_s  # the track clear and the field queued
    off = np.argmax(~on & (np.arange(len(on)) > go))
    assert off > call and s[off] < 0 and lat[off] < F.cfg.pit_wall_lat  # into the pit lane at its entry
    green = np.argmax(status == race_control.RACING)
    assert green > off and laps[green] > laps[off] and (status[green:] == race_control.RACING).all()
    assert lat[on].min() > F.cfg.fast_lat - 1.0 and lat[on].max() < F.cfg.width / 2  # on the track or in the lane
    assert int(c.breaks[1:].sum()) <= 6  # the scripted field mostly keeps its delta (not race control's job)


def test_the_vsc_delta(long_race):
    """Car 0 out beside the tarmac: a VSC. Car 1 races on at the test driver's pace and breaks the minimum time; car 2
    drives at 80% of the minimum pace's speed and car 3 at 60%: they bank time and break nothing."""
    ctl = long_race
    F, dt = ctl.field, ctl.field.cfg.dt
    designs = jnp.asarray(np.stack([rc.F1_LIKE[0]] * N))
    f = F.grid(jax.random.PRNGKey(0), 0, designs)
    out = F.place(jax.random.PRNGKey(5), 0, designs[0], float(F.tracks.length[0]) / 2, F.cfg.width / 2 + 1.5)
    cars = jax.tree.map(lambda a, b: a.at[0].set(b), f.cars, out)
    f = f._replace(cars=cars._replace(crashed=cars.crashed.at[0].set(True)))

    def body(carry, _):
        f, c = carry
        act = jax.vmap(lambda st: scripted_driver(F.game, st))(f.cars)
        v = jnp.hypot(f.cars.car.vx, f.cars.car.vy)
        slow = jnp.array([1.0, 1.0, 0.8, 0.6]) / ctl.min_pace[0, f.cars.idx]
        capped = jnp.minimum(act[:, 1], jnp.clip(0.5 * (slow - v), -0.5, 1.0))
        f, c, _ = ctl.step(f, c, act.at[:, 1].set(jnp.where(jnp.arange(N) >= 2, capped, act[:, 1])))
        return (f, c), (c.status, c.delta)
    (f, c), (status, delta) = jax.jit(lambda f: jax.lax.scan(body, (f, ctl.init(f)), None, int(160.0 / dt)))(f)
    status, delta = np.asarray(status), np.asarray(delta)
    assert status[0] == race_control.VSC and int(c.vsc_n) == 1 and int(c.sc_n) == 0
    end = np.argmax(status == race_control.RACING)
    assert abs(end * dt - (F.recover_s + ctl.vsc_end_s)) < 1.0  # recovered, then "VSC ENDING", then green
    assert int(c.breaks[1]) > 3 and int(c.breaks[2]) == 0 and int(c.breaks[3]) == 0
    assert 0 < delta[end - 1, 2] < delta[end - 1, 3]  # banked time, more for the slower car
    assert (delta[end:] == 0).all()
    sense = np.asarray(ctl.status_senses(c))
    assert sense.shape == (N, race_control.N_STATUS_SENSES) and not sense.any()


def test_the_safety_car_is_solid_and_seen():
    """A car driving into the Safety Car hits it (and pushes it); its lasers and its senses see it; off the circuit
    it's neither."""
    cfg = race.RaceConfig(time_limit=400.0, pit_lane=True)
    F = race_field.RaceField(race.make_tracks(range(1), cfg), n_cars=N, laps=2, cfg=cfg, slipstream=False)
    ctl = race_control.RaceControl(F)
    designs = jnp.asarray(np.stack([rc.F1_LIKE[0]] * N))
    w = float(ctl.sector_len(0))
    f = F.grid(jax.random.PRNGKey(0), 0, designs)
    cars = [F.place(jax.random.PRNGKey(k), 0, designs[k], (3 + 2 * k + 0.5) * w, 0.0, 20.0) for k in range(N)]
    f = f._replace(cars=jax.tree.map(lambda *a: jnp.stack(a), *cars))
    c = ctl.init(f)
    sc = F.place(jax.random.PRNGKey(9), 0, ctl.sc_design, 3.5 * w + 7.0, 0.0, 0.0)  # 7 m ahead of car 0, stopped
    c = c._replace(sc=jax.tree.map(lambda a: a[None], sc), sc_on=jnp.ones(1, bool))
    near = np.asarray(F.neighbours(f, ctl.bodies(c)))[0, 0]
    assert near[0] == 1 and abs(near[1] * F.sense_range - 7.0) < 1.0
    ahead = np.asarray(F.car_lasers(f, ctl.bodies(c)))[0, len(cfg.ray_angles) // 2]
    assert ahead < 7.0  # the middle laser: to its tail
    off = c._replace(sc_on=jnp.zeros(1, bool))
    assert np.asarray(F.neighbours(f, ctl.bodies(off)))[0, 0, 0] == 0
    coast = jnp.zeros((N, 2))
    hit = False
    for _ in range(int(1.0 / cfg.dt)):
        f, _, b = F.step_with(f, coast, ctl.bodies(c))
        hit = hit or float(f.bump[0]) > 0
        c = c._replace(sc=b.cars)
    assert hit and float(c.sc.car.vx[0]) > 0.5  # it was hit and pushed


def test_the_red_flag_three_in_a_sector_or_the_track_blocked(ctl):
    F = ctl.field
    w = float(ctl.sector_len(0))
    at = 5.5 * w
    c0 = ctl.init(racing(ctl, spread(ctl)))
    for lats, ms, want in (((-2.5, 2.5), (at, at + 4), race_control.RED),        # side by side, no 3 m gap
                           ((-4.0, 4.0), (at, at + 4), race_control.SC),         # a 5 m gap between them
                           ((-2.5, 2.5), (at, at + 40), race_control.SC),        # 40 m apart: room to weave through
                           ((5.0, 5.0, 5.0), (at - 80, at, at + 80), race_control.RED)):  # three in one sector
        spots = [(m, lat) for m, lat in zip(ms, lats)] + spread(ctl)[len(ms):]
        f = racing(ctl, spots)
        for k in range(len(ms)):
            f = stop(f, k, out=True, out_s=10.0)
        assert int(ctl.direct(f, c0).status) == want, (lats, ms)


@pytest.fixture(scope="module")
def red_race():
    """Eight cars on circuit 0, the field racing: cars 0-2 stopped side by side 1.5 km after the pit lane's middle
    (the track blocked), cars 3-7 spread round the lap behind them at 20 m/s, on `follower` (the script drives them
    under the red flag)."""
    cfg = race.RaceConfig(time_limit=900.0, pit_lane=True)
    n = 8
    F = race_field.RaceField(race.make_tracks(range(1), cfg), n_cars=n, laps=10, cfg=cfg, slipstream=False)
    ctl = race_control.RaceControl(F)
    L, dt = float(F.tracks.length[0]), cfg.dt
    designs = jnp.asarray(np.stack([rc.F1_LIKE[0]] * n))
    at = (float(F.game.pit_at[0]) + 1500.0) % L
    spots = [(at, -4.0, 0.0), (at + 3, 0.0, 0.0), (at - 2, 4.0, 0.0)] + [((at - 300 - 550 * k) % L, 0.0, 20.0)
                                                                         for k in range(5)]
    cars = [F.place(jax.random.PRNGKey(k), 0, designs[k], m, side, v) for k, (m, side, v) in enumerate(spots)]
    f = F.grid(jax.random.PRNGKey(0), 0, designs)
    f = f._replace(cars=jax.tree.map(lambda *a: jnp.stack(a), *cars), t=jnp.asarray(int(60.0 / dt)))
    f = f._replace(cars=f.cars._replace(crashed=f.cars.crashed.at[:3].set(True)), out_s=f.out_s.at[:3].set(1.0),
                   damage=f.damage._replace(wing_front=f.damage.wing_front.at[5].set(0.5)))  # car 5: a broken nose

    def body(carry, _):
        f, c = carry
        f, c, _ = ctl.step(f, c, follower(F, ctl, f, c))
        return (f, c), (c.status, c.slot, F.pit_s(f.cars), f.cars.lat, jnp.hypot(f.cars.car.vx, f.cars.car.vy),
                        f.cars.crashed | f.wrecked, f.cars.car.wear.max(-1), f.damage.wing_front, c.sc_on[0],
                        c.sc_wait, f.bump)
    (f, c), out = jax.jit(lambda f: jax.lax.scan(body, (f, ctl.init(f)), None, int(600.0 / dt)))(f)
    return ctl, c, dict(zip(("status", "slot", "ps", "lat", "v", "out", "wear", "wing", "sc_on", "sc_wait", "bump"),
                            map(np.asarray, out)))


def test_a_red_flag_period(red_race):
    """Red at once; every racing car into the pit lane at the limiter's speed, queued in the fast lane in the order
    they arrive, `slot_m` apart; once the track is clear and the field queued: repaired, new tyres, out behind the
    Safety Car (which doesn't wait), "in this lap" only once the whole field is out of the lane."""
    ctl, c, o = red_race
    F, dt = ctl.field, ctl.field.cfg.dt
    st = o["status"]
    assert st[0] == race_control.RED and int(c.red_n) == 1 and int(c.sc_n) == 0
    resume = np.argmax(st != race_control.RED)
    assert st[resume] == race_control.SC and resume * dt >= F.recover_s  # the blockage cleared first
    assert not o["out"][:resume, 3:].any() and o["bump"][:resume].max() == 0  # nobody hit anything under the red flag
    assert o["v"][:resume, 3:].max() < F.cfg.pit_limit + 0.5
    k = resume - 1  # the queue, just before the resumption
    slots, ps = o["slot"][k, 3:], o["ps"][k, 3:]
    assert sorted(slots) == list(range(5)) and (o["v"][k, 3:] < 1.0).all()
    assert np.allclose(ps, ctl.head_s - slots * ctl.slot_m, atol=2.0)  # stopped on its place
    assert np.allclose(o["lat"][k, 3:], F.cfg.fast_lat, atol=1.0)      # in the fast lane
    assert o["wear"][k, 3:].min() > 0 and (o["wear"][resume, 3:] == 0).all()  # new tyres
    assert o["wing"][k, 5] > 0 and o["wing"][resume, 5] == 0                   # its nose repaired
    assert o["sc_on"][resume] and not o["sc_wait"][resume]                     # leads them straight out
    call = np.argmax(st == race_control.SC_IN)
    assert call > resume and (o["slot"][call] < 0).all()  # "in this lap" with the field out of the lane
    assert o["lat"][resume:call, 3:].max() < F.cfg.width / 2 and not o["out"][:call, 3:].any()


def test_pit_entry_closed(ctl):
    """An out car in the pit lane closes the pit entry: the team can't call a car in, and one entering anyway is
    counted; open again once its crew has moved it."""
    F = ctl.field
    L = float(F.tracks.length[0])
    pit = float(F.game.pit_at[0])
    spots = [(pit, F.cfg.fast_lat), ((pit - 1000) % L, 0.0), ((pit - 2000) % L, 0.0), ((pit - 1500) % L, 0.0)]
    f = stop(racing(ctl, spots), 0, out=True, out_s=5.0)
    f = f._replace(box_call=f.box_call.at[1].set(True))
    c = ctl.direct(f, ctl.init(f))
    assert bool(c.pit_closed) and int(c.status) == race_control.RACING  # in the lane: no flags, no VSC
    f2, c2, _ = ctl.step(f, c, jnp.zeros((N, 2)))
    assert not bool(f2.box_call[1])
    inside = racing(ctl, [spots[0], (pit - 100, F.cfg.fast_lat)] + spots[2:])  # car 1 has entered the lane
    inside = stop(inside, 0, out=True, out_s=5.0)
    assert int(ctl.direct(inside, c).closed_in[1]) == 1
    gone = inside._replace(out_s=inside.out_s.at[0].set(F.recover_lane_s + 1.0))
    assert not bool(ctl.direct(gone, c).pit_closed)


def test_a_collision_caused_in_the_pit_lane(ctl):
    """(step 13d) a car to blame for a new contact (`Field.fault`) in the pit lane: counted, 5 s; on the track, or in
    the red flag's queue: not."""
    F = ctl.field
    L = float(F.tracks.length[0])
    pit = float(F.game.pit_at[0])
    spots = [(pit, F.cfg.fast_lat), ((pit - 1000) % L, 0.0), ((pit - 2000) % L, 0.0), ((pit - 1500) % L, 0.0)]
    f = racing(ctl, spots)
    blame = jnp.asarray([True, True, False, False])  # car 0 in the lane, car 1 on the track
    c = ctl.direct(f._replace(fault=blame), ctl.init(f))
    assert list(np.asarray(c.lane_hits)) == [1, 0, 0, 0]
    assert list(np.asarray(ctl.penalty_s(c))) == [5.0, 0.0, 0.0, 0.0]
    queued = ctl.init(f)._replace(status=jnp.asarray(race_control.RED), slot=jnp.asarray([0, -1, -1, -1], jnp.int32))
    assert int(ctl.direct(f._replace(fault=blame), queued).lane_hits[0]) == 0


def test_overtaking_under_the_vsc_given_back_or_penalised(ctl):
    """Car 0 out beside the tarmac (a VSC). Car 2 passes car 1: owed back for a sector; kept past it, 5 s; given back,
    nothing. Passing a car that's out, or passing under green, is fine."""
    F = ctl.field
    L, w = float(F.tracks.length[0]), float(ctl.sector_len(0))
    m = 6.5 * w
    side = F.cfg.width / 2 + 1.5

    def field(car2, car1=m, crash1=False, out0=True):
        f = racing(ctl, [(2.0 * w, side), (car1, 0.0), (car2, 2.0), (12.5 * w % L, 0.0)], speed=20.0)
        f = stop(f, 0, out=True, out_s=10.0) if out0 else f
        return stop(f, 1, out=True, out_s=1.0) if crash1 else f

    c = ctl.direct(field(m - 5.0), ctl.init(field(m - 5.0)))
    assert int(c.status) == race_control.VSC
    c1 = ctl.direct(field(m + 5.0), c)
    assert float(c1.owe[2, 1]) == ctl.sector_m and int(c1.passes.sum()) == 0
    kept = ctl.direct(field(m + 5.0 + ctl.sector_m + 5.0, car1=m + 5.0), c1)  # a sector on, still ahead of it
    assert int(kept.passes[2]) == 1 and float(kept.owe.max()) == 0.0
    back = ctl.direct(field(m + 10.0, car1=m + 20.0), c1)  # car 1 ahead again
    assert int(back.passes.sum()) == 0 and float(back.owe.max()) == 0.0
    out = ctl.direct(field(m + 5.0, crash1=True), ctl.direct(field(m - 5.0, crash1=True), c))
    assert float(out.owe.max()) == 0.0  # car 1 out: an obvious problem
    g0 = ctl.init(field(m - 5.0, out0=False))
    green = ctl.direct(field(m + 5.0, out0=False), ctl.direct(field(m - 5.0, out0=False), g0))
    assert int(green.status) == race_control.RACING and float(green.owe.max()) == 0.0
    sc = F.place(jax.random.PRNGKey(9), 0, ctl.sc_design, m, 0.0, 20.0)  # the Safety Car where car 1 was, leading
    on = c._replace(sc=jax.tree.map(lambda a: a[None], sc), sc_on=jnp.ones(1, bool), sc_prev=jnp.asarray(m),
                    status=jnp.asarray(race_control.SC))
    f_before = field(m - 5.0, car1=m - 60.0)
    passed = ctl.direct(field(m + 5.0, car1=m - 60.0), ctl.direct(f_before, on))
    assert float(passed.owe_sc[2]) == ctl.sector_m


def test_not_slowing_in_a_double_yellow(ctl):
    """A whole sector driven under a double yellow in under 1.4 x the reference car's time for it: 5 s."""
    w = float(ctl.sector_len(0))
    f = racing(ctl, [((s + 0.05) * w, 0.0) for s in (4, 6, 8, 10)])  # each just over a sector line
    c = ctl.init(f)
    ref = np.asarray(ctl.ref_sector_s[0])
    prev = jnp.asarray([3, 5, 7, 9])
    fast = c._replace(sec=prev, dbl_in=jnp.ones(N, bool),
                      sec_t=jnp.asarray([ref[3], 1.3 * ref[5], 1.5 * ref[7], 3.0 * ref[9]], jnp.float32))
    c1 = ctl.direct(f, fast)
    assert list(np.asarray(c1.dbl)) == [1, 1, 0, 0]
    assert list(np.asarray(ctl.penalty_s(c1))) == [5.0, 5.0, 0.0, 0.0]
    assert int(ctl.direct(f, fast._replace(dbl_in=jnp.zeros(N, bool))).dbl.sum()) == 0  # not double as it entered


def test_the_classification_after_the_stewards(ctl):
    """Finishers by race time + penalties; one compound all race: disqualified, last."""
    f = racing(ctl, spread(ctl))
    f = f._replace(finished=jnp.asarray([True, True, True, False]), finish_t=jnp.asarray([100.0, 104.0, 99.0, jnp.inf]))
    c = ctl.init(f)
    two = jax.nn.one_hot(jnp.asarray([0, 1]), len(rc.COMPOUNDS)).sum(0) > 0
    c = c._replace(used=jnp.stack([two, two, c.used[2], two]), passes=jnp.asarray([2, 0, 0, 0], jnp.int32))
    assert list(np.asarray(ctl.penalty_s(c))) == [10.0, 0.0, 0.0, 0.0]
    assert list(np.asarray(ctl.disqualified(f, c))) == [False, False, True, False]
    assert list(np.asarray(ctl.classify(f, c))) == [1, 0, 3, 2]  # 110 s, 104 s, disqualified (99 s), still running


def test_the_black_flag(ctl):
    """Five infringements of any kind (here 2 passes, 2 sector breaks, a closed pit entry): disqualified, finished or
    still running; four: not. Off (`black_flag_n` 0): never."""
    f = racing(ctl, spread(ctl))
    f = f._replace(finished=jnp.asarray([True, True, False, False]), finish_t=jnp.asarray([100.0, 104.0, jnp.inf, jnp.inf]))
    c = ctl.init(f)
    two = jax.nn.one_hot(jnp.asarray([0, 1]), len(rc.COMPOUNDS)).sum(0) > 0
    i32 = lambda x: jnp.asarray(x, jnp.int32)  # noqa: E731
    c = c._replace(used=jnp.stack([two] * N), passes=i32([2, 2, 0, 4]), breaks=i32([2, 2, 5, 0]),
                   closed_in=i32([1, 0, 0, 0]))
    assert list(np.asarray(ctl.black_flagged(c))) == [True, False, True, False]
    assert list(np.asarray(ctl.disqualified(f, c))) == [True, False, True, False]
    ctl.black_flag_n = 0
    try:
        assert not np.asarray(ctl.disqualified(f, c)).any()
    finally:
        ctl.black_flag_n = 5


def test_the_driver_senses_size_and_order():
    """Frozen: `RaceField.observe`'s 115, then race control's 15 in `DRIVER_SENSES`' order (a brain trained on them
    depends on it; the last 3, the stopped car ahead, from step 15n). A VSC (car 0 beside the tarmac in sector 3), car 1 in the sector before it, car 2 just behind the
    Safety Car, car 3 in the pit lane's fast lane; and a widened brain drives as the old one did."""
    from ailearns.agents.ppo_car import forward, init_params
    from ailearns.agents.ppo_field import widen_senses
    cfg = race.RaceConfig(time_limit=400.0, pit_lane=True)
    F = race_field.RaceField(race.make_tracks(range(1), cfg), n_cars=N, laps=2, cfg=cfg, slipstream=False)
    ctl = race_control.RaceControl(F)
    assert F.obs_size == 115 and ctl.obs_size == 130
    assert race_control.DRIVER_SENSES == (
        "vsc", "safety_car", "sc_in_this_lap", "red_flag", "my_sector_yellow", "my_sector_double",
        "next_sector_yellow", "next_sector_double", "next_sector_m", "my_delta", "safety_car_ahead", "box_side",
        "stopped_ahead", "stopped_ahead_m", "stopped_ahead_side")
    designs = jnp.asarray(np.stack([rc.F1_LIKE[0]] * N))
    w = float(ctl.sector_len(0))
    pit = float(F.game.pit_at[0])
    spots = [(3.5 * w, cfg.width / 2 + 1.5), (2.75 * w, 0.0), (8.5 * w, 0.0), (pit, cfg.fast_lat)]
    f = F.grid(jax.random.PRNGKey(0), 0, designs)
    cars = [F.place(jax.random.PRNGKey(k), 0, designs[k], m, side, 30.0) for k, (m, side) in enumerate(spots)]
    f = f._replace(cars=jax.tree.map(lambda *a: jnp.stack(a), *cars), t=jnp.asarray(int(60.0 / cfg.dt)))
    f = stop(f, 0, out=True, out_s=10.0)
    c = ctl.direct(f, ctl.init(f))
    sc = F.place(jax.random.PRNGKey(9), 0, ctl.sc_design, 8.5 * w + 20.0, 0.0, 30.0)
    c = c._replace(sc=jax.tree.map(lambda a: a[None], sc), sc_on=jnp.ones(1, bool),
                   delta=jnp.asarray([0.0, 2.5, 0.0, -10.0]))
    obs = np.asarray(ctl.observe(f, c))
    assert obs.shape == (N, 130)
    np.testing.assert_allclose(obs[:, :115], np.asarray(F.observe(f, ctl.bodies(c))))
    x = {name: obs[:, 115 + i] for i, name in enumerate(race_control.DRIVER_SENSES)}
    assert (x["vsc"] == 1).all() and not (x["safety_car"].any() or x["sc_in_this_lap"].any() or x["red_flag"].any())
    assert x["my_sector_double"][0] == 1 and x["my_sector_yellow"][0] == 0  # out beside the tarmac: double there
    assert x["my_sector_yellow"][1] == 1 and x["next_sector_double"][1] == 1  # the warning in the sector before
    assert x["next_sector_m"][1] == pytest.approx(0.25 * w / ctl.sector_m, abs=0.02)
    assert not (x["my_sector_yellow"][2:].any() or x["my_sector_double"][2:].any())
    assert list(x["my_delta"]) == [0.0, 0.5, 0.0, -1.0]
    assert list(x["safety_car_ahead"]) == [0, 0, 1, 0]
    assert x["box_side"][3] == pytest.approx((cfg.box_lat - cfg.fast_lat) / 5.0, abs=0.05)
    assert (x["box_side"][:3] == 0).all()
    # car 0 is the one stopped: each other car within the sense range behind it sees it, where it is
    L = float(F.tracks.length[0])
    gap = (np.asarray(f.cars.lap_pos)[0] - np.asarray(f.cars.lap_pos)) % L
    sees = (gap > 0) & (gap < F.sense_range)
    assert list(x["stopped_ahead"]) == list(sees.astype(float))
    np.testing.assert_allclose(x["stopped_ahead_m"], np.where(sees, gap / F.sense_range, 0.0), atol=1e-4)
    lat = np.asarray(f.cars.lat)
    np.testing.assert_allclose(x["stopped_ahead_side"], np.where(sees, (lat[0] - lat) / (cfg.width / 2), 0.0), atol=1e-4)
    old = init_params(jax.random.PRNGKey(1), 115, 256, -0.5)
    new = widen_senses(old, 130)
    np.testing.assert_allclose(np.asarray(forward(new, jnp.asarray(obs))[0]),
                               np.asarray(forward(old, jnp.asarray(obs[:, :115]))[0]), atol=1e-5)


def test_the_strategist_senses_size_and_order(ctl):
    """Frozen: the pit wall's 8 race control senses in `STRATEGIST_SENSES`' order. A VSC (car 0 out beside the
    tarmac), then a Safety Car with car 2 first in its queue and car 1 second; car 3 in the pit lane."""
    assert race_control.STRATEGIST_SENSES == (
        "vsc", "safety_car", "red_flag", "laps_since", "two_compounds", "queue_place", "to_pit_entry", "pit_closed")
    F, cfg = ctl.field, ctl.field.cfg
    w, L = float(ctl.sector_len(0)), float(F.tracks.length[0])
    pit = float(F.game.pit_at[0])
    f = stop(racing(ctl, [(3.5 * w, cfg.width / 2 + 1.5), (2.75 * w, 0.0), (8.5 * w, 0.0), (pit, cfg.fast_lat)]),
             0, out=True, out_s=10.0)
    c = ctl.direct(f, ctl.init(f))
    lead = int(ctl.leader(f))
    assert int(c.status) == race_control.VSC and float(c.since) == pytest.approx(float(f.cars.progress[lead]))
    x = {k: v for k, v in zip(race_control.STRATEGIST_SENSES, np.asarray(ctl.strategist_senses(f, c)).T)}
    assert (x["vsc"] == 1).all() and not (x["safety_car"].any() or x["red_flag"].any())
    assert (x["laps_since"] == 0).all() and not x["queue_place"].any() and not x["pit_closed"].any()
    assert not x["two_compounds"].any()
    to_entry = (-cfg.pit_entry - np.asarray(F.game.from_pit(f.cars.track, f.cars.idx))) % L
    np.testing.assert_allclose(x["to_pit_entry"], np.clip(to_entry / 1000.0, 0, 1), atol=1e-5)
    sc = F.place(jax.random.PRNGKey(9), 0, ctl.sc_design, 8.5 * w + 20.0, 0.0, 30.0)
    c = c._replace(status=jnp.asarray(race_control.SC), sc=jax.tree.map(lambda a: a[None], sc),
                   sc_on=jnp.ones(1, bool), since=c.since - 2 * L, pit_closed=jnp.asarray(True),
                   used=c.used.at[1, 0].set(True).at[1, 1].set(True))
    x = {k: v for k, v in zip(race_control.STRATEGIST_SENSES, np.asarray(ctl.strategist_senses(f, c)).T)}
    assert (x["safety_car"] == 1).all() and not x["vsc"].any()
    np.testing.assert_allclose(x["laps_since"], 0.4, atol=1e-5)
    assert x["queue_place"][2] == 0 and x["queue_place"][1] == pytest.approx(1 / 8)
    assert list(x["two_compounds"]) == [0, 1, 0, 0] and (x["pit_closed"] == 1).all()
    assert jax.jit(ctl.strategist_senses)(f, c).shape == (N, 8)


def test_scripts_drive_down_the_lane_not_the_wall():
    """(step 9) the scripts' line in the pit lane: the fast lane's middle; on episode 15's lane (no fast lane: its edge
    at the pit wall) the whole lane's middle, not the wall's own line."""
    new = race.RaceConfig(pit_lane=True)
    assert new.drive_lat == new.fast_lat
    old = race.RaceConfig(pit_lane=True, pit_outer=-14.0, pit_fast=-7.0)
    assert old.drive_lat == -10.5 and old.fast_lat == old.pit_wall_lat


def test_the_leader_and_the_queue_as_the_sorts_had_them(ctl):
    """(ep16 step 9g) `leader` and `queued` without sorting: the same answers as `F.order` and a sorted queue, on random
    fields (ties, cars out, the Safety Car out or not)."""
    F = ctl.field
    f0 = F.grid(jax.random.PRNGKey(0), 0, jnp.asarray(np.stack([rc.F1_LIKE[0]] * N)))
    c0 = ctl.init(f0)
    L = float(F.tracks.length[0])
    for k in range(30):
        ka, kb, kc, kd = jax.random.split(jax.random.PRNGKey(k), 4)
        prog = jnp.round(jax.random.uniform(ka, (N,)) * 400.0 / 10.0) * 10.0  # 10 m steps: ties
        f = f0._replace(cars=f0.cars._replace(progress=prog, lap_pos=prog % L,
                                              crashed=jax.random.uniform(kb, (N,)) < 0.2),
                        finished=jax.random.uniform(kc, (N,)) < 0.1)
        running = ~(f.cars.crashed | f.wrecked | f.finished | F.dry_out(f))
        assert int(ctl.leader(f)) == int(jnp.argmin(jnp.where(running, F.order(f), F.n + 1))), k
        sc = c0.sc._replace(lap_pos=jnp.full((1,), (450.0 + float(jax.random.uniform(kd)) * 50.0) % L))
        c = c0._replace(sc=sc, sc_on=jnp.ones((1,), bool))
        racing = running & ~jax.vmap(F.game.in_pit_lane)(f.cars)
        d = jnp.sort(jnp.where(racing, (sc.lap_pos[0] - f.cars.lap_pos) % L, jnp.inf))
        gaps = jnp.diff(jnp.concatenate([jnp.zeros(1), d]))
        want = jnp.all(jnp.where(jnp.isfinite(d), gaps <= ctl.queue_m, True)) & ~jnp.any(ctl.running(f) & (c.slot >= 0))
        assert bool(ctl.queued(f, c)) == bool(want), k


def test_the_widest_gap_as_the_sorted_sweep_had_it(ctl):
    """(ep16 step 9g) `red_needed`'s widest gap across the tarmac without sorting: on random spreads of stopped cars,
    the same widths wherever there is a gap, and the same blocked / not blocked (`< block_gap`) everywhere."""
    w, half = ctl.field.cfg.width / 2, ctl.body_half

    def sweep(lat, near):
        lo, hi = np.where(near, lat - half, np.inf), np.where(near, lat + half, -np.inf)
        o = np.argsort(lo, kind="stable")
        lo, hi = lo[o], hi[o]
        edge = np.maximum.accumulate(np.concatenate([[-w], hi[:-1]]))
        gaps = np.where(np.isfinite(lo), np.minimum(lo, w) - edge, -np.inf)
        return max(gaps.max(), w - max(hi.max(), -w))

    def fast(lat, near):
        lo, hi = jnp.where(near, lat - half, jnp.inf), jnp.where(near, lat + half, -jnp.inf)
        x = jnp.concatenate([jnp.full(1, -w), jnp.maximum(hi, -w)])
        free = ~jnp.any((lo[None, :] < x[:, None]) & (x[:, None] < hi[None, :]), 1) & (x < w)
        to = jnp.minimum(w, jnp.min(jnp.where(lo[None, :] >= x[:, None], lo[None, :], jnp.inf), 1))
        return float(jnp.max(jnp.where(free, to - x, -jnp.inf)))

    rng = np.random.default_rng(0)
    for k in range(400):
        lat = np.round(rng.uniform(-w - 3, w + 3, N) * 2) / 2  # half-metre steps: touching and repeated cars
        near = rng.uniform(size=N) < rng.uniform(0.1, 1.0)
        a, b = sweep(lat, near), fast(jnp.asarray(lat), jnp.asarray(near))
        assert (a < ctl.block_gap) == (b < ctl.block_gap), (k, a, b)
        if a > 0:
            assert abs(a - b) < 1e-5, (k, a, b)
