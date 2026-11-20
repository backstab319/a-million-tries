"""The dogfight sim (ep21): pinned to published F-16 numbers, then the gun, the ground, the rewards."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ailearns.games.dogfight import ALIVE, CRASHED, SHOT, Fighters, n_obs
from ailearns.games.rocket_jax import us76

F = Fighters()
STEP = jax.jit(F.step)
G = 9.80665


def sound(alt):
    return float(us76(jnp.float32(alt))[2])


def level(st, i, pos, heading_deg, speed):
    h = np.radians(heading_deg)
    return st._replace(pos=st.pos.at[i].set(jnp.asarray(pos, jnp.float32)),
                       vel=st.vel.at[i].set(jnp.asarray([np.cos(h) * speed, np.sin(h) * speed, 0.0], jnp.float32)),
                       up=st.up.at[i].set(jnp.asarray([0.0, 0.0, 1.0])), load=st.load.at[i].set(1.0))


def apart(st):
    """Plane 1 far away so it doesn't matter."""
    return level(st, 1, [50_000.0, 50_000.0, 8000.0], 0.0, 250.0)


def fly(st, act, seconds):
    act = jnp.asarray(act, jnp.float32)
    for _ in range(int(round(seconds / F.cfg.dt))):
        st, _, _ = STEP(st, act)
    return st


def test_instantaneous_turn_and_corner_speed():
    """Published: max instantaneous ~26 deg/s at sea level, corner speed 330-440 kt."""
    rho = us76(jnp.float32(0.0))[0]
    rates = []
    for v in np.arange(120.0, 320.0, 2.0):
        n = float(F.load_limits(0.5 * rho * v * v)[1])
        rates.append((np.degrees(G * np.sqrt(max(n * n - 1, 0)) / v), v))
    best, v = max(rates)
    assert 24.0 < best < 29.0
    assert 330 < v * 1.944 < 440


def test_sustained_turn_at_mach_0_8():
    """Published (informal sources): ~14-20 deg/s sustained at M0.8, 15,000 ft, full afterburner."""
    alt = 4572.0
    v = 0.8 * sound(alt)
    lo, hi = 1.0, 9.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if float(F.excess_power(alt, v, mid)) > 0 else (lo, mid)
    assert 13.0 < np.degrees(G * np.sqrt(lo * lo - 1) / v) < 19.0


@pytest.mark.parametrize("alt,lo,hi", [(0.0, 1.1, 1.3), (11_000.0, 1.8, 2.2)])
def test_top_speed(alt, lo, hi):
    """Published: about Mach 1.2 at sea level, Mach 2.05 high up."""
    a = sound(alt)
    top = max(m for m in np.arange(0.5, 3.0, 0.01) if float(F.excess_power(alt, m * a, 1.0)) > 0)
    assert lo < top < hi


def test_level_flight_holds_height():
    st = apart(level(F.init(jax.random.key(0)), 0, [0, 0, 5000], 0, 250))
    st = fly(st, [[1 / 9, 0, 0.5, -1], [1 / 9, 0, 0.5, -1]], 5.0)  # 1 g, wings level, part throttle
    assert abs(float(st.pos[0, 2]) - 5000.0) < 30.0
    assert int(st.cause[0]) == ALIVE


def test_simulated_turn_matches_the_formula():
    """A 6 g level-ish turn in the sim turns at g*sqrt(n^2-1)/V."""
    st = apart(level(F.init(jax.random.key(0)), 0, [0, 0, 5000], 0, 250))
    phi = np.arccos(1 / 6)
    st = st._replace(up=st.up.at[0].set(jnp.asarray([0.0, -np.sin(phi), np.cos(phi)], jnp.float32)),
                     load=st.load.at[0].set(6.0))
    a = [[6 / 9, 0, 1, -1], [1 / 9, 0, 0.5, -1]]
    st1 = fly(st, a, 1.0)
    h0, h1 = 0.0, np.degrees(np.arctan2(float(st1.vel[0, 1]), float(st1.vel[0, 0])))
    v = float(jnp.linalg.norm(st1.vel[0]))
    want = np.degrees(G * np.sqrt(35) / v)
    assert abs(abs(h1 - h0) - want) < 0.15 * want


def test_stall_slow_jet_cannot_pull_and_falls():
    st = apart(level(F.init(jax.random.key(0)), 0, [0, 0, 5000], 0, 70))
    st = fly(st, [[1, 0, -1, -1], [0.1, 0, 0.5, -1]], 4.0)  # full back stick, idle (at full afterburner the
    # engine, tilted 25 degrees up, holds it a while: the 1 g stall speed here is ~81 m/s)
    assert float(st.load[0]) < 2.0
    assert float(st.vel[0, 2]) < -5.0


def test_lift_direction_stays_perpendicular():
    st = F.init(jax.random.key(1))
    key = jax.random.key(2)
    for _ in range(600):
        key, k = jax.random.split(key)
        st, _, done = STEP(st, jax.random.uniform(k, (2, 4), minval=-1, maxval=1))
        if bool(done):
            break
    up, vhat = np.asarray(st.up), np.asarray(st.vel) / np.linalg.norm(np.asarray(st.vel), axis=1, keepdims=True)
    assert np.all(np.isfinite(np.asarray(st.pos)))
    assert np.allclose(np.linalg.norm(up, axis=1), 1.0, atol=1e-4)
    assert np.all(np.abs((up * vhat).sum(1)) < 1e-3)


def guns(range_m, off_deg=0.0, seconds=1.0):
    """Plane 0 fires straight ahead at plane 1 flying the same way, range_m ahead (off_deg sideways)."""
    st = F.init(jax.random.key(0))
    o = np.radians(off_deg)
    st = level(st, 0, [0, 0, 5000], 0, 250)
    st = level(st, 1, [0, 0, 5000], 0, 250)
    nose = np.asarray(F.nose(st)[0][0])  # at 1 g the nose rides ~2.6 deg above the flight path: aim with it
    side = np.array([-nose[1], nose[0], 0.0]) / np.linalg.norm(nose[:2])
    st = st._replace(pos=st.pos.at[1].set(jnp.asarray(
        [0, 0, 5000] + range_m * (np.cos(o) * nose + np.sin(o) * side), jnp.float32)))
    tot = np.zeros(2)
    act = jnp.asarray([[1 / 9, 0, 0.5, 1], [1 / 9, 0, 0.5, -1]], jnp.float32)
    for _ in range(int(seconds / F.cfg.dt) + 60):
        st, r, done = STEP(st, act)
        tot += np.asarray(r)
        if bool(done):
            break
    return st, tot


def test_gun_hits_close_dead_ahead():
    st, r = guns(400.0, seconds=1.0)
    assert float(st.health[1]) < 0.5
    assert r[0] > 0.5 and abs(r.sum()) < 1e-5  # zero-sum


def test_gun_misses_off_axis_and_far():
    st, _ = guns(400.0, off_deg=20.0)
    assert float(st.health[1]) == 1.0
    st, _ = guns(3500.0)
    assert float(st.health[1]) > 0.9


def test_a_long_burst_kills_and_ammo_runs_out_in_5_seconds():
    st, r = guns(300.0, seconds=6.0)
    assert int(st.cause[1]) == SHOT
    assert r[0] == pytest.approx(1.0, abs=1e-5)
    st = apart(level(F.init(jax.random.key(0)), 0, [0, 0, 5000], 0, 250))
    st = fly(st, [[1 / 9, 0, 0.5, 1], [1 / 9, 0, 0.5, -1]], 6.0)
    assert float(st.ammo[0]) == 0.0


def test_crash_is_a_loss_and_zero_sum():
    st = apart(level(F.init(jax.random.key(0)), 0, [0, 0, 300], 0, 250))
    st = st._replace(up=st.up.at[0].set(jnp.asarray([0.0, 0.0, -1.0])))  # inverted
    tot = np.zeros(2)
    for _ in range(400):
        st, r, done = STEP(st, jnp.asarray([[1, 0, 1, -1], [1 / 9, 0, 0.5, -1]], jnp.float32))
        tot += np.asarray(r)
        if bool(done):
            break
    assert int(st.cause[0]) == CRASHED
    assert tot[0] == pytest.approx(-1.0) and tot[1] == pytest.approx(1.0)


def test_scripted_target_holds_height_through_a_level_turn():
    st = F.init(jax.random.key(3))
    st = st._replace(man_kind=jnp.int32(1), man_g=jnp.float32(4.0))
    st = st._replace(pos=st.pos.at[0].set(jnp.asarray([90_000.0, 0, 8000.0])))  # attacker far away
    st = st._replace(man_alt=st.pos[1, 2])
    act0 = jnp.asarray([1 / 9, 0, 0.5, -1], jnp.float32)
    step = jax.jit(lambda s: F.step(s, jnp.stack([act0, F.scripted_action(s, 1)])))
    for _ in range(400):
        st, _, _ = step(st)
    assert abs(float(st.pos[1, 2] - st.man_alt)) < 150.0
    assert abs(float(st.load[1]) - 4.0) < 0.6


def test_observations_are_finite_and_sized():
    st = F.init(jax.random.key(4), 0.5)
    for i in (0, 1):
        o = np.asarray(F.observe(st, i))
        assert o.shape == (n_obs(),) and np.all(np.isfinite(o))
