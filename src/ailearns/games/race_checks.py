"""Reality checks for the racing car (episode 14): scripted test drives on flat ground, no AI.

Each test is what a magazine road test measures: 0-100 km/h, top speed, braking 100-0 km/h, steady cornering g.
The scripted driver is a perfect test driver: every 0.05 s it tries several pedal positions and keeps the best one
(that's launch control and threshold braking without ABS hardware). Used by tests/test_race_physics.py and the
devlog's table; the numbers it prints are the ones the episode compares with real cars.
"""

import jax
import jax.numpy as jnp

from ailearns.games import race_car as rc

DT = 0.05


def _full(p):
    """A road test measures the car, not its fuel: a tank that never runs dry (a 150 s top-speed run would empty one
    sized for a 90 s lap)."""
    return p._replace(fuel_tank=jnp.inf)


def _step_all(p, s, actions):
    """Try several actions from the same state: (N, 2) -> N next states."""
    return jax.vmap(lambda a: rc.drive(_full(p), s, a, dt=DT)[0])(actions)


_try = jax.jit(_step_all)
_drive = jax.jit(lambda p, s, a: rc.drive(_full(p), s, a, dt=DT))


def zero_to(p, kmh: float = 100.0, limit_s: float = 20.0) -> float:
    """Seconds from standing to `kmh`, with the best throttle every step (no wheelspin wasted)."""
    s = rc.car_at([0, 0], 0.0)
    pedals = jnp.stack([jnp.zeros(9), jnp.linspace(0.2, 1.0, 9)], 1)
    target, t = kmh / 3.6, 0.0
    while t < limit_s:
        nxt = _try(p, s, pedals)
        i = int(jnp.argmax(nxt.vx))
        prev, s = float(s.vx), jax.tree.map(lambda x: x[i], nxt)
        if float(s.vx) >= target:
            return t + DT * (target - prev) / (float(s.vx) - prev)
        t += DT
    return float("inf")


def top_speed(p, seconds: float = 150.0) -> float:
    """km/h after flat out for `seconds` (launching gently enough not to spin up the tyres)."""
    s = rc.car_at([0, 0], 0.0, speed=20.0)
    flat_out = jnp.array([0.0, 1.0])
    run = jax.jit(lambda s: jax.lax.fori_loop(0, int(seconds / DT), lambda _, st: _drive(p, st, flat_out)[0], s))
    return float(run(s).vx) * 3.6


def braking(p, from_kmh: float = 100.0) -> float:
    """Metres to stop from `from_kmh`, braking as hard as the tyres allow every step (no ABS, no lock-ups wasted)."""
    s = rc.car_at([0, 0], 0.0, speed=from_kmh / 3.6)
    pedals = jnp.stack([jnp.zeros(17), -jnp.linspace(0.2, 1.0, 17)], 1)
    for _ in range(int(20 / DT)):
        nxt = _try(p, s, pedals)
        i = int(jnp.argmin(nxt.vx))
        s = jax.tree.map(lambda x: x[i], nxt)
        if float(s.vx) <= 0.05:
            break
    return float(s.pos[0])


PEDALS = jnp.array([-0.3, -0.1, 0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])


def cornering_g(p, kmh: float, steers: int = 48, seconds: float = 8.0) -> float:
    """Highest steady sideways g at a constant `kmh` (the skidpad test): every steering angle from gentle to full
    lock is held while a speed controller works the pedal; the best one that stays steady (no spin, speed held)
    is the car's cornering limit at that speed. The tyres are held at their best temperature, like quoted g figures
    (warm tyres, peak grip): eight seconds at the limit would otherwise cook the outside pair (tested separately)."""
    v = kmh / 3.6
    s0 = rc.car_at([0, 0], 0.0, speed=v)
    n = int(seconds / DT)

    def hold(steer):
        def body(i, carry):
            s, acc = carry
            # the pedal (of a few) that best holds the speed: a heavy foot spins the rear tyres out of the corner
            outs = jax.vmap(lambda pd: rc.drive(_full(p), s, jnp.stack([steer, pd]), dt=DT))(PEDALS)
            k = jnp.argmin(jnp.abs(outs[0].vx - v))
            s, info = jax.tree.map(lambda x: x[k], outs)
            s = s._replace(temp=jnp.full(4, p.temp_best))
            last = i >= n - 20  # average over the final second
            beta = jnp.abs(s.vy / jnp.maximum(s.vx, 1.0))
            acc = acc + jnp.where(last, jnp.stack([info.ay, jnp.abs(s.vx - v), beta]), 0.0)
            return s, acc
        s, acc = jax.lax.fori_loop(0, n, body, (s0, jnp.zeros(3)))
        ay, verr, beta = acc / 20
        steady = (verr < 0.03 * v) & (beta < 0.15) & (s.vx > 0.9 * v)
        return jnp.where(steady, jnp.abs(ay), 0.0)

    steer = jnp.linspace(0.01, 1.0, steers) ** 2  # fine steps near straight: fast corners need little lock
    return float(jax.jit(jax.vmap(hold))(steer).max()) / rc.G


def downforce(p, kmh: float) -> float:
    """Newtons of downforce at a steady `kmh` in a straight line (the floor settles to its running height)."""
    s = rc.car_at([0, 0], 0.0, speed=kmh / 3.6, ride_height=p.ride_height)
    hold = jax.jit(lambda s: rc.drive(_full(p), s, jnp.array([0.0, 0.0]), dt=DT))
    for _ in range(10):
        s, info = hold(s._replace(vx=jnp.asarray(kmh / 3.6, jnp.float32)))
    return float(info.downforce)


def road_test(d, spec) -> dict:
    """The full road test of one design, as a dict of rounded numbers."""
    p = rc.derive(d, spec)
    return {"mass_kg": round(float(p.mass)), "front %": round(100 * float(rc.weight_front(jnp.asarray(d), spec)), 1),
            "yaw kg m2": round(float(p.yaw_inertia)), "downforce @250 kN": round(downforce(p, 250) / 1000, 1),
            "0-100 s": round(zero_to(p), 2), "0-160 s": round(zero_to(p, 160), 2),
            "0-200 s": round(zero_to(p, 200, limit_s=30), 2), "top km/h": round(top_speed(p)),
            "100-0 m": round(braking(p), 1),
            **{f"g @ {k}": round(cornering_g(p, k), 2) for k in (100, 200, 250)}}


def test_driver(game, st, grip_g=1.4, brake_g=1.5, offset=0.0):
    """A cautious scripted test driver on a circuit (no AI): follows the centreline (or a line `offset` metres to its
    left, e.g. the pit lane) and slows for the corners ahead. `game` is a race.RaceGame, `st` its State; returns the
    action. For tests and the debug view."""
    car = st.car
    p = rc.derive(st.design, game.spec)
    # pure pursuit: steer for a point on the centreline further ahead the faster it goes
    look = jnp.maximum(12.0, 0.6 * car.vx)
    P, L = game.cfg.points, game.tracks.length[st.track]
    k = ((st.lap_pos + look) / L * P).astype(jnp.int32) % P
    h = game.tracks.heading[st.track, k]
    to = game.tracks.center[st.track, k] + offset * jnp.stack([-jnp.sin(h), jnp.cos(h)]) - car.pos
    alpha = (jnp.arctan2(to[1], to[0]) - car.yaw + jnp.pi) % (2 * jnp.pi) - jnp.pi
    steer = jnp.arctan(2 * (p.a + p.b) * jnp.sin(alpha) / jnp.linalg.norm(to)) / game.spec.max_steer
    curv_here = game.tracks.curvature[st.track, st.idx]
    curv = jnp.concatenate([curv_here[None], game.road_ahead(st)])
    dist = jnp.concatenate([jnp.zeros(1), game.ahead])
    corner_v = jnp.sqrt(grip_g * rc.G / jnp.maximum(jnp.abs(curv), 1e-4))
    target = jnp.min(jnp.sqrt(corner_v ** 2 + 2 * brake_g * rc.G * dist))
    # throttle only with grip to spare: easy at low speed (wheelspin) and while cornering (power oversteer)
    spare = jnp.clip(1 - jnp.abs(curv_here) * car.vx ** 2 / (grip_g * rc.G), 0.0, 1.0)
    # never more than half brake: full pedal locks all four (no ABS) and a locked front tyre can't steer
    pedal = jnp.clip(0.5 * (target - car.vx), -0.5, jnp.minimum(0.25 + car.vx / 60, 1.0) * spare)
    return jnp.stack([jnp.clip(steer, -1, 1), pedal])


if __name__ == "__main__":
    for name, (d, spec) in {"911 Carrera (ROAD)": rc.PORSCHE_911, "F1-like (RACE)": rc.F1_LIKE}.items():
        print(name, road_test(d, spec))
