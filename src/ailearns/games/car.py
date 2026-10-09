"""A top-down race car, in JAX: thousands of cars on hundreds of tracks, stepped together.

Tracks are random closed loops (a smooth spline through random points around a
circle) with walls on both sides. Touching a wall ends the run.

The car is an arcade "bicycle": throttle and brake change its speed, the steering
angle sets how tightly it turns, and the tyres have limited grip. Above a certain
speed a corner can't be taken: the car turns as hard as its grip allows and
slides wide into the wall. So it has to learn to brake.

What the AI sees is everything in `observe`: distance "lasers" fanned out from
the nose, its speed, and its last input. No map, nothing about the track ahead.

All functions here are pure (state in, state out), so they work under jax.jit
and jax.vmap; `Tracks` is plain arrays, so a whole bank of tracks rides along.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class CarConfig:
    points: int = 256            # centreline samples per track (equally spaced)
    width: float = 12.0          # track width, metres
    radius: float = 60.0         # typical track radius, metres (a lap is ~300 m)
    dt: float = 0.1              # seconds per decision
    max_speed: float = 40.0      # m/s (144 km/h): where drag cancels full throttle
    accel: float = 8.0           # m/s^2 at full throttle
    brake: float = 16.0          # m/s^2 at full brake
    wheelbase: float = 2.6       # metres; with max_steer: a ~5 m turning circle at walking pace
    max_steer: float = 0.55      # radians
    grip: float = 10.0           # max sideways acceleration, m/s^2 (about 1 g)
    car_half_width: float = 1.0
    ray_angles: tuple = (-90, -60, -40, -25, -12, 0, 12, 25, 40, 60, 90)  # degrees, 0 = straight ahead
    ray_range: float = 60.0      # metres
    time_limit: float = 60.0     # seconds per run: score = how far it gets, so faster is better
    laps: int = 1000             # (no lap limit: a cap would stop rewarding speed once reached)
    progress_reward: float = 0.1  # per metre along the track (backwards: negative)
    crash_penalty: float = 5.0
    messy_starts: bool = False   # training restarts off-centre, angled and moving (episode 5)

    @property
    def obs_size(self) -> int:
        return len(self.ray_angles) + 3  # lasers, speed, last steer, last throttle


# ── Tracks (numpy: generated once, then handed to JAX as arrays) ─────────────

def _resample(loop: np.ndarray, n: int):
    """A closed polyline -> n points equally spaced along it, its length and the heading at each point."""
    closed = np.vstack([loop, loop[:1]])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    s = np.linspace(0, cum[-1], n, endpoint=False)
    c = np.stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])], 1)
    tang = np.roll(c, -1, 0) - np.roll(c, 1, 0)
    return c, cum[-1], np.arctan2(tang[:, 1], tang[:, 0])


def make_track(seed: int, cfg: CarConfig = CarConfig()) -> dict:
    """A random closed loop, counter-clockwise. Retries until the track is drivable:
    no corner tighter than the track is wide, and no two parts of it close enough
    to overlap."""
    rng = np.random.default_rng(seed)
    while True:
        k = int(rng.integers(10, 19))
        ang = (np.arange(k) + rng.uniform(-0.4, 0.4, k)) * 2 * np.pi / k
        r = cfg.radius * rng.uniform(0.3, 1.0, k)  # deep dents make hairpins and chicanes
        ctrl = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        t = np.linspace(0, 1, 32, endpoint=False)[:, None]
        dense = np.concatenate([  # closed Catmull-Rom spline through the control points
            0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                   + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
            for p0, p1, p2, p3 in (ctrl[[i - 1, i, (i + 1) % k, (i + 2) % k]] for i in range(k))])
        for _ in range(3):  # soften kinks at the control points
            dense = (np.roll(dense, -2, 0) + np.roll(dense, -1, 0) + dense + np.roll(dense, 1, 0)
                     + np.roll(dense, 2, 0)) / 5
        c, length, heading = _resample(dense, cfg.points)
        turn = (np.roll(heading, -1) - heading + np.pi) % (2 * np.pi) - np.pi
        tightest = 1 / np.abs(turn / (length / cfg.points)).max()  # smallest corner radius, metres
        grow = 0.8 * cfg.width / tightest
        if grow > 2.5:
            continue  # it would have to be huge to make that corner drivable
        if grow > 1:  # scale the whole track up until the tightest corner is drivable
            c, length, heading = _resample(c * grow * 1.02, cfg.points)
        d = np.linalg.norm(c[:, None] - c[None], axis=2)
        i = np.arange(cfg.points)
        along = np.abs(i[:, None] - i[None]) * length / cfg.points
        along = np.minimum(along, length - along)
        if (d[along > 3 * cfg.width] < 1.6 * cfg.width).any():
            continue  # two stretches of track would touch
        normal = np.stack([-np.sin(heading), np.cos(heading)], 1)  # points left (inside, counter-clockwise)
        half = cfg.width / 2
        return {"center": c, "heading": heading, "left": c + normal * half, "right": c - normal * half,
                "length": float(length), "seed": seed}


class Tracks(NamedTuple):
    center: jax.Array    # (T, P, 2)
    heading: jax.Array   # (T, P)  direction of travel at each centreline point
    wall_a: jax.Array    # (T, 2P, 2) wall segments a -> b (left wall, then right wall)
    wall_b: jax.Array    # (T, 2P, 2)
    length: jax.Array    # (T,)


TRAIN_SEEDS = range(0, 512)          # tracks it learns on
TEST_SEEDS = range(100_000, 100_064)  # tracks it has never seen (final numbers come from these)


def make_tracks(seeds, cfg: CarConfig = CarConfig()) -> Tracks:
    ts = [make_track(s, cfg) for s in seeds]
    walls = lambda t: (np.concatenate([t["left"], t["right"]]),
                       np.concatenate([np.roll(t["left"], -1, 0), np.roll(t["right"], -1, 0)]))
    a, b = zip(*(walls(t) for t in ts))
    return Tracks(jnp.asarray(np.stack([t["center"] for t in ts]), jnp.float32),
                  jnp.asarray(np.stack([t["heading"] for t in ts]), jnp.float32),
                  jnp.asarray(np.stack(a), jnp.float32), jnp.asarray(np.stack(b), jnp.float32),
                  jnp.asarray([t["length"] for t in ts], jnp.float32))


# ── The car (JAX) ────────────────────────────────────────────────────────────

class State(NamedTuple):
    track: jax.Array      # which track
    pos: jax.Array        # (2,) metres
    heading: jax.Array    # radians
    speed: jax.Array      # m/s, never negative (no reverse gear)
    idx: jax.Array        # nearest centreline point
    lap_pos: jax.Array    # metres along the lap, in [0, length)
    progress: jax.Array   # metres driven along the track since the start (laps add up)
    t: jax.Array          # decisions taken
    last: jax.Array       # (2,) last steer, throttle
    crashed: jax.Array
    done: jax.Array


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class CarGame:
    def __init__(self, tracks: Tracks, cfg: CarConfig = CarConfig()):
        self.tracks, self.cfg = tracks, cfg
        self.rays = jnp.deg2rad(jnp.asarray(cfg.ray_angles, jnp.float32))

    def init(self, key, track=None, random_start: bool = True, start=None) -> State:
        """A car standing still on the centreline, facing along the track (at point `start`,
        else a random point, else the start line)."""
        kt, ks = jax.random.split(key)
        P = self.cfg.points
        tr = jax.random.randint(kt, (), 0, self.tracks.length.shape[0]) if track is None else jnp.asarray(track)
        i = (jnp.asarray(start) if start is not None
             else jax.random.randint(ks, (), 0, P) if random_start else jnp.asarray(0))
        z = jnp.zeros((), jnp.float32)
        pos, heading, speed = self.tracks.center[tr, i], self.tracks.heading[tr, i], z
        if self.cfg.messy_starts and random_start and start is None:
            ko, kh, kv = jax.random.split(jax.random.fold_in(ks, 1), 3)
            side = jnp.stack([-jnp.sin(heading), jnp.cos(heading)])
            pos = pos + side * jax.random.uniform(ko, (), minval=-0.3, maxval=0.3) * self.cfg.width
            heading = heading + jax.random.uniform(kh, (), minval=-0.35, maxval=0.35)
            speed = jax.random.uniform(kv, (), maxval=0.6 * self.cfg.max_speed)
        return State(tr, pos, heading, speed, i,
                     i * self.tracks.length[tr] / P, z, jnp.asarray(0), jnp.zeros(2), jnp.asarray(False),
                     jnp.asarray(False))

    def _locate(self, st: State, pos):
        """Nearest centreline point (searched near the last one), distance from the
        centreline, and position along the lap."""
        P, c = self.cfg.points, self.tracks.center[st.track]
        cand = (st.idx + jnp.arange(-6, 15)) % P
        j = cand[jnp.argmin(jnp.sum((c[cand] - pos) ** 2, axis=1))]

        def on_segment(i0):  # project onto centreline segment i0 -> i0+1
            a, b = c[i0 % P], c[(i0 + 1) % P]
            ab = b - a
            u = jnp.clip(jnp.dot(pos - a, ab) / jnp.dot(ab, ab), 0, 1)
            return jnp.linalg.norm(a + u * ab - pos), (i0 % P) + u

        (d0, f0), (d1, f1) = on_segment(j - 1), on_segment(j)
        lateral = jnp.minimum(d0, d1)
        frac = jnp.where(d0 < d1, f0, f1)
        ds = self.tracks.length[st.track] / P
        return j, lateral, (frac * ds) % self.tracks.length[st.track]

    def step(self, st: State, action) -> tuple[State, jax.Array]:
        cfg = self.cfg
        steer, throttle = jnp.clip(action[0], -1, 1), jnp.clip(action[1], -1, 1)
        drag = cfg.accel / cfg.max_speed ** 2
        acc = jnp.where(throttle >= 0, throttle * cfg.accel, throttle * cfg.brake) - drag * st.speed ** 2
        v = jnp.clip(st.speed + acc * cfg.dt, 0.0, cfg.max_speed * 1.05)
        want = jnp.tan(steer * cfg.max_steer) / cfg.wheelbase       # curvature the wheels ask for
        can = cfg.grip / jnp.maximum(v * v, 1e-3)                    # curvature the tyres can hold
        heading = st.heading + v * jnp.clip(want, -can, can) * cfg.dt
        pos = st.pos + v * cfg.dt * jnp.stack([jnp.cos(heading), jnp.sin(heading)])
        idx, lateral, lap_pos = self._locate(st, pos)
        L = self.tracks.length[st.track]
        moved = (lap_pos - st.lap_pos + L / 2) % L - L / 2           # metres along the track this step
        crashed = lateral > cfg.width / 2 - cfg.car_half_width
        progress = st.progress + moved
        t = st.t + 1
        done = crashed | (t * cfg.dt >= cfg.time_limit) | (progress >= cfg.laps * L)
        reward = moved * cfg.progress_reward - crashed * cfg.crash_penalty
        new = State(st.track, pos, heading, v, idx, lap_pos, progress, t, jnp.stack([steer, throttle]),
                    crashed, done)
        return new, reward

    def lasers(self, st: State) -> jax.Array:
        """Distance (metres) to the nearest wall along each laser, capped at ray_range."""
        ang = st.heading + self.rays
        d = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)[:, None, :]          # (R, 1, 2)
        a, b = self.tracks.wall_a[st.track], self.tracks.wall_b[st.track]   # (S, 2)
        e, w = b - a, a - st.pos
        den = _cross(d, e)                                                   # (R, S)
        safe = jnp.where(jnp.abs(den) < 1e-9, 1e-9, den)
        t = _cross(w, e) / safe
        u = _cross(w, d) / safe
        hit = (jnp.abs(den) >= 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
        return jnp.minimum(jnp.min(jnp.where(hit, t, jnp.inf), axis=1), self.cfg.ray_range)

    def observe(self, st: State) -> jax.Array:
        return jnp.concatenate([self.lasers(st) / self.cfg.ray_range,
                                jnp.stack([st.speed / self.cfg.max_speed]), st.last])

    def reset_if(self, st: State, key) -> State:
        """Finished runs start again on a random track (for training)."""
        fresh = self.init(key)
        return jax.tree.map(lambda a, b: jnp.where(st.done, a, b), fresh, st)

    def laps(self, st: State) -> jax.Array:
        return jnp.floor(st.progress / self.tracks.length[st.track])


def centreline_limit_profile(seed: int, cfg: CarConfig = CarConfig()) -> tuple[np.ndarray, float]:
    """Speed (m/s) at every centreline point for the fastest flying lap on the middle of the road."""
    t = make_track(seed, cfg)
    P, ds = cfg.points, t["length"] / cfg.points
    h = t["heading"]
    curv = np.abs(((np.roll(h, -1) - h + np.pi) % (2 * np.pi) - np.pi) / ds)
    v = np.minimum(np.sqrt(cfg.grip / np.maximum(curv, 1e-6)), cfg.max_speed)
    drag = cfg.accel / cfg.max_speed ** 2
    for _ in range(3):  # around the loop a few times so the start line doesn't matter
        for i in range(P):  # flat out wherever possible
            j = (i + 1) % P
            v[j] = min(v[j], np.sqrt(max(v[i] ** 2 + 2 * (cfg.accel - drag * v[i] ** 2) * ds, 0)))
        for i in range(P - 1, -1, -1):  # brake in time for every bend
            j = (i - 1) % P
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * (cfg.brake + drag * v[i] ** 2) * ds))
    return v, ds


def centreline_drive(seed: int, seconds: float, cfg: CarConfig = CarConfig()) -> dict:
    """The perfect middle-of-the-road driver as a drive: standing start on the line, then the
    limit profile lap after lap. Positions/headings every cfg.dt, like a recorded AI drive."""
    t = make_track(seed, cfg)
    vlim, ds = centreline_limit_profile(seed, cfg)
    P = cfg.points
    drag = cfg.accel / cfg.max_speed ** 2
    n_laps = 1 + int(seconds * cfg.max_speed / t["length"])
    v = np.tile(vlim, n_laps + 1)
    v[0] = 0.0
    for i in range(len(v) - 1):  # the standing start: accelerate from zero, never above the limit
        v[i + 1] = min(v[i + 1], np.sqrt(max(v[i] ** 2 + 2 * (cfg.accel - drag * v[i] ** 2) * ds, 0)))
    times = np.concatenate([[0], np.cumsum(2 * ds / (v[:-1] + v[1:]))])
    s = np.arange(len(v)) * ds
    tq = np.arange(0, seconds + cfg.dt, cfg.dt)
    sq = np.interp(tq, times, s)
    k = (sq / ds).astype(int) % P
    frac = (sq / ds) % 1
    c, h = t["center"], t["heading"]
    pos = c[k] * (1 - frac[:, None]) + c[(k + 1) % P] * frac[:, None]
    return {"pos": pos, "heading": h[k], "progress": sq, "length": t["length"]}


def centreline_limit_lap(seed: int, cfg: CarConfig = CarConfig()) -> float:
    """The fastest possible flying lap for a driver who stays exactly on the middle of the
    road: at every point as fast as grip allows for that bend, braking and accelerating at
    the car's limits in between. (A racing line can beat it by straightening the bends.)"""
    v, ds = centreline_limit_profile(seed, cfg)
    return float(np.sum(ds / v))
