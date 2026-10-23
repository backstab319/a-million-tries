"""The racing sandbox (episode 14): procedural circuits, the four-tyre car of `race_car`, many cars at once in JAX.

A circuit is a random closed loop at real scale (laps of ~2-4 km): 12 m of tarmac with a kerb at each edge, a strip
of grass (slow, low grip) and then a wall. Circuits have hills (a smooth, looping elevation profile: climbs,
drops, crests, compressions) and corners banked towards their centre. Touching the wall ends the run. Progress is measured along the
centreline, so the score of a run is how far the car gets in the time limit: faster is better.

Every car carries its own design (`race_car.DESIGN`), so one batch can hold thousands of different cars; the
driver is told the design it's driving (`observe`).

What the driver sees (`observe`): distance lasers fanned from the nose (to the walls); **the road ahead**: the
track's curvature, gradient and banking at fixed distances up to 320 m ahead (what a driver who knows the circuit
has in their head; lasers can't see round corners, and at 300 km/h braking takes ~150 m); where it is on the road (offset from the
centreline, angle to it); its own motion (speed, sideways speed, yaw rate, felt accelerations); what each tyre is
on and whether it's sliding; each tyre's temperature and wear; the fuel left; its last inputs; and the car's
design numbers.

Pure functions (state in, state out): jax.jit / jax.vmap work. Nothing here draws anything.
"""

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.games import race_car as rc
from ailearns.games.car import _resample


@dataclass(frozen=True)
class RaceConfig:
    points: int = 1024           # centreline samples per circuit (~2-4 m apart)
    radius: float = 450.0        # typical circuit radius, metres
    width: float = 12.0          # tarmac, kerb to kerb
    kerb: float = 1.0            # kerb width at each edge (inside the 12 m)
    grass: float = 10.0          # run-off beyond each kerb, then the wall
    min_corner: float = 15.0     # tightest corner radius allowed (a hairpin)
    dt: float = 0.05             # seconds per decision
    substeps: int = 10           # physics steps per decision
    time_limit: float = 90.0     # seconds per run
    ray_angles: tuple = (-90, -60, -40, -25, -15, -7, 0, 7, 15, 25, 40, 60, 90)  # degrees, 0 = ahead
    ray_range: float = 200.0     # metres
    wall_window: tuple = (-10, 100)  # wall segments the lasers check, in centreline points around the car
    ahead: tuple = (10, 20, 35, 50, 75, 100, 140, 190, 250, 320)  # metres: where the driver reads the curvature
    progress_reward: float = 0.01  # per metre along the track (backwards: negative)
    crash_penalty: float = 3.0
    start_speed: float = 0.0     # training starts at a random speed up to this (m/s), on a random stretch
    design_center: tuple = ()    # random designs: empty = anywhere in the parts box; else near this design ...
    design_spread: float = 0.15  # ... within this share of each dial's range either side (specialising a driver)
    # surfaces: (grip multiplier, extra rolling resistance, bumpiness, lift in metres)
    tarmac_surface: tuple = (1.0, 0.0, 0.03, 0.0)
    kerb_surface: tuple = (0.95, 0.005, 0.15, 0.03)
    grass_surface: tuple = (0.35, 0.06, 0.30, 0.0)  # dry grass under a slick ~0.3-0.4 of tarmac, bumpier than a kerb
    dirt_grip: float = 0.10      # share of grip a tyre loses with grass and dirt on it ...
    dirt_clear: float = 3.0      # ... wearing off over this many seconds back on tarmac
    tyre_start: float = 70.0     # tyre temperature (C) at the start: out of the blankets (F1 blankets: 70 C)
    hill_range: tuple = (3.0, 40.0)  # metres from a circuit's lowest point to its highest (Monza ~10, Spa ~100)
    max_grade: float = 0.12      # steepest climb or drop (12%: Eau Rouge-steep; most circuits much less)
    bank_deg: tuple = (3.0, 10.0)  # banking in tight corners: most circuits (80%) up to the first, the rest the second

    @property
    def obs_size(self) -> int:
        return len(self.ray_angles) + 3 * len(self.ahead) + 9 + 4 + 4 + 4 + 4 + 1 + len(rc.DESIGN)

    @property
    def edge(self) -> float:
        """Distance from the centreline to the wall."""
        return self.width / 2 + self.grass


# ── Circuits (numpy: generated once, then handed to JAX as arrays) ───────────

def make_track(seed: int, cfg: RaceConfig = RaceConfig()) -> dict:
    """A random closed circuit, counter-clockwise. Retries until it's drivable: no corner tighter than
    `min_corner`, and no two stretches close enough for their run-off areas to touch."""
    rng = np.random.default_rng(seed)
    while True:
        k = int(rng.integers(10, 19))
        ang = (np.arange(k) + rng.uniform(-0.4, 0.4, k)) * 2 * np.pi / k
        r = cfg.radius * rng.uniform(0.3, 1.0, k)  # deep dents make hairpins and chicanes
        ctrl = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        t = np.linspace(0, 1, 64, endpoint=False)[:, None]
        dense = np.concatenate([  # closed Catmull-Rom spline through the control points
            0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                   + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
            for p0, p1, p2, p3 in (ctrl[[i - 1, i, (i + 1) % k, (i + 2) % k]] for i in range(k))])
        for _ in range(3):  # soften kinks at the control points
            dense = (np.roll(dense, -2, 0) + np.roll(dense, -1, 0) + dense + np.roll(dense, 1, 0)
                     + np.roll(dense, 2, 0)) / 5
        c, length, heading = _resample(dense, cfg.points)
        ds = length / cfg.points
        curv = ((np.roll(heading, -1) - np.roll(heading, 1) + np.pi) % (2 * np.pi) - np.pi) / (2 * ds)
        tightest = 1 / np.abs(curv).max()
        if tightest < cfg.min_corner:
            grow = cfg.min_corner / tightest
            if grow > 2.0:
                continue
            c, length, heading = _resample(c * grow * 1.02, cfg.points)
            ds = length / cfg.points
            curv = ((np.roll(heading, -1) - np.roll(heading, 1) + np.pi) % (2 * np.pi) - np.pi) / (2 * ds)
        d = np.linalg.norm(c[:, None] - c[None], axis=2)
        i = np.arange(cfg.points)
        along = np.abs(i[:, None] - i[None]) * ds
        along = np.minimum(along, length - along)
        if (d[along > 4 * cfg.edge] < 2.4 * cfg.edge).any():
            continue  # two stretches' run-off would touch
        normal = np.stack([-np.sin(heading), np.cos(heading)], 1)  # points left (inside, counter-clockwise)
        return {"center": c, "heading": heading, "curvature": curv, "normal": normal,
                "wall_left": c + normal * cfg.edge, "wall_right": c - normal * cfg.edge,
                "length": float(length), "seed": seed, **_hills(seed, curv, length, cfg)}


def _hills(seed: int, curv, length: float, cfg: RaceConfig) -> dict:
    """Elevation and banking for a circuit, from their own random stream (a seed's layout never changes)."""
    rng = np.random.default_rng([seed, 7])
    P = cfg.points
    ds = length / P
    pos = np.arange(P) * ds
    lo, hi = max(1, int(length / 2500)), max(2, int(length / 400))  # whole waves round the lap: 400-2500 m long
    n = rng.integers(lo, hi + 1, 4)
    z = sum(rng.uniform(0.3, 1.0) / k * np.sin(2 * np.pi * k * pos / length + rng.uniform(0, 2 * np.pi)) for k in n)
    z = (z - z.min()) / max(np.ptp(z), 1e-6) * rng.uniform(*cfg.hill_range)
    grade = (np.roll(z, -1) - np.roll(z, 1)) / (2 * ds)
    z = z * min(1.0, cfg.max_grade / np.abs(grade).max())
    grade = (np.roll(z, -1) - np.roll(z, 1)) / (2 * ds)
    vcurv = (np.roll(z, -1) - 2 * z + np.roll(z, 1)) / ds ** 2
    most = rng.uniform(0, cfg.bank_deg[0] if rng.random() < 0.8 else cfg.bank_deg[1])
    bank = np.radians(most) * np.clip(curv * 50.0, -1, 1)  # full banking in corners tighter than 50 m radius
    kernel = np.ones(15) / 15                               # eased in and out along the circuit
    bank = np.convolve(np.concatenate([bank[-7:], bank, bank[:7]]), kernel, "valid")
    return {"z": z, "grade": grade, "side": -np.tan(bank), "vcurv": vcurv}


class Tracks(NamedTuple):
    center: jax.Array     # (T, P, 2)
    heading: jax.Array    # (T, P)
    curvature: jax.Array  # (T, P) 1/m, left turns positive
    wall_a: jax.Array     # (T, 2P, 2) wall segments a -> b (left wall, then right wall)
    wall_b: jax.Array     # (T, 2P, 2)
    length: jax.Array     # (T,)
    z: jax.Array          # (T, P) height of the centreline, metres
    grade: jax.Array      # (T, P) rise per metre along the circuit
    side: jax.Array       # (T, P) rise per metre across it, to the left (banking: down towards a corner's centre)
    vcurv: jax.Array      # (T, P) 1/m, vertical curvature: > 0 in a dip (compression), < 0 over a crest


TRAIN_SEEDS = range(0, 512)
TEST_SEEDS = range(100_000, 100_064)  # circuits it never trains on: every number we say comes from these


def make_tracks(seeds, cfg: RaceConfig = RaceConfig()) -> Tracks:
    ts = [make_track(s, cfg) for s in seeds]

    def walls(t):
        a = np.concatenate([t["wall_left"], t["wall_right"]])
        b = np.concatenate([np.roll(t["wall_left"], -1, 0), np.roll(t["wall_right"], -1, 0)])
        return a, b

    a, b = zip(*(walls(t) for t in ts))
    f = lambda x: jnp.asarray(np.stack(x), jnp.float32)  # noqa: E731
    return Tracks(f([t["center"] for t in ts]), f([t["heading"] for t in ts]), f([t["curvature"] for t in ts]),
                  f(a), f(b), jnp.asarray([t["length"] for t in ts], jnp.float32),
                  *(f([t[k] for t in ts]) for k in ("z", "grade", "side", "vcurv")))


# ── Slipstream and dirty air (for races between cars) ───────────────────────

@dataclass(frozen=True)
class WakeConfig:
    """A car leaves a wake behind it: slower air (a tow: less drag for the car behind) that's also turbulent (dirty
    air: less downforce, the front wing worst hit). First estimates, to be checked against published figures before
    they're claimed on screen: ~30% less drag right behind, fading over ~25 m; ~40% less front downforce."""
    reach: float = 80.0          # metres behind a car where its wake still counts
    fade: float = 25.0           # strength falls by e every this many metres
    width: float = 1.2           # half-width of the wake right behind the car, metres ...
    spread: float = 0.03         # ... growing this much per metre back
    drag_cut: float = 0.30       # at full strength
    front_down_cut: float = 0.40
    rear_down_cut: float = 0.20


def wake(pos, yaw, cfg: WakeConfig = WakeConfig()) -> rc.Air:
    """The air each car is in, from the cars around it: pos (N, 2), yaw (N,) of the cars in one race -> rc.Air of
    (N,) multipliers. The strongest wake a car is in counts (wakes don't stack)."""
    fwd = jnp.stack([jnp.cos(yaw), jnp.sin(yaw)], -1)                  # (N, 2) each car's heading
    rel = pos[:, None, :] - pos[None, :, :]                            # (i, j): car i relative to car j
    behind = -jnp.sum(rel * fwd[None, :, :], -1)                       # metres car i is behind car j
    side = jnp.abs(rel[..., 0] * fwd[None, :, 1] - rel[..., 1] * fwd[None, :, 0])
    half = cfg.width + cfg.spread * behind
    inside = (behind > 0) & (behind < cfg.reach) & ~jnp.eye(pos.shape[0], dtype=bool)
    strength = jnp.where(inside, jnp.exp(-behind / cfg.fade) * jnp.exp(-(side / half) ** 2), 0.0).max(axis=1)
    return rc.Air(1 - cfg.drag_cut * strength, 1 - cfg.front_down_cut * strength, 1 - cfg.rear_down_cut * strength)


# ── The game ─────────────────────────────────────────────────────────────────

class State(NamedTuple):
    car: rc.CarState
    design: jax.Array     # (len(rc.DESIGN),) this car's design numbers
    track: jax.Array
    idx: jax.Array        # nearest centreline point
    lat: jax.Array        # signed offset from the centreline, metres (left positive)
    lap_pos: jax.Array    # metres along the lap
    progress: jax.Array   # metres along the track since the start (laps add up)
    t: jax.Array          # decisions taken
    last: jax.Array       # (2,) last steer, pedal
    surface: jax.Array    # (4,) grip under each tyre
    dirt: jax.Array       # (4,) grass and dirt on each tyre: 1 fresh off the grass, wears off to 0
    sliding: jax.Array    # (4,) tyre sliding
    plank: jax.Array      # (2,) newtons on the skid plank, front/rear (bottoming out)
    crashed: jax.Array
    done: jax.Array


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class RaceGame:
    def __init__(self, tracks: Tracks, cfg: RaceConfig = RaceConfig(), spec: rc.CarSpec = rc.RACE):
        self.tracks, self.cfg = tracks, cfg
        self.spec = replace(spec, race_s=cfg.time_limit) if spec.race_s > 0 else spec  # fuel for the run's length
        self.rays = jnp.deg2rad(jnp.asarray(cfg.ray_angles, jnp.float32))
        self.ahead = jnp.asarray(cfg.ahead, jnp.float32)

    # where the car is on the circuit
    def _locate(self, track, idx, pos):
        """Nearest centreline point (searched near the last one), signed offset, metres along the lap."""
        P, c = self.cfg.points, self.tracks.center[track]
        cand = (idx + jnp.arange(-6, 12)) % P
        j = cand[jnp.argmin(jnp.sum((c[cand] - pos) ** 2, axis=1))]

        def on_segment(i0):
            a, b = c[i0 % P], c[(i0 + 1) % P]
            ab = b - a
            u = jnp.clip(jnp.dot(pos - a, ab) / jnp.dot(ab, ab), 0, 1)
            near = a + u * ab
            side = jnp.sign(_cross(ab, pos - a))
            return jnp.linalg.norm(near - pos), side * jnp.linalg.norm(near - pos), (i0 % P) + u

        (d0, s0, f0), (d1, s1, f1) = on_segment(j - 1), on_segment(j)
        lat = jnp.where(d0 < d1, s0, s1)
        frac = jnp.where(d0 < d1, f0, f1)
        L = self.tracks.length[track]
        return j, lat, (frac * L / P) % L

    def _surfaces(self, st_car: rc.CarState, p: rc.CarParams, track, idx, lat):
        """What's under each tyre (rc.Surface), and how far out the furthest tyre is."""
        cfg = self.cfg
        normal_ang = self.tracks.heading[track, idx] + jnp.pi / 2
        n = jnp.stack([jnp.cos(normal_ang), jnp.sin(normal_ang)])
        wx = jnp.stack([p.a, p.a, -p.b, -p.b])
        wy = jnp.stack([p.track_width / 2, -p.track_width / 2, p.track_width / 2, -p.track_width / 2])
        cy, sy = jnp.cos(st_car.yaw), jnp.sin(st_car.yaw)
        offs = jnp.stack([wx * cy - wy * sy, wx * sy + wy * cy], 1)  # (4, 2) world offsets of the tyres
        wlat = jnp.abs(lat + offs @ n)
        half = cfg.width / 2
        on_kerb = (wlat > half - cfg.kerb) & (wlat <= half)
        on_grass = wlat > half
        pick = [jnp.where(on_grass, g, jnp.where(on_kerb, k, t))
                for t, k, g in zip(cfg.tarmac_surface, cfg.kerb_surface, cfg.grass_surface)]
        return rc.Surface(*pick), wlat.max()

    def init(self, key, design=None, track=None, start=None, random_start: bool = True) -> State:
        """A car on the centreline, facing along the circuit (at point `start`, else a random point, else the start
        line), at a random speed up to cfg.start_speed. `design`: rc.DESIGN numbers, else a random design."""
        kt, ks, kd, kv = jax.random.split(key, 4)
        P = self.cfg.points
        tr = jax.random.randint(kt, (), 0, self.tracks.length.shape[0]) if track is None else jnp.asarray(track)
        i = (jnp.asarray(start) if start is not None
             else jax.random.randint(ks, (), 0, P) if random_start else jnp.asarray(0))
        if design is None:
            lo, span = jnp.asarray(rc.DESIGN_LOW), jnp.asarray(rc.DESIGN_HIGH - rc.DESIGN_LOW)
            u = jax.random.uniform(kd, (len(rc.DESIGN),))
            if self.cfg.design_center:  # near a given design (a specialised driver's school)
                c = (jnp.asarray(self.cfg.design_center, jnp.float32) - lo) / span
                u = jnp.clip(c + self.cfg.design_spread * (2 * u - 1), 0.0, 1.0)
            design = lo + u * span
        speed = jax.random.uniform(kv, ()) * self.cfg.start_speed
        car = rc.car_at(self.tracks.center[tr, i], self.tracks.heading[tr, i], speed,
                        ride_height=design[rc.D["ride_height_mm"]] / 1000.0, tyre_temp=self.cfg.tyre_start)
        z = jnp.zeros((), jnp.float32)
        L = self.tracks.length[tr]
        return State(car, jnp.asarray(design, jnp.float32), tr, jnp.asarray(i), z, i * L / P, z, jnp.asarray(0),
                     jnp.zeros(2), jnp.ones(4), jnp.zeros(4), jnp.zeros(4, bool), jnp.zeros(2), jnp.asarray(False),
                     jnp.asarray(False))

    def step(self, st: State, action) -> tuple[State, jax.Array]:
        cfg = self.cfg
        action = jnp.clip(jnp.asarray(action, jnp.float32), -1, 1)
        p = rc.derive(st.design, self.spec)
        surf, _ = self._surfaces(st.car, p, st.track, st.idx, st.lat)
        surf = surf._replace(grip=surf.grip * (1 - cfg.dirt_grip * st.dirt))  # dirty tyres grip less
        car, info = rc.drive(p, st.car, action, surf, dt=cfg.dt, substeps=cfg.substeps, slope=self.slope(st))
        idx, lat, lap_pos = self._locate(st.track, st.idx, car.pos)
        surf_now, furthest = self._surfaces(car, p, st.track, idx, lat)
        L = self.tracks.length[st.track]
        moved = (lap_pos - st.lap_pos + L / 2) % L - L / 2
        crashed = furthest > cfg.edge - 0.3  # a tyre at the wall: the body's in it
        t = st.t + 1
        done = crashed | (t * cfg.dt >= cfg.time_limit)
        reward = moved * cfg.progress_reward - crashed * cfg.crash_penalty
        on_grass = surf_now.grip <= cfg.grass_surface[0]
        dirt = jnp.where(on_grass, 1.0, jnp.maximum(st.dirt - cfg.dt / cfg.dirt_clear, 0.0))
        new = State(car, st.design, st.track, idx, lat, lap_pos, st.progress + moved, t, action, surf_now.grip, dirt,
                    info.sliding, info.plank, crashed, done)
        return new, reward

    def slope(self, st: State) -> rc.Slope:
        """The road under the car: gravity along its surface (in the car's frame) and the load factor."""
        tr, i = st.track, st.idx
        h, yaw = self.tracks.heading[tr, i], st.car.yaw
        grade, side = self.tracks.grade[tr, i], self.tracks.side[tr, i]
        # uphill direction in the world: along the circuit by the grade, across it by the banking
        up = grade * jnp.stack([jnp.cos(h), jnp.sin(h)]) + side * jnp.stack([-jnp.sin(h), jnp.cos(h)])
        tilt = jnp.sqrt(1 + grade ** 2 + side ** 2)
        g = -rc.G * up / tilt
        v2 = st.car.vx ** 2 + st.car.vy ** 2
        load = (1 / tilt + v2 * self.tracks.vcurv[tr, i] / rc.G                        # crests lighten, dips press
                - v2 * self.tracks.curvature[tr, i] * side / jnp.sqrt(1 + side ** 2) / rc.G)  # a banked turn presses
        return rc.Slope(g[0] * jnp.cos(yaw) + g[1] * jnp.sin(yaw), -g[0] * jnp.sin(yaw) + g[1] * jnp.cos(yaw),
                        jnp.maximum(load, 0.0))

    def lasers(self, st: State) -> jax.Array:
        """Metres to the nearest wall along each laser, capped at ray_range (walls near the car only)."""
        P = self.cfg.points
        lo, hi = self.cfg.wall_window
        near = (st.idx + jnp.arange(lo, hi)) % P
        seg = jnp.concatenate([near, near + P])
        a, b = self.tracks.wall_a[st.track, seg], self.tracks.wall_b[st.track, seg]
        ang = st.car.yaw + self.rays
        d = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)[:, None, :]
        e, w = b - a, a - st.car.pos
        den = _cross(d, e)
        safe = jnp.where(jnp.abs(den) < 1e-9, 1e-9, den)
        t = _cross(w, e) / safe
        u = _cross(w, d) / safe
        hit = (jnp.abs(den) >= 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
        return jnp.minimum(jnp.min(jnp.where(hit, t, jnp.inf), axis=1), self.cfg.ray_range)

    def road_ahead(self, st: State) -> jax.Array:
        """Curvature (1/m, left positive) of the circuit at cfg.ahead metres ahead of the car."""
        P = self.cfg.points
        L = self.tracks.length[st.track]
        k = ((st.lap_pos + self.ahead) / L * P).astype(jnp.int32) % P
        return self.tracks.curvature[st.track, k]

    def hills_ahead(self, st: State) -> jax.Array:
        """Gradient and banking (rise per metre, along and across) at cfg.ahead metres ahead: (2 * len(ahead),)."""
        P = self.cfg.points
        L = self.tracks.length[st.track]
        k = ((st.lap_pos + self.ahead) / L * P).astype(jnp.int32) % P
        return jnp.concatenate([self.tracks.grade[st.track, k], self.tracks.side[st.track, k]])

    def observe(self, st: State) -> jax.Array:
        cfg, car = self.cfg, st.car
        heading_err = (car.yaw - self.tracks.heading[st.track, st.idx] + jnp.pi) % (2 * jnp.pi) - jnp.pi
        motion = jnp.stack([car.vx / 100.0, car.vy / 10.0, car.r / 2.0, car.ax / 50.0, car.ay / 50.0,
                            st.lat / (cfg.width / 2), heading_err, car.steer / self.spec.max_steer,
                            st.last[1]])
        return jnp.concatenate([
            self.lasers(st) / cfg.ray_range,
            jnp.clip(self.road_ahead(st) * 30.0, -2.0, 2.0),
            self.hills_ahead(st) * 10.0,
            motion,
            st.surface * (1 - cfg.dirt_grip * st.dirt),  # the grip each tyre has (dirt included)
            st.sliding.astype(jnp.float32),
            (car.temp - self.spec.temp_best) / self.spec.temp_window,
            car.wear,
            (1 - car.fuel_used / rc.fuel_start_kg(st.design, self.spec))[None],  # share of the fuel left
            rc.normalise(st.design),
        ])

    def reset_if(self, st: State, key) -> State:
        """Finished runs start again: a random circuit, stretch and design (for training)."""
        fresh = self.init(key)
        return jax.tree.map(lambda a, b: jnp.where(st.done, a, b), fresh, st)

    def laps(self, st: State) -> jax.Array:
        return jnp.floor(st.progress / self.tracks.length[st.track])
