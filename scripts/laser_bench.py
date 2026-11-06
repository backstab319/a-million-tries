"""Race day 2 (episode 16), step 9g: the wall lasers, several ways, timed on the GPU against each other. The profile
(`race_speed.py --profile`) had their closest-hit kernel at 23.5% of a training step: each car's window of 330 wall
segments, written out relative to the car, then read again for each of its 13 rays. Each version here gets a
training step's cars (3072 races x 8), 200 times over (moved a little each time so nothing is reused), and is checked
against `RaceGame.lasers`.

    uv run --group jax python scripts/laser_bench.py [--races 3072] [--steps 200]
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import es_field
from ailearns.games import race

ap = argparse.ArgumentParser()
ap.add_argument("--races", type=int, default=3072)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--tracks", type=int, default=64)
ap.add_argument("--chunks", default="256,512,1024,2048,4096", help="comma-separated: the current lasers over this many cars at a time")
ap.add_argument("--warps", default="4", help="comma-separated: each kernel at each warp count")
ap.add_argument("--only", default="", help="comma-separated name parts: only these versions (and now)")
a = ap.parse_args()

cfg = es_field.FieldESConfig(mode="finish", n_cars=8, n_learn=4, laps=25, gp_km=305.0, tyre_cal=1.9,
                             call_worst=True, race_control=True)
school = es_field.make_school(cfg, list(race.TRAIN_SEEDS)[:a.tracks])
g = school.field.game
race.LASER_KERNEL = "jax"  # "now" is the JAX lasers ("auto" would pick the Warp kernel on a GPU)
keys = jax.random.split(jax.random.PRNGKey(0), a.races)
cars = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), jax.jit(jax.vmap(school.start))(keys).field.cars)
lo, hi = g.cfg.wall_window
RAYS = np.deg2rad(np.asarray(g.cfg.ray_angles, np.float32))
C, S = jnp.asarray(np.cos(RAYS)), jnp.asarray(np.sin(RAYS))  # each ray's direction in the car's frame: constants


def window(st):
    start = (st.idx + lo) % g.cfg.points
    return jax.lax.dynamic_slice(g._win, (st.track, start, 0, 0), (1, hi - lo + 1) + g._win.shape[2:])[0]


def hits(wx, wy, ex, ey, real, dx, dy):
    """Distance along the ray (dx, dy) to each segment from w to w + e, inf where it misses."""
    den = dx * ey - dy * ex
    safe = jnp.where(jnp.abs(den) < 1e-9, 1e-9, den)
    t = (wx * ey - wy * ex) / safe
    u = (wx * dy - wy * dx) / safe
    hit = real & (jnp.abs(den) >= 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    return jnp.where(hit, t, jnp.inf)


def car_frame(st):
    """The window's points relative to the car, turned into its frame (x ahead, y left): (points, walls, 2)."""
    win = window(st)
    p = win - st.car.pos
    c, s = jnp.cos(st.car.yaw), jnp.sin(st.car.yaw)
    return jnp.stack([p[..., 0] * c + p[..., 1] * s, -p[..., 0] * s + p[..., 1] * c], -1), win


def frame_rows(st):  # in the car's frame, rays down and segments across (as now)
    pc, win = car_frame(st)
    real = ((win[:-1, ..., 0] < 1e5) & (win[1:, ..., 0] < 1e5)).reshape(-1)
    w, e = pc[:-1].reshape(-1, 2), (pc[1:] - pc[:-1]).reshape(-1, 2)
    t = hits(w[:, 0], w[:, 1], e[:, 0], e[:, 1], real, C[:, None], S[:, None])
    return jnp.minimum(t.min(1), g.cfg.ray_range)


def per_ray(st):  # world frame, one min per ray (13 reductions over the same segments: can they share the reads?)
    win = window(st)
    a_, b_ = win[:-1].reshape(-1, 2), win[1:].reshape(-1, 2)
    real = (a_[:, 0] < 1e5) & (b_[:, 0] < 1e5)
    w, e = a_ - st.car.pos, b_ - a_
    ang = st.car.yaw + g.rays
    out = [hits(w[:, 0], w[:, 1], e[:, 0], e[:, 1], real, jnp.cos(ang[k]), jnp.sin(ang[k])).min()
           for k in range(len(RAYS))]
    return jnp.minimum(jnp.stack(out), g.cfg.ray_range)


def frame_per_ray(st):  # the car's frame and one min per ray
    pc, win = car_frame(st)
    real = ((win[:-1, ..., 0] < 1e5) & (win[1:, ..., 0] < 1e5)).reshape(-1)
    w, e = pc[:-1].reshape(-1, 2), (pc[1:] - pc[:-1]).reshape(-1, 2)
    out = [hits(w[:, 0], w[:, 1], e[:, 0], e[:, 1], real, C[k], S[k]).min() for k in range(len(RAYS))]
    return jnp.minimum(jnp.stack(out), g.cfg.ray_range)


def scan_rays(st):  # world frame, the rays one after another in a loop (each pass: one ray over the segments)
    win = window(st)
    a_, b_ = win[:-1].reshape(-1, 2), win[1:].reshape(-1, 2)
    real = (a_[:, 0] < 1e5) & (b_[:, 0] < 1e5)
    w, e = a_ - st.car.pos, b_ - a_
    ang = st.car.yaw + g.rays
    one = lambda th: hits(w[:, 0], w[:, 1], e[:, 0], e[:, 1], real, jnp.cos(th), jnp.sin(th)).min()  # noqa: E731
    return jnp.minimum(jax.lax.map(one, ang), g.cfg.ray_range)


# hand-written kernels (Pallas): one car a program, its 16 ray slots (13 + 3 that never hit) x 128 segments a tile, so
# each segment is read once for all its rays
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import triton as plt  # noqa: E402  (Triton, not Mosaic GPU: that needs a Hopper)

GPU = jax.devices()[0].platform == "gpu"
NSEG = hi - lo  # segments per wall in the window (110)
P = g.cfg.points
base = g._win[:, :P]  # (T, P, walls, 2)
TABLE = jnp.concatenate([base, base[:, :130]], 1)  # wraps past the line, room for 128 + 1 lanes from any start
TABLE = TABLE.transpose(0, 2, 3, 1).reshape(TABLE.shape[0], -1, TABLE.shape[1])  # (T, walls x 2, points)
WALLS = base.shape[2]


def dirs(cars):
    ang = cars.car.yaw[:, None] + g.rays
    d = jnp.stack([jnp.cos(ang), jnp.sin(ang)], 1)  # (N, 2, 13)
    return jnp.pad(d, ((0, 0), (0, 0), (0, 16 - d.shape[2])))  # the 3 extra rays: no direction, never hit


def tile_min(best, wx, wy, ex, ey, real, dx, dy):
    """The closest hit so far for each ray slot, after this tile of 128 segments."""
    den = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]
    safe = jnp.where(jnp.abs(den) < 1e-9, 1e-9, den)
    t = (wx[None, :] * ey[None, :] - wy[None, :] * ex[None, :]) / safe
    u = (wx[None, :] * dy[:, None] - wy[None, :] * dx[:, None]) / safe
    hit = real[None, :] & (jnp.abs(den) >= 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
    return jnp.minimum(best, jnp.min(jnp.where(hit, t, jnp.inf), axis=1))


def kern_table(track_ref, start_ref, pos_ref, dir_ref, table_ref, out_ref):  # A: straight from the wall table
    tr, s0 = track_ref[0], start_ref[0]
    px, py = pos_ref[0, 0], pos_ref[0, 1]
    dx, dy = dir_ref[0, 0, :], dir_ref[0, 1, :]
    lane = jax.lax.broadcasted_iota(jnp.int32, (128,), 0)
    best = jnp.full((16,), jnp.inf, jnp.float32)
    for wall in range(WALLS):
        ax, ay = table_ref[tr, 2 * wall, pl.ds(s0, 128)], table_ref[tr, 2 * wall + 1, pl.ds(s0, 128)]
        bx, by = table_ref[tr, 2 * wall, pl.ds(s0 + 1, 128)], table_ref[tr, 2 * wall + 1, pl.ds(s0 + 1, 128)]
        real = (lane < NSEG) & (ax < 1e5) & (bx < 1e5)
        best = tile_min(best, ax - px, ay - py, bx - ax, by - ay, real, dx, dy)
    out_ref[0, :] = jnp.minimum(best, g.cfg.ray_range)


def by_table(cars, WARPS=4):
    n = cars.idx.shape[0]
    start = ((cars.idx + lo) % P).astype(jnp.int32)
    out = pl.pallas_call(
        kern_table, grid=(n,), interpret=not GPU, compiler_params=plt.CompilerParams(num_warps=WARPS),
        in_specs=[pl.BlockSpec((1,), lambda i: (i,)), pl.BlockSpec((1,), lambda i: (i,)),
                  pl.BlockSpec((1, 2), lambda i: (i, 0)), pl.BlockSpec((1, 2, 16), lambda i: (i, 0, 0)),
                  pl.BlockSpec(TABLE.shape, lambda i: (0, 0, 0))],
        out_specs=pl.BlockSpec((1, 16), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((n, 16), jnp.float32),
    )(cars.track.astype(jnp.int32), start, cars.car.pos, dirs(cars), TABLE)
    return out[:, :13]


def kern_rows(seg_ref, dir_ref, out_ref):  # B: each car's segments laid out by XLA, read once here
    dx, dy = dir_ref[0, 0, :], dir_ref[0, 1, :]
    best = jnp.full((16,), jnp.inf, jnp.float32)
    for wall in range(WALLS):
        ex, ey = seg_ref[0, 8 + wall, :], seg_ref[0, 12 + wall, :]
        best = tile_min(best, seg_ref[0, wall, :], seg_ref[0, 4 + wall, :], ex, ey,
                        (ex != 0) | (ey != 0), dx, dy)
    out_ref[0, :] = jnp.minimum(best, g.cfg.ray_range)


def by_rows(cars, WARPS=4):
    def lay(st):  # (16, 128): wx, wy, ex, ey for each wall (4 rows each, the 4th empty), e = 0 where no segment
        win = window(st)
        a_, b_ = win[:-1], win[1:]  # (110, walls, 2)
        real = (a_[..., 0] < 1e5) & (b_[..., 0] < 1e5)
        w, e = a_ - st.car.pos, jnp.where(real[..., None], b_ - a_, 0.0)
        rows = jnp.concatenate([w, e], -1).transpose(2, 1, 0)  # (wx wy ex ey, walls, 110)
        rows = jnp.pad(rows, ((0, 0), (0, 4 - WALLS), (0, 128 - NSEG)))
        return rows.reshape(16, 128)
    segs = jax.vmap(lay)(cars)
    n = segs.shape[0]
    out = pl.pallas_call(
        kern_rows, grid=(n,), interpret=not GPU, compiler_params=plt.CompilerParams(num_warps=WARPS),
        in_specs=[pl.BlockSpec((1, 16, 128), lambda i: (i, 0, 0)), pl.BlockSpec((1, 2, 16), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((1, 16), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((n, 16), jnp.float32),
    )(segs, dirs(cars))
    return out[:, :13]


per_car = {"now (RaceGame.lasers)": g.lasers, "car frame": frame_rows, "per ray": per_ray,
           "car frame + per ray": frame_per_ray, "rays in a loop": scan_rays}
versions = {n: (lambda c, f=f: jax.vmap(f)(c)) for n, f in per_car.items()}
def chunked(cars, n):
    """The current lasers over n cars at a time (one after another): each chunk's segments (~2.6 KB a car) might
    stay in the T4's 4 MB cache for the 13 re-reads."""
    pieces = jax.tree.map(lambda x: x.reshape((-1, n) + x.shape[1:]), cars)
    return jax.lax.map(jax.vmap(g.lasers), pieces).reshape(-1, 13)


def warp_kernel(cars):  # (race_warp) one thread a car, each segment read once from the wall table
    race.LASER_KERNEL = "warp"
    try:
        return jax.vmap(g.lasers)(cars)
    finally:
        race.LASER_KERNEL = "jax"


versions["warp kernel"] = warp_kernel
for n in map(int, a.chunks.split(",")) if a.chunks else ():
    versions[f"now, {n} cars a chunk"] = lambda c, n=n: chunked(c, n)
for w in map(int, a.warps.split(",")):
    versions |= {f"kernel A: table, {w} warps": lambda c, w=w: by_table(c, w),
                 f"kernel B: rows, {w} warps": lambda c, w=w: by_rows(c, w)}
if a.only:
    versions = {n: f for n, f in versions.items() if n.startswith("now") or any(k in n for k in a.only.split(","))}
print(f"{jax.devices()[0].device_kind}: {len(cars.idx)} cars x {a.steps}", flush=True)
ref = np.asarray(jax.jit(jax.vmap(g.lasers))(cars))
for name, fn in versions.items():
    def run(cars, fn=fn):
        def tick(acc, i):  # moved a little each time (along x, 1 mm): the same work, nothing reused
            c = cars._replace(car=cars.car._replace(pos=cars.car.pos + i * 1e-3))
            return acc + fn(c).sum(), None
        return jax.lax.scan(tick, 0.0, jnp.arange(a.steps, dtype=jnp.float32))[0]
    try:
        f = jax.jit(run)
        jax.block_until_ready(f(cars))
        t0 = time.time()
        jax.block_until_ready(f(cars))
        t = time.time() - t0
        diff = np.abs(np.asarray(jax.jit(fn)(cars)) - ref)
    except Exception as err:  # (a kernel that doesn't build on this GPU: say why, go on)
        print(f"  {name:24s} failed: {type(err).__name__}: {str(err)[:600]}", flush=True)
        continue
    print(f"  {name:24s} {t:6.3f} s   differs from now: max {diff.max():.4f} m, over 1 mm {(diff > 1e-3).mean():.3%}",
          flush=True)
