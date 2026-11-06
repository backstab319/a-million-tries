"""(ep16 step 9g) The wall lasers as one hand-written GPU kernel (NVIDIA Warp, called from JAX through its FFI).

The same sums as `RaceGame.lasers`, laid out the way a GPU likes them: one thread a car reads each of its window's
wall segments once, from one small table of every circuit's walls (it stays in the GPU's cache), and tests all its
rays against it. Written in JAX, each car's segments were laid out relative to the car (~65 MB a decision) and read
again for every ray: ~21% of a training generation on a T4, and no JAX rewrite helped (playbook 7a). Pallas would
be the JAX way to write a kernel, but it needs an Ampere or newer; Warp runs on a T4.

`wall_lasers(...)` takes cars with any batch shape: however many `vmap`s are around it, it flattens them into one
launch (`custom_vmap`), so the kernel always sees a flat list of cars.
"""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=None)
def _kernel(nseg: int, walls: int, rays: tuple, ray_range: float):
    """The kernel for one window size, wall count, set of ray angles (radians) and range: these are baked in (the
    FFI only takes arrays)."""
    import warp as wp
    from warp import jax_kernel

    R = len(rays)
    vec = wp.types.vector(length=R, dtype=wp.float32)
    RAYS = vec(*[float(r) for r in rays])
    FAR = wp.constant(1e5)  # the pit wall's start is 1e6 away where there's none (RaceGame._wall_windows)
    RANGE = wp.constant(float(ray_range))

    def lasers(table: wp.array3d(dtype=wp.float32), track: wp.array(dtype=wp.int32),
               start: wp.array(dtype=wp.int32), pos: wp.array2d(dtype=wp.float32), yaw: wp.array(dtype=wp.float32),
               out: wp.array2d(dtype=wp.float32)):
        i = wp.tid()
        tr = track[i]
        s0 = start[i]
        px = pos[i, 0]
        py = pos[i, 1]
        y = yaw[i]
        dx, dy, best = vec(), vec(), vec()
        for r in range(R):
            dx[r] = wp.cos(y + RAYS[r])
            dy[r] = wp.sin(y + RAYS[r])
            best[r] = wp.inf
        for wl in range(walls):
            for j in range(nseg):
                ax = wp.float32(table[tr, 2 * wl, s0 + j])  # (values, not references: they change every turn)
                ay = wp.float32(table[tr, 2 * wl + 1, s0 + j])
                bx = wp.float32(table[tr, 2 * wl, s0 + j + 1])
                by = wp.float32(table[tr, 2 * wl + 1, s0 + j + 1])
                if ax < FAR and bx < FAR:  # a real segment (both ends near)
                    ex = bx - ax
                    ey = by - ay
                    wx = ax - px
                    wy = ay - py
                    tn = wx * ey - wy * ex
                    for r in range(R):
                        den = dx[r] * ey - dy[r] * ex
                        if wp.abs(den) >= 1e-9:
                            t = tn / den  # along the ray
                            u = (wx * dy[r] - wy * dx[r]) / den  # along the segment
                            if t > 0.0 and u >= 0.0 and u <= 1.0:
                                best[r] = wp.min(best[r], t)
        for r in range(R):
            out[i, r] = wp.min(best[r], RANGE)

    name = f"wall_lasers_{nseg}_{walls}_{R}_{abs(hash((rays, ray_range)))}"
    return jax_kernel(wp.kernel(lasers, name=name, enable_backward=False, module="unique")), R


def wall_lasers(table, track, start, pos, yaw, *, nseg: int, rays: tuple, ray_range: float):
    """Metres to the nearest wall along each ray (as `RaceGame.lasers`). table: (circuits, walls x 2, points) the
    wall points (x, y rows per wall), padded so every window is contiguous; track, start, yaw: (...), pos: (..., 2).
    -> (..., rays)."""
    walls = table.shape[1] // 2
    call, R = _kernel(nseg, walls, tuple(float(r) for r in rays), float(ray_range))

    @jax.custom_batching.custom_vmap
    def flat(table, track, start, pos, yaw):
        shape = track.shape
        n = int(np.prod(shape)) if shape else 1
        out = call(table, track.reshape(n).astype(jnp.int32), start.reshape(n).astype(jnp.int32),
                   pos.reshape(n, 2).astype(jnp.float32), yaw.reshape(n).astype(jnp.float32),
                   launch_dims=(n,), output_dims=(n, R))
        out = out[0] if isinstance(out, (list, tuple)) else out
        return out.reshape(shape + (R,))

    @flat.def_vmap
    def _(axis_size, in_batched, table, track, start, pos, yaw):
        if in_batched[0]:
            raise ValueError("wall_lasers: the wall table can't be batched")
        args = [x if b else jnp.broadcast_to(x, (axis_size,) + x.shape)
                for x, b in zip((track, start, pos, yaw), in_batched[1:])]
        return flat(table, *args), True  # the batch axis joins the cars: still one launch

    return flat(table, track, start, pos, yaw)


@lru_cache(maxsize=None)
def _body_kernel(nseg: int, rays: tuple, ray_range: float):
    """The car lasers' kernel: each race's body edges (nseg of them) against each of its cars' rays."""
    import warp as wp
    from warp import jax_kernel

    R = len(rays)
    vec = wp.types.vector(length=R, dtype=wp.float32)
    RAYS = vec(*[float(r) for r in rays])
    RANGE = wp.constant(float(ray_range))

    def lasers(a: wp.array3d(dtype=wp.float32), b: wp.array3d(dtype=wp.float32), show: wp.array2d(dtype=wp.int32),
               owner: wp.array2d(dtype=wp.int32), pos: wp.array3d(dtype=wp.float32), yaw: wp.array2d(dtype=wp.float32),
               out: wp.array3d(dtype=wp.float32)):
        tid = wp.tid()
        n = pos.shape[1]
        g = tid // n  # the race
        i = tid - g * n  # the car in it
        px = pos[g, i, 0]
        py = pos[g, i, 1]
        y = yaw[g, i]
        dx, dy, best = vec(), vec(), vec()
        for r in range(R):
            dx[r] = wp.cos(y + RAYS[r])
            dy[r] = wp.sin(y + RAYS[r])
            best[r] = wp.inf
        for j in range(nseg):
            if show[g, j] != 0 and owner[g, j] != i:  # another body's edge, on the circuit
                ax = wp.float32(a[g, j, 0])
                ay = wp.float32(a[g, j, 1])
                ex = wp.float32(b[g, j, 0]) - ax
                ey = wp.float32(b[g, j, 1]) - ay
                wx = ax - px
                wy = ay - py
                tn = wx * ey - wy * ex
                for r in range(R):
                    den = dx[r] * ey - dy[r] * ex
                    if wp.abs(den) >= 1e-9:
                        t = tn / den
                        u = (wx * dy[r] - wy * dx[r]) / den
                        if t > 0.0 and u >= 0.0 and u <= 1.0:
                            best[r] = wp.min(best[r], t)
        for r in range(R):
            out[g, i, r] = wp.min(best[r], RANGE)

    name = f"body_lasers_{nseg}_{R}_{abs(hash((rays, ray_range)))}"
    return jax_kernel(wp.kernel(lasers, name=name, enable_backward=False, module="unique")), R


def body_lasers(a, b, show, owner, pos, yaw, *, rays: tuple, ray_range: float):
    """Metres along each car's rays to the nearest other body's edge (as `RaceField.car_lasers`). Per race: a, b
    (edges, 2) the edges' ends, show (edges,) whether that edge's body is on the circuit, owner (edges,) its body's
    index; pos (cars, 2), yaw (cars,). Any batch shape in front (a vmap over races folds into one launch).
    -> (..., cars, rays)."""
    call, R = _body_kernel(a.shape[-2], tuple(float(r) for r in rays), float(ray_range))

    @jax.custom_batching.custom_vmap
    def flat(a, b, show, owner, pos, yaw):
        lead = pos.shape[:-2]
        g = int(np.prod(lead)) if lead else 1
        S, n = a.shape[-2], pos.shape[-2]
        out = call(a.reshape(g, S, 2).astype(jnp.float32), b.reshape(g, S, 2).astype(jnp.float32),
                   show.reshape(g, S).astype(jnp.int32), owner.reshape(g, S).astype(jnp.int32),
                   pos.reshape(g, n, 2).astype(jnp.float32), yaw.reshape(g, n).astype(jnp.float32),
                   launch_dims=(g * n,), output_dims=(g, n, R))
        out = out[0] if isinstance(out, (list, tuple)) else out
        return out.reshape(lead + (n, R))

    @flat.def_vmap
    def _(axis_size, in_batched, *args):
        args = [x if bt else jnp.broadcast_to(x, (axis_size,) + x.shape) for x, bt in zip(args, in_batched)]
        return flat(*args), True

    return flat(a, b, show, owner, pos, yaw)
