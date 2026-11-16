"""The ecosystem on the GPU (plan section 4): the same world as `ecosystem.py`, in JAX, many worlds at once.

The numpy version stays the reference; everything here is checked against it (`tests/test_ecosystem_jax.py`).

Port step 1, the world: plants, roots, dead grass, fruit, rain and soil wetness. Shapes are fixed so worlds can be
stacked and run together (`jax.vmap` over worlds, `jax.lax.scan` over steps):

- The terrain comes from the numpy `World.make` (`from_numpy`), so a seed gives the same land in both.
- Fruit trees: a world has however many it has; here they are folded into one per-cell map, `fall` (each cell's
  share of all the trees' drop, summed over the trees whose crowns cover it), and a count, `n_trees`. The same
  kilojoules land on the same cells as numpy's `np.add.at`.
- Storms: numpy draws a Poisson count, then that many storms. Here at most `max_storms` a step; storms past the cap
  are counted (`Plants.storms_lost`), never silently dropped. With the defaults the cap is ~10 standard deviations
  above the mean, so the count should stay 0.
- The random draws come in as an argument (`Storms`), so a test can hand both versions the same weather.

The energy books (`Plants.sun_in`, `eaten`, `dieback`, `start`) work as in numpy: `error()` should stay at rounding
level. Dtype follows JAX's setting: float32 by default (the GPU's speed), float64 under `jax.enable_x64` (tests).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .ecosystem import World, WorldConfig


class Land(NamedTuple):
    """What doesn't change: the soil, the plants' limits, where fruit falls. (n, n) arrays unless said."""
    capacity: jax.Array        # kJ of shoots a cell holds
    root_capacity: jax.Array
    root_share: jax.Array      # root_capacity / (capacity + root_capacity), 0 where both are 0
    grazable_floor: jax.Array  # stubble * capacity: shoots below this are out of reach
    fall: jax.Array            # (7a) share of one tree's drop each cell gets, summed over all trees
    n_trees: jax.Array         # () number of fruit trees


class Plants(NamedTuple):
    """What changes, plus the books. (n, n) arrays, () scalars."""
    plants: jax.Array
    roots: jax.Array
    dead: jax.Array
    fruit: jax.Array
    wet: jax.Array
    day: jax.Array
    start: jax.Array
    sun_in: jax.Array
    eaten: jax.Array
    dieback: jax.Array
    storms_lost: jax.Array     # storms past max_storms (should stay 0)


class Storms(NamedTuple):
    """One step's storms: centres in metres, which slots are real, and how many didn't fit."""
    x: jax.Array               # (max_storms,)
    y: jax.Array
    real: jax.Array            # (max_storms,) bool
    lost: jax.Array            # () int


def max_storms(cfg: WorldConfig) -> int:
    """A cap on storms a step: the Poisson mean plus 10 sd, at least 4."""
    L = cfg.size * cfg.cell_m
    lam = cfg.storms_ha_day * L * L / 10_000 * cfg.dt_days
    return int(np.ceil(lam + 10 * np.sqrt(lam))) + 4


def from_numpy(world: World) -> tuple[Land, Plants]:
    """The JAX world for a numpy `World` (normally fresh from `World.make`), in JAX's float dtype.

    Always copies (`jnp.array`): on CPU `jnp.asarray` can share numpy's memory, and numpy's in-place updates
    (`wet *= ...`) then change the JAX world too (it made the equivalence test fail at random)."""
    cfg, n = world.cfg, world.cfg.size
    f = jnp.zeros(0).dtype
    K, RK = world.capacity, world.root_capacity
    root_share = np.divide(RK, K + RK, out=np.zeros_like(K), where=K + RK > 0)
    fall = np.zeros(n * n)
    if len(world.trees):
        cells, share = world.crown()
        np.add.at(fall, cells, share)
    land = Land(*(jnp.array(a, f) for a in (K, RK, root_share, cfg.stubble * K, fall.reshape(n, n))),
                n_trees=jnp.asarray(len(world.trees), f))
    b = world.ledger
    plants = Plants(*(jnp.array(a, f) for a in (world.plants, world.roots, world.dead, world.fruit, world.wet,
                                                    world.day, b.start, b.sun_in, b.eaten, b.dieback)),
                    storms_lost=jnp.asarray(0, jnp.int32))
    return land, plants


def season(cfg, day):
    """0 at midwinter (day 0 of each year), 1 at midsummer."""
    return 0.5 - 0.5 * jnp.cos(2 * jnp.pi * day / cfg.year_days)


def temperature(cfg, day):
    return cfg.temp_winter_c + (cfg.temp_summer_c - cfg.temp_winter_c) * season(cfg, day)


def belt(cfg, day):
    """(9a) Where the rainy front is, (x, y) metres."""
    L = cfg.size * cfg.cell_m
    a = 2 * jnp.pi * cfg.belt_per_year * day / cfg.year_days
    return L / 2 + cfg.belt_loop * L * jnp.cos(a), L / 2 + cfg.belt_loop * L * jnp.sin(a)


def draw_storms(cfg, key, day) -> Storms:
    """(9a) One step's storms, as numpy's `World.weather` draws them (its own stream, so not the same storms)."""
    L = cfg.size * cfg.cell_m
    m = max_storms(cfg)
    kk, kf, kx, ky, ux, uy = jax.random.split(key, 6)
    k = jax.random.poisson(kk, cfg.storms_ha_day * L * L / 10_000 * cfg.dt_days)
    bx, by = belt(cfg, day)
    front = jax.random.uniform(kf, (m,)) < cfg.belt_share
    x = jnp.where(front, bx + cfg.belt_m * jax.random.normal(kx, (m,)), L * jax.random.uniform(ux, (m,)))
    y = jnp.where(front, by + cfg.belt_m * jax.random.normal(ky, (m,)), L * jax.random.uniform(uy, (m,)))
    return Storms(x, y, jnp.arange(m) < k, jnp.maximum(k - m, 0).astype(jnp.int32))


def no_storms(cfg) -> Storms:
    m = max_storms(cfg)
    z = jnp.zeros(m)
    return Storms(z, z, jnp.zeros(m, bool), jnp.asarray(0, jnp.int32))


def weather(cfg, wet, storms: Storms):
    """(9a) The soil dries, the storms wet it. -> (new wet, growth factor 0..1)."""
    c = (jnp.arange(cfg.size) + 0.5) * cfg.cell_m
    wet = wet * np.exp(-cfg.dt_days / cfg.dry_days)
    gx = jnp.exp(-((c[None, :] - storms.x[:, None]) / cfg.storm_m) ** 2 / 2) * storms.real[:, None]
    gy = jnp.exp(-((c[None, :] - storms.y[:, None]) / cfg.storm_m) ** 2 / 2)
    wet = jnp.minimum(wet + cfg.storm_rain * jnp.einsum("ki,kj->ij", gx, gy, precision="highest"), 1.0)
    return wet, jnp.minimum(wet / cfg.wet_full, 1.0)


def fruit_fall(cfg, day):
    """(7a) kJ one tree drops this step: its crop over the season as a sine bump."""
    a, b = cfg.fruit_days
    y = day % cfg.year_days
    on = (y >= a) & (y < b)
    return jnp.where(on, cfg.fruit_kj * np.pi / (2 * (b - a)) * jnp.sin(np.pi * (y - a) / (b - a)) * cfg.dt_days, 0.0)


def _ratio(a, b):
    return jnp.where(b > 0, a / jnp.where(b > 0, b, 1.0), 0.0)


def step(cfg: WorldConfig, land: Land, p: Plants, storms: Storms | None = None) -> Plants:
    """One plant step (`World.step`). With `cfg.rain`, `storms` must be given (`draw_storms`)."""
    K, RK, B, R = land.capacity, land.root_capacity, p.plants, p.roots
    s, dt = season(cfg, p.day), cfg.dt_days
    wet, g, lost = p.wet, 1.0, p.storms_lost
    if cfg.rain:
        wet, g = weather(cfg, p.wet, storms)
        lost = lost + storms.lost
    room, root_room = _ratio(K - B, K), _ratio(RK - R, RK)
    q = jnp.pad(B, 1)
    neighbours = (q[:-2, 1:-1] + q[2:, 1:-1] + q[1:-1, :-2] + q[1:-1, 2:]) / 4
    grow = g * s * dt * (cfg.growth_per_day * B + cfg.seed_per_day * neighbours) * jnp.maximum(room, root_room)
    to_root = jnp.minimum(land.root_share * grow, RK - R)
    to_shoot = jnp.minimum(grow - to_root, K - B)
    to_root = jnp.minimum(grow - to_shoot, RK - R)
    grow = to_shoot + to_root
    up = jnp.minimum(g * s * dt * cfg.regrow_per_day * R * room, K - B - to_shoot)
    die = (1 - s) * dt * cfg.dieback_per_day * B
    if cfg.rain:
        die = jnp.minimum(die + (1 - g) * dt * cfg.drought_per_day * B, B)
    root_die = (2 - s) / 1.5 * dt * cfg.root_loss_per_day * R
    rot = p.dead * (1 - np.exp(-cfg.dead_decay_per_day * dt))
    drop = fruit_fall(cfg, p.day)
    fruit = p.fruit + drop * land.fall
    rate = cfg.fruit_rot_per_day * cfg.rot_q10 ** ((temperature(cfg, p.day) - cfg.temp_summer_c) / 10)
    gone = fruit * (1 - jnp.exp(-rate * dt))
    return p._replace(
        plants=B + to_shoot + up - die, roots=R + to_root - up - root_die, dead=p.dead + die - rot,
        fruit=fruit - gone, wet=wet, day=p.day + dt, storms_lost=lost,
        sun_in=p.sun_in + grow.sum() + drop * land.n_trees,
        dieback=p.dieback + root_die.sum() + rot.sum() + gone.sum())


def _take(pool, reach, cells, want):
    """Take up to `reach` from flat `pool` for eaters asking `want` on flat `cells`, shared pro rata.
    -> (new pool, what each got). Eaters that don't exist ask 0."""
    asked = jnp.zeros_like(pool).at[cells].add(want)
    share = _ratio(jnp.minimum(reach, asked), asked)
    return jnp.maximum(pool - asked * share, 0.0), want * share[cells]


def graze(land: Land, p: Plants, cells, want):
    """`World.graze`: green shoots above the stubble first, then dead grass. -> (plants, got_green, got_dead)."""
    shape = p.plants.shape
    flat = p.plants.reshape(-1)
    green_pool, green = _take(flat, jnp.maximum(flat - land.grazable_floor.reshape(-1), 0.0), cells, want)
    dead = p.dead.reshape(-1)
    dead_pool, got_dead = _take(dead, dead, cells, want - green)
    p = p._replace(plants=green_pool.reshape(shape), dead=dead_pool.reshape(shape),
                   eaten=p.eaten + green.sum() + got_dead.sum())
    return p, green, got_dead


def eat_fruit(p: Plants, cells, want):
    """`World.eat_fruit`. -> (plants, what each got)."""
    flat = p.fruit.reshape(-1)
    pool, got = _take(flat, flat, cells, want)
    return p._replace(fruit=pool.reshape(p.fruit.shape), eaten=p.eaten + got.sum()), got


def total(p: Plants):
    return p.plants.sum() + p.roots.sum() + p.dead.sum() + p.fruit.sum()


def error(p: Plants):
    """Books minus reality, as a share of all energy that entered (`World.error`)."""
    return (p.start + p.sun_in - p.eaten - p.dieback - total(p)) / jnp.maximum(p.start + p.sun_in, 1.0)


def run(cfg: WorldConfig, land: Land, p: Plants, key, steps: int) -> Plants:
    """`steps` plant steps with fresh storms each, in one compiled loop. vmap it over worlds."""
    def body(p, k):
        return step(cfg, land, p, draw_storms(cfg, k, p.day) if cfg.rain else None), None
    return jax.lax.scan(body, p, jax.random.split(key, steps))[0]
