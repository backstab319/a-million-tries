"""The ecosystem's creatures on the GPU (plan section 4): `ecosystem_life.py` in JAX, many worlds at once.

The numpy `Life` stays the reference; `tests/test_ecosystem_jax_life.py` checks this against it.

Port step 2, bodies and brains: senses, the brain, moving, paying, eating (fruit, carcass, plants), milk, carried
young, thirst, growth, ageing, the voice, memory, deaths, carcasses rotting, the plants' step every `plant_every`
ticks.

Port step 4, fights and births: chance comes in as `Draws` (one row per slot), so a test can hand in numpy's. The
mating loop becomes `match` (the same pairs, in parallel); children fill free slots in numpy's order.

Port step 5, probes: `Probe` sums deaths by cause, food by source, drinking, thirst, walking, calls and the green
probe as the world runs (`run(..., probe=no_probe())`); `scramble` gives the control (random brains, same moment).

Port step 3, who sees whom: `nearest_grid` (sight-wide buckets, the 3 x 3 around each creature, a fixed number per
bucket with overflow counted in `books.crowded`); `nearest_brute` (all pairs) is its exact reference. Shapes are fixed: every world has `cap` creature slots, `alive` marks the used ones (numpy's arrays grow; here
births past the cap are counted in `books.full`). Dead slots hold harmless values: every update is masked by `alive`.
A creature's mother is found by slot (`mom`: the slot she had when it was born, still hers while the uid there is
its parent's), not by searching uids.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import ecosystem_jax as ej
from .ecosystem import WATER, WorldConfig
from .ecosystem_life import DAY_REAL_S, N_HID, N_IN, N_OUT, N_W, SECTORS_DEG, Life, LifeConfig, keep_warm_s, plant_keep


class Ground(NamedTuple):
    """What the creatures need of the land that never changes (flat over cells)."""
    water: jax.Array           # (n*n,) bool
    shore: jax.Array           # (n*n,) bool: land next to water (drinkable)
    air_of_cell: jax.Array     # (n*n,) int: each cell's centre's cell on the air grid (fruit scent)


class Creatures(NamedTuple):
    alive: jax.Array
    pos: jax.Array             # (cap, 2) metres
    heading: jax.Array
    mass: jax.Array
    adult: jax.Array
    birth: jax.Array
    fat: jax.Array
    diet: jax.Array
    alone: jax.Array
    reserve: jax.Array
    age: jax.Array
    dry: jax.Array
    gen: jax.Array             # int32
    uid: jax.Array             # int32, -1 for never used
    parent: jax.Array          # int32 uid
    parent2: jax.Array
    mom: jax.Array             # int32 slot the parent had at birth (-1: none)
    w: jax.Array               # (cap, N_W)
    calling: jax.Array         # bool
    mem: jax.Array             # (cap, 2)


class Books(NamedTuple):
    start: jax.Array
    heat: jax.Array
    dung: jax.Array
    decay: jax.Array
    kills: jax.Array           # int32
    births: jax.Array
    births_alone: jax.Array
    deaths: jax.Array
    full: jax.Array
    capped: jax.Array          # (port 4) births past MAX_BIRTHS in a tick (should stay 0)
    crowded: jax.Array         # (port 3) creature-ticks a sight bucket was too full to list them (should stay ~0)


class World(NamedTuple):
    """One world's whole changing state (vmap over a stack of these)."""
    c: Creatures
    meat: jax.Array            # (n, n) kJ of carcass
    plants: ej.Plants
    books: Books
    next_uid: jax.Array        # int32
    ticks: jax.Array           # int32


class Static(NamedTuple):
    land: ej.Land
    ground: Ground


def air_n(wcfg: WorldConfig, cfg: LifeConfig) -> int:
    return max(int(round(wcfg.size * wcfg.cell_m / cfg.sound_cell_m)), 1)


def plant_every(wcfg: WorldConfig, cfg: LifeConfig) -> int:
    return round(wcfg.dt_days * cfg.day_s / cfg.dt)


def from_numpy(life: Life, cap: int | None = None) -> tuple[Static, World]:
    """The JAX world for a numpy `Life` (and its `World`), with `cap` slots (default: numpy's). Copies everything."""
    w, cfg = life.world, life.cfg
    wcfg, n = w.cfg, w.cfg.size
    m = len(life.alive)
    cap = cap or m
    assert not life.alive[cap:].any(), "the living don't fit in cap slots"
    m = min(m, cap)
    land, plants = ej.from_numpy(w)
    water = (w.biome == WATER)
    p = np.pad(water, 1)
    shore = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~water
    na = air_n(wcfg, cfg)
    rc = np.stack(np.divmod(np.arange(n * n), n), 1)
    ac = np.clip((((rc + 0.5) * wcfg.cell_m) / cfg.sound_cell_m).astype(int), 0, na - 1)
    ground = Ground(jnp.array(water.reshape(-1)), jnp.array(shore.reshape(-1)), jnp.array(ac[:, 0] * na + ac[:, 1]))
    f = jnp.zeros(0).dtype

    def pad(a, fill=0):
        out = np.full((cap,) + a.shape[1:], fill, a.dtype)
        out[:m] = a[:m]
        return out

    # mothers by slot: the living creature whose uid is the parent's
    slot_of = {int(u): k for k, u in enumerate(life.uid) if life.alive[k]}
    mom = np.array([slot_of.get(int(q), -1) if q >= 0 else -1 for q in life.parent])
    fl = lambda a: jnp.array(pad(a), f)
    it = lambda a, fill=-1: jnp.array(pad(a.astype(np.int64), fill), jnp.int32)
    c = Creatures(alive=jnp.array(pad(life.alive)), pos=fl(life.pos), heading=fl(life.heading), mass=fl(life.mass),
                  adult=fl(life.adult), birth=fl(life.birth), fat=fl(life.fat), diet=fl(life.diet),
                  alone=fl(life.alone), reserve=fl(life.reserve), age=fl(life.age), dry=fl(life.dry),
                  gen=it(life.gen, 0), uid=it(life.uid), parent=it(life.parent), parent2=it(life.parent2),
                  mom=it(mom), w=fl(life.w), calling=jnp.array(pad(life.calling)), mem=fl(life.mem))
    b = life.books
    i32 = lambda v: jnp.asarray(v, jnp.int32)
    books = Books(*(jnp.asarray(v, f) for v in (b.start, b.heat, b.dung, b.decay)),
                  *(i32(v) for v in (b.kills, b.births, b.births_alone, b.deaths, b.full, 0, 0)))
    world = World(c, jnp.array(life.meat.reshape(n, n), f), plants, books, i32(life.next_uid), i32(life.ticks))
    return Static(land, ground), world


# --- physiology ---------------------------------------------------------------------------------------------------
def bmr_s(cfg, mass):
    return cfg.bmr_kj_day * mass ** 0.75 / cfg.day_s


def walk_kj_m(cfg, carried):
    return cfg.cot_j_m * carried ** 0.684 / 1000


def sprint_kj_s(cfg, lean):
    return cfg.aerobic_scope * bmr_s(cfg, lean) / (DAY_REAL_S / cfg.day_s)


def reserve_cap(cfg, lean, fat):
    return fat * cfg.fat_kj_kg * lean


def lifespan(cfg, adult):
    return cfg.life_days * (adult * 1000) ** 0.25 / 10


def cell(wcfg, xy):
    """Flat cell of points, and whether each is on the map."""
    n = wcfg.size
    rc = jnp.floor(xy / wcfg.cell_m).astype(jnp.int32)
    inside = ((rc >= 0) & (rc < n)).all(-1)
    rc = jnp.clip(rc, 0, n - 1)
    return rc[..., 0] * n + rc[..., 1], inside


def _ratio(a, b):
    return jnp.where(b > 0, a / jnp.where(b > 0, b, 1.0), 0.0)


# --- who sees whom ------------------------------------------------------------------------------------------------
def nearest_brute(pos, heading, free, radius, edges=SECTORS_DEG):
    """`nearest_by_sector` over all pairs (cap^2: tests and small worlds). Only `free` creatures see or are seen.
    -> near, near_d, sec_b (cap, 3), sec_d; -1 / inf where nobody. Ties go to the lower index (argmin's first)."""
    cap = pos.shape[0]
    dx = pos[None, :, 0] - pos[:, None, 0]          # [a, b]: b seen from a
    dy = pos[None, :, 1] - pos[:, None, 1]
    d = jnp.sqrt(dx * dx + dy * dy)
    ok = free[:, None] & free[None, :] & ~jnp.eye(cap, dtype=bool) & (dx * dx + dy * dy < radius * radius)
    rel = jnp.degrees((jnp.arctan2(dy, dx) - heading[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi)

    def pick(mask):
        dm = jnp.where(mask, d, jnp.inf)
        b = jnp.argmin(dm, axis=1)
        dd = jnp.take_along_axis(dm, b[:, None], 1)[:, 0]
        return jnp.where(jnp.isfinite(dd), b, -1).astype(jnp.int32), dd

    near, near_d = pick(ok)
    secs = [pick(ok & (rel >= edges[e]) & (rel < edges[e + 1])) for e in range(3)]
    return near, near_d, jnp.stack([s[0] for s in secs], 1), jnp.stack([s[1] for s in secs], 1), jnp.int32(0)


def nearest_grid(pos, heading, free, radius, size_m, per_bucket=16, edges=SECTORS_DEG):
    """(port 3) `nearest_brute`'s answer from a grid: creatures binned into radius-wide buckets (as
    `nearest_by_sector`), each looking only at the 3 x 3 buckets around it. A bucket lists at most `per_bucket`
    (by index); the rest are unseen this tick and counted (the 5th result). Ties go to the lower index."""
    cap = pos.shape[0]
    nb = max(int(np.ceil(size_m / radius)), 1)
    bxy = jnp.clip(jnp.floor(pos / radius).astype(jnp.int32), 0, nb - 1)
    none = nb * nb                                  # an always-empty row
    bucket = jnp.where(free, bxy[:, 0] * nb + bxy[:, 1], none + 1)   # not free: listed nowhere
    order = jnp.argsort(bucket, stable=True)
    sb = bucket[order]
    start = jnp.searchsorted(sb, jnp.arange(none + 2), side="left")
    rank = jnp.arange(cap) - start[sb]
    table = jnp.full((none + 1, per_bucket), -1, jnp.int32).at[sb, rank].set(order.astype(jnp.int32), mode="drop")
    crowded = ((rank >= per_bucket) & (sb < none)).sum(dtype=jnp.int32)
    off = jnp.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
    nxy = bxy[:, None, :] + off[None]
    inb = ((nxy >= 0) & (nxy < nb)).all(-1)
    cand = table[jnp.where(inb, nxy[..., 0] * nb + nxy[..., 1], none)].reshape(cap, -1)   # (cap, 9 x per_bucket)
    cb = jnp.maximum(cand, 0)
    dx = pos[cb, 0] - pos[:, None, 0]
    dy = pos[cb, 1] - pos[:, None, 1]
    d2 = dx * dx + dy * dy
    ok = (cand >= 0) & (cand != jnp.arange(cap)[:, None]) & free[:, None] & (d2 < radius * radius)
    d = jnp.sqrt(d2)
    rel = jnp.degrees((jnp.arctan2(dy, dx) - heading[:, None] + jnp.pi) % (2 * jnp.pi) - jnp.pi)

    def pick(mask):
        dm = jnp.where(mask, d, jnp.inf)
        best = dm.min(1)
        b = jnp.where(mask & (dm == best[:, None]), cand, cap).min(1)
        return jnp.where(jnp.isfinite(best), b, -1).astype(jnp.int32), best

    near, near_d = pick(ok)
    secs = [pick(ok & (rel >= edges[e]) & (rel < edges[e + 1])) for e in range(3)]
    return near, near_d, jnp.stack([s[0] for s in secs], 1), jnp.stack([s[1] for s in secs], 1), crowded


def grid_for(wcfg, per_bucket=16):
    """The grid search for this map (what `tick` and `run` use unless told otherwise)."""
    return lambda pos, heading, free, radius: nearest_grid(pos, heading, free, radius, wcfg.size * wcfg.cell_m,
                                                           per_bucket)


# --- the air (sound, scent) ---------------------------------------------------------------------------------------
def blur(field, passes):
    """[1, 2, 1] / 4 each way, `passes` times, zero outside (as numpy's `_air`)."""
    for _ in range(passes):
        for ax in (0, 1):
            p = jnp.pad(field, [(1, 1) if a == ax else (0, 0) for a in (0, 1)])
            n = field.shape[ax]
            lo, mid, hi = (jax.lax.slice_in_dim(p, k, k + n, axis=ax) for k in (0, 1, 2))
            field = (lo + 2 * mid + hi) / 4
    return field


def ears(cfg, field, pos, heading, scale):
    """(strength squashed 0..1, (right - left) / (right + left)), sampled `ear_m` either side."""
    na = field.shape[0]
    side = heading + jnp.pi / 2
    ear = cfg.ear_m * jnp.stack([jnp.cos(side), jnp.sin(side)], 1)

    def at(xy):
        c = jnp.clip(jnp.trunc(xy / cfg.sound_cell_m).astype(jnp.int32), 0, na - 1)
        return field[c[:, 0], c[:, 1]]
    left, right = at(pos + ear), at(pos - ear)
    return jnp.stack([1 - jnp.exp(-(left + right) / 2 / scale), _ratio(right - left, right + left)], 1)


def sound(wcfg, cfg, c: Creatures):
    na = air_n(wcfg, cfg)
    who = c.alive & c.calling
    ac = jnp.clip(jnp.trunc(c.pos / cfg.sound_cell_m).astype(jnp.int32), 0, na - 1)
    field = jnp.zeros(na * na, c.pos.dtype).at[ac[:, 0] * na + ac[:, 1]].add(jnp.where(who, 1.0, 0.0))
    return blur(field.reshape(na, na), 2)


def scent(wcfg, cfg, ground: Ground, fruit):
    na = air_n(wcfg, cfg)
    field = jnp.zeros(na * na, fruit.dtype).at[ground.air_of_cell].add(fruit.reshape(-1))
    return blur(field.reshape(na, na), 3)


# --- senses and brain ---------------------------------------------------------------------------------------------
def riders(cfg, c: Creatures, mass, adult):
    """(8h) Who rides on a living mother: (bool mask, mother slot)."""
    if not cfg.carry:
        return jnp.zeros_like(c.alive), jnp.zeros_like(c.mom)
    mom = jnp.maximum(c.mom, 0)
    ride = (c.alive & (mass < cfg.wean_frac * adult) & (c.parent >= 0) & (c.mom >= 0)
            & c.alive[mom] & (c.uid[mom] == c.parent) & (mom != jnp.arange(len(mom))))
    return ride, mom


def senses(wcfg, cfg, st: Static, wd: World, mass, adult, fat, nearest):
    """The brain's 33 inputs for every slot, and who's nearest (any direction) and straight ahead."""
    c, pl = wd.c, wd.plants
    per_cell = wcfg.plant_kj_m2 * wcfg.cell_m ** 2
    food = (pl.plants + pl.dead).reshape(-1)
    meat, fruit = wd.meat.reshape(-1), pl.fruit.reshape(-1)
    here, _ = cell(wcfg, c.pos)
    ang = c.heading[:, None] + jnp.radians(jnp.array([-cfg.sight_deg, 0.0, cfg.sight_deg]))
    eye = c.pos[:, None] + cfg.sight_m * jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)
    cells, inside = cell(wcfg, eye)
    blocked = ~inside | st.ground.water[cells]
    seen = lambda v: jnp.where(blocked, 0.0, v)
    carried = mass + c.reserve / cfg.fat_kj_kg
    ride, mom = riders(cfg, c, mass, adult)
    free = c.alive & ~ride
    near, near_d, sec_b, sec_d, crowded = nearest(c.pos, c.heading, free, cfg.sight_m)
    has = sec_b >= 0
    b = jnp.maximum(sec_b, 0)
    close = jnp.where(has, 1 - sec_d / cfg.sight_m, 0.0)
    size = jnp.where(has, jnp.clip(jnp.log(carried[b] / carried[:, None]), -3, 3) / 3, 0.0)
    ahead, ahead_d = sec_b[:, 1], sec_d[:, 1]
    k = b[:, 1]
    ua, ub, pa, pb, qa, qb = c.uid, c.uid[k], c.parent, c.parent[k], c.parent2, c.parent2[k]
    kin = ((pb == ua) | (qb == ua) | (pa == ub) | (qa == ub) | ((pa == pb) & (pa >= 0)) | ((qa == qb) & (qa >= 0)))
    nose = jnp.stack([jnp.where(has[:, 1], 1 - jnp.abs(c.diet - c.diet[k]), 0.0),
                      jnp.where(has[:, 1], kin, False).astype(c.pos.dtype)], 1)
    fs = cfg.fruit_sense_kj
    x = jnp.concatenate([
        (food[here] / per_cell)[:, None],                                  # 0
        seen(food[cells] / per_cell),                                      # 1-3
        blocked.astype(c.pos.dtype),                                       # 4-6
        _ratio(c.reserve, reserve_cap(cfg, mass, fat))[:, None],          # 7
        (mass / adult)[:, None],                                           # 8
        jnp.ones_like(mass)[:, None],                                      # 9
        jnp.minimum(meat[here] / (5 * cfg.body_kj_kg), 1.0)[:, None],     # 10
        close, size,                                                       # 11-16
        seen(jnp.minimum(meat[cells] / (5 * cfg.body_kj_kg), 1.0)),       # 17-19
        nose,                                                              # 20-21
        ears(cfg, sound(wcfg, cfg, c), c.pos, c.heading, 1.0),            # 22-23
        c.mem,                                                             # 24-25
        (1 - jnp.exp(-fruit[here] / fs))[:, None],                        # 26
        seen(1 - jnp.exp(-fruit[cells] / fs)),                            # 27-29
        ears(cfg, scent(wcfg, cfg, st.ground, pl.fruit), c.pos, c.heading, fs),   # 30-31
        (c.dry / (cfg.dry_max_l_kg * mass))[:, None],                     # 32
    ], 1)
    sees = dict(near=near, near_d=near_d, ahead=ahead, ahead_d=ahead_d, ride=ride, mom=mom, here=here, crowded=crowded)
    return x, sees


def think(w, x):
    """The brains: 33 -> 8 tanh -> 8, each creature its own weights."""
    a = 0
    w1 = w[:, a:a + N_IN * N_HID].reshape(-1, N_IN, N_HID); a += N_IN * N_HID
    b1 = w[:, a:a + N_HID]; a += N_HID
    w2 = w[:, a:a + N_HID * N_OUT].reshape(-1, N_HID, N_OUT); a += N_HID * N_OUT
    b2 = w[:, a:a + N_OUT]
    h = jnp.tanh(jnp.einsum("ni,nih->nh", x, w1, precision="highest") + b1)
    return jnp.einsum("nh,nho->no", h, w2, precision="highest") + b2


def _take(pool, reach, cells, want):
    asked = jnp.zeros_like(pool).at[cells].add(want)
    share = _ratio(jnp.minimum(reach, asked), asked)
    return jnp.maximum(pool - asked * share, 0.0), want * share[cells]


# --- chance ------------------------------------------------------------------------------------------------------
MAX_BIRTHS = 128   # children a world can have in one tick (brains for more aren't drawn); the rest: `books.capped`
PRIO_END = 1 << 30


class Draws(NamedTuple):
    """One tick's random numbers (a test hands in numpy's): one row per slot, except the brains' (one row per child,
    in birth order: drawing 337 numbers for every slot every tick cost more than the rest of the tick, devlog 10e)."""
    settle: jax.Array          # (cap,) uniform: is this attacker's fight settled this second
    won: jax.Array             # (cap,) uniform: did the attacker win
    prio: jax.Array            # (cap,) int32, unique, < PRIO_END: the mating order (numpy: a random permutation)
    solo: jax.Array            # (cap,) uniform: breeds alone this tick
    pick: jax.Array            # (cap, 5) uniform: each gene from this parent (< 0.5) or the partner
    gene: jax.Array            # (cap, 5) standard normal: each gene's mutation (adult, birth, fat, diet, alone)
    heading: jax.Array         # (cap,) uniform in [-pi, pi): the child's heading
    w_pick: jax.Array          # (max_births, N_W) uniform: each brain weight from the parent or the partner
    w_noise: jax.Array         # (max_births, N_W) standard normal: the child's brain mutation


def draw(key, cap, dtype=None, max_births=MAX_BIRTHS) -> Draws:
    f = dtype or jnp.zeros(0).dtype
    k = jax.random.split(key, 9)
    u = lambda kk, shape: jax.random.uniform(kk, shape, f)
    # a random order without a sort: random high bits, the slot in the low ones (unique, so no ties)
    prio = jax.random.randint(k[2], (cap,), 0, PRIO_END // cap) * cap + jnp.arange(cap)
    return Draws(u(k[0], (cap,)), u(k[1], (cap,)), prio.astype(jnp.int32), u(k[3], (cap,)), u(k[4], (cap, 5)),
                 jax.random.normal(k[5], (cap, 5), f), jax.random.uniform(k[7], (cap,), f, -jnp.pi, jnp.pi),
                 u(k[6], (max_births, N_W)), jax.random.normal(k[8], (max_births, N_W), f))


def match(propose, near, prio):
    """(port 4) Pairs for mating, as numpy's loop makes them: proposers in `prio` order, each taking its `near` if
    neither is taken yet. In parallel: a proposal is taken when it has the lowest priority among the open proposals
    at both its ends; repeat until none is open (the same pairs as the loop, for the same order).
    -> (taken: bool per proposer, paired: bool per slot)."""
    cap = propose.shape[0]
    big = jnp.int32(PRIO_END)
    tgt = jnp.maximum(near, 0)

    def body(state):
        taken, paired, _ = state
        open_ = propose & ~paired & ~paired[tgt]
        p = jnp.where(open_, prio, big)
        low = jnp.full(cap, big).at[jnp.arange(cap)].min(p).at[tgt].min(p)
        take = open_ & (low == p) & (low[tgt] == p)
        paired = paired | take
        paired = paired.at[tgt].max(take)
        return taken | take, paired, take.any()

    taken, paired, _ = jax.lax.while_loop(lambda s: s[2], body,
                                          (jnp.zeros(cap, bool), jnp.zeros(cap, bool), jnp.bool_(True)))
    return taken, paired


def births(cfg, wd: World, events, partner, sexual, order, d: Draws):
    """(6a, 6b) One child per event (a parent slot; `partner` the other parent, itself when alone), if both can
    pay their share; children take free slots in `order` (lowest first, as numpy's two `_births` calls), at most
    `max_births` (the brain draws' rows) a tick, the rest counted in `books.capped`.
    -> (world, born mask over parents)."""
    c, bk = wd.c, wd.books
    cap = c.alive.shape[0]
    q = jnp.where(sexual, partner, jnp.arange(cap))

    def pick(v, g):
        return jnp.where(sexual & (d.pick[:, g] >= 0.5), v[q], v)
    adult = pick(c.adult, 0) * jnp.exp(cfg.sigma_gene * d.gene[:, 0])
    frac = jnp.clip(pick(c.birth, 1) * jnp.exp(cfg.sigma_gene * d.gene[:, 1]), 0.01, 0.9)
    fat = jnp.clip(pick(c.fat, 2) * jnp.exp(cfg.sigma_gene * d.gene[:, 2]), 0.01, 1.0)
    diet = jnp.clip(pick(c.diet, 3) + cfg.sigma_diet * d.gene[:, 3], 0.0, 1.0)
    alone = jnp.clip(pick(c.alone, 4) + cfg.sigma_alone * d.gene[:, 4], 0.0, 1.0)
    m0 = frac * adult
    r0 = 0.5 * reserve_cap(cfg, m0, fat)
    cost = m0 * cfg.body_kj_kg * (1 + cfg.build_overhead) + r0
    share = jnp.where(sexual, 0.5, 1.0)
    want = events & (c.reserve >= share * cost) & (c.reserve[q] >= share * cost)
    # children in `order` into the free slots, lowest first
    key = jnp.where(want, order, jnp.iinfo(jnp.int32).max)
    by = jnp.argsort(key, stable=True)
    rank = jnp.zeros(cap, jnp.int32).at[by].set(jnp.arange(cap, dtype=jnp.int32))
    free = jnp.nonzero(~c.alive, size=cap, fill_value=cap)[0].astype(jnp.int32)
    kid = jnp.where(want, free[jnp.minimum(rank, cap - 1)], cap)
    room = want & (kid < cap)
    born = room & (rank < d.w_pick.shape[0])
    full = (want & ~room).sum(dtype=jnp.int32)
    capped = (room & ~born).sum(dtype=jnp.int32)
    kid = jnp.where(born, kid, cap)                   # out of range: dropped by the scatters
    pay = jnp.where(born, share * cost, 0.0)
    reserve = c.reserve - pay
    reserve = reserve.at[jnp.where(born & sexual, q, cap)].add(-pay, mode="drop")
    put = lambda arr, v: arr.at[kid].set(v, mode="drop")
    c = c._replace(
        reserve=put(reserve, r0), alive=put(c.alive, True), pos=put(c.pos, c.pos), heading=put(c.heading, d.heading),
        adult=put(c.adult, adult), birth=put(c.birth, frac), mass=put(c.mass, m0), fat=put(c.fat, fat),
        diet=put(c.diet, diet), alone=put(c.alone, alone), age=put(c.age, 0.0), dry=put(c.dry, 0.0),
        gen=put(c.gen, jnp.maximum(c.gen, c.gen[q]) + 1), parent=put(c.parent, c.uid),
        parent2=put(c.parent2, jnp.where(sexual, c.uid[q], -1)),
        uid=put(c.uid, wd.next_uid + rank), mom=put(c.mom, jnp.arange(cap, dtype=jnp.int32)),
        calling=put(c.calling, False), mem=put(c.mem, 0.0))
    ev = by[:d.w_pick.shape[0]]                       # the parents in birth order, one row of brain draws each
    w = jnp.where(sexual[ev][:, None] & (d.w_pick >= 0.5), c.w[q[ev]], c.w[ev]) + cfg.sigma_w * d.w_noise
    c = c._replace(w=c.w.at[kid[ev]].set(w, mode="drop"))
    n = born.sum(dtype=jnp.int32)
    bk = bk._replace(heat=bk.heat + jnp.where(born, m0 * cfg.body_kj_kg * cfg.build_overhead, 0.0).sum(),
                     births=bk.births + n, full=bk.full + full, capped=bk.capped + capped)
    return wd._replace(c=c, books=bk, next_uid=wd.next_uid + n), born


# --- one tick -----------------------------------------------------------------------------------------------------
def tick(wcfg: WorldConfig, cfg: LifeConfig, st: Static, wd: World, key, nearest=None, draws: Draws | None = None):
    """One creature tick (`Life.step` up to the plants' step). `nearest`: who sees whom (default: the grid,
    `grid_for`). The chance comes from `key`, or from `draws` if given."""
    nearest = nearest or grid_for(wcfg)
    d = draws if draws is not None else draw(key, wd.c.alive.shape[0], wd.c.pos.dtype)
    dt = cfg.dt
    day = dt / cfg.day_s
    c, pl, bk = wd.c, wd.plants, wd.books
    c0_pos = c.pos
    a = c.alive
    af = a.astype(c.pos.dtype)
    m = jnp.where(a, c.mass, 1.0)          # safe stand-ins in dead slots (no 0 ** x, no x / 0)
    adult = jnp.where(a, c.adult, 1.0)
    fat = jnp.where(a, c.fat, 0.1)
    diet = c.diet
    x, sees = senses(wcfg, cfg, st, wd, m, adult, fat, nearest)
    out = think(c.w, x)
    turn, speed = jnp.tanh(out[:, 0]), 1 / (1 + jnp.exp(-out[:, 1]))
    eat, split, attack = out[:, 2] > 0, out[:, 3] > 0, out[:, 4] > 0
    call = out[:, 5] > 0
    carried = m + c.reserve / cfg.fat_kj_kg
    ride, mom = sees["ride"], sees["mom"]
    load = carried + jnp.zeros_like(carried).at[mom].add(jnp.where(ride, carried, 0.0))
    speed = jnp.where(ride, 0.0, speed)
    attack = attack & ~ride & a

    # move; water and the map's edge stop you
    heading = jnp.where(a, (c.heading + turn * cfg.turn_rad_s * dt + jnp.pi) % (2 * jnp.pi) - jnp.pi, c.heading)
    v = speed * sprint_kj_s(cfg, m) / walk_kj_m(cfg, load)
    stepv = (v * dt)[:, None] * jnp.stack([jnp.cos(heading), jnp.sin(heading)], 1)
    to, inside = cell(wcfg, c.pos + stepv)
    ok = inside & ~st.ground.water[to] & a
    pos = c.pos + jnp.where(ok[:, None], stepv, 0.0)
    walked = jnp.where(ok, v * dt, 0.0)
    pos = jnp.where(ride[:, None], pos[mom], pos)

    # attack the nearest straight ahead, if in reach: a fight that may kill either (the bigger usually wins)
    tgt, td = sees["ahead"], sees["ahead_d"]
    hit = attack & (tgt >= 0) & (td < cfg.reach_m)
    t_ = jnp.maximum(tgt, 0)
    arm = 1 + cfg.weapon * diet
    ca, ct = carried * arm, carried[t_] * arm[t_]
    settled = hit & (d.settle < cfg.fight_per_s * dt)
    won = d.won < ca ** 2 / (ca ** 2 + ct ** 2)
    killed = (settled & ~won) | jnp.zeros_like(a).at[jnp.where(settled & won, t_, len(a))].set(True, mode="drop")
    effort = jnp.where(attack, sprint_kj_s(cfg, m) * dt, 0.0)
    rest = bmr_s(cfg, m) * (1 + cfg.call_bmr * call)
    warm = keep_warm_s(cfg, m, ej.temperature(wcfg, wd.plants.day), rest, jnp.maximum)   # (11b)
    cost = af * ((rest + warm) * dt + walk_kj_m(cfg, load) * walked + effort)
    reserve = c.reserve - cost
    heat = bk.heat + cost.sum()

    # eat: fruit, then carcass, then plants
    keep_meat = cfg.digest_meat_grazer + (cfg.digest_meat - cfg.digest_meat_grazer) * diet
    keep_plant = plant_keep(cfg, diet, m, jnp.exp)
    keep_fruit = cfg.digest_fruit + (cfg.digest_fruit_hunter - cfg.digest_fruit) * diet
    room = jnp.maximum(reserve_cap(cfg, m, fat) - reserve, 0.0)
    want = af * eat * (1 - speed) * cfg.eat_bmr_day * bmr_s(cfg, m) * dt
    here, _ = cell(wcfg, pos)
    pl, got_f = ej.eat_fruit(pl, here, jnp.minimum(want, room / keep_fruit))
    room = room - keep_fruit * got_f
    want = want - got_f
    want_m = jnp.where(keep_meat > 0.01, jnp.minimum(want, room / jnp.maximum(keep_meat, 1e-9)), 0.0)
    meat, got_m = _take(wd.meat.reshape(-1), wd.meat.reshape(-1), here, want_m)
    room = room - keep_meat * got_m
    want_p = jnp.where(keep_plant > 0.01, jnp.minimum(want - got_m, room / jnp.maximum(keep_plant, 1e-9)), 0.0)
    pl, green, dead = ej.graze(st.land, pl, here, jnp.maximum(want_p, 0.0))
    got_p = keep_plant * (green + wcfg.dead_quality * dead)
    reserve = reserve + keep_meat * got_m + got_p + keep_fruit * got_f
    dung = bk.dung + ((1 - keep_meat) * got_m + green + dead - got_p + (1 - keep_fruit) * got_f).sum()

    # (8f) milk: young within reach of their (living, unkilled) mother drink from her spare reserve
    milk = jnp.zeros_like(reserve)
    if cfg.milk:
        mslot = jnp.maximum(c.mom, 0)
        young = (a & (m < cfg.wean_frac * adult) & (c.parent >= 0) & ~killed & (c.mom >= 0) & a[mslot]
                 & (c.uid[mslot] == c.parent) & ~killed[mslot]
                 & (jnp.hypot(*(pos - pos[mslot]).T) < cfg.reach_m))
        room_y = jnp.maximum(reserve_cap(cfg, m, fat) - reserve, 0.0)
        ask = jnp.where(young, jnp.minimum(cfg.eat_bmr_day * bmr_s(cfg, m) * dt, room_y / cfg.milk_eff), 0.0)
        spare = jnp.maximum(reserve - cfg.milk_floor * reserve_cap(cfg, m, fat), 0.0)
        asked = jnp.zeros_like(reserve).at[mslot].add(ask)
        give = ask * jnp.minimum(1.0, _ratio(spare[mslot], asked[mslot]))
        reserve = reserve.at[mslot].add(-give) + cfg.milk_eff * give
        heat = heat + ((1 - cfg.milk_eff) * give).sum()
        milk = cfg.milk_eff * give

    dry = c.dry
    drank = jnp.zeros_like(a)
    if cfg.thirst:
        wet_cost = cost - af * (1 - cfg.breath_water_per_kj / cfg.water_per_kj) * warm * dt if cfg.cold_breath else cost
        dry = dry + cfg.water_per_kj * wet_cost - (cfg.water_green * green + cfg.water_dead * dead + cfg.water_meat * got_m
                                                + cfg.water_fruit * got_f + cfg.water_milk * milk)
        drinkable = st.ground.shore | (pl.wet.reshape(-1) > cfg.puddle_wet) if wcfg.rain else st.ground.shore
        drank = eat & drinkable[here] & a
        dry = jnp.maximum(jnp.where(drank, 0.0, dry), 0.0)

    # grow towards adult mass while the reserve is comfortable
    g = cfg.growth_a * (m * 1000) ** 0.75 * (1 - (m / adult) ** 0.25) * day / 1000
    per_kg = cfg.body_kj_kg * (1 + cfg.build_overhead)
    spare = jnp.maximum(reserve - cfg.grow_above * reserve_cap(cfg, m, fat), 0.0)
    g = af * jnp.maximum(jnp.minimum(g, spare / per_kg), 0.0)
    mass = c.mass + g
    reserve = reserve - g * per_kg
    heat = heat + (g * cfg.body_kj_kg * cfg.build_overhead).sum()

    c = c._replace(pos=pos, heading=heading, mass=mass, reserve=jnp.where(a, reserve, c.reserve), dry=dry,
                   age=c.age + af * day, calling=jnp.where(a, call, c.calling),
                   mem=jnp.where(a[:, None], jnp.tanh(out[:, 6:8]), c.mem))
    wd = wd._replace(c=c, meat=meat.reshape(wd.meat.shape), plants=pl, books=bk._replace(heat=heat, dung=dung,
                                                                           crowded=bk.crowded + sees["crowded"]))
    ready = a & split & (mass >= 0.95 * c.adult) & ~killed
    cap = len(a)
    near, nd = sees["near"], sees["near_d"]
    n_ = jnp.maximum(near, 0)
    if cfg.sexual:   # (6a) each ready one with the nearest, if in reach, ready too and (8g) alike
        cand = (ready & (near >= 0) & (nd < cfg.reach_m) & ready[n_]
                & (jnp.abs(c.diet - c.diet[n_]) <= cfg.mate_gap))
        taken, paired = match(cand, near, d.prio)
        # (6b) ready, no mate: alone with a chance of `alone` a wait of alone_wait_days
        wait = ready & ~paired & ~killed
        solo = wait & (d.solo < c.alone * dt / (cfg.alone_wait_days * cfg.day_s))
        events, sexual = taken | solo, taken
        order = jnp.where(taken, d.prio, PRIO_END + jnp.arange(cap, dtype=jnp.int32))
    else:
        cand = taken = paired = wait = jnp.zeros_like(a)
        solo = events = ready
        sexual = jnp.zeros_like(a)
        order = jnp.arange(cap, dtype=jnp.int32)
    wd, born = births(cfg, wd, events, n_, sexual, order, d)
    if cfg.sexual:
        wd = wd._replace(books=wd.books._replace(births_alone=wd.books.births_alone
                                                 + (born & solo).sum(dtype=jnp.int32)))
    wd, cause = deaths(wcfg, cfg, wd, a, killed)
    info = dict(x=x, out=out, killed=killed, ready=ready, drank=drank, milk=milk, sees=sees, speed=speed,
                got_f=got_f, got_m=got_m, got_p=got_p, hit=hit, settled=settled, won=won, cand=cand, taken=taken,
                paired=paired, wait=wait, solo=solo, born=born, attack=attack, cause=cause, walked=walked, moved=jnp.where(a, jnp.hypot(*(pos - c0_pos).T), 0.0),
                green=green, dead=dead, alive=a, call=call)
    return wd._replace(ticks=wd.ticks + 1), info


def deaths(wcfg, cfg, wd: World, was_alive, killed):
    """Killed, starved, old or parched (only those alive at the tick's start): body and fat become a carcass."""
    c, bk = wd.c, wd.books
    mass_s = jnp.where(was_alive, c.mass, 1.0)
    starved, parched = c.reserve < 0, c.dry > cfg.dry_max_l_kg * mass_s
    old = c.age > lifespan(cfg, jnp.where(was_alive, c.adult, 1.0))
    die = was_alive & (killed | starved | old | parched)
    # (port 5) one cause each, in this order (a creature can starve and parch in the same tick)
    cause = jnp.stack([die & killed, die & ~killed & parched, die & ~killed & ~parched & starved,
                       die & ~killed & ~parched & ~starved & old])
    where, _ = cell(wcfg, c.pos)
    meat = wd.meat.reshape(-1).at[where].add(jnp.where(die, c.reserve + cfg.body_kj_kg * c.mass, 0.0))
    c = c._replace(alive=c.alive & ~die, reserve=jnp.where(die, 0.0, c.reserve), mass=jnp.where(die, 0.0, c.mass))
    bk = bk._replace(kills=bk.kills + killed.sum(dtype=jnp.int32), deaths=bk.deaths + die.sum(dtype=jnp.int32))
    return wd._replace(c=c, meat=meat.reshape(wd.meat.shape), books=bk), cause


def plant_step(wcfg, cfg, st: Static, wd: World, storms):
    """Carcasses rot (by the air's warmth), then the plants' step."""
    rate = cfg.meat_rot_per_day * cfg.rot_q10 ** ((ej.temperature(wcfg, wd.plants.day) - wcfg.temp_summer_c) / 10)
    rot = wd.meat * (1 - jnp.exp(-rate * wcfg.dt_days))
    return wd._replace(meat=wd.meat - rot, books=wd.books._replace(decay=wd.books.decay + rot.sum()),
                       plants=ej.step(wcfg, st.land, wd.plants, storms))


class Probe(NamedTuple):
    """(port 5) What happened over a stretch of ticks, summed (`observe`); per world under vmap. Food is what the
    eaters kept (digested kJ); milk is what the young kept."""
    steps: jax.Array           # creature-ticks
    drink: jax.Array           # creature-ticks drinking
    drink_thirst: jax.Array    # their thirst (0..1) when they drank, summed
    thirst: jax.Array          # thirst (0..1), summed
    moved: jax.Array           # metres moved (carried young too, as numpy's 9d measure)
    calls: jax.Array
    attacks: jax.Array
    food: jax.Array            # (5,) kJ kept: fruit, carcass, green grass, dead grass, milk
    dead: jax.Array            # (4,) deaths: killed, thirst, starved, old
    green_here: jax.Array      # grass-eaters (diet < 0.5): food under them / a full cell, summed (9a's green probe)
    green_land: jax.Array      # ... and the land's average, as many times
    births: jax.Array


def no_probe(dtype=None) -> Probe:
    f = dtype or jnp.zeros(0).dtype
    z = jnp.zeros((), f)
    return Probe(z, z, z, z, z, z, z, jnp.zeros(5, f), jnp.zeros(4, jnp.int32), z, z, jnp.zeros((), jnp.int32))


def observe(wcfg, cfg, st: Static, wd_before: World, info, p: Probe) -> Probe:
    a, x = info["alive"], info["x"]
    f = x.dtype
    th = jnp.where(a, x[:, 32], 0.0)
    keep_plant = plant_keep(cfg, wd_before.c.diet, wd_before.c.mass, jnp.exp)
    food = jnp.stack([info["got_f"] * (cfg.digest_fruit + (cfg.digest_fruit_hunter - cfg.digest_fruit)
                                       * wd_before.c.diet),
                      info["got_m"] * (cfg.digest_meat_grazer + (cfg.digest_meat - cfg.digest_meat_grazer)
                                       * wd_before.c.diet),
                      keep_plant * info["green"], keep_plant * wcfg.dead_quality * info["dead"], info["milk"]], 1)
    g = a & (wd_before.c.diet < 0.5)
    land = ~st.ground.water
    per_cell = wcfg.plant_kj_m2 * wcfg.cell_m ** 2
    pl = wd_before.plants
    avg = jnp.where(land, (pl.plants + pl.dead).reshape(-1), 0.0).sum() / land.sum() / per_cell
    return Probe(p.steps + a.sum(), p.drink + info["drank"].sum(), p.drink_thirst + jnp.where(info["drank"], th, 0).sum(),
                 p.thirst + th.sum(), p.moved + info["moved"].sum(), p.calls + (info["call"] & a).sum(),
                 p.attacks + info["attack"].sum(), p.food + jnp.where(a[:, None], food, 0.0).sum(0),
                 p.dead + info["cause"].sum(1, dtype=jnp.int32), p.green_here + jnp.where(g, x[:, 0], 0.0).sum(),
                 p.green_land + g.sum() * avg, p.births + info["born"].sum(dtype=jnp.int32))


def run(wcfg, cfg, st: Static, wd: World, key, plant_steps: int, nearest=None, probe: Probe | None = None):
    """`plant_steps` x (plant_every ticks, then the plants' step), in one compiled loop. vmap it over worlds.
    With `probe`: also sum what happened into it (`observe`) -> (world, probe)."""
    per = plant_every(wcfg, cfg)

    def one(carry, kk):
        w, p = carry
        w2, info = tick(wcfg, cfg, st, w, kk, nearest)
        return (w2, observe(wcfg, cfg, st, w, info, p) if probe is not None else p), None

    def body(carry, k):
        kt, ks = jax.random.split(k)
        wd, p = jax.lax.scan(one, carry, jax.random.split(kt, per))[0]
        storms = ej.draw_storms(wcfg, ks, wd.plants.day) if wcfg.rain else None
        return (plant_step(wcfg, cfg, st, wd, storms), p), None
    wd, p = jax.lax.scan(body, (wd, probe if probe is not None else jnp.zeros(())),
                         jax.random.split(key, plant_steps))[0]
    return (wd, p) if probe is not None else wd


def scramble(wd: World, key) -> World:
    """(9c) The control: the same world with every living creature's brain redrawn at random (N(0, 1), as a first
    creature's), as `eco_kills.control`."""
    c = wd.c
    w = jnp.where(c.alive[:, None], jax.random.normal(key, c.w.shape, c.w.dtype), c.w)
    return wd._replace(c=c._replace(w=w))


def energy(wd: World, cfg):
    c = wd.c
    return jnp.where(c.alive, c.reserve + cfg.body_kj_kg * c.mass, 0.0).sum()


def error(wd: World, cfg):
    """Books minus reality, plants and creatures together (`Life.error`)."""
    p, b = wd.plants, wd.books
    entered = p.start + p.sun_in + b.start
    expected = entered - p.dieback - b.heat - b.dung - b.decay
    return (expected - ej.total(p) - energy(wd, cfg) - wd.meat.sum()) / entered
