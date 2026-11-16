"""The ecosystem, step 1: a world of soil, water and plants, with every unit of energy booked.

The world is a square grid of cells. Each cell has a biome (water, desert, grassland,
forest) and a soil fertility, which together set how much plant energy it can hold
(its carrying capacity). Plants are the only energy source the creatures will ever
have: they grow on sunlight, faster in summer, towards that capacity (logistic growth),
seed bare cells next to them, and die back a little in winter.

Each plant is two pools. Shoots (`plants`) are what grazers can eat and what catches the
sun. Roots (`roots`) are underground, out of a grazer's reach: part of each day's growth
goes to them (towards a real root:shoot ratio for the biome), they push new shoots up when
the shoots have been grazed or died back, and they lose only a little in winter. That's
how real grassland survives being grazed flat and comes back each spring. And no mouth
bites to the ground: a short stubble of shoots (`stubble` of capacity) is out of reach too,
and keeps catching the sun when everything above it has been eaten.

(Ep17 step 3h) Shoots that die back don't vanish: they stand as dead grass (`dead`), the hay
that feeds real grazers through winter. It can be eaten (poorer food: `dead_quality` of green's
digestible energy, applied by the eater) and weathers away slowly (`dead_decay_per_day`, booked
as die-back). Grazers take green shoots first, then the dead.

(Step 7a) Fruit trees: a few, scattered (`fruit_trees_ha` per hectare of each biome, mostly forest), each dropping
its crop (`fruit_kj`, ~100 kg of fruit at ~2.2 MJ/kg: an apple tree's order) under its crown from late summer into
autumn (`fruit_days`). Only what's on the ground can be eaten (`fruit`, per cell); it rots like carcasses, fast when
warm, slow when cold. A rich food, in a few places, a few months a year: worth finding and remembering. The fruit
comes from the sun (booked as sun in), its rot as die-back.

The energy ledger books every kilojoule: sunlight fixed by plants comes in, and plant
energy leaves only by being eaten (`graze`, shoots only) or by die-back. `World.error()`
is the gap between the books and what is actually in the world; it should stay at
rounding level forever. Sim exploits later get fixed here (the physics), never as rules
on the creatures.

Plain numpy, no rendering: `step()` runs headless and fast.
"""

from dataclasses import dataclass, field

import numpy as np

WATER, DESERT, GRASS, FOREST = 0, 1, 2, 3
BIOMES = ("water", "desert", "grass", "forest")


@dataclass(frozen=True)
class WorldConfig:
    size: int = 128                  # cells per side
    cell_m: float = 2.0              # each cell is cell_m x cell_m metres
    dt_days: float = 0.25            # days per plant step
    year_days: float = 360.0
    # Our estimates, to be checked when creatures arrive: grassland holds ~0.3 kg dry plant per m2 at ~17 kJ/g.
    plant_kj_m2: float = 5000.0      # most plant energy a square metre of the best grassland soil holds
    growth_per_day: float = 0.05     # logistic growth rate at midsummer
    seed_per_day: float = 0.02       # how fast neighbours seed a cell, per unit of their plant energy
    dieback_per_day: float = 0.01    # share of plant energy lost per day at midwinter
    water_frac: float = 0.08         # share of the map under water
    desert_frac: float = 0.15        # shares of the land
    forest_frac: float = 0.25
    capacity: tuple = (0.0, 0.1, 1.0, 2.0)   # shoot carrying capacity by biome, relative to grassland
    # Roots: root:shoot ratios inside the ranges Mokany et al. (2006) report (grassland ~1-4, forest ~0.2-0.3,
    # dry shrubland ~1); the rates are our estimates.
    root_ratio: tuple = (0.0, 1.0, 2.0, 0.25)  # root capacity / shoot capacity, by biome
    regrow_per_day: float = 0.05     # share of root energy pushed up as new shoots per day, when shoots have room
    root_loss_per_day: float = 0.002 # root turnover (~half a year's worth lost a year), more in winter
    stubble: float = 0.1             # share of shoot capacity too close to the ground to graze (our estimate)
    # (3h) standing dead grass: cured grass keeps ~0.45-0.5 digestible against green's ~0.7 (forage tables), so
    # ~0.65 of the energy a grazer gets from green; weathering and leaching take roughly a third of it over a
    # ~100-day winter (our estimate from that)
    dead_quality: float = 0.65
    dead_decay_per_day: float = 0.004
    start_fill: float = 0.5          # plants start at this share of capacity
    # (step 4a) the biggest patches (hills, wet and dry land) are about this many metres across, whatever the map's
    # size: a bigger map has more patches, not bigger ones (64 m: the 256 m map's patches until now)
    patch_m: float = 64.0
    # (4e) the air: a temperate grassland's (our estimate), for anything that runs faster when warm (rot)
    temp_winter_c: float = 0.0
    temp_summer_c: float = 20.0
    # (7a) fruit trees (our estimates): wild fruit trees in temperate woodland run ~5-20 a hectare
    fruit_trees_ha: tuple = (0.0, 0.0, 1.0, 10.0)   # by biome: water, desert, grass, forest
    fruit_kj: float = 220_000.0                      # a tree's crop a year (~100 kg x 2.2 MJ/kg)
    fruit_days: tuple = (200.0, 300.0)               # when it falls, days into the year (midsummer is 180)
    crown_m: float = 4.0                             # it falls within this far of the trunk
    fruit_rot_per_day: float = 0.15                  # at summer warmth (a week or two on the ground); Q10 below
    rot_q10: float = 2.5
    # (9a) rain (our estimates): grass grows only where the soil is wet. Storms drop rain in splashes ~storm_m across;
    # most fall around a rainy front that circles the map twice a year (as the rains that herds follow), the rest
    # anywhere. The soil dries with a time constant of dry_days; growth (and regrowth from roots) runs at
    # min(1, wet / wet_full) of full, and dry grass browns into dead grass at drought_per_day.
    rain: bool = False
    storms_ha_day: float = 0.1       # storms a day per hectare of map
    storm_m: float = 30.0            # a storm's splash, gaussian sd in metres
    storm_rain: float = 0.8          # soil wetness it adds at its centre (wetness is 0..1)
    belt_share: float = 0.8          # share of storms around the rainy front
    belt_m: float = 40.0             # the front's spread (gaussian sd), metres
    belt_loop: float = 0.3           # its circle's radius, as a share of the map's side
    belt_per_year: float = 2.0       # loops a year
    dry_days: float = 15.0
    wet_full: float = 0.4
    drought_per_day: float = 0.02


def _smooth_noise(rng, n, scales=(4, 8, 16), weights=(0.6, 0.3, 0.1)):
    """Smooth random field on an n x n grid, rank-transformed to uniform [0, 1) so fractions are exact. `scales`:
    lattice points across the map, coarse to fine."""
    out = np.zeros((n, n))
    for s, w in zip(scales, weights):
        g = rng.random((s + 1, s + 1))
        x = np.linspace(0, s, n, endpoint=False)
        i, f = x.astype(int), x - x.astype(int)
        rows = g[i] * (1 - f)[:, None] + g[i + 1] * f[:, None]
        out += w * (rows[:, i] * (1 - f) + rows[:, i + 1] * f)
    ranks = np.empty(n * n)
    ranks[np.argsort(out, axis=None)] = np.arange(n * n) / (n * n)
    return ranks.reshape(n, n)


@dataclass
class Ledger:
    """Energy books for the whole world, in kJ (float64)."""
    start: float = 0.0       # plant energy at the start
    sun_in: float = 0.0      # fixed from sunlight by plant growth
    eaten: float = 0.0       # moved out of plants into whoever ate them
    dieback: float = 0.0     # lost to root turnover and dead grass weathering away (heat, decay)

    def expected_plants(self):
        return self.start + self.sun_in - self.eaten - self.dieback


@dataclass
class World:
    cfg: WorldConfig
    biome: np.ndarray        # (n, n) int8
    capacity: np.ndarray     # (n, n) kJ each cell's plants can hold
    plants: np.ndarray       # (n, n) kJ of shoots (what can be eaten) in each cell
    roots: np.ndarray        # (n, n) kJ of roots, out of reach
    root_capacity: np.ndarray
    day: float = 0.0
    ledger: Ledger = field(default_factory=Ledger)
    dead: np.ndarray = None  # (n, n) kJ of standing dead grass (3h)
    trees: np.ndarray = None # (7a) (k, 2) the fruit trees' cells
    fruit: np.ndarray = None # (7a) (n, n) kJ of fallen fruit
    wet: np.ndarray = None   # (9a) (n, n) soil wetness 0..1
    rng: object = None       # (9a) the weather's own random stream

    def __post_init__(self):
        if self.dead is None:
            self.dead = np.zeros_like(self.plants)
        if self.fruit is None:
            self.fruit = np.zeros_like(self.plants)
        if self.wet is None:
            self.wet = np.full_like(self.plants, 0.5)
        if self.rng is None:
            self.rng = np.random.default_rng(0)
        if self.trees is None:
            self.trees = np.zeros((0, 2), int)
        self._crown = None

    @classmethod
    def make(cls, seed, cfg=WorldConfig()):
        rng = np.random.default_rng(seed)
        n = cfg.size
        coarse = max(round(n * cfg.cell_m / cfg.patch_m), 1)
        scales = (coarse, 2 * coarse, 4 * coarse)
        height, moisture, soil = (_smooth_noise(rng, n, scales) for _ in range(3))
        biome = np.full((n, n), GRASS, np.int8)
        land = height >= cfg.water_frac
        m = np.where(land, moisture, np.nan)
        lo, hi = np.nanquantile(m, [cfg.desert_frac, 1 - cfg.forest_frac])
        biome[land & (moisture < lo)] = DESERT
        biome[land & (moisture >= hi)] = FOREST
        biome[~land] = WATER
        fertility = 0.5 + 0.5 * soil
        capacity = cfg.plant_kj_m2 * cfg.cell_m ** 2 * fertility * np.asarray(cfg.capacity)[biome]
        plants = capacity * cfg.start_fill
        root_capacity = capacity * np.asarray(cfg.root_ratio)[biome]
        roots = root_capacity * cfg.start_fill
        # (7a) the fruit trees: their own random stream, so the land and the grass are as they were
        trng = np.random.default_rng([seed, 7])
        ha = (cfg.cell_m ** 2) / 10_000
        trees = np.argwhere(trng.random((n, n)) < np.asarray(cfg.fruit_trees_ha)[biome] * ha)
        return cls(cfg, biome, capacity, plants, roots, root_capacity,
                   ledger=Ledger(start=float(plants.sum() + roots.sum())), trees=trees,
                   rng=np.random.default_rng([seed, 9]))

    def total_plants(self):
        return float(self.plants.sum() + self.roots.sum() + self.dead.sum() + self.fruit.sum())

    def food(self):
        """(n, n) kJ of plant a grazer could find on each cell: green shoots and standing dead grass."""
        return self.plants + self.dead

    def season(self):
        """0 at midwinter (day 0 of each year), 1 at midsummer."""
        return 0.5 - 0.5 * np.cos(2 * np.pi * self.day / self.cfg.year_days)

    def temperature(self):
        """(4e) degrees C, coldest at midwinter, warmest at midsummer."""
        return self.cfg.temp_winter_c + (self.cfg.temp_summer_c - self.cfg.temp_winter_c) * self.season()

    def crown(self):
        """(7a) Flat cell indices under each tree's crown, and each cell's share of its tree's fall."""
        if self._crown is None:
            cfg, n = self.cfg, self.cfg.size
            r = int(np.ceil(cfg.crown_m / cfg.cell_m))
            off = np.array([(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1)
                            if (a * a + b * b) * cfg.cell_m ** 2 <= cfg.crown_m ** 2])
            cells, share = [], []
            for t in self.trees:
                c = t + off
                c = c[((c >= 0) & (c < n)).all(1)]
                c = c[self.biome[c[:, 0], c[:, 1]] != WATER]
                cells.append(c[:, 0] * n + c[:, 1])
                share.append(np.full(len(c), 1.0 / len(c)))
            self._crown = (np.concatenate(cells) if cells else np.zeros(0, int),
                           np.concatenate(share) if share else np.zeros(0))
        return self._crown

    def fruit_fall(self, day, dt):
        """(7a) kJ a tree drops between `day` and `day + dt`: its crop over the season as a sine bump."""
        a, b = self.cfg.fruit_days
        y = day % self.cfg.year_days
        if not a <= y < b:
            return 0.0
        return self.cfg.fruit_kj * np.pi / (2 * (b - a)) * np.sin(np.pi * (y - a) / (b - a)) * dt

    def belt(self):
        """(9a) Where the rainy front is, (x, y) metres: on a circle round the map's middle, belt_per_year loops."""
        cfg = self.cfg
        L = cfg.size * cfg.cell_m
        a = 2 * np.pi * cfg.belt_per_year * self.day / cfg.year_days
        return L / 2 + cfg.belt_loop * L * np.cos(a), L / 2 + cfg.belt_loop * L * np.sin(a)

    def weather(self, dt):
        """(9a) The soil dries; storms fall (most in the belt) and wet it. -> how wet each cell grows, 0..1."""
        cfg, n = self.cfg, self.cfg.size
        L = n * cfg.cell_m
        self.wet *= np.exp(-dt / cfg.dry_days)
        k = self.rng.poisson(cfg.storms_ha_day * L * L / 10_000 * dt)
        if k:
            bx, by = self.belt()
            front = self.rng.random(k) < cfg.belt_share
            x = np.where(front, self.rng.normal(bx, cfg.belt_m, k), self.rng.uniform(0, L, k))
            y = np.where(front, self.rng.normal(by, cfg.belt_m, k), self.rng.uniform(0, L, k))
            c = (np.arange(n) + 0.5) * cfg.cell_m
            for cx, cy in zip(x, y):
                gx = np.exp(-((c - cx) / cfg.storm_m) ** 2 / 2)
                gy = np.exp(-((c - cy) / cfg.storm_m) ** 2 / 2)
                self.wet += cfg.storm_rain * np.outer(gx, gy)
            np.minimum(self.wet, 1.0, out=self.wet)
        return np.minimum(self.wet / cfg.wet_full, 1.0)

    def step(self):
        cfg, B, K, R, RK = self.cfg, self.plants, self.capacity, self.roots, self.root_capacity
        s, dt = self.season(), cfg.dt_days
        g = self.weather(dt) if cfg.rain else 1.0   # (9a) growth only where the soil is wet
        room = np.divide(K - B, K, out=np.zeros_like(K), where=K > 0)
        root_room = np.divide(RK - R, RK, out=np.zeros_like(RK), where=RK > 0)
        p = np.pad(B, 1)
        neighbours = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
        # Sunlight caught by the shoots (and seed from the neighbours), while either pool has room.
        both_room = np.maximum(room, root_room)
        grow = g * s * dt * (cfg.growth_per_day * B + cfg.seed_per_day * neighbours) * both_room
        # Split it between the pools in the root:shoot ratio, each up to its capacity; what one can't take goes to the other.
        to_root = np.divide(RK, K + RK, out=np.zeros_like(K), where=K + RK > 0) * grow
        to_root = np.minimum(to_root, RK - R)
        to_shoot = np.minimum(grow - to_root, K - B)
        to_root = np.minimum(grow - to_shoot, RK - R)
        grow = to_shoot + to_root
        # Roots push up new shoots where the shoots have room (spring, or after grazing); energy moves, none is made.
        up = np.minimum(g * s * dt * cfg.regrow_per_day * R * room, K - B - to_shoot)
        die = (1 - s) * dt * cfg.dieback_per_day * B          # shoots dying back stand as dead grass
        if cfg.rain:   # (9a) and in a drought green grass browns
            die = np.minimum(die + (1 - g) * dt * cfg.drought_per_day * B, B)
        root_die = (2 - s) / 1.5 * dt * cfg.root_loss_per_day * R
        rot = self.dead * (1 - np.exp(-cfg.dead_decay_per_day * dt))
        self.plants = B + to_shoot + up - die
        self.roots = R + to_root - up - root_die
        self.dead = self.dead + die - rot
        self.ledger.sun_in += float(grow.sum())
        self.ledger.dieback += float(root_die.sum() + rot.sum())
        # (7a) fruit falls under the trees in its season, and rots on the ground (fast when warm)
        if len(self.trees):
            drop = self.fruit_fall(self.day, dt)
            if drop > 0:
                cells, share = self.crown()
                np.add.at(self.fruit.reshape(-1), cells, drop * share)
                self.ledger.sun_in += drop * len(self.trees)
            rate = cfg.fruit_rot_per_day * cfg.rot_q10 ** ((self.temperature() - cfg.temp_summer_c) / 10)
            gone = self.fruit * (1 - np.exp(-rate * dt))
            self.fruit -= gone
            self.ledger.dieback += float(gone.sum())
        self.day += dt

    def graze(self, cells, want, split=False):
        """Eaters at flat cell indices `cells` each try to eat `want` kJ; returns what each got (with `split`: the
        green and the dead grass apart, (got_green, got_dead)).

        Only shoots above the stubble can be eaten, then (3h) the standing dead grass. Eaters sharing a cell that
        can't feed them all split what is there in proportion to what they wanted.
        """
        cells, want = np.asarray(cells), np.asarray(want, float)
        flat = self.plants.reshape(-1)
        reach = np.maximum(flat - self.cfg.stubble * self.capacity.reshape(-1), 0.0)
        green = self._take(flat, reach, cells, want)
        dead = self.dead.reshape(-1)
        got_dead = self._take(dead, dead.copy(), cells, want - green)
        self.ledger.eaten += float(green.sum() + got_dead.sum())
        return (green, got_dead) if split else green + got_dead

    def eat_fruit(self, cells, want):
        """(7a) Eaters at `cells` take up to `want` kJ of fallen fruit each (shared pro rata); returns what each got."""
        flat = self.fruit.reshape(-1)
        got = self._take(flat, flat.copy(), np.asarray(cells), np.asarray(want, float))
        self.ledger.eaten += float(got.sum())
        return got

    @staticmethod
    def _take(pool, reach, cells, want):
        """Take up to `reach` from `pool` (flat, in place) for eaters asking `want` on `cells`, shared pro rata."""
        asked = np.zeros(pool.size)
        np.add.at(asked, cells, want)
        share = np.divide(np.minimum(reach, asked), asked, out=np.zeros_like(asked), where=asked > 0)
        pool -= asked * share
        np.maximum(pool, 0.0, out=pool)
        return want * share[cells]

    def error(self):
        """Books minus reality, as a share of all energy that entered the world (should stay at rounding level)."""
        book = self.ledger
        return (book.expected_plants() - self.total_plants()) / max(book.start + book.sun_in, 1.0)
