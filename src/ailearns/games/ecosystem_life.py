"""The ecosystem, step 2: plant-eaters with bodies, tiny random brains, and nothing telling them what to do.

There is no reward. A creature that eats enough grows, splits off children (mutated
copies of itself) and passes its genes on; one that doesn't starves. Every rule here is
physics or physiology, from real scaling laws where we have one:

- Resting cost: Kleiber's law, 293 kJ/day x mass^0.75 (mass in kg).
- Walking cost: Taylor et al. (1982), 10.7 x mass^0.684 J per metre.
- Top speed: what ~10x resting power (the aerobic scope) can pay for at that cost per metre.
- Growth: West et al. (2001), dm/dt = a m^0.75 (1 - (m/M)^0.25), paid for from the energy reserve.
- Lifespan ~ mass^0.25; body tissue holds ~7 MJ/kg.
- Fat: how much a creature can store (kg of fat per kg of lean body) is a gene. Fat weighs what it weighs: walking is
  paid on lean + fat, while top speed comes from the muscle (lean) power, so a fat creature is slower and pays more
  per metre. Resting cost is on lean mass (fat tissue costs almost nothing to keep; real metabolic rate follows
  fat-free mass). Mammals run ~5-20% fat; hibernators fatten to 30-50%.
- Diet is a gene from 0 (plant-eater) to 1 (meat-eater). Guts trade off: a plant-eater's keeps ~50% of plant
  energy, a meat-eater's ~85% of meat. Plants fall with diet^2 (4d: only a plant-eater's gut ferments fibre well,
  but an omnivore's does fairly: pigs on forage keep ~0.3-0.45, so at diet 0.5 ~0.38; a cat's gut none; before 4d
  the fall was linear, 0.25 at diet 0.5, and diet never drifted up); (4c)
  meat is good food for any gut: a plant-eater keeps ~60% of it (cattle digest the protein in fishmeal or
  meat-and-bone meal well; deer and hippos eat carrion), rising to 85% for a meat-eater. What a grazer lacks is a
  way to get meat, not to digest it. (Before 4c: 85% x diet, so a grazer kept ~4% of a carcass and thousands of
  carcasses a month rotted uneaten.) The rest leaves as dung.
- Meat: anything that dies (starved, old or killed) leaves its body and fat as a carcass on its cell, which rots
  (energy lost to decay; 4e: fast when warm, slow when cold, Q10 2.5: ~20%/day at midsummer, ~3% at midwinter) and can be eaten by anyone standing on it. An attack starts a fight with the nearest
  creature straight ahead within reach, and a fight goes both ways: each second it may be settled, and one of the
  two dies, the attacker winning with mass^2 / (mass^2 + target^2) (50/50 against an equal, 80% against half its
  size). Each second of attack costs all-out effort.
  Nothing says who should hunt whom.
- (Step 4b) A nose: for the creature straight ahead (the one an attack would hit), how much it smells like its own
  kind (1 - the gap between their diet genes) and whether it smells like family (its parent, child or sibling), as
  real animals know kin and species by smell. Information only: what a brain does with it is up to evolution. (Before
  it, a creature saw only distance and size, and couldn't tell its own newborn from a small grazer: 49-56% of the
  meat-eaters killed by meat-eaters were the killer's own child, ep17 4b.)
- (Step 5a) A voice: a creature can call (a sound with no meaning; calling costs `call_bmr` x its resting rate while
  it lasts) and hears the calls around it: sound spreads through the air on a coarse grid (`sound_cell_m` cells,
  softened twice by their neighbours: it carries a few tens of metres, well past the eyes) and each creature senses
  how loud it is where it stands and whether it's louder on its left or its right (sampled `ear_m` to each side, a
  stand-in for real ears' timing and level cues). It hears its own call too. Whatever a call comes to mean is up to
  evolution.
- (Step 6a) Sex: a grown creature that's ready to breed (output 3) breeds only with the nearest creature within reach
  (`reach_m`, any direction: mating takes contact, not facing), if that one is grown and ready too. No sexes (as snails and earthworms). Each parent
  pays half the child's cost; every gene and every brain weight comes from one parent or the other at random, then
  mutates as before; the child records both parents (the nose's family sense counts either). `sexual=False`: the
  old way, a creature splits on its own.
- (Step 8g) Only alike kinds breed together: a pair whose diet genes differ by more than `mate_gap` has no child
  (as different species can't interbreed). Each just has no mate, so it may breed alone (6b).
- (Step 6b) Breeding alone when no mate comes (parthenogenesis, as aphids, water fleas, some lizards and sharks):
  a gene `alone` (0..1). A ready creature with no ready partner in reach splits alone with a chance of `alone` per
  `alone_wait_days` (alone = 1: after ~a day of waiting; 0: never, sex only). A solo child costs the full price and
  is a copy plus mutation; a sexual one costs half and mixes two. Evolution sets the gene: the twofold cost of sex
  against the mixing.
- (Step 7b) Fruit (the world's fruit trees, 7a): eaten first when there's any underfoot (the richest food); sugar
  that most guts handle (a grazer keeps 80%, a meat-eater 50%: our estimates). Seen by the eyes (fruit underfoot and
  in each eye's cell), and smelt from afar: scent drifts from fallen fruit on the same coarse grid as sound, a little
  further (three softenings), and each creature senses how strong and from which side.
- (Step 11a, `gut=True`) A gut that fits the body: grass is cell contents (sugars, protein: any gut takes ~90%) and
  fibre (cell walls), which only fermentation breaks down, at a first-order rate, for as long as the food stays in
  the gut. Gut capacity grows with mass^1 and need with mass^0.75, so food stays ~mass^0.25 as long (Demment & Van
  Soest 1985): a big grazer ferments most of its fibre, a mouse passes it nearly whole. Energy kept = (1 - diet^2)
  x ME/DE x (contents + fibre digested). At 2 kg (the middle of the first creatures' sizes) this is ~0.49, about
  the flat 0.5 before; 0.04 kg keeps ~0.38, 8 kg ~0.53. It's why no real grass-eater is smaller than a vole.
- (Step 11b, `cold=True`) Keeping warm: a body at 37 C loses heat to the air at its thermal conductance C x (37 C -
  air), C = 1.02 ml O2 / (g h C) x mass_g^-0.505 per gram (Herreid & Kessel 1967; ~20.1 J per ml O2), so in total
  ~mass^0.5: per kg, small bodies lose far more. Resting metabolism is heat already; only the shortfall is extra
  cost. A 40 g mouse at 0 C pays ~4.3x its resting rate (real mice: ~4-5x), at 20 C ~2x; a 2 kg body needs nothing
  extra above ~14 C (a rabbit's lower critical temperature is ~15-20 C). No nests, burrows or huddling: everyone
  pays by its own (lean) size, carried young too. (11d, `cold_breath=True`) That heat costs only the water breathed
  out with it (~0.03 ml/kJ), not the 0.2 ml/kJ of 9b: water is mostly lost shedding heat, which a cold body doesn't.
- (Step 5b) Memory: two numbers each brain writes every step (outputs 6-7, squashed to -1..1) and reads back as two
  senses the next step. Nothing says what to keep in them; a newborn's start at 0.

The clock: slow things are time-lapsed, fast things run in real time. A sim day lasts `day_s` sim seconds (60: a
day is a minute), and plants, metabolism, growth and ageing run on that day. Moving, chasing and fighting run at real
speed and cost what they really cost per metre and per second of effort. So a creature covers only ~100 m in a day:
it's limited by time rather than by the energy of walking, as most real foragers are. (First try, ep17 3e: every
metre billed as the 1,440 real metres it "stood for" in a one-minute day made a 5 m chase cost like 7 km of running,
and even scripted hunters starved in under a day.)

Energy books: plants + creatures (reserve + body) + carcasses = start + sun - die-back - heat - dung - decay. The world's books
cover the plants; `Life.error()` checks the whole lot.
"""

from dataclasses import dataclass

import numpy as np

from ailearns.games.ecosystem import WATER, World
from ailearns.games.ecosystem_space import nearest_by_sector

N_IN, N_HID, N_OUT = 33, 8, 8   # (9b) sense 32: thirst; (4b) senses 20-21: the nose; (5a) senses 22-23 + output 5: the voice; (5b) senses
                                # 24-25 + outputs 6-7: memory; (7b) senses 26-31: fruit (underfoot, 3 eyes, scent)
SECTORS_DEG = (-60.0, -20.0, 20.0, 60.0)   # what each eye sees of other creatures: left, ahead, right
N_W = N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT


try:   # (ep17 speed) the brains, compiled; the numpy version in `think` is the fallback
    import numba as _nb

    @_nb.njit(cache=True)
    def _think_nb(w, idx, x, n_in, n_hid, n_out):
        n = len(idx)
        out = np.empty((n, n_out))
        h = np.empty(n_hid)
        ob1 = n_in * n_hid
        ow2 = ob1 + n_hid
        ob2 = ow2 + n_hid * n_out
        for r in range(n):
            k = idx[r]
            for j in range(n_hid):
                s = w[k, ob1 + j]
                for q in range(n_in):
                    s += x[r, q] * w[k, q * n_hid + j]
                h[j] = np.tanh(s)
            for o in range(n_out):
                s = w[k, ob2 + o]
                for j in range(n_hid):
                    s += h[j] * w[k, ow2 + j * n_out + o]
                out[r, o] = s
        return out
except ImportError:  # pragma: no cover
    _think_nb = None
DAY_REAL_S = 86_400.0


@dataclass(frozen=True)
class LifeConfig:
    day_s: float = 60.0              # sim seconds in a sim day
    dt: float = 1.0                  # sim seconds per tick
    max_creatures: int = 6000        # arrays' first size; (3i) they double when full, so it limits nothing
    ceiling: int = 100_000           # ... up to this (memory: ~1.7 kB a creature); births past it are logged (`full`)
    # Physiology (real scaling laws; the constants are textbook values for mammals)
    bmr_kj_day: float = 293.0        # Kleiber: resting kJ/day per kg^0.75
    cot_j_m: float = 10.7            # Taylor: J per metre per kg^0.684
    aerobic_scope: float = 10.0      # sustained power / resting power
    body_kj_kg: float = 7000.0       # energy in body tissue
    build_overhead: float = 0.3      # extra cost of making tissue (lost as heat)
    fat_kj_kg: float = 37000.0
    digest_plant: float = 0.5        # share of plant energy a pure plant-eater keeps (x (1 - diet^plant_curve))
    plant_curve: float = 2.0         # (4d) 1: linear (an omnivore keeps half as much); 2: omnivores keep ~3/4
    # (11a) a gut that fits the body (`gut`): see the module notes. Grasses are ~55-65% cell wall (NDF) by dry matter
    # (forage tables), the contents ~90% digestible by any gut; grass NDF ferments at ~3-8%/h with ~75% of it
    # digestible at all (lignin); mean retention time ~60 h in a ~600 kg cow, x mass^0.25 (Demment & Van Soest 1985;
    # later data scale more weakly, so this is the strong end); metabolisable / digestible energy ~0.82 (ARC).
    # (11b) keeping warm (`cold`): see the module notes; the air's temperature is the world's (0 C midwinter, 20 C
    # midsummer)
    cold: bool = False
    body_c: float = 37.0
    conduct_ml_o2: float = 1.02       # Herreid & Kessel (1967): ml O2 per g per hour per C at 1 g ...
    conduct_exp: float = -0.505       # ... x mass_g^this
    j_per_ml_o2: float = 20.1
    # (11d, `cold_breath`) the heat made to keep warm costs only the water breathed out with it, not the full
    # water_per_kj (most water turnover is evaporation to shed heat, which a cold body doesn't do): rodents lose
    # ~0.5-0.7 mg of water per ml O2 breathed (our reading of the respiratory-water-loss literature), ~20 J per ml O2
    cold_breath: bool = False
    breath_water_per_kj: float = 0.03e-3
    gut: bool = False
    gut_contents: float = 0.4        # share of grass energy in cell contents
    gut_contents_dig: float = 0.9    # ... digested by any gut
    gut_fibre_max: float = 0.75      # share of the fibre that can be fermented at all
    gut_fibre_k_h: float = 0.05      # fermentation rate, per hour in the gut
    gut_mrt_h: float = 60.0          # mean retention time at gut_ref_kg, real hours
    gut_ref_kg: float = 600.0
    gut_me_de: float = 0.82
    digest_meat: float = 0.85        # share of meat energy a pure meat-eater keeps
    digest_meat_grazer: float = 0.6  # (4c) ... and a pure plant-eater (our estimate from cattle fed animal protein)
    meat_rot_per_day: float = 0.2    # share of a carcass lost to rot each day at summer warmth (our estimate: days)
    rot_q10: float = 2.5             # (4e) rot slows this many times per 10 C colder (decomposition: ~2-3)
    reach_m: float = 1.5             # how close an attack has to be
    sexual: bool = True              # (6a) a partner is needed to breed (False: split alone, as before 6a)
    mate_gap: float = 0.25           # (8g) two can have a child only if their diet genes differ by at most this
                                     #      (unlike kinds can't interbreed, as species; our estimate)
    alone_wait_days: float = 1.0     # (6b) alone = 1 splits alone after ~this long ready without a mate
    alone: tuple = (0.0, 1.0)        # (6b) the alone gene at the start (uniform)
    sigma_alone: float = 0.03        # ... and its mutation
    # (5a) the voice: calling costs this many times the resting rate on top (real callers: ~2-10x; our estimate, low)
    call_bmr: float = 2.0
    sound_cell_m: float = 8.0        # the air's grid for sound (and, 7b, scent)
    digest_fruit: float = 0.8        # (7b) share of fruit energy a grazer keeps ...
    digest_fruit_hunter: float = 0.5 # ... and a meat-eater (our estimates: sugar, easy for most guts)
    fruit_sense_kj: float = 2000.0   # (7b) a fruit pile this big reads ~0.63 to the eyes and the scent
    ear_m: float = 4.0               # where each "ear" samples it, either side
    # (8c) a meat-eater is built to kill (teeth, claws, a grip): it fights as if its mass were x (1 + weapon x diet),
    # so a pure meat-eater fights like an animal twice its size (owner's pick); its cost is the gut (plants digest worse)
    weapon: float = 1.0
    # (8f) milk: young under wean_frac of adult mass drink from their mother (the parent that bore them) while within
    # reach, up to what they could eat, from her reserve above milk_floor of full; lactation keeps ~70% of the energy
    # (our estimates: small mammals wean at ~30-40% of adult mass)
    milk: bool = True
    wean_frac: float = 0.4
    milk_eff: float = 0.7
    milk_floor: float = 0.3
    # (8h) carried young (as marsupials, bats, many primates): young under wean_frac ride on their living mother: at
    # her place (so always in reach of milk), not walking, not attacking, unseen by others; she carries their weight
    # (slower, dearer walking). If she dies they're on their own.
    carry: bool = True
    # (9b) thirst (our estimates): every kJ spent costs water_per_kj litres (mammals turn over ~1 ml per kcal);
    # food brings water: green grass ~75% water, dead grass ~10%, meat ~70%, fruit ~85%, milk ~88% (per kJ below).
    # Drinking (eating while at a lake's shore, or on a puddle: soil wetter than puddle_wet) fills you up. Dead of
    # thirst at a loss of dry_max_l_kg litres per kg of body (~15%).
    thirst: bool = False
    water_per_kj: float = 0.2e-3
    water_green: float = 0.18e-3
    water_dead: float = 0.007e-3
    water_meat: float = 0.1e-3
    water_fruit: float = 0.39e-3
    water_milk: float = 0.3e-3
    dry_max_l_kg: float = 0.15
    puddle_wet: float = 0.6
    fight_per_s: float = 0.5         # chance a second that a fight is settled, one of the two dying (our estimate)
    growth_a: float = 1.0            # West: g^0.25 per day
    life_days: float = 300.0         # lifespan = life_days x (mass in g)^0.25 / 10  (our estimate)
    # Our estimates, to be measured
    eat_bmr_day: float = 12.0        # most a creature can eat in a day, in multiples of its resting cost
    turn_rad_s: float = 2.0
    sight_m: float = 6.0             # how far ahead it looks (fixed in step 2)
    sight_deg: float = 40.0          # left and right eyes look this far off straight ahead
    grow_above: float = 0.3          # grows only while its reserve is over this share of full
    # Genes at the start and mutation
    mass_kg: tuple = (0.5, 8.0)      # adult mass, log-uniform at the start
    birth_frac: tuple = (0.05, 0.3)  # newborn mass / adult mass
    fat_frac: tuple = (0.05, 0.4)    # most fat it can store, kg per kg of lean body (gene; uniform at the start)
    diet: tuple = (0.0, 1.0)         # 0 plant-eater .. 1 meat-eater (gene; uniform at the start)
    sigma_diet: float = 0.03         # diet mutates by adding this much noise
    start_fill: float = 0.5          # the first creatures start at this share of a full reserve
    sigma_w: float = 0.1
    sigma_gene: float = 0.05


def plant_keep(cfg, diet, mass, exp=np.exp):
    """Share of plant energy a creature keeps: (4d) falls with diet^plant_curve; (11a, `cfg.gut`) and with a small
    gut, which passes fibre before it ferments. `exp`: np.exp or jnp.exp (the JAX port shares this)."""
    curve = 1 - diet ** cfg.plant_curve
    if not cfg.gut:
        return cfg.digest_plant * curve
    hours = cfg.gut_mrt_h * (mass / cfg.gut_ref_kg) ** 0.25
    fibre = cfg.gut_fibre_max * (1 - exp(-cfg.gut_fibre_k_h * hours))
    return curve * cfg.gut_me_de * (cfg.gut_contents * cfg.gut_contents_dig + (1 - cfg.gut_contents) * fibre)


def keep_warm_s(cfg, mass, air_c, rest_s, maximum=np.maximum):
    """(11b) Extra kJ a sim second to stay at body temperature in air at `air_c`, when resting already makes `rest_s`
    kJ a sim second (time-lapsed like the resting cost: a day's loss over a sim day). 0 without `cfg.cold`."""
    if not cfg.cold:
        return 0.0 * rest_s
    g = mass * 1000
    c_day = cfg.conduct_ml_o2 * g ** (1 + cfg.conduct_exp) * cfg.j_per_ml_o2 * 24 / 1000   # kJ per day per C
    return maximum(c_day * (cfg.body_c - air_c) / cfg.day_s - rest_s, 0.0)


@dataclass
class Books:
    start: float = 0.0       # creature energy when they were placed
    heat: float = 0.0        # resting, walking, building tissue
    dung: float = 0.0        # undigested food
    decay: float = 0.0       # carcasses rotting
    kills: int = 0
    births: int = 0
    births_alone: int = 0    # (6b) of the births, split alone (no mate)
    deaths: int = 0
    full: int = 0            # births lost because the arrays were full


class Life:
    """Creatures in a `World`. Arrays of size max_creatures, `alive` marks the used slots."""

    def __init__(self, world: World, n: int, seed: int, cfg: LifeConfig = LifeConfig()):
        self.world, self.cfg = world, cfg
        self.rng = np.random.default_rng(seed)
        self.real_s_per_s = DAY_REAL_S / cfg.day_s
        self.t = 0.0
        self.plant_every = round(world.cfg.dt_days * cfg.day_s / cfg.dt)
        self.ticks = 0
        m = cfg.max_creatures
        self.alive = np.zeros(m, bool)
        self.pos = np.zeros((m, 2))
        self.heading = np.zeros(m)
        self.mass = np.zeros(m)          # current kg
        self.adult = np.zeros(m)         # adult mass gene, kg
        self.birth = np.zeros(m)         # newborn share gene
        self.fat = np.zeros(m)           # fat capacity gene, kg fat per kg lean
        self.diet = np.zeros(m)          # diet gene, 0 plant-eater .. 1 meat-eater
        self.meat = np.zeros(world.cfg.size ** 2)   # kJ of carcass on each cell
        self.reserve = np.zeros(m)       # kJ
        self.age = np.zeros(m)           # days
        self.gen = np.zeros(m, int)
        self.uid = np.full(m, -1)        # every creature ever gets its own id (for lineages and the family tree)
        self.parent = np.full(m, -1)     # its parent's id (-1: one of the first creatures)
        self.next_uid = 0
        self.w = np.zeros((m, N_W))
        self.calling = np.zeros(m, bool)  # (5a) called last step (heard this step)
        self.mem = np.zeros((m, 2))       # (5b) what each brain wrote to its memory last step
        self.parent2 = np.full(m, -1)     # (6a) the other parent's id (-1: none, or split alone)
        self.alone = np.zeros(m)          # (6b) gene: how readily it breeds alone with no mate about
        self.dry = np.zeros(m)            # (9b) litres of water short of full
        self.books = Books()
        # (step 3g) an observer, called each step with what happened (attacks, fights, kills, meals) before the dead
        # are removed; it only looks (scripts/eco_kills.py)
        self.trace = None
        self.add(n)

    # --- physiology -------------------------------------------------------------------------------------------------
    def bmr_s(self, mass):
        """Resting kJ per sim second."""
        return self.cfg.bmr_kj_day * mass ** 0.75 / self.cfg.day_s

    def walk_kj_m(self, carried):
        """kJ per metre walked carrying `carried` kg (Taylor; real cost, movement runs in real time)."""
        return self.cfg.cot_j_m * carried ** 0.684 / 1000

    def sprint_kj_s(self, lean):
        """kJ per second of all-out effort: aerobic_scope x real resting power."""
        return self.cfg.aerobic_scope * self.bmr_s(lean) / self.real_s_per_s

    def top_speed(self, lean, carried=None):
        """m/s that all-out effort (from the lean body) pays for, carrying `carried` kg."""
        return self.sprint_kj_s(lean) / self.walk_kj_m(lean if carried is None else carried)

    def reserve_cap(self, lean, fat):
        return fat * self.cfg.fat_kj_kg * lean

    def carried(self, i):
        """Lean body plus the fat it's carrying, kg."""
        return self.mass[i] + self.reserve[i] / self.cfg.fat_kj_kg

    def lifespan(self, adult):
        return self.cfg.life_days * (adult * 1000) ** 0.25 / 10

    def energy(self):
        a = self.alive
        return float((self.reserve[a] + self.cfg.body_kj_kg * self.mass[a]).sum())

    # --- setup ------------------------------------------------------------------------------------------------------
    def _land_points(self, k):
        w = self.world
        cells = np.flatnonzero(w.biome.reshape(-1) != WATER)
        c = self.rng.choice(cells, k)
        n = w.cfg.size
        return (np.stack([c // n, c % n], 1) + self.rng.random((k, 2))) * w.cfg.cell_m

    GENES = ("adult", "birth", "fat", "alone", "diet", "w")

    def genes(self, idx):
        """(8) The genes and brain weights of creatures `idx`, as a dict of arrays (to save, or to copy with `add`)."""
        return {g: getattr(self, g)[idx].copy() for g in self.GENES}

    def add(self, k, diet=None, like=None, p=None):
        """Drop k new random creatures onto random land (the first ones, or later an "invasive species").

        `diet` (lo, hi) overrides the starting diet range. Their energy is booked as entering the world.
        (8) `like`: a `genes()` dict; each newcomer copies one of those creatures (chosen with probabilities `p`),
        mutated as a child would be, arriving full grown. Returns the new creatures' indices.
        """
        cfg = self.cfg
        self._grow(k)
        idx = np.flatnonzero(~self.alive)[:k]
        k = len(idx)
        lo, hi = np.log(cfg.mass_kg)
        self.alive[idx] = True
        self.pos[idx] = self._land_points(k)
        self.heading[idx] = self.rng.uniform(-np.pi, np.pi, k)
        self.adult[idx] = np.exp(self.rng.uniform(lo, hi, k))
        self.mass[idx] = self.adult[idx]
        self.birth[idx] = self.rng.uniform(*cfg.birth_frac, k)
        self.fat[idx] = self.rng.uniform(*cfg.fat_frac, k)
        self.alone[idx] = self.rng.uniform(*cfg.alone, k)
        self.diet[idx] = self.rng.uniform(*(cfg.diet if diet is None else diet), k)
        self.gen[idx] = 0
        self.parent[idx] = -1
        self.reserve[idx] = cfg.start_fill * self.reserve_cap(self.mass[idx], self.fat[idx])
        self.age[idx] = 0.0
        self.w[idx] = self.rng.normal(0, 1, (k, N_W))
        self.calling[idx] = False
        self.mem[idx] = 0.0
        self.parent2[idx] = -1
        self.dry[idx] = 0.0
        if like is not None:
            rng, src = self.rng, self.rng.choice(len(like["diet"]), k, p=p)
            e = lambda: np.exp(rng.normal(0, cfg.sigma_gene, k))
            self.adult[idx] = like["adult"][src] * e()
            self.mass[idx] = self.adult[idx]
            self.birth[idx] = np.clip(like["birth"][src] * e(), 0.01, 0.9)
            self.fat[idx] = np.clip(like["fat"][src] * e(), 0.01, 1.0)
            self.alone[idx] = np.clip(like["alone"][src] + rng.normal(0, cfg.sigma_alone, k), 0.0, 1.0)
            self.diet[idx] = np.clip(like["diet"][src] + rng.normal(0, cfg.sigma_diet, k), 0.0, 1.0)
            self.w[idx] = like["w"][src] + rng.normal(0, cfg.sigma_w, (k, N_W))
            self.reserve[idx] = cfg.start_fill * self.reserve_cap(self.mass[idx], self.fat[idx])
        self.uid[idx] = self.next_uid + np.arange(k)
        self.next_uid += k
        self.books.start += float((self.reserve[idx] + cfg.body_kj_kg * self.mass[idx]).sum())
        return idx

    # --- senses and brain -------------------------------------------------------------------------------------------
    def _cell(self, xy):
        """Flat cell index of points, and whether each is on the map."""
        w = self.world
        n, cm = w.cfg.size, w.cfg.cell_m
        rc = np.floor(xy / cm).astype(int)
        inside = ((rc >= 0) & (rc < n)).all(-1)
        rc = np.clip(rc, 0, n - 1)
        return rc[..., 0] * n + rc[..., 1], inside

    def senses(self, i):
        cfg, w = self.cfg, self.world
        per_cell = w.cfg.plant_kj_m2 * w.cfg.cell_m ** 2
        plants, biome = w.food().reshape(-1), w.biome.reshape(-1)   # (3h) green and standing dead
        here, _ = self._cell(self.pos[i])
        ang = self.heading[i][:, None] + np.radians([-cfg.sight_deg, 0.0, cfg.sight_deg])
        eye = self.pos[i][:, None] + cfg.sight_m * np.stack([np.cos(ang), np.sin(ang)], -1)
        cells, inside = self._cell(eye)
        blocked = ~inside | (biome[cells] == WATER)
        x = np.empty((len(i), N_IN))
        x[:, 0] = plants[here] / per_cell
        x[:, 1:4] = np.where(blocked, 0.0, plants[cells] / per_cell)
        x[:, 4:7] = blocked
        x[:, 7] = self.reserve[i] / self.reserve_cap(self.mass[i], self.fat[i])
        x[:, 8] = self.mass[i] / self.adult[i]
        x[:, 9] = 1.0
        x[:, 10] = np.minimum(self.meat[here] / (5 * cfg.body_kj_kg), 1.0)   # carcass under it, up to ~5 kg of body
        x[:, 11:17] = 0.0
        x[:, 17:20] = np.where(blocked, 0.0, np.minimum(self.meat[cells] / (5 * cfg.body_kj_kg), 1.0))  # carcasses seen
        x[:, 20:22] = 0.0        # the nose (4b): nobody ahead, nothing to smell
        x[:, 22:24] = self.hear(i)   # (5a) the voice: how loud, and louder left (-) or right (+)
        x[:, 24:26] = self.mem[i]    # (5b) memory: what it wrote last step
        fruit = w.fruit.reshape(-1)   # (7b) fruit underfoot, in each eye's cell, and its scent
        x[:, 26] = 1 - np.exp(-fruit[here] / cfg.fruit_sense_kj)
        x[:, 27:30] = np.where(blocked, 0.0, 1 - np.exp(-fruit[cells] / cfg.fruit_sense_kj))
        x[:, 30:32] = self.smell_fruit(i)
        x[:, 32] = self.dry[i] / (cfg.dry_max_l_kg * self.mass[i])   # (9b) thirst, 0 .. 1 (dead)
        # Other creatures: the nearest in each eye's sector, how close (1 = touching, 0 = at sight range) and how big
        # against itself (log of the mass ratio, /3).
        self._ahead = np.full(len(i), -1)       # nearest creature straight ahead (index into i), for attacks
        self._ahead_d = np.full(len(i), np.inf)
        self._near = np.full(len(i), -1)        # (6a) nearest creature in any direction (index into i), for mating
        self._near_d = np.full(len(i), np.inf)
        # (ep17 speed) one compiled pass finds the nearest in any direction and in each eye's sector
        # (8h) carried young are in the fur or the pouch: they don't see others and nobody sees them
        self._ride = self._riders(i)
        free = np.ones(len(i), bool)
        free[self._ride[0]] = False
        f = np.flatnonzero(free)
        near, near_d, sec_b, sec_d = nearest_by_sector(self.pos[i][f], self.heading[i][f], cfg.sight_m,
                                                       w.cfg.size * w.cfg.cell_m, SECTORS_DEG)
        self._near[f] = np.where(near >= 0, f[np.maximum(near, 0)], -1)
        self._near_d[f] = near_d
        a, sector = np.nonzero(sec_b >= 0)
        if len(a):
            b, d = f[sec_b[a, sector]], sec_d[a, sector]
            a = f[a]
            carried = self.carried(i)
            x[a, 11 + sector] = 1 - d / cfg.sight_m
            x[a, 14 + sector] = np.clip(np.log(carried[b] / carried[a]), -3, 3) / 3
            mid = sector == 1
            self._ahead[a[mid]], self._ahead_d[a[mid]] = b[mid], d[mid]
            # (4b) the nose, on the creature ahead: own kind (diet genes alike), family (parent, child or sibling)
            a, b = a[mid], b[mid]
            ua, ub, pa, pb = self.uid[i[a]], self.uid[i[b]], self.parent[i[a]], self.parent[i[b]]
            qa, qb = self.parent2[i[a]], self.parent2[i[b]]   # (6a) the other parent
            x[a, 20] = 1 - np.abs(self.diet[i[a]] - self.diet[i[b]])
            x[a, 21] = ((pb == ua) | (qb == ua) | (pa == ub) | (qa == ub)
                        | ((pa == pb) & (pa >= 0)) | ((qa == qb) & (qa >= 0)))
        return x

    def _air(self, xy, amount, passes):
        """(5a, 7b) Something in the air (sound, scent): `amount` added at points `xy` on the coarse grid, softened
        `passes` times by the neighbours ([1, 2, 1] / 4 each way, so it carries a few cells)."""
        cfg, w = self.cfg, self.world
        n = max(int(round(w.cfg.size * w.cfg.cell_m / cfg.sound_cell_m)), 1)
        field = np.zeros((n, n))
        if len(xy):
            c = np.clip((xy / cfg.sound_cell_m).astype(int), 0, n - 1)
            np.add.at(field, (c[:, 0], c[:, 1]), amount)
            for _ in range(passes):
                for ax in (0, 1):
                    p = np.pad(field, [(1, 1) if a == ax else (0, 0) for a in (0, 1)])
                    lo, mid, hi = (np.take(p, range(k, k + n), axis=ax) for k in (0, 1, 2))
                    field = (lo + 2 * mid + hi) / 4
        return field

    def _ears(self, i, field, scale):
        """(5a, 7b) Sampled `ear_m` to each side: (strength squashed to 0..1, (right - left) / (right + left))."""
        cfg = self.cfg
        out = np.zeros((len(i), 2))
        if not field.any():
            return out
        n = field.shape[0]
        side = self.heading[i] + np.pi / 2              # left of the heading
        ear = cfg.ear_m * np.stack([np.cos(side), np.sin(side)], 1)
        at = lambda xy: field[tuple(np.clip((xy / cfg.sound_cell_m).astype(int), 0, n - 1).T)]
        left, right = at(self.pos[i] + ear), at(self.pos[i] - ear)
        out[:, 0] = 1 - np.exp(-(left + right) / 2 / scale)
        out[:, 1] = np.divide(right - left, right + left, out=np.zeros(len(i)), where=right + left > 0)
        return out

    def sound(self):
        """(5a) The air: each creature that called last step adds 1 to its sound cell, softened twice."""
        who = self.alive & self.calling
        return self._air(self.pos[who], 1.0, 2)

    def hear(self, i):
        """(5a) What creatures `i` hear: (loudness squashed to 0..1, (right - left) / (right + left))."""
        return self._ears(i, self.sound(), 1.0)

    def smell_fruit(self, i):
        """(7b) What creatures `i` smell of fallen fruit: (strength 0..1, (right - left) / (right + left)); the scent
        drifts from every fruit cell, softened three times (a little further than a call)."""
        w = self.world
        f = w.fruit.reshape(-1)
        cells = np.flatnonzero(f > 0)
        n = w.cfg.size
        xy = (np.stack([cells // n, cells % n], 1) + 0.5) * w.cfg.cell_m
        return self._ears(i, self._air(xy, f[cells], 3), self.cfg.fruit_sense_kj)

    def think(self, i, x):
        if _think_nb is not None:   # (ep17 speed) compiled: reads the weights in place (no 80 MB copy at 30k)
            return _think_nb(self.w, np.ascontiguousarray(i, np.int64), np.ascontiguousarray(x, float),
                             N_IN, N_HID, N_OUT)
        w = self.w[i]
        a = 0
        w1 = w[:, a:a + N_IN * N_HID].reshape(-1, N_IN, N_HID); a += N_IN * N_HID
        b1 = w[:, a:a + N_HID]; a += N_HID
        w2 = w[:, a:a + N_HID * N_OUT].reshape(-1, N_HID, N_OUT); a += N_HID * N_OUT
        b2 = w[:, a:a + N_OUT]
        h = np.tanh(np.einsum("ni,nih->nh", x, w1) + b1)
        return np.einsum("nh,nho->no", h, w2) + b2

    # --- one tick ---------------------------------------------------------------------------------------------------
    def step(self):
        cfg, w, b = self.cfg, self.world, self.books
        dt = cfg.dt
        day = dt / cfg.day_s
        i = np.flatnonzero(self.alive)
        if len(i):
            x = self.senses(i)
            out = self.think(i, x)
            turn, speed = np.tanh(out[:, 0]), 1 / (1 + np.exp(-out[:, 1]))
            eat, split, attack = out[:, 2] > 0, out[:, 3] > 0, out[:, 4] > 0
            call = out[:, 5] > 0                           # (5a) heard by everyone next step
            m, fat, diet = self.mass[i], self.fat[i], self.diet[i]
            carried = self.carried(i)
            ry, rm = self._ride                            # (8h) carried young and their mothers
            load = carried.copy()
            np.add.at(load, rm, carried[ry])               # a mother carries her young's weight
            speed[ry] = 0.0
            attack[ry] = False

            # Move; water and the map edge stop you (you pay only for the distance you covered).
            self.heading[i] = (self.heading[i] + turn * cfg.turn_rad_s * dt + np.pi) % (2 * np.pi) - np.pi
            v = speed * self.top_speed(m, load)
            step = v[:, None] * dt * np.stack([np.cos(self.heading[i]), np.sin(self.heading[i])], 1)
            cell, inside = self._cell(self.pos[i] + step)
            ok = inside & (w.biome.reshape(-1)[cell] != WATER)
            self.pos[i] += np.where(ok[:, None], step, 0.0)
            walked = np.where(ok, v * dt, 0.0)
            self.pos[i[ry]] = self.pos[i[rm]]               # (8h) the young ride along

            # Attack the nearest creature straight ahead, if it's within reach: a fight that may kill either of you
            # (the bigger one usually wins), at a second of all-out effort.
            tgt, td = self._ahead, self._ahead_d
            hit = attack & (tgt >= 0) & (td < cfg.reach_m)
            killed = np.zeros(len(i), bool)
            att, settled, won = np.zeros(0, int), np.zeros(0, bool), np.zeros(0, bool)
            if hit.any():
                att = np.flatnonzero(hit)
                arm = 1 + cfg.weapon * self.diet
                ca, ct = carried[att] * arm[i[att]], carried[tgt[att]] * arm[i[tgt[att]]]
                settled = self.rng.random(len(att)) < cfg.fight_per_s * dt
                won = self.rng.random(len(att)) < ca ** 2 / (ca ** 2 + ct ** 2)
                killed[tgt[att][settled & won]] = True
                killed[att[settled & ~won]] = True
            effort = np.where(attack, self.sprint_kj_s(m) * dt, 0.0)

            # Pay for resting, walking and attacking.
            rest = self.bmr_s(m) * (1 + cfg.call_bmr * call)
            warm = keep_warm_s(cfg, m, self.world.temperature(), rest)   # (11b) the cold's extra, if any
            cost = (rest + warm) * dt + self.walk_kj_m(load) * walked + effort
            self.reserve[i] -= cost
            b.heat += float(cost.sum())

            # Eat from the cell you're on (a mouth can't eat at full speed), up to a full reserve: carcass first, then
            # plants, each digested as well as the diet allows.
            keep_meat = cfg.digest_meat_grazer + (cfg.digest_meat - cfg.digest_meat_grazer) * diet
            keep_plant = plant_keep(cfg, diet, m)
            room = np.maximum(self.reserve_cap(m, fat) - self.reserve[i], 0.0)
            want = eat * (1 - speed) * cfg.eat_bmr_day * self.bmr_s(m) * dt
            here, _ = self._cell(self.pos[i])
            # (7b) fruit first (the richest), then carcass, then plants
            keep_fruit = cfg.digest_fruit + (cfg.digest_fruit_hunter - cfg.digest_fruit) * diet
            got_f = w.eat_fruit(here, np.minimum(want, room / keep_fruit))
            room -= keep_fruit * got_f
            want = want - got_f
            want_m = np.where(keep_meat > 0.01, np.minimum(want, room / np.maximum(keep_meat, 1e-9)), 0.0)
            got_m = self._take_meat(here, want_m)
            room -= keep_meat * got_m
            want_p = np.where(keep_plant > 0.01, np.minimum(want - got_m, room / np.maximum(keep_plant, 1e-9)), 0.0)
            green, dead = w.graze(here, np.maximum(want_p, 0.0), split=True)
            got_p = keep_plant * (green + w.cfg.dead_quality * dead)    # (3h) dead grass is poorer food
            self.reserve[i] += keep_meat * got_m + got_p + keep_fruit * got_f
            b.dung += float(((1 - keep_meat) * got_m + green + dead - got_p + (1 - keep_fruit) * got_f).sum())
            milk = self._nurse(i, m, fat, killed) if cfg.milk else np.zeros(len(i))
            if cfg.thirst:   # (9b) water out with the energy spent, in with the food and milk; drink to full
                wet_cost = cost - (1 - cfg.breath_water_per_kj / cfg.water_per_kj) * warm * dt if cfg.cold_breath else cost
                self.dry[i] += cfg.water_per_kj * wet_cost   # (11d) the warming heat at the breath's rate only
                self.dry[i] -= (cfg.water_green * green + cfg.water_dead * dead + cfg.water_meat * got_m
                                + cfg.water_fruit * got_f + cfg.water_milk * milk)
                self.drank = eat & self.drinkable()[here]          # (len(i),) who drank this step
                self.dry[i[self.drank]] = 0.0
                np.maximum(self.dry, 0.0, out=self.dry)
            if self.trace is not None:
                self.trace(self, i=i, attack=attack, reach=(tgt >= 0) & (td < cfg.reach_m), att=att, tgt=tgt[att],
                           settled=settled, won=won, killed=killed, meat=keep_meat * got_m, plant=got_p, carcass=got_m,
                           fruit=keep_fruit * got_f,
                           x=x, turn=turn, speed=speed, call=call, mem=np.tanh(out[:, 6:8]),
                           ready=split & (self.mass[i] >= 0.95 * self.adult[i]), milk=milk)

            # Grow towards adult mass while the reserve is comfortable (West et al.), paying for the tissue.
            g = cfg.growth_a * (m * 1000) ** 0.75 * (1 - (m / self.adult[i]) ** 0.25) * day / 1000
            per_kg = cfg.body_kj_kg * (1 + cfg.build_overhead)
            spare = np.maximum(self.reserve[i] - cfg.grow_above * self.reserve_cap(m, fat), 0.0)
            g = np.clip(np.minimum(g, spare / per_kg), 0.0, None)
            self.mass[i] += g
            self.reserve[i] -= g * per_kg
            b.heat += float((g * cfg.body_kj_kg * cfg.build_overhead).sum())

            self.age[i] += day
            self.calling[i] = call
            self.mem[i] = np.tanh(out[:, 6:8])
            ready = split & (self.mass[i] >= 0.95 * self.adult[i]) & ~killed
            if cfg.sexual:   # (6a) each ready one with the nearest, if within reach and ready too; one child a pair
                near, nd = self._near, self._near_d
                cand = np.flatnonzero(ready & (near >= 0) & (nd < cfg.reach_m))
                cand = cand[ready[near[cand]]]
                # (8g) only alike kinds are fertile together; an unlike neighbour is no mate (may breed alone)
                cand = cand[np.abs(self.diet[i[cand]] - self.diet[i[near[cand]]]) <= cfg.mate_gap]
                if len(cand):
                    used, ma, pa = set(), [], []
                    for a in self.rng.permutation(cand):
                        b_ = int(near[a])
                        if a not in used and b_ not in used:
                            used |= {int(a), b_}
                            ma.append(a)
                            pa.append(b_)
                    self._births(i[np.array(ma)], i[np.array(pa)])
                # (6b) ready with no ready mate in reach: alone, with a chance of `alone` a wait of alone_wait_days
                mated = np.zeros(len(i), bool)
                if len(cand):
                    mated[np.array(ma)] = mated[np.array(pa)] = True
                wait = ready & ~mated & ~killed
                if wait.any():
                    p_ = self.alone[i] * dt / (cfg.alone_wait_days * cfg.day_s)
                    solo = wait & (self.rng.random(len(i)) < p_)
                    if solo.any():
                        n0 = b.births
                        self._births(i[solo])
                        b.births_alone += b.births - n0
            else:
                self._births(i[ready])
            self._deaths(i, killed)

        self.ticks += 1
        self.t += dt
        if self.ticks % self.plant_every == 0:
            w = self.world   # (4e) rot runs on the temperature: weeks in the cold, days in the warm
            rate = cfg.meat_rot_per_day * cfg.rot_q10 ** ((w.temperature() - w.cfg.temp_summer_c) / 10)
            rot = self.meat * (1 - np.exp(-rate * w.cfg.dt_days))
            self.meat -= rot
            b.decay += float(rot.sum())
            w.step()

    def parched(self, i):
        """(9b) Dead of thirst: lost more water than the body can spare."""
        return self.dry[i] > self.cfg.dry_max_l_kg * self.mass[i]

    def drinkable(self):
        """(9b) Flat mask of cells where one can drink: land at a lake's shore, or (with rain) a fresh puddle."""
        w = self.world
        if getattr(self, "_shore", None) is None:
            water = w.biome == WATER
            p = np.pad(water, 1)
            near = p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
            self._shore = (near & ~water).reshape(-1)
        if w.cfg.rain:
            return self._shore | (w.wet.reshape(-1) > self.cfg.puddle_wet)
        return self._shore

    def _riders(self, i):
        """(8h) Young (under wean_frac) whose mother is alive: (young, mother), both indexes into i."""
        cfg = self.cfg
        e = np.zeros(0, int)
        if not cfg.carry:
            return e, e
        young = np.flatnonzero((self.mass[i] < cfg.wean_frac * self.adult[i]) & (self.parent[i] >= 0))
        if not len(young):
            return e, e
        order = np.argsort(self.uid[i])
        k = np.minimum(np.searchsorted(self.uid[i][order], self.parent[i][young]), len(i) - 1)
        mom = order[k]
        ok = (self.uid[i][mom] == self.parent[i][young]) & (mom != young)
        return young[ok], mom[ok]

    def _nurse(self, i, m, fat, killed):
        """(8f) Young near their mother drink her milk. -> (len(i),) kJ each young got this step."""
        cfg, dt = self.cfg, self.cfg.dt
        got = np.zeros(len(i))
        young = np.flatnonzero((m < cfg.wean_frac * self.adult[i]) & (self.parent[i] >= 0) & ~killed)
        if not len(young):
            return got
        order = np.argsort(self.uid[i])
        k = np.minimum(np.searchsorted(self.uid[i][order], self.parent[i][young]), len(i) - 1)
        mom = order[k]
        ok = (self.uid[i][mom] == self.parent[i][young]) & ~killed[mom]
        ok &= np.hypot(*(self.pos[i][young] - self.pos[i][mom]).T) < cfg.reach_m
        young, mom = young[ok], mom[ok]
        if not len(young):
            return got
        room = np.maximum(self.reserve_cap(m[young], fat[young]) - self.reserve[i[young]], 0.0)
        ask = np.minimum(cfg.eat_bmr_day * self.bmr_s(m[young]) * dt, room / cfg.milk_eff)
        spare = np.maximum(self.reserve[i] - cfg.milk_floor * self.reserve_cap(m, fat), 0.0)
        asked = np.bincount(mom, ask, len(i))
        give = ask * np.minimum(1.0, np.divide(spare[mom], asked[mom], out=np.zeros(len(mom)), where=asked[mom] > 0))
        np.subtract.at(self.reserve, i[mom], give)
        self.reserve[i[young]] += cfg.milk_eff * give
        self.books.heat += float(((1 - cfg.milk_eff) * give).sum())
        got[young] = cfg.milk_eff * give
        return got

    def _take_meat(self, cells, want):
        """Like `World.graze`, for carcasses: eaters sharing a cell split what's there in proportion to their wants."""
        asked = np.zeros(self.meat.size)
        np.add.at(asked, cells, want)
        share = np.divide(np.minimum(self.meat, asked), asked, out=np.zeros_like(asked), where=asked > 0)
        self.meat -= asked * share
        np.maximum(self.meat, 0.0, out=self.meat)
        return want * share[cells]

    # (3i) every per-creature array (the carcass grid `meat` is per cell, not per creature)
    PER_CREATURE = ("alive", "pos", "heading", "mass", "adult", "birth", "fat", "diet", "reserve", "age", "gen", "uid",
                    "parent", "w", "calling", "mem", "parent2", "alone", "dry")

    def _grow(self, need):
        """Double the arrays (up to `ceiling`) until `need` more creatures fit; new slots are empty."""
        m = len(self.alive)
        free = m - int(self.alive.sum())
        new = m
        while free + new - m < need and new < self.cfg.ceiling:
            new = min(2 * new, self.cfg.ceiling)
        if new == m:
            return
        for name in self.PER_CREATURE:
            a = getattr(self, name)
            fill = -1 if name in ("uid", "parent", "parent2") else 0
            b = np.full((new,) + a.shape[1:], fill, a.dtype)
            b[:m] = a
            setattr(self, name, b)

    def _births(self, parents, partners=None):
        """A parent splits off one child at birth_frac of the adult mass, with a half-full reserve, if it can pay.
        (6a) With `partners`: each pair has one child, each paying half; every gene and weight from one or the other."""
        cfg, b = self.cfg, self.books
        if not len(parents):
            return
        rng = self.rng
        k = len(parents)
        q = parents if partners is None else partners
        pick = (lambda a: a[parents]) if partners is None else (
            lambda a: np.where(rng.random(k) < 0.5, a[parents], a[q]))   # a gene from one parent or the other
        adult = pick(self.adult) * np.exp(rng.normal(0, cfg.sigma_gene, k))
        frac = np.clip(pick(self.birth) * np.exp(rng.normal(0, cfg.sigma_gene, k)), 0.01, 0.9)
        fat = np.clip(pick(self.fat) * np.exp(rng.normal(0, cfg.sigma_gene, k)), 0.01, 1.0)
        diet = np.clip(pick(self.diet) + rng.normal(0, cfg.sigma_diet, k), 0.0, 1.0)
        alone = np.clip(pick(self.alone) + rng.normal(0, cfg.sigma_alone, k), 0.0, 1.0)
        w = self.w[parents] if partners is None else np.where(rng.random((k, N_W)) < 0.5, self.w[parents], self.w[q])
        m0 = frac * adult
        r0 = 0.5 * self.reserve_cap(m0, fat)
        cost = m0 * cfg.body_kj_kg * (1 + cfg.build_overhead) + r0
        share = 1.0 if partners is None else 0.5
        can = (self.reserve[parents] >= share * cost) & (self.reserve[q] >= share * cost)
        g = [parents, q, adult, frac, fat, diet, m0, r0, cost, w, alone]
        g = [a[can] for a in g]
        if len(self.alive) - self.alive.sum() < len(g[0]):
            self._grow(len(g[0]))
        free = np.flatnonzero(~self.alive)
        if len(free) < len(g[0]):
            b.full += len(g[0]) - len(free)
            g = [a[:len(free)] for a in g]
        parents, q, adult, frac, fat, diet, m0, r0, cost, w, alone = g
        kids = free[:len(parents)]
        self.reserve[parents] -= share * cost
        if partners is not None:
            self.reserve[q] -= share * cost
        b.heat += float((m0 * cfg.body_kj_kg * cfg.build_overhead).sum())
        self.alive[kids] = True
        self.pos[kids] = self.pos[parents]
        self.heading[kids] = rng.uniform(-np.pi, np.pi, len(kids))
        self.adult[kids], self.birth[kids], self.mass[kids], self.reserve[kids] = adult, frac, m0, r0
        self.fat[kids] = fat
        self.diet[kids] = diet
        self.alone[kids] = alone
        self.age[kids] = 0.0
        self.dry[kids] = 0.0
        self.gen[kids] = np.maximum(self.gen[parents], self.gen[q]) + 1
        self.parent[kids] = self.uid[parents]
        self.parent2[kids] = -1 if partners is None else self.uid[q]
        self.uid[kids] = self.next_uid + np.arange(len(kids))
        self.next_uid += len(kids)
        self.w[kids] = w + rng.normal(0, cfg.sigma_w, (len(kids), N_W))
        self.calling[kids] = False
        self.mem[kids] = 0.0
        b.births += len(kids)

    def _deaths(self, i, killed=None):
        """Killed, starved (the reserve ran out: the shortfall comes out of the body) or old. The body and whatever
        fat is left become a carcass on the cell where it died."""
        cfg, b = self.cfg, self.books
        k = np.zeros(len(i), bool) if killed is None else killed
        dead = i[k | (self.reserve[i] < 0) | (self.age[i] > self.lifespan(self.adult[i])) | self.parched(i)]
        b.kills += int(k.sum())
        if not len(dead):
            return
        cell, _ = self._cell(self.pos[dead])
        np.add.at(self.meat, cell, self.reserve[dead] + cfg.body_kj_kg * self.mass[dead])
        self.alive[dead] = False
        self.reserve[dead] = 0.0
        self.mass[dead] = 0.0
        b.deaths += len(dead)

    def error(self):
        """Books minus reality for plants and creatures together, as a share of all energy that ever entered."""
        wb, b = self.world.ledger, self.books
        entered = wb.start + wb.sun_in + b.start
        expected = entered - wb.dieback - b.heat - b.dung - b.decay
        return (expected - self.world.total_plants() - self.energy() - float(self.meat.sum())) / entered
