"""Who kills whom, and why everyone attacks (ep17 step 3g): a world as `eco_run.py` runs it, watched by `Life.trace`.

Every `--every` sim days it prints, for plant-eaters (P, diet < 0.5) and meat-eaters (M):
- deaths by cause (killed, starved, old) and the kills as killer -> victim (PP, PM, MP, MM), how many fights the
  attacker won (the rest: the defender killed it), the median killer / victim mass, and how many kills the killer fed
  on within a day (a hunt, not a brawl); (4b) for each pair of kinds, how many victims were the killer's own child,
  its sibling, its parent, and how many were under a day old;
- the aggression probe: the share of steps each kind attacks, with a creature in reach and with nobody there
  (swinging at the air), and hungry (reserve under half full) v fed;
- (4e) the carcasses: on the ground now, eaten and rotted in the window (GJ);
- (5a) the voice: how often creatures call, hungry v fed, with a creature in reach, on a carcass; and what a call
  does to those who hear it (loud enough: over 0.1): do they turn towards it (+) or away (-), and how fast they go,
  against those who hear nothing;
- (6a) sex: (6b) births in the window and how many were alone, the alone gene (median, p10, p90); how often a
  grown creature is ready to breed; calling while ready v not (a mating call?); do ready
  hearers turn towards calls more than the rest;
- (7c) fruit: the share of all food that was fruit, how many creatures stand on fruit, and do those who smell it
  turn towards the scent (+) against those who smell none;
- (5b) memory: how big the values kept are, how much they change a step, and how often one is held steady
  (|change| < 0.01 while |value| > 0.5: a latch);
- the energy each kind got from meat and from plants; the world's green shoots and standing dead grass (3h); the
  meat-eaters' furthest generation (4c: does any line of them last?).

uv run python scripts/eco_kills.py --days 540 --n 1000 --invade 240:200 --invade 480:200 --seed 0
    --seeds 0 1 2 --out runs/eco-x: those worlds side by side (one process each), runs/eco-x/seed-<s>.log each
    (a Kaggle CPU job: ailearns cloud script "scripts/eco_kills.py ... --seeds 0 1 2 --out runs/eco-x" --on kaggle)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ailearns.games.ecosystem import World, WorldConfig
from ailearns.games.ecosystem_life import Life, LifeConfig


class Watch:
    """Sums what `Life.trace` reports over a window; `report()` prints it and starts the next one."""

    def __init__(self, per_day: int):
        self.per_day = per_day
        self.births0 = (0, 0)                       # (6b) the books' births at the window's start
        self.fed_uid, self.fed_tick = np.zeros(0, int), np.zeros(0, int)   # killers waiting for a meal (a day)
        self.reset()

    def reset(self):
        self.kills = np.zeros((2, 2), int)          # [killer is M, victim is M]
        self.kin = np.zeros((2, 2, 4), int)         # ... and the victim was: 0 its child, 1 sibling, 2 parent, 3 young
        self.att_won = self.fights = self.fed = self.starved = self.old = self.parched = 0
        self.water = np.zeros(3)                    # (9b) [steps, steps drinking, sum of thirst]
        self.ratio = []
        self.steps = np.zeros((2, 2, 2))            # [is M, someone in reach, hungry]
        self.attacks = np.zeros((2, 2, 2))
        self.meat, self.plant = np.zeros(2), np.zeros(2)
        self.milk = np.zeros((2, 3))                # (8f) [kind] x [young steps, nursing steps, milk kJ]
        self.eaten = 0.0                            # (4e) carcass energy eaten (before digestion)
        self.voice = np.zeros((4, 2))               # (5a) [hungry, fed, someone in reach, on a carcass] x [steps, calls]
        self.ear = np.zeros((2, 3))                 # [hears nothing, hears a call] x [steps, turn towards, speed]
        self.memo = np.zeros(4)                     # (5b) [values, sum |value|, sum |change|, latched]
        self.fruit = np.zeros(5)                    # (7c) [fruit kJ, all food kJ, steps on fruit, steps, ...]
        self.nose_f = np.zeros((2, 2))              # [smells none, smells fruit] x [steps, turn towards]
        self.sex = np.zeros((2, 4))                 # (6a) [not ready, ready] x [steps, calls, hear a call, turn towards]
        self.decay0 = None                          # the books' decay at the window's start
        self.green = np.zeros(2)                    # (9a) grass-eaters' [food under them, the land's average] summed

    def __call__(self, life, i, attack, reach, att, tgt, settled, won, killed, meat, plant, carcass=None, x=None,
                 turn=None, speed=None, call=None, mem=None, ready=None, fruit=None, milk=None):
        if x is not None:   # (9a) do grass-eaters stand where the grass is? (x[:, 0]: food here / a full cell)
            w = life.world
            if not hasattr(self, "_land"):
                from ailearns.games.ecosystem import WATER
                self._land = w.biome != WATER
            g = life.diet[i] < 0.5
            full = w.cfg.plant_kj_m2 * w.cfg.cell_m ** 2
            self.green += [x[g, 0].sum(), g.sum() * w.food()[self._land].mean() / full]
        if fruit is not None:   # (7c) the fruit probe
            self.fruit[:4] += [fruit.sum(), fruit.sum() + meat.sum() + plant.sum(), (x[:, 26] > 0.05).sum(), len(i)]
            smell = x[:, 30] > 0.05
            toward = turn * np.sign(x[:, 31])
            for k, mask in enumerate((~smell, smell)):
                self.nose_f[k] += [mask.sum(), toward[mask].sum()]
        if ready is not None:   # (6a) the sex probe
            loud = x[:, 22] > 0.1
            toward = turn * np.sign(x[:, 23])
            for k, mask in enumerate((~ready, ready)):
                self.sex[k] += [mask.sum(), (call & mask).sum(), (loud & mask).sum(), toward[loud & mask].sum()]
        if mem is not None:   # (5b) the memory probe: old values are senses 24-25, new ones what it wrote now
            old = x[:, 24:26]
            self.memo += [mem.size, np.abs(mem).sum(), np.abs(mem - old).sum(),
                          ((np.abs(mem - old) < 0.01) & (np.abs(mem) > 0.5)).sum()]
        if call is not None:   # (5a) the voice probe
            fill = life.reserve[i] / life.reserve_cap(life.mass[i], life.fat[i])
            for k, mask in enumerate((fill < 0.5, fill >= 0.5, reach, x[:, 10] > 0)):
                self.voice[k] += [mask.sum(), (call & mask).sum()]
            loud = x[:, 22] > 0.1
            for k, mask in enumerate((~loud, loud)):
                self.ear[k] += [mask.sum(), (turn * np.sign(x[:, 23]))[mask].sum(), speed[mask].sum()]
        if self.decay0 is None:
            self.decay0 = life.books.decay
        if carcass is not None:
            self.eaten += float(carcass.sum())
        m = (life.diet[i] > 0.5).astype(int)
        hungry = (life.reserve[i] / life.reserve_cap(life.mass[i], life.fat[i]) < 0.5).astype(int)
        r = reach.astype(int)
        np.add.at(self.steps, (m, r, hungry), 1)
        np.add.at(self.attacks, (m, r, hungry), attack)
        np.add.at(self.meat, m, meat)
        np.add.at(self.plant, m, plant)
        if milk is not None:   # (8f) the milk probe: how often the young are nursing, and how much milk
            young = life.mass[i] < life.cfg.wean_frac * life.adult[i]
            np.add.at(self.milk, (m[young], 0), 1)
            np.add.at(self.milk, (m[young], 1), milk[young] > 0)
            np.add.at(self.milk, (m, 2), milk)
        # kills: each settled fight's winner and loser (a victim counted once, if two fights settle on it)
        a, t, w = att[settled], tgt[settled], won[settled]
        killer, victim = np.where(w, a, t), np.where(w, t, a)
        victim, first = np.unique(victim, return_index=True)
        killer, w = killer[first], w[first]
        np.add.at(self.kills, (m[killer], m[victim]), 1)
        uk, uv, pk, pv = life.uid[i[killer]], life.uid[i[victim]], life.parent[i[killer]], life.parent[i[victim]]
        qk, qv = life.parent2[i[killer]], life.parent2[i[victim]]   # (6a) the other parent
        for r, hit in enumerate(((pv == uk) | (qv == uk), (pk == pv) & (pv >= 0), (pk == uv) | (qk == uv),
                                 life.age[i[victim]] < 1.0)):
            np.add.at(self.kin[..., r], (m[killer], m[victim]), hit.astype(int))
        self.att_won += int(w.sum())
        self.fights += len(victim)
        self.ratio += list(life.mass[i[killer]] / life.mass[i[victim]])
        # did the killer eat meat within a day of a kill?
        tick = life.ticks
        keep = tick - self.fed_tick < self.per_day
        self.fed_uid, self.fed_tick = self.fed_uid[keep], self.fed_tick[keep]
        ate = life.uid[i[meat > 0]]
        hit = np.isin(self.fed_uid, ate)
        self.fed += int(hit.sum())
        self.fed_uid, self.fed_tick = self.fed_uid[~hit], self.fed_tick[~hit]
        self.fed_uid = np.concatenate([self.fed_uid, life.uid[i[killer]]])
        self.fed_tick = np.concatenate([self.fed_tick, np.full(len(killer), tick)])
        # the other deaths, as Life._deaths decides them
        starve = (life.reserve[i] < 0) & ~killed
        self.starved += int(starve.sum())
        parched = life.parched(i) & ~killed & ~starve          # (9b)
        self.parched += int(parched.sum())
        self.old += int(((life.age[i] > life.lifespan(life.adult[i])) & ~killed & ~starve & ~parched).sum())
        if life.cfg.thirst and x is not None:   # x is this step's senses: x[:, 32] the thirst before drinking
            self.water += [len(i), int(life.drank.sum()), x[:, 32].sum()]

    def report(self, day, life):
        al = life.alive
        n_m = int((life.diet[al] > 0.5).sum())
        k = self.kills.sum()
        dead = k + self.starved + self.old + self.parched
        pct = lambda x, n: f"{100 * x / n:3.0f}%" if n else "  - "
        rate = lambda kind, r, h: pct(self.attacks[kind, r, h], self.steps[kind, r, h])
        mg = int(life.gen[al & (life.diet > 0.5)].max()) if n_m else 0   # (4c) the meat-eaters' furthest generation
        dq = np.quantile(life.diet[al], [0.5, 0.9, 0.99]) if al.any() else np.zeros(3)   # (4c) is diet drifting?
        print(f"day {day:4.0f}  alive P {al.sum() - n_m:5d} M {n_m:4d} (gen {mg:2d}) | deaths {dead:6d}: killed {pct(k, dead)} "
              f"starved {pct(self.starved, dead)} thirst {pct(self.parched, dead)} old {pct(self.old, dead)} | kills PP {self.kills[0, 0]} "
              f"PM {self.kills[0, 1]} MP {self.kills[1, 0]} MM {self.kills[1, 1]}, attacker won {pct(self.att_won, self.fights)}, "
              f"killer/victim mass {np.median(self.ratio) if self.ratio else 0:.2f}, fed within a day {pct(self.fed, k)} | "
              f"green {life.world.plants.sum() / 1e6:5.1f} GJ, dead grass {life.world.dead.sum() / 1e6:5.1f} GJ | "
              f"diet median {dq[0]:.2f} p90 {dq[1]:.2f} p99 {dq[2]:.2f}, max gen {life.gen[al].max() if al.any() else 0}")
        w = life.world
        print(f"          grass: grass-eaters stand on {self.green[0] / max(self.green[1], 1e-9):.2f}x the land's "
              f"average food" + (f"; soil wet {w.wet[self._land].mean():.2f} on average, rain front at "
              f"({w.belt()[0]:.0f}, {w.belt()[1]:.0f}) m" if w.cfg.rain and hasattr(self, '_land') else ""))
        if self.water[0]:
            print(f"          water: drinking {pct(self.water[1], self.water[0])} of steps, thirst {self.water[2] / self.water[0]:.2f} on average")
        kin = [f"{a}{b} {self.kills[x, y]}: child {pct(self.kin[x, y, 0], self.kills[x, y])} sib "
               f"{pct(self.kin[x, y, 1], self.kills[x, y])} parent {pct(self.kin[x, y, 2], self.kills[x, y])} "
               f"young {pct(self.kin[x, y, 3], self.kills[x, y])}"
               for x, a in ((0, "P"), (1, "M")) for y, b in ((0, "P"), (1, "M")) if self.kills[x, y]]
        if kin:
            print("          kin: " + "; ".join(kin))
        if self.voice[0, 0] + self.voice[1, 0]:
            v, e = self.voice, self.ear
            calls = lambda k: pct(v[k, 1], v[k, 0])
            avg = lambda k, j: f"{e[k, j] / e[k, 0]:+.2f}" if e[k, 0] else "  -  "
            print(f"          voice: calling {pct(v[0, 1] + v[1, 1], v[0, 0] + v[1, 0])} of steps (hungry {calls(0)}, fed "
                  f"{calls(1)}, someone in reach {calls(2)}, on a carcass {calls(3)}); hearing a call "
                  f"{pct(e[1, 0], e[0, 0] + e[1, 0])} of steps: turn towards it {avg(1, 1)} (deaf baseline {avg(0, 1)}), "
                  f"speed {avg(1, 2)} v {avg(0, 2)}")
        if self.fruit[3]:
            f, nf = self.fruit, self.nose_f
            tw = lambda k: f"{nf[k, 1] / nf[k, 0]:+.2f}" if nf[k, 0] else "  -  "
            print(f"          fruit: {pct(f[0], f[1])} of all food; on fruit {pct(f[2], f[3])} of steps; smelling fruit "
                  f"{pct(nf[1, 0], nf.sum(0)[0])} of steps, turn towards it {tw(1)} (none smelt {tw(0)}); "
                  f"fallen fruit {life.world.fruit.sum() / 1e6:.2f} GJ")
        if self.sex[:, 0].sum():
            x_ = self.sex
            tw = lambda k: f"{x_[k, 3] / x_[k, 2]:+.2f}" if x_[k, 2] else "  -  "
            nb, na = life.books.births - self.births0[0], life.books.births_alone - self.births0[1]
            g = np.quantile(life.alone[al], [0.5, 0.1, 0.9]) if al.any() else np.zeros(3)
            print(f"          sex: births {nb} ({pct(na, nb)} alone), alone gene {g[0]:.2f} ({g[1]:.2f}-{g[2]:.2f}); ready {pct(x_[1, 0], x_[:, 0].sum())} of steps; calling ready {pct(x_[1, 1], x_[1, 0])} v "
                  f"not {pct(x_[0, 1], x_[0, 0])}; turn towards a call ready {tw(1)} v not {tw(0)}")
        if self.memo[0]:
            m = self.memo
            print(f"          memory: |value| {m[1] / m[0]:.2f}, change a step {m[2] / m[0]:.3f}, latched {pct(m[3], m[0])}")
        print(f"          carcasses: on the ground {life.meat.sum() / 1e6:6.2f} GJ, this window eaten {self.eaten / 1e6:6.2f} GJ, "
              f"rotted {(life.books.decay - (self.decay0 or 0.0)) / 1e6:6.2f} GJ")
        for kind, name in ((0, "P"), (1, "M")):
            if self.steps[kind].sum():
                print(f"          {name} attacks: someone in reach {rate(kind, 1, 1)} hungry / {rate(kind, 1, 0)} fed, "
                      f"nobody there {rate(kind, 0, 1)} hungry / {rate(kind, 0, 0)} fed; "
                      f"in reach {pct(self.steps[kind, 1].sum(), self.steps[kind].sum())} of steps; "
                      f"food: meat {self.meat[kind] / 1e3:8.0f} MJ, plants {self.plant[kind] / 1e3:8.0f} MJ, milk "
                      f"{self.milk[kind, 2] / 1e3:6.0f} MJ (young nursing {pct(self.milk[kind, 1], self.milk[kind, 0])})",
                      flush=True)
        self.reset()
        self.decay0 = life.books.decay
        self.births0 = (life.books.births, life.books.births_alone)


def control(life, per_day, days=5.0):
    """(9c) The green probe's control: the same world, evolved brains v random ones, from the same moment.
    (9d) And what they do: how often they drink and how thirsty they are when they do, how far from water they
    keep, how fast they go, how often they eat."""
    import copy
    from scipy.ndimage import distance_transform_edt
    from ailearns.games.ecosystem import WATER
    land = life.world.biome != WATER
    full = life.world.cfg.plant_kj_m2 * life.world.cfg.cell_m ** 2
    n, cm = life.world.cfg.size, life.world.cfg.cell_m
    out = []
    for scramble in (False, True):
        c = copy.deepcopy(life)
        c.trace = None
        if scramble:
            c.w[:] = c.rng.normal(0, 1, c.w.shape)
        on = avg = steps = drinks = thirst_at = dist = moved = thirst = 0.0
        n0 = int(c.alive.sum())
        far = None
        for t in range(round(days * per_day)):
            if c.cfg.thirst and t % round(per_day / 4) == 0:   # metres to the nearest drinkable cell (puddles move)
                far = distance_transform_edt(~c.drinkable().reshape(n, n)) * cm
            i = np.flatnonzero(c.alive)
            g = c.diet[i] < 0.5
            here, _ = c._cell(c.pos[i])
            if g.any():
                on += c.world.food().reshape(-1)[here[g]].sum() / full
                avg += g.sum() * c.world.food()[land].mean() / full
            th = c.dry[i] / (c.cfg.dry_max_l_kg * c.mass[i]) if c.cfg.thirst else np.zeros(len(i))
            p0 = c.pos[i].copy()
            c.step()
            steps += len(i)
            thirst += th.sum()
            moved += np.hypot(*(c.pos[i] - p0).T).sum() / c.cfg.dt
            if far is not None:
                dist += far.reshape(-1)[here].sum()
                d = c.drank
                drinks += d.sum()
                thirst_at += th[d].sum()
        tag = "random" if scramble else "evolved"
        line = f"{tag} brains {on / max(avg, 1e-9):.2f}x (alive {n0} -> {int(c.alive.sum())})"
        if far is not None and steps:
            line += (f" [drink {100 * drinks / steps:.0f}% of steps at thirst {thirst_at / max(drinks, 1):.2f}, "
                     f"thirst {thirst / steps:.2f}, {dist / steps:.1f} m from water, {moved / steps:.2f} m/s]")
        out.append(line)
    print(f"          control ({days:.0f} days from here): " + "; ".join(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=540)
    ap.add_argument("--n", type=int, default=1000, help="first plant-eaters (diet 0-0.1)")
    ap.add_argument("--mixed", action="store_true", help="(4d) the first creatures get random diets 0-1 instead")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--gut", action="store_true", help="(11a) a gut that fits the body: small ones pass fibre unfermented")
    ap.add_argument("--cold", action="store_true", help="(11b) keeping warm costs: heat lost ~mass^0.5 x (37 C - air)")
    ap.add_argument("--cold-breath", action="store_true", help="(11d) keeping warm costs only the water breathed out")
    ap.add_argument("--thirst", action="store_true", help="(9b) creatures need water: from food, a lake's shore "
                    "or a rain puddle")
    ap.add_argument("--rain", action="store_true", help="(9a) grass grows only where storms wet the soil; a rainy "
                    "belt sweeps the map once a year")
    ap.add_argument("--invade", action="append", default=[], help="DAY:K drops K meat-eaters (diet 0.9-1) on DAY")
    ap.add_argument("--school", default="", help="(8) K:EVERY, the hunting school: every EVERY days, top the "
                    "meat-eaters (diet >= 0.5) back up to K, copying the living ones (old and well fed more often), or random "
                    "founders (diet 0.9-1) if none are alive")
    ap.add_argument("--pure", action="store_true", help="(8) the school's meat-eaters eat only meat: diet 1 and no "
                    "diet mutation in this world (copying survivors let them drift back to grass)")
    ap.add_argument("--school-from", type=float, default=0.0, help="(8c) open the school on this day (prey is "
                    "dense only in the boom years)")
    ap.add_argument("--school-until", type=float, default=1e9, help="(8) stop topping up after this day")
    ap.add_argument("--save-hunters", default="", help="(8) save the living meat-eaters' genes here at "
                    "each report (.npz; {seed} is filled in)")
    ap.add_argument("--hunters", action="append", default=[], help="(8) PATH:DAY:K drops K copies of the hunters "
                    "saved in PATH on DAY")
    ap.add_argument("--every", type=float, default=30)
    ap.add_argument("--control-at", type=float, action="append", default=[], help="(9c) on this day, copy the world "
                    "twice, give one copy random brains, run both 5 days: food underfoot / the land's average, each")
    ap.add_argument("--hours", type=float, default=0.0, help="stop cleanly after this long (Kaggle ends a session at "
                    "12 h and keeps nothing: 11 for a long run; 0: no limit)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[], help="with --out: these worlds at once, one log each")
    ap.add_argument("--out", default="", help="a folder (runs/<name>) for the --seeds logs")
    a = ap.parse_args()
    if a.seeds:
        out = Path(a.out or "runs/eco-kills")
        out.mkdir(parents=True, exist_ok=True)
        args = [x for x in sys.argv[1:]]
        for flag in ("--seeds", "--out"):   # drop --seeds/--out and their values: each child runs one world
            k = args.index(flag) if flag in args else -1
            if k >= 0:
                j = k + 1
                while j < len(args) and not args[j].startswith("--"):
                    j += 1
                del args[k:j]
        procs = [subprocess.Popen([sys.executable, __file__, *args, "--seed", str(s)],
                                  stdout=open(out / f"seed-{s}.log", "w"), stderr=subprocess.STDOUT) for s in a.seeds]
        codes = [p.wait() for p in procs]
        print(f"seeds {a.seeds}: exit {codes}; logs in {out}")
        sys.exit(max(codes))
    world = World.make(a.seed, WorldConfig(size=a.size, rain=a.rain))
    life = Life(world, 0, a.seed, LifeConfig(sigma_diet=0.0, thirst=a.thirst, gut=a.gut, cold=a.cold, cold_breath=a.cold_breath) if a.pure else LifeConfig(thirst=a.thirst, gut=a.gut, cold=a.cold, cold_breath=a.cold_breath))
    life.add(a.n, diet=(0.0, 1.0) if a.mixed else (0.0, 0.1))
    per_day = round(life.cfg.day_s / life.cfg.dt)
    life.trace = watch = Watch(per_day)
    invade = {int(float(d) * per_day): int(k) for d, k in (x.split(":") for x in a.invade)}
    drops = {}
    for x in a.hunters:
        path, d, k = x.rsplit(":", 2)
        drops[int(float(d) * per_day)] = (dict(np.load(path)), int(k))
    school_k, school_every = (int(a.school.split(":")[0]), float(a.school.split(":")[1])) if a.school else (0, 0)
    copied = founders = 0
    t0 = time.perf_counter()
    for tick in range(int(a.days * per_day)):
        if tick in invade:
            life.add(invade[tick], diet=(0.9, 1.0))
            print(f"day {tick / per_day:.0f}: {invade[tick]} meat-eaters dropped in", flush=True)
        if tick in drops:
            like, k = drops[tick]
            life.add(k, like=like)
            print(f"day {tick / per_day:.0f}: {k} school hunters dropped in (from {len(like['diet'])} saved)", flush=True)
        open_ = a.school_from <= tick / per_day <= a.school_until
        if school_k and open_ and tick % round(school_every * per_day) == 0:
            m = np.flatnonzero(life.alive & (life.diet >= 0.5))
            need = school_k - len(m)
            if need > 0 and len(m):
                # well-fed survivors are copied most: age x how full the reserve is (age alone picked fat, not skill)
                p = (life.age[m] + 1.0) * life.reserve[m] / life.reserve_cap(life.mass[m], life.fat[m])
                life.add(need, like=life.genes(m), p=p / p.sum())
                copied += need
            elif need > 0:
                life.add(need, diet=(1.0, 1.0) if a.pure else (0.9, 1.0))
                founders += need
        if any(tick == round(d * per_day) for d in a.control_at):
            control(life, per_day)
        life.step()
        # the limit is checked every day, not every report: late 90-day blocks with big herds outran 1.5 h of margin
        late = a.hours and (tick + 1) % per_day == 0 and time.perf_counter() - t0 > a.hours * 3600
        if (tick + 1) % round(a.every * per_day) == 0 or late:
            watch.report((tick + 1) / per_day, life)
            if school_k:
                print(f"          school: topped up with {copied} copies of living hunters, {founders} random founders",
                      flush=True)
                copied = founders = 0
            if a.save_hunters:
                m = np.flatnonzero(life.alive & (life.diet >= 0.5))
                if len(m):
                    np.savez(a.save_hunters.format(seed=a.seed), **life.genes(m))
            if late:
                print(f"stopped at the --hours limit ({a.hours} h), day {(tick + 1) / per_day:.0f}", flush=True)
                break
            if not life.alive.any():
                print("everyone died")
                break
    print(f"{time.perf_counter() - t0:.0f} s wall")


if __name__ == "__main__":
    main()
