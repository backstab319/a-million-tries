"""Who survives the winter crash? Run a world for some years and, each year, compare the creatures alive at the
year's population peak with the ones of them still alive next spring (day 90).

uv run python scripts/eco_survivors.py --seed 0 --years 5 --out out/ecosystem/survivors-s0.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ailearns.games.ecosystem import BIOMES, World, WorldConfig
from ailearns.games.ecosystem_life import Life, LifeConfig


def snapshot(life):
    i = np.flatnonzero(life.alive)
    cell, _ = life._cell(life.pos[i])
    return {
        "uid": life.uid[i].copy(),
        "fat": life.fat[i].copy(),
        "adult": life.adult[i].copy(),
        "grown": life.mass[i] / life.adult[i],
        "fill": life.reserve[i] / life.reserve_cap(life.mass[i], life.fat[i]),
        "age": life.age[i].copy(),
        "biome": life.world.biome.reshape(-1)[cell].copy(),
    }


def compare(peak, alive_uids):
    s = np.isin(peak["uid"], alive_uids)
    row = {"peak_n": int(len(s)), "survivors": int(s.sum())}
    for k in ("fat", "adult", "grown", "fill", "age"):
        row[k] = [float(np.median(peak[k])), float(np.median(peak[k][s])) if s.any() else None]
    row["biome_all"] = {BIOMES[b]: float((peak["biome"] == b).mean()) for b in range(1, 4)}
    row["biome_surv"] = {BIOMES[b]: float((peak["biome"][s] == b).mean()) for b in range(1, 4)} if s.any() else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    world = World.make(a.seed, WorldConfig())
    life = Life(world, a.n, a.seed, LifeConfig())
    per_day = round(life.cfg.day_s / life.cfg.dt)
    year = int(world.cfg.year_days)
    pop, years, peak, peak_n = [], [], None, -1
    t0 = time.perf_counter()
    for day in range(a.years * year):
        for _ in range(per_day):
            life.step()
        n = int(life.alive.sum())
        pop.append(n)
        doy = day % year
        if doy >= 90 and n > peak_n:           # the year's peak, counted from spring
            peak, peak_n = snapshot(life), n
        if doy == 89 and peak is not None:     # next spring: who of last year's peak is still here?
            row = compare(peak, life.uid[life.alive])
            row.update(year=day // year, peak_day=None, spring_n=n)
            years.append(row)
            peak, peak_n = None, -1
            print(f"year {row['year']}: peak {row['peak_n']}, {row['survivors']} of them alive in spring "
                  f"(spring total {n}); fat {row['fat']}, adult kg {row['adult']}, fill {row['fill']}, "
                  f"age {row['age']}, biome {row['biome_surv']} v {row['biome_all']}", flush=True)
        if n == 0:
            print(f"extinct on day {day}")
            break
    out = {"seed": a.seed, "pop": pop, "years": years, "extinct_day": len(pop) if pop[-1] == 0 else None,
           "births": life.books.births, "lost_full": life.books.full, "error": life.error(),
           "wall_s": time.perf_counter() - t0}
    print(f"seed {a.seed}: {'extinct day ' + str(out['extinct_day']) if out['extinct_day'] else 'alive'}, "
          f"{out['wall_s']:.0f} s, books {out['error']:.0e}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
