"""Run an ecosystem world headless and print how the plant-eaters are doing each sim day.

uv run python scripts/eco_run.py --days 60 --n 300 --seed 0
"""

import argparse
import time

import numpy as np

from ailearns.games.ecosystem import World, WorldConfig
from ailearns.games.ecosystem_life import Life, LifeConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=60)
    ap.add_argument("--n", type=int, default=300, help="first creatures, random diets (or plant-eaters with --carn)")
    ap.add_argument("--carn", type=int, default=0, help="with this or --invade: --n plant-eaters (diet 0-0.1) + this many meat-eaters (0.9-1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--invade", action="append", default=[], help="DAY:K drops K meat-eaters (diet 0.9-1) on DAY")
    ap.add_argument("--every", type=float, default=5, help="print every this many sim days")
    a = ap.parse_args()
    world = World.make(a.seed, WorldConfig(size=a.size))
    mixed = a.carn or a.invade          # plant-eaters first, meat-eaters on their own
    life = Life(world, 0 if mixed else a.n, a.seed, LifeConfig())
    if mixed:
        life.add(a.n, diet=(0.0, 0.1))
        life.add(a.carn, diet=(0.9, 1.0))
    per_day = round(life.cfg.day_s / life.cfg.dt)
    t0 = time.perf_counter()
    print("day  alive  births deaths  kills  max_gen  mass_kg(med)  fat%(med)  meat-eaters%  diet(med)  reserve%  plants%  err")
    invade = {int(float(d) * per_day): int(k) for d, k in (x.split(":") for x in a.invade)}
    for tick in range(int(a.days * per_day)):
        if tick in invade:
            life.add(invade[tick], diet=(0.9, 1.0))
            print(f"day {tick / per_day:.0f}: {invade[tick]} meat-eaters dropped in", flush=True)
        life.step()
        if (tick + 1) % round(a.every * per_day) == 0:
            al = life.alive
            k = al.sum()
            if k:
                x = life.senses(np.flatnonzero(al))
                fill = np.median(life.reserve[al] / life.reserve_cap(life.mass[al], life.fat[al])) * 100
                food = (x[:, 0] > 0.05).mean() * 100
                med = np.median(life.mass[al])
                fat = np.median(life.fat[al]) * 100
                carn = (life.diet[al] > 0.5).mean() * 100
                diet = np.median(life.diet[al])
            else:
                fill = food = med = fat = carn = diet = 0
            b = life.books
            print(f"{(tick + 1) / per_day:4.0f} {k:6d} {b.births:7d} {b.deaths:6d} {b.kills:6d} {life.gen[al].max() if k else 0:7d}"
                  f" {med:12.2f} {fat:10.0f} {carn:13.0f} {diet:10.2f} {fill:9.0f} {world.plants.sum() / world.capacity.sum() * 100:8.0f}"
                  f"  {life.error():.0e}")
            if not k:
                print("everyone died")
                break
    print(f"{time.perf_counter() - t0:.0f} s wall")


if __name__ == "__main__":
    main()
