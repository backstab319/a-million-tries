"""Can hunting pay at all in our physics? Drop hand-scripted hunters (not evolved brains) into a world of grazers.

The scripted hunter: turn towards the nearest creature it sees, run at it, attack when it's within reach, eat when
standing on meat, rest when full; (8e) breed when well fed, its children follow the script too (the hunters are the
diet-1 creatures), so it tests whether a hunter *line* can live. If even this starves, the physics makes hunting impossible (fix the physics); if it
thrives, the problem is only that random brains don't find the skill in time.

uv run python scripts/eco_hunter_check.py --seed 1 --at 240 --hunters 50 --follow 60
"""

import argparse

import numpy as np

from ailearns.games.ecosystem import World, WorldConfig
from ailearns.games.ecosystem_life import Life, LifeConfig


class Scripted(Life):
    def think(self, i, x):
        out = super().think(i, x)
        h = self.diet[i] >= 0.99                 # (8e) the hunters and their children (grazers start at 0-0.1)
        if not h.any():
            return out
        y = x[h]
        close = y[:, 11:14]
        seen = close.max(1) > 0
        side = np.argmax(close, 1)           # 0 left eye (lower angle), 1 ahead, 2 right
        turn = np.where(seen, np.array([-3.0, 0.0, 3.0])[side], 2.0)   # nobody seen: circle to search
        hungry = y[:, 7] < 0.95
        meat_here = y[:, 10] > 0
        eat = meat_here & hungry
        speed = np.where(eat, -9.0, np.where(seen & hungry, 9.0, np.where(hungry, 0.0, -9.0)))
        reach = 1 - self.cfg.reach_m / self.cfg.sight_m
        attack = (close[:, 1] > reach) & hungry & ~meat_here
        breed = y[:, 7] > 0.8
        out[h, :5] = np.stack([turn, speed, np.where(eat, 9.0, -9.0), np.where(breed, 9.0, -9.0),
                               np.where(attack, 9.0, -9.0)], 1)
        out[h, 5:] = [-5.0, 0.0, 0.0]            # quiet, memory 0
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--at", type=float, default=240)
    ap.add_argument("--hunters", type=int, default=50)
    ap.add_argument("--follow", type=float, default=60)
    ap.add_argument("--grazers", type=int, default=1000)
    a = ap.parse_args()
    world = World.make(a.seed, WorldConfig())
    life = Scripted(world, 0, a.seed, LifeConfig())
    life.add(a.grazers, diet=(0.0, 0.1))
    per_day = round(life.cfg.day_s / life.cfg.dt)
    for _ in range(int(a.at * per_day)):
        life.step()
    life.cfg = LifeConfig(sigma_diet=0.0)   # (8e) diet fixed from here: the script follows diet 1, and only them
    idx = life.add(a.hunters, diet=(1.0, 1.0))
    life.alone[idx] = 1.0                    # they breed alone when no mate is about
    print(f"day {a.at:.0f}: {life.alive.sum() - a.hunters} grazers, {a.hunters} scripted hunters (diet 1, breed)")
    kills0 = life.books.kills
    print("day  grazers  hunters  hunter_reserve%(med)  hunter_kg(med)  kills  max_gen")
    for d in range(1, int(a.follow) + 1):
        for _ in range(per_day):
            life.step()
        if d % 5 == 0:
            al = life.alive
            hu = al & (life.diet >= 0.99)
            fill = np.median(life.reserve[hu] / life.reserve_cap(life.mass[hu], life.fat[hu])) * 100 if hu.any() else 0
            kg = np.median(life.mass[hu]) if hu.any() else 0
            print(f"{a.at + d:4.0f} {(al & ~hu).sum():8d} {hu.sum():8d} {fill:21.0f} {kg:15.2f} "
                  f"{life.books.kills - kills0:6d} {life.gen[hu].max() if hu.any() else 0:8d}", flush=True)
    print(f"books {life.error():.0e}")


if __name__ == "__main__":
    main()
