"""Do they learn to graze? Ask the brains, every N days of a world: each creature's brain gets its own real senses
with one thing changed, and we look at what it does.

- eat:  food under it (a full cell) v bare ground -> share that eat on food minus share that eat on bare ground
- turn: food in the left eye only v the right eye only -> share turning towards the food (left is negative turn)
- full: reserve empty v full, standing on food -> share that eat when empty minus share that eat when full
- walk: bare ground v food under it -> mean speed on bare ground minus on food (leave bare ground, stay on food)

Random brains score ~0 on all four. Plus the same scores for the first creatures (day 0) as the baseline.

uv run python scripts/eco_probe.py --seed 0 --days 720 --every 60
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ailearns.games.ecosystem import World, WorldConfig
from ailearns.games.ecosystem_life import Life, LifeConfig

PLANTS_HERE, EYES, BLOCKED, RESERVE = 0, slice(1, 4), slice(4, 7), 7


def probe(life):
    i = np.flatnonzero(life.alive)
    if not len(i):
        return None
    x = life.senses(i)
    x[:, BLOCKED] = 0.0

    def out(**change):
        y = x.copy()
        for k, v in change.items():
            y[:, {"here": PLANTS_HERE, "eyes": EYES, "reserve": RESERVE}[k]] = v
        o = life.think(i, y)
        return np.tanh(o[:, 0]), 1 / (1 + np.exp(-o[:, 1])), o[:, 2] > 0

    full_cell = 1.0
    _, v_food, eat_food = out(here=full_cell)
    _, v_bare, eat_bare = out(here=0.0)
    turn_l, _, _ = out(eyes=[full_cell, 0.0, 0.0])     # eye 1 is the left one (heading - 40 deg)
    turn_r, _, _ = out(eyes=[0.0, 0.0, full_cell])
    _, _, eat_empty = out(here=full_cell, reserve=0.0)
    _, _, eat_full = out(here=full_cell, reserve=1.0)
    # Turning: heading - 40 deg is the eye with the lower angle, so turning towards it is a negative turn.
    towards = ((turn_l < 0).mean() + (turn_r > 0).mean()) / 2 - 0.5
    return {
        "n": int(len(i)),
        "eat": float(eat_food.mean() - eat_bare.mean()),
        "turn": float(towards * 2),                    # -1..1, 0 = random
        "full": float(eat_empty.mean() - eat_full.mean()),
        "walk": float(v_bare.mean() - v_food.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=float, default=720)
    ap.add_argument("--every", type=float, default=60)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    world = World.make(a.seed, WorldConfig(size=a.size))
    life = Life(world, a.n, a.seed, LifeConfig())
    per_day = round(life.cfg.day_s / life.cfg.dt)
    rows = [dict(day=0, **probe(life))]
    print("day     n    eat   turn   full   walk")
    print("{day:4.0f} {n:5d} {eat:+.2f} {turn:+.2f} {full:+.2f} {walk:+.2f}".format(**rows[0]), flush=True)
    for tick in range(int(a.days * per_day)):
        life.step()
        if (tick + 1) % round(a.every * per_day) == 0:
            p = probe(life)
            if p is None:
                print(f"{(tick + 1) / per_day:4.0f} extinct")
                break
            rows.append(dict(day=(tick + 1) / per_day, **p))
            print("{day:4.0f} {n:5d} {eat:+.2f} {turn:+.2f} {full:+.2f} {walk:+.2f}".format(**rows[-1]), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"seed": a.seed, "rows": rows}))


if __name__ == "__main__":
    main()
