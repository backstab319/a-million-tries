"""numpy v JAX ecosystem worlds, side by side (plan section 4: a port that changes the biology is a bug).

Reads two `report.jsonl` files from `scripts/eco_jax_run.py` (same seeds, same settings) and prints, for each report
day, each measure's mean +- sd across worlds for both engines; `!` marks gaps over 3 standard errors.

    uv run python scripts/eco_jax_compare.py runs/eco-np-ref/report.jsonl runs/eco-jax-ref/report.jsonl
"""

import json
import sys
from collections import defaultdict

import numpy as np

MEASURES = ("alive", "births", "dead.killed", "dead.thirst", "dead.starved", "food.carcass", "food.green",
            "food.milk", "drink", "thirst", "speed", "calls", "attacks", "green", "diet", "gen_max", "mass")


def load(path, seeds=None):
    by = defaultdict(list)
    for ln in open(path):
        r = json.loads(ln)
        if seeds is None or r["seed"] in seeds:
            by[r["day"]].append(r)
    return by


def get(r, m):
    if "." in m:
        a, b = m.split(".")
        v = r[a][b]
        if a == "food":   # food as a share of all kept food
            v = 100 * v / max(sum(r["food"].values()), 1e-9)
        return v
    return r[m]


def main():
    a, b = load(sys.argv[1]), None
    seeds = {r["seed"] for rs in a.values() for r in rs}
    b = load(sys.argv[2], seeds)
    flags = 0
    for day in sorted(set(a) & set(b)):
        print(f"day {day:.0f} ({len(a[day])} numpy worlds, {len(b[day])} JAX)")
        for m in MEASURES:
            x = np.array([get(r, m) for r in a[day]], float)
            y = np.array([get(r, m) for r in b[day]], float)
            se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)) if len(x) > 1 and len(y) > 1 else 0
            gap = abs(x.mean() - y.mean())
            bad = se > 0 and gap > 3 * se
            flags += bad
            print(f"  {m:14s} numpy {x.mean():10.3f} +- {x.std():8.3f}   jax {y.mean():10.3f} +- {y.std():8.3f}"
                  + ("   !" if bad else ""))
    print(f"{flags} gaps over 3 standard errors (expect ~0.3% of comparisons by chance)")


if __name__ == "__main__":
    main()
