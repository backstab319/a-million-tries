"""The final race (episode 14): each finalist car with its own specialised driver, all on the same unseen circuits,
60 s from a standing start. Also shows every car with the general driver (race-t4-1:572) as the baseline.

    uv run --group jax python scripts/race_final.py [--first 40 --n 24] name=runs/<driver>:<design.json> ...
Circuits default to race.TEST_SEEDS[40:64]: never used to pick anything (TEST_SEEDS[8:40] chose designs all episode).
"""

import argparse
import json

import numpy as np

from ailearns.agents.ppo_car import load
from ailearns.agents.race_engineer import judge
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race
from ailearns.games import race_car as rc

ap = argparse.ArgumentParser()
ap.add_argument("finalists", nargs="+", help="name=runs/<driver>[@<update>]:<design.json>")
ap.add_argument("--general", default="runs/race-t4-1:572")
ap.add_argument("--first", type=int, default=40)
ap.add_argument("--n", type=int, default=24)
a = ap.parse_args()


def brain(spec):
    path, _, ck = spec.replace("@", ":").partition(":")
    return load(pick_checkpoint(path, ck or "last"))[0]


cfg = race.RaceConfig()
seeds = list(race.TEST_SEEDS)[a.first:a.first + a.n]
game = race.RaceGame(race.make_tracks(seeds, cfg), cfg)
general = brain(a.general)
own_runs = {}
print(f"final race on test circuits {seeds[0]}-{seeds[-1]} ({len(seeds)}), 60 s from a standing start\n")
print(f"{'car':10s} {'cost':>7s} | {'own driver':>10s} {'crash':>5s} | {'general driver':>14s} {'crash':>5s}")
for f in a.finalists:
    name, _, rest = f.partition("=")
    driver, _, design_file = rest.rpartition(":")
    d = np.array(json.load(open(design_file))["history"][-1]["design"], np.float32)
    own = judge(brain(driver), game, d[None])
    gen = judge(general, game, d[None])
    own_runs[name] = np.asarray(own["per_track"][0])
    print(f"{name:10s} ${float(rc.cost_k(d)):5.0f}k | {float(own['metres'][0]):8.0f} m {float(own['crash'][0]):5.2f}"
          f" | {float(gen['metres'][0]):12.0f} m {float(gen['crash'][0]):5.2f}")

print("\nhead to head, own drivers (paired by circuit: mean difference +- its standard error, circuits won)")
names = list(own_runs)
for i, x in enumerate(names):
    for y in names[i + 1:]:
        diff = own_runs[y] - own_runs[x]
        print(f"  {y} - {x}: {diff.mean():+.0f} m +- {diff.std(ddof=1) / np.sqrt(len(diff)):.0f}, "
              f"{y} wins {int((diff > 0).sum())}/{len(diff)}")
