"""Compare designs fairly (episode 14): every design judged by the same driver on 32 test circuits nobody trained or
evolved on (race.TEST_SEEDS[8:40]), 60 s from a standing start.

    uv run --group jax python scripts/race_compare.py runs/<driver>[:<update>] runs/eng-a runs/eng-b ...
Each engineer run contributes its first and last design; the sweep's best (450 kW, full wings) is the reference.
"""

import json
import sys

import numpy as np

from ailearns.agents.ppo_car import load
from ailearns.agents.race_engineer import judge
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race
from ailearns.games import race_car as rc

driver, *engs = sys.argv[1:]
path, _, ck = driver.partition(":")
params, _ = load(pick_checkpoint(path, ck or "last"))
cfg = race.RaceConfig()
game = race.RaceGame(race.make_tracks(list(race.TEST_SEEDS)[8:40], cfg), cfg)
cands = {"sweep best: 450 kW, full wings, RWD": rc.design(power_kw=450, wing_front=1, wing_rear=1, drive_front=0.0)}
for e in engs:
    h = json.load(open(f"{e}/history.json"))["history"]
    cands.setdefault("start (mid-range)", np.array(h[0]["design"]))
    cands[f"{e.split('/')[-1]} gen {h[-1]['gen']}"] = np.array(h[-1]["design"])
out = judge(params, game, np.stack(list(cands.values())))
print(f"judged by {driver} on test circuits 100008-100039")
for (k, d), m, c in zip(cands.items(), out["metres"], out["crash"]):
    print(f"{k:38s} {m:6.0f} m  crash {c:.2f}  ${float(rc.cost_k(d)):.0f}k")
last = np.array(json.load(open(f"{engs[-1]}/history.json"))["history"][-1]["design"])
print("\n" + engs[-1] + " design:", ", ".join(f"{n} {v:.2f}" for n, v in zip(rc.DESIGN_NAMES, last)))
