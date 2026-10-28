"""Race day (episode 15): step 7's grid races (`scripts/race_grid.py --save`, the ep14 driver x8, no pit lane, no tyre
or fuel model yet) in the shape `scripts/race_tv.py` draws, so the first start can be shown on the TV view.

Fills what step 7 didn't save: distance along the circuit (laps + the nearest centreline point), the running order
from it (cars out drop to the back), and neutral pit/tyre/fuel columns (no stops, mediums, a full tank: those
systems didn't exist yet). Only the first `--seconds` are kept.

    uv run python scripts/race_grid_tv.py out/ep15/grid-blind.npz [--seconds 120]
    -> out/ep15/race-grid-blind.npz, then: uv run python scripts/race_tv.py out/ep15/race-grid-blind.npz --speed 2
"""

import argparse
from pathlib import Path

import numpy as np

from ailearns.games import race

ap = argparse.ArgumentParser()
ap.add_argument("grid")
ap.add_argument("--seconds", type=float, default=120.0)
ap.add_argument("--laps", type=int, default=25)
a = ap.parse_args()

z = np.load(a.grid)
dt = float(z["dt"])
n = min(int(a.seconds / dt), z["pos"].shape[1])
seeds = z["seeds"]
cfg = race.RaceConfig(pit_lane=True)
prog = np.zeros(z["pos"].shape[:1] + (n,) + z["pos"].shape[2:3], np.float32)
lengths = []
for k, seed in enumerate(seeds):
    tr = race.make_track(int(seed), cfg)
    assert np.allclose(tr["center"], race.make_track(int(seed), race.RaceConfig())["center"]), "pit lane moved the road"
    c, L = np.asarray(tr["center"]), float(tr["length"])
    lengths.append(L)
    for j in range(n):  # nearest centreline point: metres into the lap
        d = ((z["pos"][k, j][:, None, :] - c[None]) ** 2).sum(-1)
        prog[k, j] = z["laps"][k, j] * L + d.argmin(1) * L / len(c)
out = z["walled"][:, :n] | z["wrecked"][:, :n]
rank = np.where(out, -1e9 + prog, prog)  # cars out drop behind every running car
order = np.argsort(np.argsort(-rank, axis=-1), axis=-1)
zeros = np.zeros_like(out)
dest = Path(a.grid).with_name("race-" + Path(a.grid).stem + ".npz")
np.savez_compressed(dest, seeds=seeds, dt=dt, laps=a.laps, length=np.asarray(lengths), pos=z["pos"][:, :n],
                    yaw=z["yaw"][:, :n], progress=prog, order=order, finished=z["finished"][:, :n],
                    walled=z["walled"][:, :n], wrecked=z["wrecked"][:, :n], dry=zeros, pit=zeros,
                    stops=zeros.astype(np.int8), wear=zeros.astype(np.float32), fuel=np.ones_like(prog),
                    bump=z["bump"][:, :n], compound=np.ones_like(out, np.int8))
print("->", dest, f"{n} frames ({n * dt:.0f} s) x {len(seeds)} circuits")
