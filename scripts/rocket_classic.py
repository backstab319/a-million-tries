"""Fly the hand-written hoverslam controller on many random landing burns and report how they end.

    PYTHONPATH=src python scripts/rocket_classic.py [n] [seed]
"""

import sys

import numpy as np

from ailearns.agents.rocket_classic import act
from ailearns.games.rocket import CRASHED, LANDED, TIMEOUT, Boosters


def main(n=1000, seed=0):
    b = Boosters(int(n), seed=int(seed))
    vy, vx, tilt, x, fuel = (np.zeros(b.n) for _ in range(5))
    while True:
        flying = b.outcome == 0
        _, _, done = b.step(act(b))
        now = flying & done
        vy[now], vx[now], tilt[now], x[now], fuel[now] = b.vy[now], b.vx[now], b.tilt[now], b.x[now], b.fuel[now]
        if done.all():
            break
    for name, code in (("landed", LANDED), ("crashed", CRASHED), ("timed out", TIMEOUT)):
        print(f"{name:10s} {np.mean(b.outcome == code):6.1%}")
    crash = b.outcome == CRASHED
    if crash.any():
        cfg = b.cfg
        why = {"too fast down": np.abs(vy) > cfg.land_vy, "sliding": np.abs(vx) > cfg.land_vx,
               "leaning": np.abs(tilt) > cfg.land_tilt, "off the pad": np.abs(x) > cfg.pad_radius}
        print("crash reasons:", {k: int((v & crash).sum()) for k, v in why.items()})
    ok = b.outcome == LANDED
    if ok.any():
        print(f"touchdown speed: mean {np.mean(-vy[ok]):.2f} m/s, worst {np.max(-vy[ok]):.2f}; "
              f"miss mean {np.mean(np.abs(x[ok])):.1f} m; fuel left mean {np.mean(fuel[ok]):.0f} kg, "
              f"min {np.min(fuel[ok]):.0f}")
    return b


if __name__ == "__main__":
    main(*sys.argv[1:])
