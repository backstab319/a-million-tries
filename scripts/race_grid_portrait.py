"""Race day (episode 15), the before/after Short: step 7's first start (8 copies of the ep14 driver, blind to other
cars) drawn in portrait at real scale, the camera on the pack. Pure drawing from `scripts/race_grid_tv.py`'s file.

    uv run python scripts/race_grid_portrait.py out/ep15/race-grid-blind.npz --circuit 1 [--seconds 11]
    -> out/ep15/shorts/grid-<name>-c<circuit>-portrait.mp4
"""

import argparse
from pathlib import Path

import numpy as np
import pygame

from ailearns.games import race
from ailearns.games import race_car as rc
from ailearns.media.video import VideoWriter
from ailearns.render import theme as T
from ailearns.render.race_view import CircuitView

ap = argparse.ArgumentParser()
ap.add_argument("race")
ap.add_argument("--circuit", type=int, default=1)
ap.add_argument("--seconds", type=float, default=11.0)
ap.add_argument("--zoom", type=float, default=14.0, help="px per metre")
a = ap.parse_args()

z = np.load(a.race)
k, dt = a.circuit, float(z["dt"])
seed = int(z["seeds"][k])
pos, yaw, bump = z["pos"][k], z["yaw"][k], z["bump"][k]
out = z["walled"][k] | z["wrecked"][k]
n, N = pos.shape[:2]
W, H = 1080, 1920
T.init()
cfg = race.RaceConfig()
view = CircuitView(race.make_track(seed, cfg), cfg, pygame.Rect(0, 0, W, H))
design = np.asarray(rc.F1_LIKE[0])
dest = Path("out/ep15/shorts") / f"grid-{Path(a.race).stem}-c{k}-portrait.mp4"
dest.parent.mkdir(parents=True, exist_ok=True)
writer = VideoWriter(str(dest), (W, H))
surf = pygame.Surface((W, H))
cam = None
try:
    for fr in range(int(a.seconds * T.FPS)):
        i = min(fr / T.FPS / dt, n - 1.001)
        j, u = int(i), i - int(i)
        p = pos[j] * (1 - u) + pos[j + 1] * u
        dy = (yaw[j + 1] - yaw[j] + np.pi) % (2 * np.pi) - np.pi
        h = yaw[j] + dy * u
        run = ~out[j]
        target = p[run].mean(0) if run.any() else p.mean(0)
        cam = target if cam is None else cam + (target - cam) * 0.08
        surf.fill(T.BG)
        view.follow(cam, a.zoom)
        view.draw(surf)
        for c in range(N):
            col = T.mix(T.CARS[c], T.BG, 0.55) if out[j, c] else T.CARS[c]
            if bump[j, c]:
                T.glow(surf, view.px(p[c]), 40, T.TEXT, strength=70)
            view.draw_car(surf, design, p[c], h[c], color=col)
        T.text(surf, f"{int((~out[j]).sum())} OF 8 CARS RUNNING", (W // 2, 330), "display", 56,
               T.RED if out[j].any() else T.TEXT, anchor="midtop")
        T.text(surf, f"0:{fr / T.FPS:04.1f}", (W // 2, 410), "semibold", 40, T.MUTED, anchor="midtop")
        writer.write(surf)
finally:
    writer.close()
print("->", dest)
