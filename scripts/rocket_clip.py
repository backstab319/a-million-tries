"""Render one booster flight in 2D (render/rocket_view.py) to an mp4, or a contact sheet of stills to check it.

    PYTHONPATH=src python scripts/rocket_clip.py <pilot> <seed> [--world entry] [--label "ppo25"] [--speed 4]
        [--from 40 --to 85] [--airflow] [--versus <pilot> --world2 entry --label2 "TRY 29"]
        [--out out/rocket/clips/x.mp4] [--stills out/rocket/x.png] [--hold 2.5]

pilot: a checkpoint path, 'classic' or 'entry-script' (games.rocket_flight.pilot). --speed plays the flight faster
(the panel's clock still shows flight time); --hold keeps the end on screen (the explosion, the verdict).
--from/--to: a stretch of the flight (s). --airflow: the line of the airflow and the angle off it (ep18 12p).
--versus: a second pilot on the same seed, side by side (each half its own camera and panel), timed together.
"""

import argparse

import numpy as np
import pygame

from ailearns.games.rocket_flight import record
from ailearns.media.video import VideoWriter
from ailearns.render import theme as T
from ailearns.render.rocket_view import RocketView


def main():
    a = argparse.ArgumentParser()
    a.add_argument("pilot")
    a.add_argument("seed", type=int)
    a.add_argument("--world", default=None)
    a.add_argument("--label", default="")
    a.add_argument("--versus", default=None)
    a.add_argument("--world2", default=None)
    a.add_argument("--label2", default="")
    a.add_argument("--speed", type=float, default=1.0)
    a.add_argument("--from", dest="t0", type=float, default=0.0)
    a.add_argument("--to", dest="t1", type=float, default=None)
    a.add_argument("--airflow", action="store_true")
    a.add_argument("--hold", type=float, default=2.5)
    a.add_argument("--out", default=None)
    a.add_argument("--stills", default=None, help="a 3x3 sheet of frames instead of a video")
    a.add_argument("--fps", type=int, default=T.FPS)
    args = a.parse_args()

    T.init()
    recs = [record(args.pilot, args.seed, args.world)]
    if args.versus:
        recs.append(record(args.versus, args.seed, args.world2 or args.world))
    half = (T.WIDTH // len(recs), T.HEIGHT)
    views = [RocketView(r, lab, size=half) for r, lab in zip(recs, (args.label, args.label2))]
    for v in views:
        v.airflow = args.airflow
    surf = pygame.Surface((T.WIDTH, T.HEIGHT))
    parts = [pygame.Surface(half) for _ in views]
    end = max(float(r["time"][-1]) for r in recs)
    t1 = min(args.t1, end) if args.t1 is not None else end
    for r in recs:
        print(f"{r['pilot']}: {r['outcome']}, {r['time'][-1]:.1f} s of flight")

    def draw(t):
        for k, (v, part) in enumerate(zip(views, parts)):
            v.frame(part, t)
            surf.blit(part, (k * half[0], 0))
        if len(views) > 1:
            pygame.draw.line(surf, T.PANEL_EDGE, (half[0], 0), (half[0], T.HEIGHT), 4)

    if args.stills:
        sheet = pygame.Surface((T.WIDTH, T.HEIGHT))
        times = list(np.linspace(args.t0, t1, 8)) + [t1 + 0.8]
        for k, t in enumerate(times):
            draw(t)
            sheet.blit(pygame.transform.smoothscale(surf, (T.WIDTH // 3, T.HEIGHT // 3)),
                       ((k % 3) * T.WIDTH // 3, (k // 3) * T.HEIGHT // 3))
        pygame.image.save(sheet, args.stills)
        print(args.stills)
        return
    out = args.out or f"out/rocket/clips/{args.label or 'flight'}-s{args.seed}.mp4".replace(" ", "_")
    span = (t1 - args.t0) / args.speed
    hold = args.hold if t1 >= end else 0.0  # a stretch that stops before the end doesn't hold
    frames = int((span + hold) * args.fps)
    with VideoWriter(out, (T.WIDTH, T.HEIGHT), args.fps) as v:
        for f in range(frames):
            wall = f / args.fps
            t = args.t0 + wall * args.speed if wall <= span else t1 + (wall - span)
            draw(t)
            v.write(surf)
    print(out, f"{frames / args.fps:.1f} s")


if __name__ == "__main__":
    main()
