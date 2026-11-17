"""The booster's brain as a clip (render/rocket_brain.py): ppo25's network lighting up as it flies, optionally with
the flight itself in a corner (a 3D render or any mp4 of the same stretch).

    PYTHONPATH=src python scripts/rocket_brain_clip.py runs/rocket-ppo25/checkpoints/u15258.pkl 7 --world entry
        [--from 99.93] [--to 107.45] [--speed 1] [--hold 2] [--inset out/rocket/3d/clips/ppo25-land-chase.mp4]
        [--out out/rocket/brain/ppo25-s7.mp4] [--stills out/rocket/brain/ppo25-s7.png]

--from/--to are flight seconds (the 3D chase clip's frames 3000:3312 at 30 fps, speed 1, are 99.93 s to the end).
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pygame

from ailearns.games.rocket_flight import record
from ailearns.media.video import VideoWriter
from ailearns.render import theme as T
from ailearns.render.rocket_brain import BrainView


def main():
    a = argparse.ArgumentParser()
    a.add_argument("checkpoint")
    a.add_argument("seed", type=int)
    a.add_argument("--world", default=None)
    a.add_argument("--label", default="")
    a.add_argument("--from", dest="t0", type=float, default=0.0)
    a.add_argument("--to", dest="t1", type=float, default=None)
    a.add_argument("--speed", type=float, default=1.0)
    a.add_argument("--hold", type=float, default=2.0)
    a.add_argument("--inset", default=None, help="an mp4 of the same stretch, shown bottom left over the senses' end")
    a.add_argument("--beside", default=None, help="a 1920x1080 mp4 of the same stretch: its centre, cut to a tall "
                   "panel, goes on the right, the brain on the left (the 3D clips keep the booster centred)")
    a.add_argument("--panel", type=int, default=740, help="--beside: the flight panel's width")
    a.add_argument("--out", default=None)
    a.add_argument("--stills", default=None)
    a.add_argument("--fps", type=int, default=T.FPS)
    args = a.parse_args()

    T.init()
    rec = record(args.checkpoint, args.seed, args.world)
    size = (T.WIDTH - args.panel, T.HEIGHT) if args.beside else (T.WIDTH, T.HEIGHT)
    view = BrainView(rec, args.checkpoint, args.label, size=size, controls=(200, 560) if args.inset else None)
    surf = pygame.Surface(size)
    end = float(rec["time"][-1])
    t1 = min(args.t1 if args.t1 is not None else end, end)
    print(f"{rec['outcome']}, {end:.1f} s of flight; {view.net.n_in} senses")
    if args.stills:
        Path(args.stills).parent.mkdir(parents=True, exist_ok=True)
        sheet = pygame.Surface(size)
        for k, t in enumerate(np.linspace(args.t0, t1, 9)):
            view.shown = None
            view.frame(surf, t)
            sheet.blit(pygame.transform.smoothscale(surf, (size[0] // 3, size[1] // 3)),
                       ((k % 3) * size[0] // 3, (k // 3) * size[1] // 3))
        pygame.image.save(sheet, args.stills)
        print(args.stills)
        return
    out = Path(args.out or f"out/rocket/brain/{Path(args.checkpoint).parent.parent.name}-s{args.seed}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".brain.mp4") if args.inset or args.beside else out
    hold = args.hold
    if args.beside:  # run the brain as long as the flight clip (its end hold included)
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                args.beside], capture_output=True, text=True, check=True)
        hold = max(hold, float(probe.stdout) - (t1 - args.t0) / args.speed)
    frames = int(((t1 - args.t0) / args.speed + hold) * args.fps)
    with VideoWriter(str(raw), size, args.fps) as v:
        for f in range(frames):
            view.frame(surf, min(args.t0 + f / args.fps * args.speed, t1 + (f / args.fps - (t1 - args.t0) / args.speed)))
            v.write(surf)
    if args.inset:  # the flight in a framed box at the right, under the controls' readings
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-i", args.inset, "-filter_complex",
                        "[1:v]scale=640:-2,pad=iw+6:ih+6:3:3:color=0x262b3a[s];[0:v][s]overlay=W-w-40:H-h-40:shortest=0",
                        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out)], check=True)
    if args.beside:  # flight on the right in a tall panel, a thin edge between
        p = args.panel
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-i", args.beside, "-filter_complex",
                        f"[1:v]scale=-2:{T.HEIGHT},crop={p - 4}:{T.HEIGHT},pad={p}:{T.HEIGHT}:4:0:color=0x262b3a[s];"
                        "[0:v][s]hstack=inputs=2", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an",
                        str(out)], check=True)
    print(out, f"{frames / args.fps:.1f} s")


if __name__ == "__main__":
    main()
