"""2D debug view of the racing sandbox (episode 14): record a drive, render it as a chase-camera video.

The scripted test driver (race_checks.test_driver, no AI) drives a car of a chosen design; the video shows the car
as its design makes it (body, tyre widths, wings, engine), sliding tyres in red, sparks when it bottoms out, and the
dashboard (speed, gear, g, pedals, what each tyre is doing). For checking the sim by eye.

    uv run --group jax python scripts/race_debug.py --design f1 --track 0 --seconds 30 [--still 12]
    -> out/ep14/debug-<design>-t<track>.mp4 (and .png with --still)
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from ailearns.games import race, race_checks
from ailearns.games import race_car as rc
from ailearns.media.video import VideoWriter
from ailearns.render import theme as T
from ailearns.render.race_view import CircuitView

DESIGNS = {
    "f1": rc.F1_LIKE[0],
    "low-soft": rc.design(wing_front=1, wing_rear=1, ride_height_mm=15, spring_n_mm=30, power_kw=500),
    "road-ish": rc.design(power_kw=200, wing_front=0, wing_rear=0, tyre_front_mm=225, tyre_rear_mm=255,
                          engine_pos=0.1, ride_height_mm=110, spring_n_mm=40),
}


def record(design, track: int, seconds: float, cfg=race.RaceConfig(), run: str | None = None,
           start_kmh: float = 0.0) -> dict:
    """Drive with the scripted test driver, or with an AI driver's brain (run = "runs/x" or "runs/x:ckpt").
    `start_kmh`: a rolling start, as in driver school (training starts at up to 144 km/h)."""
    game = race.RaceGame(race.make_tracks([track], cfg), cfg)
    if run:
        from ailearns.agents.ppo_car import forward, load
        from ailearns.checkpoints import pick_checkpoint
        path, _, ck = run.partition(":")
        params, meta = load(pick_checkpoint(path, ck or "last"))
        policy = lambda g, s: jnp.clip(forward(params, g.observe(s)[None])[0][0], -1, 1)  # noqa: E731 (best guess)
        label = ("AI DRIVER", f"after {meta['steps'] / 1e6:.0f}M steps of practice")
    else:
        policy, label = race_checks.test_driver, ("TEST DRIVER", "scripted, not the AI")
    st = game.init(jax.random.PRNGKey(0), design=jnp.asarray(design), track=0, random_start=False)
    if start_kmh:
        car = rc.car_at(game.tracks.center[0, 0], game.tracks.heading[0, 0], start_kmh / 3.6,
                        ride_height=float(design[rc.D["ride_height_mm"]]) / 1000.0, tyre_temp=cfg.tyre_start)
        st = st._replace(car=car)
    step = jax.jit(lambda s: game.step(s, policy(game, s))[0])
    rows = [st]
    for _ in range(int(seconds / cfg.dt)):
        st = step(st)
        rows.append(st)
        if bool(st.done):
            break
    stack = jax.tree.map(lambda *x: np.stack(x), *rows)
    return {"st": stack, "track": race.make_track(track, cfg), "cfg": cfg, "design": np.asarray(design),
            "label": label}


def _bar(surf, x, y, w, v, label, centered, color):
    T.text(surf, label, (x, y), "semibold", 22, T.MUTED)
    r = pygame.Rect(x, y + 30, w, 16)
    pygame.draw.rect(surf, T.PANEL_EDGE, r, border_radius=8)
    if centered:
        mid = x + w // 2
        end = mid + int(v * w / 2)
        pygame.draw.rect(surf, color, pygame.Rect(min(mid, end), y + 30, abs(end - mid) + 1, 16), border_radius=8)
    else:
        pygame.draw.rect(surf, color, pygame.Rect(x, y + 30, int(max(v, 0) * w), 16), border_radius=8)


def render(rec: dict, out: str, playback: float = 1.0, still: float | None = None, zoom: float = 10.0,
           start: float = 0.0) -> None:
    T.init()
    st, cfg, d = rec["st"], rec["cfg"], rec["design"]
    W, H = T.WIDTH, T.HEIGHT
    surf = pygame.Surface((W, H))
    view = CircuitView(rec["track"], cfg, pygame.Rect(0, 0, 1300, H))
    mini = CircuitView(rec["track"], cfg, pygame.Rect(1380, 640, 420, 420))
    n = len(st.t)
    rng = np.random.default_rng(0)
    trail: list = []
    frames = int((n - 1) * cfg.dt / playback * T.FPS)
    targets = [int(still / playback * T.FPS)] if still is not None else range(int(start / playback * T.FPS), frames)
    writer = None if still is not None else VideoWriter(out, (W, H))
    try:
        for f in targets:
            i = min(f / T.FPS * playback / cfg.dt, n - 1.001)
            k, u = int(i), i - int(i)
            pos = st.car.pos[k] * (1 - u) + st.car.pos[k + 1] * u
            dyaw = (st.car.yaw[k + 1] - st.car.yaw[k] + np.pi) % (2 * np.pi) - np.pi
            yaw = st.car.yaw[k] + dyaw * u
            trail = (trail + [pos])[-300:]
            surf.fill(T.BG)
            view.follow(pos, zoom)
            view.draw(surf)
            surf.set_clip(view.rect)
            view.draw_trail(surf, trail)
            crashed = bool(st.crashed[k + 1])
            view.draw_car(surf, d, pos, yaw, st.car.steer[k + 1], st.sliding[k + 1], st.plank[k + 1],
                          color=T.RED if crashed else T.GREEN, rng=rng)
            surf.set_clip(None)
            x = 1380
            T.text(surf, rec["label"][0], (x, 50), "display", 50, T.TEXT)
            T.text(surf, rec["label"][1], (x, 115), "semibold", 26, T.GREEN if "AI" in rec["label"][0] else T.AMBER)
            v = float(st.car.vx[k + 1]) * 3.6
            T.text(surf, f"{v:3.0f} km/h", (x, 170), "display", 64, T.TEXT)
            T.text(surf, f"gear {int(st.car.gear[k + 1])}", (x + 330, 190), "semibold", 32, T.MUTED)
            g = np.hypot(st.car.ax[k + 1], st.car.ay[k + 1]) / rc.G
            T.text(surf, f"{g:.1f} g", (x + 330, 232), "semibold", 32, T.MUTED)
            a = st.last[k + 1]
            _bar(surf, x, 290, 440, float(a[0]), "STEERING", True, T.BLUE)
            _bar(surf, x, 360, 440, float(max(a[1], 0)), "THROTTLE", False, T.GREEN)
            _bar(surf, x, 430, 440, float(max(-a[1], 0)), "BRAKE", False, T.RED)
            T.text(surf, "TYRES", (x, 505), "semibold", 22, T.MUTED)
            grass = cfg.grass_surface[0]
            for j, (dx, dy) in enumerate(((0, 0), (60, 0), (0, 60), (60, 60))):  # front-left, front-right, rear...
                grip = float(st.surface[k + 1][j])
                col = (T.RED if st.sliding[k + 1][j] else T.ORANGE if np.isclose(grip, grass)
                       else T.GREEN if grip >= 0.99 else T.AMBER)
                pygame.draw.rect(surf, col, pygame.Rect(x + dx, 540 + dy, 44, 50), border_radius=6)
            T.text(surf, "green tarmac, amber kerb,", (x + 150, 600), "semibold", 18, T.MUTED)
            T.text(surf, "orange grass, red sliding", (x + 150, 624), "semibold", 18, T.MUTED)
            on_grass = int(np.isclose(st.surface[k + 1], grass).sum())
            if on_grass:
                T.text(surf, f"{on_grass} TYRE{'S' * (on_grass > 1)} ON GRASS", (650, 60), "display", 56, T.ORANGE,
                       anchor="center")
            if float(st.plank[k + 1].sum()) > 0:  # (beside the tyres)
                T.text(surf, "BOTTOMING OUT", (x + 150, 560), "semibold", 30, T.ORANGE)
            mini.fit()
            mini.draw(surf)
            mini.draw_car(surf, d, pos, yaw, color=T.GREEN, min_px=18)
            T.text(surf, f"{float(st.progress[k + 1]):,.0f} m   {float(st.t[k + 1]) * cfg.dt:5.1f} s", (x, 1040),
                   "semibold", 26, T.MUTED)
            if crashed:
                T.text(surf, "CRASHED", (650, 120), "display", 110, T.RED, anchor="center")
            if writer is None:
                pygame.image.save(surf, out)
            else:
                writer.write(surf)
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="f1", help=f"one of {sorted(DESIGNS)}, or a history.json (its last design)")
    ap.add_argument("--run", default=None, help="drive with this AI driver (runs/x or runs/x:ckpt) instead")
    ap.add_argument("--track", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--playback", type=float, default=1.0, help="2 = twice real speed")
    ap.add_argument("--start", type=float, default=0.0, help="video starts at this many seconds into the drive")
    ap.add_argument("--still", type=float, default=None, help="save one frame at this many seconds instead")
    ap.add_argument("--zoom", type=float, default=10.0, help="pixels per metre")
    args = ap.parse_args()
    if args.design.endswith(".json"):
        import json
        design = np.array(json.loads(Path(args.design).read_text())["history"][-1]["design"], np.float32)
        args.design = Path(args.design).stem
    else:
        design = DESIGNS[args.design]
    rec = record(design, args.track, args.seconds, run=args.run)
    who = Path(args.run.split(":")[0]).name if args.run else "test"
    stem = Path("out/ep14") / f"debug-{who}-{args.design}-t{args.track}"
    out = str(stem.with_suffix(".png" if args.still is not None else ".mp4"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    render(rec, out, args.playback, args.still, args.zoom, args.start)
    print(out, f"{len(rec['st'].t) - 1} steps, crashed={bool(rec['st'].crashed[-1])}")
