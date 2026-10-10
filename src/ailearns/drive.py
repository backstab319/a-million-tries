"""Drive the race car yourself (arrow keys / WASD) and set the lap time the AI has to beat.

    uv run --group jax ailearns drive                  # a test track (the AI never trains on these)
    uv run --group jax ailearns drive --track 100003

Same game code the AI uses (games/car.py), same 10 decisions per second. Every
lap is saved to out/human-laps/ (time + the full drive, for ghost replays).
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from ailearns.games.car import CarConfig, CarGame, make_track, make_tracks
from ailearns.render import theme as T
from ailearns.render.car_view import TrackView

OUT = Path(__file__).resolve().parents[2] / "out" / "human-laps"
W, H = 1600, 900


def _input(keys, steer_now: float) -> tuple[float, float]:
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    want = (1.0 if left else 0.0) - (1.0 if right else 0.0)  # + = left (counter-clockwise)
    steer = steer_now + np.clip(want - steer_now, -0.35, 0.35)  # keyboards are on/off: ramp the wheel
    throttle = (1.0 if keys[pygame.K_UP] or keys[pygame.K_w] else 0.0) - (
        1.0 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0.0)
    return float(steer), throttle


def run(track_seed: int = 100_000) -> None:
    cfg = replace(CarConfig(), time_limit=600.0)
    game = CarGame(make_tracks([track_seed], cfg), cfg)
    track = make_track(track_seed, cfg)
    step = jax.jit(game.step)
    T.init(headless=False)
    window = T.Window((W, H), "ai-learns — drive")
    screen = window.canvas
    clock = pygame.time.Clock()
    view = TrackView(track, pygame.Rect(0, 0, W, H))
    mini = TrackView(track, pygame.Rect(W - 330, H - 330, 310, 310))
    OUT.mkdir(parents=True, exist_ok=True)
    best = None
    laps_file = OUT / "laps.jsonl"
    if laps_file.exists():
        times = [json.loads(l)["seconds"] for l in laps_file.read_text().splitlines()
                 if json.loads(l)["track"] == track_seed]
        best = min(times) if times else None

    def fresh():
        return game.init(jax.random.PRNGKey(0), track=0, random_start=False)

    st, prev, steer, acc, lap_start, path = fresh(), None, 0.0, 0.0, 0.0, []
    countdown, crashed_at = 3.0, None
    while True:
        dt = clock.tick(60) / 1000
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_r, pygame.K_SPACE) and crashed_at is not None:
                st, prev, steer, acc, lap_start, path, countdown, crashed_at = fresh(), None, 0.0, 0.0, 0.0, [], 3.0, None
        if countdown > 0:
            countdown -= dt
        elif crashed_at is None:
            acc += dt
            while acc >= cfg.dt:  # the game runs at 10 decisions per second, like the AI's
                acc -= cfg.dt
                steer, throttle = _input(pygame.key.get_pressed(), steer)
                prev = st
                st, _ = step(st, jnp.asarray([steer, throttle], jnp.float32))
                path.append((float(st.pos[0]), float(st.pos[1]), float(st.heading), float(st.speed), steer, throttle))
                if bool(st.crashed):
                    crashed_at = time.time()
                    with (OUT / "crashes.jsonl").open("a") as f:  # every try counts, for the episode
                        f.write(json.dumps({"track": track_seed, "after": round(int(st.t) * cfg.dt, 1),
                                            "when": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
                    break
                t = int(st.t) * cfg.dt
                if game.laps(st) > game.laps(prev):  # crossed the line: one full lap since the last
                    lap = t - lap_start
                    rec = {"track": track_seed, "seconds": round(lap, 2), "flying": lap_start > 0,
                           "when": time.strftime("%Y-%m-%d %H:%M"),
                           "drive": f"{track_seed}-{int(time.time())}.npy"}
                    np.save(OUT / rec["drive"], np.asarray(path, dtype=np.float32))
                    with laps_file.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    best = lap if best is None else min(best, lap)
                    lap_start, path = t, []
        # draw, interpolating between the last two physics steps for smooth motion
        a = 1.0 if prev is None or crashed_at is not None or countdown > 0 else min(1.0, acc / cfg.dt)
        p0 = np.asarray(prev.pos if prev is not None else st.pos)
        pos = p0 + (np.asarray(st.pos) - p0) * a
        h0 = float(prev.heading if prev is not None else st.heading)
        heading = h0 + (float(st.heading) - h0) * a
        screen.fill(T.BG)
        view.follow(pos, 9.0)
        view.draw_track(screen)
        view.draw_car(screen, pos, heading, color=T.AMBER)
        mini.draw_track(screen)
        mini.draw_car(screen, pos, heading, color=T.AMBER, outline=False)
        now = int(st.t) * cfg.dt - lap_start
        T.text(screen, f"{float(st.speed) * 3.6:3.0f} km/h", (40, 30), "display", 56, T.TEXT)
        T.text(screen, f"lap {now:5.1f} s", (40, 100), "semibold", 40, T.TEXT)
        T.text(screen, f"best {best:.2f} s" if best else "best: -", (40, 150), "semibold", 32, T.GREEN)
        T.text(screen, f"track {track_seed}   arrows/WASD   R restart   Esc quit", (40, H - 50), "regular", 24,
               T.MUTED)
        if countdown > 0:
            T.text(screen, str(int(countdown) + 1), (W / 2, H / 2 - 150), "display", 160, T.AMBER, anchor="center")
        if crashed_at is not None:
            T.text(screen, "CRASHED", (W / 2, H / 2 - 150), "display", 120, T.RED, anchor="center")
            T.text(screen, "press R to go again", (W / 2, H / 2 - 50), "semibold", 36, T.TEXT, anchor="center")
        window.show()
