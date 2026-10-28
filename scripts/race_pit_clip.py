"""Race day (episode 15), footage for the pit saga (beat 4): one called-in car from the race school, from above at real
scale, its own box ringed. The same stretches as `scripts/race_box_calls.py` (every car called in at the start, one
brain driving every car, best guess, no noise); this only records them and draws one car.

    uv run --group jax python scripts/race_pit_clip.py runs/race-field-5 --lessons 1 --pick past   # beside its box, rolls by
    uv run --group jax python scripts/race_pit_clip.py runs/race-bc-2 --lessons 0 --pick stop      # from the racing line, stops
    -> out/ep15/pit-<run>-<pick>.mp4

--pick past: a car that started in the lane and never stopped; stop: the first car that entered the lane from the
track and stopped on its mark; skip: a car that reached the pit zone and never entered the lane.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from ailearns.agents.ppo_car import forward, load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_field, race_school
from ailearns.games import race_car as rc
from ailearns.media.video import VideoWriter
from ailearns.render import theme as T
from ailearns.render.race_view import TARMAC, CircuitView

ap = argparse.ArgumentParser()
ap.add_argument("brain")
ap.add_argument("--lessons", type=float, default=1.0)
ap.add_argument("--pick", choices=["past", "stop", "skip"], default="past")
ap.add_argument("--races", type=int, default=16)
ap.add_argument("--tracks", type=int, default=8)
ap.add_argument("--seconds", type=float, default=30.0)
ap.add_argument("--nth", type=int, default=0, help="which matching car (0: the first)")
ap.add_argument("--title", default="")
ap.add_argument("--zoom", type=float, default=9.0, help="px per metre")
a = ap.parse_args()

cfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
seeds = list(race.TEST_SEEDS)[:a.tracks]
F = race_field.RaceField(race.make_tracks(seeds, cfg), cfg=cfg)
S = race_school.RaceSchool(F, race_school.SchoolConfig(stretch_s=a.seconds, grid_share=0.0, call_rate_s=1e9,
                                                         pit_share=a.lessons))
g, n, steps = F.game, F.n, int(a.seconds / cfg.dt)
path, _, ck = a.brain.partition(":")
params, meta = load(pick_checkpoint(path, ck or "last"))


def one(key):
    les = S.start(key)
    les = les._replace(field=les.field._replace(box_call=jnp.ones(n, bool)))

    def tick(les, k):
        mean, _, _ = forward(params, S.observe(les))
        les, *_ = S.step(les, jnp.clip(mean, -1, 1), k)
        f = les.field
        c = f.cars
        return les, (c.car.pos, c.car.yaw, jnp.hypot(c.car.vx, c.car.vy), jax.vmap(g.in_pit_lane)(c),
                     jax.vmap(g.in_pit_zone)(c.track, c.idx), f.stops, c.crashed | f.wrecked, c.track)
    return jax.lax.scan(tick, les, jax.random.split(key, steps))[1]


pos, yaw, v, lane, zone, stops, out, track = (np.asarray(x) for x in jax.jit(jax.vmap(one))(
    jax.random.split(jax.random.key(5), a.races)))   # (races, steps, cars, ...)
found = []
for r in range(a.races):
    for c in range(n):
        ok = {"past": lane[r, 0, c] and stops[r, -1, c] == 0 and not out[r, -1, c],
              "stop": not lane[r, 0, c] and stops[r, -1, c] > 0,
              "skip": zone[r, :, c].any() and not zone[r, :int(4 / cfg.dt), c].any() and not lane[r, :, c].any()
                      and not out[r, -1, c]}   # reaches the entry on camera[a.pick]
        if ok:
            found.append((r, c))
print(f"{a.brain}: {len(found)} cars match '{a.pick}' of {a.races * n}")
r, c = found[a.nth]
seed = seeds[int(track[r, 0, c])]
# the stretch around what matters: 'stop' from 6 s before it enters the lane to 4 s after it leaves its box
j0, j1 = 0, steps - 1
if a.pick == "stop":
    j_in = int(np.argmax(lane[r, :, c]))
    j_go = int(np.argmax(stops[r, :, c] > 0))
    j0, j1 = max(0, j_in - int(6 / cfg.dt)), min(steps - 1, j_go + int(6 / cfg.dt))
elif a.pick == "skip":
    j_z = int(np.argmax(zone[r, :, c]))
    j0, j1 = max(0, j_z - int(4 / cfg.dt)), min(steps - 1, j_z + int(10 / cfg.dt))
else:
    j1 = min(steps - 1, int(13 / cfg.dt))
print(f"race {r} car {c + 1} circuit {seed}: {j0 * cfg.dt:.1f}-{j1 * cfg.dt:.1f} s, stopped {stops[r, j1, c] > 0}")

T.init()
W, H = T.WIDTH, T.HEIGHT
tr = race.make_track(seed, cfg)
view = CircuitView(tr, cfg, pygame.Rect(0, 0, W, H))
L, P = float(tr["length"]), len(tr["center"])
mid = float(race.pit_spots(race.make_tracks([seed], cfg), cfg)[0])
cen, nrm = np.asarray(tr["center"]), np.asarray(tr["normal"])
box_s = np.asarray(F.box_s)


def at(m, lat):
    i = int(round(m / L * P)) % P
    return cen[i] + nrm[i] * lat


def draw_pits(surf):
    ms = np.linspace(mid - cfg.pit_entry, mid + cfg.pit_exit, 120)
    lane = [view.px(at(m, -cfg.width / 2)) for m in ms] + [view.px(at(m, cfg.pit_outer)) for m in ms[::-1]]
    pygame.draw.polygon(surf, TARMAC, lane)
    pygame.draw.polygon(surf, T.AMBER, lane, 3)   # outlined, as on the TV view
    wall = [view.px(at(m, cfg.pit_wall_lat)) for m in np.linspace(mid - cfg.pit_wall_len, mid + cfg.pit_wall_len, 60)]
    pygame.draw.lines(surf, T.TEXT, False, wall, 5)   # the pit wall, bright: at this scale WALL vanishes
    for b in range(n):
        p0, p1 = view.px(at(mid + box_s[b] - 2.5, cfg.pit_outer - 0.5)), view.px(at(mid + box_s[b] + 2.5, cfg.pit_outer - 0.5))
        pygame.draw.line(surf, T.CARS[b], p0, p1, 6 if b == c else 3)
    T.glow(surf, view.px(at(mid + box_s[c], cfg.pit_outer - 2)), 34, T.CARS[c], strength=60)
    T.text(surf, "ITS BOX", view.px(at(mid + box_s[c], cfg.pit_outer + 4)), "semibold", 28, T.CARS[c], anchor="center")


design = np.asarray(rc.F1_LIKE[0])
name = f"pit-{Path(path).name}-{a.pick}"
dest = Path("out/ep15") / f"{name}.mp4"
writer = VideoWriter(str(dest), (W, H))
surf = pygame.Surface((W, H))
cam = None
try:
    for fr in range(int((j1 - j0) * cfg.dt * T.FPS)):
        i = j0 + fr / T.FPS / cfg.dt
        j, u = int(i), i - int(i)
        j2 = min(j + 1, j1)
        p = pos[r, j] * (1 - u) + pos[r, j2] * u
        dy = (yaw[r, j2] - yaw[r, j] + np.pi) % (2 * np.pi) - np.pi
        h = yaw[r, j] + dy * u
        cam = p[c] if cam is None else cam + (p[c] - cam) * 0.15
        surf.fill(T.BG)
        view.follow(cam, a.zoom)
        view.draw(surf)
        draw_pits(surf)
        for k in range(n):
            col = T.CARS[k] if k == c else T.mix(T.CARS[k], T.BG, 0.5)
            view.draw_car(surf, design, p[k], h[k], color=col)
        pygame.draw.circle(surf, T.TEXT, view.px(p[c]), 30, 2)
        T.panel(surf, pygame.Rect(40, H - 170, 620, 130), radius=14)
        state = ("STOPPED ON ITS MARK" if stops[r, j, c] > 0 else "IN THE PIT LANE" if lane[r, j, c] else "ON TRACK")
        T.text(surf, f"CALLED IN · {state}", (64, H - 150), "semibold", 32,
               T.GREEN if stops[r, j, c] > 0 else T.AMBER if lane[r, j, c] else T.TEXT)
        T.text(surf, f"{v[r, j, c] * 3.6:.0f} km/h", (64, H - 100), "display", 44, T.TEXT)
        if a.title:
            T.text(surf, a.title, (W // 2, 50), "semibold", 36, T.TEXT, anchor="midtop")
        writer.write(surf)
finally:
    writer.close()
print("->", dest)
