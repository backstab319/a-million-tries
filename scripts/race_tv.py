"""Race day (episode 15): a full race as TV shows it. The circuit from above with every car, and a timing tower:
position, lap, distance behind the leader, tyres (compound and wear), fuel left, stops, and who's in the pits, out,
wrecked or dry. Pure drawing from `scripts/race_full.py --save`; nothing is simulated here.
(Episode 16) A race recorded with race control (`scripts/race_incidents.py --save`) also shows it: the race's status
(green, VSC, Safety Car, "in this lap", red flag) and a closed pit entry, each yellow / double yellow sector on the
map, the Safety Car, and the stewards' penalties and disqualifications in the tower (classified once the leader is
home).

    uv run python scripts/race_tv.py out/ep15/race-race-fin-4-100.npz [--circuit 0] [--speed 20] [--still 60] [--start 340 --end 460]
    -> out/ep15/tv-<race>-c<circuit>.mp4 (or .png with --still <race seconds>); out/ep16/ for race control
"""

import argparse
from pathlib import Path

import numpy as np
import pygame

from ailearns.games import race
from ailearns.games import race_car as rc
from ailearns.media.video import VideoWriter
from ailearns.render import theme as T
from ailearns.render.race_view import TARMAC, WALL, CircuitView

ap = argparse.ArgumentParser()
ap.add_argument("race")
ap.add_argument("--circuit", type=int, default=0, help="which of the saved races")
ap.add_argument("--speed", type=float, default=20.0, help="race seconds per video second")
ap.add_argument("--still", type=float, default=None, help="one frame at this many race seconds -> png")
ap.add_argument("--follow", type=int, default=0, help="the battle camera stays on this car (its tower number)")
ap.add_argument("--start", type=float, default=0.0, help="video from this many race seconds")
ap.add_argument("--end", type=float, default=None, help="video to this many race seconds (default: the race's end)")
ap.add_argument("--driver", default="", help="named on screen (default: race-fin-5, or race-fin-7 with race control)")
ap.add_argument("--strategist", default="strat-2:43", help="named on screen, when the race has two teams")
ap.add_argument("--note", default="", help="instead of who calls the stops (step 7's grid: no pit stops yet)")
ap.add_argument("--title", default="", help="the tower's title (default: RACE DAY, or RACE DAY 2 with race control)")
a = ap.parse_args()

z = np.load(a.race)
k = a.circuit
seed, dt, laps, L = int(z["seeds"][k]), float(z["dt"]), int(z["laps"]), float(z["length"][k])
pos, yaw, prog = z["pos"][k], z["yaw"][k], z["progress"][k]
order, fin, walled, wrecked, dry = z["order"][k], z["finished"][k], z["walled"][k], z["wrecked"][k], z["dry"][k]
pit, stops, wear, fuel, bump, comp = z["pit"][k], z["stops"][k], z["wear"][k], z["fuel"][k], z["bump"][k], z["compound"][k]
n, N = pos.shape[:2]
team = z["team"] if "team" in z.files else np.zeros(N, bool)
gone = z["gone"][k] if "gone" in z.files else np.zeros((n, N), bool)  # (step 9l) recovered by the marshals: not drawn  # (step 9j) cars run by the strategist; else all the rule
# (ep16) race control, if the race had it
CONTROL = "status" in z.files
if CONTROL:
    status, flags, sc_pos, sc_yaw = z["status"][k], z["flags"][k], z["sc_pos"][k], z["sc_yaw"][k]
    sc_on, penalty, pit_closed, dsq, classified = z["sc_on"][k], z["penalty"][k], z["pit_closed"][k], z["dsq"][k], \
        z["classified"][k]
    black = z["black"][k] if "black" in z.files else np.zeros_like(dsq)  # (step 9h) recorded from then on
    n_sec = int(z["sectors"][k])
    sec_len = L / n_sec
over = np.nonzero((fin | walled | wrecked).all(1))[0]
end = int(over[0]) + int(4 / dt) if len(over) else n - 1  # a few seconds after the last car is home or out
end = min(end, n - 1)

# the lane the race ran on (episode 15's recordings don't say: one 7 m lane, boxes 12 m apart)
cfg = race.RaceConfig(pit_lane=True, **({k: float(z[k]) for k in ("pit_outer", "pit_fast") if k in z}
                                        or {"pit_outer": -14.0, "pit_fast": -14.0}))
BOX_GAP = float(z["box_gap"]) if "box_gap" in z else 12.0
W, H = T.WIDTH, T.HEIGHT
T.init()
view = CircuitView(race.make_track(seed, cfg), cfg, pygame.Rect(0, 0, 1180, 720))
# (owner: an overtake "morphs through" the other car) on the overview a car is drawn ~17x its size (the track is ~3 px
# wide), so a clean side-by-side pass 1 m apart looks like one car driving through another. TV's answer: a second
# camera at real scale on the closest fight (or the leader)
BATTLE = pygame.Rect(20, 735, 1140, 300)
battle = CircuitView(race.make_track(seed, cfg), cfg, BATTLE)
BATTLE_ZOOM = 11.0  # px per metre: the inset shows ~100 x 27 m
BATTLE_MIN_ZOOM = 3.0  # ... zoomed out to fit a pair: ~380 x 100 m
circuit = pygame.Surface((W, H))
circuit.fill(T.BG)
view.draw(circuit)  # the circuit doesn't move: draw it once


# (ep16) with race control, amber means a yellow flag: the pit lane's outline goes grey (episode 15's footage unchanged)
PIT_EDGE = T.MUTED if CONTROL else T.AMBER


def draw_pits(surf, track, cfg, n_cars: int, box_gap: float = 12.0, view=view, label: bool = True) -> None:
    """The pit lane as the sim has it (race.RaceConfig): tarmac from the track's right edge out to `pit_outer`, from
    the entry to the exit around the circuit's straightest stretch, the pit wall's line, and a box per car (its
    colour) `box_gap` apart along the lane (race_field.RaceField's boxes)."""
    tr = race.make_tracks([seed], cfg)
    mid, P = float(race.pit_spots(tr, cfg)[0]), len(track["center"])
    c, nrm = np.asarray(track["center"]), np.asarray(track["normal"])

    def at(m, lat):  # metres along the circuit, metres left of the centreline -> world
        i = int(round(m / L * P)) % P
        return c[i] + nrm[i] * lat

    ms = np.linspace(mid - cfg.pit_entry, mid + cfg.pit_exit, 60)
    lane = [view.px(at(m, -cfg.width / 2)) for m in ms] + [view.px(at(m, cfg.pit_outer)) for m in ms[::-1]]
    pygame.draw.polygon(surf, TARMAC, lane)
    pygame.draw.polygon(surf, PIT_EDGE, lane, 2)  # outlined: at the overview's scale the lane is a few pixels wide
    wall = [view.px(at(m, cfg.pit_wall_lat)) for m in np.linspace(mid - cfg.pit_wall_len, mid + cfg.pit_wall_len, 30)]
    pygame.draw.lines(surf, WALL, False, wall, 2)
    for b in range(n_cars):
        m = mid + (b - (n_cars - 1) / 2) * box_gap
        pygame.draw.circle(surf, T.CARS[b], view.px(at(m, cfg.box_lat if cfg.pit_fast > cfg.pit_outer
                                                       else cfg.pit_outer + 1.0)), 3)
    out = view.px(at(mid, cfg.pit_outer - 1)) - view.px(at(mid, cfg.pit_outer))  # away from the track, on screen
    label_at = view.px(at(mid, cfg.pit_outer)) + out / (np.hypot(*out) or 1) * 40
    if not view.rect.inflate(-60, -60).collidepoint(label_at):  # off the map's edge: put it on the inside instead
        label_at = view.px(at(mid, cfg.pit_outer)) - out / (np.hypot(*out) or 1) * 60
    if label:
        T.text(surf, "PIT", label_at, "semibold", 22, PIT_EDGE, anchor="center")


draw_pits(circuit, race.make_track(seed, cfg), cfg, N, BOX_GAP)
design = np.asarray(rc.F1_LIKE[0])
TYRE = [T.RED, T.AMBER, T.TEXT]  # soft, medium, hard (TV's colours)


def at(i: float):
    """Every car's position and heading at frame i (between saved frames: blended)."""
    j, u = int(i), i - int(i)
    j2 = min(j + 1, n - 1)
    p = pos[j] * (1 - u) + pos[j2] * u
    dy = (yaw[j2] - yaw[j] + np.pi) % (2 * np.pi) - np.pi
    return p, yaw[j] + dy * u, j


def sc_at(i: float):
    """The Safety Car's position and heading at frame i (blended, as `at`)."""
    j, u = int(i), i - int(i)
    j2 = min(j + 1, n - 1)
    dy = (sc_yaw[j2] - sc_yaw[j] + np.pi) % (2 * np.pi) - np.pi
    return sc_pos[j] * (1 - u) + sc_pos[j2] * u, sc_yaw[j] + dy * u


# the race's status as TV shows it: words, colour (race_control.RACING, VSC, SC, SC_IN, RED)
STATUS = [("GREEN FLAG", T.GREEN), ("VIRTUAL SAFETY CAR", T.AMBER), ("SAFETY CAR", T.AMBER),
          ("SAFETY CAR IN THIS LAP", T.AMBER), ("RED FLAG", T.RED)]


def draw_flags(surf, j: int) -> None:
    """Each yellow sector over the track on the map (double yellow: a wide band split by a dark line, two flags);
    a red flag tints the whole track red, the VSC and the Safety Car amber (every post shows yellow)."""
    c = np.asarray(view.track["center"])
    P = len(c)
    st = int(status[j])
    if st == 4:
        pygame.draw.lines(surf, T.RED, True, [view.px(q) for q in c[::4]], 7)
    elif st in (1, 2, 3):  # VSC / Safety Car: yellow at every marshal post (F1), so the whole track
        pygame.draw.lines(surf, T.AMBER, True, [view.px(q) for q in c[::4]], 7)
    for sec in range(n_sec):
        lv = int(flags[j, sec])
        if lv == 0 or st != 0:
            continue
        idx = np.arange(int(sec * sec_len / L * P), int((sec + 1) * sec_len / L * P) + 1) % P
        pts = [view.px(q) for q in c[idx]]
        pygame.draw.lines(surf, T.AMBER, False, pts, 11 if lv == 2 else 7)
        if lv == 2:
            pygame.draw.lines(surf, T.BG, False, pts, 3)


def status_board(surf, j: int, x: int, y: int) -> None:
    """The race's status and a closed pit entry, a line in the tower (a board on the map
    covered the track and the cars on some circuits)."""
    st = int(status[j])
    words, col = STATUS[st]
    if st == 0 and flags[j, :n_sec].any():
        secs = [str(s_ + 1) for s_ in range(n_sec) if flags[j, s_]]
        words, col = ("DOUBLE YELLOW" if (flags[j] == 2).any() else "YELLOW") + " · sector " + ", ".join(secs), T.AMBER
    pygame.draw.rect(surf, col, (x, y + 4, 12, 28), border_radius=4)
    T.text(surf, words, (x + 24, y), "semibold", 28, col)
    if pit_closed[j]:
        T.text(surf, "PIT ENTRY CLOSED", (x + 630, y + 6), "semibold", 20, T.RED, anchor="topright")


def bar(surf, x, y, w, v, color):
    pygame.draw.rect(surf, T.PANEL_EDGE, (x, y, w, 10), border_radius=5)
    pygame.draw.rect(surf, color, (x, y, max(2, int(w * float(np.clip(v, 0, 1)))), 10), border_radius=5)


cam = {"pair": None, "zoom": None}


def battle_cam(surf, p, h, j, out) -> None:
    """The closest fight between running cars (kept while they stay within 90 m), else the leader, at real scale.
    (Owner: cars kept leaving the frame. The camera trailed them, easing 20% a frame: at 20x a car moves ~30 m a frame,
    so it sat ~120 m behind in a ~104 x 27 m inset. Now it sits on the pair and zooms out to fit both.)"""
    run = [c for c in range(N) if not (out[c] or fin[j, c])]
    if a.follow:  # one car, whatever it's doing (footage of one car's story)
        cam["pair"] = None
        run = []
    pair = cam["pair"]
    if pair is None or not all(c in run for c in pair) or np.hypot(*(p[pair[0]] - p[pair[1]])) > 90:
        best = min(((np.hypot(*(p[a] - p[b])), (a, b)) for a in run for b in run if a < b), default=(1e9, None))
        pair = best[1] if best[0] < 80 else None
        cam["pair"] = pair
    lead = a.follow - 1 if a.follow else int(np.argmin(order[j]))
    target = (p[pair[0]] + p[pair[1]]) / 2 if pair else p[lead]
    span = np.abs(p[pair[0]] - p[pair[1]]) if pair else np.zeros(2)
    fit = float(np.clip(min(BATTLE.w / (span[0] + 30), BATTLE.h / (span[1] + 30)), BATTLE_MIN_ZOOM, BATTLE_ZOOM))
    z = cam["zoom"] or fit  # out at once (both cars stay in), back in gently
    cam["zoom"] = min(fit, z + (fit - z) * 0.1)
    T.panel(surf, BATTLE.inflate(8, 8), radius=12)
    battle.follow(target, cam["zoom"])
    battle.draw(surf)
    clip = surf.get_clip()
    surf.set_clip(BATTLE)
    draw_pits(surf, battle.track, cfg, N, BOX_GAP, view=battle, label=False)
    for c in sorted(range(N), key=lambda c: -order[j, c]):
        if gone[j, c]:
            continue
        col = T.mix(T.CARS[c], T.BG, 0.6) if out[c] or fin[j, c] else T.CARS[c]
        battle.draw_car(surf, design, p[c], h[c], color=col)
        if BATTLE.collidepoint(battle.px(p[c])):
            T.text(surf, f"{c + 1}", battle.px(p[c]) + [0, -40], "semibold", 20, col, anchor="center")
    if CONTROL and sc_on[j]:
        q, qh = sc_at(i_now[0])
        battle.draw_car(surf, design, q, qh, color=T.TEXT)
        if BATTLE.collidepoint(battle.px(q)):
            T.text(surf, "SC", battle.px(q) + [0, -40], "semibold", 20, T.TEXT, anchor="center")
    surf.set_clip(clip)
    who = (f"BATTLE · car {pair[0] + 1} v car {pair[1] + 1}" if pair else f"{'CAMERA' if a.follow else 'LEADER'} · car {lead + 1}")
    T.text(surf, who, (BATTLE.x + 14, BATTLE.y + 10), "semibold", 22, T.TEXT)


i_now = [0.0]  # the frame being drawn (the battle camera draws the Safety Car from it)


def frame(surf, i: float) -> None:
    p, h, j = at(i)
    i_now[0] = i
    surf.blit(circuit, (0, 0))
    if CONTROL:
        draw_flags(surf, j)
    out = walled[j] | wrecked[j] | (dry[j] & ~fin[j])
    for c in sorted(range(N), key=lambda c: -order[j, c]):  # the leader drawn last (on top)
        if gone[j, c]:
            continue
        col = T.mix(T.CARS[c], T.BG, 0.6) if out[c] or fin[j, c] else T.CARS[c]
        if bump[j, c]:
            T.glow(surf, view.px(p[c]), 16, T.TEXT, strength=50)
        view.draw_car(surf, design, p[c], h[c], color=col, min_px=22)
    if CONTROL and sc_on[j]:
        q, qh = sc_at(i)
        view.draw_car(surf, design, q, qh, color=T.TEXT, min_px=22)
        T.text(surf, "SC", view.px(q) + [0, -26], "semibold", 18, T.TEXT, anchor="center")
    battle_cam(surf, p, h, j, out)
    # the timing tower
    x0 = 1200
    T.panel(surf, pygame.Rect(x0, 30, 690, 1020))
    lead = prog[j].max()
    lap = min(int(lead // L) + 1, laps)
    T.text(surf, a.title or ("RACE DAY 2" if CONTROL else "RACE DAY"), (x0 + 30, 50), "display", 44, T.TEXT)
    T.text(surf, f"LAP {lap}/{laps}" if not fin[j].any() else "FINISHED", (x0 + 660, 56), "semibold", 34,
           T.AMBER if fin[j].any() else T.TEXT, anchor="topright")
    t = j * dt
    T.text(surf, f"{int(t // 60)}:{int(t % 60):02d}  ·  circuit {seed}  ·  {L / 1000:.1f} km", (x0 + 30, 110),
           "semibold", 24, T.MUTED)
    # (ep16) once the leader is home, the classification after the stewards (penalties, DSQ last)
    rank = classified[j] if CONTROL and fin[j].any() else order[j]
    top = 170
    if CONTROL:
        status_board(surf, j, x0 + 30, 146)
        top = 205
    for c in np.argsort(rank):
        r = int(rank[c])
        y = top + r * 106
        T.text(surf, f"{r + 1}", (x0 + 50, y + 18), "display", 40, T.TEXT, anchor="midtop")
        pygame.draw.rect(surf, T.CARS[c], (x0 + 84, y + 14, 10, 54), border_radius=4)
        T.text(surf, f"Car {c + 1}", (x0 + 110, y + 10), "semibold", 30, T.TEXT)
        if team.any():
            T.text(surf, "AI" if team[c] else "RULE", (x0 + 215, y + 18), "semibold", 20,
                   T.GREEN if team[c] else T.MUTED)
        gap = lead - prog[j, c]
        state = ("BLACK FLAG" if CONTROL and black[j, c] else "DSQ" if CONTROL and dsq[j, c] else "FINISHED" if fin[j, c] else "WRECKED" if wrecked[j, c] else "OUT" if walled[j, c]
                 else "NO FUEL" if dry[j, c] else "IN PIT" if pit[j, c] else
                 "leader" if r == 0 else f"+{gap / L:.1f} laps" if gap >= L else f"+{gap:,.0f} m")
        sc = (T.RED if CONTROL and dsq[j, c] else T.GREEN if fin[j, c] else T.RED if out[c] else T.AMBER if pit[j, c] else T.MUTED)
        T.text(surf, state, (x0 + 110, y + 50), "semibold", 22, sc)
        if CONTROL and penalty[j, c] > 0:
            T.text(surf, f"+{penalty[j, c]:.0f} s", (x0 + 290, y + 16), "semibold", 22, T.RED, anchor="topright")
        T.text(surf, rc.COMPOUNDS[int(comp[j, c])][0].upper(), (x0 + 330, y + 12), "display", 34,
               TYRE[int(comp[j, c])], anchor="midtop")
        bar(surf, x0 + 360, y + 22, 120, 1 - wear[j, c], TYRE[int(comp[j, c])])
        T.text(surf, "tyres", (x0 + 360, y + 40), "semibold", 18, T.MUTED)
        bar(surf, x0 + 500, y + 22, 100, fuel[j, c], T.BLUE if fuel[j, c] > 0.1 else T.RED)
        T.text(surf, "fuel", (x0 + 500, y + 40), "semibold", 18, T.MUTED)
        T.text(surf, f"{int(stops[j, c])}", (x0 + 650, y + 12), "display", 34, T.TEXT, anchor="midtop")
    T.text(surf, "stops", (x0 + 650, top - 20), "semibold", 18, T.MUTED, anchor="midtop")
    ai = np.nonzero(team)[0] + 1
    who = (f"pit calls: AI strategist {a.strategist} (cars {ai[0]}-{ai[-1]}) v the rule" if team.any()
           else "the team calls the pit stops")
    who = a.note or who
    driver = a.driver or ("race-fin-7, blind to race control" if CONTROL else "race-fin-5")
    T.text(surf, f"one AI driver ({driver}) in every car · {who} · {a.speed:.0f}x speed",
           (590, 712), "semibold", 20, T.MUTED, anchor="center")


name = Path(a.race).stem if Path(a.race).stem != "race" else Path(a.race).parent.name  # runs/<name>/race.npz
OUT = "out/ep16" if CONTROL else "out/ep15"
Path(OUT).mkdir(parents=True, exist_ok=True)
surf = pygame.Surface((W, H))
if a.still is not None:
    frame(surf, min(a.still / dt, end))
    out = f"{OUT}/tv-{name}-c{k}.png"
    pygame.image.save(surf, out)
else:
    window = f"-{a.start:.0f}-{a.end:.0f}" if a.start or a.end is not None else ""
    window += f"-car{a.follow}" if a.follow else ""
    out = f"{OUT}/tv-{name}-c{k}{window}.mp4"
    stop = min(a.end / dt, end) if a.end is not None else end
    writer = VideoWriter(out, (W, H))
    try:
        for f in range(int((stop - a.start / dt) * dt / a.speed * T.FPS)):
            frame(surf, a.start / dt + f / T.FPS * a.speed / dt)
            writer.write(surf)
    finally:
        writer.close()
print(out)
