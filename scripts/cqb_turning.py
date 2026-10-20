"""How robotic is their turning (ep13 step 8.13)? Both sides of a match, under one brain's rules, on report houses:
per operator decision while up, the turn chosen (24 degrees a decision at full rate), how much it changes from one
decision to the next (the jitter a bodycam shows), left<->right flicks (sign flips, both over 0.3), full-speed turning
and full spins (15 decisions the same way at full rate); and how much of their time they face a wall or tall furniture
from close up (what fills a bodycam's frame); how they move (speed, running, sideways or backwards). Swings (8.15): a turn one way until the facing comes back 30 deg, so 90+, 180+ and back-and-forths a minute up.

    uv run --group jax python scripts/cqb_turning.py runs/cqb-e15-2v2-home:200 runs/cqb-d24-2v2-home:300 \\
        [--rules runs/cqb-d24-2v2-home:300] [--rounds 32]"""

import argparse

import numpy as np

from ailearns import record_cqb as R
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S

HOLD = np.radians(30)   # a swing ends when the facing comes back this far from its furthest point


def swings(face: np.ndarray, alive: np.ndarray) -> list[tuple[float, int]]:
    """One operator's facing (t,) -> its swings: (degrees turned one way, decisions it took). A swing ends when the
    facing comes back HOLD from its furthest point (or he goes down), so a turn and a look back are two swings."""
    out, f = [], np.unwrap(face)
    start = ext = None
    for t in range(len(f)):
        if not alive[t]:
            if start is not None and abs(f[ext] - f[start]) > 0:
                out.append((float(np.degrees(abs(f[ext] - f[start]))), ext - start))
            start = ext = None
            continue
        if start is None:
            start = ext = t
            continue
        d = np.sign(f[ext] - f[start]) or np.sign(f[t] - f[start])
        if (f[t] - f[ext]) * d >= 0:
            ext = t
        elif abs(f[t] - f[ext]) > HOLD:
            out.append((float(np.degrees(abs(f[ext] - f[start]))), ext - start))
            start = ext
            ext = t
    if start is not None and ext != start:
        out.append((float(np.degrees(abs(f[ext] - f[start]))), ext - start))
    return out


def sweep_habits(face: np.ndarray, alive: np.ndarray, dt: float) -> dict:
    """face, alive: (rounds, t, ops). Swings of 90+ and 180+ degrees a minute up, and back-and-forths: a swing of 60+
    straight after one of 60+ the other way (a swing ends on a 30 degree turn back, so each swing has one direction)."""
    big = half = back = 0
    turned = 0.0
    for i in range(face.shape[0]):
        for k in range(face.shape[2]):
            sw = swings(face[i, :, k], alive[i, :, k])
            big += sum(a >= 90 for a, _ in sw)
            half += sum(a >= 180 for a, _ in sw)
            back += sum(a >= 60 and b >= 60 for (a, _), (b, _) in zip(sw, sw[1:]))
            turned += sum(a for a, _ in sw)
    minutes = alive.sum() * dt / 60
    return {"swing90": big / minutes, "swing180": half / minutes, "back_forth": back / minutes,
            "deg_per_s": turned / (minutes * 60)}


def habits(turn: np.ndarray, alive: np.ndarray) -> dict:
    """turn, alive: (rounds, t, ops)."""
    n = alive.sum()
    step = np.abs(turn[:, 1:] - turn[:, :-1])
    both = alive[:, 1:] & alive[:, :-1]
    flip = (np.sign(turn[:, 1:]) * np.sign(turn[:, :-1]) < 0) & (np.abs(turn[:, 1:]) > 0.3) & (np.abs(turn[:, :-1]) > 0.3)
    spins = 0
    for i in range(turn.shape[0]):
        for k in range(turn.shape[2]):
            run, sign = 0, 0
            for x, a in zip(turn[i, :, k], alive[i, :, k]):
                s_ = np.sign(x) if (abs(x) > 0.9 and a) else 0
                run = run + 1 if (s_ != 0 and s_ == sign) else (1 if s_ != 0 else 0)
                sign = s_
                spins += run == 15
    return {"turn_deg": float(np.abs(turn)[alive].mean() * 24), "change_deg": float(step[both].mean() * 24),
            "flicks": float((flip & both).sum() / both.sum()), "full_speed": float(((np.abs(turn) > 0.9) & alive).sum() / n),
            "spins_per_op_round": spins / (turn.shape[0] * turn.shape[2])}


def door_habits(event: np.ndarray, at: np.ndarray, alive: np.ndarray, dt: float, within: float = 3.0) -> dict:
    """event (rounds, t, ops): started moving a door; at (rounds, t, ops, 2): which (its middle). -> doors moved a minute
    up, and the share moved again by the same operator within `within` s (open-close-open: a loop)."""
    n = again = 0
    w = int(round(within / dt))
    for i in range(event.shape[0]):
        for k in range(event.shape[2]):
            ts = np.flatnonzero(event[i, :, k])
            n += len(ts)
            again += sum(t1 - t0 <= w and np.allclose(at[i, t0, k], at[i, t1, k], atol=0.05) for t0, t1 in zip(ts, ts[1:]))
    return {"per_min": n / max(alive.sum() * dt / 60, 1e-9), "again": again / max(n, 1)}


def ahead(pos: np.ndarray, face: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Distance straight ahead to the nearest box (lo, hi: (boxes, 2)) from pos (..., 2) along face (...)."""
    d = np.stack([np.cos(face), np.sin(face)], -1)[..., None, :]
    p = pos[..., None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = (lo - p) / d, (hi - p) / d
    tn = np.nanmax(np.minimum(t0, t1), -1)
    tf = np.nanmin(np.maximum(t0, t1), -1)
    hit = (tf >= np.maximum(tn, 0)) & (tf > 0)
    return np.where(hit, np.maximum(tn, 0), np.inf).min(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assault")
    ap.add_argument("defense")
    ap.add_argument("--rules", default="runs/cqb-d24-2v2-home:300")
    ap.add_argument("--rounds", type=int, default=32)
    ap.add_argument("--web", default="3,12,25", help="also these web page homes alone (3d.html?home&house=k is seed 100064+k)")
    a = ap.parse_args()
    cfg = R.game_config(a.rules)
    seeds = list(cqb.TEST_SEEDS)[64:64 + a.rounds]
    g = cqb.CQBGame(cqb.make_buildings(seeds, cfg), cfg)
    r = S.play_rounds(g, R.player(a.assault, 0), R.player(a.defense, g.nA), np.arange(len(seeds)), seed=5000, record=True)
    h = r["history"]
    turn = np.asarray(h.turn)[:, 1:]
    alive = np.asarray(h.alive)[:, :-1] & ~np.asarray(h.done)[:, :-1, None]
    # facing a wall or tall furniture (not a door) closer than 60 cm, from the body's centre: what fills a bodycam frame
    pos, face = np.asarray(h.pos)[:, :-1], np.asarray(h.face)[:, :-1]
    wall_dist = np.stack([ahead(pos[i], face[i], np.concatenate([g.b.wall_lo[i], g.b.cover_lo[i]]),
                                np.concatenate([g.b.wall_hi[i], g.b.cover_hi[i]])) for i in range(len(seeds))])
    # moving: how fast, how much of it running, and how much sideways or backwards (more than 60 degrees off the facing)
    vel = (np.asarray(h.pos)[:, 1:] - np.asarray(h.pos)[:, :-1]) / cfg.dt
    speed = np.linalg.norm(vel, axis=-1)
    off = np.abs(np.angle(np.exp(1j * (np.arctan2(vel[..., 1], vel[..., 0]) - face))))
    print(f"Assault won {np.mean(np.asarray(r['outcome']) == 1):.0%} of {len(seeds)} rounds under {a.rules}")
    # (8.16) defenders seen by attackers in the first 3 s (from outside: through windows, or a door just opened), down early
    seen = np.asarray(h.seen)
    early = seen[:, :30, :g.nA, g.nA:].any((1, 2, 3))
    down3 = ~np.asarray(h.alive)[:, min(30, seen.shape[1] - 1), g.nA:].all(-1)
    print(f"a defender seen by an attacker in the first 3 s: {early.mean():.0%} of rounds; a defender down by 3 s: "
          f"{down3.mean():.0%} (of those, seen in the first 3 s: {(early & down3).sum()}/{down3.sum()})")
    for side, name, ops in (("Assault", a.assault, list(range(g.nA))), ("Defense", a.defense, list(range(g.nA, g.N)))):
        m = habits(turn[:, :, ops], alive[:, :, ops])
        print(f"{side:8s} {name:32s} turn {m['turn_deg']:4.1f} deg/decision  change {m['change_deg']:4.1f} deg  "
              f"flicks {m['flicks']:.0%}  full-speed {m['full_speed']:.0%}  spins/op-round {m['spins_per_op_round']:.2f}  "
              f"facing a wall <0.6 m {(wall_dist[:, :, ops] < 0.6)[alive[:, :, ops]].mean():.0%} "
              f"(<1 m {(wall_dist[:, :, ops] < 1.0)[alive[:, :, ops]].mean():.0%}) of their time up")
        up, mv = alive[:, :, ops], alive[:, :, ops] & (speed[:, :, ops] > 0.3)
        sw = sweep_habits(face[:, :, ops], alive[:, :, ops], cfg.dt)
        print(f"{'':41s} turning {sw['deg_per_s']:.0f} deg/s up; a minute up: swings of 90+ deg {sw['swing90']:.1f}, "
              f"180+ {sw['swing180']:.1f}, back-and-forths (60+ then 60+ back) {sw['back_forth']:.1f}")
        for k in (int(x) for x in a.web.split(",") if x) if side == "Assault" else ():
            i = seeds.index(100_064 + k) if 100_064 + k in seeds else None
            if i is not None:
                w = sweep_habits(face[i:i + 1][:, :, ops], alive[i:i + 1][:, :, ops], cfg.dt)
                print(f"{'':41s}   web house {k:2d}: {w['deg_per_s']:.0f} deg/s, 90+ {w['swing90']:.1f}/min, "
                      f"180+ {w['swing180']:.1f}/min, back-and-forths {w['back_forth']:.1f}/min, "
                      f"{alive[i][:, ops].sum(0).max() * cfg.dt:.0f} s up")
        dm = door_habits(np.asarray(h.door_event)[:, 1:][:, :, ops], np.asarray(h.door_at)[:, 1:][:, :, ops],
                         alive[:, :, ops], cfg.dt)
        print(f"{'':41s} doors: {dm['per_min']:.1f} moved a minute up; {dm['again']:.0%} of them the same door again by "
              f"the same operator within 3 s (a door loop)")
        print(f"{'':41s} moving {mv.sum() / up.sum():.0%} of the time, {speed[:, :, ops][mv].mean():.2f} m/s; faster than a "
              f"walk {(speed[:, :, ops] > cfg.walk * 1.1)[mv].mean():.0%} of it; sideways or backwards {(off[:, :, ops] > np.pi / 3)[mv].mean():.0%}")


if __name__ == "__main__":
    main()
