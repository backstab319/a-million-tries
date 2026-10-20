"""The window-spawn bug for ep13's 2D picture: one home, its defender spots before the fix (143ebbe) and after.

The fix redraws a spot from its own random stream, so stubbing `_seen_from` to "never seen" gives the old spots in the
same layout. Of the web game's homes (seed 100064 + k), picks, of the moved spots an attacker start sees
within 6 m, the one whose new spot is hidden from that start by the plainest wall (furthest from a door they open
seconds later, or a window, which reads as a near miss), and writes
the house (walls, doors, windows, furniture), the attackers' starts and both spots to JSON.

    uv run --group jax python scripts/cqb_window_spawn.py runs/cqb-e17-2v2-home-doors:150 \\
        --out episodes/13-the-house/manim/window_spawn.json
"""

import argparse
import json

import numpy as np

from ailearns import record_cqb as R
from ailearns.games import cqb_home as H


def eyes_of(b, cfg):
    return [(i, e) for i, (p, f, _) in enumerate(b["entries"])
            for e in (p, p - np.array([np.cos(f), np.sin(f)]) * cfg.stack_gap)]


def blockers(b):
    walls = list(zip(b["wall_lo"], b["wall_hi"]))
    tall = list(zip(b["cover_lo"], b["cover_hi"]))
    doors = [(d["lo"], d["hi"]) for d in b["doors"]]
    lo, hi = zip(*(walls + tall + doors))
    return np.array(lo, float), np.array(hi, float)


def first_block(e, q, lo, hi) -> tuple[int, np.ndarray]:
    """The index (into blockers) of the first box the line e -> q enters, and where."""
    d = q - e
    with np.errstate(divide="ignore", invalid="ignore"):
        t0, t1 = (lo - e) / d, (hi - e) / d
    tn, tf = np.nanmax(np.minimum(t0, t1), -1), np.nanmin(np.maximum(t0, t1), -1)
    tn = np.where((tf >= np.maximum(tn, 0)) & (tn <= 1), tn, np.inf)
    i = int(np.argmin(tn))
    return i, e + d * tn[i]


def plain_wall(e, q, b, lo, hi) -> float:
    """How far from any door (one the attackers open seconds later) or window (a picture that reads as a near miss)
    the wall is that hides the new spot; 0 if it's hidden by something else."""
    i, at = first_block(e, q, lo, hi)
    gaps = [(d["lo"], d["hi"]) for d in b["doors"]] + list(zip(b["win_lo"], b["win_hi"]))
    near = min(np.linalg.norm(at - np.clip(at, l, h)) for l, h in gaps)
    return float(near) if i < len(b["wall_lo"]) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rules")
    ap.add_argument("--houses", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = R.game_config(a.rules)
    seen = H._seen_from
    best = None
    for k in range(a.houses):
        new = H.make_home(100_064 + k, cfg)
        H._seen_from = lambda *_: False
        old = H.make_home(100_064 + k, cfg)
        H._seen_from = seen
        assert np.allclose(old["wall_lo"], new["wall_lo"]), "the layout changed: the stub broke the random streams"
        lo, hi = blockers(new)
        for s, (p, q) in enumerate(zip(old["spawn"], new["spawn"])):
            if np.allclose(p, q):
                continue
            for i, e in eyes_of(new, cfg):
                dist = float(np.linalg.norm(p - e))
                if dist < 6 and seen(p, [e], lo, hi):
                    clear = plain_wall(e, q, new, lo, hi)
                    if best is None or clear > best[0]:
                        best = (clear, dist, k, s, i, e, old, new)
    clear, dist, k, s, i, e, old, new = best
    print(f"web house {k} (seed {100_064 + k}): spot {s} seen from attacker start {i} at {dist:.1f} m, through a window;"
          f" the new spot hidden by a wall {clear:.1f} m from any door or window;"
          f" moved {np.linalg.norm(old['spawn'][s] - new['spawn'][s]):.1f} m")
    L = lambda x: np.round(np.asarray(x, float), 3).tolist()
    out = {"house": k, "seed": 100_064 + k, "lo": L(new["lo"]), "hi": L(new["hi"]),
           "walls": [[L(l), L(h), bool(soft)] for l, h, soft in zip(new["wall_lo"], new["wall_hi"], new["wall_soft"])],
           "doors": [[L(d["lo"]), L(d["hi"])] for d in new["doors"]],
           "windows": [[L(l), L(h)] for l, h in zip(new["win_lo"], new["win_hi"])],
           "tall": [[L(l), L(h)] for l, h in zip(new["cover_lo"], new["cover_hi"])],
           "low": [[L(l), L(h)] for l, h in zip(new["low_lo"], new["low_hi"])],
           "entries": [[L(p), float(f)] for p, f, _ in new["entries"]],
           "eye": L(e), "old": L(old["spawn"][s]), "new": L(new["spawn"][s])}
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
