"""The facing trace for ep13's "spinning" beat: which way an e15 attacker and a d24 defender face over the same 20 s
of one round, with their swings (cqb_turning.py's definition: a turn one way until the facing comes back 30 deg).

Same rounds as cqb_turning.py (report houses, seed 5000). The attacker is not the worst one: of the attackers up for
the whole window, the one whose back-and-forths a minute is closest to the e15 average.

    uv run --group jax python scripts/cqb_facing_trace.py runs/cqb-e15-2v2-home:200 runs/cqb-d24-2v2-home:300 \\
        --out episodes/13-the-house/manim/facing.json
"""

import argparse
import json

import numpy as np

from ailearns import record_cqb as R
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S
from cqb_turning import HOLD


def swing_spans(face: np.ndarray) -> list[tuple[int, int, float]]:
    """cqb_turning.swings for one operator who is up throughout, as (start, end, degrees) steps."""
    out, f = [], np.unwrap(face)
    start = ext = 0
    for t in range(1, len(f)):
        d = np.sign(f[ext] - f[start]) or np.sign(f[t] - f[start])
        if (f[t] - f[ext]) * d >= 0:
            ext = t
        elif abs(f[t] - f[ext]) > HOLD:
            out.append((start, ext, float(np.degrees(abs(f[ext] - f[start])))))
            start = ext
            ext = t
    if ext != start:
        out.append((start, ext, float(np.degrees(abs(f[ext] - f[start])))))
    return out


def back_forths(sw: list[tuple[int, int, float]]) -> int:
    return sum(a[2] >= 60 and b[2] >= 60 for a, b in zip(sw, sw[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("assault")
    ap.add_argument("defense")
    ap.add_argument("--rules", default="runs/cqb-d24-2v2-home:300")
    ap.add_argument("--rounds", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = R.game_config(a.rules)
    seeds = list(cqb.TEST_SEEDS)[64:64 + a.rounds]
    g = cqb.CQBGame(cqb.make_buildings(seeds, cfg), cfg)
    r = S.play_rounds(g, R.player(a.assault, 0), R.player(a.defense, g.nA), np.arange(len(seeds)), seed=5000, record=True)
    h = r["history"]
    face = np.asarray(h.face)[:, :-1]
    alive = np.asarray(h.alive)[:, :-1] & ~np.asarray(h.done)[:, :-1, None]
    n = int(round(a.seconds / cfg.dt))

    # every (round, attacker) up for the first n decisions, with a defender up for the same n: back-and-forths a minute
    cands = []
    for i in range(len(seeds)):
        defs = [k for k in range(g.nA, g.N) if alive[i, :n, k].all()]
        for k in range(g.nA):
            if alive[i, :n, k].all() and defs:
                cands.append((back_forths(swing_spans(face[i, :n, k])) / (a.seconds / 60), i, k, defs[0]))
    mean = np.mean([c[0] for c in cands])
    rate, i, k, d = min(cands, key=lambda c: abs(c[0] - mean))
    print(f"{len(cands)} attackers up for {a.seconds:.0f} s; mean {mean:.1f} back-and-forths a minute; "
          f"picked house {seeds[i]} attacker {k} ({rate:.1f}/min) vs defender {d}")

    def op(j):
        f = face[i, :n, j]
        sw = swing_spans(f)
        print(f"  op {j}: swings {[round(s[2]) for s in sw]}, back-and-forths {back_forths(sw)}")
        return {"deg": np.round(np.degrees(np.unwrap(f) - np.unwrap(f)[0]), 1).tolist(),
                "swings": [{"start": s, "end": e, "deg": round(dg)} for s, e, dg in sw if dg >= 60]}

    out = {"dt": cfg.dt, "seconds": a.seconds, "house": int(seeds[i]), "assault": a.assault, "defense": a.defense,
           "attacker": op(k), "defender": op(d), "mean_back_forth": round(float(mean), 1)}
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
