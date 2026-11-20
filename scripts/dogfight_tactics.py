"""How does a dogfight brain fly? Measured over many fights, not eyeballed from a few replays (ep21 step 3f).

    uv run python scripts/dogfight_tactics.py runs/dogfight-self2/checkpoints/u00500.pkl \
        --vs runs/dogfight-chase5/checkpoints/u00700.pkl --fights 512 --seed 4242

For each jet (0 = the first brain, 1 = --vs or the script), over all flying time:
  energy height   h + v²/2g (m): the "bank account" pilots fight with; and the 0-minus-1 gap
  corner share    time within 330-440 kt, where the F-16 turns fastest
  hard g share    time above 7 g (turning hard bleeds energy)
  vertical share  time with the nose more than 30° above / below the horizon
  zooms           climbs of 500+ m with the nose 30°+ up, trading speed for height
  overshoots      the jet behind (within 1 km, in the other's rear half) ends up in front of it
And at each shoot-down: the killer's energy minus the victim's, and the range.
"""

import argparse
import json

import numpy as np

from dogfight_replay import run  # same folder

G = 9.80665


def rear_half(pos, nose, i):
    """True where jet i is behind jet j (in j's rear half)."""
    j = 1 - i
    d = pos[..., i, :] - pos[..., j, :]
    return (d * nose[..., j, :]).sum(-1) < 0


def tactics(game, ends, frames, dt):
    alive = frames["alive"]                     # (F, T)
    pos, nose, v, g = frames["pos"], frames["nose"], frames["speed"], frames["load"]   # (F,T,2,3), (F,T,2)
    e = pos[..., 2] + v ** 2 / (2 * G)
    rng = frames["range"]
    w = alive[..., None].astype(float)
    mean = lambda x: (x * w).sum((0, 1)) / w.sum((0, 1))  # noqa: E731
    kt = v * 1.944
    pitch = np.degrees(np.arcsin(np.clip(nose[..., 2], -1, 1)))
    out = {"energy_m": mean(e), "corner_share": mean((kt > 330) & (kt < 440)), "hard_g_share": mean(g > 7.0),
           "nose_up_share": mean(pitch > 30), "nose_down_share": mean(pitch < -30), "speed_ms": mean(v), "g": mean(g)}
    gap = ((e[..., 0] - e[..., 1]) * alive).sum() / alive.sum()
    # zooms: from a local start, height up 500 m within 15 s while nose >30° up at some point and speed falls
    zooms = np.zeros(2)
    over = np.zeros(2)
    F, T = alive.shape
    step = int(round(1.0 / dt))
    for f in range(F):
        n = int(alive[f].sum())
        for i in (0, 1):
            h, s, pt = pos[f, :n, i, 2], v[f, :n, i], pitch[f, :n, i]
            t = 0
            while t < n - step:
                end = min(n, t + 15 * step)
                k = t + int(np.argmax(h[t:end]))
                if h[k] - h[t] > 500 and pt[t:k + 1].max() > 30 and s[k] < s[t] - 30:
                    zooms[i] += 1
                    t = k + step
                else:
                    t += step
            behind = rear_half(pos[f, :n], nose[f, :n], i) & (rng[f, :n] < 1000)
            ahead = rear_half(pos[f, :n], nose[f, :n], 1 - i) & (rng[f, :n] < 1000)
            # behind at some step, the other's... in front of it within 3 s after
            idx = np.flatnonzero(behind[:-1] & ~behind[1:])
            over[i] += sum(ahead[k + 1:k + 1 + 3 * step].any() for k in idx)
    fights = F
    out["zooms_per_fight"] = zooms / fights
    out["overshoots_per_fight"] = over / fights
    # at each shoot-down
    kills = []
    for f in range(F):
        c = ends.cause[f]
        for i in (0, 1):
            if c[1 - i] == 1 and c[i] == 0:   # i shot the other down
                n = int(alive[f].sum()) - 1
                kills.append((i, e[f, n, i] - e[f, n, 1 - i], rng[f, n]))
    res = {k: np.round(np.asarray(val, float), 3).tolist() for k, val in out.items()}
    res["energy_gap_0_minus_1_m"] = round(float(gap), 1)
    for i in (0, 1):
        ks = [k for k in kills if k[0] == i]
        res[f"kills_by_{i}"] = len(ks)
        if ks:
            res[f"kills_by_{i}_energy_edge_m"] = round(float(np.median([k[1] for k in ks])), 0)
            res[f"kills_by_{i}_range_m"] = round(float(np.median([k[2] for k in ks])), 0)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ckpt")
    p.add_argument("--vs", default=None)
    p.add_argument("--fights", type=int, default=512)
    p.add_argument("--level", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=4242)
    a = p.parse_args()
    game, ends, frames = run(a.ckpt, a.fights, a.level, a.seed, sample=True, vs=a.vs)
    print(json.dumps({"ckpt": a.ckpt, "vs": a.vs or "script", **tactics(game, ends, frames, game.cfg.dt)}))


if __name__ == "__main__":
    main()
