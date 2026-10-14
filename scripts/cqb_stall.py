"""Why does a free callout cost wins (ep11 4b/5)? The same pair on the same report houses with shared sightings on and
off: in rounds that time out, where are they and what are they doing, and how often does an operator "know" the
defender (the team saw him) without seeing him itself. `uv run --group jax python scripts/cqb_stall.py
runs/cqb-p4-stun:650 [--split] [--rounds 64]`"""

import argparse
from dataclasses import replace

import numpy as np

from ailearns import record_cqb as R
from ailearns.games import cqb, cqb_scripts as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair")
    ap.add_argument("--against", default="runs/cqb-d5-guns:900")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--rounds", type=int, default=64)
    a = ap.parse_args()
    base = replace(R.game_config(a.pair), split_entry=a.split)
    seeds = list(cqb.TEST_SEEDS)[64:64 + a.rounds]
    print(f"{a.pair} vs {a.against}, {len(seeds)} report houses{', split entry' if a.split else ''}")
    print("| shared sightings | won | time-outs | in time-outs: inside | speed (m/s) | still (<0.3 m/s) | "
          "knows him, can't see him (all rounds, alive) | ... and then standing still |")
    print("|---|---|---|---|---|---|---|---|")
    for share in (True, False):
        cfg = replace(base, share_sight=share)
        g = cqb.CQBGame(cqb.make_buildings(seeds, cfg), cfg)
        r = S.play_rounds(g, R.player(a.pair, 0), R.player(a.against, g.nA), np.arange(len(seeds)), seed=5000,
                          record=True)
        h = r["history"]
        out, alive_end = np.asarray(r["outcome"]), np.asarray(r["alive"])
        tout = alive_end[:, :2].any(1) & alive_end[:, 2]
        live = ~np.asarray(h.done)[:, :-1]
        alive = np.asarray(h.alive)[:, :-1, :2] & live[..., None]
        pos = np.asarray(h.pos)[:, :, :2]
        speed = np.linalg.norm(pos[:, 1:] - pos[:, :-1], axis=-1) / cfg.dt          # (rounds, t, 2)
        lo, hi = np.asarray(g.b.foot_lo)[:, None, None], np.asarray(g.b.foot_hi)[:, None, None]
        inside = ((pos[:, :-1] > lo) & (pos[:, :-1] < hi)).all(-1)
        seen = np.asarray(h.seen)[:, :-1, :2, 2]
        since = np.asarray(h.since)[:, :-1, :2, 2]
        knows = (since < g.T) & ~seen & alive                                          # remembered, not seen now
        m = alive & tout[:, None, None]
        print(f"| {share} | {np.mean(out == 1):.0%} | {tout.mean():.0%} | {inside[m].mean():.0%} | "
              f"{speed[m].mean():.2f} | {np.mean(speed[m] < 0.3):.0%} | {knows[alive].mean():.0%} | "
              f"{np.mean(speed[knows] < 0.3):.0%} |", flush=True)


if __name__ == "__main__":
    main()
