"""Why do they shoot at walls? (owner, after ep11 10zm.) Every shot each side fires, sorted by what the shooter knew:
an enemy in sight (an aimed shot), none in sight but one seen in the last 2 s (at where he was: suppressing / "pre-
firing"), or nobody seen lately (blind). In this game a shot can only hit an enemy the shooter sees (walls stop
every bullet), so the last two kinds can't hit; they cost ammo and recoil, flash, and the whole building hears them.
Each Assault brain under its own run's rules against a Defense, report houses (test 64-255), seed 5000, best guess.
No drawing code: runs on Modal.

    uv run --group jax --group modal --env-file .env ailearns cloud script \
        "scripts/cqb_blind_fire.py runs/cqb-r9-callout-critic:572 runs/cqb-v3-evolve-aim-more:200" --on modal
"""

import argparse
from pathlib import Path

import numpy as np

from ailearns.agents import cqb_eval as E
from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S

REPORT = list(cqb.TEST_SEEDS)[64:256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brains", nargs="+", help="Assault: runs/x:update or a script name (with --rules)")
    ap.add_argument("--against", default="runs/cqb-d5-guns:900")
    ap.add_argument("--rules", default="", help="runs/x:update whose rules everyone plays (default: each its own)")
    a = ap.parse_args()
    print("| Assault | vs | side | shots a round | aimed (enemy in sight) | at a spot seen < 2 s ago | blind | "
          "hits: from aimed / from not-in-sight shots |\n|---|---|---|---|---|---|---|---|")
    for spec in a.brains:
        if spec in S.SCRIPTS:
            player, ck = spec, A.load(Path(a.rules.partition(":")[0]) / "checkpoints" /
                                     f"u{int(a.rules.partition(':')[2]):05d}.pt")[1]
        else:
            run, _, u = spec.partition(":")
            params, ck = A.load(Path(run) / "checkpoints" / f"u{int(u):05d}.pt")
            player = A.brain(params, cqb.ASSAULT)
        if a.rules and spec not in S.SCRIPTS:
            ck = A.load(Path(a.rules.partition(":")[0]) / "checkpoints" / f"u{int(a.rules.partition(':')[2]):05d}.pt")[1]
        cfg = E._game_cfg(ck)
        g = cqb.CQBGame(cqb.make_buildings(REPORT, cfg), cfg)
        r = S.play_rounds(g, player, A.opponent_player(a.against, g.nA), np.arange(len(REPORT)), seed=5000,
                          record=True)
        h = r["history"]
        live = ~np.asarray(h.done)[:, :-1]                                       # (rounds, t) before the round ends
        fired = np.asarray(h.fired)[:, 1:] & live[..., None]                      # (rounds, t, N): fired this decision
        seen = np.asarray(h.seen)[:, :-1]                                         # what each saw when it decided
        since = np.asarray(h.since)[:, :-1]
        hp = np.asarray(h.hp) if cfg.health else np.asarray(h.alive).astype(float)
        hurt = (hp[:, 1:] < hp[:, :-1]) & live[..., None]                         # (rounds, t, N) hit this decision
        by = np.asarray(h.hit_by)[:, 1:]                                          # who last hit each one
        for side in (cqb.ASSAULT, cqb.DEFENSE):
            ops, en = g.side_ops(side), g.side_ops(1 - side)
            f = fired[:, :, ops]
            sees = seen[:, :, ops][..., en].any(-1)
            recent = (since[:, :, ops][..., en] < 20).any(-1) & ~sees
            n = f.sum()
            share = lambda m: f"{(f & m).sum() / max(n, 1):.0%}"                 # noqa: E731
            hit = np.zeros_like(f)                                                # did this shot hit an enemy
            for k, op in enumerate(ops):
                hit[:, :, k] = f[:, :, k] & (hurt[:, :, en] & (by[:, :, en] == op)).any(-1)
            hits = f"{(hit & sees).sum()} / {(hit & ~sees).sum()}"
            print(f"| {spec.split('/')[-1]} | {a.against.split('/')[-1]} | {'Assault' if side == cqb.ASSAULT else 'Defense'} | "
                  f"{n / len(REPORT):.1f} | {share(sees)} | {share(recent)} | {share(~sees & ~recent)} | {hits} |", flush=True)


if __name__ == "__main__":
    main()
