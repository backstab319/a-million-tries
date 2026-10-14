"""ep11 10p: who sees whom first? For each Assault player (scripts or brains) in 1v1 under one run's rules, against one
Defense: how often the attacker sees the defender first (or both at once), how far its aim is off him at the first
sighting either way, and the win rate in each case. Report houses (test 64-255), seed 5000, best guess. No drawing
code, so it runs on Modal:

    uv run --group jax --group modal --env-file .env ailearns cloud script \
        "scripts/cqb_first_sight.py walker walker_feed runs/cqb-s1-solo-big-feed:200 --rules runs/cqb-s1-solo-big-feed:200" \
        --on modal
"""

import argparse
from pathlib import Path

import numpy as np

from ailearns.agents import cqb_eval as E
from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S

REPORT = list(cqb.TEST_SEEDS)[64:256]


def _ckpt(spec: str) -> Path:
    run, _, u = spec.partition(":")
    return Path(run) / "checkpoints" / f"u{int(u):05d}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("players", nargs="+", help="Assault: script names or runs/x:update")
    ap.add_argument("--rules", required=True, help="runs/x:update whose rules everyone plays")
    ap.add_argument("--against", default="runs/cqb-d5-guns:900")
    a = ap.parse_args()
    cfg = E._game_cfg(A.load(_ckpt(a.rules))[1])
    game = cqb.CQBGame(cqb.make_buildings(REPORT, cfg), cfg)
    print(f"rules of {a.rules} (rooms {cfg.rooms}, feed {cfg.feed}, route {cfg.route}) vs {a.against}")
    print("| Assault | won | sees him first | he sees first | both at once | never met | aim off him at first sight "
          "(median) | won when it saw first | won when he did |\n|---|---|---|---|---|---|---|---|---|")
    for p in a.players:
        player = p if p in S.SCRIPTS else A.brain(A.load(_ckpt(p))[0], cqb.ASSAULT)
        r = S.play_rounds(game, player, A.opponent_player(a.against, 1), np.arange(len(REPORT)), seed=5000,
                          record=True)
        h = r["history"]
        seen = np.asarray(h.seen)                                                    # (rounds, t, 2, 2)
        live = ~np.asarray(h.done)
        T = seen.shape[1]
        first = lambda m: np.where((m & live).any(1), np.argmax(m & live, axis=1), T)   # noqa: E731
        ta, td = first(seen[:, :, 0, 1]), first(seen[:, :, 1, 0])
        met = np.minimum(ta, td) < T
        a_first, d_first, both = met & (ta < td), met & (td < ta), met & (ta == td)
        t0 = np.minimum(np.minimum(ta, td), T - 1)
        pos, face = np.asarray(h.pos), np.asarray(h.face)
        k = np.arange(len(t0))
        d = pos[k, t0, 1] - pos[k, t0, 0]
        off = np.degrees(np.abs((np.arctan2(d[:, 1], d[:, 0]) - face[k, t0, 0] + np.pi) % (2 * np.pi) - np.pi))
        won = np.asarray(r["outcome"]) == 1
        rate = lambda x: f"{x.mean():.0%}" if x.size else "-"                        # noqa: E731
        name = p.split("/")[-1]
        print(f"| {name} | {won.mean():.0%} | {a_first.mean():.0%} | {d_first.mean():.0%} | {both.mean():.0%} | "
              f"{(~met).mean():.0%} | {np.median(off[met]):.0f} deg | {rate(won[a_first])} | {rate(won[d_first])} |",
              flush=True)


if __name__ == "__main__":
    main()
