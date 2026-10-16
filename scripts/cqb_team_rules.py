"""Episode 12 (teams, less ammo): how a team game plays before anyone trains for it. Every operator of a side plays its
side's brain (one brain, as in training); the same houses under several ammo loads, so the only thing that changes is
how many rounds each operator carries. Per load: who wins, how often time runs out, how long rounds take, shots per
operator per round (and how many were aimed: an enemy in sight), and how many operators run dry before the round ends.

    uv run --group jax python scripts/cqb_team_rules.py --teams 2,2 --house big --spare 3,1,0 [--rounds 64]
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ailearns.agents import cqb_eval as E
from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S


def player(spec: str, side: int, cfg=None):
    """A script, or a saved brain fitted to the game's rules (ep12: a 2v2 brain in a 3v2 game, `A.fit`)."""
    if spec in S.SCRIPTS:
        return spec
    run, _, u = spec.partition(":")
    if cfg is not None:
        params, ck = A.load(Path(run) / "checkpoints" / f"u{int(u):05d}.pt")
        return A.brain(A.fit(params, ck, cfg, side), side)
    return A.brain(A.load(Path(run) / "checkpoints" / f"u{int(u):05d}.pt")[0], side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assault", default="runs/cqb-p4-stun:650", help="runs/x:update or a script")
    ap.add_argument("--defense", default="runs/cqb-d8-vs-pair:350", help="runs/x:update or a script")
    ap.add_argument("--rules", default="runs/cqb-p4-stun:650", help="whose rules everyone plays (then changed below)")
    ap.add_argument("--teams", default="2,2", help="operators per side, Assault,Defense")
    ap.add_argument("--house", default="big", help="a cqb.HOUSES preset")
    ap.add_argument("--mag", type=int, default=0, help="rounds per magazine (0: the rules' own)")
    ap.add_argument("--spare", default="3,1,0", help="spare magazines to compare, comma-separated")
    ap.add_argument("--rounds", type=int, default=64, help="report houses (test 64+)")
    ap.add_argument("--time", type=float, default=0, help="seconds a round lasts (0: the rules' own)")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--hostage", action="store_true", help="a hostage: securing it wins for Assault (ep12)")
    ap.add_argument("--hostage-deep", type=float, default=0.0, help="the hostage at least this far (m) from every entry point")
    ap.add_argument("--extract", action="store_true", help="secured, the hostage must be walked out to win (ep12)")
    ap.add_argument("--blind-away", type=float, default=1.0, help="flashbang blindness facing away (ep12: 0.25)")
    ap.add_argument("--split", action="store_true", help="Assault starts at different doors (ep11 split entry)")
    ap.add_argument("--flash-defense", type=int, default=0, help="flashbangs for each defender (0: none, as before)")
    a = ap.parse_args()

    run, _, u = a.rules.partition(":")
    base = E._game_cfg(A.load(Path(run) / "checkpoints" / f"u{int(u):05d}.pt")[1])
    teams = tuple(int(x) for x in a.teams.split(","))
    houses = list(cqb.TEST_SEEDS)[64:64 + a.rounds]
    print(f"{a.assault} (Assault) vs {a.defense} (Defense), {teams[0]}v{teams[1]}, house '{a.house}', "
          f"{len(houses)} houses, seed {a.seed}\n")
    print("| rounds carried | Assault wins | Defense wins (Assault all down) | time-outs (both still standing) | mean round | shots / operator (Assault, Defense) "
          "| aimed | ran dry (Assault, Defense) |\n|---|---|---|---|---|---|---|---|")
    for spare in (int(x) for x in a.spare.split(",")):
        cfg = replace(base, team_sizes=teams, spare_mags=spare, split_entry=a.split, blind_away=a.blind_away, hostage=a.hostage, hostage_deep=a.hostage_deep, extract=a.extract,
                      flashbangs=(base.flashbangs[0], a.flash_defense or base.flashbangs[1]), **({"magazine": a.mag} if a.mag else {}),
                      **({"time_limit": a.time} if a.time else {}),
                      **cqb.HOUSES[a.house])
        g = cqb.CQBGame(cqb.make_buildings(houses, cfg), cfg)
        r = S.play_rounds(g, player(a.assault, cqb.ASSAULT, cfg), player(a.defense, cqb.DEFENSE, cfg), np.arange(len(houses)),
                          seed=a.seed, record=True)
        h, alive = r["history"], np.asarray(r["alive"])
        # who's left: the game scores a time-out as Defense holding (outcome -1), so tell them apart by who's standing
        a_up, d_up = alive[:, g.side_ops(cqb.ASSAULT)].any(1), alive[:, g.side_ops(cqb.DEFENSE)].any(1)
        won, held, timeout = a_up & ~d_up, d_up & ~a_up, a_up & d_up
        live = ~np.asarray(h.done)[:, :-1]
        fired = np.asarray(h.fired)[:, 1:] & live[..., None]                   # (rounds, t, N)
        seen = np.asarray(h.seen)[:, :-1]
        dry = ((np.asarray(h.mag) == 0) & (np.asarray(h.reserve) == 0)).any(1)  # (rounds, N) emptied everything
        cells = []
        for side in (cqb.ASSAULT, cqb.DEFENSE):
            ops, en = g.side_ops(side), g.side_ops(1 - side)
            f = fired[:, :, ops]
            cells.append((f.sum((0, 1)).mean() / len(houses), (f & seen[:, :, ops][..., en].any(-1)).sum(), f.sum(),
                          dry[:, ops].mean()))
        aimed = (cells[0][1] + cells[1][1]) / max(cells[0][2] + cells[1][2], 1)
        carried = cfg.magazine * (1 + spare)
        print(f"| {carried} ({1 + spare} x {cfg.magazine}) | {won.mean():.0%} | {held.mean():.0%} | "
              f"{timeout.mean():.0%} | {np.asarray(r['t']).mean() * cfg.dt:.1f} s | "
              f"{cells[0][0]:.0f}, {cells[1][0]:.0f} | {aimed:.0%} | {cells[0][3]:.0%}, {cells[1][3]:.0%} |", flush=True)


if __name__ == "__main__":
    main()
