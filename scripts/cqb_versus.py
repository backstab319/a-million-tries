"""Like-for-like CQB matchups (ep12): every Assault player against every Defense player, under one set of rules, on
the same test houses and seeds, so the numbers compare. Players are scripts (`camper`, `guard`, ...) or saved brains
(`runs/<name>:<update>`, fitted to the rules: `ppo_cqb.fit`). Rounds count as Assault wins (and how many with the
hostage secured or out while a defender still stood), Defense wins by kills, and time-outs (both sides standing).

    uv run --group jax python scripts/cqb_versus.py --teams 2,2 --split --hostage --hostage-deep 6 --extract \\
        --assault runs/cqb-e14-2v2-extract:300 --defense runs/cqb-d20-2v2-hostage-coached:200,camper,guard

Heavy on a laptop's CPU: run it on a Kaggle CPU with `ailearns cloud script "<this command>" --on kaggle`.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ailearns.agents import cqb_eval as E
from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S

ROOT = Path(__file__).resolve().parents[1]


def player(spec: str, side: int, cfg):
    if spec in S.SCRIPTS:
        return spec
    run, _, u = spec.partition(":")
    params, ck = A.load(ROOT / run / "checkpoints" / f"u{int(u):05d}.pt")
    return A.brain(A.fit(params, ck, cfg, side), side)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assault", required=True, help="comma-separated players")
    ap.add_argument("--defense", required=True, help="comma-separated players")
    ap.add_argument("--rules", default="runs/cqb-p4-stun:650", help="whose rules (then changed by the flags below)")
    ap.add_argument("--teams", default="2,2")
    ap.add_argument("--spare", type=int, default=1)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--blind-away", type=float, default=0.25)
    ap.add_argument("--hostage", action="store_true")
    ap.add_argument("--hostage-deep", type=float, default=0.0)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--hostage-noise", type=float, default=0.0, help="Defense hears the hostage being moved this far (m)")
    ap.add_argument("--flash-defense", type=int, default=0, help="flashbangs for each defender (0: the rules' own)")
    ap.add_argument("--houses", default="64-191", help="test house indices a-b")
    ap.add_argument("--seeds", default="5000,5001")
    a = ap.parse_args()

    run, _, u = a.rules.partition(":")
    cfg = replace(E._game_cfg(A.load(ROOT / run / "checkpoints" / f"u{int(u):05d}.pt")[1]),
                  team_sizes=tuple(int(x) for x in a.teams.split(",")), spare_mags=a.spare, split_entry=a.split,
                  blind_away=a.blind_away, hostage=a.hostage, hostage_deep=a.hostage_deep, extract=a.extract, hostage_noise=a.hostage_noise)
    cfg = replace(cfg, flashbangs=(cfg.flashbangs[0], a.flash_defense or cfg.flashbangs[1]))
    lo, _, hi = a.houses.partition("-")
    houses = list(cqb.TEST_SEEDS)[int(lo):int(hi or lo) + 1]
    g = cqb.CQBGame(cqb.make_buildings(houses, cfg), cfg)
    seeds = [int(s) for s in a.seeds.split(",")]
    goal = "out" if a.extract else "secured"
    print(f"{a.teams.replace(',', 'v')}, split {a.split}, hostage {a.hostage} (deep {a.hostage_deep}, extract "
          f"{a.extract}), {len(houses)} houses x {len(seeds)} seeds\n")
    print(f"| Assault | Defense | Assault wins | hostage {goal} | Defense by kills | time-outs | mean round |\n"
          "|---|---|---|---|---|---|---|", flush=True)
    for att in a.assault.split(","):
        for dfn in a.defense.split(","):
            res = []
            for seed in seeds:
                r = S.play_rounds(g, player(att, cqb.ASSAULT, cfg), player(dfn, cqb.DEFENSE, cfg),
                                  np.arange(len(houses)), seed=seed)
                al, out = np.asarray(r["alive"]), np.asarray(r["outcome"])
                a_up = al[:, g.side_ops(cqb.ASSAULT)].any(1)
                d_up = al[:, g.side_ops(cqb.DEFENSE)].any(1)
                res.append((np.mean(out == 1), np.mean((out == 1) & d_up), np.mean(d_up & ~a_up),
                            np.mean(a_up & d_up & (out != 1)), np.mean(r["t"]) * cfg.dt))
            o = np.mean(res, 0)
            name = lambda s: s.split("/")[-1].replace("cqb-", "")   # noqa: E731
            print(f"| {name(att)} | {name(dfn)} | {o[0]:.0%} | {o[1]:.0%} | {o[2]:.0%} | {o[3]:.0%} | {o[4]:.1f} s |",
                  flush=True)


if __name__ == "__main__":
    main()
