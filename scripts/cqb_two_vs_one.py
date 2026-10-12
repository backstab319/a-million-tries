"""Episode 10's first question: does a second operator help, before anyone trains for it? (ep10 step 4)

Each Assault player plays 1v1 and then as a pair (the same player for both operators: scripts, or a 1v1 brain that has
never seen a teammate), against each Defense player, under episode 9's full rules (flashbangs 2,0, 30-round magazines,
health 100, reaction time 0.3 s). Report buildings (test 64-255, seed 5000, best guess), as cqb_eval.

    uv run --group jax --group modal --env-file .env ailearns cloud script "scripts/cqb_two_vs_one.py" --on modal
    ... "scripts/cqb_two_vs_one.py --assault runs/cqb-p1-pair:500,runs/cqb-a12-react:300" --on modal
-> a markdown table, and the numbers as one JSON line. A brain trained in a team reads more numbers than 1v1 has: it
only plays 2v1 ("-" in the 1v1 column).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S

FULL = {"flashbangs": (2, 0), "magazine": 30, "health": 100.0, "react": 0.3}
ASSAULT = ["walker", "walker_flash", "runs/cqb-a12-react:300"]
DEFENSE = ["camper", "runs/cqb-d5-guns:900"]
REPORT = list(cqb.TEST_SEEDS)[64:256]
SEED = 5000


def _player(spec: str, who: int):
    if spec in S.SCRIPTS:
        return S.SCRIPTS[spec]
    run, _, u = spec.partition(":")
    return A.brain(A.load(Path(run) / "checkpoints" / f"u{int(u):05d}.pt")[0], who)


def _short(spec: str) -> str:
    return spec.split("/")[-1].replace("cqb-", "")


def _team_brain(spec: str) -> bool:
    if spec in S.SCRIPTS:
        return False
    run, _, u = spec.partition(":")
    return tuple(A.load(Path(run) / "checkpoints" / f"u{int(u):05d}.pt")[1]["cqb"].get("team_sizes", (1, 1))) != (1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assault", default=",".join(ASSAULT))
    ap.add_argument("--defense", default=",".join(DEFENSE))
    ap.add_argument("--rules", default="", help="runs/x:u: play under that brain's rules instead of episode 9's full rules")
    ap.add_argument("--only-2v1", action="store_true")
    args = ap.parse_args()
    rules = FULL
    if args.rules:
        run, _, u = args.rules.partition(":")
        saved = A.load(Path(run) / "checkpoints" / f"u{int(u):05d}.pt")[1]["cqb"]
        rules = {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k != "team_sizes"}
    assault, defense = args.assault.split(","), args.defense.split(",")
    out = {}
    for teams in ((2, 1),) if args.only_2v1 else ((1, 1), (2, 1)):
        cfg = replace(cqb.CQBConfig(), team_sizes=teams, **rules)
        game = cqb.CQBGame(cqb.make_buildings(REPORT, cfg), cfg)
        nA = game.nA
        for a in assault:
            for d in defense:
                if teams == (1, 1) and (_team_brain(a) or _team_brain(d)):
                    continue
                pa, pd = _player(a, cqb.ASSAULT), _player(d, cqb.DEFENSE)
                last, _, _ = jax.jit(lambda b, pa=pa, pd=pd, game=game: S.rounds(game, pa, pd, b, SEED))(
                    jnp.arange(len(REPORT)))
                last = jax.device_get(last)
                o, alive, hit_by = np.asarray(last.outcome), np.asarray(last.alive), np.asarray(last.hit_by)
                a_out, d_out = ~alive[:, :nA].any(1), ~alive[:, nA:].any(1)
                friendly = ((hit_by[:, :nA] >= 0) & (hit_by[:, :nA] < nA)).sum(1)   # Assault downed by Assault
                out[f"{_short(a)} vs {_short(d)} {teams[0]}v{teams[1]}"] = {
                    "assault_won": float((o == 1).mean()), "trades": float((a_out & d_out).mean()),
                    "timeouts": float((~a_out & ~d_out).mean()),
                    "assault_all_alive_when_won": float(alive[o == 1, :nA].all(1).mean()) if (o == 1).any() else None,
                    "friendly_fire_downs_per_round": float(friendly.mean()),
                    "flashbangs_per_round": float((2 * nA - np.asarray(last.bangs)[:, :nA].sum(1)).mean()),
                    "seconds": float(np.asarray(last.t).mean() * cfg.dt)}
    print("| Assault vs Defense | 1v1 Assault won | 2v1 Assault won | 2v1 trades | 2v1 time-outs | 2v1 won with both "
          "standing | friendly-fire downs / round (2v1) |")
    print("|---|---|---|---|---|---|---|")
    for a in assault:
        for d in defense:
            k = f"{_short(a)} vs {_short(d)}"
            one, two = out.get(k + " 1v1"), out[k + " 2v1"]
            both = "-" if two["assault_all_alive_when_won"] is None else f"{two['assault_all_alive_when_won']:.0%}"
            solo = "-" if one is None else f"{one['assault_won']:.0%}"
            print(f"| {k} | {solo} | {two['assault_won']:.0%} | {two['trades']:.0%} | "
                  f"{two['timeouts']:.0%} | {both} | {two['friendly_fire_downs_per_round']:.2f} |")
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
