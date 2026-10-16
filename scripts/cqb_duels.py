"""Why do attackers lose? (owner, ep12) Every kill in a set of recorded rounds, from both sides: who fired it, how fast
the shooter and the victim were moving and turning when it happened, the distance, and whether the victim had seen
the shooter at all; plus how the rounds end. Small houses, 60 rounds, best-guess play, test houses 64+.

    uv run --group jax python scripts/cqb_duels.py --assault runs/cqb-e3b-2v2-split:200 \\
        --defense runs/cqb-d12-2v2-split-evolve:200 --teams 2,2 --split [--rounds 128]
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ailearns.agents import cqb_eval as E
from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assault", required=True)
    ap.add_argument("--defense", required=True)
    ap.add_argument("--rules", default="runs/cqb-p4-stun:650")
    ap.add_argument("--teams", default="2,2")
    ap.add_argument("--hostage", action="store_true", help="a hostage: securing it wins for Assault (ep12)")
    ap.add_argument("--blind-away", type=float, default=1.0, help="flashbang blindness facing away (ep12: 0.25)")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--rounds", type=int, default=128)
    ap.add_argument("--seed", type=int, default=5000)
    a = ap.parse_args()
    run, _, u = a.rules.partition(":")
    cfg = replace(E._game_cfg(A.load(Path(run) / "checkpoints" / f"u{int(u):05d}.pt")[1]),
                  team_sizes=tuple(int(x) for x in a.teams.split(",")), spare_mags=1, split_entry=a.split, blind_away=a.blind_away, hostage=a.hostage)

    def brain(spec, side):
        r, _, c = spec.partition(":")
        p, ck = A.load(Path(r) / "checkpoints" / f"u{int(c):05d}.pt")
        return A.brain(A.fit(p, ck, cfg, side), side)

    houses = list(cqb.TEST_SEEDS)[64:64 + a.rounds]
    g = cqb.CQBGame(cqb.make_buildings(houses, cfg), cfg)
    r = S.play_rounds(g, brain(a.assault, cqb.ASSAULT), brain(a.defense, cqb.DEFENSE), np.arange(len(houses)),
                      seed=a.seed, record=True)
    h = r["history"]
    alive, vel, turn = np.asarray(h.alive), np.asarray(h.vel), np.asarray(h.turn)
    pos, seen, hit_by = np.asarray(h.pos), np.asarray(h.seen), np.asarray(h.hit_by)
    fired = np.asarray(h.fired)
    A_ops, D_ops = g.side_ops(cqb.ASSAULT), g.side_ops(cqb.DEFENSE)
    side = {o: "A" for o in A_ops} | {o: "D" for o in D_ops}
    kills = []
    R, T, N = alive.shape
    for i in range(R):
        for t in range(1, T):
            for v in range(N):
                if alive[i, t - 1, v] and not alive[i, t, v]:
                    k = int(hit_by[i, t, v])
                    if k < 0:
                        continue
                    ever_saw = seen[i, :t, v, k].any()                              # victim ever saw its killer
                    kills.append(dict(killer=side[k], k_speed=np.linalg.norm(vel[i, t - 1, k]),
                                      v_speed=np.linalg.norm(vel[i, t - 1, v]), k_turn=abs(turn[i, t - 1, k]),
                                      dist=np.linalg.norm(pos[i, t, k] - pos[i, t, v]), saw=ever_saw,
                                      k_shots=fired[i, max(0, t - 30):t + 1, k].sum()))
    out = np.asarray(r["alive"])
    a_up, d_up = out[:, A_ops].any(1), out[:, D_ops].any(1)
    print(f"{Path(a.assault.split(':')[0]).name} vs {Path(a.defense.split(':')[0]).name}, {a.teams}, "
          f"split {a.split}, {R} rounds: Assault wins {np.mean(a_up & ~d_up):.0%}, Defense by kills "
          f"{np.mean(d_up & ~a_up):.0%}, time-outs {np.mean(a_up & d_up):.0%}\n")
    print("| kills by | how many | shooter moving (m/s, median) | shooter still (<0.3 m/s) | victim moving | "
          "victim still | victim never saw the shooter | distance (median) |\n|---|---|---|---|---|---|---|---|")
    for s in ("A", "D"):
        ks = [x for x in kills if x["killer"] == s]
        if not ks:
            continue
        f = lambda key: np.array([x[key] for x in ks])   # noqa: E731
        print(f"| {'Assault' if s == 'A' else 'Defense'} | {len(ks)} | {np.median(f('k_speed')):.2f} | "
              f"{np.mean(f('k_speed') < 0.3):.0%} | {np.median(f('v_speed')):.2f} | {np.mean(f('v_speed') < 0.3):.0%} | "
              f"{np.mean(~f('saw')):.0%} | {np.median(f('dist')):.1f} m |")


if __name__ == "__main__":
    main()
