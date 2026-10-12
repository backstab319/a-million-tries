"""How calm are they (ep10 step 9)? Per Assault player, 32 report rounds vs d5:900 under episode 9's full rules (as
a pair unless it says solo): won, full-speed turning, full spins, time touching a wall, and (second table) left/right
flicks, speed off walls, time stuck still against a wall. `uv run --group jax python scripts/cqb_habits.py [specs] [--rules runs/x:u]`"""

import sys
from dataclasses import replace
import numpy as np
from ailearns import record_cqb as R
from ailearns.games import cqb, cqb_scripts as S
full = replace(cqb.CQBConfig(), flashbangs=(2, 0), magazine=30, health=100.0, react=0.3)
seeds = list(cqb.TEST_SEEDS)[64:96]
D = "runs/cqb-d5-guns:900"
PLAYERS = [("walker (script) pair", "walker", (2, 1)), ("ep9 a12 solo", "runs/cqb-a12-react:300", (1, 1)),
           ("ep9 a12 as a pair", "runs/cqb-a12-react:300", (2, 1)), ("trained pair p1", "runs/cqb-p1-pair:750", (2, 1)),
           ("gallery flashbang 2-2", "runs/cqb-g2-gallery/cells/2-2:0", (2, 1))]
args = sys.argv[1:]
RULES = None                     # --rules runs/x:u: everyone plays under that brain's rules (a fair before / after)
if "--rules" in args:
    k = args.index("--rules")
    RULES, args = R.game_config(args[k + 1]), args[:k] + args[k + 2:]
if args:
    PLAYERS = [(x, x, (2, 1)) for x in args]
for name, spec, teams in PLAYERS:
    base = RULES or (full if spec in S.SCRIPTS else R.game_config(spec))
    cfg = replace(base, team_sizes=teams)
    g = cqb.CQBGame(cqb.make_buildings(seeds, cfg), cfg)
    r = S.play_rounds(g, R.player(spec, 0), R.player(D, g.nA), np.arange(len(seeds)), seed=5000, record=True)
    h = r["history"]
    ops = list(range(g.nA))
    turn = np.asarray(h.turn)[:, 1:, ops]                       # (rounds, t, ops) the turn chosen
    alive = np.asarray(h.alive)[:, :-1, ops] & ~np.asarray(h.done)[:, :-1, None]
    seeing = np.asarray(h.seen)[:, :-1][:, :, ops][:, :, :, g.nA:].any(-1)
    full_turn = (np.abs(turn) > 0.9) & alive
    # spins: runs of >= 15 decisions turning the same way at > 0.9 (15 x 24 deg = a full circle)
    spins = 0
    for i in range(turn.shape[0]):
        for k in range(len(ops)):
            run, sign = 0, 0
            for x, a in zip(turn[i, :, k], alive[i, :, k]):
                s_ = np.sign(x) if (abs(x) > 0.9 and a) else 0
                run = run + 1 if (s_ != 0 and s_ == sign) else (1 if s_ != 0 else 0)
                sign = s_
                if run == 15:
                    spins += 1
    # pressed on a wall: within 2 cm of radius from a wall or closed door box
    pos = np.asarray(h.pos)[:, :-1][:, :, ops]
    wall = 0
    for i, s in enumerate(seeds):
        lo, hi = g.b.wall_lo[i], g.b.wall_hi[i]
        lo, hi = np.asarray(lo), np.asarray(hi)
        p = pos[i][:, :, None, :]
        d = np.linalg.norm(p - np.clip(p, lo, hi), axis=-1).min(-1)
        wall += ((d < cfg.radius + 0.02) & alive[i]).sum()
    flip = (np.sign(turn[:, 1:]) * np.sign(turn[:, :-1]) < 0) & (np.abs(turn[:, 1:]) > 0.3) & \
        (np.abs(turn[:, :-1]) > 0.3) & alive[:, 1:]
    p_all = np.asarray(h.pos)[:, :, ops]
    speed = np.linalg.norm(p_all[:, 1:] - p_all[:, :-1], axis=-1) / cfg.dt
    touch = np.zeros_like(alive)
    for i in range(len(seeds)):
        lo, hi = np.asarray(g.b.wall_lo[i]), np.asarray(g.b.wall_hi[i])
        q = p_all[i, :-1][:, :, None, :]
        touch[i] = np.linalg.norm(q - np.clip(q, lo, hi), axis=-1).min(-1) < cfg.radius + 0.02
    t_ = touch & alive
    n = alive.sum()
    print(f"{name:24s} won {np.mean(np.asarray(r['outcome']) == 1):.0%}  full-speed turning {full_turn.sum() / n:.0%} "
          f"(while seeing an enemy {(full_turn & seeing).sum() / max((alive & seeing).sum(), 1):.0%})  "
          f"full spins per operator-round {spins / (turn.shape[0] * len(ops)):.2f}  pressed on a wall {wall / n:.0%}  "
          f"flicks {flip.sum() / n:.0%}  speed off walls {speed[alive & ~touch].mean():.2f} m/s  "
          f"stuck on a wall {np.mean(speed[t_] < 0.1) if t_.any() else 0:.0%} of wall time", flush=True)
