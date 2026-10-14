"""The numbers we say about a CQB run (ep9 3aj), done the honest way (playbook section 4): choose a checkpoint on test
buildings 0-63, then report it on 64-255, which nothing was chosen with.

- choose: every `every`-th checkpoint and the last, best guess, one round per building, against each member of the
  opponents it trained against (normal starts); the best mean win rate wins (ties: the later one).
- report: the chosen checkpoint against a standard list (both sides' scripts and the reference brains; `against`),
  normal starts, and drill starts too for a run that trained with drills.

Each opponent's rounds are compiled once with the brain's weights as an input, so every checkpoint plays through the
same compiled code. Runs at the end of a Kaggle training job (on the GPU it's there anyway) or on its own
(`ailearns cloud eval runs/x`, a Kaggle CPU job: no GPU slot, no GPU quota), or here:

    uv run --group jax python -m ailearns.agents.cqb_eval runs/cqb-a5-drill-mask
-> runs/<run>/eval.json and a markdown table.
"""

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S

STANDARD = {  # who a run is reported against (the reference brains of devlog 3s/3v)
    "assault": "camper,wanderer,runs/cqb-d1-vs-v13:381,runs/cqb-d2-vs-walker:381,runs/cqb-d3-pool:953,runs/cqb-d5-guns:900",
    "defense": "walker,walker_flash,runs/cqb-v13-camper-from-512:1250,runs/cqb-a2-pool:953",
}
CHOOSE, REPORT = (0, 64), (64, 256)   # test buildings
SEED_CHOOSE, SEED_REPORT = 1000, 5000


def brains_needed(cfg, against: str | None = None) -> list[str]:
    """The run names (in runs/) a report reads besides the run itself."""
    side = cfg["side"] if isinstance(cfg, dict) else cfg.side
    opp = cfg["opponent"] if isinstance(cfg, dict) else cfg.opponent
    specs = (against or STANDARD[side]).split(",") + opp.split(",")
    return sorted({Path(m.strip().partition(":")[0]).name for m in specs if m.strip() and m.strip() not in S.SCRIPTS})


def _game_cfg(ck: dict) -> cqb.CQBConfig:
    return cqb.CQBConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cqb"].items()})


def _matchup(game: cqb.CQBGame, me: int, template: dict, opponent: str, n: int, seed: int, stacked: bool = False,
             sample_words: bool = False):
    """A compiled function (params) -> results of one best-guess round per building 0..n-1 against `opponent`."""
    other = S.SCRIPTS.get(opponent) or A.opponent_player(opponent, 1 - me)

    @jax.jit
    def play(params):
        mine = A.brain(params, me, sample_words=sample_words)
        a, d = (mine, other) if me == cqb.ASSAULT else (other, mine)
        last, first, _ = S.rounds(game, a, d, jnp.arange(n), seed, stacked=stacked)
        return {"outcome": last.outcome, "alive": last.alive, "bangs": last.bangs}
    return play


def _summary(r: dict, me: int, flashbangs: int, game: cqb.CQBGame | None = None) -> dict:
    """Won, trades (both sides out), time-outs (neither), flashbangs thrown a round (by the whole side). Without a
    game: 1v1."""
    o, alive = np.asarray(r["outcome"]), np.asarray(r["alive"])
    won = o == (1 if me == cqb.ASSAULT else -1)
    n_a, mine = (game.nA, game.side_ops(me)) if game is not None else (1, [me])
    a_out, d_out = game.sides_out(alive) if game is not None else (~alive[:, :n_a].any(1), ~alive[:, n_a:].any(1))
    return {"won": float(won.mean()), "trades": float((a_out & d_out).mean()),
            "timeouts": float((~a_out & ~d_out).mean()),
            "flashbangs_used": float((flashbangs * len(mine) - np.asarray(r["bangs"])[:, mine].sum(1)).mean())
            if flashbangs else 0.0}


def report(run: str | Path, against: str | None = None, every: int = 50) -> dict:
    run = Path(run)
    cks = sorted(run.glob("checkpoints/u*.pt"))
    updates = [int(p.stem[1:]) for p in cks]
    last = max(updates)
    pick = sorted({u for u in updates if u and u % every == 0} | {last})
    _, ck = A.load(run / "checkpoints" / f"u{last:05d}.pt")
    cfg, game_cfg = ck["config"], _game_cfg(ck)
    me = cqb.ASSAULT if cfg["side"] == "assault" else cqb.DEFENSE
    bangs = game_cfg.flashbangs[me]
    test = list(cqb.TEST_SEEDS)
    t0 = time.time()
    g_choose = cqb.CQBGame(cqb.make_buildings(test[CHOOSE[0]:CHOOSE[1]], game_cfg), game_cfg)
    g_report = cqb.CQBGame(cqb.make_buildings(test[REPORT[0]:REPORT[1]], game_cfg), game_cfg)
    params = {u: A.load(run / "checkpoints" / f"u{u:05d}.pt")[0] for u in pick}

    # choose, against what it trained against
    trained = [m.strip() for m in cfg["opponent"].split(",") if m.strip()]
    choose = {u: {} for u in pick}
    for m in trained:
        play = _matchup(g_choose, me, params[last], m, CHOOSE[1] - CHOOSE[0], SEED_CHOOSE)
        for u in pick:
            choose[u][m] = _summary(jax.device_get(play(params[u])), me, bangs, g_choose)["won"]
    score = {u: float(np.mean(list(v.values()))) for u, v in choose.items()}
    best = max(pick, key=lambda u: (score[u], u))
    print(f"chose update {best} (mean win on test {CHOOSE[0]}-{CHOOSE[1] - 1} vs {', '.join(trained)}: "
          f"{score[best]:.0%}; last {last}: {score[last]:.0%})", flush=True)

    # report, against the standard list
    starts = [False, True] if cfg.get("stacked") else [False]
    table = {}
    for m in (x.strip() for x in (against or STANDARD[cfg["side"]]).split(",") if x.strip()):
        for stacked in starts:
            play = _matchup(g_report, me, params[best], m, REPORT[1] - REPORT[0], SEED_REPORT, stacked)
            table[f"{m}{' (drill starts)' if stacked else ''}"] = _summary(jax.device_get(play(params[best])), me, bangs,
                                                                           g_report)
    out = {"run": run.name, "side": cfg["side"], "chosen_update": best, "last_update": last,
           "choose": {"buildings": list(CHOOSE), "seed": SEED_CHOOSE, "opponents": trained,
                      "mean_win_by_update": score},
           "report": {"buildings": list(REPORT), "seed": SEED_REPORT, "table": table},
           "flashbangs": list(game_cfg.flashbangs), "seconds": round(time.time() - t0, 1)}
    (run / "eval.json").write_text(json.dumps(out, indent=1))
    print(f"| {run.name} u{best} vs | won | trades | time-outs |" + (" flashbangs |" if bangs else ""))
    print("|---|---|---|---|" + ("---|" if bangs else ""))
    for k, v in table.items():
        print(f"| {k} | {v['won']:.0%} | {v['trades']:.0%} | {v['timeouts']:.0%} |"
              + (f" {v['flashbangs_used']:.2f} |" if bangs else ""))
    print(f"(report: test buildings {REPORT[0]}-{REPORT[1] - 1}, seed {SEED_REPORT}; {out['seconds']} s)", flush=True)
    return out


if __name__ == "__main__":
    report(sys.argv[1], *(sys.argv[2:3] or [None]))
