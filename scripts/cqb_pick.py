"""Pick footage rounds: every round of a batch over test buildings, one line each, for `cqb_round` shots
(`batch: [first, n]` replays a round exactly as it was played here; ep9 4a).

  uv run --group jax python scripts/cqb_pick.py <assault> <defense> [--sample assault] [--first 64 --n 64 --seed 5000]
      [--rules runs/x:ck]   # whose rules (cqb_round's rules_from)
  uv run --group jax --group modal --env-file .env ailearns cloud script "scripts/cqb_pick.py ..." --on modal

Per round: building, result, length, who saw whom first (decision), shots fired by each, share of the building
Assault looked at."""

import argparse

import jax
import numpy as np

from ailearns.agents.ppo_cqb import brain, load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S


# record_cqb's player / game_config, without pygame (Modal's image has none)
def player(spec: str, who: int, sample: bool = False):
    if spec in S.SCRIPTS:
        return S.SCRIPTS[spec]
    run, _, ck = spec.partition(":")
    return brain(load(pick_checkpoint(run, ck or "last"))[0], who, sample_moves=sample)


def game_config(*specs: str) -> cqb.CQBConfig:
    cfg = cqb.CQBConfig()
    for spec in specs:
        if spec in S.SCRIPTS:
            continue
        run, _, ck = spec.partition(":")
        saved = load(pick_checkpoint(run, ck or "last"))[1]["cqb"]
        c = cqb.CQBConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})
        if c.obs_size > cfg.obs_size:
            cfg = c
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("assault")
    ap.add_argument("defense")
    ap.add_argument("--sample", default="")
    ap.add_argument("--rules", default="", help="whose rules (runs/x:ck), as cqb_round's rules_from (e.g. a script in 2v2)")
    ap.add_argument("--first", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=5000)
    a = ap.parse_args()
    cfg = game_config(a.rules) if a.rules else game_config(a.assault, a.defense)
    seeds = list(cqb.TEST_SEEDS)[a.first:a.first + a.n]
    game = cqb.CQBGame(cqb.make_buildings(seeds, cfg), cfg)
    A = player(a.assault, cqb.ASSAULT, a.sample in ("assault", "both"))
    D = player(a.defense, cqb.DEFENSE, a.sample in ("defense", "both"))
    r = S.play_rounds(game, A, D, np.arange(len(seeds)), seed=a.seed, record=True)
    h = r["history"]
    print(f"{a.assault} vs {a.defense} (sample: {a.sample or 'none'}), buildings {seeds[0]}..{seeds[-1]}, seed {a.seed}")
    print("building  result  secs  first-seen(A,D)  shots(A,D)  A looked at")
    for i, s in enumerate(seeds):
        hi = jax.tree.map(lambda x: np.asarray(x[i]), h)
        last = int(np.argmax(hi.done)) if hi.done.any() else game.T
        shots = hi.fired[:last + 1].sum(0)
        res = {1: "A wins", -1: "D wins", 0: "trade"}[int(hi.outcome[last])]
        if res == "D wins" and hi.alive[last][cqb.ASSAULT]:
            res = "time up"
        print(f"{s:8d}  {res:7s} {last * cfg.dt:5.1f}  {str(tuple(int(x) for x in r['first_seen'][i])):15s}"
              f"  {str(tuple(int(x) for x in shots)):10s}  {hi.cleared[last][cqb.ASSAULT].mean():.0%}")
    print(f"won {np.mean(r['outcome'] == 1):.0%}")


if __name__ == "__main__":
    main()
