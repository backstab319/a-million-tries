"""ep11 10p: is a gap between the choice houses (test 0-63, seed 1000) and the report houses (64-255, seed 5000) the
houses or the dice? Each brain under its own rules against each Defense, on houses 0-63 with seed 1000 and 5000,
and on 64-255 with 5000 and 1000 (cqb_eval's matchup, best guess). Runs on Modal:

    uv run --group jax --group modal --env-file .env ailearns cloud script \
        "scripts/cqb_house_split.py runs/cqb-s1-solo-big-feed:200 runs/cqb-s1-solo-big-feed:572" --on modal
"""

import argparse
from pathlib import Path

import jax
import numpy as np

from ailearns.agents import cqb_eval as E
from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brains", nargs="+")
    ap.add_argument("--against", default="camper,runs/cqb-d5-guns:900,runs/cqb-d7-react:325")
    a = ap.parse_args()
    test = list(cqb.TEST_SEEDS)
    print("| brain | vs | houses 0-63, seed 1000 | 0-63, seed 5000 | 64-255, seed 5000 | 64-255, seed 1000 | 64-255, both |\n|---|---|---|---|---|---|---|")
    for spec in a.brains:
        run, _, u = spec.partition(":")
        params, ck = A.load(Path(run) / "checkpoints" / f"u{int(u):05d}.pt")
        cfg = E._game_cfg(ck)
        games = {(0, 64): cqb.CQBGame(cqb.make_buildings(test[0:64], cfg), cfg),
                 (64, 256): cqb.CQBGame(cqb.make_buildings(test[64:256], cfg), cfg)}
        for m in (x.strip() for x in a.against.split(",") if x.strip()):
            cells, rep = [], []
            for (lo, hi), seed in (((0, 64), 1000), ((0, 64), 5000), ((64, 256), 5000), ((64, 256), 1000)):
                g = games[(lo, hi)]
                r = jax.device_get(E._matchup(g, cqb.ASSAULT, params, m, hi - lo, seed)(params))
                won = E._summary(r, cqb.ASSAULT, 0, g)["won"]
                cells.append(f"{won:.0%}")
                if lo == 64:
                    rep.append(won)
            cells.append(f"**{np.mean(rep):.0%}**")
            print(f"| {spec.split('/')[-1]} | {m.split('/')[-1]} | " + " | ".join(cells) + " |", flush=True)


if __name__ == "__main__":
    main()
