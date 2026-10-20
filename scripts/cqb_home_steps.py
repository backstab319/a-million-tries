"""Episode 13, before training: what each new rule does to the episode 12 brains (like for like, cqb_versus.py),
adding one at a time: the kill house (episode 12's game), a home, + bullets through drywall, + the hostage can be hit.
Rules: episode 12's (2v2, split entry, one spare magazine, the flashbang fix, the hostage 6 m deep, walked out, heard).

    uv run --group jax python scripts/cqb_home_steps.py --assault runs/cqb-e14-2v2-extract:300 \\
        --defense runs/cqb-d23-2v2-extract-noise:300,camper
    uv run ailearns cloud script "scripts/cqb_home_steps.py --assault ... --defense ..." --on kaggle
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EP12 = ["--teams", "2,2", "--split", "--spare", "1", "--blind-away", "0.25", "--hostage", "--hostage-deep", "6",
        "--extract", "--hostage-noise", "8"]
STEPS = [("the kill house (episode 12)", []), ("a home", ["--house", "home"]),
         ("a home + drywall", ["--house", "home", "--through-drywall", "0.5"]),
         ("a home + drywall + the hostage can be hit", ["--house", "home", "--through-drywall", "0.5", "--hostage-hit"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assault", required=True)
    ap.add_argument("--defense", required=True)
    ap.add_argument("--houses", default="64-191")
    ap.add_argument("--rules", default="runs/cqb-d23-2v2-extract-noise:300")
    a = ap.parse_args()
    for name, extra in STEPS:
        print(f"\n### {name}\n", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "cqb_versus.py"), "--assault", a.assault, "--defense",
                        a.defense, "--rules", a.rules, "--houses", a.houses, *EP12, *extra], check=True)


if __name__ == "__main__":
    main()
