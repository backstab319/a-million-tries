"""The whole episode-14 recipe in one go on one GPU machine (a Runpod pod, a Kaggle box, any CUDA host), for a
retrain after a physics change. Each step leaves runs/<tag>/DONE-<step> so a rerun picks up where it stopped:

  1. general driver   runs/race-<tag>-1       from scratch, 300M steps, random designs
  2. sweep            the power x wing grid judged by it; its best car under budget -> finalists/sweep-<tag>.json
  3. engineer         runs/eng-<tag>-1        from the sweep's best, judged by the general driver (steadier settings)
  4. its driver       runs/race-<tag>-spec1   on the engineer's car (100M, +-15%)
  5. engineer again   runs/eng-<tag>-2        judged by that driver, from round 1's car
  6. own drivers      runs/race-<tag>-{sweep,f1,eng2}   each finalist its own driver (same recipe)

    PYTHONPATH=src python scripts/race_pipeline.py v4      # from the repo root; then fetch runs/ and finalists/
The final race runs on the laptop (scripts/race_final.py), as every number in the devlog does.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIN = ROOT / "episodes" / "14-designs-its-own-car" / "finalists"


def main(tag: str) -> None:
    import jax
    print("jax", jax.__version__, "| devices:", jax.devices(), flush=True)
    if jax.devices()[0].platform == "cpu":
        raise SystemExit("no GPU")
    from ailearns.agents.ppo_race import RacePPOConfig, train
    from ailearns.games import race_car as rc
    marks = ROOT / "runs" / f"pipeline-{tag}"
    marks.mkdir(parents=True, exist_ok=True)

    def step(name, fn):
        if (marks / f"DONE-{name}").exists():
            print("skip", name, flush=True)
            return
        print("START", name, flush=True)
        fn()
        (marks / f"DONE-{name}").write_text("ok")
        print("DONE", name, flush=True)

    def last(run):
        return int(sorted((ROOT / "runs" / run / "checkpoints").glob("u*.pt"))[-1].stem[1:])

    def driver(run, design_file=None, steps=100e6):
        center, init = (), ""
        if design_file:
            center = tuple(json.loads(Path(design_file).read_text())["history"][-1]["design"])
            init = f"runs/race-{tag}-1:{last(f'race-{tag}-1')}"
        train(RacePPOConfig(total_steps=int(steps), n_envs=4096, init_from=init, design_center=center,
                            design_spread=0.15 if center else 0.0), str(ROOT / "runs" / run), max_hours=2)

    def engineer(judge, out, start):
        subprocess.run([sys.executable, str(ROOT / "src/ailearns/agents/race_engineer.py"),
                        f"runs/{judge}:{last(judge)}", f"runs/{out}", "--gens", "150", "--pop", "96", "--circuits", "48",
                        "--anneal", "0.3", "--start-from", str(start)], check=True, cwd=ROOT)

    def sweep():
        subprocess.run([sys.executable, str(ROOT / "scripts/race_sweep.py"), f"runs/race-{tag}-1"], check=True, cwd=ROOT)
        res = json.loads((ROOT / "out" / "ep14" / f"sweep-race-{tag}-1.json").read_text())
        m, cost = np.asarray(res["metres"]), np.asarray(res["cost_k"])
        m = np.where(cost <= res["budget_k"], m, -np.inf)
        i, j = np.unravel_index(np.argmax(m), m.shape)
        pw, w = res["power_kw"][i], res["wing"][j]
        d = np.asarray(rc.design(power_kw=pw, wing_front=w, wing_rear=w, drive_front=0.0), float)
        (FIN / f"sweep-{tag}.json").write_text(json.dumps(
            {"label": f"the sweep's best on physics {tag}: {pw:.0f} kW, wings {w:.2f}, RWD ({m[i, j]:.0f} m)",
             "history": [{"design": d.round(4).tolist()}]}, indent=1))
        print(f"sweep best under budget: {pw:.0f} kW, wings {w:.2f}: {m[i, j]:.0f} m", flush=True)

    def finalist(src_run, name, label):
        h = json.loads((ROOT / "runs" / src_run / "history.json").read_text())["history"][-1]
        (FIN / f"{name}.json").write_text(json.dumps({"label": label, "history": [{"design": h["design"]}]}, indent=1))

    step("driver", lambda: driver(f"race-{tag}-1", steps=300e6))
    step("sweep", sweep)
    step("eng1", lambda: (engineer(f"race-{tag}-1", f"eng-{tag}-1", FIN / f"sweep-{tag}.json"),
                          finalist(f"eng-{tag}-1", f"eng-{tag}-1", f"eng-{tag}-1 gen 149 (the engineer, physics {tag})")))
    step("spec1", lambda: driver(f"race-{tag}-spec1", FIN / f"eng-{tag}-1.json"))
    step("eng2", lambda: (engineer(f"race-{tag}-spec1", f"eng-{tag}-2", FIN / f"eng-{tag}-1.json"),
                          finalist(f"eng-{tag}-2", f"eng-{tag}-2",
                                   f"eng-{tag}-2 gen 149 (the engineer, physics {tag}, judged by its own car's driver)")))
    step("own-sweep", lambda: driver(f"race-{tag}-sweep", FIN / f"sweep-{tag}.json"))
    step("own-f1", lambda: driver(f"race-{tag}-f1", FIN / "f1.json"))
    step("own-eng2", lambda: driver(f"race-{tag}-eng2", FIN / f"eng-{tag}-2.json"))
    (marks / "DONE").write_text("ok")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
