"""The go/no-go test (racing plan section 7): is the best car in the middle of the range, or always at a limit?

A grid of designs over engine power x wing size (both wings the same), everything else mid-range and rear drive;
the driver judges each one (race_engineer.judge: average metres in 60 s from a standing start on training circuits).
Designs over the $1M budget are judged too but greyed out. Prints the table and draws the map (footage).

    uv run --group jax python scripts/race_sweep.py runs/race-d1 [--circuits 8] [--powers 10] [--wings 6]
    -> out/ep14/sweep-<run>.png and .json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pygame

from ailearns.agents.ppo_car import load
from ailearns.agents.race_engineer import judge
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race
from ailearns.games import race_car as rc
from ailearns.render import theme as T


def sweep(run: str, circuits: int, powers: int, wings: int, seconds: float) -> dict:
    path, _, ck = run.partition(":")
    params, meta = load(pick_checkpoint(path, ck or "last"))
    cfg = race.RaceConfig()
    game = race.RaceGame(race.make_tracks(list(race.TRAIN_SEEDS)[:circuits], cfg), cfg)
    kw = np.linspace(75, 750, powers)
    wing = np.linspace(0, 1, wings)
    designs = np.stack([rc.design(power_kw=p, wing_front=w, wing_rear=w, drive_front=0.0) for p in kw for w in wing])
    out = judge(params, game, designs, seconds=seconds)
    cost = np.array([float(rc.cost_k(d)) for d in designs])
    grid = lambda a: np.asarray(a).reshape(powers, wings).round(3).tolist()  # noqa: E731
    return {"run": run, "steps": meta["steps"], "circuits": circuits, "seconds": seconds, "power_kw": kw.tolist(),
            "wing": wing.tolist(), "metres": grid(out["metres"]), "crash": grid(out["crash"]), "cost_k": grid(cost),
            "budget_k": rc.BUDGET_K}


def draw(res: dict, out: str) -> None:
    T.init()
    W, H = T.WIDTH, T.HEIGHT
    surf = pygame.Surface((W, H))
    surf.fill(T.BG)
    m, cost, crash = (np.array(res[k]) for k in ("metres", "cost_k", "crash"))
    ok = cost <= res["budget_k"]
    lo, hi = m.min(), m.max()
    best = np.unravel_index(np.where(ok, m, -1).argmax(), m.shape)
    x0, y0, cw, chh = 330, 170, 130, 70
    T.text(surf, "WHICH CAR IS FASTEST?", (x0, 50), "display", 56, T.TEXT)
    T.text(surf, f"engine power x wing size, judged by the AI driver ({res['steps'] / 1e6:.0f}M steps of practice); "
           f"grey: over the ${res['budget_k'] / 1000:.0f}M budget", (x0, 118), "semibold", 26, T.MUTED)
    for i, kw in enumerate(res["power_kw"]):
        T.text(surf, f"{kw:.0f} kW", (x0 - 20, y0 + i * chh + chh // 2), "semibold", 26, T.MUTED, anchor="midright")
        for j, w in enumerate(res["wing"]):
            r = pygame.Rect(x0 + j * cw, y0 + i * chh, cw - 4, chh - 4)
            t = (m[i, j] - lo) / max(hi - lo, 1e-6)
            col = T.mix(T.PANEL, T.GREEN, t) if ok[i, j] else T.mix(T.PANEL, T.MUTED, 0.25)
            pygame.draw.rect(surf, col, r, border_radius=8)
            T.text(surf, f"{m[i, j] / 1000:.2f}", r.center, "semibold", 24, T.TEXT if ok[i, j] else T.MUTED,
                   anchor="center")
            if crash[i, j] > 0:
                T.text(surf, f"{crash[i, j]:.0%} crash", (r.centerx, r.bottom - 12), "semibold", 14, T.RED,
                       anchor="center")
            if (i, j) == best:
                pygame.draw.rect(surf, T.AMBER, r, 4, border_radius=8)
    for j, w in enumerate(res["wing"]):
        T.text(surf, f"wing {w:.1f}", (x0 + j * cw + cw // 2, y0 + len(res["power_kw"]) * chh + 20), "semibold", 24,
               T.MUTED, anchor="center")
    T.text(surf, "km covered in 60 s (avg of circuits)", (x0, H - 60), "semibold", 26, T.MUTED)
    pygame.image.save(surf, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--circuits", type=int, default=8)
    ap.add_argument("--powers", type=int, default=10)
    ap.add_argument("--wings", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=60.0)
    a = ap.parse_args()
    res = sweep(a.run, a.circuits, a.powers, a.wings, a.seconds)
    stem = Path("out/ep14") / f"sweep-{Path(a.run.split(':')[0]).name}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1))
    draw(res, str(stem.with_suffix(".png")))
    m, cost = np.array(res["metres"]), np.array(res["cost_k"])
    for i, kw in enumerate(res["power_kw"]):
        print(f"{kw:4.0f} kW  " + "  ".join(f"{v:6.0f}{'*' if c > res['budget_k'] else ' '}"
                                            for v, c in zip(m[i], cost[i])))
    print("(* over budget)", stem.with_suffix(".png"))
