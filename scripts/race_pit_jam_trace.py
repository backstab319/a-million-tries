"""Episode 16 step 14i: why do cars sit in the pit lane under the Safety Car (14h: one stopped 768 s)? Runs races with
all 8 cars on the team's rule (as 14h) and sorts every second a car is stopped (< 1 m/s) in the pit lane:
  work      the crew is on it (tyres 2.6 s, a nose 8 s, pulling it back to its mark)
  release   the work's done, held: a car is coming down the fast lane (`RaceField.pit`'s unsafe release)
  queued    another car is stopped or crawling within 15 m ahead of it in the lane
  served    already served, nothing ahead: sitting after its stop
  missed    within 15 m of its own box, not being served (not on its marks: `pit` needs it across, stopped, in the box)
  nothing   none of those: the driver just stopped
by the status when it entered the lane (green, VSC, SC), and prints the worst passes (sampled seconds).

    uv run --group jax python scripts/race_pit_jam_trace.py runs/strat-ep16-1:0 --driver runs/race-d1-8:8 [--races 32]
"""

import argparse

import jax
import numpy as np
import torch

from ailearns.agents.es_strategist import StratConfig, make_field, make_race
from ailearns.agents.ppo_car import load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_control

ap = argparse.ArgumentParser()
ap.add_argument("strategist")
ap.add_argument("--driver", required=True)
ap.add_argument("--races", type=int, default=32)
ap.add_argument("--worst", type=int, default=6)
a = ap.parse_args()

sp, _, sk = a.strategist.partition(":")
ck = torch.load(pick_checkpoint(sp, sk or "last"), map_location="cpu", weights_only=False)
cfg = StratConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["config"].items()
                     if k in StratConfig.__dataclass_fields__})
cfg = StratConfig(**{**cfg.__dict__, "driver": a.driver})
F = make_field(cfg, seeds=list(race.TEST_SEEDS))
dp, _, dk = a.driver.partition(":")
driver, _ = load(pick_checkpoint(dp, dk or "last"))
params = {k: jax.numpy.asarray(v.numpy()) for k, v in ck["model"].items()}
play = jax.jit(make_race(cfg, F, driver, "rule", trace=True))
o = jax.device_get(play(params, jax.random.split(jax.random.PRNGKey(4242), a.races)))
tr = {k: np.asarray(v) for k, v in o["trace"].items()}
dt = F.cfg.dt
KINDS = ("work", "release", "queued", "served", "missed", "nothing")
names = {0: "green", race_control.VSC: "VSC", race_control.SC: "SC", race_control.RED: "red"}


def sort(r, t, car):
    """what a car stopped in the lane at tick t is doing."""
    sv = tr["service"][r, t, car]
    if sv >= 0.01:
        return "work"
    if sv > 0:
        return "release"
    ps, lane, v = tr["pit_s"][r, t], tr["lane"][r, t].copy(), tr["v"][r, t]
    lane[car] = False
    if (lane & (ps > ps[car]) & (ps - ps[car] < 15.0) & (v < 3.0)).any():
        return "queued"
    if tr["served"][r, t, car]:
        return "served"
    if abs(tr["to_box"][r, t, car]) < 15.0:
        return "missed"
    return "nothing"


tot = {k: dict.fromkeys(KINDS, 0.0) for k in ("green", "VSC", "SC")}
n = dict.fromkeys(tot, 0)
passes = []
for r in range(a.races):
    lane, st = tr["lane"][r], tr["st"][r]
    for car in range(lane.shape[1]):
        ins = np.flatnonzero(~lane[:-1, car] & lane[1:, car]) + 1
        outs = np.flatnonzero(lane[:-1, car] & ~lane[1:, car]) + 1
        for t_in in ins:
            later = outs[outs > t_in]
            t_out = later[0] if len(later) else lane.shape[0]
            kind = names.get(int(st[t_in]), "?")
            if kind not in tot:
                continue
            n[kind] += 1
            secs = dict.fromkeys(KINDS, 0.0)
            for t in np.flatnonzero(tr["v"][r, t_in:t_out, car] < 1.0) + t_in:
                secs[sort(r, t, car)] += dt
            for s in KINDS:
                tot[kind][s] += secs[s]
            passes.append((sum(secs.values()), r, car, t_in, t_out, kind, secs))

print(f"{a.races} races; seconds stopped in the pit lane a pass, by what it was doing:")
print("| entered under | passes | " + " | ".join(KINDS) + " | all |")
print("|---|---|" + "---|" * (len(KINDS) + 1))
for k in tot:
    if n[k]:
        print(f"| {k} | {n[k]} | " + " | ".join(f"{tot[k][s] / n[k]:.1f}" for s in KINDS)
              + f" | {sum(tot[k].values()) / n[k]:.1f} |")

print("\nthe worst passes:")
for total, r, car, t_in, t_out, kind, secs in sorted(passes, key=lambda x: -x[0])[: a.worst]:
    print(f"race {r} car {car + 1}, entered under {kind}, {(t_out - t_in) * dt:.0f} s in the lane, stopped {total:.0f} s: "
          + ", ".join(f"{s} {v:.0f} s" for s, v in secs.items() if v > 0))
    stopped = np.flatnonzero(tr["v"][r, t_in:t_out, car] < 1.0) + t_in
    for t in stopped[:: max(len(stopped) // 6, 1)][:6]:
        ps, lane = tr["pit_s"][r, t], tr["lane"][r, t]
        near = [f"car {j + 1} {ps[j] - ps[car]:+.0f} m at {tr['v'][r, t, j] * 3.6:.0f} km/h"
                + (" (in its box)" if tr["service"][r, t, j] > 0 else "")
                for j in np.flatnonzero(lane) if j != car and abs(ps[j] - ps[car]) < 60]
        print(f"    t={t * dt:6.0f} s {sort(r, t, car):8s} box {tr['to_box'][r, t, car]:+.0f} m, "
              f"lat {tr['lat'][r, t, car]:+.1f} m, status {names.get(int(tr['st'][r, t]), '?')}; "
              + ("; ".join(near) or "nobody within 60 m"), flush=True)
