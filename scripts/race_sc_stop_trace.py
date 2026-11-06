"""Episode 16 step 14h: why is a stop under the Safety Car hardly cheaper than under green (14g)? Runs races with all 8
cars on the team's rule (race control, breakdowns, the ep16 driver), finds stops that start under the SC, and prints
each one's timeline: the deployment, the pit entry and exit, the car's speed in the lane and its time held in the box,
the field's median speed (racing cars on track) before, during and after, and the seconds lost to the field.

    uv run --group jax python scripts/race_sc_stop_trace.py runs/strat-ep16-1:0 --driver runs/race-d1-8:8 [--races 16]
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
ap.add_argument("--races", type=int, default=16)
ap.add_argument("--show", type=int, default=6, help="stops to print in detail")
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
tr = {k: np.asarray(v) for k, v in o["trace"].items()}  # (races, ticks, [cars])
dt = F.cfg.dt
names = {0: "green", race_control.VSC: "VSC", race_control.SC: "SC", race_control.RED: "red"}


def field_v(r, t0, t1, me):
    """median speed over [t0, t1) of the racing cars on track (not me)."""
    m = tr["alive"][r, t0:t1] & ~tr["lane"][r, t0:t1]
    m[:, me] = False
    v = np.where(m, tr["v"][r, t0:t1], np.nan)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")


rows, shown = [], 0
for r in range(a.races):
    st, lane = tr["st"][r], tr["lane"][r]
    for car in range(lane.shape[1]):
        ins = np.flatnonzero(~lane[:-1, car] & lane[1:, car]) + 1
        outs = np.flatnonzero(lane[:-1, car] & ~lane[1:, car]) + 1
        for t_in in ins:
            later = outs[outs > t_in]
            if not len(later) or not tr["alive"][r, t_in, car]:
                continue
            t_out = later[0]
            kind = names.get(int(st[t_in]), "?")
            dep = t_in
            while dep > 0 and st[dep - 1] == st[t_in]:
                dep -= 1
            held = float((tr["service"][r, t_in:t_out, car] > 0).sum() * dt)
            stopped = float((tr["v"][r, t_in:t_out, car] < 1.0).sum() * dt)
            lane_s = (t_out - t_in) * dt
            v_lane = float(np.median(tr["v"][r, t_in:t_out, car]))
            fv = field_v(r, t_in, t_out, car)
            fadv = np.nanmedian(np.where(tr["alive"][r, t_out] & ~tr["lane"][r, t_out], tr["prog"][r, t_out], np.nan)) \
                - np.nanmedian(np.where(tr["alive"][r, t_in] & ~tr["lane"][r, t_in], tr["prog"][r, t_in], np.nan))
            cadv = tr["prog"][r, t_out, car] - tr["prog"][r, t_in, car]
            loss = lane_s * (1 - cadv / max(fadv, 1.0))
            row = dict(kind=kind, race=r, car=car + 1, since_dep=(t_in - dep) * dt, lane_s=lane_s, held=held,
                       stopped=stopped, v_lane=v_lane * 3.6, v_field=fv * 3.6,
                       v_before=field_v(r, max(t_in - 600, 0), t_in, car) * 3.6, loss=loss,
                       status_out=names.get(int(st[t_out]), "?"))
            rows.append(row)
            if kind == "SC" and shown < a.show:
                shown += 1
                print(f"race {r} car {row['car']}: SC for {row['since_dep']:.0f} s when it entered; {lane_s:.1f} s in "
                      f"the lane (held {held:.1f} s, stopped {stopped:.1f} s, median {row['v_lane']:.0f} km/h); field "
                      f"{row['v_before']:.0f} km/h in the 30 s before, {row['v_field']:.0f} km/h during; status at exit "
                      f"{row['status_out']}; lost {loss:.1f} s", flush=True)

print(f"\n{a.races} races, {len(rows)} pit lane passes")
print("| entered under | passes | lane s | held s | stopped s | car km/h | field km/h before | field km/h during | "
      "s since deployment | loss s |")
print("|---|---|---|---|---|---|---|---|---|---|")
for kind in ("green", "VSC", "SC"):
    k = [x for x in rows if x["kind"] == kind]
    if k:
        med = lambda f: np.nanmedian([x[f] for x in k])  # noqa: E731
        print(f"| {kind} | {len(k)} | {med('lane_s'):.1f} | {med('held'):.1f} | {med('stopped'):.1f} | "
              f"{med('v_lane'):.0f} | {med('v_before'):.0f} | {med('v_field'):.0f} | "
              f"{med('since_dep'):.0f} | {med('loss'):.1f} |", flush=True)
