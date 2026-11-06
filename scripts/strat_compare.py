"""Race day (episode 15), step 9: is the strategist better than the team's rule? Each strategist checkpoint runs cars
1-4 against the rule's cars 5-8 (grid slots shuffled, random designs, GP tyres, the frozen driver) in many races on the
test circuits (never trained on). u00000 is the rule itself (the strategist starts as it): the control, it should
come out level.
(ep16 step 14e) The race (race control, breakdowns, the driver, how many cars it runs) comes from the first
strategist's checkpoint config, so an ep16 pit wall is judged in the races it trained in; DSQs and Safety Car boxing
are shown when race control is on.

    uv run --group jax python scripts/strat_compare.py runs/strat-1:0 runs/strat-1:20 [--races 128]
"""

import argparse

import jax
import numpy as np

from ailearns.agents.es_strategist import StratConfig, make_field, make_race
from ailearns.agents.ppo_car import load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_field

ap = argparse.ArgumentParser()
ap.add_argument("strategists", nargs="+")
ap.add_argument("--races", type=int, default=128)
ap.add_argument("--learner-rule", choices=["rule", "no_sc"], default=None,
                help="(ep16 14f) cars 1-4 drive a scripted rule instead (the strategist's weights are unused)")
ap.add_argument("--driver", default="", help="default: the strategist's own (ep15: runs/race-fin-7:50)")
a = ap.parse_args()

import torch  # noqa: E402

p0, _, k0 = a.strategists[0].partition(":")
c0 = torch.load(pick_checkpoint(p0, k0 or "last"), map_location="cpu", weights_only=False).get("config", {})
cfg = StratConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in c0.items() if k in StratConfig.__dataclass_fields__})
cfg = StratConfig(**{**cfg.__dict__, "driver": a.driver or cfg.driver or "runs/race-fin-7:50"})
F = make_field(cfg, seeds=list(race.TEST_SEEDS))
dp, _, dk = cfg.driver.partition(":")
driver, _ = load(pick_checkpoint(dp, dk or "last"))
play = jax.jit(make_race(cfg, F, driver, a.learner_rule))
keys = jax.random.split(jax.random.PRNGKey(4242), a.races)
K, rc = cfg.n_learn, cfg.race_control
print(f"{a.races} races on {len(race.TEST_SEEDS)} test circuits, strategist cars 1-{K} vs the rule's {K + 1}-{cfg.n_cars}, "
      f"driver {cfg.driver}, race control {rc}, {cfg.incidents} breakdowns a race"
      + (f", cars 1-{K} on the scripted rule '{a.learner_rule}'" if a.learner_rule else ""))
print("| strategist | its position | rule's position | finished (its / rule) | stops a car (its / rule) | starts soft / hard |"
      + (" DSQ a race (its / rule) | boxed under SC/VSC (of its cars' chances) |" if rc else ""))
print("|---|---|---|---|---|---|" + ("---|---|" if rc else ""))
for s in a.strategists:
    sp, _, sk = s.partition(":")
    ck = torch.load(pick_checkpoint(sp, sk or "last"), map_location="cpu", weights_only=False)
    params = {k: jax.numpy.asarray(v.numpy()) for k, v in ck["model"].items()}
    o = {k: np.asarray(v) for k, v in jax.device_get(play(params, keys)).items()}
    se = lambda x: x.std() / np.sqrt(len(x))  # noqa: E731
    print(f"| {s} | P{o['pos'].mean() + 1:.2f} ± {se(o['pos']):.2f} | P{o['rule_pos'].mean() + 1:.2f} | "
          f"{o['finished'].mean() / K:.0%} / {o['rule_finished'].mean() / K:.0%} | "
          f"{o['stops'].mean() / K:.2f} / {o['rule_stops'].mean() / K:.2f} | "
          f"{o['soft_start'].mean() / K:.0%} / {o['hard_start'].mean() / K:.0%} |"
          + (f" {o['dsq'].mean():.2f} / {o['rule_dsq'].mean():.2f} | {o['sc_calls'].mean():.2f} of {o['sc_moments'].mean():.2f} |"
             if rc else ""), flush=True)
    if "lane_loss" in o:  # (ep16 step 14g) seconds a pass through the pit lane loses to the field, by the status at entry
        L, C = o["lane_loss"].ravel(), o["lane_cat"].ravel()
        parts = []
        for j, w in enumerate(("green", "vsc", "sc", "red")):
            x = L[C == j]
            if len(x):
                q = np.percentile(x, [25, 50, 75])
                parts.append(f"{w} median {q[1]:.1f} s (IQR {q[0]:.1f}-{q[2]:.1f}; {len(x) / a.races:.2f} a race)")
        print("    pit lane loss to the field (all 8 cars): " + ", ".join(parts), flush=True)
