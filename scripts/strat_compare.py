"""Race day (episode 15), step 9: is the strategist better than the team's rule? Each strategist checkpoint runs cars
1-4 against the rule's cars 5-8 (grid slots shuffled, random designs, GP tyres, the frozen driver) in many races on the
test circuits (never trained on). u00000 is the rule itself (the strategist starts as it): the control, it should
come out level.

    uv run --group jax python scripts/strat_compare.py runs/strat-1:0 runs/strat-1:20 [--races 128]
"""

import argparse

import jax
import numpy as np

from ailearns.agents.es_strategist import StratConfig, make_race
from ailearns.agents.ppo_car import load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_field

ap = argparse.ArgumentParser()
ap.add_argument("strategists", nargs="+")
ap.add_argument("--races", type=int, default=128)
ap.add_argument("--driver", default="runs/race-fin-7:50")
a = ap.parse_args()

import torch  # noqa: E402

cfg = StratConfig(driver=a.driver)
rcfg = race.RaceConfig(time_limit=cfg.cap_s, pit_lane=True)
F = race_field.RaceField(race.make_tracks(list(race.TEST_SEEDS), rcfg), n_cars=cfg.n_cars, laps=cfg.laps, cfg=rcfg,
                         gp_km=cfg.gp_km, tyre_cal=cfg.tyre_cal)
dp, _, dk = a.driver.partition(":")
driver, _ = load(pick_checkpoint(dp, dk or "last"))
play = jax.jit(make_race(cfg, F, driver))
keys = jax.random.split(jax.random.PRNGKey(4242), a.races)
print(f"{a.races} races on {len(race.TEST_SEEDS)} test circuits, strategist cars 1-4 vs the rule's 5-8, driver {a.driver}")
print("| strategist | its position | rule's position | finished (its / rule) | stops a car (its / rule) | starts soft / hard |")
print("|---|---|---|---|---|---|")
for s in a.strategists:
    sp, _, sk = s.partition(":")
    ck = torch.load(pick_checkpoint(sp, sk or "last"), map_location="cpu", weights_only=False)
    params = {k: jax.numpy.asarray(v.numpy()) for k, v in ck["model"].items()}
    o = {k: np.asarray(v) for k, v in jax.device_get(play(params, keys)).items()}
    se = lambda x: x.std() / np.sqrt(len(x))  # noqa: E731
    print(f"| {s} | P{o['pos'].mean() + 1:.2f} ± {se(o['pos']):.2f} | P{o['rule_pos'].mean() + 1:.2f} | "
          f"{o['finished'].mean() / 4:.0%} / {o['rule_finished'].mean() / 4:.0%} | "
          f"{o['stops'].mean() / 4:.2f} / {o['rule_stops'].mean() / 4:.2f} | "
          f"{o['soft_start'].mean() / 4:.0%} / {o['hard_start'].mean() / 4:.0%} |", flush=True)
