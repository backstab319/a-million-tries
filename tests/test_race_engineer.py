"""Judging designs with a driver (episode 14)."""

import jax
import numpy as np

from ailearns.agents.ppo_car import init_params
from ailearns.agents.race_engineer import judge
from ailearns.games import race
from ailearns.games import race_car as rc


def test_judge_scores_every_design_on_every_circuit():
    cfg = race.RaceConfig()
    game = race.RaceGame(race.make_tracks(range(2), cfg), cfg)
    params = init_params(jax.random.key(0), cfg.obs_size, 32, -0.5)  # an untrained brain: just the plumbing
    designs = np.stack([rc.design(power_kw=100), rc.design(power_kw=700), rc.F1_LIKE[0]])
    out = judge(params, game, designs, seconds=3.0)
    assert out["metres"].shape == (3,) and out["per_track"].shape == (3, 2)
    assert np.all(np.isfinite(out["metres"])) and np.all((out["crash"] >= 0) & (out["crash"] <= 1))


def test_evolve_writes_every_generation(tmp_path):
    import json
    from ailearns.agents.race_engineer import evolve
    cfg = race.RaceConfig()
    game = race.RaceGame(race.make_tracks(range(2), cfg), cfg)
    params = init_params(jax.random.key(0), cfg.obs_size, 32, -0.5)
    hist = evolve(params, game, tmp_path, gens=2, pop=4, circuits=1, seconds=2.0)
    saved = json.loads((tmp_path / "history.json").read_text())
    assert len(hist) == 2 and len(saved["history"]) == 2
    assert len(saved["history"][-1]["design"]) == len(rc.DESIGN)
    d = np.array(saved["history"][-1]["design"])
    assert np.all(d >= rc.DESIGN_LOW - 1e-3) and np.all(d <= rc.DESIGN_HIGH + 1e-3)
