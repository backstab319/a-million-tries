"""The pit wall (es_strategist): widened for race control, and a race with race control on."""

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import es_strategist as es
from ailearns.agents.ppo_car import init_params as driver_params
from ailearns.games import race, race_control, race_field


def test_widened_it_decides_as_before():
    cfg = es.StratConfig()
    old = es.init_params(cfg, jax.random.PRNGKey(0))
    old = dict(old, w1=old["w1"] + 0.3 * jax.random.normal(jax.random.PRNGKey(1), old["w1"].shape))
    rc_cfg = es.StratConfig(race_control=True)
    new = es.widen(old, es.n_senses(rc_cfg))
    assert es.n_senses(rc_cfg) == 33 and new["w1"].shape == (33, cfg.hidden) and new["lin"].shape == (33, 4)
    x = jax.random.uniform(jax.random.PRNGKey(2), (16, 33))
    for a, b in zip(es.decide(new, x), es.decide(old, x[:, :25])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert len(race_control.STRATEGIST_SENSES) == 8


def test_a_race_with_race_control():
    """A short race runs, a 115-sense driver is widened, and the stewards' numbers come back."""
    cfg = es.StratConfig(race_control=True, n_cars=4, n_learn=2, laps=1, cap_s=30.0)
    rcfg = race.RaceConfig(time_limit=cfg.cap_s, pit_lane=True)
    F = race_field.RaceField(race.make_tracks(range(1), rcfg), n_cars=cfg.n_cars, laps=cfg.laps, cfg=rcfg)
    driver = driver_params(jax.random.PRNGKey(3), F.obs_size, 256, -0.5)
    play = es.make_race(cfg, F, driver)
    out = jax.jit(play)(es.init_params(cfg, jax.random.PRNGKey(0)), jax.random.split(jax.random.PRNGKey(4), 2))
    for k in ("pos", "rule_pos", "dsq", "penalty_s", "sc_calls", "vsc", "sc", "red"):
        assert out[k].shape == (2,)
    assert jnp.all((out["pos"] >= 0) & (out["pos"] <= 3))
