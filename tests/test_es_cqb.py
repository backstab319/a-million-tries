"""Evolution strategies for CQB (ep9 3az): the ranking and one generation's plumbing."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import es_cqb as ES
from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S


def test_centered_ranks_share_ties():
    r = np.asarray(ES.centered_ranks(jnp.asarray([0.0, 0.0, 1.0])))
    assert np.allclose(r, [-0.25, -0.25, 0.5])
    r = np.asarray(ES.centered_ranks(jnp.asarray([3.0, 1.0, 2.0, 0.0])))
    assert np.allclose(r, [0.5, -1 / 6, 1 / 6, -0.5])


def _setup(**kw):
    rules = replace(cqb.CQBConfig(), flashbangs=(2, 0), magazine=30, health=100.0, react=0.3, time_limit=3.0)
    game = cqb.CQBGame(cqb.make_buildings(list(cqb.TRAIN_SEEDS)[:4], rules), rules)
    params = A.init_params(jax.random.PRNGKey(0), rules.obs_size, 16, -0.5, A.switches_for(rules, cqb.ASSAULT))
    cfg = ES.CQBESConfig(pop=4, rounds=2, buildings=4, **kw)
    sig = ES._sigmas(cfg, params, A.N_MOVE + A.SWITCHES.index("throw"))
    gen = jax.jit(ES.make_generation(cfg, game, cqb.ASSAULT, [S.SCRIPTS["camper"]], sig))
    groups = [(jnp.asarray([0, 3]), jax.random.split(jax.random.PRNGKey(1), 2), jnp.asarray([True, False]))]
    return params, sig, gen, groups


def test_no_change_no_gradient():
    """Members that are as good as the brain itself (changes far too small to change a decision) score the same, so
    the ranked changes cancel: no step."""
    params, _, gen, groups = _setup(sigma=1e-7, sigma_throw=1e-7)
    grad, stats = gen(params, jax.random.PRNGKey(2), groups)
    assert all(float(jnp.abs(g).max()) == 0 for g in grad.values())
    assert set(grad) == set(ES.EVOLVED) and stats["score"] == stats["score_brain"]


def test_sigmas_and_a_generation():
    """The throw output's row and bias explore with sigma_throw, the rest with sigma; a generation gives a finite
    step for every evolved weight and nothing else, and stats in range."""
    params, sig, gen, groups = _setup(sigma=0.02, sigma_throw=0.5)
    row = A.N_MOVE + A.SWITCHES.index("throw")
    pw = np.asarray(sig["policy.weight"])
    assert np.allclose(pw[row], 0.5) and np.allclose(np.delete(pw, row, 0), 0.02)
    assert np.isclose(float(sig["policy.bias"][row]), 0.5) and np.isclose(float(sig["body.2.weight"].max()), 0.02)
    grad, stats = gen(params, jax.random.PRNGKey(2), groups)
    for n in ES.EVOLVED:
        assert grad[n].shape == params[n].shape and bool(jnp.isfinite(grad[n]).all())
    assert -1 <= float(stats["score"]) <= 1 and 0 <= float(stats["members_throwing"]) <= 1
    assert float(stats["won_brain"]) == (float(stats["won_brain_full"]) + float(stats["won_brain_drill"])) / 2


def test_guard_counts_the_brains_wins_and_coaching_is_off_by_default():
    """The guard's yardstick (wins only) agrees with the generation's own count of the brain's wins; without a
    bonus the members' score is their result alone."""
    params, sig, gen, groups = _setup(sigma=0.02, sigma_throw=0.5)
    assert ES.CQBESConfig().blind_bonus == 0 and ES.CQBESConfig().self_blind_cost == 0 and ES.CQBESConfig().guard < 0
    rules = replace(cqb.CQBConfig(), flashbangs=(2, 0), magazine=30, health=100.0, react=0.3, time_limit=3.0)
    game = cqb.CQBGame(cqb.make_buildings(list(cqb.TRAIN_SEEDS)[:4], rules), rules)
    judge = jax.jit(ES.make_judge(game, cqb.ASSAULT, [S.SCRIPTS["camper"]]))
    _, stats = gen(params, jax.random.PRNGKey(2), groups)
    assert int(judge(params, groups)) == round(float(stats["won_brain"]) * 2)
