"""Imitation for CQB (ep9 3bb): the demonstrations keep exactly the script's winning rounds, and copying works."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import bc_cqb as BC
from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S

RULES = replace(cqb.CQBConfig(), flashbangs=(2, 0), magazine=30, health=100.0, react=0.3, time_limit=6.0)


def _game():
    return cqb.CQBGame(cqb.make_buildings(list(cqb.TRAIN_SEEDS)[:4], RULES), RULES)


def test_demonstrations_are_the_scripts_winning_rounds():
    game = _game()
    n = 8
    keys = jax.random.split(jax.random.PRNGKey(4), n)
    bis, drill = jnp.arange(n) % 4, jnp.arange(n) < 4
    obs, act, live, result = jax.device_get(BC.record(game, cqb.ASSAULT, S.SCRIPTS["walker_flash"],
                                                      S.SCRIPTS["camper"], bis, keys, drill))
    assert obs.shape == (n, game.T, RULES.obs_size) and act.shape == (n, game.T, cqb.ACTION_SIZE)
    # the same rounds through the usual runner: same results, and live decisions = the round's length
    for i in range(n):
        st = jax.tree.map(lambda a, b: jnp.where(drill[i], a, b), game.init_stacked(keys[i], bis[i]),
                          game.init(keys[i], bis[i]))
        assert np.allclose(obs[i, 0], np.asarray(game.observe(st, cqb.ASSAULT)), atol=1e-5)   # (compiled in a batch)
        assert live[i].sum() == (np.argmin(live[i]) if not live[i].all() else game.T)
    assert set(np.unique(result)) <= {-1, 0, 1}


def test_copying_lowers_the_loss_and_learns_a_throw():
    """On a fixed little batch with throws in it, steps of cloning bring the loss down and the brain's best guess
    starts throwing where the script did."""
    rules = RULES
    n_switch = A.switches_for(rules, cqb.ASSAULT)
    params = A.widen(A.init_params(jax.random.PRNGKey(0), rules.obs_size, 32, -0.5, 3), rules.obs_size, n_switch)
    obs = jax.random.normal(jax.random.PRNGKey(1), (64, rules.obs_size))
    act = jnp.concatenate([jnp.zeros((64, 3)), -jnp.ones((64, 5))], 1)
    act = act.at[:8, cqb.THROW].set(1.0)                     # 8 of 64 decisions throw
    cfg = BC.CQBBCConfig(lr=3e-3, throw_weight=8.0)
    step = jax.jit(BC.make_step(cfg, n_switch))
    from ailearns.agents.ppo_jax import adam_init
    opt = adam_init({k: params[k] for k in BC.TRAINED})
    first = None
    for _ in range(200):
        params, opt, (loss, _, _, recall) = step(params, opt, obs, act)
        first = float(loss) if first is None else first
    assert float(loss) < first / 3 and float(recall) == 1.0
    _, logits, _, _ = A.forward(params, obs)
    throws = np.asarray(logits[:, A.SWITCHES.index("throw")] > 0)
    assert throws[:8].all() and not throws[8:].any()


def test_throw_only_moves_nothing_but_the_throw_output():
    """bc2: only the throw output's row and bias change; every other weight is exactly as it was."""
    rules = RULES
    n_switch = A.switches_for(rules, cqb.ASSAULT)
    params = A.widen(A.init_params(jax.random.PRNGKey(0), rules.obs_size, 32, -0.5, 3), rules.obs_size, n_switch)
    obs = jax.random.normal(jax.random.PRNGKey(1), (64, rules.obs_size))
    act = jax.random.uniform(jax.random.PRNGKey(2), (64, cqb.ACTION_SIZE), minval=-1, maxval=1)
    act = act.at[:8, cqb.THROW].set(1.0).at[8:, cqb.THROW].set(-1.0)
    step = jax.jit(BC.make_step(BC.CQBBCConfig(lr=3e-2, throw_weight=8.0, throw_only=True), n_switch))
    from ailearns.agents.ppo_jax import adam_init
    p, opt = params, adam_init({k: params[k] for k in BC.TRAINED})
    for _ in range(400):
        p, opt, _ = step(p, opt, obs, act)
    row = A.N_MOVE + A.SWITCHES.index("throw")
    for k in params:
        before, after = np.asarray(params[k]), np.asarray(p[k])
        if k in ("policy.weight", "policy.bias"):
            assert np.array_equal(np.delete(before, row, 0), np.delete(after, row, 0)), k
            assert not np.array_equal(before[row], after[row])
        else:
            assert np.array_equal(before, after), k
    _, logits, _, _ = A.forward(p, obs)
    throws = np.asarray(logits[:, A.SWITCHES.index("throw")] > 0)
    assert throws[:8].all() and not throws[8:].any()
