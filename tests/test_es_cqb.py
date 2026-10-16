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


def test_aim_at_first_sight_matches_the_recorded_round():
    """ep11 10v: the rollout's "aim off him when we first saw each other" equals the same worked out from the round's
    recorded states (first decision either one sees the other: the angle from the attacker's facing to him); rounds
    where they never meet read pi; and the bonus is off by default."""
    assert ES.CQBESConfig().aim_bonus == 0
    rules = replace(cqb.CQBConfig(), flashbangs=(2, 0), magazine=30, health=100.0, react=0.3, time_limit=12.0)
    game = cqb.CQBGame(cqb.make_buildings(list(cqb.TRAIN_SEEDS)[:4], rules), rules)
    p, _ = A.load("runs/cqb-a12-react/checkpoints/u00300.pt") if __import__("os").path.exists(
        "runs/cqb-a12-react/checkpoints/u00300.pt") else (None, None)
    params = p if p is not None else A.init_params(jax.random.PRNGKey(0), rules.obs_size, 16, -0.5,
                                                   A.switches_for(rules, cqb.ASSAULT))
    n = 8
    bis, keys, drill = jnp.arange(n) % 4, jax.random.split(jax.random.PRNGKey(5), n), jnp.zeros(n, bool)
    opp = S.SCRIPTS["camper"]
    *_, off, _ = jax.device_get(ES._rollout(game, cqb.ASSAULT, params, opp, bis, keys, drill))
    n_in = params["body.0.weight"].shape[1]
    step = jax.jit(lambda st, k: game.step(st, jnp.stack([
        cqb.full_action(A.best_guess(params, game.observe(st, 0)[None, :n_in])[0]),
        cqb.full_action(opp(game, st, 1, k))]))[0])
    met = 0
    for i in range(n):                                          # the same round, step by step, every state kept
        st = game.init(keys[i], bis[i])
        want = np.pi
        for k in jax.random.split(jax.random.fold_in(keys[i], 1), game.T):
            new = step(st, k)
            if (bool(new.seen[0, 1]) or bool(new.seen[1, 0])) and not bool(st.done):
                d = np.asarray(new.pos[1] - new.pos[0])
                want = abs((np.arctan2(d[1], d[0]) - float(new.face[0]) + np.pi) % (2 * np.pi) - np.pi)
                break
            st = new
        met += want < np.pi
        assert np.isclose(off[i], want, atol=5e-3), (i, off[i], want)       # float noise: batch vs one by one
    assert met >= 2 or p is None   # some rounds do meet (the trained brain; a fresh one, in the public repo
                                   # without runs/, may not: every round's reading is still checked above)


def test_a_2v1_generation_with_a_radio():
    """ep11 10x: with teams the brain plays both attackers; with a radio its word head evolves too; the aim at first
    sight is kept per attacker."""
    rules = replace(cqb.CQBConfig(), flashbangs=(2, 0), magazine=30, health=100.0, react=0.3, time_limit=4.0,
                    team_sizes=(2, 1), share_sight=False, split_entry=True, radio=(4, 0))
    game = cqb.CQBGame(cqb.make_buildings(list(cqb.TRAIN_SEEDS)[:4], rules), rules)
    params = A.add_radio(A.init_params(jax.random.PRNGKey(0), rules.obs_size, 16, -0.5,
                                       A.switches_for(rules, cqb.ASSAULT)), 4)
    cfg = ES.CQBESConfig(pop=4, rounds=2, buildings=4, sigma=0.02, sigma_throw=0.02, aim_bonus=0.25)
    sig = ES._sigmas(cfg, params, A.N_MOVE + A.SWITCHES.index("throw"))
    assert set(sig) == set(ES.EVOLVED) | {"say.weight", "say.bias"}
    groups = [(jnp.asarray([0, 3]), jax.random.split(jax.random.PRNGKey(1), 2), jnp.asarray([False, False]))]
    grad, stats = jax.jit(ES.make_generation(cfg, game, cqb.ASSAULT, [S.SCRIPTS["camper"]], sig))(
        params, jax.random.PRNGKey(2), groups)
    assert set(grad) == set(sig) and all(bool(jnp.isfinite(g).all()) for g in grad.values())
    *_, off, _ = jax.device_get(ES._rollout(game, cqb.ASSAULT, params, S.SCRIPTS["camper"], *groups[0]))
    assert off.shape == (2, 2) and ((off >= 0) & (off <= np.pi)).all()


def test_side_counts_are_what_the_round_did():
    """ep11 10y: the watched counts (crashes, footsteps heard, friendly downs, words) equal the same counted from a
    step-by-step replay of the round."""
    rules = replace(cqb.CQBConfig(), flashbangs=(2, 0), magazine=30, health=100.0, react=0.3, time_limit=10.0,
                    team_sizes=(2, 1), share_sight=False, split_entry=True, radio=(4, 0), bump=15.0, stun=0.5)
    game = cqb.CQBGame(cqb.make_buildings(list(cqb.TRAIN_SEEDS)[:4], rules), rules)
    params = A.add_radio(A.init_params(jax.random.PRNGKey(0), rules.obs_size, 16, -0.5,
                                       A.switches_for(rules, cqb.ASSAULT)), 4)
    params = {**params, "say.bias": jnp.asarray([0.0, 3.0, 0.0, 0.0, 0.0])}           # it says "Alpha" a lot
    params = {**params, "policy.bias": params["policy.bias"].at[0].set(3.0)}           # and charges forward
    n = 8
    bis, keys, drill = jnp.arange(n) % 4, jax.random.split(jax.random.PRNGKey(5), n), jnp.ones(n, bool)   # drills: they meet
    opp = S.SCRIPTS["camper"]
    _, _, _, _, side, _, _ = jax.device_get(ES._rollout(game, cqb.ASSAULT, params, opp, bis, keys, drill))
    step = jax.jit(lambda st, k: game.step(st, jnp.stack([
        cqb.full_action(A.best_guess(params, game.observe(st, 0)[None])[0]),
        cqb.full_action(A.best_guess(params, game.observe(st, 1)[None])[0]),
        cqb.full_action(opp(game, st, 2, k))]))[0])
    for i in range(n):
        st, want = game.init_stacked(keys[i], bis[i]), np.zeros(5, int)
        for k in jax.random.split(jax.random.fold_in(keys[i], 1), game.T):
            if bool(st.done):
                break
            new = step(st, k)
            fell = np.asarray(st.alive[:2]) & ~np.asarray(new.alive[:2])
            want += [1, int(np.asarray(new.bumped[:2]).sum()), int(np.asarray(new.sound_age[2, 0]) == 0),
                     int((fell & np.isin(np.asarray(new.hit_by[:2]), [0, 1])).sum()),
                     int(((np.asarray(new.said_age[:2]) == 0) & (np.asarray(new.said[:2]) > 0)).sum())]
            st = new
        assert (side[i] == want).all(), (i, side[i], want)
    assert side[:, 4].sum() > 0 and side[:, 1].sum() > 0                                  # words and crashes happened
    assert (side[:, 0] < game.T).any()                                                    # and some round ended early
