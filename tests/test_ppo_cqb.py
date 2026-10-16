"""CQB PPO: the brain's two kinds of output, and a tiny run end to end (train, resume, evaluate)."""

import json

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo_cqb import (CQBPPOConfig, best_guess, evaluate, forward, init_params, log_prob, sample,
                                     to_action, train)
from ailearns.games.cqb import CQBConfig

CFG = CQBConfig()


def test_switches_are_coins_with_a_learned_bias():
    params = init_params(jax.random.key(0), CFG.obs_size, 32, -0.5)
    params["policy.bias"] = params["policy.bias"].at[3:].set(jnp.array([2.0, -1.0, -2.0]))  # fire likely, the others not
    obs = jax.random.normal(jax.random.key(1), (4000, CFG.obs_size)) * 0.01
    mean, logits, log_std, _ = forward(params, obs)
    moves, switches = sample(jax.random.key(2), mean, logits, log_std)
    p = np.asarray(switches.mean(0))
    np.testing.assert_allclose(p, 1 / (1 + np.exp(-np.array([2.0, -1.0, -2.0]))), atol=0.03)
    # log prob = Gaussian moves + Bernoulli switches, worked out by hand for one sample
    m, l, s, a, b = map(np.asarray, (mean[0], logits[0], log_std, moves[0], switches[0]))
    want = (-0.5 * ((a - m) / np.exp(s)) ** 2 - s - 0.5 * np.log(2 * np.pi)).sum()
    want += sum(np.log(1 / (1 + np.exp(-x))) if on else np.log(1 / (1 + np.exp(x))) for x, on in zip(l, b))
    np.testing.assert_allclose(float(log_prob(mean, logits, log_std, moves, switches)[0]), want, rtol=1e-4)
    act = np.asarray(best_guess(params, obs[:1]))[0]
    assert list(act[3:]) == [1.0, -1.0, -1.0]                    # best guess: on only when the chance is over half
    assert list(np.asarray(to_action(moves[:1], switches[:1]))[0, 3:]) == list(2 * np.asarray(switches[0]) - 1)


def test_tiny_run_trains_resumes_and_evaluates(tmp_path):
    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=32, buildings=8)
    run = tmp_path / "cqb"
    train(CQBPPOConfig(**small, total_steps=2 * 32 * 16), run)
    train(CQBPPOConfig(**small, total_steps=3 * 32 * 16), run, resume=True)
    lines = [json.loads(x) for x in open(run / "metrics.jsonl")]
    assert [x["update"] for x in lines] == [1, 2, 3]
    assert {"win_rate", "met_rate", "cells_seen", "p_fire"} <= set(lines[0])
    assert (run / "checkpoints" / "u00003.pt").exists()
    r = evaluate(str(run), "last", first=0, n=4)
    assert r["update"] == 3 and 0.0 <= r["win_rate"] <= 1.0 and r["rounds"] == 4


def test_time_cost_reaches_the_rewards(tmp_path):
    # same seed, same first rollout: only the cost per second differs (0.1 s per decision)
    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=32, buildings=8, total_steps=32 * 16)
    train(CQBPPOConfig(**small), tmp_path / "free")
    train(CQBPPOConfig(**small, time_cost=1.0), tmp_path / "paid")
    free, paid = (json.loads(open(tmp_path / r / "metrics.jsonl").readline()) for r in ("free", "paid"))
    assert -0.1 <= paid["reward_per_step"] - free["reward_per_step"] < -0.09   # rounds that ended don't pay
    assert json.loads((tmp_path / "paid" / "config.json").read_text())["cqb"]["time_cost"] == 1.0


def test_init_from_starts_with_that_brain(tmp_path):
    import torch
    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=32, buildings=8, total_steps=32 * 16)
    train(CQBPPOConfig(**small), tmp_path / "a")
    train(CQBPPOConfig(**small, init_from=f"{tmp_path / 'a'}:1", reset_std=True, init_log_std=-1.0), tmp_path / "b")
    a = torch.load(tmp_path / "a" / "checkpoints" / "u00001.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "checkpoints" / "u00000.pt", weights_only=False)["model"]
    for k in a:
        if k == "log_std":
            assert torch.allclose(b[k], torch.full_like(a[k], -1.0))   # fresh move noise
        else:
            assert torch.equal(a[k], b[k]), k


def test_rounds_start_staggered():
    from ailearns.agents.ppo_cqb import start_rounds
    from ailearns.games.cqb import TRAIN_SEEDS, CQBGame, make_buildings
    game = CQBGame(make_buildings(list(TRAIN_SEEDS)[:4], CFG), CFG)
    t = np.asarray(start_rounds(game, jax.random.key(0), 256).t)
    assert t.min() >= 0 and t.max() < game.T
    assert len(np.unique(t)) > 150 and abs(t.mean() - game.T / 2) < game.T / 10   # spread over the whole clock


def test_pool_slot_i_plays_member_i_mod_k():
    from ailearns.agents.ppo_cqb import pool_player
    from ailearns.games import cqb_scripts as S
    from ailearns.games.cqb import DEFENSE, TRAIN_SEEDS, CQBGame, make_buildings
    game = CQBGame(make_buildings(list(TRAIN_SEEDS)[:4], CFG), CFG)
    pool = pool_player("dummy,wanderer", DEFENSE)
    keys = jax.random.split(jax.random.key(0), 6)
    st = jax.vmap(game.init)(keys)
    for _ in range(30):  # let the wanderer get moving, so the two members act differently
        acts = jax.vmap(lambda s, k: jnp.stack([S.SCRIPTS["walker"](game, s, 0, k), S.wanderer(game, s, 1, k)]))(st, keys)
        st = jax.vmap(lambda s, a: game.step(s, a)[0])(st, acts)
    got = jax.vmap(lambda s, k, i: pool(game, s, DEFENSE, k, i))(st, keys, jnp.arange(6))
    want_dummy = jax.vmap(lambda s, k: S.dummy(game, s, DEFENSE, k))(st, keys)
    want_wander = jax.vmap(lambda s, k: S.wanderer(game, s, DEFENSE, k))(st, keys)
    want = jnp.where((jnp.arange(6) % 2 == 0)[:, None], want_dummy, want_wander)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert not np.array_equal(np.asarray(want_dummy), np.asarray(want_wander))   # the test can tell them apart
    # pool_actions (training's version: each member works only on its own rounds) gives exactly the same
    from ailearns.agents.ppo_cqb import pool_actions, pool_members
    fast = pool_actions(pool_members("dummy,wanderer", DEFENSE), game, st, DEFENSE, keys)
    np.testing.assert_array_equal(np.asarray(fast), np.asarray(got))


def test_widened_brain_plays_as_before_and_learns_to_throw(tmp_path):
    """A brain from before flashbangs, widened for a game with them: the same moves and switches from the same
    first 107 numbers whatever the 8 new ones say, a throw switch that starts mostly off; and a tiny run with
    flashbangs starts from it, logs p_throw, saves the wide brain, and evaluates in the flashbang game."""
    import torch
    from dataclasses import replace
    from ailearns.agents.ppo_cqb import THROW_BIAS, load, widen
    bang = replace(CFG, flashbangs=(2, 0))
    old = init_params(jax.random.key(0), CFG.obs_size, 32, -0.5)
    wide = widen(old, bang.obs_size, 4)
    obs = jax.random.normal(jax.random.key(1), (64, CFG.obs_size))
    extra = jax.random.normal(jax.random.key(2), (64, bang.obs_size - CFG.obs_size))
    m0, l0, _, v0 = forward(old, obs)
    m1, l1, _, v1 = forward(wide, jnp.concatenate([obs, extra], 1))
    np.testing.assert_allclose(m1, m0, atol=1e-5)
    np.testing.assert_allclose(l1[:, :3], l0, atol=1e-5)
    np.testing.assert_allclose(v1, v0, atol=1e-5)
    np.testing.assert_allclose(l1[:, 3], THROW_BIAS, atol=1e-5)
    assert (np.asarray(best_guess(wide, jnp.concatenate([obs, extra], 1)))[:, 6] == -1).all()

    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=32, buildings=8, total_steps=2 * 32 * 16)
    train(CQBPPOConfig(**small), tmp_path / "old")
    train(CQBPPOConfig(**small, init_from=f"{tmp_path / 'old'}:2", flashbangs=(2, 0)), tmp_path / "new")
    lines = [json.loads(x) for x in open(tmp_path / "new" / "metrics.jsonl")]
    assert "p_throw" in lines[0] and 0 < lines[0]["p_throw"] < 0.1
    params, ck = load(tmp_path / "new" / "checkpoints" / "u00002.pt")
    assert params["body.0.weight"].shape[1] == bang.obs_size and params["policy.weight"].shape[0] == 7
    assert tuple(ck["cqb"]["flashbangs"]) == (2, 0)
    a = torch.load(tmp_path / "old" / "checkpoints" / "u00002.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "new" / "checkpoints" / "u00000.pt", weights_only=False)["model"]
    assert torch.equal(b["body.0.weight"][:, :CFG.obs_size], a["body.0.weight"])
    assert torch.equal(b["body.0.weight"][:, CFG.obs_size:], torch.zeros(32, bang.obs_size - CFG.obs_size))
    r = evaluate(str(tmp_path / "new"), "last", first=0, n=4)
    assert r["rounds"] == 4
    # an old Defense brain as the opponent of a flashbang run (it reads its first 107)
    train(CQBPPOConfig(**small, flashbangs=(2, 0), opponent=f"camper,{tmp_path / 'old'}:2", side="assault"),
          tmp_path / "pool")


def test_entry_drills(tmp_path):
    """--stacked: the first share of each device's slots start (and restart) as entry drills, logged apart."""
    from ailearns.agents.ppo_cqb import start_rounds
    from ailearns.games.cqb import ASSAULT, TRAIN_SEEDS, CQBGame, make_buildings
    game = CQBGame(make_buildings(list(TRAIN_SEEDS)[:8], CFG), CFG)
    env = start_rounds(game, jax.random.key(0), 16, n_local=8, n_drill=4)
    normal = jax.vmap(game.init)(jax.random.split(jax.random.split(jax.random.key(0))[0], 16))
    same = np.asarray((env.pos[:, ASSAULT] == normal.pos[:, ASSAULT]).all(1))
    assert list(same) == [False] * 4 + [True] * 4 + [False] * 4 + [True] * 4      # per device: drills first
    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=32, buildings=8, total_steps=2 * 32 * 16)
    train(CQBPPOConfig(**small, stacked=0.5, flashbangs=(2, 0)), tmp_path / "drill")
    lines = [json.loads(x) for x in open(tmp_path / "drill" / "metrics.jsonl")]
    assert "drill_win_rate" in lines[-1] and "full_win_rate" in lines[-1]
    train(CQBPPOConfig(**small, stacked=1.0, flashbangs=(2, 0)), tmp_path / "drills-only")
    lines = [json.loads(x) for x in open(tmp_path / "drills-only" / "metrics.jsonl")]
    assert lines[-1]["full_win_rate"] is None and lines[-1]["drill_win_rate"] is not None


def test_dead_throw_is_left_out():
    """With no flashbang in hand (or one in the air), the throw coin is left out of log prob and entropy; and
    live_switches says when that is."""
    from ailearns.agents.ppo_cqb import entropy, live_switches
    from ailearns.games.cqb import ASSAULT, TRAIN_SEEDS, CQBGame, make_buildings
    from dataclasses import replace
    params = init_params(jax.random.key(0), CFG.obs_size, 32, -0.5, n_switch=4)
    obs = jax.random.normal(jax.random.key(1), (8, CFG.obs_size))
    mean, logits, log_std, _ = forward(params, obs)
    moves, switches = sample(jax.random.key(2), mean, logits, log_std)
    full = jnp.ones((8, 4))
    dead = full.at[:, 3].set(0.0)
    lp, ld = log_prob(mean, logits, log_std, moves, switches, full), log_prob(mean, logits, log_std, moves, switches, dead)
    t = switches[:, 3] * jax.nn.log_sigmoid(logits[:, 3]) + (1 - switches[:, 3]) * jax.nn.log_sigmoid(-logits[:, 3])
    np.testing.assert_allclose(lp - ld, t, rtol=1e-5)
    np.testing.assert_allclose(lp, log_prob(mean, logits, log_std, moves, switches), rtol=1e-6)
    assert float(entropy(logits, log_std, dead)) < float(entropy(logits, log_std, full))
    bang = replace(CFG, flashbangs=(2, 0))
    game = CQBGame(make_buildings(list(TRAIN_SEEDS)[:2], bang), bang)
    st = jax.vmap(game.init)(jax.random.split(jax.random.key(3), 3))
    st = st._replace(bangs=st.bangs.at[1, ASSAULT].set(0), bang_timer=st.bang_timer.at[2, ASSAULT].set(5))
    assert np.asarray(live_switches(game, st, ASSAULT, 4))[:, 3].tolist() == [1.0, 0.0, 0.0]
    assert np.asarray(live_switches(game, st, ASSAULT, 3)).min() == 1.0


def test_memory_brain(tmp_path):
    """add_memory: plays exactly as before until memory.weight moves; replay matches step-by-step play (memory wiped
    where a round ended); play_rounds carries it; a tiny run with --memory trains, resumes, evaluates and renders."""
    from ailearns.agents.ppo_cqb import add_memory, brain, forward_mem, has_memory, load, replay
    from ailearns.games import cqb_scripts as S
    from ailearns.games.cqb import TEST_SEEDS, CQBGame, make_buildings
    old = init_params(jax.random.key(0), CFG.obs_size, 32, -0.5)
    mem = add_memory(old, 8, jax.random.key(1))
    obs = jax.random.normal(jax.random.key(2), (5, CFG.obs_size))
    h = jax.random.normal(jax.random.key(3), (5, 8))
    a, b = forward(old, obs), forward_mem(mem, obs, h)
    for x, y in zip(a, b[:4]):
        np.testing.assert_allclose(x, y, atol=1e-6)
    assert not np.allclose(b[4], h)                                          # it does remember something
    # replay == one decision at a time
    mem["memory.weight"] = jax.random.normal(jax.random.key(4), mem["memory.weight"].shape)
    T, E = 6, 3
    seq = jax.random.normal(jax.random.key(5), (T, E, CFG.obs_size))
    done = jnp.zeros((T, E), bool).at[2, 1].set(True)
    h = jnp.zeros((E, 8))
    want = []
    for t in range(T):
        m, lg, _, v, h = forward_mem(mem, seq[t], h)
        h = jnp.where(done[t][:, None], 0.0, h)
        want.append(v)
    got = replay(mem, seq, done, jnp.zeros((E, 8)))[3].reshape(T, E)
    np.testing.assert_allclose(got, jnp.stack(want), atol=1e-5)
    # pieces of 2 decisions, each replayed from the memory it had in play (training's way), give the same
    from ailearns.agents.ppo_cqb import to_chunks
    h, hs = jnp.zeros((E, 8)), []
    for t in range(T):
        hs.append(h)
        h = jnp.where(done[t][:, None], 0.0, forward_mem(mem, seq[t], h)[4])
    hs = jnp.stack(hs)                                                       # (T, E, 8): the memory each decision had
    pieces = replay(mem, to_chunks(seq, 2), to_chunks(done, 2), hs[::2].reshape(-1, 8))[3].reshape(2, T // 2 * E)
    np.testing.assert_allclose(pieces, to_chunks(jnp.stack(want), 2), atol=1e-5)
    # play_rounds: with memory.weight at zero, the same rounds as the brain without memory
    game = CQBGame(make_buildings(list(TEST_SEEDS)[:4], CFG), CFG)
    plain = S.play_rounds(game, brain(old, 0), "camper", np.arange(4), seed=1)
    zero = S.play_rounds(game, brain(add_memory(old, 8, jax.random.key(1)), 0), "camper", np.arange(4), seed=1)
    assert np.array_equal(plain["t"], zero["t"]) and np.allclose(plain["pos"], zero["pos"], atol=1e-5)
    # training
    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=32, buildings=8, mem_chunk=4)
    train(CQBPPOConfig(**small, total_steps=2 * 32 * 16), tmp_path / "plain")
    train(CQBPPOConfig(**small, total_steps=2 * 32 * 16, memory=8, init_from=f"{tmp_path / 'plain'}:2"),
          tmp_path / "mem")
    train(CQBPPOConfig(**small, total_steps=3 * 32 * 16, memory=8), tmp_path / "mem", resume=True)
    params, _ = load(tmp_path / "mem" / "checkpoints" / "u00003.pt")
    assert has_memory(params) and params["gru.wh"].shape == (24, 8)
    lines = [json.loads(x) for x in open(tmp_path / "mem" / "metrics.jsonl")]
    assert [x["update"] for x in lines] == [1, 2, 3] and lines[0]["clipfrac"] < 0.5
    r = evaluate(str(tmp_path / "mem"), "last", first=0, n=4)
    assert r["rounds"] == 4
    from ailearns import record_cqb as R
    tr = R.rollout(f"{tmp_path / 'mem'}:last", "camper", 100_000)
    assert tr["steps"] > 0


def test_plateau_stops_a_run(tmp_path):
    """--plateau: a run with nothing to learn (lr 0) stops once 25-update averages stop setting new bests."""
    small = dict(n_envs=16, n_steps=8, minibatch=64, hidden=16, buildings=4, total_steps=200 * 16 * 8, lr=0.0,
                 opponent="dummy", plateau=10)
    train(CQBPPOConfig(**small), tmp_path / "flat")
    lines = [json.loads(x) for x in open(tmp_path / "flat" / "metrics.jsonl")]
    assert 35 <= len(lines) < 200
    assert (tmp_path / "flat" / "checkpoints" / f"u{len(lines):05d}.pt").exists()


def test_report_chooses_on_0_63_and_matches_evaluate(tmp_path):
    """cqb_eval.report: chooses among every n-th checkpoint and the last on test 0-63, then reports on 64-255 with
    the same numbers evaluate() gives for that checkpoint, opponent and seed."""
    from ailearns.agents.cqb_eval import report
    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=16, buildings=8, total_steps=4 * 32 * 16,
                 opponent="camper", early_checkpoints=(0, 1, 2, 3, 4))
    train(CQBPPOConfig(**small), tmp_path / "r")
    out = report(tmp_path / "r", against="camper,dummy", every=2)
    assert sorted(out["choose"]["mean_win_by_update"]) == [2, 4]
    assert out["chosen_update"] in (2, 4)
    ev = evaluate(str(tmp_path / "r"), str(out["chosen_update"]), first=64, n=192, seed=5000, opponent="dummy")
    assert abs(out["report"]["table"]["dummy"]["won"] - ev["win_rate"]) < 1e-9
    assert json.loads((tmp_path / "r" / "eval.json").read_text())["chosen_update"] == out["chosen_update"]


def test_memory_learns_at_its_own_rate(tmp_path):
    """mem_lr scales only the memory's own weights' steps: 0 keeps them as they started, the rest still learns."""
    import torch
    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=16, buildings=8, total_steps=2 * 32 * 16, memory=8,
                 mem_chunk=4)
    train(CQBPPOConfig(**small, mem_lr=0.0), tmp_path / "frozen")
    a = torch.load(tmp_path / "frozen" / "checkpoints" / "u00000.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "frozen" / "checkpoints" / "u00002.pt", weights_only=False)["model"]
    for k in ("gru.wi", "gru.wh", "gru.bi", "gru.bh", "memory.weight"):
        assert torch.equal(a[k], b[k]), k
    assert not torch.equal(a["body.2.weight"], b["body.2.weight"])
    train(CQBPPOConfig(**small, mem_lr=1.0), tmp_path / "free")
    c = torch.load(tmp_path / "free" / "checkpoints" / "u00002.pt", weights_only=False)["model"]
    assert not torch.equal(a["memory.weight"], c["memory.weight"])


def test_vf_warmup_holds_the_policy(tmp_path):
    """ep11 10u: for the first vf_warmup updates only the value head learns; every other weight (and the move noise)
    is exactly as it started; after that the policy learns too."""
    import torch
    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=32, buildings=8)
    run = tmp_path / "warm"
    train(CQBPPOConfig(**small, total_steps=3 * 32 * 16, vf_warmup=2), run)
    ck = [torch.load(run / "checkpoints" / f"u{u:05d}.pt", weights_only=False)["model"] for u in (0, 1, 2, 3)]
    for k in ck[0]:
        if k.startswith("value."):
            assert not torch.equal(ck[0][k], ck[2][k]), k                  # the value head learned
        else:
            assert torch.equal(ck[0][k], ck[1][k]) and torch.equal(ck[0][k], ck[2][k]), k   # held
    assert not torch.equal(ck[2]["policy.weight"], ck[3]["policy.weight"])  # then the policy moves


def test_transplant_moves_each_part_of_the_observation():
    """ep11 10x: a 1v1 brain (route + feed at the end) moved into a 2v1 game with a radio (team numbers, rifle, stun
    and radio before route + feed): its first layer computes exactly what it did on the numbers it knew, wherever
    they now sit; the new parts read zero weights."""
    from dataclasses import replace
    from ailearns.agents import ppo_cqb as A
    from ailearns.games import cqb
    one = replace(CQBConfig(), flashbangs=(2, 0), magazine=30, health=100.0, react=0.3, route=True, feed=1,
                  **cqb.HOUSES["big"])
    two = replace(one, team_sizes=(2, 1), bump=15.0, ready=0.3, stun=0.5, share_sight=False, split_entry=True,
                  radio=(4, 0))
    p = init_params(jax.random.key(0), one.obs_size, 32, -0.5)
    q = A.transplant(p, one, two)
    assert q["body.0.weight"].shape == (32, two.obs_size)
    obs2 = jax.random.normal(jax.random.key(1), (16, two.obs_size))
    blocks2 = dict(); at = 0
    for name, n in A.obs_blocks(two):
        blocks2[name] = obs2[:, at:at + n]; at += n
    obs1 = jnp.concatenate([blocks2[name] for name, _ in A.obs_blocks(one)], 1)   # the same numbers in 1v1's order
    np.testing.assert_allclose(np.asarray(obs2 @ q["body.0.weight"].T), np.asarray(obs1 @ p["body.0.weight"].T),
                               rtol=1e-5, atol=1e-5)
    at = 0
    for name, n in A.obs_blocks(two):
        if name in ("teams", "ready", "stun", "radio"):
            assert not np.asarray(q["body.0.weight"][:, at:at + n]).any(), name
        at += n
    for c in (one, two, CQBConfig(), replace(two, radio=(0, 0), feed=0)):                 # the parts add up
        assert sum(n for _, n in A.obs_blocks(c)) == c.obs_size


def test_transplant_equals_widen_when_new_parts_come_last():
    """ep11 10zb: every earlier init_from only added parts at the end (e.g. a 1v1 brain into 2v1); there the transplant
    gives exactly what widening did, so those runs start the same."""
    from dataclasses import replace
    from ailearns.agents import ppo_cqb as A
    one = replace(CQBConfig(), flashbangs=(2, 0), magazine=30, health=100.0, react=0.3)
    two = replace(one, team_sizes=(2, 1), bump=15.0, ready=0.3, stun=0.5, radio=(4, 0))
    p = init_params(jax.random.key(0), one.obs_size, 32, -0.5)
    a = A.transplant(p, one, two)["body.0.weight"]
    b = A.widen(p, two.obs_size, A.switches_for(two, 0))["body.0.weight"]
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_asymmetric_critic(tmp_path):
    """ep11 10zh critic_priv: a separate value network reads the observation plus where the enemy really is; it
    learns in a tiny run, its value moves when only the privileged numbers change, and the brain's play (moves and
    switches) never depends on them."""
    import torch
    from dataclasses import replace
    from ailearns.agents import ppo_cqb as A
    from ailearns.games import cqb
    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=32, buildings=8)
    run = tmp_path / "crit"
    train(CQBPPOConfig(**small, total_steps=2 * 32 * 16, critic_priv=True), run)
    ck = [torch.load(run / "checkpoints" / f"u{u:05d}.pt", weights_only=False)["model"] for u in (0, 2)]
    assert "critic.0.weight" in ck[0] and not torch.equal(ck[0]["critic.0.weight"], ck[1]["critic.0.weight"])
    params, _ = A.load(run / "checkpoints" / "u00002.pt")
    obs = jax.random.normal(jax.random.key(1), (8, CQBConfig().obs_size))
    p1, p2 = jnp.zeros((8, A.PRIV)), jax.random.normal(jax.random.key(2), (8, A.PRIV))
    assert not np.allclose(np.asarray(A.critic_value(params, obs, p1)), np.asarray(A.critic_value(params, obs, p2)))
    m, l, _, _ = forward(params, obs)                                                  # play: obs only
    assert m.shape == (8, 3) and np.isfinite(np.asarray(l)).all()
    g = cqb.CQBGame(cqb.make_buildings(list(cqb.TEST_SEEDS)[:2], CQBConfig()), CQBConfig())
    st = g.init(jax.random.key(0))
    assert np.allclose(np.asarray(g.privileged(st, 0)), np.asarray(
        cqb.CQBGame(g.b, replace(CQBConfig(), feed=1)).observe(st, 0)[-cqb.FEED_OBS:]))   # = what the feed shows


def test_guess_head(tmp_path):
    """ep12 guess_coef: a side head on the brain's body guesses the PRIV numbers (where the nearest enemy really is);
    it's scored against the truth in training (its error is logged and its weights move), the body learns from it
    too, and play never reads it: off (0) the brain has no guess head at all."""
    import json
    import torch
    from ailearns.agents import ppo_cqb as A
    small = dict(n_envs=32, n_steps=16, minibatch=256, hidden=32, buildings=8)
    run = tmp_path / "guess"
    train(CQBPPOConfig(**small, total_steps=2 * 32 * 16, guess_coef=1.0), run)
    ck = [torch.load(run / "checkpoints" / f"u{u:05d}.pt", weights_only=False)["model"] for u in (0, 2)]
    assert "guess.weight" in ck[0] and not torch.equal(ck[0]["guess.weight"], ck[1]["guess.weight"])
    rows = [json.loads(line) for line in open(run / "metrics.jsonl")]
    assert all(np.isfinite(r["guess_err"]) for r in rows)
    params, _ = A.load(run / "checkpoints" / "u00002.pt")
    obs = jax.random.normal(jax.random.key(1), (8, CQBConfig().obs_size))
    assert A.guess(params, obs).shape == (8, A.PRIV)
    without = {k: v for k, v in params.items() if not k.startswith("guess.")}
    for a, b in zip(forward(params, obs), forward(without, obs)):                    # play: the same without it
        assert np.allclose(np.asarray(a), np.asarray(b))
    off = tmp_path / "off"
    train(CQBPPOConfig(**small, total_steps=32 * 16), off)
    assert "guess.weight" not in torch.load(off / "checkpoints" / "u00001.pt", weights_only=False)["model"]


def test_fit_adds_team_slots_and_plays_as_before():
    """ep12 fit/transplant: a 2v2 brain in a 3v2 game reads its old numbers with the same weights (the enemy and
    teammate slots it knew, which one of its team it is) and zero weights on the new slots, so with the new slots
    empty it decides exactly as before; and a defender brain already has a throw switch (a never-live placeholder since
    magazines), so flashbangs for the defence need no widening."""
    from dataclasses import replace
    from ailearns.agents import ppo_cqb as A
    from ailearns.games import cqb
    small = replace(CQBConfig(), flashbangs=(2, 0), magazine=30, health=100.0, react=0.3)
    two = replace(small, team_sizes=(2, 2))
    three = replace(small, team_sizes=(3, 2))
    params = A.init_params(jax.random.key(0), two.obs_size, 32, -0.5, A.switches_for(two, cqb.ASSAULT))
    ck = {"cqb": {k: list(v) if isinstance(v, tuple) else v for k, v in two.__dict__.items()}}
    p3 = A.fit(params, ck, three, cqb.ASSAULT)
    assert p3["body.0.weight"].shape[1] == three.obs_size
    # the same situation in both layouts: the 2v2 numbers, and the 3v2 numbers with the extra slots empty
    obs2 = jax.random.normal(jax.random.key(1), (4, two.obs_size))
    blocks2, blocks3 = dict(A.obs_blocks(two)), dict(A.obs_blocks(three))
    parts, at = [], 0
    for name, n in A.obs_blocks(two):
        x = obs2[:, at:at + n]
        if name == "teams":
            E, M = cqb.ENEMY_OBS, cqb.MATE_OBS
            x = jnp.concatenate([x[:, :E], jnp.zeros((4, E)), x[:, E:E + M], jnp.zeros((4, M)), x[:, -1:]], 1)
            assert x.shape[1] == blocks3["teams"]
        parts.append(x)
        at += n
    obs3 = jnp.concatenate(parts, 1)
    for a, b in zip(forward(params, obs2), forward(p3, obs3)):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-5)
    flash = replace(two, flashbangs=(2, 2))
    d = A.init_params(jax.random.key(2), two.obs_size, 32, -0.5, A.switches_for(two, cqb.DEFENSE))
    dw = A.fit(d, ck, flash, cqb.DEFENSE)
    assert A.switches_for(flash, cqb.DEFENSE) == A.switches_for(two, cqb.DEFENSE)
    assert dw["policy.weight"].shape == d["policy.weight"].shape                # the placeholder becomes live
