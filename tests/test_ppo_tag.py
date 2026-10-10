"""Tag self-play plumbing: frozen brains (and pools of them) as opponents, and the runner learning."""

import json

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo_car import forward, init_params
from ailearns.agents.ppo_tag import TagPPOConfig, pool_player, train
from ailearns.games.tag import RUNNER, TagConfig, TagGame, make_arenas

CFG = TagConfig()


def test_pool_member_by_slot():
    """Round i plays member i % K, with that member's best guess."""
    game = TagGame(make_arenas([0, 1], CFG), CFG)
    members = [init_params(jax.random.key(i), CFG.obs_size, 32, -0.5) for i in range(3)]
    pool = jax.tree.map(lambda *xs: jnp.stack(xs), *members)
    play = pool_player(pool, RUNNER)
    st = game.init(jax.random.PRNGKey(0), 0)
    obs = game.observe(st, RUNNER)[None]
    for slot in range(5):
        want = jnp.clip(forward(members[slot % 3], obs)[0][0], -1, 1)
        np.testing.assert_allclose(play(game, st, None, slot), want, atol=1e-6)


def test_runner_trains_against_a_frozen_brain(tmp_path):
    """A tiny chaser run, then a runner run against it: the runner's rewards are the chaser's, flipped."""
    small = dict(n_envs=64, n_steps=16, minibatch=256, hidden=32, arenas=8, total_steps=2 * 64 * 16)
    train(TagPPOConfig(**small), tmp_path / "chaser")
    train(TagPPOConfig(**small, side="runner", opponent=f"{tmp_path / 'chaser'}:2,{tmp_path / 'chaser'}:0"),
          tmp_path / "runner")
    lines = [json.loads(x) for x in open(tmp_path / "runner" / "metrics.jsonl")]
    assert [x["update"] for x in lines] == [1, 2]
    assert json.loads((tmp_path / "runner" / "config.json").read_text())["side"] == "runner"
