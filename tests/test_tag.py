"""Tag: fair arenas, solid obstacles, honest lasers and line of sight, catches that count."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ailearns.games.tag import (CHASER, RUNNER, Arenas, TagConfig, TagGame, chaser_naive, make_arena,
                                make_arenas, play_rounds, runner_flee, runner_smart, runner_still)

CFG = TagConfig()
SEEDS = list(range(16)) + [100_000, 100_001]


@pytest.fixture(scope="module")
def game():
    return TagGame(make_arenas(SEEDS, CFG), CFG)


def _box_game(lo, hi):
    """One arena with the given obstacles (the rest of the slots far away)."""
    K = CFG.max_obstacles
    pad = lambda x: np.vstack([np.asarray(x, float).reshape(-1, 2), np.full((K - len(x), 2), -1000.0)])
    xs = np.arange(2.0, CFG.size - 1, 1.0)
    pts = np.stack(np.meshgrid(xs, xs), -1).reshape(-1, 2)
    if len(lo):
        pts = pts[(np.linalg.norm(pts[:, None] - np.clip(pts[:, None], np.asarray(lo, float)[None],
                                                          np.asarray(hi, float)[None]), axis=2) > 1).all(axis=1)]
    spawn = pts[np.random.default_rng(0).choice(len(pts), CFG.spawn_points)]
    f = lambda x: jnp.asarray(x[None], jnp.float32)
    return TagGame(Arenas(f(pad(lo)), f(pad(hi)), f(spawn)), CFG)


def _state(game, chaser, runner, vel=((0, 0), (0, 0))):
    st = game.init(jax.random.PRNGKey(0), 0)
    pos = jnp.asarray([chaser, runner], jnp.float32)
    seen = game.visible(0, pos[0], pos[1])
    return st._replace(pos=pos, vel=jnp.asarray(vel, jnp.float32), mem=pos[::-1], seen=seen)


@pytest.mark.parametrize("seed", SEEDS)
def test_arenas_have_room_everywhere(seed):
    a = make_arena(seed, CFG)
    lo, hi = a["lo"], a["hi"]
    assert 3 <= len(lo) <= 7
    assert (lo >= CFG.min_gap - 1e-9).all() and (hi <= CFG.size - CFG.min_gap + 1e-9).all()
    for i in range(len(lo)):  # every gap is wider than a player
        for j in range(i):
            d = np.maximum(0, np.maximum(lo[j] - hi[i], lo[i] - hi[j]))
            assert np.hypot(*d) >= CFG.min_gap
    closest = np.clip(a["spawn"][:, None], lo[None], hi[None])
    assert (np.linalg.norm(a["spawn"][:, None] - closest, axis=2) > CFG.radius).all()  # starts are free


def test_players_start_apart_and_still(game):
    st = jax.vmap(lambda k: game.init(k))(jax.random.split(jax.random.PRNGKey(3), 256))
    d = np.linalg.norm(np.asarray(st.pos[:, 0] - st.pos[:, 1]), axis=1)
    assert (d >= CFG.spawn_dist).mean() > 0.99
    assert (np.asarray(st.vel) == 0).all() and not np.asarray(st.done).any()


def test_lasers_measure_walls_and_obstacles():
    g = _box_game([[20, 10]], [[22, 30]])  # a wall 2 m wide, from y=10 to 30
    d = np.asarray(g.lasers(0, jnp.asarray([10.0, 20.0])))
    east, north, west, south = d[0], d[CFG.n_rays // 4], d[CFG.n_rays // 2], d[3 * CFG.n_rays // 4]
    assert abs(east - 10) < 1e-3 and abs(west - 10) < 1e-3
    assert abs(north - 20) < 1e-3 and abs(south - 20) < 1e-3
    d = np.asarray(g.lasers(0, jnp.asarray([2.0, 2.0])))
    assert abs(d[0] - 30.0) < 1e-3  # the far wall is out of range: capped


def test_obstacles_block_sight():
    g = _box_game([[20, 10]], [[22, 30]])
    assert not bool(g.visible(0, jnp.asarray([10.0, 20.0]), jnp.asarray([30.0, 20.0])))
    assert bool(g.visible(0, jnp.asarray([10.0, 5.0]), jnp.asarray([30.0, 5.0])))  # passes below it
    assert bool(g.visible(0, jnp.asarray([10.0, 20.0]), jnp.asarray([15.0, 25.0])))  # stops short of it


def test_nobody_goes_through_walls(game):
    """Random flailing for a long time: never inside an obstacle or outside the arena."""
    def run(key):
        st = game.init(key)

        def tick(st, k):
            st, _ = game.step(st, jax.random.uniform(k, (2, 2), minval=-1, maxval=1) * 3)
            return st._replace(done=jnp.asarray(False), t=jnp.asarray(0)), st.pos
        return jax.lax.scan(tick, st, jax.random.split(key, 400))[1]

    paths = np.asarray(jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(5), 64)))
    r = CFG.radius
    assert (paths >= r - 1e-4).all() and (paths <= CFG.size - r + 1e-4).all()
    lo, hi = np.asarray(game.arenas.lo), np.asarray(game.arenas.hi)
    st = jax.vmap(lambda k: game.init(k))(jax.random.split(jax.random.PRNGKey(5), 64))
    for n, a in enumerate(np.asarray(st.arena)):
        p = paths[n].reshape(-1, 1, 2)
        gap = np.linalg.norm(p - np.clip(p, lo[a][None], hi[a][None]), axis=2)
        assert gap.min() > r - 0.05


def test_top_speed(game):
    g = _box_game([], [])
    st = _state(g, (2, 2), (2, 38))
    for _ in range(40):
        st, _ = g.step(st._replace(done=jnp.asarray(False)), jnp.asarray([[1.0, 1.0], [1.0, -1.0]]))
    speed = np.linalg.norm(np.asarray(st.vel), axis=1)
    assert abs(speed[CHASER] - CFG.chaser_speed) < 0.1 and abs(speed[RUNNER] - CFG.runner_speed) < 0.1


def test_catch_counts_even_between_frames():
    g = _box_game([], [])
    # they run past each other at full speed; at no frame are they close, but they cross
    st = _state(g, (20.0, 20.0), (20.6, 21.1), vel=((6, 0), (-5.5, 0)))
    assert not bool(st.caught)  # 1.25 m apart before, 1.23 m after
    st, r = g.step(st, jnp.asarray([[1.0, 0.0], [-1.0, 0.0]]))
    assert bool(st.caught) and bool(st.done) and float(r) > 0.9


def test_timeout_and_rewards():
    g = _box_game([], [])
    st = _state(g, (2.0, 2.0), (38.0, 38.0))
    total, n = 0.0, 0
    while not bool(st.done):
        st, r = g.step(st, jnp.zeros((2, 2)))
        total, n = total + float(r), n + 1
    assert n == round(CFG.time_limit / CFG.dt) and not bool(st.caught)
    assert abs(total - (-CFG.timeout_penalty - CFG.time_penalty * CFG.time_limit)) < 1e-3


def test_memory_only_updates_when_visible():
    g = _box_game([[20, 10]], [[22, 30]])
    st = _state(g, (10.0, 20.0), (30.0, 20.0))  # wall in between
    assert not bool(st.seen)
    st2, _ = g.step(st, jnp.asarray([[0.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(st2.mem[CHASER], st.mem[CHASER]) and int(st2.since) == 1
    obs = np.asarray(g.observe(st2, CHASER))
    assert obs.shape == (CFG.obs_size,) and obs[CFG.n_rays + 4] == 0  # "can't see it"
    assert (obs[CFG.n_rays + 7: CFG.n_rays + 9] == 0).all()           # no velocity for what you can't see


def test_scripted_balance(game):
    """The naive chaser always gets a runner that stands still in the open, and only sometimes
    a fleeing one: room for the AI to do better."""
    open_game = _box_game([], [])
    assert play_rounds(open_game, chaser_naive, runner_still, 32)["catch_rate"] == 1.0
    rate = play_rounds(game, chaser_naive, runner_flee, 256)["catch_rate"]
    assert 0.2 < rate < 0.7
    assert play_rounds(game, chaser_naive, runner_smart, 256)["catch_rate"] < rate  # the smart one is harder


def test_closer_reward_pays_per_metre():
    from dataclasses import replace
    cfg = replace(CFG, closer_reward=0.1)
    g = TagGame(_box_game([], []).arenas, cfg)
    st = _state(g, (10.0, 20.0), (30.0, 20.0), vel=((6, 0), (0, 0)))
    st2, r = g.step(st, jnp.asarray([[1.0, 0.0], [0.0, 0.0]]))
    gained = 20.0 - float(np.linalg.norm(np.asarray(st2.pos[1] - st2.pos[0])))
    assert abs(float(r) - (0.1 * gained - cfg.time_penalty * cfg.dt)) < 1e-4 and gained > 0.5


def test_remember_heading():
    from dataclasses import replace
    cfg = replace(CFG, remember_heading=True)
    g = TagGame(_box_game([[20, 10]], [[22, 30]]).arenas, cfg)
    st = _state(g, (10.0, 5.0), (25.0, 5.0), vel=((0, 0), (0, 4.0)))  # in view, runner heading north
    st, _ = g.step(st, jnp.asarray([[0.0, 0.0], [0.0, 1.0]]))
    assert bool(st.seen)
    st = st._replace(pos=st.pos.at[CHASER].set(jnp.asarray([10.0, 20.0])))  # chaser now behind the wall
    st, _ = g.step(st, jnp.asarray([[0.0, 0.0], [0.0, 1.0]]))
    assert not bool(st.seen)
    obs = np.asarray(g.observe(st, CHASER))
    assert obs[CFG.n_rays + 8] > 0.5  # still knows it was running north
