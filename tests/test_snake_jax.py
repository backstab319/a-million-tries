"""The JAX Snake must play exactly like the numpy reference (food synced, as in test_snake_torch)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ailearns.games.snake import LEFT, STRAIGHT, SnakeConfig, SnakeGames  # noqa: E402
from ailearns.games.snake_jax import JaxSnake  # noqa: E402


def _same(ref: SnakeGames, st) -> None:
    np.testing.assert_array_equal(np.asarray(st.board), ref.board)
    np.testing.assert_array_equal(np.asarray(st.head), ref.head)
    np.testing.assert_array_equal(np.asarray(st.direction), ref.direction)
    np.testing.assert_array_equal(np.asarray(st.length), ref.length)
    np.testing.assert_array_equal(np.asarray(st.hunger), ref.hunger)


@pytest.mark.parametrize("size", [6, 10])
def test_jax_snake_matches_numpy(size):
    n = 64
    cfg = SnakeConfig(size=size, hunger_base=12, hunger_per_length=1)
    ref, game = SnakeGames(n, cfg, seed=1), JaxSnake(cfg)
    init, step = jax.jit(jax.vmap(game.init)), jax.jit(jax.vmap(game.step))
    reset = jax.jit(jax.vmap(game.reset_if))
    observe = {v: jax.jit(jax.vmap(lambda s, v=v: game.observe(s, v))) for v in ("board", "ego", "ego2")}
    key = jax.random.key(0)
    st = init(jax.random.split(key, n))
    st = st._replace(food=jnp.asarray(ref.food, jnp.int32))
    rng = np.random.default_rng(3)
    ended = {"dead": 0, "starved": 0}
    for t in range(600):
        a = rng.choice(3, size=n, p=[0.6, 0.2, 0.2])
        r1, d1, w1 = ref.step(a)
        st, r2, d2, w2 = step(st, jnp.asarray(a), jax.random.split(jax.random.fold_in(key, t), n))
        np.testing.assert_array_equal(np.asarray(r2), r1)
        np.testing.assert_array_equal(np.asarray(d2), d1)
        np.testing.assert_array_equal(np.asarray(w2), w1)
        np.testing.assert_array_equal(np.asarray(st.starved), ref.starved)
        ended["starved"] += int(ref.starved.sum())
        ended["dead"] += int((d1 & ~w1 & ~ref.starved).sum())
        ref.reset(d1)
        st = reset(st, jnp.asarray(d1), jax.random.split(jax.random.fold_in(key, 10_000 + t), n))
        st = st._replace(food=jnp.asarray(ref.food, jnp.int32))
        _same(ref, st)
        for v, f in observe.items():
            np.testing.assert_allclose(np.asarray(f(st)), ref.observe(v), atol=1e-6, err_msg=v)
    assert ended["dead"] > 0 and ended["starved"] > 0


def test_jax_snake_win():
    """One move from a full 4x4 board: eating the last apple is a win (+11) in both."""
    cfg = SnakeConfig(size=4)
    ref, game = SnakeGames(1, cfg, seed=0), JaxSnake(cfg)
    path = [(r, c if r % 2 == 0 else 3 - c) for r in range(4) for c in range(4)]
    board = np.zeros((4, 4), np.int32)
    for k, (r, c) in enumerate(path[:15]):
        board[r, c] = k + 1
    ref.board[0], ref.head[0], ref.direction[0], ref.length[0], ref.food[0] = board, path[14], LEFT, 15, path[15]
    st = game.init(jax.random.key(0))._replace(board=jnp.asarray(board), head=jnp.asarray(path[14], jnp.int32),
                                               direction=jnp.int32(LEFT), length=jnp.int32(15),
                                               food=jnp.asarray(path[15], jnp.int32))
    r1, d1, w1 = ref.step(np.array([STRAIGHT]))
    st, r2, d2, w2 = game.step(st, STRAIGHT, jax.random.key(1))
    assert w1[0] and bool(w2) and float(r2) == r1[0] == 11.0 and bool(d2)
    np.testing.assert_array_equal(np.asarray(st.board), ref.board[0])
