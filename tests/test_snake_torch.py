"""The torch Snake must play exactly like the numpy reference.

Food is random in both (with different random streams), so after every step
the torch games are given the numpy games' food. Everything else must match:
boards, heads, rewards, endings and every observation view.
"""

import numpy as np
import pytest
import torch

from ailearns.games.snake import SnakeConfig, SnakeGames
from ailearns.games.snake_torch import TorchSnakeGames


def _sync_food(ref: SnakeGames, tg: TorchSnakeGames) -> None:
    tg.food = torch.from_numpy(ref.food.copy())


def _same_state(ref: SnakeGames, tg: TorchSnakeGames) -> None:
    np.testing.assert_array_equal(tg.board.numpy(), ref.board)
    np.testing.assert_array_equal(tg.head.numpy(), ref.head)
    np.testing.assert_array_equal(tg.direction.numpy(), ref.direction)
    np.testing.assert_array_equal(tg.length.numpy(), ref.length)
    np.testing.assert_array_equal(tg.hunger.numpy(), ref.hunger)
    for view in ("board", "ego", "ego2"):
        np.testing.assert_allclose(tg.observe(view).numpy(), ref.observe(view), atol=1e-6, err_msg=view)


@pytest.mark.parametrize("size", [6, 10])
def test_torch_snake_matches_numpy(size):
    n = 64
    cfg = SnakeConfig(size=size, hunger_base=12, hunger_per_length=1)  # short fuse so starving happens too
    ref = SnakeGames(n, cfg, seed=1)
    tg = TorchSnakeGames(n, cfg, seed=2)
    _sync_food(ref, tg)
    _same_state(ref, tg)
    rng = np.random.default_rng(3)
    endings = {"dead": 0, "starved": 0, "won": 0}
    for _ in range(600):
        # mostly straight, so games last long enough to eat and grow
        actions = rng.choice(3, size=n, p=[0.6, 0.2, 0.2])
        r1, d1, w1 = ref.step(actions)
        r2, d2, w2 = tg.step(torch.from_numpy(actions))
        np.testing.assert_array_equal(r2.numpy(), r1)
        np.testing.assert_array_equal(d2.numpy(), d1)
        np.testing.assert_array_equal(w2.numpy(), w1)
        np.testing.assert_array_equal(tg.starved.numpy(), ref.starved)
        endings["starved"] += int(ref.starved.sum())
        endings["won"] += int(w1.sum())
        endings["dead"] += int((d1 & ~w1 & ~ref.starved).sum())
        _sync_food(ref, tg)  # ate -> new food: take the reference's
        # board/head of finished games are only compared after the reset
        ref.reset(d1)
        tg.reset(torch.from_numpy(d1))
        _sync_food(ref, tg)
        _same_state(ref, tg)
    assert endings["dead"] > 0 and endings["starved"] > 0


def test_torch_snake_win():
    """One move from a full 4x4 board: eating the last apple must be a win in both."""
    from ailearns.games.snake import LEFT, STRAIGHT
    cfg = SnakeConfig(size=4)
    ref, tg = SnakeGames(1, cfg, seed=0), TorchSnakeGames(1, cfg, seed=0)
    path = [(r, c if r % 2 == 0 else 3 - c) for r in range(4) for c in range(4)]  # serpentine
    board = np.zeros((4, 4), dtype=np.int32)
    for k, (r, c) in enumerate(path[:15]):  # tail (1) at path[0], head (15) at path[14]
        board[r, c] = k + 1
    ref.board[0], ref.head[0], ref.direction[0], ref.length[0], ref.food[0] = board, path[14], LEFT, 15, path[15]
    tg.board[0] = torch.from_numpy(board)
    tg.head[0], tg.direction[0], tg.length[0] = torch.tensor(path[14]), LEFT, 15
    _sync_food(ref, tg)
    _same_state(ref, tg)
    r1, d1, w1 = ref.step(np.array([STRAIGHT]))
    r2, d2, w2 = tg.step(torch.tensor([STRAIGHT]))
    assert w1[0] and d1[0] and r1[0] == 11.0
    np.testing.assert_array_equal(r2.numpy(), r1)
    np.testing.assert_array_equal(d2.numpy(), d1)
    np.testing.assert_array_equal(w2.numpy(), w1)
    np.testing.assert_array_equal(tg.board.numpy(), ref.board)
