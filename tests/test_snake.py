import numpy as np

from ailearns.games.snake import (
    STRAIGHT, TURN_LEFT, TURN_RIGHT, SnakeConfig, SnakeGames,
)


def test_starts_centered_moving_right():
    g = SnakeGames(1, SnakeConfig(size=10), seed=0)
    assert tuple(g.head[0]) == (5, 5)
    assert (g.board[0] > 0).sum() == 3
    assert g.board[0, 5, 5] == 3 and g.board[0, 5, 3] == 1


def test_moving_keeps_length_and_vacates_tail():
    g = SnakeGames(1, SnakeConfig(size=10), seed=0)
    g.food[0] = (0, 0)  # keep the food out of the way
    g.step(np.array([STRAIGHT]))
    assert tuple(g.head[0]) == (5, 6)
    assert (g.board[0] > 0).sum() == 3
    assert g.board[0, 5, 3] == 0


def test_eating_grows_and_rewards():
    g = SnakeGames(1, SnakeConfig(size=10), seed=0)
    g.food[0] = (5, 6)
    reward, done, _ = g.step(np.array([STRAIGHT]))
    assert reward[0] == 1.0 and not done[0]
    assert g.length[0] == 4 and (g.board[0] > 0).sum() == 4
    assert g.board[0, 5, 3] == 1  # tail did not move
    assert g.board[0][tuple(g.food[0])] == 0  # new food on an empty cell


def test_wall_kills():
    g = SnakeGames(1, SnakeConfig(size=10), seed=0)
    g.food[0] = (9, 9)
    for _ in range(4):
        _, done, _ = g.step(np.array([STRAIGHT]))
        assert not done[0]
    _, done, _ = g.step(np.array([STRAIGHT]))  # col 10 is off the board
    assert done[0]


def test_biting_yourself_kills_but_chasing_tail_is_fine():
    g = SnakeGames(1, SnakeConfig(size=10, start_length=5), seed=0)
    g.food[0] = (0, 0)
    # A length-5 snake turning in a tight square bites its own body.
    g.step(np.array([TURN_RIGHT]))
    g.step(np.array([TURN_RIGHT]))
    _, done, _ = g.step(np.array([TURN_RIGHT]))
    assert done[0]
    # A length-4 snake can follow its own tail around a 2x2 square forever.
    g = SnakeGames(1, SnakeConfig(size=10, start_length=4), seed=0)
    g.food[0] = (0, 0)
    for _ in range(12):
        _, done, _ = g.step(np.array([TURN_LEFT]))
        assert not done[0]


def test_filling_the_board_wins():
    g = SnakeGames(1, SnakeConfig(size=2, start_length=3), seed=0)
    # 2x2 board: snake laid out at row 1 cols 1,0 and off-board; rebuild by hand.
    g.board[0] = [[0, 0], [2, 3]]
    g.head[0] = (1, 1)
    g.length[0] = 3
    g.direction[0] = 0  # up
    g.food[0] = (0, 1)
    reward, done, won = g.step(np.array([STRAIGHT]))
    assert won[0] and done[0] and reward[0] > 1


def test_batch_games_are_independent():
    g = SnakeGames(4, SnakeConfig(size=8), seed=1)
    for _ in range(50):
        _, done, _ = g.step(np.random.default_rng(0).integers(0, 3, 4))
        g.reset(done)
    obs = g.observe()
    assert obs.shape == (4, 4, 10, 10)
    assert (obs[:, 0].sum(axis=(1, 2)) == 1).all()
    assert (obs[:, 2].sum(axis=(1, 2)) == 1).all()


def test_ego_view_puts_heading_up():
    from ailearns.games.snake import DOWN, LEFT, RIGHT, UP
    for heading in (UP, RIGHT, DOWN, LEFT):
        g = SnakeGames(1, SnakeConfig(size=10), seed=0)
        g.direction[0] = heading
        dr, dc = [(-1, 0), (0, 1), (1, 0), (0, -1)][heading]
        g.food[0] = (5 + 2 * dr, 5 + 2 * dc)  # two cells straight ahead
        obs = g.observe("ego")
        assert obs.shape == (1, 3, 19, 19)
        fr, fc = np.argwhere(obs[0, 1] == 1)[0]
        assert (fr, fc) == (9 - 2, 9), heading  # centre is (9, 9); ahead = up
    # Head at row 5: rows 0..4 above it are board, the 6th cell up is wall.
    g = SnakeGames(1, SnakeConfig(size=10), seed=0)
    g.direction[0] = UP
    wall = g.observe("ego")[0, 2]
    assert wall[9 - 5, 9] == 0 and wall[9 - 6, 9] == 1


def test_starving_is_punished():
    g = SnakeGames(1, SnakeConfig(size=10, start_length=4, hunger_base=10, hunger_per_length=0), seed=0)
    g.food[0] = (0, 0)
    for _ in range(10):
        _, done, _ = g.step(np.array([TURN_LEFT]))  # chase own tail, never eat
        assert not done[0]
    reward, done, _ = g.step(np.array([TURN_LEFT]))
    assert done[0] and reward[0] == -1.0


def test_ego2_sees_the_whole_body():
    g = SnakeGames(1, SnakeConfig(size=10, start_length=3), seed=0)
    obs = g.observe("ego2")
    assert obs.shape == (1, 4, 19, 19)
    occupied, fade = obs[0, 0], obs[0, 1]
    # head at centre is not drawn; the 2 body cells behind it are fully visible
    assert occupied[9, 9] == 0 and occupied.sum() == 2
    assert np.allclose(sorted(fade[occupied == 1]), [1 / 3, 2 / 3])
    # ego (old) is unchanged: 3 planes
    assert g.observe("ego").shape == (1, 3, 19, 19)
