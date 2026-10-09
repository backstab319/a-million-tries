"""Snake once more, in JAX, so the whole training loop can be compiled (TPUs, and faster GPUs).

Same rules, rewards and observations as `snake.py` (the reference; recording
and analysis use it, and `tests/test_snake_jax.py` checks the two agree move for
move). Written for one game as pure functions of a `State`; `jax.vmap` turns
them into thousands of games, `jax.jit` compiles them.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.games.snake import (DELTAS, REWARD_DEATH, REWARD_FOOD, REWARD_WIN, RIGHT, TURN_LEFT, TURN_RIGHT,
                                  SnakeConfig)


class State(NamedTuple):
    board: jax.Array      # (s, s) int32: turns each body cell stays (head = length)
    head: jax.Array       # (2,) int32
    direction: jax.Array  # () int32
    food: jax.Array       # (2,) int32
    length: jax.Array     # () int32
    hunger: jax.Array     # () int32
    steps: jax.Array      # () int32
    starved: jax.Array    # () bool: starved on the last step


class JaxSnake:
    """Functions for one game of Snake; vmap them for many."""

    def __init__(self, config: SnakeConfig | None = None):
        self.cfg = cfg = config or SnakeConfig()
        s = cfg.size
        start = np.zeros((s, s), np.int32)
        for k in range(cfg.start_length):  # horizontal, head on the right, as in snake.py
            start[s // 2, s // 2 - k] = cfg.start_length - k
        self.start_board = jnp.asarray(start)
        self.deltas = jnp.asarray(DELTAS, jnp.int32)
        w = 2 * s - 1  # snake's-eye view: rotation as index tables, same as np.rot90(crop, k=d)
        I, J = np.meshgrid(np.arange(w), np.arange(w), indexing="ij")
        self.rot_i = jnp.asarray(np.stack([np.rot90(I, d) for d in range(4)]).copy(), jnp.int32)
        self.rot_j = jnp.asarray(np.stack([np.rot90(J, d) for d in range(4)]).copy(), jnp.int32)

    def _random_empty(self, key, board):
        flat = board.reshape(-1)
        scores = jnp.where(flat == 0, jax.random.uniform(key, flat.shape), -1.0)
        cell = jnp.argmax(scores)
        return jnp.stack([cell // self.cfg.size, cell % self.cfg.size]).astype(jnp.int32)

    def init(self, key) -> State:
        s, L = self.cfg.size, self.cfg.start_length
        return State(board=self.start_board, head=jnp.array([s // 2, s // 2], jnp.int32),
                     direction=jnp.int32(RIGHT), food=self._random_empty(key, self.start_board),
                     length=jnp.int32(L), hunger=jnp.int32(0), steps=jnp.int32(0), starved=jnp.bool_(False))

    def step(self, st: State, action, key):
        """One move. Returns (state, reward, done, won); a finished game is not reset."""
        s, cfg = self.cfg.size, self.cfg
        turn = jnp.where(action == TURN_RIGHT, 1, jnp.where(action == TURN_LEFT, 3, 0))
        direction = (st.direction + turn) % 4
        new_head = st.head + self.deltas[direction]
        r, c = new_head[0], new_head[1]
        hit_wall = (r < 0) | (r >= s) | (c < 0) | (c >= s)
        rr, cc = jnp.clip(r, 0, s - 1), jnp.clip(c, 0, s - 1)
        ate = ~hit_wall & (r == st.food[0]) & (c == st.food[1])
        hit_self = ~hit_wall & (st.board[rr, cc] > 1)  # the tail (1) moves out this turn
        dead = hit_wall | hit_self
        alive = ~dead

        board = jnp.where(alive & ~ate, jnp.maximum(st.board - 1, 0), st.board)
        length = st.length + ate.astype(jnp.int32)
        board = board.at[rr, cc].set(jnp.where(alive, length, board[rr, cc]))
        head = jnp.where(alive, new_head, st.head)
        won = ate & (length == s * s)
        food = jnp.where(ate & ~won, self._random_empty(key, board), st.food)
        hunger = jnp.where(ate, 0, st.hunger + 1)
        starved = alive & ~won & (hunger > cfg.hunger_base + cfg.hunger_per_length * length)
        reward = ate * REWARD_FOOD + won * REWARD_WIN
        reward = jnp.where(dead | starved, REWARD_DEATH, reward).astype(jnp.float32)
        st = State(board, head, direction, food, length, hunger, st.steps + 1, starved)
        return st, reward, dead | won | starved, won

    def reset_if(self, st: State, done, key) -> State:
        fresh = self.init(key)
        return jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, st)

    def observe(self, st: State, view: str = "ego2"):
        """Same planes as SnakeGames.observe, for one game: (channels, h, w) float32."""
        s = self.cfg.size
        hr, hc = st.head[0], st.head[1]
        body = (st.board / st.length).astype(jnp.float32).at[hr, hc].set(0.0)
        food = jnp.zeros((s, s), jnp.float32).at[st.food[0], st.food[1]].set(1.0)
        if view == "board":
            head = jnp.zeros((s, s), jnp.float32).at[hr, hc].set(1.0)
            obs = jnp.pad(jnp.stack([head, body, food]), ((0, 0), (1, 1), (1, 1)))
            wall = jnp.pad(jnp.zeros((1, s, s), jnp.float32), ((0, 0), (1, 1), (1, 1)), constant_values=1.0)
            return jnp.concatenate([obs, wall])
        if view in ("ego", "ego2"):
            planes = [body, food] if view == "ego" else \
                [(st.board > 0).astype(jnp.float32).at[hr, hc].set(0.0), body, food]
            pad = s - 1
            inner = jnp.stack(planes + [jnp.zeros((s, s), jnp.float32)])
            world = jnp.pad(inner, ((0, 0), (pad, pad), (pad, pad)))
            wall = jnp.pad(jnp.zeros((s, s), jnp.float32), pad, constant_values=1.0)
            world = world.at[-1].set(wall)
            rows = hr + self.rot_i[st.direction]  # head lands at the centre
            cols = hc + self.rot_j[st.direction]
            return world[:, rows, cols]
        raise ValueError(view)
