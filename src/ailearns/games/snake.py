"""Snake, written so that many games run at once as numpy arrays.

The board stores, for every cell, how many more turns a body segment will
stay there. The head is written with the snake's length, every step all
positive cells count down by one, and the tail disappears when it hits zero.
Eating simply skips the countdown for one step, so the snake grows.
That one trick makes moving, growing and collision checks plain array maths.
"""

from dataclasses import dataclass

import numpy as np

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DELTAS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])  # (row, col) per direction

# Actions are relative to where the snake is heading, so "reverse into
# yourself" is never an option and the agent only has three choices.
STRAIGHT, TURN_RIGHT, TURN_LEFT = 0, 1, 2
N_ACTIONS = 3

REWARD_FOOD = 1.0
REWARD_DEATH = -1.0
REWARD_WIN = 10.0


@dataclass
class SnakeConfig:
    size: int = 10
    start_length: int = 3
    # Turns allowed without eating before the game is cut short; scales with
    # length, because a long snake needs a long route to reach the food.
    hunger_base: int = 100
    hunger_per_length: int = 2


class SnakeGames:
    """`n` independent games of Snake that step together."""

    def __init__(self, n: int = 1, config: SnakeConfig | None = None, seed: int | None = None):
        self.n = n
        self.cfg = config or SnakeConfig()
        self.rng = np.random.default_rng(seed)
        s = self.cfg.size
        self.board = np.zeros((n, s, s), dtype=np.int32)
        self.head = np.zeros((n, 2), dtype=np.int64)
        self.direction = np.zeros(n, dtype=np.int64)
        self.food = np.zeros((n, 2), dtype=np.int64)
        self.length = np.zeros(n, dtype=np.int64)
        self.hunger = np.zeros(n, dtype=np.int64)
        self.steps = np.zeros(n, dtype=np.int64)
        self.starved = np.zeros(n, dtype=bool)  # which games starved on the last step
        self.reset()

    @property
    def score(self) -> np.ndarray:
        return self.length - self.cfg.start_length

    @property
    def max_length(self) -> int:
        return self.cfg.size * self.cfg.size

    def reset(self, mask: np.ndarray | None = None) -> None:
        idx = np.arange(self.n) if mask is None else np.flatnonzero(mask)
        if idx.size == 0:
            return
        s, L = self.cfg.size, self.cfg.start_length
        row, col = s // 2, s // 2
        self.board[idx] = 0
        for k in range(L):  # lay the snake out horizontally, head on the right
            self.board[idx, row, col - k] = L - k
        self.head[idx] = (row, col)
        self.direction[idx] = RIGHT
        self.length[idx] = L
        self.hunger[idx] = 0
        self.steps[idx] = 0
        self._place_food(idx)

    def _place_food(self, idx: np.ndarray) -> None:
        # Pick a uniformly random empty cell per game: random scores on empty
        # cells, -1 on occupied ones, take the argmax.
        flat = self.board[idx].reshape(idx.size, -1)
        scores = np.where(flat == 0, self.rng.random(flat.shape), -1.0)
        cell = scores.argmax(axis=1)
        self.food[idx, 0], self.food[idx, 1] = np.divmod(cell, self.cfg.size)

    def step(self, actions: np.ndarray):
        """Advance every game by one move.

        Returns (reward, done, won). Finished games are NOT reset here; the
        caller decides (training resets them, recording may linger on them).
        """
        actions = np.asarray(actions)
        turn = np.select([actions == TURN_RIGHT, actions == TURN_LEFT], [1, 3], 0)
        self.direction = (self.direction + turn) % 4
        new_head = self.head + DELTAS[self.direction]
        r, c = new_head[:, 0], new_head[:, 1]
        s = self.cfg.size
        all_ = np.arange(self.n)

        hit_wall = (r < 0) | (r >= s) | (c < 0) | (c >= s)
        rr, cc = np.clip(r, 0, s - 1), np.clip(c, 0, s - 1)
        ate = ~hit_wall & (r == self.food[:, 0]) & (c == self.food[:, 1])
        # The tail (value 1) moves out this turn, so only cells > 1 are solid.
        hit_self = ~hit_wall & (self.board[all_, rr, cc] > 1)
        dead = hit_wall | hit_self
        alive = ~dead

        # Move: count down every body cell (unless we ate), then write the head.
        countdown = alive & ~ate
        self.board[countdown] = np.maximum(self.board[countdown] - 1, 0)
        self.length[ate] += 1
        a = np.flatnonzero(alive)
        self.board[a, rr[a], cc[a]] = self.length[a]
        self.head[a] = new_head[a]

        won = ate & (self.length == self.max_length)
        needs_food = ate & ~won
        if needs_food.any():
            self._place_food(np.flatnonzero(needs_food))

        self.steps += 1
        self.hunger = np.where(ate, 0, self.hunger + 1)
        starved = alive & ~won & (
            self.hunger > self.cfg.hunger_base + self.cfg.hunger_per_length * self.length
        )

        self.starved = starved
        reward = np.zeros(self.n, dtype=np.float32)
        reward[ate] = REWARD_FOOD
        reward[won] += REWARD_WIN
        # Starving must hurt as much as dying, or the agent learns that
        # spinning in circles forever is the safest strategy (it did).
        reward[dead | starved] = REWARD_DEATH
        done = dead | won | starved
        return reward, done, won

    def observe(self, view: str = "board") -> np.ndarray:
        """What the agent sees, as (n, channels, h, w) float32 planes.

        view="board": the whole board as-is, with a ring of wall cells.
            planes: head, body, food, wall
        view="ego2": same, plus an "occupied" plane (1.0 on every body cell).
            planes: occupied, body fade, food, wall
        view="ego": the snake's-eye view. The world is cropped around the
            head and rotated so the snake always faces up. "Food ahead and to
            the left" then looks the same wherever the snake is, so the agent
            learns that skill once instead of once per board position.
            planes: body, food, wall   (the head is always the centre cell)

        Body cells fade from 1.0 at the neck to ~0 at the tail, which says
        which way the snake is going and which cells will clear soonest.
        """
        n, s = self.n, self.cfg.size
        all_ = np.arange(n)
        hr, hc = self.head[:, 0], self.head[:, 1]
        body = (self.board / self.length[:, None, None]).astype(np.float32)
        body[all_, hr, hc] = 0.0
        food = np.zeros((n, s, s), dtype=np.float32)
        food[all_, self.food[:, 0], self.food[:, 1]] = 1.0

        if view == "board":
            head = np.zeros((n, s, s), dtype=np.float32)
            head[all_, hr, hc] = 1.0
            planes = np.stack([head, body, food], axis=1)
            obs = np.pad(planes, ((0, 0), (0, 0), (1, 1), (1, 1)))
            wall = np.ones((n, 1, s + 2, s + 2), dtype=np.float32)
            wall[:, :, 1:-1, 1:-1] = 0.0
            return np.concatenate([obs, wall], axis=1)

        if view in ("ego", "ego2"):
            # ego2 adds an "occupied" plane: 1.0 on every body cell. With only
            # the fade, the tail end of a long snake is ~0.02, i.e. invisible,
            # and 80% of the episode-1 agent's deaths were crashes into it.
            planes = [body, food] if view == "ego" else [(self.board > 0).astype(np.float32), body, food]
            planes[0][all_, hr, hc] = 0.0  # the head is the centre, not body
            k = len(planes)
            w, pad = 2 * s - 1, s - 1
            world = np.zeros((n, s + 2 * pad, s + 2 * pad, k + 1), dtype=np.float32)
            world[..., k] = 1.0  # everything outside the board is wall
            for j, plane in enumerate(planes):
                world[:, pad:pad + s, pad:pad + s, j] = plane
            world[:, pad:pad + s, pad:pad + s, k] = 0.0
            rows = hr[:, None] + np.arange(w)  # head lands at (pad, pad)
            cols = hc[:, None] + np.arange(w)
            crop = world[all_[:, None, None], rows[:, :, None], cols[:, None, :]]  # (n, w, w, k+1)
            for d in range(1, 4):  # rotate so the heading points up
                m = self.direction == d
                if m.any():
                    crop[m] = np.rot90(crop[m], k=d, axes=(1, 2))
            return np.ascontiguousarray(crop.transpose(0, 3, 1, 2))

        raise ValueError(view)

    @classmethod
    def from_snapshot(cls, snap: dict) -> "SnakeGames":
        """A one-game batch in exactly the state of `snap` (for showing what the agent saw)."""
        g = cls(1, SnakeConfig(size=snap["size"]))
        g.board[0] = snap["board"]
        g.head[0] = snap["head"]
        g.direction[0] = snap["direction"]
        g.food[0] = snap["food"]
        g.length[0] = snap["length"]
        return g

    def snapshot(self, i: int = 0) -> dict:
        """A copy of one game's state, for the renderer."""
        return {
            "board": self.board[i].copy(),
            "head": tuple(self.head[i]),
            "direction": int(self.direction[i]),
            "food": tuple(self.food[i]),
            "length": int(self.length[i]),
            "score": int(self.score[i]),
            "size": self.cfg.size,
        }
