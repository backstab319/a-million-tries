"""Snake again, in torch, so thousands of games can run on a GPU next to the network.

Same rules, same rewards and same observations as `snake.py` (which stays the
reference: recording, analysis and the renderer use it; `tests/` checks the two
agree move for move). The differences are all about never making the GPU wait
for the CPU:

- no "pick these games" index lists: every update is computed for all games
  and kept with a mask (`torch.where`),
- food: a candidate cell is drawn for every game every step; only games that
  just ate (or reset) take it,
- the snake's-eye rotation is one gather with precomputed index grids instead
  of a Python loop over headings.
"""

import numpy as np
import torch

from ailearns.games.snake import (DELTAS, REWARD_DEATH, REWARD_FOOD, REWARD_WIN, RIGHT, TURN_LEFT, TURN_RIGHT,
                                  SnakeConfig)


class TorchSnakeGames:
    """`n` games of Snake as tensors on `device`. Mirrors `SnakeGames`."""

    def __init__(self, n: int = 1, config: SnakeConfig | None = None, seed: int = 0, device: str = "cpu"):
        self.n = n
        self.cfg = config or SnakeConfig()
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        s, L, dev = self.cfg.size, self.cfg.start_length, self.device
        self.deltas = torch.as_tensor(DELTAS, device=dev)
        self.all = torch.arange(n, device=dev)

        start = torch.zeros((s, s), dtype=torch.int32, device=dev)
        for k in range(L):  # horizontal, head on the right, as in snake.py
            start[s // 2, s // 2 - k] = L - k
        self.start_board = start
        self.start_head = torch.tensor([s // 2, s // 2], device=dev)

        self.board = torch.zeros((n, s, s), dtype=torch.int32, device=dev)
        self.head = torch.zeros((n, 2), dtype=torch.long, device=dev)
        self.direction = torch.zeros(n, dtype=torch.long, device=dev)
        self.food = torch.zeros((n, 2), dtype=torch.long, device=dev)
        self.length = torch.zeros(n, dtype=torch.long, device=dev)
        self.hunger = torch.zeros(n, dtype=torch.long, device=dev)
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.starved = torch.zeros(n, dtype=torch.bool, device=dev)

        # Snake's-eye view: for heading d, output cell (i, j) comes from crop cell (rot_i[d][i,j], rot_j[d][i,j]),
        # the same as np.rot90(crop, k=d) in snake.py.
        w = 2 * s - 1
        I, J = np.meshgrid(np.arange(w), np.arange(w), indexing="ij")
        self.rot_i = torch.as_tensor(np.stack([np.rot90(I, d) for d in range(4)]).copy(), device=dev)
        self.rot_j = torch.as_tensor(np.stack([np.rot90(J, d) for d in range(4)]).copy(), device=dev)
        self.reset()

    @property
    def score(self) -> torch.Tensor:
        return self.length - self.cfg.start_length

    @property
    def max_length(self) -> int:
        return self.cfg.size * self.cfg.size

    def _random_empty_cell(self) -> torch.Tensor:
        """A uniformly random empty cell for every game, as (n, 2)."""
        flat = self.board.view(self.n, -1)
        scores = torch.where(flat == 0, torch.rand(flat.shape, generator=self.gen, device=self.device), -1.0)
        cell = scores.argmax(dim=1)
        return torch.stack([cell // self.cfg.size, cell % self.cfg.size], dim=1)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        m = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else mask
        self.board = torch.where(m[:, None, None], self.start_board, self.board)
        self.head = torch.where(m[:, None], self.start_head, self.head)
        self.direction = torch.where(m, RIGHT, self.direction)
        self.length = torch.where(m, self.cfg.start_length, self.length)
        self.hunger = torch.where(m, 0, self.hunger)
        self.steps = torch.where(m, 0, self.steps)
        self.food = torch.where(m[:, None], self._random_empty_cell(), self.food)

    def step(self, actions: torch.Tensor):
        """Advance every game by one move. Returns (reward, done, won); finished games are not reset."""
        s = self.cfg.size
        turn = torch.where(actions == TURN_RIGHT, 1, torch.where(actions == TURN_LEFT, 3, 0))
        self.direction = (self.direction + turn) % 4
        new_head = self.head + self.deltas[self.direction]
        r, c = new_head[:, 0], new_head[:, 1]

        hit_wall = (r < 0) | (r >= s) | (c < 0) | (c >= s)
        rr, cc = r.clamp(0, s - 1), c.clamp(0, s - 1)
        ate = ~hit_wall & (r == self.food[:, 0]) & (c == self.food[:, 1])
        hit_self = ~hit_wall & (self.board[self.all, rr, cc] > 1)  # the tail (1) moves out this turn
        dead = hit_wall | hit_self
        alive = ~dead

        countdown = alive & ~ate
        self.board = torch.where(countdown[:, None, None], (self.board - 1).clamp(min=0), self.board)
        self.length = self.length + ate.long()
        under_head = self.board[self.all, rr, cc]
        self.board[self.all, rr, cc] = torch.where(alive, self.length.int(), under_head)
        self.head = torch.where(alive[:, None], new_head, self.head)

        won = ate & (self.length == self.max_length)
        self.food = torch.where((ate & ~won)[:, None], self._random_empty_cell(), self.food)

        self.steps = self.steps + 1
        self.hunger = torch.where(ate, 0, self.hunger + 1)
        starved = alive & ~won & (self.hunger > self.cfg.hunger_base + self.cfg.hunger_per_length * self.length)
        self.starved = starved

        reward = ate.float() * REWARD_FOOD + won.float() * REWARD_WIN
        reward = torch.where(dead | starved, REWARD_DEATH, reward)
        return reward, dead | won | starved, won

    def observe(self, view: str = "board") -> torch.Tensor:
        """Same planes as SnakeGames.observe, as a (n, channels, h, w) float tensor."""
        n, s = self.n, self.cfg.size
        hr, hc = self.head[:, 0], self.head[:, 1]
        body = self.board.float() / self.length[:, None, None]
        body[self.all, hr, hc] = 0.0
        food = torch.zeros((n, s, s), device=self.device)
        food[self.all, self.food[:, 0], self.food[:, 1]] = 1.0

        if view == "board":
            head = torch.zeros((n, s, s), device=self.device)
            head[self.all, hr, hc] = 1.0
            obs = torch.nn.functional.pad(torch.stack([head, body, food], dim=1), (1, 1, 1, 1))
            wall = torch.ones((n, 1, s + 2, s + 2), device=self.device)
            wall[:, :, 1:-1, 1:-1] = 0.0
            return torch.cat([obs, wall], dim=1)

        if view in ("ego", "ego2"):
            if view == "ego":
                planes = [body, food]
            else:
                occupied = (self.board > 0).float()
                occupied[self.all, hr, hc] = 0.0  # the head is the centre, not body
                planes = [occupied, body, food]
            k, pad = len(planes), s - 1
            inner = torch.stack(planes + [torch.zeros((n, s, s), device=self.device)], dim=1)
            world = torch.nn.functional.pad(inner, (pad, pad, pad, pad))
            world[:, k] = 1.0  # wall everywhere ...
            world[:, k, pad:pad + s, pad:pad + s] = 0.0  # ... except on the board
            rows = hr[:, None, None] + self.rot_i[self.direction]  # (n, w, w); head lands at the centre
            cols = hc[:, None, None] + self.rot_j[self.direction]
            crop = world[self.all[:, None, None], :, rows, cols]  # (n, w, w, k+1)
            return crop.permute(0, 3, 1, 2).contiguous()

        raise ValueError(view)
