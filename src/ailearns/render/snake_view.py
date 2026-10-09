"""Draws a Snake game state into any rectangle of a surface."""

from functools import lru_cache

import numpy as np
import pygame

from ailearns.games.snake import DELTAS
from ailearns.render import theme as T


def body_order(board: np.ndarray, head) -> list[tuple[int, int]]:
    """Body cells from head to tail (each step: the neighbour one value lower)."""
    order = [tuple(head)]
    s = board.shape[0]
    while True:
        r, c = order[-1]
        v = board[r, c]
        if v <= 1:
            return order
        for dr, dc in DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < s and 0 <= nc < s and board[nr, nc] == v - 1:
                order.append((nr, nc))
                break
        else:
            return order


@lru_cache(maxsize=32)
def _board_background(s: int, px: int, radius: int, show_grid: bool) -> pygame.Surface:
    """Checkerboard with rounded corners, cached because grids redraw it a lot."""
    surf = pygame.Surface((px, px), pygame.SRCALPHA)
    surf.fill(T.GRID_A)
    cell = px / s
    if show_grid and cell >= 6:
        for r in range(s):
            for c in range(s):
                if (r + c) % 2:
                    pygame.draw.rect(surf, T.GRID_B, (c * cell, r * cell, cell + 1, cell + 1))
    mask = pygame.Surface((px, px), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, px, px), border_radius=radius)
    surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return surf


def draw(surface: pygame.Surface, rect: pygame.Rect, snap: dict, *, dead: bool = False,
         won: bool = False, t: float = 0.0, show_grid: bool = True) -> None:
    """`t` is a time in seconds, used only for idle animation (food pulse)."""
    s = snap["size"]
    cell = min(rect.w, rect.h) / s
    ox = rect.x + (rect.w - cell * s) / 2
    oy = rect.y + (rect.h - cell * s) / 2
    board_rect = pygame.Rect(round(ox), round(oy), round(cell * s), round(cell * s))
    radius = max(4, int(cell * 0.25))

    surface.blit(_board_background(s, board_rect.w, radius, show_grid), board_rect)
    edge = T.GREEN if won else (T.RED if dead else T.PANEL_EDGE)
    pygame.draw.rect(surface, edge, board_rect.inflate(4, 4), width=max(2, int(cell * 0.06)),
                     border_radius=radius)

    def center(rc):
        return (ox + (rc[1] + 0.5) * cell, oy + (rc[0] + 0.5) * cell)

    # Food: pulsing glow + dot.
    fc = center(snap["food"])
    if not won:
        pulse = 0.5 + 0.5 * np.sin(t * 5)
        T.glow(surface, fc, cell * (0.7 + 0.1 * pulse), T.RED, strength=40)
        pygame.draw.circle(surface, T.RED, fc, cell * 0.32)
        pygame.draw.circle(surface, (255, 160, 180), (fc[0] - cell * 0.09, fc[1] - cell * 0.1), cell * 0.08)

    # Snake: thick segments joined head to tail, fading green, grey if dead.
    order = body_order(snap["board"], snap["head"])
    n = len(order)
    pad = cell * 0.12
    for i in range(n - 1, -1, -1):
        f = i / max(n - 1, 1)
        color = T.mix(T.GREEN, T.GREEN_DARK, f * 0.85)
        if dead:
            color = T.mix(color, (90, 95, 110), 0.7)
        r, c = order[i]
        cell_rect = pygame.Rect(ox + c * cell + pad, oy + r * cell + pad, cell - 2 * pad, cell - 2 * pad)
        if i + 1 < n:  # bridge toward the next segment (toward the tail)
            nr, nc = order[i + 1]
            bridge = cell_rect.union(cell_rect.move((nc - c) * cell, (nr - r) * cell))
            pygame.draw.rect(surface, color, bridge,
                             border_radius=int(cell * 0.2))
        pygame.draw.rect(surface, color, cell_rect, border_radius=int(cell * 0.28))

    # Eyes on the head, looking where it's going.
    if cell >= 10:
        hc = center(snap["head"])
        dr, dc = DELTAS[snap["direction"]]
        side = np.array([-dr, dc])  # perpendicular, in (x, y)
        for sgn in (-1, 1):
            ex = hc[0] + dc * cell * 0.15 + side[0] * sgn * cell * 0.18
            ey = hc[1] + dr * cell * 0.15 + side[1] * sgn * cell * 0.18
            pygame.draw.circle(surface, (245, 245, 250), (ex, ey), cell * 0.13)
            if dead:
                k = cell * 0.06
                pygame.draw.line(surface, (20, 20, 25), (ex - k, ey - k), (ex + k, ey + k), max(1, int(cell * .03)))
                pygame.draw.line(surface, (20, 20, 25), (ex - k, ey + k), (ex + k, ey - k), max(1, int(cell * .03)))
            else:
                pygame.draw.circle(surface, (15, 17, 24), (ex + dc * cell * 0.04, ey + dr * cell * 0.04), cell * 0.07)
