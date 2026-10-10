"""Play Snake yourself (arrow keys / WASD). Good for the 'can I even beat it?' shot."""

import numpy as np
import pygame

from ailearns.games.snake import DOWN, LEFT, RIGHT, UP, STRAIGHT, TURN_LEFT, TURN_RIGHT, SnakeConfig, SnakeGames
from ailearns.render import snake_view, theme as T

KEYS = {
    pygame.K_UP: UP, pygame.K_w: UP, pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
    pygame.K_DOWN: DOWN, pygame.K_s: DOWN, pygame.K_LEFT: LEFT, pygame.K_a: LEFT,
}


def relative_action(current: int, wanted: int | None) -> int:
    if wanted is None or wanted == current or (wanted - current) % 4 == 2:
        return STRAIGHT
    return TURN_RIGHT if (wanted - current) % 4 == 1 else TURN_LEFT


def run(size: int = 10, moves_per_second: float = 8.0, record: str | None = None) -> None:
    T.init(headless=False)
    window = T.Window((900, 1000), "ai-learns — Snake")
    screen = window.canvas
    clock = pygame.time.Clock()
    game = SnakeGames(1, SnakeConfig(size=size))
    writer = None
    if record:
        from ailearns.media.video import VideoWriter
        writer = VideoWriter(record, (900, 1000), fps=60)

    queued: list[int] = []
    dead = won = False
    best, t, since_move = 0, 0.0, 0.0
    while True:
        dt = clock.tick(60) / 1000
        t += dt
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                if writer:
                    writer.close()
                pygame.quit()
                return
            if e.type == pygame.KEYDOWN:
                if e.key in KEYS and len(queued) < 2:
                    queued.append(KEYS[e.key])
                elif e.key in (pygame.K_SPACE, pygame.K_RETURN) and (dead or won):
                    game.reset()
                    dead = won = False

        since_move += dt
        if not (dead or won) and since_move >= 1 / moves_per_second:
            since_move = 0.0
            wanted = queued.pop(0) if queued else None
            _, done, w = game.step(np.array([relative_action(int(game.direction[0]), wanted)]))
            won = bool(w[0])
            dead = bool(done[0]) and not won
            best = max(best, int(game.score[0]))

        screen.fill(T.BG)
        snake_view.draw(screen, pygame.Rect(50, 140, 800, 800), game.snapshot(), dead=dead, won=won, t=t)
        T.text(screen, f"SCORE {game.score[0]}", (50, 50), "display", 56)
        T.text(screen, f"BEST {best}", (850, 62), "bold", 36, T.MUTED, anchor="topright")
        if dead or won:
            T.text(screen, "YOU WIN!" if won else "space to restart", (450, 965),
                   "bold", 28, T.GREEN if won else T.MUTED, anchor="center")
        window.show()
        if writer:
            writer.write(screen)
