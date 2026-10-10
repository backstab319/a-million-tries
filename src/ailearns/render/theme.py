"""The channel's look. Every renderer pulls colors and fonts from here."""

import os
from functools import cache
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

ASSETS = Path(__file__).resolve().parents[3] / "assets"

WIDTH, HEIGHT, FPS = 1920, 1080, 60

BG = (13, 15, 22)
PANEL = (22, 25, 35)
PANEL_EDGE = (38, 43, 58)
GRID_A = (24, 27, 38)
GRID_B = (28, 32, 44)
TEXT = (232, 234, 240)
MUTED = (120, 128, 148)

GREEN = (74, 222, 128)
GREEN_DARK = (21, 128, 61)
RED = (244, 63, 94)
AMBER = (251, 191, 36)
BLUE = (96, 165, 250)
PURPLE = (167, 139, 250)
ORANGE = (251, 146, 60)

FONT_FILES = {
    "regular": "Inter-Regular.ttf",
    "semibold": "Inter-SemiBold.ttf",
    "bold": "Inter-Bold.ttf",
    "display": "InterDisplay-Black.ttf",
    "mono": "JetBrainsMono-Regular.ttf",
    "mono-bold": "JetBrainsMono-Bold.ttf",
}


def init(headless: bool = True) -> None:
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()


class Window:
    """A game window that fits the screen (and can be resized). Draw on `canvas`, always
    full size, then `show()` scales it into the window."""

    def __init__(self, size: tuple[int, int], caption: str):
        w, h = size
        dw, dh = pygame.display.get_desktop_sizes()[0]
        fit = min(1.0, dw * 0.95 / w, dh * 0.85 / h)  # leave room for the title bar and the taskbar
        self.window = pygame.display.set_mode((int(w * fit), int(h * fit)), pygame.RESIZABLE)
        pygame.display.set_caption(caption)
        self.canvas = pygame.Surface(size)

    def show(self) -> None:
        (w, h), (ww, wh) = self.canvas.get_size(), self.window.get_size()
        k = min(ww / w, wh / h)
        size = (int(w * k), int(h * k))
        self.window.fill(BG)
        self.window.blit(pygame.transform.smoothscale(self.canvas, size), ((ww - size[0]) // 2, (wh - size[1]) // 2))
        pygame.display.flip()


@cache
def font(kind: str, size: int) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame.font.Font(str(ASSETS / "fonts" / FONT_FILES[kind]), size)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def mix(c1, c2, t: float) -> tuple[int, int, int]:
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))


def ease_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_in_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def text(surface, s: str, pos, kind="semibold", size=32, color=TEXT, anchor="topleft", alpha=255):
    img = font(kind, size).render(s, True, color)
    if alpha < 255:
        img.set_alpha(alpha)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


def panel(surface, rect, radius=24, color=PANEL, edge=PANEL_EDGE):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    pygame.draw.rect(surface, edge, rect, width=2, border_radius=radius)


def glow(surface, center, radius: float, color, strength=60, layers=6):
    """Soft additive halo, used for food and highlights."""
    size = int(radius * 2 + 4)
    halo = pygame.Surface((size, size), pygame.SRCALPHA)
    for i in range(layers, 0, -1):
        r = radius * i / layers
        a = int(strength * (1 - i / (layers + 1)))
        pygame.draw.circle(halo, (*color, a), (size / 2, size / 2), r)
    surface.blit(halo, (center[0] - size / 2, center[1] - size / 2))

# Chart series, in fixed order. Validated (dataviz validate_palette.js, dark
# mode, surface BG): lightness band, chroma, adjacent CVD all pass. Green vs
# amber is only 6.2 dE for protans, so charts always direct-label each line.
SERIES = [(22, 163, 74), (59, 130, 246), (217, 119, 6)]
GRIDLINE = (30, 34, 46)


class Confetti:
    """A one-shot burst of confetti, for wins. Call draw(surface, seconds_since_burst)."""

    def __init__(self, origin, n: int = 160, seed: int = 0, spread: float = 900):
        import random
        rng = random.Random(seed)
        self.origin = origin
        self.parts = [(rng.uniform(-spread, spread), rng.uniform(-1500, -700), rng.uniform(-8, 8),
                       rng.choice([GREEN, AMBER, BLUE, PURPLE, RED, TEXT]), rng.uniform(16, 28)) for _ in range(n)]

    def draw(self, surface, t: float) -> None:
        import math
        if t < 0 or t > 3.5:
            return
        for vx, vy, spin, color, size in self.parts:
            x = self.origin[0] + vx * t
            y = self.origin[1] + vy * t + 0.5 * 1800 * t * t
            w = size * abs(math.cos(spin * t))  # flips as it tumbles
            pygame.draw.rect(surface, color, (x - w / 2, y - size / 4, max(2, w), size / 2))
