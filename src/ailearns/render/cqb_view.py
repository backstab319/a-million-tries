"""Drawing CQB top-down (the plain view for building and checking the game; the isometric look
comes later). Pure drawing: takes a building from games.cqb.make_building and positions in
metres, never touches the simulation.
"""

import numpy as np
import pygame

from ailearns.render import theme as T

YARD = (18, 21, 29)
FLOOR = (34, 38, 50)
WALL = (150, 158, 178)
DOOR = T.AMBER
COVER = (104, 96, 84)
ASSAULT = T.BLUE
DEFENSE = T.RED


class View:
    """Metres -> pixels for the part of the yard from `lo` to `lo + size` drawn into `rect`
    (y up in the game, down on screen)."""

    def __init__(self, rect: pygame.Rect, size: tuple[float, float], lo: tuple[float, float] = (0.0, 0.0)):
        self.k = min(rect.w / size[0], rect.h / size[1])
        self.lo = np.asarray(lo, float)
        self.x0 = rect.x + (rect.w - size[0] * self.k) / 2 - self.lo[0] * self.k
        self.y0 = rect.y + (rect.h + size[1] * self.k) / 2 + self.lo[1] * self.k
        self.yard = size

    def px(self, p) -> tuple[float, float]:
        return float(self.x0 + p[0] * self.k), float(self.y0 - p[1] * self.k)

    def box(self, lo, hi) -> pygame.Rect:
        x0, y1 = self.px(lo)
        x1, y0 = self.px(hi)
        return pygame.Rect(round(x0), round(y0), max(1, round(x1 - x0)), max(1, round(y1 - y0)))


def draw_building(surf: pygame.Surface, b: dict, view: View, doors_open=None, starts: bool = False) -> None:
    """The yard, floors, walls, cover and doors (closed: a bar in the doorway; open: just the gap).
    `starts` also marks Assault's entry points (blue arrows) and Defense's start spots (red dots)."""
    surf.fill(YARD, view.box(view.lo, view.lo + np.asarray(view.yard)))
    for lo, hi in b["rooms"]:
        surf.fill(FLOOR, view.box(lo, hi))
    for lo, hi in zip(b["cover_lo"], b["cover_hi"]):
        surf.fill(COVER, view.box(lo, hi))
    for lo, hi in zip(b["wall_lo"], b["wall_hi"]):
        surf.fill(WALL, view.box(lo, hi))
    for i, d in enumerate(b["doors"]):
        if doors_open is None or not doors_open[i]:
            surf.fill(DOOR, view.box(d["lo"], d["hi"]))
    if starts:
        for q in b["spawn"]:
            pygame.draw.circle(surf, DEFENSE, view.px(q), max(2, view.k * 0.12))
        for p, face, _ in b["entries"]:
            d = np.array([np.cos(face), np.sin(face)])
            n = np.array([-d[1], d[0]])
            tri = [view.px(p + d * 0.8), view.px(p - d * 0.4 + n * 0.5), view.px(p - d * 0.4 - n * 0.5)]
            pygame.draw.polygon(surf, ASSAULT, tri)


# ── A round ──────────────────────────────────────────────────────────────────

DARK = (4, 5, 9)
LIGHT = (255, 236, 190)       # the weapon light's warm tint
FLASH = (255, 214, 120)
BANG = (240, 244, 255)        # a flashbang's light: cold white
GRENADE = (70, 84, 70)        # the flashbang itself, olive


def light_polygon(view: View, p, face: float, angles, reach) -> list:
    """The lit area: the operator, then the end of each ray of its light's fan."""
    pts = [view.px(p)]
    for a, r in zip(angles, reach):
        pts.append(view.px(p + r * np.array([np.cos(face + a), np.sin(face + a)])))
    return pts


def draw_fog(surf: pygame.Surface, view: View, lights: list, remembered=None, dark: int = 215, memory: int = 150) -> None:
    """Darkness over everything except the lit polygons; cells already looked at (rects) stay a
    little less dark: what this side has seen of the building."""
    fog = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    fog.fill((*DARK, dark))
    for lo, hi in remembered or []:
        fog.fill((*DARK, memory), view.box(lo, hi))
    for poly in lights:
        pygame.draw.polygon(fog, (0, 0, 0, 0), poly)
    surf.blit(fog, (0, 0))


def draw_light_tint(surf: pygame.Surface, poly: list, alpha: int = 48) -> None:
    tint = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(tint, (*LIGHT, alpha), poly)
    surf.blit(tint, (0, 0))


def draw_operator(surf: pygame.Surface, view: View, p, face: float, color, alpha: int = 255, down: bool = False) -> None:
    """A disc the operator's true size, with its weapon pointing where it faces; down: a cross."""
    r = max(3.0, 0.3 * view.k)
    c = view.px(p)
    layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    if down:
        d = r * 0.9
        for s in (1, -1):
            pygame.draw.line(layer, (*color, alpha), (c[0] - d, c[1] - s * d), (c[0] + d, c[1] + s * d), max(2, int(r / 3)))
    else:
        pygame.draw.circle(layer, (*color, alpha), c, r)
        tip = view.px(p + 0.75 * np.array([np.cos(face), np.sin(face)]))
        pygame.draw.line(layer, (*T.TEXT, alpha), c, tip, max(2, int(r / 3)))
    surf.blit(layer, (0, 0))


def draw_flash(surf: pygame.Surface, view: View, p, face: float, strength: float = 1.0) -> None:
    tip = np.asarray(p) + 0.6 * np.array([np.cos(face), np.sin(face)])
    glow = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    for k, rad in enumerate((1.1, 0.6, 0.25)):
        pygame.draw.circle(glow, (*FLASH, int(strength * (40 + 70 * k))), view.px(tip), rad * view.k)
    surf.blit(glow, (0, 0))


def draw_heard(surf: pygame.Surface, view: View, p, direction, loudness: float, color=T.TEXT) -> None:
    """Heard, not seen: an arc just outside the listener, on the side the sound came from."""
    ang = np.arctan2(direction[1], direction[0])
    c = view.px(p)
    r = 1.1 * view.k
    rect = pygame.Rect(c[0] - r, c[1] - r, 2 * r, 2 * r)
    layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    pygame.draw.arc(layer, (*color, int(90 + 165 * loudness)), rect, ang - 0.45, ang + 0.45, max(2, int(view.k * 0.12)))
    surf.blit(layer, (0, 0))


def draw_flashbang(surf: pygame.Surface, view: View, p, blink: bool) -> None:
    """A live flashbang lying on the floor (its fuse light blinks)."""
    c = view.px(p)
    r = max(4.0, 0.22 * view.k)                 # bigger than life (a real one is ~6 cm): it has to read on a phone
    pygame.draw.circle(surf, GRENADE, c, r)
    pygame.draw.circle(surf, T.TEXT, c, r, max(1, int(r / 4)))
    if blink:
        pygame.draw.circle(surf, T.AMBER, c, max(1.5, r * 0.45))


def draw_bang(surf: pygame.Surface, area: list, strength: float) -> None:
    """The bang: the area its light reaches (walls stop it), white, fading as strength goes 1 -> 0."""
    layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(layer, (*BANG, int(235 * strength)), area)
    surf.blit(layer, (0, 0))


def draw_health(surf: pygame.Surface, view: View, p, frac: float) -> None:
    """A small health bar under the operator: green, amber when wounded (under half), red when nearly down."""
    c = view.px(p)
    w, h = max(14.0, 0.9 * view.k), max(3.0, 0.12 * view.k)
    y = c[1] + max(6.0, 0.5 * view.k)
    pygame.draw.rect(surf, T.PANEL_EDGE, pygame.Rect(c[0] - w / 2, y, w, h))
    color = T.GREEN if frac >= 0.5 else (T.AMBER if frac >= 0.25 else T.RED)
    pygame.draw.rect(surf, color, pygame.Rect(c[0] - w / 2, y, w * max(frac, 0.0), h))


def draw_reloading(surf: pygame.Surface, view: View, p, done: float) -> None:
    """Reloading: an amber ring around the operator, filling up as the reload gets done (0 -> 1)."""
    c = view.px(p)
    r = max(6.0, 0.55 * view.k)
    rect = pygame.Rect(c[0] - r, c[1] - r, 2 * r, 2 * r)
    width = max(2, int(view.k * 0.08))
    pygame.draw.circle(surf, T.MUTED, c, r, width)
    if done > 0:
        pygame.draw.arc(surf, T.AMBER, rect, np.pi / 2 - 2 * np.pi * done, np.pi / 2, width)


def draw_blind(surf: pygame.Surface, view: View, p, strength: float) -> None:
    """Blinded: a white haze around the operator (strength 1 -> 0 as it wears off)."""
    layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    for k, rad in enumerate((1.0, 0.7, 0.45)):
        pygame.draw.circle(layer, (*BANG, int(strength * (35 + 45 * k))), view.px(p), rad * view.k)
    surf.blit(layer, (0, 0))
