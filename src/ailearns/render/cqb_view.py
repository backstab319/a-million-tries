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
# (ep13) homes, as the web game's top-down (web/cqb/app.js): inside walls, furniture under 1 m, windows
DRYWALL = (108, 116, 137)
LOW = (75, 73, 84)
WINDOW = (125, 211, 252)
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
    """The yard, floors, walls, cover and doors (closed: a bar in the doorway; open: just the gap). Homes (ep13) also
    draw furniture under 1 m, windows and inside walls (drywall) in their own colours, as the web game's top-down.
    `starts` also marks Assault's entry points (blue arrows) and Defense's start spots (red dots)."""
    surf.fill(YARD, view.box(view.lo, view.lo + np.asarray(view.yard)))
    for lo, hi in b["rooms"]:
        surf.fill(FLOOR, view.box(lo, hi))
    for lo, hi in zip(b.get("low_lo", ()), b.get("low_hi", ())):
        surf.fill(LOW, view.box(lo, hi))
    for lo, hi in zip(b["cover_lo"], b["cover_hi"]):
        surf.fill(COVER, view.box(lo, hi))
    soft = b.get("wall_soft", np.zeros(len(b["wall_lo"]), bool))
    for lo, hi, dry in zip(b["wall_lo"], b["wall_hi"], soft):
        surf.fill(DRYWALL if dry else WALL, view.box(lo, hi))
    for lo, hi in zip(b.get("win_lo", ()), b.get("win_hi", ())):
        surf.fill(WINDOW, view.box(lo, hi))
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


def draw_pilot(surf: pygame.Surface, view: View, p, color) -> None:
    """(ep11) The drone pilot: a tablet with an antenna, no weapon (it doesn't fight), labelled PILOT."""
    c = view.px(p)
    w, h = max(10.0, 0.8 * view.k), max(7.0, 0.55 * view.k)
    rect = pygame.Rect(c[0] - w / 2, c[1] - h / 2, w, h)
    pygame.draw.rect(surf, color, rect, border_radius=max(2, int(h / 5)))
    pygame.draw.rect(surf, T.BG, rect.inflate(-max(3, w / 5), -max(3, h / 4)), border_radius=max(1, int(h / 8)))
    tip = (rect.right - w * 0.18, rect.top - h * 0.7)
    pygame.draw.line(surf, color, (rect.right - w * 0.18, rect.top), tip, max(2, int(view.k * 0.08)))
    pygame.draw.circle(surf, T.AMBER, tip, max(2, int(view.k * 0.1)))
    T.text(surf, "PILOT", (c[0], rect.bottom + max(4, 0.15 * view.k)), kind="bold", size=max(14, int(0.42 * view.k)),
           color=color, anchor="midtop")


def draw_feed_ring(surf: pygame.Surface, view: View, p, phase: float) -> None:
    """(ep11) The drone feed's lock on someone: a ring pulsing outward (the pilot knows where he is, through walls)."""
    c = view.px(p)
    layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    for k in range(2):
        f = (phase + k / 2) % 1.0
        r = (0.45 + 0.9 * f) * view.k
        pygame.draw.circle(layer, (*T.AMBER, int(170 * (1 - f))), c, r, max(2, int(view.k * 0.07)))
    surf.blit(layer, (0, 0))


def draw_call(surf: pygame.Surface, view: View, p, word: str, direction, strength: float) -> None:
    """(ep11) A radio call arriving at an operator: the word in a bubble above it and (direction: a world angle, or
    None) an arrow the way the call points; fades with strength 1 -> 0."""
    a = int(255 * max(0.0, min(strength, 1.0)))
    c = view.px(p)
    layer = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    if direction is not None:
        d = np.array([np.cos(direction), np.sin(direction)])
        start, end = view.px(np.asarray(p) + 0.55 * d), view.px(np.asarray(p) + 2.2 * d)
        width = max(3, int(view.k * 0.14))
        pygame.draw.line(layer, (*T.AMBER, a), start, end, width)
        side = np.array([-d[1], d[0]])
        head = [end, view.px(np.asarray(p) + 1.75 * d + 0.35 * side), view.px(np.asarray(p) + 1.75 * d - 0.35 * side)]
        pygame.draw.polygon(layer, (*T.AMBER, a), head)
    size = max(16, int(0.55 * view.k))
    img = T.font("bold", size).render(word, True, T.BG)
    box = img.get_rect(midbottom=(c[0], c[1] - 0.75 * view.k)).inflate(size * 0.7, size * 0.35)
    pygame.draw.rect(layer, (*T.AMBER, a), box, border_radius=int(size * 0.35))
    pygame.draw.polygon(layer, (*T.AMBER, a), [(box.centerx - size * 0.25, box.bottom), (box.centerx + size * 0.25, box.bottom),
                                                 (box.centerx, box.bottom + size * 0.3)])
    img.set_alpha(a)
    layer.blit(img, img.get_rect(center=box.center))
    surf.blit(layer, (0, 0))
