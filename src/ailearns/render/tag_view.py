"""Drawing tag: the arena, the two characters, what the chaser can't see, trails and pop-ups.

Pure drawing: takes an arena (from games.tag.make_arena) and positions in metres,
never touches the simulation.

The chaser is the channel's robot (green head, dark visor, glowing eyes); the runner
is an amber blob with big eyes that watch the chaser. Both are drawn bigger than
their 1 m collision disc so their faces read on a phone; the soft shadow under each
one is its true size. Sprites are drawn at 4x and scaled down, for smooth edges.
"""

from functools import lru_cache

import numpy as np
import pygame

from ailearns.render import theme as T

FLOOR_A = (36, 42, 58)
FLOOR_B = (40, 46, 63)
BLOCK_TOP = (150, 158, 178)    # obstacles and walls: bright enough to read at a glance
BLOCK_SIDE = (92, 99, 120)
BLOCK_EDGE = (190, 197, 214)
INK = (5, 6, 10)
FOG = (8, 10, 16)              # where the chaser can't see
CHASER_COLOR = T.GREEN         # the AI (episode 6)
RUNNER_COLOR = T.AMBER
WALL = 0.8                     # metres: how thick the edge wall is drawn
SIDE = 0.45                    # metres of an obstacle's footprint shown as its shaded side
LOOK = 2.3                     # characters are drawn this much bigger than their collision disc
SS = 4                         # supersampling for sprites


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain), counter-clockwise."""
    pts = sorted(map(tuple, pts))

    def half(points):
        out = []
        for p in points:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return np.array(half(pts) + half(pts[::-1]))


def _unit(v, fallback=(0.0, -1.0)) -> np.ndarray:
    """Screen-space unit vector (y down) from a world vector (y up)."""
    v = np.asarray(v, float) * [1, -1]
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.asarray(fallback)


# ── sprites (cached; `look` is quantised so the cache stays small) ──────────────

@lru_cache(maxsize=512)
def _robot(px: int, look: tuple, mood: str) -> pygame.Surface:
    """The channel's robot head, `px` wide. mood: hunt | lost | happy."""
    S = px * SS
    pad = S // 4
    surf = pygame.Surface((S + 2 * pad, S + 2 * pad), pygame.SRCALPHA)
    c = np.array([S / 2 + pad, S / 2 + pad])
    w, h = S * 0.86, S * 0.74
    ear = pygame.Rect(0, 0, S * 0.14, S * 0.30)
    for side in (-1, 1):  # ears
        ear.center = (c[0] + side * w / 2, c[1])
        pygame.draw.rect(surf, T.GREEN_DARK, ear, border_radius=int(S * 0.06))
    head = pygame.Rect(0, 0, w, h)
    head.center = c
    pygame.draw.rect(surf, T.GREEN_DARK, head.inflate(S * 0.06, S * 0.06), border_radius=int(S * 0.24))
    pygame.draw.rect(surf, CHASER_COLOR, head, border_radius=int(S * 0.22))
    visor = head.inflate(-S * 0.20, -S * 0.22)
    pygame.draw.rect(surf, (24, 28, 42), visor, border_radius=int(S * 0.15))
    d = np.asarray(look) * S * 0.06  # eyes shift towards where it's looking
    glow = (140, 255, 180)
    for side in (-1, 1):
        e = c + [side * S * 0.17, -S * 0.02] + d
        if mood == "happy":    # ^ ^
            r = pygame.Rect(0, 0, S * 0.18, S * 0.18)
            r.center = e + [0, S * 0.04]
            pygame.draw.arc(surf, glow, r, 0.2, np.pi - 0.2, int(S * 0.065))
        elif mood == "lost":   # round eyes
            pygame.draw.circle(surf, glow, e, S * 0.08)
        else:                  # hunting: narrowed eyes, slanting in towards the middle
            dy = S * 0.035
            pygame.draw.line(surf, glow, e + [-S * 0.09, side * -dy], e + [S * 0.09, side * dy], int(S * 0.10))
    m = c + [0, S * 0.17] + d * 0.5
    if mood == "happy":
        r = pygame.Rect(0, 0, S * 0.26, S * 0.16)
        r.center = m + [0, -S * 0.04]
        pygame.draw.arc(surf, glow, r, np.pi + 0.3, 2 * np.pi - 0.3, int(S * 0.04))
    elif mood == "lost":
        pygame.draw.line(surf, glow, m + [-S * 0.06, 0], m + [S * 0.06, 0], int(S * 0.035))
    return pygame.transform.smoothscale(surf, (surf.get_width() // SS, surf.get_height() // SS))


@lru_cache(maxsize=512)
def _blob(px: int, look: tuple, scared: bool) -> pygame.Surface:
    """The runner: an amber blob with big eyes, `px` wide."""
    S = px * SS
    pad = S // 4
    surf = pygame.Surface((S + 2 * pad, S + 2 * pad), pygame.SRCALPHA)
    c = np.array([S / 2 + pad, S / 2 + pad])
    dark = T.mix(RUNNER_COLOR, INK, 0.45)
    pygame.draw.circle(surf, dark, c, S * 0.5)
    pygame.draw.circle(surf, RUNNER_COLOR, c, S * 0.45)
    pygame.draw.circle(surf, T.mix(RUNNER_COLOR, (255, 255, 255), 0.35), c + [-S * 0.14, -S * 0.16], S * 0.10)
    d = np.asarray(look)
    er = S * (0.15 if scared else 0.13)
    for side in (-1, 1):
        e = c + [side * S * 0.16, -S * 0.04] + d * S * 0.05
        pygame.draw.circle(surf, (250, 250, 252), e, er)
        pygame.draw.circle(surf, INK, e + d * er * 0.45, er * (0.42 if scared else 0.5))
    m = c + [0, S * 0.20] + d * S * 0.03
    if scared:
        pygame.draw.ellipse(surf, INK, pygame.Rect(m[0] - S * 0.06, m[1] - S * 0.05, S * 0.12, S * 0.10))
    else:
        r = pygame.Rect(0, 0, S * 0.20, S * 0.12)
        r.center = m + [0, -S * 0.04]
        pygame.draw.arc(surf, INK, r, np.pi + 0.4, 2 * np.pi - 0.4, int(S * 0.035))
    return pygame.transform.smoothscale(surf, (surf.get_width() // SS, surf.get_height() // SS))


def _q(v: np.ndarray, steps: int = 4) -> tuple:
    return tuple(np.round(np.asarray(v) * steps) / steps)


def draw_clock(surf, rect: pygame.Rect, t: float, limit: float, caught: bool, done: bool, size: int = 44) -> None:
    """Countdown: how long the runner still has to survive. The label sits above the bar."""
    left = max(0.0, limit - t)
    if caught and done:
        label, color = "TAGGED in %.1f s" % t, CHASER_COLOR
    elif done:
        label, color = "ESCAPED!", RUNNER_COLOR
    else:
        label, color = "%.1f s left" % left, T.TEXT
    T.text(surf, label, (rect.x, rect.y - 12), kind="mono-bold", size=size, color=color, anchor="bottomleft")
    T.panel(surf, rect, radius=rect.h // 2)
    inner = rect.inflate(-8, -8)
    w = int(inner.w * left / limit)
    if w > 0:
        pygame.draw.rect(surf, T.RED if left < 5 else T.AMBER, (inner.x, inner.y, w, inner.h),
                         border_radius=inner.h // 2)


class ArenaView:
    """An arena fitted into `rect` (or just the `window` of it, (x0, y0, x1, y1) metres, to
    zoom in); the static picture is drawn once and reused."""

    def __init__(self, arena: dict, size: float, rect: pygame.Rect, margin: float = 0.03, window=None):
        self.arena, self.size, self.rect = arena, size, pygame.Rect(rect)
        x0, y0, x1, y1 = window or (-WALL, -WALL, size + WALL, size + WALL)
        self.scale = min(rect.w, rect.h) * (1 - 2 * margin) / max(x1 - x0, y1 - y0)
        mid = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        self.offset = np.array(self.rect.center) - mid * [self.scale, -self.scale]
        self.bg = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        self._draw_static()

    def px(self, p) -> np.ndarray:
        """World metres -> screen pixels (y up in the world, down on screen)."""
        p = np.asarray(p, dtype=float)
        return p * [self.scale, -self.scale] + self.offset

    def box_rect(self, lo, hi) -> pygame.Rect:
        a, b = self.px([lo[0], hi[1]]), self.px([hi[0], lo[1]])
        return pygame.Rect(a, b - a)

    def _block(self, surf, lo, hi, o, shadow: bool = True) -> None:
        """An obstacle with a little depth: soft drop shadow, shaded front side, lit top."""
        r = self.box_rect(lo, hi).move(-o)
        rad = max(1, int(self.scale * 0.25))
        if shadow:
            big = r.inflate(self.scale, self.scale)
            sh = pygame.Surface(big.size, pygame.SRCALPHA)
            pygame.draw.rect(sh, (0, 0, 0, 90), sh.get_rect().inflate(-self.scale * 0.5, -self.scale * 0.5),
                             border_radius=rad * 2)
            small = (max(1, sh.get_width() // 6), max(1, sh.get_height() // 6))
            sh = pygame.transform.smoothscale(pygame.transform.smoothscale(sh, small), big.size)
            surf.blit(sh, big.move(self.scale * 0.35, self.scale * 0.45))
        pygame.draw.rect(surf, BLOCK_SIDE, r, border_radius=rad)
        side = min(SIDE * self.scale, r.h * 0.4)
        top = pygame.Rect(r.x, r.y, r.w, r.h - side)
        pygame.draw.rect(surf, BLOCK_TOP, top, border_radius=rad)
        pygame.draw.line(surf, BLOCK_EDGE, (top.x + rad, top.y + 1), (top.right - rad - 1, top.y + 1),
                         max(1, int(self.scale * 0.08)))

    def _draw_static(self) -> None:
        o = np.array(self.rect.topleft)
        S, w = self.size, WALL
        floor = self.box_rect([0, 0], [S, S]).move(-o)
        pygame.draw.rect(self.bg, FLOOR_A, floor)
        tile = 2.0  # faint 2 m tiles, so speed reads on screen
        for i in range(int(S / tile)):
            for j in range(int(S / tile)):
                if (i + j) % 2:
                    pygame.draw.rect(self.bg, FLOOR_B, self.box_rect([i * tile, j * tile],
                                                                     [(i + 1) * tile, (j + 1) * tile]).move(-o))
        for lo, hi in ([[-w, -w], [S + w, 0]], [[-w, S], [S + w, S + w]],     # the edge wall
                       [[-w, 0], [0, S]], [[S, 0], [S + w, S]]):
            pygame.draw.rect(self.bg, BLOCK_SIDE, self.box_rect(lo, hi).move(-o))
        pygame.draw.rect(self.bg, BLOCK_TOP, self.box_rect([-w, -w], [S + w, S + w]).move(-o),
                         max(2, int(self.scale * w * 0.55)), border_radius=max(2, int(self.scale * 0.4)))
        for lo, hi in zip(self.arena["lo"], self.arena["hi"]):
            self._block(self.bg, lo, hi, o)

    def draw_arena(self, surface) -> None:
        clip = surface.get_clip()
        surface.set_clip(self.rect)
        surface.blit(self.bg, self.rect.topleft)
        surface.set_clip(clip)

    def draw_shadows(self, surface, eye, alpha: int = 165) -> None:
        """Fog over everything the player at `eye` can't see (behind obstacles), with soft edges."""
        r = self.rect
        k = 2  # drawn at half size, then averaged down 4x more and scaled back up: soft, smooth edges
        layer = pygame.Surface((max(1, r.w // k), max(1, r.h // k)), pygame.SRCALPHA)
        eye = np.asarray(eye, float)
        far = self.size * 3
        for lo, hi in zip(self.arena["lo"], self.arena["hi"]):
            corners = np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]], float)
            out = corners - eye
            proj = corners + out / np.linalg.norm(out, axis=1, keepdims=True) * far
            poly = (self.px(_hull(np.vstack([corners, proj]))) - r.topleft) / k
            pygame.draw.polygon(layer, (*FOG, alpha), poly)
        small = (max(1, r.w // (4 * k)), max(1, r.h // (4 * k)))
        layer = pygame.transform.smoothscale(pygame.transform.smoothscale(layer, small), r.size)
        clip = surface.get_clip()
        surface.set_clip(self.box_rect([0, 0], [self.size, self.size]).clip(r))
        surface.blit(layer, r.topleft)
        surface.set_clip(clip)
        for lo, hi in zip(self.arena["lo"], self.arena["hi"]):  # obstacles back on top of their own fog
            self._block(surface, lo, hi, np.zeros(2), shadow=False)

    def _ground_shadow(self, surface, pos, radius) -> None:
        r = radius * self.scale
        c = self.px(pos)
        sh = pygame.Surface((max(2, int(r * 2.6)), max(2, int(r * 1.4))), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 110), sh.get_rect())
        surface.blit(sh, (c[0] - sh.get_width() / 2, c[1] - sh.get_height() / 2 + r * 0.55))

    def _sprite(self, surface, img, pos, vel, bob: float = 0.0) -> None:
        speed = float(np.linalg.norm(vel)) if vel is not None else 0.0
        if speed > 0.5:  # lean into the run
            tilt = -_unit(vel)[0] * min(speed, 6) * 2.0
            img = pygame.transform.rotozoom(img, tilt, 1.0)
        c = self.px(pos) + [0, -bob]
        surface.blit(img, img.get_rect(center=(int(c[0]), int(c[1]))))

    def draw_chaser(self, surface, pos, vel=None, look=None, mood: str = "hunt", radius: float = 0.5,
                    t: float = 0.0, min_px: float = 0) -> None:
        """The robot. look: world direction it's looking (default: where it's moving)."""
        px = int(max(2 * radius * LOOK * self.scale, min_px))
        self._ground_shadow(surface, pos, radius)
        d = _unit(look if look is not None else (vel if vel is not None else (0, 0)), fallback=(0.0, 0.0))
        speed = float(np.linalg.norm(vel)) if vel is not None else 0.0
        bob = abs(np.sin(t * 14)) * px * 0.05 * min(speed / 3, 1)
        self._sprite(surface, _robot(px, _q(d), mood), pos, vel, bob)

    def draw_runner(self, surface, pos, vel=None, threat=None, scared: bool = False, radius: float = 0.5,
                    t: float = 0.0, min_px: float = 0) -> None:
        """The blob. threat: where it thinks the chaser is (it looks there)."""
        px = int(max(2 * radius * LOOK * self.scale, min_px))
        self._ground_shadow(surface, pos, radius)
        look = (np.asarray(threat) - pos) if threat is not None else vel
        d = _unit(look if look is not None else (0, 0), fallback=(0.0, 0.0))
        speed = float(np.linalg.norm(vel)) if vel is not None else 0.0
        bob = abs(np.sin(t * 16 + 1)) * px * 0.06 * min(speed / 3, 1)
        self._sprite(surface, _blob(px, _q(d), scared), pos, vel, bob)

    def draw_player(self, surface, pos, radius: float, color, vel=None, alpha: int = 255, min_px: float = 0) -> None:
        """A plain disc (small grid cells, ghosts)."""
        r = max(radius * self.scale, min_px / 2)
        c = self.px(pos)
        layer = pygame.Surface((int(2 * r + 4),) * 2, pygame.SRCALPHA)
        pygame.draw.circle(layer, (*color, alpha), (r + 2, r + 2), r)
        if alpha == 255:
            pygame.draw.circle(layer, INK, (r + 2, r + 2), r, max(1, int(r * 0.18)))
        surface.blit(layer, c - r - 2)

    def draw_bubble(self, surface, pos, text: str, color, age: float, radius: float = 0.5) -> None:
        """A pop-up bubble ("!" / "?") above a character; age: seconds since it appeared."""
        if age < 0 or age > 1.2:
            return
        k = T.ease_out(age / 0.18) * (1 - T.ease_in_out((age - 0.9) / 0.3))
        if k <= 0.01:
            return
        size = radius * LOOK * self.scale
        r = max(size * 0.7, 9) * k
        c = self.px(pos) + [size * 0.6, -size * 1.3]
        pygame.draw.circle(surface, INK, c, r + max(2, r * 0.15))
        pygame.draw.circle(surface, (250, 250, 252), c, r)
        T.text(surface, text, c, kind="display", size=max(8, int(r * 1.4)), color=color, anchor="center")

    def draw_tag(self, surface, pos, age: float) -> None:
        """The catch: an expanding ring and "TAG!"; age: seconds since the catch."""
        if age < 0 or age > 1.5:
            return
        c = self.px(pos)
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for k, col in ((0.0, T.TEXT), (0.12, CHASER_COLOR)):
            a = (age - k) / 0.6
            if 0 < a < 1:
                pygame.draw.circle(layer, (*col, int(255 * (1 - a))), c, self.scale * (0.8 + 4.5 * T.ease_out(a)),
                                   max(2, int(self.scale * 0.35 * (1 - a))))
        surface.blit(layer, (0, 0))
        s = T.ease_out(age / 0.25)
        fade = 1 - T.ease_in_out((age - 1.1) / 0.4)
        if fade > 0:
            T.text(surface, "TAG!", c + [0, -self.scale * 3.2], kind="display",
                   size=max(12, int(self.scale * 2.4 * s)), color=CHASER_COLOR, anchor="center", alpha=int(255 * fade))

    def draw_last_seen(self, surface, pos, radius: float, color, vel=None) -> None:
        """A dashed ring where the other player was last seen (with an arrow: which way it went)."""
        r = max(radius * LOOK * self.scale * 0.5, 5)
        c = self.px(pos)
        n = 12
        for k in range(0, n, 2):  # dashed
            a0, a1 = 2 * np.pi * k / n, 2 * np.pi * (k + 1) / n
            pygame.draw.arc(surface, color, pygame.Rect(c - r, (2 * r, 2 * r)), a0, a1, max(1, int(r * 0.18)))
        if vel is not None and np.linalg.norm(vel) > 0.5:
            d = _unit(vel)
            tip, base = c + d * r * 2.2, c + d * r * 1.3
            n_ = np.array([-d[1], d[0]])
            pygame.draw.polygon(surface, color, [tip, base + n_ * r * 0.5, base - n_ * r * 0.5])
        else:
            T.text(surface, "?", c, kind="bold", size=max(10, int(r * 1.3)), color=color, anchor="center")

    def draw_lasers(self, surface, pos, dirs, dists, max_range: float, alpha: int = 210) -> None:
        """One line per laser, coloured by how close the wall is (red = close)."""
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        start = self.px(pos)
        for d, dist in zip(np.asarray(dirs), dists):
            end = self.px(np.asarray(pos) + d * dist)
            color = T.mix(T.RED, T.GREEN, min(1.0, dist / (0.4 * max_range)))
            pygame.draw.line(layer, (*color, alpha), start, end, max(2, int(self.scale * 0.12)))
            if dist < max_range:
                pygame.draw.circle(layer, (*color, 255), end, max(3, int(self.scale * 0.3)))
        surface.blit(layer, (0, 0))

    def draw_trail(self, surface, points, color, alpha: int = 140, fade: bool = True) -> None:
        """The path so far; with fade, older parts thin out and disappear."""
        if len(points) < 2:
            return
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pts = self.px(points)
        n = len(pts)
        w = max(1.0, self.scale * 0.3)
        for i in range(1, n):
            k = i / (n - 1) if fade else 1.0
            a = int(alpha * k)
            if a < 4:
                continue
            width = max(1, int(w * (0.3 + 0.7 * k)))
            pygame.draw.line(layer, (*color, a), pts[i - 1], pts[i], width)
            if width > 2:
                pygame.draw.circle(layer, (*color, a), pts[i], width / 2)
        surface.blit(layer, (0, 0))
