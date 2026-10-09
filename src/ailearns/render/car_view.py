"""Drawing the race car game: track, cars, the lasers the AI sees, and trails.

Pure drawing: takes a track (from games.car.make_track) and car positions in
metres, never touches the simulation.
"""

import numpy as np
import pygame

from ailearns.render import theme as T

ASPHALT = (34, 38, 52)
KERB = (150, 158, 178)  # walls: bright enough to read at a glance
CAR_SIZE = (4.6, 2.0)  # metres, length x width


class TrackView:
    """A track fitted into `rect`; the static picture is drawn once and reused."""

    def __init__(self, track: dict, rect: pygame.Rect, margin: float = 0.06):
        self.track, self.rect = track, pygame.Rect(rect)
        pts = np.concatenate([track["left"], track["right"]])
        lo, hi = pts.min(0), pts.max(0)
        span = hi - lo
        self.scale = min(rect.w, rect.h) * (1 - 2 * margin) / max(span)
        self.offset = np.array(rect.center) - (lo + span / 2) * [self.scale, -self.scale]
        self.bg = pygame.Surface(rect.size, pygame.SRCALPHA)
        self._draw_static()

    def px(self, p) -> np.ndarray:
        """World metres -> screen pixels (y up in the world, down on screen)."""
        p = np.asarray(p, dtype=float)
        return p * [self.scale, -self.scale] + self.offset

    def _draw_static(self) -> None:
        o = np.array(self.rect.topleft)
        outer, inner = self.px(self.track["right"]) - o, self.px(self.track["left"]) - o
        pygame.draw.polygon(self.bg, ASPHALT, outer)
        pygame.draw.polygon(self.bg, (0, 0, 0, 0), inner)
        w = max(2, int(self.scale * 0.7))
        pygame.draw.lines(self.bg, KERB, True, outer, w)
        pygame.draw.lines(self.bg, KERB, True, inner, w)
        a, b = outer[0], inner[0]  # start/finish: a chequered line across the track at point 0
        for k in range(8):
            p0, p1 = a + (b - a) * k / 8, a + (b - a) * (k + 1) / 8
            pygame.draw.line(self.bg, T.TEXT if k % 2 == 0 else (5, 6, 10), p0, p1, max(3, int(self.scale * 1.2)))

    def follow(self, center, px_per_metre: float) -> None:
        """Chase camera: centre the view on `center` at a fixed zoom (the track is then drawn live)."""
        self.scale = px_per_metre
        self.offset = np.array(self.rect.center) - np.asarray(center) * [self.scale, -self.scale]
        self.live = True

    def draw_track(self, surface: pygame.Surface) -> None:
        if not getattr(self, "live", False):
            surface.blit(self.bg, self.rect.topleft)
            return
        clip = surface.get_clip()
        surface.set_clip(self.rect)
        outer, inner = self.px(self.track["right"]), self.px(self.track["left"])
        pygame.draw.polygon(surface, ASPHALT, outer)
        pygame.draw.polygon(surface, T.BG, inner)
        w = max(2, int(self.scale * 0.7))
        pygame.draw.lines(surface, KERB, True, outer, w)
        pygame.draw.lines(surface, KERB, True, inner, w)
        a, b = outer[0], inner[0]
        for k in range(8):
            p0, p1 = a + (b - a) * k / 8, a + (b - a) * (k + 1) / 8
            pygame.draw.line(surface, T.TEXT if k % 2 == 0 else (5, 6, 10), p0, p1, max(3, int(self.scale * 1.2)))
        surface.set_clip(clip)

    def draw_car(self, surface, pos, heading, color=T.GREEN, alpha: int = 255, outline=True,
                 min_px: float = 0) -> None:
        """min_px: draw the car at least this many pixels long (zoomed-out views), keeping its shape."""
        L, W = CAR_SIZE
        if min_px and L * self.scale < min_px:
            k = min_px / (L * self.scale)
            L, W = L * k, W * k
        c, s = np.cos(heading), np.sin(heading)
        f, r = np.array([c, s]), np.array([-s, c])
        corners = [pos + f * L / 2 + r * W / 2, pos + f * L / 2 - r * W / 2,
                   pos - f * L / 2 - r * W / 2, pos - f * L / 2 + r * W / 2]
        poly = self.px(corners)
        if alpha < 255:
            lo = poly.min(0).astype(int) - 2
            layer = pygame.Surface((poly.max(0) - lo + 4).astype(int), pygame.SRCALPHA)
            pygame.draw.polygon(layer, (*color, alpha), poly - lo)
            surface.blit(layer, lo)
            return
        pygame.draw.polygon(surface, color, poly)
        if outline:
            pygame.draw.polygon(surface, (5, 6, 10), poly, max(1, int(self.scale * 0.25)))
        glass = self.px([pos + f * L * 0.12 + r * W * 0.32, pos + f * L * 0.12 - r * W * 0.32,
                         pos - f * L * 0.05 - r * W * 0.32, pos - f * L * 0.05 + r * W * 0.32])
        pygame.draw.polygon(surface, (5, 6, 10), glass)

    def draw_lasers(self, surface, pos, heading, angles_deg, dists, max_range: float, alpha: int = 200) -> None:
        """One line per laser, coloured by how close the wall is (red = close)."""
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        start = self.px(pos)
        for a, d in zip(np.deg2rad(angles_deg), dists):
            end = self.px(pos + d * np.array([np.cos(heading + a), np.sin(heading + a)]))
            color = T.mix(T.RED, T.GREEN, min(1.0, d / (0.5 * max_range)))
            pygame.draw.line(layer, (*color, alpha), start, end, max(1, int(self.scale * 0.3)))
            if d < max_range:
                pygame.draw.circle(layer, (*color, 255), end, max(3, int(self.scale * 0.6)))
        surface.blit(layer, (0, 0))

    def draw_trail(self, surface, points, color=T.GREEN, alpha: int = 120) -> None:
        if len(points) < 2:
            return
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.lines(layer, (*color, alpha), False, self.px(points), max(1, int(self.scale * 0.35)))
        surface.blit(layer, (0, 0))
