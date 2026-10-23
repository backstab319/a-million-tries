"""Drawing the racing sandbox (episode 14), top-down: circuits and cars built from their design numbers.

Pure drawing: takes a circuit (games.race.make_track) and recorded car states, never touches the simulation.
The car's picture follows its design: body length from the wheelbase, tyre widths per axle, wing sizes, where the
engine sits; sliding tyres glow red and a car on its skid plank throws sparks.
"""

import numpy as np
import pygame

from ailearns.games import race_car as rc
from ailearns.render import theme as T

TARMAC = (40, 44, 58)
GRASS = (22, 54, 38)
WALL = (150, 158, 178)
KERB_A, KERB_B = T.RED, T.TEXT
LINE = (5, 6, 10)
BODY_W = 0.9  # metres: half the body's width (between the tyres)


class CircuitView:
    """A circuit seen from above, either fitted into `rect` (overview) or followed by a camera."""

    def __init__(self, track: dict, cfg, rect: pygame.Rect, margin: float = 0.05):
        self.track, self.cfg, self.rect = track, cfg, pygame.Rect(rect)
        pts = np.concatenate([track["wall_left"], track["wall_right"]])
        lo, hi = pts.min(0), pts.max(0)
        self.fit_scale = min(rect.w, rect.h) * (1 - 2 * margin) / max(hi - lo)
        self.fit_offset = np.array(self.rect.center) - (lo + (hi - lo) / 2) * [self.fit_scale, -self.fit_scale]
        self.scale, self.offset = self.fit_scale, self.fit_offset
        n, c = track["normal"], track["center"]
        half = cfg.width / 2
        self.edges = {name: c + n * off for name, off in (
            ("wall_l", cfg.edge), ("wall_r", -cfg.edge), ("tar_l", half), ("tar_r", -half),
            ("kerb_l", half - cfg.kerb), ("kerb_r", -(half - cfg.kerb)))}

    def px(self, p) -> np.ndarray:
        """World metres -> screen pixels (y up in the world, down on screen)."""
        return np.asarray(p, dtype=float) * [self.scale, -self.scale] + self.offset

    def fit(self) -> None:
        self.scale, self.offset = self.fit_scale, self.fit_offset

    def follow(self, center, px_per_metre: float) -> None:
        self.scale = px_per_metre
        self.offset = np.array(self.rect.center) - np.asarray(center) * [self.scale, -self.scale]

    def _visible(self, pts):
        """Indices of centreline points near the view (so a zoomed-in camera doesn't draw the whole circuit)."""
        scr = self.px(self.track["center"])
        r = self.rect.inflate(self.rect.w, self.rect.h)
        ok = (scr[:, 0] > r.left) & (scr[:, 0] < r.right) & (scr[:, 1] > r.top) & (scr[:, 1] < r.bottom)
        return ok

    def _band(self, surface, a, b, color, ok):
        """Fill the strip between edge polylines a and b, one quad per visible segment."""
        P = len(a)
        pa, pb = self.px(a), self.px(b)
        for i in np.nonzero(ok)[0]:
            j = (i + 1) % P
            pygame.draw.polygon(surface, color, [pa[i], pa[j], pb[j], pb[i]])

    def draw(self, surface: pygame.Surface) -> None:
        clip = surface.get_clip()
        surface.set_clip(self.rect)
        e = self.edges
        ok = self._visible(self.track["center"])
        self._band(surface, e["wall_l"], e["wall_r"], GRASS, ok)
        self._band(surface, e["tar_l"], e["tar_r"], TARMAC, ok)
        if self.scale > 1.5:  # kerbs: red and white stripes, only worth drawing up close
            P = len(self.track["center"])
            for side in ("l", "r"):
                pa, pb = self.px(e[f"tar_{side}"]), self.px(e[f"kerb_{side}"])
                for i in np.nonzero(ok)[0]:
                    j = (i + 1) % P
                    pygame.draw.polygon(surface, KERB_A if i % 2 else KERB_B, [pa[i], pa[j], pb[j], pb[i]])
        w = max(1, int(self.scale * 0.6))
        for side in ("wall_l", "wall_r"):
            pygame.draw.lines(surface, WALL, True, self.px(e[side]), w)
        a, b = self.px(e["tar_l"][0]), self.px(e["tar_r"][0])  # start/finish line
        for k in range(8):
            p0, p1 = a + (b - a) * k / 8, a + (b - a) * (k + 1) / 8
            pygame.draw.line(surface, T.TEXT if k % 2 == 0 else LINE, p0, p1, max(2, int(self.scale * 1.2)))
        surface.set_clip(clip)

    def draw_trail(self, surface, points, color=T.GREEN, alpha: int = 110) -> None:
        if len(points) < 2:
            return
        layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.lines(layer, (*color, alpha), False, self.px(points), max(1, int(self.scale * 0.35)))
        surface.blit(layer, (0, 0))

    def draw_car(self, surface, design, pos, yaw, steer=0.0, sliding=None, plank=None, color=T.GREEN,
                 spec: rc.CarSpec = rc.RACE, min_px: float = 0.0, rng=None) -> None:
        """The car as its design makes it. min_px: draw at least this long (overview maps), keeping the shape."""
        d = np.asarray(design, dtype=float)
        L = d[rc.D["wheelbase"]]
        a = float(L * (1 - rc.weight_front(d, spec)))  # front axle -> centre of mass: follows the parts
        length = L + spec.overhang_front + spec.overhang_rear
        k = max(1.0, min_px / (length * self.scale)) if min_px else 1.0
        c, s = np.cos(yaw), np.sin(yaw)
        fwd, left = np.array([c, s]), np.array([-s, c])

        def at(x, y):  # car frame (x forward from the front axle, y left) -> world
            return pos + (fwd * (a - x) + left * y) * k

        def quad(x0, x1, y0, y1, col, ang=0.0, cx=None):
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
            if ang:  # steered front tyre: rotate about its centre
                ctr = np.array([cx, (y0 + y1) / 2])
                rot = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
                pts = (pts - ctr) @ rot.T + ctr
            pygame.draw.polygon(surface, col, self.px([at(x, y) for x, y in pts]))

        tw = spec.track_width / 2
        tyre_len = 0.66
        for i, (x, y, mm) in enumerate(((0.0, tw, d[rc.D["tyre_front_mm"]]), (0.0, -tw, d[rc.D["tyre_front_mm"]]),
                                        (L, tw, d[rc.D["tyre_rear_mm"]]), (L, -tw, d[rc.D["tyre_rear_mm"]]))):
            half_w = mm / 2000.0
            slid = sliding is not None and bool(sliding[i])
            col = T.RED if slid else (18, 20, 26)
            quad(x - tyre_len / 2, x + tyre_len / 2, y - half_w, y + half_w, col,
                 ang=float(steer) if i < 2 else 0.0, cx=x)
        quad(-spec.overhang_front * 0.6, L + spec.overhang_rear * 0.5, -BODY_W * 0.55, BODY_W * 0.55, color)
        quad(0.15 * L, 0.85 * L, -BODY_W, BODY_W, color)   # sidepods
        ex = L * (-0.05 + 1.15 * d[rc.D["engine_pos"]])    # engine cover where the engine is
        size = 0.25 + 0.5 * (d[rc.D["power_kw"]] - 75) / 675
        quad(ex - size, ex + size, -0.35, 0.35, T.mix(color, LINE, 0.45))
        quad(0.35 * L, 0.5 * L, -0.28, 0.28, LINE)          # cockpit
        for wing, x, span in ((d[rc.D["wing_front"]], -spec.overhang_front, 1.0),
                              (d[rc.D["wing_rear"]], L + spec.overhang_rear, 0.75)):
            if wing > 0.02:
                chord = 0.12 + 0.35 * wing
                quad(x - chord / 2, x + chord / 2, -span * (0.5 + 0.5 * wing), span * (0.5 + 0.5 * wing), T.TEXT)
        if plank is not None and float(np.sum(plank)) > 0:  # sparks off the skid plank
            rng = rng or np.random.default_rng()
            for _ in range(int(min(30, 4 + np.sum(plank) / 400))):
                p0 = at(L * rng.uniform(0.3, 1.0), rng.uniform(-0.3, 0.3))
                p1 = p0 - fwd * rng.uniform(1.0, 4.0) * k + left * rng.normal(0, 0.6) * k
                pygame.draw.line(surface, T.mix(T.AMBER, T.ORANGE, rng.uniform()), self.px(p0), self.px(p1), 2)
