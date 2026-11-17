"""Drawing a booster flight in 2D (episode 18): a camera that follows the booster, the pad or the drone ship, and a
side panel of the numbers that matter (height, speed, fuel, engines, heat and air load against their limits).

Pure drawing: takes a recorded flight (games.rocket_flight.record) and a time, never touches the simulation.
"""

import numpy as np
import pygame

from ailearns.render import theme as T

PANEL_W = 460
BEND_REF = 800e3  # (12m) Pa.deg: games.rocket.BEND_LIMIT, shown for flights made before it existed
SKY_LOW = (30, 52, 84)  # sky at the ground; fades to BG (space) with height
SEA = (16, 36, 58)
SEA_LINE = (40, 74, 110)
GROUND = (34, 38, 46)
CONCRETE = (92, 98, 110)
DECK = (70, 76, 88)
BODY = (236, 238, 242)
SOOT = (58, 54, 52)  # the engine end goes dark after entry
LEGS = (40, 42, 50)  # interstage, ship hull
LEG_GREY = (150, 156, 170)  # landing legs: light enough to read against the sea
FLAME_CORE = (255, 236, 190)
SHIP_W = 52.0  # m, the deck drawn (the sim's landing box is smaller: the legs must stay on it)


def at(rec: dict, t: float) -> dict:
    """The flight at time t (s), linearly between recorded steps; past the end it holds the last step."""
    ts = rec["time"]
    i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
    f = float(np.clip((t - ts[i]) / (ts[i + 1] - ts[i]), 0.0, 1.0))
    s = {k: rec[k][i] + (rec[k][i + 1] - rec[k][i]) * f for k in ("x", "y", "vx", "vy", "tilt", "thrust", "gimbal",
                                                                    "fin", "fuel", "heat", "q", "deck_y", "wind")}
    for k in ("engines", "fin_state"):
        s[k] = rec[k][i + (f > 0.5)]
    s["time"], s["i"] = t, i
    # (12m) the angle between the body and the airflow, and the bending load q x angle (Pa.deg)
    rvx = s["vx"] - s["wind"]
    st, ct = np.sin(s["tilt"]), np.cos(s["tilt"])
    s["alpha"] = float(np.degrees(np.arctan2(abs(rvx * ct - s["vy"] * st), abs(rvx * st + s["vy"] * ct))))
    s["bend"] = s["q"] * s["alpha"]
    return s


class RocketView:
    """One flight on a 1920x1080 canvas: the view on the left, the numbers panel on the right."""

    def __init__(self, rec: dict, label: str = "", size=(T.WIDTH, T.HEIGHT)):
        self.rec, self.label = rec, label
        self.w, self.h = size
        self.pw = PANEL_W if self.w >= 1500 else 330  # (12p) half-width views (side by side) get a narrower panel
        self.view = pygame.Rect(0, 0, self.w - self.pw, self.h)
        self.airflow = False  # (12p) draw the airflow line and the angle off it (the swing, ep18 twist)
        self.fuel0 = float(rec["fuel"][0])
        self.end = float(rec["time"][-1])
        self.rng = np.random.default_rng(int(rec["seed"]))
        self.stars = self.rng.random((160, 2))

    # -- camera ---------------------------------------------------------------
    def scale(self, y: float) -> float:
        """Metres per pixel: close in near the deck, further out higher up (the booster stays ~120-250 px tall)."""
        return float(np.clip(0.17 + (y - 100.0) / 4000.0, 0.17, 0.30))

    def camera(self, s: dict):
        m = self.scale(s["y"])
        cy = max(s["y"] + self.rec["length"] * 0.45, self.h * 0.36 * m)  # low down, keep the deck in the bottom
        return s["x"], cy, m

    def px(self, cam, x, y):
        cx, cy, m = cam
        return (self.view.centerx + (x - cx) / m, self.view.centery - (y - cy) / m)

    # -- drawing --------------------------------------------------------------
    def frame(self, surf: pygame.Surface, t: float) -> None:
        s = at(self.rec, t)
        cam = self.camera(s)
        surf.set_clip(self.view)
        self.sky(surf, s, cam)
        self.surface(surf, s, cam)
        self.trail(surf, s, cam)
        after = t - self.end
        if self.rec["outcome"] == "crashed" and after > 0:
            self.explosion(surf, s, cam, after)
        else:
            self.booster(surf, s, cam)
            if self.airflow:
                self.flow(surf, s, cam)
        surf.set_clip(None)
        self.panel(surf, s, after)

    def sky(self, surf, s, cam):
        y = s["y"]
        k = float(np.clip(np.log1p(y / 800.0) / np.log1p(70e3 / 800.0), 0, 1))  # 0 at the ground, 1 at 70 km
        top, low = T.mix(SKY_LOW, T.BG, min(1.0, k + 0.35)), T.mix(SKY_LOW, T.BG, k)
        for r in range(0, self.view.h, 8):
            pygame.draw.rect(surf, T.mix(top, low, r / self.view.h), (0, r, self.view.w, 8))
        if k > 0.5:  # stars come out high up
            a = (k - 0.5) * 2
            for sx, sy in self.stars:
                pygame.draw.circle(surf, T.mix(T.BG, T.MUTED, a), (sx * self.view.w, sy * self.view.h), 1)
        # speed streaks: marks fixed in the air (moving with the wind), so they stream past at the booster's speed
        cx, cy, m = cam
        step = 60.0 * m / 0.17
        speed = np.hypot(s["vx"], s["vy"] - 0.0)
        if speed > 5:
            ux, uy = s["vx"] / speed, s["vy"] / speed
            tail = min(speed * 0.05, 140.0) / m * 0.9  # px: what one 20th of a second of flight sweeps
            colour = T.mix(T.BG, T.TEXT, 0.18 * (1 - k) + 0.1)
            x0, y0 = np.floor((cx - self.view.w / 2 * m) / step), np.floor((cy - self.view.h / 2 * m) / step)
            for i in range(int(self.view.w * m / step) + 2):
                for j in range(int(self.view.h * m / step) + 2):
                    wx, wy = (x0 + i) * step, (y0 + j) * step
                    jx, jy = np.sin(wx * 12.9898 + wy * 78.233) * 0.5, np.cos(wx * 4.1 + wy * 9.7) * 0.5
                    p = self.px(cam, wx + jx * step, wy + jy * step)
                    if wy + jy * step > 2:
                        pygame.draw.line(surf, colour, p, (p[0] + ux * tail, p[1] - uy * tail), 1)

    def surface(self, surf, s, cam):
        ground_y = self.px(cam, 0, 0)[1]
        if ground_y > self.view.h + 5:
            return
        if self.rec["ship"]:
            pygame.draw.rect(surf, SEA, (0, ground_y, self.view.w, self.view.h))
            t = s["time"]
            for k in range(6):  # swell lines
                yy = ground_y + 10 + k * 22
                pts = [(x, yy + 4 * np.sin(x / 60.0 + t * 1.3 + k)) for x in range(0, self.view.w + 20, 20)]
                pygame.draw.lines(surf, SEA_LINE, False, pts, 1)
            d = s["deck_y"]
            l, r = self.px(cam, -SHIP_W / 2, d), self.px(cam, SHIP_W / 2, d)
            hull = self.px(cam, SHIP_W / 2 + 4, d - 9)
            pygame.draw.polygon(surf, LEGS, [(l[0] - 4 / cam[2], l[1]), (r[0] + 4 / cam[2], r[1]), (hull[0], hull[1]),
                                             (l[0] - (hull[0] - r[0]), hull[1])])
            pygame.draw.rect(surf, DECK, (l[0], l[1] - 3, r[0] - l[0], 6))
            pad = self.rec["pad_radius"]
            a, b = self.px(cam, -pad, d), self.px(cam, pad, d)
            pygame.draw.line(surf, T.AMBER, (a[0], a[1] - 3), (b[0], b[1] - 3), 2)
        else:
            pygame.draw.rect(surf, GROUND, (0, ground_y, self.view.w, self.view.h))
            pad = self.rec["pad_radius"]
            a, b = self.px(cam, -pad, 0), self.px(cam, pad, 0)
            pygame.draw.rect(surf, CONCRETE, (a[0], a[1] - 2, b[0] - a[0], 8))
            c = self.px(cam, 0, 0)
            pygame.draw.line(surf, T.AMBER, (c[0] - 30, c[1]), (c[0] + 30, c[1]), 3)
        c = self.px(cam, 0, s["deck_y"])
        if not (0 <= c[0] <= self.view.w):  # the pad is off screen: say where, and how far
            right = c[0] > self.view.w
            yy = min(max(c[1] - 40, 60), self.view.h - 60)
            msg = f"PAD {abs(s['x']):.0f} m " + ("→" if right else "←")
            T.text(surf, msg if right else "← " + msg[:-2], (self.view.w - 30 if right else 30, yy), "bold", 30,
                   T.AMBER, anchor="midright" if right else "midleft")

    def trail(self, surf, s, cam):
        """Exhaust left behind while lit: grey puffs at past positions (faint, thinning with height)."""
        i, rec = s["i"], self.rec
        lo = max(0, i - 120)
        for j in range(lo, i, 3):
            if rec["engines"][j] > 0:
                a = (j - lo) / max(1, i - lo)
                p = self.px(cam, rec["x"][j], rec["y"][j])
                r = (4 + 10 * (1 - a)) * (0.5 + 0.5 * rec["engines"][j] / 3) / (cam[2] / 0.17)
                pygame.draw.circle(surf, T.mix(T.BG, T.MUTED, 0.25 + 0.3 * a), p, max(1.0, r))

    def booster(self, surf, s, cam):
        m, L, D = cam[2], self.rec["length"], self.rec["diameter"]
        tilt = s["tilt"]
        up = np.array([np.sin(tilt), np.cos(tilt)])  # world, y up
        side = np.array([np.cos(tilt), -np.sin(tilt)])
        base = np.array([s["x"], s["y"]])

        def P(v):
            return self.px(cam, v[0], v[1])
        # flame first (behind the body)
        if s["engines"] > 0 and s["thrust"] > 0:
            g = s["gimbal"]
            down = -(up * np.cos(g) + side * np.sin(g))
            n = int(s["engines"])
            power = float(np.clip(s["thrust"] / (845e3 * n), 0.3, 1.2))
            length = (11.0 + 14.0 * power) * (1.0 + 0.6 * (n > 1)) * (1 + 0.25 * np.sin(s["time"] * 47))
            thin = float(np.clip(s["y"] / 50e3, 0, 1))  # the plume balloons in thin air (entry burns high up)
            spread = D * (0.32 if n == 1 else 0.5) * (1 + 2.5 * thin)
            tip = base + down * length
            for k, col in ((1.0, T.ORANGE), (0.55, T.AMBER), (0.25, FLAME_CORE)):
                pts = [P(base + side * spread * k), P(base + down * length * 0.25 * k + side * spread * k * 1.3),
                       P(base + (tip - base) * k), P(base + down * length * 0.25 * k - side * spread * k * 1.3),
                       P(base - side * spread * k)]
                pygame.draw.polygon(surf, col, pts)
            T.glow(surf, P(base + down * 2), 26 / (m / 0.17) * (1 + 0.5 * (n > 1)), T.ORANGE, strength=50)
        # re-entry glow on the engine end: heating rate ~ the air load times speed
        hot = float(np.clip(s["q"] * np.hypot(s["vx"], s["vy"]) / 3e7, 0, 1))
        if hot > 0.05:
            T.glow(surf, P(base - up * 1.5), (18 + 30 * hot) / (m / 0.17), T.ORANGE, strength=int(40 + 100 * hot))
        # legs: stowed along the body until the last few hundred metres, then spread (with the real timing)
        open_ = float(np.clip((400.0 - (s["y"] - s["deck_y"])) / 250.0, 0, 1))
        for k in (-1, 1):
            hinge = base + up * 8.0 + side * D / 2 * k
            stowed = base + up * 1.0 + side * D / 2 * k
            spread = base + side * (D / 2 + 4.6) * k - up * 0.6
            foot = stowed + (spread - stowed) * open_
            pygame.draw.line(surf, LEG_GREY, P(hinge), P(foot), max(3, int(1.1 / m)))
        # body: white with a soot-dark engine end once it has flown through entry
        soot = float(np.clip(s["heat"] / 6e5, 0, 1))
        corners = lambda a, b: [P(base + up * a + side * D / 2), P(base + up * b + side * D / 2),  # noqa: E731
                                P(base + up * b - side * D / 2), P(base + up * a - side * D / 2)]
        pygame.draw.polygon(surf, BODY, corners(0, L))
        if soot > 0:
            pygame.draw.polygon(surf, T.mix(BODY, SOOT, 0.7 * soot), corners(0, 3 + 9 * soot))
        pygame.draw.polygon(surf, LEGS, corners(L - 4.5, L))  # the interstage
        if self.rec["fins"]:
            colour = ((200, 204, 214), T.RED, T.MUTED)[int(s["fin_state"])]
            at_ = base + up * (L - 6.0)
            vane = np.array([np.sin(tilt - s["fin"]), np.cos(tilt - s["fin"])]) * 1.6
            for k in (-1, 1):
                c = at_ + side * (D / 2 + 1.2) * k
                pygame.draw.line(surf, colour, P(c - vane), P(c + vane), max(2, int(0.7 / m)))

    def flow(self, surf, s, cam):
        """(12p) The airflow past the booster (a dashed line along its motion through the air) and the angle between
        it and the body: a booster falling straight keeps it near 0."""
        rvx, vy = s["vx"] - s["wind"], s["vy"]
        sp = np.hypot(rvx, vy)
        if sp < 30 or s["y"] - s["deck_y"] < 50:
            return
        d = np.array([rvx, vy]) / sp  # direction of motion
        mid = np.array([s["x"], s["y"]]) + np.array([np.sin(s["tilt"]), np.cos(s["tilt"])]) * self.rec["length"] / 2
        for k in range(-7, 8, 2):  # dashes, from ahead of the booster to behind it
            a, b = mid + d * k * 6.0, mid + d * (k + 1) * 6.0
            pygame.draw.line(surf, T.TEXT, self.px(cam, *a), self.px(cam, *b), 3)
        col = T.GREEN if s["alpha"] < 10 else T.AMBER if s["alpha"] < 20 else T.RED
        p = self.px(cam, *(mid + np.array([abs(d[1]), 0.0]) * 30))  # well to the right of the booster
        T.text(surf, f"{s['alpha']:.0f}°", (p[0], p[1] - 18), "bold", 40, col, anchor="midleft")
        T.text(surf, "off the airflow", (p[0], p[1] + 18), "semibold", 22, T.MUTED, anchor="midleft")
        if self.rec.get("entry", False):  # (12s) the bending load, big enough for a phone
            bl = float(self.rec.get("bend_limit", 0.0)) or BEND_REF
            bc = T.RED if s["bend"] > bl else T.AMBER if s["bend"] > 0.75 * bl else T.PURPLE
            T.text(surf, f"{s['bend'] / 1000:,.0f}", (p[0], p[1] + 70), "bold", 40, bc, anchor="midleft")
            T.text(surf, f"bending (limit {bl / 1000:.0f})", (p[0], p[1] + 106), "semibold", 22, T.MUTED,
                   anchor="midleft")

    def explosion(self, surf, s, cam, after):
        """Puffs thrown up and out; each cools white-hot -> amber -> orange -> red -> smoke and fades as it rises."""
        zoom = 0.17 / cam[2]
        p = np.array(self.px(cam, s["x"], s["y"] + 4))
        R = 150 * zoom
        layer = pygame.Surface((int(R * 6), int(R * 6)), pygame.SRCALPHA)
        o = np.array(layer.get_size()) / 2
        ramp = [FLAME_CORE, T.AMBER, T.ORANGE, T.RED, T.mix(T.BG, T.MUTED, 0.5)]
        rng = np.random.default_rng(3)
        puffs = []
        for _ in range(28):
            ang = rng.uniform(0.05, 0.95) * np.pi
            u = rng.uniform(0.0, 1.0) ** 0.7
            spread = R * u * (0.3 + 1.1 * T.ease_out(min(after / 0.9, 1)))
            c = o + [np.cos(ang) * spread, -np.sin(ang) * spread * 0.8 - R * 0.35 * min(after, 3) / 3]
            size = R * rng.uniform(0.28, 0.5) * (0.6 + 0.6 * min(after / 1.5, 1))
            age = float(np.clip(after * 1.6 + u * 0.9, 0, 4))  # outer puffs cool first
            a, f = int(age), age - int(age)
            col = T.mix(ramp[a], ramp[min(a + 1, 4)], f)
            alpha = int(235 * min(1.0, after * 8) * max(0.0, 1 - max(0.0, after - 1.5) / 2.5))
            puffs.append((age, c, size, col, alpha))
        for age, c, size, col, alpha in sorted(puffs, key=lambda q: -q[0]):  # smoke behind, fire in front
            pygame.draw.circle(layer, (*col, alpha), c, size)
        surf.blit(layer, p - o)
        hot = max(0.0, 1 - after / 0.8)
        if hot > 0:
            T.glow(surf, p, R * 1.8, T.ORANGE, strength=int(150 * hot))

    # -- the numbers ------------------------------------------------------------
    def panel(self, surf, s, after):
        x0 = self.view.right
        pygame.draw.rect(surf, T.BG, (x0, 0, self.pw, self.h))
        pygame.draw.line(surf, T.PANEL_EDGE, (x0, 0), (x0, self.h), 2)
        x, y = x0 + 40, 48
        if self.label:
            T.text(surf, self.label, (x, y), "bold", 34)
            y += 64
        alt = 0.0 if (after > 0 and self.rec["outcome"] == "landed") else s["y"] - s["deck_y"]
        speed = np.hypot(s["vx"], s["vy"])
        rows = [("HEIGHT", f"{alt / 1000:.1f} km" if alt >= 1000 else f"{max(alt, 0):.0f} m"),
                ("SPEED", f"{speed:.0f} m/s" if speed >= 10 else f"{speed:.1f} m/s"),
                ("TIME", f"{min(s['time'], self.end):.1f} s")]
        for name, val in rows:
            T.text(surf, name, (x, y), "semibold", 22, T.MUTED)
            T.text(surf, val, (x, y + 26), "bold", 46)
            y += 92
        # engines: three bells, lit ones orange
        T.text(surf, "ENGINES", (x, y), "semibold", 22, T.MUTED)
        for k in range(3):
            on = k < s["engines"] and s["thrust"] > 0
            c = (x + 22 + k * 52, y + 52)
            if on:
                T.glow(surf, c, 26, T.ORANGE, strength=70)
            pygame.draw.circle(surf, T.ORANGE if on else T.PANEL_EDGE, c, 18)
        y += 100
        self.bar(surf, x, y, "FUEL", s["fuel"] / max(self.fuel0, 1), f"{s['fuel'] / 1000:.1f} t", T.BLUE)
        y += 78
        if self.rec.get("entry", False):
            hl, ql = self.rec["heat_limit"], self.rec["q_limit"]  # past these it breaks up
            self.bar(surf, x, y, "HEAT", s["heat"] / hl, f"{s['heat'] / 1e6:.2f} / {hl / 1e6:.1f} MJ/m²", T.ORANGE)
            y += 78
            self.bar(surf, x, y, "AIR LOAD", s["q"] / ql, f"{s['q'] / 1000:.0f} / {ql / 1000:.0f} kPa", T.AMBER)
            y += 78
            bl = float(self.rec.get("bend_limit", 0.0)) or BEND_REF  # (12m) before the limit: against the real one
            self.bar(surf, x, y, "BENDING", s["bend"] / bl, f"{s['bend'] / 1000:,.0f} / {bl / 1000:.0f}", T.PURPLE)
            y += 78
        if abs(s["wind"]) > 0.5:
            T.text(surf, "WIND", (x, y), "semibold", 22, T.MUTED)
            T.text(surf, f"{abs(s['wind']):.0f} m/s {'→' if s['wind'] > 0 else '←'}", (x, y + 26), "bold", 36)
            y += 80
        if after > 0:  # the verdict, once it's down
            ok = self.rec["outcome"] == "landed"
            r = pygame.Rect(x, self.h - 150, self.pw - 80, 100)
            T.panel(surf, r, 18, T.GREEN_DARK if ok else (90, 20, 36), T.GREEN if ok else T.RED)
            vy = abs(self.rec["vy"][-1] - 0.0)
            verdict = "LANDED" if ok else self.rec["outcome"].upper()
            last = at(self.rec, self.end)
            if not ok and last["y"] - last["deck_y"] > 50:  # (12s) it came apart in the air, not on the deck
                bl = float(self.rec.get("bend_limit", 0.0))
                verdict = "BUCKLED" if bl and last["bend"] > bl else "BROKE UP"
            T.text(surf, verdict, (r.centerx, r.y + 14), "bold", 40, anchor="midtop")
            T.text(surf, f"{vy:.1f} m/s, {abs(self.rec['x'][-1]):.0f} m off", (r.centerx, r.y + 62), "semibold", 24,
                   anchor="midtop")

    def bar(self, surf, x, y, name, frac, val, colour):
        T.text(surf, name, (x, y), "semibold", 22, T.MUTED)
        T.text(surf, val, (x + self.pw - 80, y), "semibold", 22, anchor="topright")
        w = self.pw - 80
        pygame.draw.rect(surf, T.PANEL, (x, y + 32, w, 18), border_radius=9)
        f = float(np.clip(frac, 0, 1))
        if f > 0:
            pygame.draw.rect(surf, T.RED if frac > 0.85 and name != "FUEL" else colour, (x, y + 32, max(18, w * f), 18),
                             border_radius=9)
