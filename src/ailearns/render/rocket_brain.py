"""The booster's brain, live (episode 18 cold open): every neuron of a PPO pilot lighting up as it flies.

Senses on the left (named in plain words, with a bar for each value), two hidden layers of neurons in the middle,
the controls on the right. Each frame re-runs the network on what the booster sensed at that moment (the flight
recorder keeps `obs` per step), so what lights up is exactly what flew. Only the strongest links are drawn: a link's
strength is (what flows in) x (its weight), green when it pushes a neuron up, blue when it pulls it down.

Pure drawing: a recorded flight (games.rocket_flight.record) + a checkpoint, never touches the simulation.
"""

import pickle
from pathlib import Path

import numpy as np
import pygame

from ailearns.render import theme as T

UP, DOWN = T.GREEN, T.BLUE  # positive / negative: a neuron firing or a link pushing up v pulling down
EDGE_TOP = (90, 140, 70)  # strongest links drawn per layer gap: senses->h1, h1->h2, h2->controls
SMOOTH = 0.35  # how fast the shown values follow the real ones per frame (the net is jumpy step to step)

BASE = ["drift from the deck", "height above deck", "sideways speed", "falling speed", "lean (sin)", "lean (cos)",
        "spin", "fuel left", "thrust", "engine angle", "engine lit", "lights left", "drift (close up)",
        "height (close up)", "sideways (close up)", "falling (close up)", "wind", "deck heave"]
FINS, FLUID, FAULTS = ["grid fins"], ["fin fluid"], ["fins stuck", "engine stuck"]
ENTRY = ["altitude (km scale)", "fall speed (km scale)", "side speed (km scale)", "downrange", "Mach",
         "peak air load", "heat soaked", "start sequence"]
CONTROLS = ["engine", "engine angle", "thrusters", "grid fins"]


def sense_names(rec: dict, n: int) -> list[str]:
    """Names for the first n senses, in the order games.rocket.Boosters.obs stacks them."""
    names = BASE + (FINS if rec["fins"] else [])
    rest = n - len(names) - (len(ENTRY) if rec["entry"] else 0)
    names += (FLUID if rest in (1, 3) else []) + (FAULTS if rest >= 2 else [])
    names += ENTRY if rec["entry"] else []
    return (names + [f"sense {i}" for i in range(len(names), n)])[:n]


class Brain:
    """A checkpoint's policy network as numpy, returning every layer's values."""

    def __init__(self, checkpoint):
        c = pickle.loads(Path(checkpoint).read_bytes())
        p = {k: np.asarray(v, np.float64) for k, v in c["params"].items()}
        n = len(c["config"]["hidden"])
        self.w = [p[f"pi{i}.w"] for i in range(n)] + [p["pi.out.w"]]
        self.b = [p[f"pi{i}.b"] for i in range(n)] + [p["pi.out.b"]]
        self.n_in = self.w[0].shape[0]

    def layers(self, obs: np.ndarray) -> list[np.ndarray]:
        """[senses, hidden 1, ..., controls (clipped to -1..1 as the sim does)]."""
        x = [np.asarray(obs[: self.n_in], np.float64)]
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            z = x[-1] @ w + b
            x.append(np.tanh(z) if i < len(self.w) - 1 else np.clip(z, -1, 1))
        return x


def tint(color, v: float, floor=0.12):
    """A colour at strength |v| (0..1) over the background."""
    return T.mix(T.BG, color, floor + (1 - floor) * min(abs(v), 1.0))


class BrainView:
    """The network on a canvas (default 1920x1080, fills it)."""

    def __init__(self, rec: dict, checkpoint, label: str = "", size=(T.WIDTH, T.HEIGHT), controls=None):
        """controls: (top, bottom) px for the control dots (default: the middle of the canvas); leave the bottom right
        free for an inset of the flight."""
        self.rec, self.label = rec, label
        self.net = Brain(checkpoint)
        self.w, self.h = size
        self.names = sense_names(rec, self.net.n_in)
        widths = [self.net.n_in] + [len(b) for b in self.net.b]
        self.n_out = widths[-1]
        top, bottom = 150, self.h - 70
        self.narrow = self.w < 1500  # beside the flight (--beside): a tighter layout
        xs = [370, 575, 750, 860] if self.narrow else [430, 830, 1190, 1450]  # senses, hidden 1, hidden 2, controls
        self.pos = []
        for k, n in enumerate(widths):
            if k == len(widths) - 1:  # the few controls: spread out, big
                ys = np.linspace(*(controls or (top + 160, bottom - 160)), n)
            else:
                ys = np.linspace(top, bottom, n)
            self.pos.append(np.stack([np.full(n, xs[k]), ys], 1))
        self.shown = None  # smoothed layer values
        self.edges = pygame.Surface(size)

    def values(self, t: float) -> list[np.ndarray]:
        ts = self.rec["time"]
        self.i = i = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1))
        now = self.net.layers(self.rec["obs"][i])
        if self.shown is None:
            self.shown = now
        else:
            self.shown = [s + (n - s) * SMOOTH for s, n in zip(self.shown, now)]
        return self.shown

    def frame(self, surf: pygame.Surface, t: float) -> None:
        x = self.values(t)
        surf.fill(T.BG)
        self.links(surf, x)
        self.neurons(surf, x)
        self.header(surf, t)

    # -- parts ----------------------------------------------------------------
    def links(self, surf, x):
        self.edges.fill((0, 0, 0))
        for k, top in enumerate(EDGE_TOP):
            flow = x[k][:, None] * self.net.w[k]  # (from, to): what each link carries now
            flat = np.abs(flow).ravel()
            idx = np.argpartition(flat, -top)[-top:]
            peak = max(float(flat[idx].max()), 1e-9)
            a, b = self.pos[k], self.pos[k + 1]
            for j in idx[np.argsort(flat[idx])]:  # weakest first, strongest on top
                i, o = divmod(int(j), flow.shape[1])
                v = flat[j] / peak
                col = tuple(int(c * (0.15 + 0.85 * v)) for c in (UP if flow[i, o] > 0 else DOWN))
                pygame.draw.line(self.edges, col, a[i], b[o], 1 if v < 0.6 else 2)
        surf.blit(self.edges, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def neurons(self, surf, x):
        # senses: name, dot, and a small bar of the value
        for (px, py), v, name in zip(self.pos[0], x[0], self.names):
            T.text(surf, name, (px - 120, py), "regular", 22 if self.narrow else 19, T.TEXT, anchor="midright")
            bar = pygame.Rect(px - 108, py - 4, 84, 8)
            pygame.draw.rect(surf, T.PANEL_EDGE, bar, border_radius=4)
            fill = int(42 * min(abs(v), 1.0))
            pygame.draw.rect(surf, UP if v > 0 else DOWN,
                             pygame.Rect(bar.centerx if v > 0 else bar.centerx - fill, py - 4, fill, 8), border_radius=4)
            pygame.draw.circle(surf, tint(UP if v > 0 else DOWN, v), (px, py), 6)
        # hidden neurons
        for k in range(1, len(x) - 1):
            for (px, py), v in zip(self.pos[k], x[k]):
                col = UP if v > 0 else DOWN
                if abs(v) > 0.8:
                    T.glow(surf, (px, py), 9, col, strength=40, layers=4)
                pygame.draw.circle(surf, tint(col, v), (px, py), 3.4)
        # controls: big dots, names and the value as a centred bar
        for (px, py), v, name in zip(self.pos[-1], x[-1], CONTROLS):
            col = UP if v > 0 else DOWN
            T.glow(surf, (px, py), 26, col, strength=int(25 + 60 * min(abs(v), 1)), layers=6)
            pygame.draw.circle(surf, tint(col, v, 0.25), (px, py), 15)
            T.text(surf, name, (px + 34, py - 14), "semibold", 28 if self.narrow else 30, T.TEXT, anchor="midleft")
            bar = pygame.Rect(px + 34, py + 14, 110 if self.narrow else 220, 12)
            pygame.draw.rect(surf, T.PANEL_EDGE, bar, border_radius=6)
            fill = int(bar.w / 2 * min(abs(v), 1.0))
            pygame.draw.rect(surf, col, pygame.Rect(bar.centerx if v > 0 else bar.centerx - fill, bar.y, fill, 12),
                             border_radius=6)
            pygame.draw.line(surf, T.MUTED, (bar.centerx, bar.y - 4), (bar.centerx, bar.bottom + 4), 2)
            T.text(surf, self.reading(name, v), (bar.right + 12, bar.centery), "regular", 20 if self.narrow else 22, T.MUTED,
                   anchor="midleft")

    def reading(self, name: str, v: float) -> str:
        if name == "engine":  # > +0.5 lights it, < -0.5 shuts it down; while lit, 0..1 is min..max throttle
            n, kn = int(self.rec["engines"][self.i]), self.rec["thrust"][self.i] / 1000
            return (f"{n} lit, {kn / 1000:.1f} MN" if kn >= 1000 else f"{n} lit, {kn:.0f} kN") if n else "off"
        return f"{v:+.2f}"

    def header(self, surf, t: float):
        sizes = " > ".join(str(len(b)) for b in [self.pos[0]] + self.net.b)
        T.text(surf, self.label or "the AI's brain", (60, 40), "bold", 44)
        T.text(surf, f"{sizes} neurons" + ("" if self.narrow else ": senses > two hidden layers > controls"), (60, 96),
               "regular", 24, T.MUTED)
        T.text(surf, "green: firing / pushing up    blue: pulling down", (self.w - 60, 52), "regular", 22, T.MUTED,
               anchor="topright")
        ts, ys = self.rec["time"], self.rec["y"] - self.rec["deck_y"]
        h = float(np.interp(t, ts, ys))
        height = f"{h / 1000:.2f} km" if h >= 1000 else f"{max(h, 0):.0f} m"
        T.text(surf, f"t {min(t, ts[-1]):.1f} s    height {height}", (self.w - 60, 90), "semibold", 26, T.TEXT,
               anchor="topright")
