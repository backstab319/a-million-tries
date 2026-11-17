"""A hand-written entry controller (ep18 step 11): from 70 km down to where the classic landing controller takes
over. No learning; it proves the descent is possible with our numbers and gives the training ladder its starts.

- **Entry burn:** light the three engines at `light_h`, pointing exactly against the motion (retrograde), and shut
  them down once the speed is under `cut_speed`. That keeps the heating and the air load under their limits.
- **The long fall:** lean off retrograde so the air pushes the booster across (body lift) towards the ship. It
  asks for the sideways acceleration that arrives over the ship at `handover_h` with no sideways speed (the textbook
  a = -6x/t^2 - 4v/t), and turns that into a lean with the body-lift formula (a = rho v^2 A angle / m), up to
  `max_lean`. The fins and the cold-gas thrusters hold the lean (the engine is off).
- **Below `handover_h`:** the classic hoverslam controller (rocket_classic.act) from the moment it lights the
  engine, fins held straight. Until then this keeps holding the attitude: the classic only steers with the gimbal,
  and coasting unheld, body lift turns the booster and blows it hundreds of metres off the deck.

Real boosters fly this with guidance we don't have the details of; this is simpler.
"""

from dataclasses import dataclass

import numpy as np

from ailearns.agents import rocket_classic
from ailearns.games.rocket import Boosters


@dataclass
class EntryGains:
    light_h: float = 67e3  # m
    cut_speed: float = 700.0  # m/s
    max_lean: float = np.radians(10.0)
    handover_h: float = 3000.0  # m: the classic flies from here
    kp: float = 3.0  # attitude gains, per radian of error and per rad/s of spin
    kd: float = 2.0


def act(b: Boosters, g: EntryGains | None = None) -> np.ndarray:
    g = g or EntryGains()
    speed = np.hypot(b.vx, b.vy)
    retro = -np.arctan2(b.vx, -b.vy)  # the tilt that points the engines along the motion
    entry_burn = b.lit & (b.ignitions == 1) & (b.cfg.ignitions == 2)
    t = np.maximum((b.y - g.handover_h) / np.maximum(-b.vy, 50.0), 2.0)
    a_want = -6 * b.x / t**2 - 4 * b.vx / t
    area = np.pi * (b.cfg.diameter / 2) ** 2
    lean = -a_want * b.mass / (b.air_density(b.y) * speed**2 * area + 1e-9)  # leaning + pushes towards -x
    lean = np.where(entry_burn, 0.0, np.clip(lean, -g.max_lean, g.max_lean))
    err = b.tilt - (retro + lean)
    light = (b.ignitions == 2) & (b.y <= g.light_h)
    engine = np.where((light | entry_burn) & (speed > g.cut_speed), 1.0, -1.0)
    gimbal = np.clip(-g.kp * 2.7 * err - g.kd * 2 * b.spin, -1, 1)
    rcs = np.clip(-g.kp * 1.7 * err - g.kd * 2.5 * b.spin, -1, 1)
    fin = np.clip(g.kp * err + g.kd * b.spin, -1, 1)
    high = np.stack([engine, gimbal, rcs, fin], axis=1)
    low = np.concatenate([rocket_classic.act(b), np.zeros((b.n, 1))], axis=1)
    classic = (b.y <= g.handover_h) & (b.lit | (low[:, 0] > 0)) & (b.ignitions < b.cfg.ignitions)
    low[:, 0] = np.where(~b.lit & (low[:, 0] > 0), 1.0, low[:, 0])  # (11g) to light, ask past the switch's band
    high[:, 0] = np.where(b.y <= g.handover_h, -1.0, high[:, 0])
    return np.where(classic[:, None], low, high)
