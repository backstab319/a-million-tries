"""A hand-written hoverslam controller for the 2D booster: the classic baseline, no learning.

It reads the true state (no noise) and uses textbook rules:

- **When to light:** stopping from speed v within height h needs a steady braking of v^2 / 2h. The booster coasts
  until that need reaches what the engine (at `ignite_at` throttle) plus air drag can give, then lights. Drag
  fades as it slows (with v^2), so the plan counts a third of it.
- **Throttle:** from then on, it asks for exactly the braking that stops it at the ground, minus what drag gives.
- **Sideways:** with t seconds left to touchdown (about 2h / v at a steady slowdown), it wants a sideways speed
  that closes the gap to the pad, capped so it can still brake that speed to zero before touchdown with the
  push it has, and leans to chase that speed. (The textbook a = -6x/t^2 - 4vx/t asks for more push than a
  booster has near the ground, and it landed sliding.) To get the push, it picks a lean angle. A tilted booster falling fast is also pushed sideways by the air, *against*
  its lean, and at high speed that can beat the thrust's sideways part. So it tries a fan of lean angles, works
  out the total sideways force of each (thrust plus air), picks the best, and turns there with the gimbal.
- **Touchdown:** like a real booster, it lands with the engine still burning, aiming to meet the ground at
  `touch_speed`; it only cuts the engine early if it stops and starts to rise.

Real boosters use convex-optimisation guidance (G-FOLD style); this is simpler and only proves a landing is
possible with our numbers.
"""

from dataclasses import dataclass

import numpy as np

from ailearns.games.rocket import Boosters


@dataclass
class ClassicGains:
    ignite_at: float = 0.5  # throttle (0 = lowest, 1 = full) the ignition rule plans for, leaving room both ways
    touch_speed: float = 1.5  # m/s: aim to meet the ground at this speed, engine lit
    max_lean: float = np.radians(4.0)
    lean_choices: int = 41
    min_time_left: float = 1.0  # s: the guidance stops sharpening this close to touchdown
    brake_share: float = 0.3  # sideways speed is capped to what 30% of the push could brake before touchdown
    speed_time: float = 1.5  # s: how quickly the lean chases the wanted sideways speed
    tilt_kp: float = 2.0  # gimbal (radians) per radian of tilt error
    tilt_kd: float = 2.5  # gimbal (radians) per rad/s of spin


def act(b: Boosters, g: ClassicGains | None = None) -> np.ndarray:
    g = g or ClassicGains()
    c = b.cfg
    m = b.mass
    h = np.maximum(b.lowest_foot(), 0.1)
    v = -b.vy  # falling speed
    drag = 0.5 * b.air_density(b.y) * c.cd_axial * np.pi * (c.diameter / 2) ** 2 * v * np.abs(v)
    need = np.maximum(v * np.abs(v) - g.touch_speed**2, 0.0) / (2 * h)  # braking to meet the ground at touch_speed
    thrust_for = lambda u: c.thrust_min + (c.thrust_max - c.thrust_min) * u  # noqa: E731
    # Drag fades with v^2 as the booster slows: over a steady slowdown it averages a third of today's.
    can = thrust_for(g.ignite_at) / m - c.gravity + drag / 3 / m

    # Throttle: the thrust that gives `need`, as a command in (0, 1]; > 0 keeps the engine lit.
    want_thrust = m * (need + c.gravity) - drag
    u = np.clip((want_thrust - c.thrust_min) / (c.thrust_max - c.thrust_min), 1e-3, 1.0)
    burning = b.lit | (need >= can)
    rising = b.lit & (b.vy > 0)
    engine = np.where(burning & ~rising, u, -1.0)

    # Sideways: lean so the thrust pushes towards the pad.
    t = np.maximum(2 * h / np.maximum(v, 1.0), g.min_time_left)
    lean = np.linspace(-g.max_lean, g.max_lean, g.lean_choices)[None, :]
    lean_x = sideways_accel(b, lean, np.maximum(b.thrust, thrust_for(0.0) * burning)[:, None])
    push = np.minimum(lean_x.max(axis=1), -lean_x.min(axis=1)).clip(0.0)  # sideways push it can give either way
    vx_cap = g.brake_share * push * t
    vx_want = np.clip(-b.x / np.maximum(t / 2, g.min_time_left), -vx_cap, vx_cap)
    a_x = (vx_want - b.vx) / g.speed_time
    lean = lean[0, np.argmin(np.abs(lean_x - a_x[:, None]), axis=1)]
    gimbal = g.tilt_kp * (lean - b.tilt) - g.tilt_kd * b.spin
    gimbal = np.clip(gimbal / c.gimbal_max, -1.0, 1.0)
    return np.stack([engine, gimbal, np.zeros(b.n)], axis=1)


def sideways_accel(b: Boosters, lean: np.ndarray, thrust: np.ndarray) -> np.ndarray:
    """Sideways acceleration of each booster (rows) if it held each lean angle (columns), not spinning, the
    gimbal centred: the thrust's sideways part plus the air's push along and across the body."""
    c = b.cfg
    st, ct = np.sin(lean), np.cos(lean)
    rho = b.air_density(b.y)[:, None]
    rvx, vy = (b.vx - c.wind)[:, None], b.vy[:, None]
    v_ax = rvx * st + vy * ct
    v_side = rvx * ct - vy * st
    f_ax = -0.5 * rho * c.cd_axial * np.pi * (c.diameter / 2) ** 2 * np.abs(v_ax) * v_ax
    f_side = -0.5 * rho * c.cd_side * c.length * c.diameter * np.abs(v_side) * v_side
    if c.body_lift:  # (ep18 11b) the slender-body lift, when the world has it (Boosters.step)
        f_side = f_side - rho * np.pi * (c.diameter / 2) ** 2 * v_side * np.abs(v_ax)
    return (thrust * st + f_ax * st + f_side * ct) / b.mass[:, None]
