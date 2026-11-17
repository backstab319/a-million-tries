"""A rocket booster's landing burn in 2D (the vertical plane), as numpy arrays.

Many boosters fly at once, one per row. The booster falls engine-first towards
the pad and has one engine, which it may light only once. Every number is in
SI units and comes from devlog step 1 (public figures or our labelled estimates).

The state is the position of the base (where the legs touch down), the speed,
the tilt from vertical (positive = top leaning towards +x) and the spin, the
fuel left, and the engine: is it lit, how hard it pushes now, and where the
gimbal points. Nothing here draws anything.
"""

from dataclasses import dataclass, replace

import numpy as np

G0 = 9.80665  # standard gravity, for Isp -> exhaust speed

FLYING, LANDED, CRASHED, TIMEOUT = 0, 1, 2, 3
N_ACTIONS = 3  # engine (> 0 lights it; once lit, 0..1 = min..max throttle), gimbal, cold-gas thrusters
N_OBS = 18  # 12, then (ep18 7b) height, drift and speeds again on a square-root scale, then (9b) wind and deck
# (ep18 10a) a booster with grid fins has one more control (the fins' angle) and one more sense (where they point)
REWARD_LANDED = 1.0
REWARD_TIMEOUT = -10.0  # never came down: worse than any crash (a 300 m/s hit 1.5 km off scores ~-6.1), so it pays to come down


@dataclass
class RocketConfig:
    # Booster (devlog step 1).
    dry_mass: float = 22_200.0  # kg, public
    length: float = 42.6  # m, public
    diameter: float = 3.7  # m, public
    cg_height: float = 15.0  # m above the base; estimate (engines make it bottom-heavy)
    leg_span: float = 18.0  # m between opposite feet; estimate
    # Engine: one Merlin-like engine for the landing burn.
    thrust_max: float = 845_200.0  # N, sea level, public
    thrust_min: float = 482_000.0  # N, lowest throttle (57%), public
    isp: float = 282.0  # s, sea level, public
    ignitions: int = 1  # the landing burn gets one light
    thrust_lag: float = 0.3  # s, time constant of the throttle; estimate
    gimbal_max: float = np.radians(5.0)  # estimate
    gimbal_rate: float = np.radians(20.0)  # per second; estimate
    rcs_force: float = 1_000.0  # N, cold-gas thrusters at the top; estimate
    # Air.
    gravity: float = 9.81
    rho0: float = 1.225  # kg/m^3 at sea level
    scale_height: float = 8_500.0  # m
    cd_axial: float = 0.8  # falling engine-first; estimate
    cd_side: float = 1.2  # a cylinder broadside; estimate
    wind: float = 0.0  # m/s along x
    # Weather (ep18 9b), drawn per burn, uniform in each range; 0 = calm. The wind blows at `wind_mean` 10 m up
    # (either way), stronger higher up (power law with `wind_shear`, the usual over open sea), plus gusts of up to
    # `wind_gust`. The deck heaves up and down `heave` m on a swell of `heave_period` s.
    wind_mean: tuple = (0.0, 0.0)
    wind_gust: tuple = (0.0, 0.0)
    wind_shear: float = 0.14
    heave: tuple = (0.0, 0.0)
    heave_period: float = 8.0
    # Grid fins (ep18 10a): lattice fins near the top that steer the fall through the air. Four in an X; seen in
    # this plane they act like two. 0 area = no fins (every run before 10a). Estimates, from ~1.2 x 1.5 m
    # titanium fins (public-ish); the body's axial drag (cd_axial) already stands for the whole stack, fins included.
    fin_area: float = 0.0  # m^2, the pair's planform together (with fins: 2 x 1.8)
    fin_height: float = 40.0  # m above the base (the interstage, near the top); estimate
    fin_cn: float = 2.5  # normal force per radian of angle to the air, per planform area; estimate
    fin_max: float = np.radians(20.0)  # how far they swing; estimate
    fin_rate: float = np.radians(40.0)  # per second (hydraulic); estimate
    fin_stall: float = np.radians(25.0)  # the force stops growing past this angle to the air; estimate
    # (10c) Open-loop hydraulics, like Falcon 9's first drone-ship try (CRS-5, Jan 2015): every move of the fins
    # vents fluid overboard, and with the tank dry the fins go limp (they float with the air and push nothing).
    # CRS-5's ran dry just before landing. Drawn per burn: the fin travel left when the landing burn starts, after
    # whatever entry used. Our estimate (the real figure isn't public). (0, 0) = closed loop, never runs out.
    fin_fluid: tuple = (0.0, 0.0)  # rad of fin travel
    # (10e) Hardware faults, each on its own in `fault_p` of burns, at a time uniform in `fault_window`: the fins
    # stick where they are (CRS-16, Dec 2018: a fin pump stalled and the booster spun into the sea), the gimbal
    # jams where it is; and from the start, the engine gives only `engine_weak` of the thrust it's asked for.
    # The flight computer knows when an actuator stops answering (two senses); a weak engine shows in the thrust.
    fault_p: float = 0.0
    fault_window: tuple = (0.0, 15.0)  # s
    engine_weak: tuple = (0.80, 0.95)
    drag_slices: int = 8  # side drag is summed over slices of the body, so turning through air is resisted
    # Start of the landing burn (ranges are uniform).
    start_height: tuple = (2_000.0, 2_500.0)
    start_fall: tuple = (250.0, 300.0)  # m/s downwards
    start_x: tuple = (-50.0, 50.0)  # grid fins have already lined it up; estimate (wider starts need fins)
    start_vx: tuple = (-3.0, 3.0)
    start_tilt: tuple = (np.radians(-3.0), np.radians(3.0))
    start_spin: tuple = (np.radians(-0.5), np.radians(0.5))
    start_fuel: tuple = (5_500.0, 6_500.0)  # kg; estimate
    # Touchdown judge; estimates (legs absorb the last few m/s).
    land_vy: float = 3.0
    land_vx: float = 1.0
    land_tilt: float = np.radians(8.0)
    land_spin: float = np.radians(5.0)
    pad_radius: float = 43.0
    dt: float = 0.05
    max_time: float = 60.0
    # (ep18 11a) From 70 km. Every default keeps the worlds before step 11 exactly as they were.
    atmosphere: str = "exp"  # "us76": the 1976 US Standard Atmosphere table (density, pressure, speed of sound)
    cd_mach: tuple = ()  # ((mach, factor on cd_axial), ...): drag through the sound barrier; () = no Mach effect
    thrust_vac: float = 0.0  # N per engine in vacuum (public: 914 kN); 0 = sea-level thrust at any height
    light_engines: tuple = (1,)  # engines each light uses, in order (Falcon 9: entry burn 3, landing burn 1)
    heat_limit: float = 0.0  # J/m^2 on the engine end (Sutton-Graves, below) before it fails; 0 = no limit
    q_limit: float = 0.0  # Pa of dynamic pressure the structure takes; 0 = no limit
    bend_limit: float = 0.0  # (12m) Pa x degrees: dynamic pressure times the angle off the airflow (q-alpha), the
    # sideways bending load a long thin tank takes before it buckles; launchers are built for a few hundred kPa.deg
    # 0 = no limit (before 12m a booster could swing +-30 deg across the airflow at 58 kPa)
    heat_radius: float = 1.85  # m, the blunt engine end's radius for the heating formula
    start_fpa: tuple = ()  # (11b) rad below the horizon: start_fall is then the speed along this path, flown
    # engine first (start_tilt off it), moving towards +x; () = falling straight down as before
    start_seq: float = 0.0  # (11e) s of spin-up to light (or to shut the entry burn down): the pumps spin up while
    # the command asks for it and back down while it doesn't; then the igniter fires. Estimate. 0 = instant
    switch_band: float = 0.0  # (11g) light only above +band, shut down only below -band: lowest throttle (0) no
    # longer sits on the off switch. 0 = the old switch at 0
    body_lift: bool = False  # slender-body lift (normal force 2 q A sin a cos a, at the leading end); textbook physics


FIN_AREA = 3.6  # m^2, the fitted fins (RocketConfig.fin_area)


FAULT_P = 0.1  # (10e) each fault's share of burns (RocketConfig.fault_p)
FIN_FLUID = (np.radians(100.0), np.radians(250.0))  # (10c) the open-loop fins' travel left (RocketConfig.fin_fluid)


def n_actions(cfg: RocketConfig) -> int:
    return N_ACTIONS + (cfg.fin_area > 0)


def n_obs(cfg: RocketConfig) -> int:
    return N_OBS + (cfg.fin_area > 0) + (cfg.fin_fluid[1] > 0) + 2 * (cfg.fault_p > 0) + ENTRY_OBS * entry(cfg)


ENTRY_OBS = 8  # (11e) + how far through a start sequence; (11b) height, speeds and drift on the scale of the whole descent, Mach, peak q and heat so far


def entry(cfg: RocketConfig) -> bool:
    return cfg.heat_limit > 0 or cfg.q_limit > 0


# (11a) 1976 US Standard Atmosphere: height (m), density (kg/m^3), pressure (Pa), temperature (K); public tables
US76 = np.array([
    [0, 1.225, 101325, 288.15], [5e3, 0.7364, 54048, 255.65], [10e3, 0.4135, 26500, 223.15],
    [11e3, 0.3639, 22632, 216.65], [15e3, 0.1948, 12111, 216.65], [20e3, 0.08891, 5529, 216.65],
    [25e3, 0.04008, 2549, 221.65], [30e3, 0.01841, 1197, 226.65], [32e3, 0.01322, 868.0, 228.65],
    [40e3, 0.003996, 287.1, 251.05], [47e3, 0.001426, 110.9, 270.65], [50e3, 0.001027, 79.78, 270.65],
    [51e3, 0.0008598, 66.94, 270.65], [60e3, 0.0003097, 21.96, 247.02], [70e3, 8.283e-5, 5.221, 219.59],
    [71e3, 6.420e-5, 3.956, 214.65], [80e3, 1.846e-5, 1.052, 196.65], [90e3, 3.40e-6, 0.1836, 186.87]])
SUTTON_GRAVES = 1.7415e-4  # Earth air, SI: heat flux = k sqrt(rho / radius) v^3
CD_MACH = ((0.0, 1.0), (0.8, 1.0), (1.05, 1.4), (1.3, 1.5), (2.0, 1.3), (3.0, 1.15), (5.0, 1.05), (8.0, 1.0))  # estimate


def us76(y):
    """-> density, pressure / sea level, speed of sound at height y (m), from the table (log for the first two)."""
    y = np.clip(y, 0.0, US76[-1, 0])
    rho = np.exp(np.interp(y, US76[:, 0], np.log(US76[:, 1])))
    p = np.exp(np.interp(y, US76[:, 0], np.log(US76[:, 2]))) / US76[0, 2]
    return rho, p, np.sqrt(1.4 * 287.05 * np.interp(y, US76[:, 0], US76[:, 3]))


BEND_LIMIT = 800e3  # (12m) Pa.deg, an estimate: ~10 deg off the airflow at 80 kPa, ~14 deg at 58. A hands-off
# booster weathervanes to ~380, the hand-written entry ~520; ppo25 swung to ~1450 (devlog 12m)
ENTRY = dict(  # (11b) devlog step 11 has the numbers and where they come from
    atmosphere="us76", cd_mach=CD_MACH, thrust_vac=914e3, light_engines=(3, 1), ignitions=2, body_lift=True,
    heat_limit=1.5e6, q_limit=150e3, bend_limit=BEND_LIMIT, start_height=(68e3, 72e3), start_fall=(1600.0, 1900.0),
    start_fpa=(np.radians(55.0), np.radians(65.0)), start_x=(-29e3, -25e3), start_fuel=(19e3, 21e3),
    max_time=240.0, start_seq=0.5, switch_band=0.5)
NEVER = 1e9  # s: a fault time that never comes
DECK_WIDTH = 52.0  # m, a drone ship's landing deck across (public figures for the barges: ~52 x ~91 m)


def world(name: str = "pad") -> RocketConfig:
    """Where it lands (ep18 step 9). "pad": the land pad (43 m). "ship": a drone ship's deck, where all four feet
    must stay on the 52 m deck, so the base may land at most 52/2 - 18/2 = 17 m off the centre line. "storm": the
    ship in weather from calm to a gale (wind, gusts, a heaving deck). Add "+fins" (10a) for grid fins: "storm+fins",
    and "+fluid" (10c) to run them on open-loop hydraulics that can run dry: "storm+fins+fluid"; "+faults" (10e) for
    hardware that breaks: "storm+fins+faults". "+nobend" (12p): without the bending limit, as before 12m."""
    if name.endswith("+nobend"):  # (12p) the physics before the bending limit (ep18 footage of the swing)
        return replace(world(name.removesuffix("+nobend")), bend_limit=0.0)
    if name.endswith("+faults"):
        return replace(world(name.removesuffix("+faults")), fault_p=FAULT_P)
    if name.endswith("+fluid"):
        base = world(name.removesuffix("+fluid"))
        if base.fin_area <= 0:
            raise ValueError("+fluid needs +fins first")
        return replace(base, fin_fluid=FIN_FLUID)
    if name == "entry":  # (11b) from 70 km: entry burn, the long fall, landing burn; on the ship, calm, with fins
        return replace(world("ship+fins"), **ENTRY)
    if name.endswith("+fins"):
        return replace(world(name.removesuffix("+fins")), fin_area=FIN_AREA)
    c = RocketConfig()
    if name == "pad":
        return c
    ship = replace(c, pad_radius=DECK_WIDTH / 2 - c.leg_span / 2)
    if name == "ship":
        return ship
    if name == "storm":  # the ship, any day from calm to a gale: up to 12 m/s at the deck (~25 at 2 km), gusts
        return replace(ship, wind_mean=(0.0, 12.0), wind_gust=(0.0, 4.0), heave=(0.0, 1.5))
    raise ValueError(f"no world {name!r}")


class Boosters:
    """`n` boosters on their landing burns, stepped together."""

    def __init__(self, n: int = 1, config: RocketConfig | None = None, seed: int | None = None):
        self.n = n
        self.cfg = config or RocketConfig()
        self.rng = np.random.default_rng(seed)
        z = lambda: np.zeros(n)  # noqa: E731
        self.x, self.y, self.vx, self.vy = z(), z(), z(), z()
        self.tilt, self.spin = z(), z()
        self.fuel, self.thrust, self.gimbal, self.fin, self.fluid = z(), z(), z(), z(), z()
        self.fin_fail_t, self.gimbal_fail_t, self.engine_eff = z() + NEVER, z() + NEVER, z() + 1.0
        self.bend = z()  # (12m) the last step's q-alpha (Pa.deg), for the breakup check and the renderers
        self.heat, self.peak_q, self.hold = z(), z(), z()  # hold: s the command has asked for a light / shutdown  # (11a) J/m^2 taken on the engine end so far, the highest q so far
        self.time = z()
        self.lit = np.zeros(n, dtype=bool)
        self.ignitions = np.zeros(n, dtype=np.int64)
        self.outcome = np.zeros(n, dtype=np.int64)
        self.wind0, self.gust, self.gust_phase, self.heave, self.heave_phase = z(), z(), z(), z(), z()
        self.reset()

    # -- derived quantities -------------------------------------------------
    @property
    def mass(self) -> np.ndarray:
        return self.cfg.dry_mass + self.fuel

    @property
    def inertia(self) -> np.ndarray:
        """Moment of inertia about the centre of mass, as a uniform rod whose mass centre is lowered."""
        c = self.cfg
        return self.mass * (c.length**2 / 12 + (c.length / 2 - c.cg_height) ** 2)

    @property
    def exhaust_speed(self) -> float:
        return self.cfg.isp * G0

    def air_density(self, y: np.ndarray) -> np.ndarray:
        if self.cfg.atmosphere == "us76":
            return us76(y)[0]
        return self.cfg.rho0 * np.exp(-np.maximum(y, 0.0) / self.cfg.scale_height)

    def overshoot(self) -> np.ndarray:
        """How far past the touchdown limits, each in units of its own limit, summed (0 = a soft landing)."""
        c = self.cfg
        return (np.maximum(np.abs(self.vy - self.deck_vy()) - c.land_vy, 0) / c.land_vy
                + np.maximum(np.abs(self.vx) - c.land_vx, 0) / c.land_vx
                + np.maximum(np.abs(self.tilt) - c.land_tilt, 0) / c.land_tilt
                + np.maximum(np.abs(self.spin) - c.land_spin, 0) / c.land_spin
                + np.maximum(np.abs(self.x) - c.pad_radius, 0) / c.pad_radius)

    def lowest_foot(self) -> np.ndarray:
        """Height of the lowest leg foot above the deck: a tilted booster touches down on one side first."""
        return self.y - self.cfg.leg_span / 2 * np.abs(np.sin(self.tilt)) - self.deck_y()

    def wind_at(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        """The wind along x at height y, time t: steady (stronger higher up) plus two gust waves."""
        c = self.cfg
        steady = self.wind0 * (np.maximum(y, 1.0) / 10.0) ** c.wind_shear
        gusts = self.gust * (0.6 * np.sin(2 * np.pi * t / 7.0 + self.gust_phase)
                             + 0.4 * np.sin(2 * np.pi * t / 2.3 + 2.7 * self.gust_phase))
        return c.wind + steady + gusts

    def deck_y(self) -> np.ndarray:
        """The deck's height (0 on land), heaving on the swell."""
        return self.heave * np.sin(2 * np.pi * self.time / self.cfg.heave_period + self.heave_phase)

    def deck_vy(self) -> np.ndarray:
        w = 2 * np.pi / self.cfg.heave_period
        return self.heave * w * np.cos(w * self.time + self.heave_phase)

    # -- episode ------------------------------------------------------------
    def reset(self, mask: np.ndarray | None = None) -> None:
        idx = np.arange(self.n) if mask is None else np.flatnonzero(mask)
        k, c, u = len(idx), self.cfg, self.rng.uniform
        self.x[idx] = u(*c.start_x, k)
        self.y[idx] = u(*c.start_height, k)
        self.vx[idx] = u(*c.start_vx, k)
        self.vy[idx] = -u(*c.start_fall, k)
        self.tilt[idx] = u(*c.start_tilt, k)
        self.spin[idx] = u(*c.start_spin, k)
        self.fuel[idx] = u(*c.start_fuel, k)
        self.thrust[idx] = 0.0
        self.gimbal[idx] = 0.0
        self.fin[idx] = 0.0
        self.heat[idx] = 0.0
        self.hold[idx] = 0.0
        self.peak_q[idx] = 0.0
        self.time[idx] = 0.0
        self.lit[idx] = False
        self.ignitions[idx] = c.ignitions
        self.outcome[idx] = FLYING
        if max(c.wind_mean[1], c.wind_gust[1], c.heave[1]) > 0:  # drawn last, so calm worlds draw as before
            self.wind0[idx] = u(*c.wind_mean, k) * self.rng.choice([-1.0, 1.0], k)
            self.gust[idx] = u(*c.wind_gust, k)
            self.gust_phase[idx] = u(0.0, 2 * np.pi, k)
            self.heave[idx] = u(*c.heave, k)
            self.heave_phase[idx] = u(0.0, 2 * np.pi, k)
        if c.fin_fluid[1] > 0:  # (10c) drawn after the weather, so other worlds draw as before
            self.fluid[idx] = u(*c.fin_fluid, k)
        if c.fault_p > 0:  # (10e) drawn last, so other worlds draw as before
            hit = lambda: self.rng.random(k) < c.fault_p  # noqa: E731
            self.fin_fail_t[idx] = np.where(hit(), u(*c.fault_window, k), NEVER)
            self.gimbal_fail_t[idx] = np.where(hit(), u(*c.fault_window, k), NEVER)
            self.engine_eff[idx] = np.where(hit(), u(*c.engine_weak, k), 1.0)
        if c.start_fpa:  # (11b) coming in along a slanted path, engine first
            path = u(*c.start_fpa, k)
            speed = -self.vy[idx]
            self.vx[idx], self.vy[idx] = speed * np.cos(path), -speed * np.sin(path)
            self.tilt[idx] = self.tilt[idx] - np.arctan2(self.vx[idx], -self.vy[idx])

    def leg_wear(self) -> np.ndarray:
        """(ep18 8a) How much of the legs' crush cores a touchdown at this speed uses up: they absorb its energy, so
        it goes with speed squared. 0 for a feather landing, 1 at the rated limits (half from falling, half from
        sliding). A landing scores REWARD_LANDED * (1 - wear): a soft one leaves a booster ready to fly again."""
        c = self.cfg
        return 0.5 * (((self.vy - self.deck_vy()) / c.land_vy) ** 2 + (self.vx / c.land_vx) ** 2)

    def obs(self) -> np.ndarray:
        c = self.cfg
        fins = [self.fin / c.fin_max] if c.fin_area > 0 else []  # (10a) where the grid fins point
        fins += [self.fluid / c.fin_fluid[1]] if c.fin_fluid[1] > 0 else []  # (10c) the fluid left
        if c.fault_p > 0:  # (10e) the actuators that stopped answering
            fins += [(self.time >= self.fin_fail_t).astype(float), (self.time >= self.gimbal_fail_t).astype(float)]
        if entry(c):  # (11b) the whole descent's scale
            mach = np.hypot(self.vx - self.wind_at(self.y, self.time), self.vy) / us76(self.y)[2]
            fins += [self.y / 70e3, self.vy / 2000.0, self.vx / 1000.0, self.x / 30e3, mach / 6.0,
                     self.peak_q / c.q_limit, self.heat / c.heat_limit, self.hold / max(c.start_seq, 1e-9)]
        return np.stack([
            self.x / 500.0, (self.y - self.deck_y()) / 2500.0, self.vx / 50.0, self.vy / 300.0,
            np.sin(self.tilt), np.cos(self.tilt), self.spin,
            self.fuel / c.start_fuel[1], self.thrust / c.thrust_max, self.gimbal / c.gimbal_max,
            self.lit.astype(float), self.ignitions.astype(float),
            # the same on a square-root scale, sharp near zero: 10 m up reads 0.06 here (0.004 above), 3 m/s 0.1
            np.sign(self.x) * np.sqrt(np.abs(self.x) / 500.0), np.sqrt(np.maximum(self.y - self.deck_y(), 0.0) / 2500.0),
            np.sign(self.vx) * np.sqrt(np.abs(self.vx) / 50.0), np.sign(self.vy) * np.sqrt(np.abs(self.vy) / 300.0),
            # (9b) the weather: the wind where it is, and how fast the deck is rising or falling
            self.wind_at(self.y, self.time) / 20.0, self.deck_vy() / 3.0, *fins,
        ], axis=1)

    # -- physics ------------------------------------------------------------
    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance every booster still flying by one dt. `action` is (n, 3) in [-1, 1], or (n, 4) with the
        grid fins' angle last (ignored on a booster without fins; 3 columns hold the fins straight).

        Returns (obs, reward, done). The reward comes only at the end: +1 landed; a crash scores
        -(1 + log(1 + overshoot)), so hitting at 5 m/s beats hitting at 250 (-1.5 v -5.4) and learning has a
        slope to climb; a timeout scores REWARD_TIMEOUT. Nothing is paid along the way.
        """
        c, dt = self.cfg, self.cfg.dt
        a = np.asarray(action, dtype=float).reshape(self.n, -1)
        a = np.clip(np.pad(a, ((0, 0), (0, N_ACTIONS + 1 - a.shape[1]))), -1.0, 1.0)
        if c.fin_area <= 0:
            a[:, 3] = 0.0
        fly = self.outcome == FLYING

        # Engine: like a real landing burn, once lit it burns until touchdown or an empty tank (no shutdown
        # in flight); the command then sets the throttle, and 0 or below means the lowest throttle.
        # (11a) With a light still in hand (the entry burn), a command below 0 shuts it down instead.
        shut = fly & self.lit & (self.ignitions > 0) & (a[:, 0] < -c.switch_band)
        light = fly & (a[:, 0] > c.switch_band) & ~self.lit & (self.ignitions > 0) & (self.fuel > 0)
        if c.start_seq > 0:  # (11e) only a command held through the start sequence lights it (or shuts it down)
            # the pumps spin up while it's asked for and back down while it isn't (they don't reset on a flicker)
            self.hold = np.where(fly, np.clip(self.hold + np.where(shut | light, dt, -dt), 0.0, c.start_seq), self.hold)
            done_seq = self.hold >= c.start_seq - 1e-9
            shut, light = shut & done_seq, light & done_seq
            self.hold = np.where(shut | light, 0.0, self.hold)
        self.lit &= ~shut
        self.ignitions -= light
        self.lit = (self.lit | light) & (self.fuel > 0)
        engines = np.asarray(c.light_engines)[np.clip(c.ignitions - self.ignitions - 1, 0, len(c.light_engines) - 1)]
        target = np.where(self.lit, c.thrust_min + (c.thrust_max - c.thrust_min) * np.maximum(a[:, 0], 0.0), 0.0)
        target = target * engines  # (11a) the entry burn lights three
        target = target * self.engine_eff  # (10e) a weak engine gives less than it's asked for
        lag = 1.0 if c.thrust_lag <= 0 else 1.0 - np.exp(-dt / c.thrust_lag)
        self.thrust = np.where(fly, self.thrust + (target - self.thrust) * lag, self.thrust)
        step_g = c.gimbal_rate * dt
        self.gimbal = np.where(fly & (self.time < self.gimbal_fail_t), self.gimbal + np.clip(a[:, 1] * c.gimbal_max - self.gimbal, -step_g, step_g),
                               self.gimbal)
        step_f = c.fin_rate * dt
        move = np.where(fly & (self.time < self.fin_fail_t),  # (10e) stuck fins stay where they are
                        np.clip(a[:, 3] * c.fin_max - self.fin, -step_f, step_f), 0.0)
        if c.fin_fluid[1] > 0:  # (10c) open loop: a move uses its travel in fluid; what's left limits it
            move = np.clip(move, -self.fluid, self.fluid)
            self.fluid = self.fluid - np.abs(move)
        self.fin = self.fin + move
        limp = (c.fin_fluid[1] > 0) & (self.fluid <= 0)  # dry: the fins float with the air

        # Fuel burnt this step; the speed it gives is exact (rocket equation over the step).
        m0 = self.mass
        burn = np.where(fly, np.minimum(self.thrust / self.exhaust_speed * dt, self.fuel), 0.0)
        self.fuel = self.fuel - burn
        # (11a) the throttle sets the flow; thin air lets the same flow push harder (sea level -> vacuum thrust)
        boost = 1.0
        if c.thrust_vac > 0:
            boost = c.thrust_vac / c.thrust_max - (c.thrust_vac / c.thrust_max - 1.0) * us76(self.y)[1]
        dv_thrust = self.exhaust_speed * boost * np.log(m0 / self.mass)
        thrust_used = np.where(burn > 0, self.thrust * boost, 0.0)
        out = self.fuel <= 0  # tank dry: the engine starves
        self.lit &= ~out
        self.thrust = np.where(out, 0.0, self.thrust)

        # Body axes: `up` along the booster, `side` across it.
        st, ct = np.sin(self.tilt), np.cos(self.tilt)
        dirx, diry = np.sin(self.tilt - self.gimbal), np.cos(self.tilt - self.gimbal)

        # Drag, split into along-body and across-body parts. Across the body, each slice feels its own
        # speed, which includes the turning (a spinning tube is braked at both ends). Most of the side area
        # is above the centre of mass, so falling engine-first is stable, like a dart.
        rho = self.air_density(self.y)
        rvx = self.vx - self.wind_at(self.y, self.time)
        v_ax = rvx * st + self.vy * ct
        v_side = rvx * ct - self.vy * st
        area_ax = np.pi * (c.diameter / 2) ** 2
        speed = np.hypot(rvx, self.vy)
        cd_ax = c.cd_axial
        if c.cd_mach:  # (11a) drag peaks through the sound barrier
            table = np.asarray(c.cd_mach)
            cd_ax = c.cd_axial * np.interp(speed / us76(self.y)[2], table[:, 0], table[:, 1])
        f_ax = -0.5 * rho * cd_ax * area_ax * np.abs(v_ax) * v_ax
        # (11a) heating on the blunt engine end (Sutton-Graves) and dynamic pressure, against their limits
        q = 0.5 * rho * speed**2
        self.heat = self.heat + np.where(fly, SUTTON_GRAVES * np.sqrt(rho / c.heat_radius) * speed**3 * dt, 0.0)
        self.peak_q = np.where(fly, np.maximum(self.peak_q, q), self.peak_q)
        self.bend = q * np.degrees(np.arctan2(np.abs(v_side), np.abs(v_ax)))  # (12m) q-alpha, Pa.deg
        k = c.drag_slices
        arm = (np.arange(k) + 0.5) * c.length / k - c.cg_height  # slice centres above the centre of mass
        v_slice = v_side[:, None] + self.spin[:, None] * arm
        f_slice = -0.5 * rho[:, None] * c.cd_side * (c.length / k * c.diameter) * np.abs(v_slice) * v_slice
        f_side = f_slice.sum(axis=1)
        aero_torque = (f_slice * arm).sum(axis=1)
        if c.body_lift:  # (11a) a tube at an angle to the air is pushed across, linearly in the angle, at its
            # leading end (engine first: below the centre of mass, so it tries to turn it round; fins fight that)
            lift = -rho * area_ax * v_side * np.abs(v_ax)  # = 2 q A sin(a) cos(a), against the crossflow
            lead = np.where(v_ax < 0, -c.cg_height, c.length - c.cg_height)
            f_side = f_side + lift
            aero_torque = aero_torque + lift * lead
        # Grid fins (10a): a plate at angle `fin` to the body, high up. The air's speed across the plate (falling
        # plus any sideways and turning motion there) sets its angle of attack; the force is normal to the plate,
        # grows with the angle up to the stall, and with the air's speed squared. Undeflected, they resist the
        # top swinging sideways (they steady the fall like feathers on a dart); swung, they steer it.
        arm_f = c.fin_height - c.cg_height
        v_fin = v_side + self.spin * arm_f
        sf, cf = np.sin(self.fin), np.cos(self.fin)
        w_n, w_t = v_ax * sf - v_fin * cf, -v_ax * cf - v_fin * sf  # the air's speed across and along the plate
        aoa = np.arctan2(w_n, np.abs(w_t))
        f_fin = np.where(limp, 0.0, 0.5 * rho * (v_ax**2 + v_fin**2) * c.fin_area * c.fin_cn
                         * np.clip(aoa, -c.fin_stall, c.fin_stall))
        f_ax = f_ax - f_fin * sf
        f_side = f_side + f_fin * cf
        aero_torque = aero_torque + f_fin * cf * arm_f
        m = m0
        ax = (f_ax * st + f_side * ct) / m
        ay = (f_ax * ct - f_side * st) / m - c.gravity

        # Turning: gimbal (engine below the centre of mass), side drag, thrusters at the top.
        torque = (c.cg_height * thrust_used * np.sin(self.gimbal) + aero_torque
                  + a[:, 2] * c.rcs_force * (c.length - c.cg_height))
        spin_new = self.spin + torque / self.inertia * dt

        vx_new = self.vx + ax * dt + dv_thrust * dirx
        vy_new = self.vy + ay * dt + dv_thrust * diry
        f = fly.astype(float)
        self.x = self.x + f * 0.5 * (self.vx + vx_new) * dt
        self.y = self.y + f * 0.5 * (self.vy + vy_new) * dt
        self.vx = np.where(fly, vx_new, self.vx)
        self.vy = np.where(fly, vy_new, self.vy)
        self.tilt = self.tilt + f * 0.5 * (self.spin + spin_new) * dt
        self.spin = np.where(fly, spin_new, self.spin)
        self.time = self.time + f * dt

        # Touchdown judge.
        down = fly & (self.lowest_foot() <= 0.0)
        soft = ((np.abs(self.vy - self.deck_vy()) <= c.land_vy) & (np.abs(self.vx) <= c.land_vx)
                & (np.abs(self.tilt) <= c.land_tilt) & (np.abs(self.spin) <= c.land_spin)
                & (np.abs(self.x) <= c.pad_radius))
        self.outcome[down & soft] = LANDED
        self.outcome[down & ~soft] = CRASHED
        broke = fly & ~down & (((c.heat_limit > 0) & (self.heat > c.heat_limit))  # (11a) burnt or torn apart
                               | ((c.q_limit > 0) & (self.peak_q > c.q_limit))
                               | ((c.bend_limit > 0) & (self.bend > c.bend_limit)))  # (12m) buckled sideways
        self.outcome[broke] = CRASHED
        self.outcome[fly & ~down & (self.time >= c.max_time)] = TIMEOUT
        ended = fly & (self.outcome != FLYING)
        self.lit &= ~ended
        self.thrust = np.where(ended, 0.0, self.thrust)

        crash = -1.0 - np.log1p(self.overshoot())
        reward = np.where(self.outcome == LANDED, REWARD_LANDED * (1.0 - self.leg_wear()),
                          np.where(self.outcome == TIMEOUT, REWARD_TIMEOUT, crash))
        reward = np.where(ended, reward, 0.0)
        return self.obs(), reward, self.outcome != FLYING
