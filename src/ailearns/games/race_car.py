"""The racing sandbox's car: real forces on four tyres, in JAX (episode 14).

A car is built from its **design numbers** (`DESIGN`: engine power and position, front and rear wings, front and rear
tyre widths, wheelbase, drive layout, brake and roll balance, gearing, ride height, springs) on top of a **class
spec** (`CarSpec`: chassis mass, tyre compound, floor; the AI's cars use `RACE`, the reality checks also use `ROAD`).
`derive()` turns them into physics constants; `drive()` advances the car one decision (several physics substeps).
Nothing here knows about tracks or drawing: it's a car on a road with a surface under each tyre, the air around it
and the road's slope (`Surface`, `Air`, `Slope`; flat by default), so the reality checks
(tests/test_race_physics.py) drive it directly.

The model (body frame: x forward, y left; SI units throughout):
- Body: position, heading, forward and sideways speed, yaw rate (3 degrees of freedom).
- Where the weight is: every part (chassis, driver, fuel, engine, wings, wheels, four-wheel-drive hardware) has a
  mass and a place along the car, so the weight balance and how readily the car rotates (yaw inertia) follow from
  the design: a mid-engined car turns in more eagerly than a rear-engined one of the same balance.
- Four tyres, each with its own load and slip angle. Lateral force: a simplified Pacejka "magic formula", scaled by
  load-sensitive grip (more load, less grip per newton: weight transfer costs grip; wider tyres suffer less).
  Braking/driving force shares the same grip (friction ellipse). Ask for more than the tyre has and it slides: a
  locked wheel gives 85% of its grip, straight against its own motion, so it can't steer; a spinning wheel keeps a
  little sideways grip (power oversteer).
- Tyre temperature: each tyre heats from the work its grip does (slip and sliding) and from flexing as it rolls,
  and cools in the airflow (faster at speed). Grip peaks in a window around the compound's best temperature: cold
  tyres and overheated tyres grip less. Wider tyres hold more heat: slower to warm up, slower to overheat.
- Tyre wear: the same friction work wears the tread (faster when the tyre is too hot; narrow tyres faster for the
  same work). Grip fades slowly as the tread goes, then falls off a cliff near the end.
- Fuel: the car starts with fuel for the run (enough at its engine's burn rate, plus a reserve: bigger engines start
  heavier) and burns it with the power it uses (petrol's energy at a racing engine's efficiency), getting lighter.
  When the tank is empty the engine stops pushing (it still drags: engine braking).
- Weight transfer: forward/back under braking and acceleration, left/right in corners (split between the axles by
  the roll-stiffness balance), from the accelerations a moment ago (soft springs settle slower).
- Aero: drag, and downforce from the front wing, rear wing and floor, all growing with speed squared; the balance
  follows from the wings. The floor works harder the closer it runs to the ground (ground effect), and makes most of
  the downforce (as in today's F1: wings set the balance, the floor carries the load); but within ~1 cm of the road
  the air under it chokes and the floor stalls, so a car riding on its plank loses downforce (porpoising's cause). Sliding
  sideways costs downforce (wings and floor are built for air from straight ahead) and adds drag (the car's side
  meets the air).
- Suspension: downforce and braking squash the springs. Too low or too soft and the car bottoms out (more so on a
  kerb): it rides on its skid plank, the tyres lose that load, the plank drags and the floor stalls. Stiff springs
  stop that but lose grip on bumps and kerbs.
- Engine: constant torque up to 80% of the rev limit, constant power above; an automatic gearbox (always the gear
  with the most push; each change cuts the drive briefly). Spinning parts: the wheels, and through the gears the
  engine, have to be spun up too, so the car accelerates as if heavier, most in first gear (the engine's share grows
  with the gear ratio squared). A reverse gear (F1 rules require one), selected at a standstill, pushes like first.
  Off the throttle the engine holds the driven wheels back (engine braking, growing
  with the revs). Front, rear or four-wheel drive (any split), a
  limited-slip differential on each driven axle. Brakes with a front/rear bias. A rate-limited steering rack.
- Air: drag and downforce multipliers per car (`Air`), so a car in another's wake can get a tow and lose downforce
  (slipstream and dirty air, for races between cars; a lone car has clean air).
- Hills and banking (`Slope`): gravity pulls along the road's slope (uphill slows, downhill speeds up; the springs
  feel the weight move back on a climb), a crest lightens the car and a dip presses it down (it can bottom out in a
  compression), and a banked corner tilts the road so gravity helps the car round.
- Below ~5 m/s the sideways motion blends into a kinematic model (no slip): slip formulas break down at walking
  pace, and this only affects standing starts.

Not modelled (yet): suspension geometry, car-to-car contact.
"""

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

G, RHO = 9.81, 1.225  # gravity m/s^2, air density kg/m^3
PETROL_J_PER_KG = 43e6  # energy in a kilogram of petrol

# The design numbers the engineer chooses: (name, lowest, highest). A design is an array in these units.
DESIGN = (
    ("power_kw", 75.0, 750.0),         # peak engine power
    ("wing_front", 0.0, 1.0),          # front wing size: 0 = none, 1 = F1-like
    ("wing_rear", 0.0, 1.0),           # rear wing size
    ("tyre_front_mm", 185.0, 405.0),   # front tyre width
    ("tyre_rear_mm", 185.0, 405.0),    # rear tyre width
    ("wheelbase", 2.3, 3.6),           # metres
    ("engine_pos", 0.0, 1.0),          # 0 = ahead of the front axle, ~0.6 = behind the driver, 1 = behind the rear axle
    ("drive_front", 0.0, 1.0),         # share of the push to the front wheels: 0 = rear drive, 1 = front drive
    ("brake_front", 0.40, 0.80),       # share of the braking on the front axle
    ("roll_front", 0.30, 0.70),        # share of the cornering weight transfer taken by the front axle
    ("top_gear_kmh", 180.0, 380.0),    # speed at the rev limit in top gear (gearing: acceleration vs top speed)
    ("ride_height_mm", 15.0, 120.0),   # floor height at rest
    ("spring_n_mm", 30.0, 300.0),      # spring stiffness at each wheel
)
DESIGN_NAMES = tuple(d[0] for d in DESIGN)
DESIGN_LOW = np.array([d[1] for d in DESIGN], np.float32)
DESIGN_HIGH = np.array([d[2] for d in DESIGN], np.float32)
D = {name: i for i, name in enumerate(DESIGN_NAMES)}


def design(**values) -> np.ndarray:
    """A design array from keyword values; anything not given sits mid-range."""
    unknown = set(values) - set(DESIGN_NAMES)
    if unknown:
        raise ValueError(f"unknown design numbers: {sorted(unknown)}")
    return np.array([values.get(n, (lo + hi) / 2) for n, lo, hi in DESIGN], np.float32)


def normalise(d):
    """Design numbers -> [-1, 1] each (what the driver is told about its car)."""
    return 2 * (d - DESIGN_LOW) / (DESIGN_HIGH - DESIGN_LOW) - 1


@dataclass(frozen=True)
class CarSpec:
    """A class of car: what the design numbers sit on. Places are fractions of the wheelbase behind the front axle."""
    chassis_kg: float = 400.0     # everything not listed below
    chassis_x: float = 0.52
    driver_kg: float = 75.0
    driver_x: float = 0.45
    fuel_kg: float = 1.0          # fuel left at the end of a run (a reserve; real series: a 1-litre sample) ...
    race_s: float = 90.0          # ... plus fuel for a run this long ...
    full_throttle_share: float = 0.6  # ... at full power this share of the time (F1: ~60-70% of a lap)
    thermal_eff: float = 0.35     # share of the fuel's energy the engine turns into work (racing petrol engine)
    fuel_x: float = 0.55
    engine_kg_base: float = 40.0
    engine_kg_per_kw: float = 0.25
    wing_front_kg: float = 12.0   # at full size
    wing_rear_kg: float = 18.0
    overhang_front: float = 0.9   # metres ahead of the front axle (front wing)
    overhang_rear: float = 0.6    # metres behind the rear axle (rear wing)
    wheel_kg: float = 12.0        # one wheel and tyre at 185 mm
    wheel_kg_per_mm: float = 0.03
    awd_kg: float = 70.0          # extra shafts and differentials for four-wheel drive
    awd_loss: float = 0.03        # ... and the power they cost
    cog_base: float = 0.27        # centre of mass height (m) with the floor on the ground
    track_width: float = 1.60     # metres between left and right tyres
    cla_floor: float = 3.0        # floor/body downforce coefficient x area (m^2) at the reference ride height:
                                  # with ground effect the floor makes most of it (~60% on the F1-like car; 2022+ F1: 50-65%)
    floor_front: float = 0.45     # share of the floor's downforce on the front axle
    ground_gain: float = 0.6      # floor downforce +60% from the reference height down to its best height ...
    ground_ref: float = 0.06      # ... reference ride height, metres
    ground_peak: float = 0.012    # the floor's best height (m); closer, the air under it chokes and the floor stalls
    ground_stall: float = 0.45    # ... down to this share of its best on the ground (est.: the stall behind porpoising)
    cla_wing_front: float = 1.0   # added by a full-size front wing (F1-like total CL.A ~7.2 at 250 km/h: sources
    cla_wing_rear: float = 1.6    # range ~5.5-9; it pulls 5.6 g there, F1 4-6.5 g)
    cda_base: float = 0.55        # drag coefficient x area (m^2) with no wings (body, wheels, the floor's own drag)
    cda_wing_front: float = 0.17  # the wings keep their lift-to-drag (6 front, 4.2 rear)
    cda_wing_rear: float = 0.38
    cda_per_tyre_mm: float = 0.0008  # wider tyres, more drag (per mm over 185, all four)
    plank_mu: float = 0.4         # friction of the skid plank on the road when the car bottoms out
    kerb_height: float = 0.03     # metres a kerb lifts a tyre
    spring_soft: float = 30.0     # N/mm: settle time and grip-on-bumps scale between these
    spring_stiff: float = 300.0
    mu: float = 1.9               # tyre grip coefficient at the nominal load (cornering)
    mu_long: float = 1.1          # braking/driving grip relative to cornering grip (tyres grip harder lengthways)
    load_sens: float = 0.15       # grip lost per unit of load above nominal
    fz_nominal_245: float = 4000.0  # nominal load (N) of a 245 mm tyre; scales with width
    pacejka_b: float = 10.0       # magic formula stiffness, shape, curvature
    pacejka_c: float = 1.9
    pacejka_e: float = 0.97
    slide: float = 0.85           # grip of a sliding (locked or spinning) tyre, as a share of its peak
    drivetrain: float = 0.90      # share of engine power reaching the road
    diff_bias: float = 2.0        # limited-slip differential: most push one tyre of an axle can get vs the other
    gears: int = 7
    gear_spread: float = 3.2      # top gear's speed at the limiter / first gear's
    shift_s: float = 0.05         # seconds without drive per gear change (racing gearbox: near-seamless)
    brake_g: float = 6.5          # brake system capacity, in g of the car's mass
    max_steer: float = 0.35       # road-wheel lock, radians
    steer_rate: float = 1.2       # road-wheel radians per second
    roll_resist: float = 0.010    # rolling resistance at 0 km/h (grows with speed)
    wheel_spin_share: float = 0.6  # a wheel's spin inertia / (mass x radius^2): most of the mass is out at the tyre
    engine_inertia: float = 0.03  # kg m^2 at the crank (engine, clutch, gearbox): tiny in a racing engine ...
    rev_limit_rpm: float = 12000.0  # ... spinning this fast at each gear's top speed
    engine_brake: float = 0.15    # off-throttle drag at the limiter, as a share of that gear's full-power push
    yaw_down_loss: float = 8.0    # downforce x (1 - this x slide angle^2): ~25% gone at 10 degrees sideways
    cda_side: float = 1.5         # drag area of the car's side (m^2), felt as sin^2 of the slide angle
    temp_best: float = 100.0      # tyre temperature (C) with the most grip (racing slicks: ~90-110)
    temp_window: float = 50.0     # grip x (1 - temp_loss x ((T - best) / window)^2), floored at temp_floor
    temp_loss: float = 0.3
    temp_floor: float = 0.5
    temp_air: float = 30.0        # air and track temperature (C): where a tyre cools towards
    tyre_heat_cap: float = 5000.0  # J per C a 245 mm tyre's working layer holds (scales with width)
    tyre_cool: float = 30.0       # W per C a 245 mm tyre sheds standing still ...
    tyre_cool_v: float = 20.0     # ... and that much more again every this many m/s
    heat_share: float = 0.5       # share of the tyre's friction work that heats the tyre (the rest heats the road)
    wear_rate: float = 3e-8       # tread used (0-1) per joule of friction work on a 245 mm tyre (F1-like car at
                                  # race pace: ~150 km, like a medium compound) ...
    wear_hot: float = 30.0        # ... doubling for every this many C over the best temperature
    wear_loss: float = 0.12       # grip lost by a fully worn tyre, before ...
    wear_cliff: float = 0.8       # ... the cliff: past this much wear, grip falls fast


RACE = CarSpec()  # the AI's class: an open-wheel racing car (F1-like at the top of every range)
ROAD = replace(   # a road sports car (for the reality checks): heavier, road tyres, no wings, no ground effect
    RACE, chassis_kg=1286.0, chassis_x=0.60, driver_x=0.50, fuel_x=0.30, fuel_kg=50.0, race_s=0.0, wing_front_kg=0.0,
    wing_rear_kg=0.0, overhang_front=1.0, overhang_rear=1.0, cog_base=0.37, track_width=1.55, cla_floor=0.05,
    ground_gain=0.0, cla_wing_front=0.0, cla_wing_rear=0.0, cda_base=0.65, cda_wing_front=0.0, cda_wing_rear=0.0,
    cda_per_tyre_mm=0.0, mu=1.15, load_sens=0.12, pacejka_b=8.0, drivetrain=0.85, gears=7, gear_spread=4.0,
    shift_s=0.2, brake_g=1.4, max_steer=0.6, steer_rate=1.5, engine_inertia=0.12, rev_limit_rpm=7500.0,
    temp_best=80.0, temp_window=60.0, temp_loss=0.1)  # road tyres: a wide, forgiving window

# Race day's tyre compounds (episode 15), on top of a class spec: (grip, wear, best temperature shift in C).
# Grip: ~0.5-0.7 s a lap between neighbours on a ~90 s lap (Pirelli's usual soft/medium/hard gaps), checked in the
# devlog. Wear is x the spec's wear_rate (~150 km a set): real F1 sets last ~20-40 laps of a 90 min race, so ours
# wear ~2-3x faster to give a 25 min race the same choice (softs ~40 km, mediums ~70, hards ~100; Pirelli-like
# 1 : 1.6 : 2.4). Softer compounds work at lower temperatures (Pirelli's "low" and "high working range" compounds).
COMPOUNDS = ("soft", "medium", "hard")
COMPOUND_GRIP = (1.03, 1.0, 0.975)
COMPOUND_WEAR = (3.6, 2.2, 1.5)
COMPOUND_TEMP = (-5.0, 0.0, 5.0)


def with_compound(p: "CarParams", compound) -> "CarParams":
    """The car on these tyres (an index into COMPOUNDS)."""
    pick = lambda xs: jnp.asarray(xs, jnp.float32)[compound]
    return p._replace(mu=p.mu * pick(COMPOUND_GRIP), wear_rate=p.wear_rate * pick(COMPOUND_WEAR),
                      temp_best=p.temp_best + pick(COMPOUND_TEMP))


class CarParams(NamedTuple):
    """Physics constants of one car (scalars unless noted; vmap over many cars)."""
    mass: jax.Array
    yaw_inertia: jax.Array
    a: jax.Array          # centre of mass -> front axle, metres
    b: jax.Array          # centre of mass -> rear axle
    cog_height: jax.Array
    track_width: jax.Array
    cla_floor: jax.Array
    floor_front: jax.Array
    cla_front: jax.Array  # wings only
    cla_rear: jax.Array
    ground_gain: jax.Array
    ground_ref: jax.Array
    ground_peak: jax.Array
    ground_stall: jax.Array
    cda: jax.Array
    ride_height: jax.Array
    spring: jax.Array     # N/m per wheel
    settle: jax.Array     # seconds for weight transfer to settle
    bump_loss: jax.Array  # share of grip lost per unit of surface roughness (stiff springs lose more)
    plank_mu: jax.Array
    brake_front: jax.Array
    roll_front: jax.Array
    drive_front: jax.Array
    mu: jax.Array
    mu_long: jax.Array
    load_sens: jax.Array
    fz_nominal: jax.Array  # (4,)
    power: jax.Array      # watts at the road (after drivetrain losses)
    gear_top: jax.Array   # m/s at the limiter in each gear, (gears,)
    shift_s: jax.Array
    diff_bias: jax.Array
    brake_force: jax.Array
    pb: jax.Array
    pc: jax.Array
    pe: jax.Array
    slide: jax.Array
    max_steer: jax.Array
    steer_rate: jax.Array
    roll_resist: jax.Array
    spin_wheels: jax.Array  # kg: the wheels' spin inertia as extra mass to accelerate
    spin_engine: jax.Array  # kg m^2/s^2: engine inertia x (rad/s at the limiter)^2; / gear_top^2 = extra mass
    engine_brake: jax.Array
    yaw_down_loss: jax.Array
    cda_side: jax.Array
    tyre_size: jax.Array  # (4,) width / 245 mm: heat held and shed scale with it
    temp_best: jax.Array
    temp_window: jax.Array
    temp_loss: jax.Array
    temp_floor: jax.Array
    temp_air: jax.Array
    tyre_heat_cap: jax.Array
    tyre_cool: jax.Array
    tyre_cool_v: jax.Array
    heat_share: jax.Array
    wear_rate: jax.Array
    wear_hot: jax.Array
    wear_loss: jax.Array
    wear_cliff: jax.Array
    fuel_per_j: jax.Array  # kg of fuel per joule at the wheels
    fuel_tank: jax.Array   # kg of fuel at the start: when it's all burned, the engine stops


def _awd(d):
    """1 if the car needs four-wheel-drive hardware (any real split), easing in over the first 10%."""
    s = d[D["drive_front"]]
    return jnp.clip(10 * jnp.minimum(s, 1 - s), 0.0, 1.0)


def fuel_start_kg(d, spec: CarSpec = RACE):
    """Fuel at the start: the reserve plus the run at the engine's full-power burn for full_throttle_share of it."""
    burn = 1000.0 * d[D["power_kw"]] / (spec.thermal_eff * PETROL_J_PER_KG)  # kg/s at full power
    return spec.fuel_kg + burn * spec.race_s * spec.full_throttle_share


def parts(d, spec: CarSpec = RACE):
    """Every part's mass (kg) and place (metres behind the front axle): the car's weight comes from its parts."""
    L = d[D["wheelbase"]]
    wheel = lambda mm: 2 * (spec.wheel_kg + spec.wheel_kg_per_mm * (mm - 185.0))  # noqa: E731 (a pair)
    return [
        (spec.chassis_kg, spec.chassis_x * L),
        (spec.driver_kg, spec.driver_x * L),
        (fuel_start_kg(d, spec), spec.fuel_x * L),
        (spec.engine_kg_base + spec.engine_kg_per_kw * d[D["power_kw"]], L * (-0.05 + 1.15 * d[D["engine_pos"]])),
        (spec.wing_front_kg * d[D["wing_front"]], -spec.overhang_front),
        (spec.wing_rear_kg * d[D["wing_rear"]], L + spec.overhang_rear),
        (wheel(d[D["tyre_front_mm"]]), 0.0 * L),
        (wheel(d[D["tyre_rear_mm"]]), L),
        (spec.awd_kg * _awd(d), 0.5 * L),
    ]


# ── The cost cap (owner, 2026-10-20: "whichever is more fun") ───────────────
# Prices in thousands of dollars. Buying everything costs ~$1.5M; the budget is $1M, so something has to give: a big
# engine or big wings, not both. Setup (balances, gearing, ride height, springs, where the engine sits) is free.
BUDGET_K = 1000.0
PRICES = {
    "chassis": 250.0,            # fixed, plus 40 per metre of wheelbase over 2.3 m
    "engine": (50.0, 0.5, 0.0005),  # 50 + 0.5 per kW + 0.0005 per kW^2: big engines get dear fast
    "wing_front": 80.0,          # at full size
    "wing_rear": 120.0,
    "tyres": (25.0, 0.35),       # per axle: 25 + 0.35 per mm over 185
    "awd": 100.0,                # four-wheel-drive hardware
}


def cost_breakdown(d) -> dict:
    """What each part of the design costs ($k)."""
    e0, e1, e2 = PRICES["engine"]
    t0, t1 = PRICES["tyres"]
    kw = d[D["power_kw"]]
    return {
        "chassis": PRICES["chassis"] + 40.0 * (d[D["wheelbase"]] - 2.3),
        "engine": e0 + e1 * kw + e2 * kw ** 2,
        "wings": PRICES["wing_front"] * d[D["wing_front"]] + PRICES["wing_rear"] * d[D["wing_rear"]],
        "tyres": 2 * t0 + t1 * ((d[D["tyre_front_mm"]] - 185.0) + (d[D["tyre_rear_mm"]] - 185.0)),
        "awd": PRICES["awd"] * _awd(d),
    }


def cost_k(d):
    """The design's price in $k (the budget is BUDGET_K)."""
    return sum(cost_breakdown(d).values())


def mass_kg(d, spec: CarSpec = RACE):
    return sum(m for m, _ in parts(d, spec))


def weight_front(d, spec: CarSpec = RACE):
    """Share of the car's weight on the front axle (follows from where the parts are)."""
    x = sum(m * x for m, x in parts(d, spec)) / mass_kg(d, spec)
    return 1 - x / d[D["wheelbase"]]


def derive(d, spec: CarSpec = RACE) -> CarParams:
    """Design numbers + class spec -> physics constants. Works under jit/vmap."""
    d = jnp.asarray(d, jnp.float32)

    def f(x):
        return jnp.asarray(x, jnp.float32)

    ps = parts(d, spec)
    m = sum(pm for pm, _ in ps)
    L = d[D["wheelbase"]]
    x_cg = sum(pm * px for pm, px in ps) / m
    a = jnp.clip(x_cg, 0.15 * L, 0.85 * L)
    b = L - a
    length = L + spec.overhang_front + spec.overhang_rear
    inertia = (sum(pm * (px - x_cg) ** 2 for pm, px in ps)
               + spec.chassis_kg * length ** 2 / 12                      # the chassis is long, not a point
               + (ps[6][0] + ps[7][0]) * (spec.track_width / 2) ** 2)     # wheels sit out to the sides
    top = d[D["top_gear_kmh"]] / 3.6
    gear_top = top * spec.gear_spread ** (jnp.arange(spec.gears) / (spec.gears - 1) - 1)
    k_nmm = d[D["spring_n_mm"]]
    stiff = jnp.clip((k_nmm - spec.spring_soft) / (spec.spring_stiff - spec.spring_soft), 0.0, 1.0)
    rh = d[D["ride_height_mm"]] / 1000.0
    wf, wr = d[D["wing_front"]], d[D["wing_rear"]]
    tf, tr = d[D["tyre_front_mm"]], d[D["tyre_rear_mm"]]
    return CarParams(
        mass=m, yaw_inertia=inertia, a=a, b=b, cog_height=spec.cog_base + rh, track_width=f(spec.track_width),
        cla_floor=f(spec.cla_floor), floor_front=f(spec.floor_front),
        cla_front=spec.cla_wing_front * wf, cla_rear=spec.cla_wing_rear * wr,
        ground_gain=f(spec.ground_gain), ground_ref=f(spec.ground_ref), ground_peak=f(spec.ground_peak),
        ground_stall=f(spec.ground_stall),
        cda=(spec.cda_base + spec.cda_wing_front * wf + spec.cda_wing_rear * wr
             + spec.cda_per_tyre_mm * ((tf - 185.0) + (tr - 185.0)) / 2),
        ride_height=rh, spring=k_nmm * 1000.0, settle=0.05 * jnp.sqrt(100.0 / k_nmm),
        bump_loss=stiff, plank_mu=f(spec.plank_mu),
        brake_front=d[D["brake_front"]], roll_front=d[D["roll_front"]], drive_front=d[D["drive_front"]],
        mu=f(spec.mu), mu_long=f(spec.mu_long), load_sens=f(spec.load_sens),
        fz_nominal=spec.fz_nominal_245 * jnp.stack([tf, tf, tr, tr]) / 245.0,
        power=(spec.drivetrain - spec.awd_loss * _awd(d)) * 1000.0 * d[D["power_kw"]], gear_top=gear_top,
        shift_s=f(spec.shift_s), diff_bias=f(spec.diff_bias),
        brake_force=spec.brake_g * G * m, pb=f(spec.pacejka_b), pc=f(spec.pacejka_c), pe=f(spec.pacejka_e),
        slide=f(spec.slide), max_steer=f(spec.max_steer), steer_rate=f(spec.steer_rate),
        roll_resist=f(spec.roll_resist),
        spin_wheels=spec.wheel_spin_share * (ps[6][0] + ps[7][0]),
        spin_engine=f(spec.engine_inertia * (spec.rev_limit_rpm * 2 * np.pi / 60) ** 2),
        engine_brake=f(spec.engine_brake), yaw_down_loss=f(spec.yaw_down_loss), cda_side=f(spec.cda_side),
        tyre_size=jnp.stack([tf, tf, tr, tr]) / 245.0, temp_best=f(spec.temp_best),
        temp_window=f(spec.temp_window), temp_loss=f(spec.temp_loss), temp_floor=f(spec.temp_floor),
        temp_air=f(spec.temp_air), tyre_heat_cap=f(spec.tyre_heat_cap), tyre_cool=f(spec.tyre_cool),
        tyre_cool_v=f(spec.tyre_cool_v), heat_share=f(spec.heat_share), wear_rate=f(spec.wear_rate),
        wear_hot=f(spec.wear_hot), wear_loss=f(spec.wear_loss), wear_cliff=f(spec.wear_cliff),
        fuel_per_j=f(1.0 / (spec.thermal_eff * PETROL_J_PER_KG * spec.drivetrain)), fuel_tank=fuel_start_kg(d, spec))


class Surface(NamedTuple):
    """What's under each tyre, (4,) each: front-left, front-right, rear-left, rear-right."""
    grip: jax.Array      # grip multiplier (tarmac 1)
    rolling: jax.Array   # extra rolling resistance coefficient
    rough: jax.Array     # bumpiness: stiff springs lose this share of grip
    lift: jax.Array      # metres the surface lifts the tyre (a kerb)


class Air(NamedTuple):
    """The air the car is in: multipliers on drag, on the front and rear wings' downforce and on the floor's (clean
    air: all 1). Turbulent air hurts wings far more than a ground-effect floor (why F1 went back to floors in 2022)."""
    drag: jax.Array
    down_front: jax.Array
    down_rear: jax.Array
    floor: jax.Array = 1.0


class Slope(NamedTuple):
    """The road under the car: gravity along the road surface in the car's frame (m/s^2, forward and left) and the
    load factor (1 on the flat; > 1 in a dip or banked corner, < 1 over a crest)."""
    gx: jax.Array
    gy: jax.Array
    load: jax.Array


TARMAC = Surface(jnp.ones(4), jnp.zeros(4), jnp.full(4, 0.03), jnp.zeros(4))
CLEAN_AIR = Air(jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0))
FLAT = Slope(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0))


class CarState(NamedTuple):
    pos: jax.Array      # (2,) metres, world frame
    yaw: jax.Array      # heading, radians
    vx: jax.Array       # forward speed, m/s (body frame)
    vy: jax.Array       # sideways speed, m/s (left positive)
    r: jax.Array        # yaw rate, rad/s (counter-clockwise positive)
    steer: jax.Array    # road-wheel angle now, radians
    ax: jax.Array       # felt accelerations a moment ago (for weight transfer), m/s^2
    ay: jax.Array
    gear: jax.Array     # 1-based; 0: reverse
    shift: jax.Array    # seconds left of the current gear change (no drive meanwhile)
    ride: jax.Array     # (2,) floor height at the front and rear axle a moment ago, metres (negative: bottomed)
    temp: jax.Array     # (4,) tyre temperatures, C
    wear: jax.Array     # (4,) tread used: 0 new, 1 gone
    fuel_used: jax.Array  # kg burned since the start (the car is that much lighter)


class CarInfo(NamedTuple):
    """What the car was doing (for the reality checks, the driver's senses and the footage)."""
    load: jax.Array      # (4,) tyre loads, newtons
    sliding: jax.Array   # (4,) bool: locked or spinning
    gear: jax.Array      # 1-based
    ax: jax.Array        # felt acceleration this step, m/s^2
    ay: jax.Array
    plank: jax.Array     # (2,) newtons on the skid plank at the front/rear (bottoming out: sparks)
    downforce: jax.Array  # newtons


def car_at(pos, yaw, speed=0.0, ride_height=0.05, tyre_temp=90.0) -> CarState:
    z = jnp.zeros((), jnp.float32)
    return CarState(jnp.asarray(pos, jnp.float32), jnp.asarray(yaw, jnp.float32), jnp.asarray(speed, jnp.float32),
                    z, z, z, z, z, jnp.asarray(1, jnp.int32), z, jnp.full(2, ride_height, jnp.float32),
                    jnp.full(4, tyre_temp, jnp.float32), jnp.zeros(4, jnp.float32), z)


def temp_grip(p: CarParams, temp):
    """Grip multiplier of tyres at these temperatures (1 in the window's middle)."""
    off = (temp - p.temp_best) / p.temp_window
    return jnp.clip(1 - p.temp_loss * off ** 2, p.temp_floor, 1.0)


def wear_grip(p: CarParams, wear):
    """Grip multiplier of tyres this worn: a slow fade, then the cliff."""
    return 1 - p.wear_loss * wear - 1.0 * jnp.maximum(wear - p.wear_cliff, 0.0)


def engine_force(p: CarParams, vx, gear=None):
    """Push at the wheels at full throttle (N) and the gear for it. Constant torque up to 80% of the rev limit,
    constant power above; below first gear's constant-torque speed the clutch slips (same torque). Given the current
    gear, it stays there unless another gear pushes clearly (5%) harder: in the constant-power range several gears
    push the same, and a gearbox that chased rounding errors would never stop shifting."""
    v = jnp.maximum(vx, 0.0)
    per_gear = p.power / jnp.maximum(v, 0.8 * p.gear_top)    # F = P / v, capped at the constant-torque push
    per_gear = jnp.where(v <= p.gear_top, per_gear, 0.0)      # over the limiter: no push in that gear
    g = jnp.argmax(per_gear)
    if gear is not None:
        cur = per_gear[gear - 1]
        g = jnp.where(cur >= 0.95 * per_gear[g], gear - 1, g)
    return per_gear[g], (g + 1).astype(jnp.int32)


def _magic(p: CarParams, alpha):
    bx = p.pb * alpha
    return jnp.sin(p.pc * jnp.arctan(bx - p.pe * (bx - jnp.arctan(bx))))


def _axle_split(total, cap_l, cap_r, bias):
    """A limited-slip differential: an even split until the weaker tyre is at its limit, then the excess goes to the
    other, up to `bias` times as much. Returns (left, right)."""
    weak = jnp.minimum(cap_l, cap_r)
    weak_ask = jnp.where(total / 2 <= weak, total / 2, total / (1 + bias))
    left = jnp.where(cap_l <= cap_r, weak_ask, total - weak_ask)
    return left, total - left


def _substep(p: CarParams, s: CarState, steer_in, pedal, surf: Surface, air: Air, slope: Slope, dt, reverse=0.0):
    """One physics step. steer_in, pedal in [-1, 1] (pedal > 0 throttle, < 0 brake); reverse > 0.5 asks for
    reverse gear (see `drive`)."""
    # steering rack: moves toward the asked angle at a limited rate
    want = jnp.clip(steer_in, -1, 1) * p.max_steer
    steer = s.steer + jnp.clip(want - s.steer, -p.steer_rate * dt, p.steer_rate * dt)
    speed = jnp.hypot(s.vx, s.vy)
    L, tw, h, m = p.a + p.b, p.track_width, p.cog_height, p.mass - s.fuel_used
    q = 0.5 * RHO * speed ** 2

    # downforce: wings, and a floor that works harder near the ground (stalls once it's on the ground)
    floor_h = jnp.mean(s.ride)
    # rises as the floor nears the ground, best at ground_peak; closer than that the flow under it chokes and the floor
    # stalls (to ground_stall of its best on the ground): riding on the plank costs downforce, as on a real car
    ground = jnp.clip(1 + p.ground_gain * (p.ground_ref - floor_h) / (p.ground_ref - p.ground_peak), 0.0,
                      1 + p.ground_gain)
    near = jnp.clip(floor_h / p.ground_peak, 0.0, 1.0)
    ground = jnp.where(floor_h < p.ground_peak, (1 + p.ground_gain) * (p.ground_stall + (1 - p.ground_stall) * near ** 2),
                       ground)
    beta = jnp.arctan2(jnp.abs(s.vy), jnp.maximum(jnp.abs(s.vx), 5.0))  # slide angle (floored at walking pace)
    yaw_keep = jnp.clip(1 - p.yaw_down_loss * beta ** 2, 0.0, 1.0)       # sideways air: less downforce
    floor = q * p.cla_floor * ground * yaw_keep
    floor = floor * air.floor
    down_f = floor * p.floor_front + q * p.cla_front * yaw_keep * air.down_front
    down_r = floor * (1 - p.floor_front) + q * p.cla_rear * yaw_keep * air.down_rear

    # axle loads: static (times the road's load factor: crests, dips, banking) + downforce + longitudinal weight
    # transfer (from the accelerations a moment ago)
    static_f, static_r = m * G * p.b / L, m * G * p.a / L
    front = static_f * slope.load + down_f - m * s.ax * h / L
    rear = static_r * slope.load + down_r + m * s.ax * h / L

    # suspension: the extra load squashes the springs; past the ride height the plank takes the rest
    lift = surf.lift.reshape(2, 2).max(axis=1)                       # a kerb under either tyre lifts the axle
    squash = (jnp.stack([front, rear]) - jnp.stack([static_f, static_r])) / (2 * p.spring)
    ride = p.ride_height - squash - lift
    plank = jnp.maximum(-ride, 0.0) * 2 * p.spring                    # newtons the springs can't take
    front, rear = front - plank[0], rear - plank[1]

    lat = m * s.ay * h / tw  # ay > 0 (turning left) loads the right-hand tyres
    load = jnp.maximum(jnp.stack([front / 2 - p.roll_front * lat, front / 2 + p.roll_front * lat,
                                  rear / 2 - (1 - p.roll_front) * lat, rear / 2 + (1 - p.roll_front) * lat]), 0.0)

    # each tyre's peak grip: load-sensitive, times the surface under it (stiff springs lose grip on bumps)
    mu = p.mu * jnp.clip(1 - p.load_sens * (load / p.fz_nominal - 1), 0.3, 1 + p.load_sens)
    fmax = mu * load * surf.grip * (1 - surf.rough * p.bump_loss) * temp_grip(p, s.temp) * wear_grip(p, s.wear) + 1e-3

    # where each tyre is and how it's moving, in its own frame (front tyres turned by the steering)
    wx = jnp.stack([p.a, p.a, -p.b, -p.b])
    wy = jnp.stack([tw / 2, -tw / 2, tw / 2, -tw / 2])
    delta = jnp.stack([steer, steer, 0.0, 0.0])
    ux, uy = s.vx - s.r * wy, s.vy + s.r * wx                       # contact-patch velocity, body frame
    c, sn = jnp.cos(delta), jnp.sin(delta)
    u, w = ux * c + uy * sn, -ux * sn + uy * c                      # ... in each tyre's frame
    alpha = jnp.arctan2(w, jnp.maximum(jnp.abs(u), 1.0))            # slip angle (floored at walking pace)

    # what the driver asks of each tyre: drive (split front/rear by the layout, then across each axle by its
    # differential) or brake (by the bias)
    push, best = engine_force(p, s.vx, jnp.maximum(s.gear, 1))
    # reverse gear (F1 cars must have one): selected only at a standstill, and it stays in until the car stops again
    # with it released. It pushes like first gear (same ratio), backwards.
    rev = jnp.where(jnp.abs(s.vx) < 1.0, reverse > 0.5, s.gear == 0)
    back = -s.vx
    push_r = jnp.where(back <= p.gear_top[0], p.power / jnp.maximum(back, 0.8 * p.gear_top[0]), 0.0)
    push, best = jnp.where(rev, push_r, push), jnp.where(rev, 0, best).astype(jnp.int32)
    shifting = best != s.gear
    shift = jnp.where(shifting, p.shift_s, jnp.maximum(s.shift - dt, 0.0))
    push = jnp.where(shift > 0, 0.0, push)  # the clutch is open while the gear changes
    push = jnp.where(s.fuel_used < p.fuel_tank, push, 0.0)  # out of fuel: the engine stops
    throttle, brake = jnp.maximum(pedal, 0.0), jnp.maximum(-pedal, 0.0)
    gt = p.gear_top[jnp.maximum(best, 1) - 1]
    engaged = (shift <= 0) & (s.vx > 0.8 * p.gear_top[0]) & ~rev   # clutch closed: not shifting, not slipping at a launch
    revs = jnp.clip(s.vx / gt, 0.0, 1.0)
    engine_drag = jnp.where(engaged, p.engine_brake * p.power / gt * revs, 0.0)
    total = jnp.where(rev, -throttle * push, throttle * push - (1 - throttle) * engine_drag)
    cap = p.mu_long * fmax
    fl, fr = _axle_split(total * p.drive_front, cap[0], cap[1], p.diff_bias)
    rl, rr = _axle_split(total * (1 - p.drive_front), cap[2], cap[3], p.diff_bias)
    drive = jnp.stack([fl, fr, rl, rr])
    bshare = jnp.stack([p.brake_front / 2, p.brake_front / 2, (1 - p.brake_front) / 2, (1 - p.brake_front) / 2])
    stop_cap = m * jnp.abs(s.vx) / dt                               # brakes can stop the car, never reverse it
    braking = brake * jnp.minimum(p.brake_force, stop_cap) * bshare
    fx_ask = drive - braking * jnp.sign(u + 1e-6)

    # friction ellipse: longitudinal first, lateral gets what's left; over the limit the tyre slides
    fy_pure = -fmax * _magic(p, alpha)
    fmax_x = p.mu_long * fmax
    ratio = fx_ask / fmax_x
    sliding = jnp.abs(ratio) > 1.0
    gripping_fy = fy_pure * jnp.sqrt(jnp.maximum(1 - ratio ** 2, 0.0))
    locked = sliding & ((fx_ask < 0) != rev)  # against the gear's drive: braking (in reverse, a push forward)
    vmag = jnp.hypot(u, w) + 1e-3                                   # a locked tyre: friction against its motion
    lock_fx, lock_fy = -p.slide * fmax_x * u / vmag, -p.slide * fmax * w / vmag
    spin_fx = p.slide * fmax_x * jnp.sign(fx_ask)                   # a spinning tyre: most grip goes to pushing
    spin_fy = fy_pure * jnp.sqrt(1 - p.slide ** 2)
    fx = jnp.where(sliding, jnp.where(locked, lock_fx, spin_fx), fx_ask)
    fy = jnp.where(sliding, jnp.where(locked, lock_fy, spin_fy), gripping_fy)

    # tyre forces -> body frame; add drag, rolling resistance and the plank scraping
    bfx, bfy = fx * c - fy * sn, fx * sn + fy * c
    roll_coef = p.roll_resist * (1 + speed * 3.6 / 160) + surf.rolling  # (4,): grows with speed; grass adds more

    # tyre temperature: friction work (sideways slip; lengthways a little slip when gripping, all of it when locked,
    # a guess when spinning) and rolling flex heat it; the airflow cools it
    spinning = sliding & ~locked
    slip_x = jnp.where(locked, jnp.abs(u), jnp.where(spinning, 0.2 * jnp.abs(u) + 2.0, 0.03 * jnp.abs(ratio) * jnp.abs(u)))
    work = jnp.abs(fy) * jnp.abs(w) + jnp.abs(fx) * slip_x
    heat = p.heat_share * work + roll_coef * load * speed
    cool = p.tyre_cool * p.tyre_size * (1 + speed / p.tyre_cool_v) * (s.temp - p.temp_air)
    temp = s.temp + (heat - cool) / (p.tyre_heat_cap * p.tyre_size) * dt
    hot = 2.0 ** (jnp.maximum(s.temp - p.temp_best, 0.0) / p.wear_hot)
    wear = jnp.minimum(s.wear + p.wear_rate * work * hot / p.tyre_size * dt, 1.0)
    cda = p.cda * air.drag + p.cda_side * jnp.sin(beta) ** 2              # ... and more drag
    resist = (jnp.sum(roll_coef * load) + q * cda + p.plank_mu * plank.sum())
    dirx = s.vx / (speed + 0.5)                                      # resistance acts against the motion
    diry = s.vy / (speed + 0.5)
    fx_tot = jnp.sum(bfx) - resist * dirx
    fy_tot = jnp.sum(bfy) - resist * diry
    moment = jnp.sum(wx * bfy - wy * bfx)

    # integrate (semi-implicit Euler)
    m_long = m + p.spin_wheels + jnp.where(engaged, p.spin_engine / gt ** 2, 0.0)  # spinning parts speed up too
    vx = s.vx + ((fx_tot + m * slope.gx) / m_long + s.r * s.vy) * dt  # gravity along the slope moves the car ...
    vy = s.vy + ((fy_tot + m * slope.gy) / m - s.r * s.vx) * dt     # ... but the springs only feel the tyres
    r = s.r + moment / p.yaw_inertia * dt
    way = jnp.where(rev, -1.0, 1.0)  # braking to a stop doesn't roll the other way (or against the gear from a standstill)
    vx = jnp.where((brake > 0) & ((s.vx * vx < 0) | ((s.vx == 0) & (vx * way < 0))), 0.0, vx)

    # walking pace: blend into the kinematic (no-slip) model
    blend = jnp.clip((jnp.abs(vx) - 3.0) / 3.0, 0.0, 1.0)
    vy = blend * vy + (1 - blend) * vx * jnp.tan(steer) * p.b / L
    r = blend * r + (1 - blend) * vx * jnp.tan(steer) / L

    yaw = s.yaw + r * dt
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    pos = s.pos + jnp.stack([vx * cy - vy * sy, vx * sy + vy * cy]) * dt

    felt_x, felt_y = fx_tot / m_long, fy_tot / m
    k = dt / (dt + p.settle)  # soft springs: the weight takes longer to move
    burned = p.fuel_per_j * jnp.maximum(total * s.vx, 0.0) * dt  # the work at the wheels (either way), as fuel
    new = CarState(pos, yaw, vx, vy, r, steer, s.ax + (felt_x - s.ax) * k, s.ay + (felt_y - s.ay) * k, best, shift,
                   ride, temp, wear, s.fuel_used + burned)
    return new, CarInfo(load, sliding, best, felt_x, felt_y, plank, down_f + down_r)


def drive(p: CarParams, s: CarState, action, surf: Surface = TARMAC, air: Air = CLEAN_AIR, dt: float = 0.05,
          substeps: int = 10, slope: Slope = FLAT):
    """One decision: action = (steer, pedal[, reverse]), each in [-1, 1], held for dt seconds, on `surf` in `air` on
    `slope`. reverse > 0.5 at a standstill selects reverse gear (then the throttle drives backwards); without it, never.
    Returns the new state and the last substep's CarInfo."""
    reverse = action[2] if action.shape[0] > 2 else 0.0

    def body(_, carry):
        st, _info = carry
        return _substep(p, st, action[0], action[1], surf, air, slope, dt / substeps, reverse)

    # the first substep outside the loop gives the loop's carry real (per-device) types under shard_map
    first = _substep(p, s, action[0], action[1], surf, air, slope, dt / substeps, reverse)
    return jax.lax.fori_loop(1, substeps, body, first)


# Reference cars for the reality checks (built from published specs, see the episode 14 devlog).
PORSCHE_911 = (design(power_kw=283, wing_front=0.0, wing_rear=0.0, tyre_front_mm=245, tyre_rear_mm=305,
                      wheelbase=2.45, engine_pos=1.0, drive_front=0.0, brake_front=0.62, roll_front=0.55,
                      top_gear_kmh=310, ride_height_mm=110, spring_n_mm=35), ROAD)
F1_LIKE = (design(power_kw=740, wing_front=1.0, wing_rear=1.0, tyre_front_mm=305, tyre_rear_mm=405, wheelbase=3.6,
                  engine_pos=0.6, drive_front=0.0, brake_front=0.58, roll_front=0.5, top_gear_kmh=350,
                  ride_height_mm=30, spring_n_mm=300), RACE)   # stiff, as real ground-effect cars (stays above the stall)
