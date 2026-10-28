"""Race day (episode 15): several cars racing on one circuit, built on `race.RaceGame` (which stays untouched).

A race starts from a staggered grid behind the start line (two columns, `grid_gap` metres between grid slots) and
runs a number of laps. When the leader finishes, the chequered flag is out and every other car finishes the next
time it crosses the line (real rules). A car that hits the wall is out of the race and stops where it is.

Contact: every car is a box (wheelbase + overhangs long, track + tyre wide). After each decision, every overlapping
pair gets a rigid-body impulse at the contact point (bounce + body-on-body friction), so a hit at the rear corner spins
a car the way a real tap does, and the boxes are pushed apart. Out cars are fixed obstacles where they stopped until the
marshals recover them (`recover_s`, then ghosts); finished cars are ghosts (in real life they peel off on the slow-down
lap).

Damage (physics only, no reward penalty): the energy a hit destroys goes to the part of each car it landed on (nose:
front wing; tail: rear wing; a wheel: puncture, and a bent steering arm at the front; the side between: the floor).
Broken parts change the car through `race.RaceGame.step(..., damage)`; one big enough hit retires the car.

Fuel: every car starts with an allowance below what it would burn flat out over the race distance (`fuel_share` of
its engine's flat-out burn over the race, calibrated on the ep14 driver), like F1's 110 kg limit: the driver has to save
some (lift and coast). An empty tank stops the engine and retires the car: a real driver pulls off the racing line,
so it leaves the circuit at once (a ghost, like a finished car; step 9l).

Pit stops: a lane beside each circuit's straightest stretch (`race.RaceConfig.pit_lane`: behind a pit wall, an automatic 80 km/h limiter)
with a box for each car, spaced along it. A car that stops in its own box is held there while the crew works: new
tyres (`Field.plan`'s compound, a flat fixed too) in `tyre_s`, plus `nose_s` for a new front wing if it's broken. No
refuelling (F1 rules). The time a stop costs is the lane: 80 km/h instead of racing speed, and the stop itself. The
team covers a stop it called (`Field.box_call`): stopping on its mark pays the driver back a typical stop's time at its
own race pace (`stop_pay_s`), so answering the call is never a loss for the driver; whether the stop was worth it is
the strategist's question (a reward term, not physics).

Senses (`RaceField.observe`): each car's own view from `race.RaceGame.observe`, with its lasers stopped by other cars'
bodies as well as walls; then the nearest two cars ahead and two behind along the track (gap, side offset, speed
difference); then what's broken on its own car; then its tyre compound and how much of the race is left (to judge its
fuel); then the pits (called in?, in the lane, its box (a square-root scale: sharp near it), the entry ahead, the stop's time left). A new observation, so drivers trained alone can't load it.

Every leaf of `Field.cars` is a `race.State` with a leading (n_cars,) axis, so a batch of races is just one more
vmap. Pure functions, nothing drawn here.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ailearns.games import race
from ailearns.games import race_car as rc


SLOTS = ("ahead", "ahead", "behind", "behind")  # the nearest cars each driver senses, nearest first
PARTS = ("wing_front", "wing_rear", "floor", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")


@dataclass(frozen=True)
class DamageConfig:
    """Joules (absorbed by this car in one hit) it takes to break each part. Estimates, sized against the FIA crash
    tests the survival cell must pass (780 kg at 15 m/s head-on ~88 kJ, 10 m/s side-on ~39 kJ, 11 m/s rear ~47 kJ):
    the wings and suspension give up long before the cell does."""
    wing_front_j: float = 3000.0  # the whole front wing gone (a 2 m/s nudge into a gearbox: ~10% of it)
    wing_rear_j: float = 6000.0   # the rear wing sits on the rear crash structure: sturdier
    floor_j: float = 10000.0      # the whole floor's downforce gone
    toe_deg_per_kj: float = 0.5   # a front-wheel hit bends the steering arm ...
    toe_max_deg: float = 5.0      # ... up to this
    puncture_j: float = 1500.0    # a wheel hit this hard cuts the tyre
    suspension_j: float = 8000.0  # a wheel hit this hard breaks the suspension: out
    retire_j: float = 25000.0     # this much in one hit wrecks the car: out
    wheel_reach: float = 0.45     # metres around a wheel's centre that count as hitting it (tyre radius ~0.33)


TYRE_SETS = 4  # sets of tyres a car's failure points are drawn for (a set per stop; the last repeats after 3 stops)


class Field(NamedTuple):
    cars: race.State       # (n_cars, ...) per leaf
    finished: jax.Array    # (n_cars,) took the chequered flag
    finish_t: jax.Array    # (n_cars,) seconds at the finish (inf until then)
    flag: jax.Array        # the leader has finished
    flag_laps: jax.Array   # (n_cars,) laps each car had done when the flag came out
    bump: jax.Array        # (n_cars,) contact impulse this decision, newton-seconds (0: no contact)
    hit: jax.Array         # (n_cars, len(PARTS)) joules each part took this decision
    damage: race.Damage    # (n_cars, ...) what's broken so far
    wrecked: jax.Array     # (n_cars,) out from a big hit: slides to a stop (brakes locked), then parked
    setup: race.Setup      # (n_cars, ...) each car's tyres and fuel load
    plan: jax.Array        # (n_cars,) the compound fitted at its next stop
    box_call: jax.Array    # (n_cars,) the team has called it in (set from outside: the strategist)
    service: jax.Array     # (n_cars,) seconds of work left in its box (0: not stopped)
    served: jax.Array      # (n_cars,) stopped in its box on this visit (no second stop before it drives away)
    stops: jax.Array       # (n_cars,) pit stops made
    t: jax.Array           # decisions taken
    done: jax.Array        # every car finished or out, or the time cap
    burst_at: jax.Array    # (n_cars, TYRE_SETS, 4) the wear at which each tyre of each set fails (drawn on the grid)
    flat_m: jax.Array      # (n_cars, 4) metres driven on each flat tyre (0: not flat)
    out_s: jax.Array       # (n_cars,) seconds stopped out of the race (the marshals recover it at `recover_s`)


class RaceField:
    def __init__(self, tracks: race.Tracks, n_cars: int = 8, laps: int = 25,
                 cfg: race.RaceConfig = race.RaceConfig(time_limit=3600.0, pit_lane=True), spec: rc.CarSpec = rc.RACE,
                 grid_gap: float = 8.0, grid_side: float = 3.0, pole_back: float = 5.0, contact: bool = True,
                 restitution: float = 0.2, body_friction: float = 0.4, contact_iters: int = 4,
                 damage: DamageConfig | None = DamageConfig(), slipstream: bool = True,
                 wake: race.WakeConfig = race.WakeConfig(), sense_range: float = 150.0,
                 fuel_share: float = 0.92, fuel_speed: float = 62.0, box_gap: float = 12.0, box_half: float = 2.5,
                 tyre_s: float = 2.5, nose_s: float = 8.0, stop_pay_s: float = 21.0,
                 burst_wear: tuple = (0.9, 1.0), shred_m: float = 2000.0, shred_floor: float = 0.25,
                 rim_m: float = 4000.0, gp_km: float = 0.0, tyre_cal: float = 1.0, recover_s: float = 10.0):
        self.game = race.RaceGame(tracks, cfg, spec)
        self.tracks, self.cfg = tracks, cfg
        self.n, self.laps = n_cars, laps
        self.grid_gap, self.grid_side, self.pole_back = grid_gap, grid_side, pole_back
        self.contact, self.restitution, self.body_friction, self.contact_iters = (
            contact, restitution, body_friction, contact_iters)
        self.dmg = damage
        self.slipstream, self.wake_cfg = slipstream, wake
        self.sense_range = sense_range  # metres along the track a driver notices another car
        self.fuel_share, self.fuel_speed = fuel_share, fuel_speed
        # (step 9e, owner: "scale tyre wear based on our sim distance") tyres wear x `gp_km` / this race's distance, so a
        # set lasts the same share of our race as of a Grand Prix (F1: ~305 km), as F1 games do for short races; and x
        # `tyre_cal`, calibrating our wear to real lifespans with the race driver of the day (race-fin-5: softs to the
        # cliff in ~190 km; Pirelli's softs ~100 km of a GP: 1.9). gp_km 0: off (wear as before)
        self.gp_km, self.tyre_cal = gp_km, tyre_cal
        # (step 9l, owner: "the cars wreck in the pits") a stopped car stayed put all race, often on the racing line or in
        # the pit lane, and 27 of 39 wrecks in 8 races were cars running into an earlier wreck. Real marshals clear a car
        # (crane, or the crew pushes it out of the lane) within a minute or two: after `recover_s` stopped it's gone
        # (a ghost, like a finished car). 0: never (as before). Owner (2026-10-27): 10 s, faster than real marshals (60 s
        # still let cars on their next lap hit the wreck); yellow flags, the real answer, are for the next episode
        self.recover_s = recover_s
        # boxes: centred on the pit lane's middle, `box_gap` apart along the lane (F1 garages ~12-15 m wide); a stop counts with
        # the car's centre within `box_half` of its box mark. Stop times: F1 tyre stops ~2-3 s, a nose ~8-10 s more.
        self.box_s = (jnp.arange(n_cars) - (n_cars - 1) / 2) * box_gap
        self.box_half, self.tyre_s, self.nose_s = box_half, tyre_s, nose_s
        # the team covers a stop it called: the driver is paid back this many seconds at its own race pace when it stops
        # on its mark (a stop costs ~19 s, scripts/race_pitstop.py; +10% so a clean stop pays, a sloppy one doesn't)
        self.stop_pay_s = stop_pay_s
        # (step 8l, owner: a fully worn tyre should fail) a tyre worn through its tread fails: each tyre gets a failure
        # point uniform in `burst_wear` when it's fitted (the grip cliff starts at 0.8; wear 1 is no tread left), and
        # reaching it punctures it (the same puncture a hit makes: a stop fixes it). What follows is the physics: limp
        # back on a straight, or spin off in a fast corner. Real cause: Silverstone 2020, three failures on the last laps
        # of very long stints. The range is our choice (unsourced). Drawn on the grid, so a race is deterministic.
        self.burst_wear = burst_wear
        # (step 8o, owner: cars mustn't crawl to the flag on flat tyres) a flat tyre shreds after `shred_m` metres, and
        # the flailing rubber takes `shred_floor` of the floor's downforce; the car then runs on the rim, and after
        # `rim_m` more metres on it the wheel or the suspension breaks: out (wrecked: it slides to a stop). Real cars
        # nurse a flat back to the pits, ~a lap, often with the floor damaged; nobody races on rims. Distances are our
        # estimates (unsourced); the flat's grip stays `race.RaceConfig.flat_grip` (0.4: a rim on tarmac is ~that).
        self.shred_m, self.shred_floor, self.rim_m = shred_m, shred_floor, rim_m

    def fuel_allowance(self, track, designs) -> jax.Array:
        """(n_cars,) kg each car starts with: `fuel_share` of its full-power burn for the class's full-throttle share
        of a race run at `fuel_speed` m/s, plus the reserve. `fuel_speed` is calibrated so a share of 1 is what the
        ep14 driver burns flat out in an F1-like car (0.47 kg/km, `scripts/race_compounds.py`; F1's ~0.36 kg/km is a
        hybrid at ~50% thermal efficiency, ours burn petrol at 35%). Re-measure with the race-trained driver."""
        spec = self.game.spec
        race_s = self.laps * self.tracks.length[track] / self.fuel_speed
        burn = 1000.0 * designs[:, rc.D["power_kw"]] / (spec.thermal_eff * rc.PETROL_J_PER_KG)
        return spec.fuel_kg + self.fuel_share * burn * spec.full_throttle_share * race_s

    def grid(self, key, track, designs, compounds=None, fuel=None) -> Field:
        """The cars on the grid of circuit `track`, pole first; `designs`: (n_cars, len(rc.DESIGN)); `compounds`:
        (n_cars,) the tyres each starts on (indices into rc.COMPOUNDS; default mediums); `fuel`: (n_cars,) kg each
        starts with (default the allowance)."""
        L = self.tracks.length[track]
        designs = jnp.asarray(designs, jnp.float32)

        def place(k, d, design):
            back = self.pole_back + self.grid_gap * k
            side = jnp.where(k % 2 == 0, self.grid_side, -self.grid_side)  # pole on the left, then alternate
            st = self.place(d, track, design, L - back, side)
            return st._replace(progress=st.lap_pos - L)

        k = jnp.arange(self.n)
        cars = jax.vmap(place)(k, jax.random.split(key, self.n), designs)
        n = self.n
        compounds = jnp.full(n, rc.COMPOUNDS.index("medium")) if compounds is None else compounds
        fuel = self.fuel_allowance(track, designs) if fuel is None else fuel
        compounds = jnp.asarray(compounds, jnp.int32)
        wear = (jnp.full(n, self.tyre_cal * self.gp_km * 1000.0 / (self.laps * L)) if self.gp_km
                else jnp.ones(n))
        lo, hi = self.burst_wear
        burst_at = jax.random.uniform(jax.random.fold_in(key, 7), (n, TYRE_SETS, 4), minval=lo, maxval=hi)
        return Field(cars, jnp.zeros(n, bool), jnp.full(n, jnp.inf), jnp.asarray(False), jnp.zeros(n),
                     jnp.zeros(n), jnp.zeros((n, len(PARTS))), jax.tree.map(lambda a: jnp.stack([a] * n), race.NO_DAMAGE),
                     jnp.zeros(n, bool), race.Setup(compounds, jnp.asarray(fuel, jnp.float32), wear), compounds,
                     jnp.zeros(n, bool), jnp.zeros(n), jnp.zeros(n, bool), jnp.zeros(n, jnp.int32),
                     jnp.asarray(0), jnp.asarray(False), burst_at, jnp.zeros((n, 4)), jnp.zeros(n))

    def place(self, key, track, design, metres, side, speed=0.0) -> race.State:
        """One car `metres` along the lap of circuit `track`, `side` metres left of the centreline, facing along the
        track at `speed` m/s, tyres off the blankets. Its progress is its lap position (lap 1)."""
        g, P = self.game, self.cfg.points
        L = self.tracks.length[track]
        c, h = self.tracks.center[track], self.tracks.heading[track]
        s = (metres % L) / L * P
        i = jnp.floor(s).astype(jnp.int32) % P
        u = s - jnp.floor(s)
        yaw = h[i]
        normal = jnp.stack([-jnp.sin(yaw), jnp.cos(yaw)])
        pos = c[i] + u * (c[(i + 1) % P] - c[i]) + side * normal
        st = g.init(key, design=design, track=track, start=i)
        car = rc.car_at(pos, yaw, speed, ride_height=design[rc.D["ride_height_mm"]] / 1000.0,
                        tyre_temp=self.cfg.tyre_start)
        idx, lat, lap_pos = g._locate(st.track, i, pos)
        return st._replace(car=car, idx=idx, lat=lat, lap_pos=lap_pos, progress=lap_pos)

    def laps_done(self, f: Field) -> jax.Array:
        return jnp.floor(f.cars.progress / self.tracks.length[f.cars.track])

    def step(self, f: Field, actions) -> tuple[Field, jax.Array]:
        """One decision for every car: actions (n_cars, 2). Returns the field and each car's reward."""
        moving = ~(f.cars.crashed | f.finished | self.dry_out(f))  # out at the wall, dry or finished: frozen; a wreck slides
        racing = moving & ~f.wrecked
        held = f.wrecked | (f.service > 0)  # a wreck locks its brakes; a car in its box is held while the crew works
        actions = jnp.where(held[:, None], jnp.zeros_like(actions).at[:, 1].set(-1.0), actions)
        air = self.air(f, moving) if self.slipstream else rc.CLEAN_AIR
        new, reward = jax.vmap(self.game.step, in_axes=(0, 0, 0, 0 if self.slipstream else None, 0))(
            f.cars, actions, f.damage, air, f.setup)
        cars = jax.tree.map(lambda a, b: jnp.where(moving.reshape(moving.shape + (1,) * (a.ndim - 1)), a, b),
                            new, f.cars)
        reward = jnp.where(racing, reward, 0.0)
        bump, hit, damage, wrecked = jnp.zeros(self.n), jnp.zeros((self.n, len(PARTS))), f.damage, f.wrecked
        if self.contact:
            cars, bump, hit = self.collide(cars, solid=~(f.finished | self.gone(f)), moving=moving, setup=f.setup)
            if self.dmg is not None:
                damage, wrecked_now = self.break_parts(f.damage, hit)
                wrecked = wrecked | (wrecked_now & racing)
        parked = wrecked & (jnp.abs(cars.car.vx) < 0.5)  # a stopped wreck stays put (an obstacle, like an out car)
        cars, damage, setup, plan, box_call, service, served, stops = self.pit(f, cars, damage, racing & ~wrecked)
        limit = f.burst_at[jnp.arange(self.n), jnp.minimum(stops, TYRE_SETS - 1)]  # this set's failure points
        damage = damage._replace(puncture=damage.puncture | ((cars.car.wear >= limit) & racing[:, None]))
        moved = jnp.hypot(cars.car.vx, cars.car.vy) * self.cfg.dt
        flat_m = jnp.where(damage.puncture, f.flat_m + moved[:, None] * racing[:, None], 0.0)  # a stop's new tyre: 0
        shredded = ((flat_m >= self.shred_m) & (f.flat_m < self.shred_m)).sum(-1)
        damage = damage._replace(floor=jnp.minimum(damage.floor + self.shred_floor * shredded, 1.0))
        wrecked = wrecked | (racing & (flat_m >= self.shred_m + self.rim_m).any(-1))  # a wheel or its suspension gone
        pace = jnp.maximum(f.cars.progress, 0.0) / jnp.maximum(f.t * self.cfg.dt, 1.0)  # m/s so far this race
        reward = reward + ((stops > f.stops) & f.box_call) * self.stop_pay_s * pace * self.cfg.progress_reward
        cars = cars._replace(crashed=cars.crashed | parked, done=cars.done | wrecked)
        t = f.t + 1
        laps = jnp.floor(cars.progress / self.tracks.length[cars.track])
        leader_done = racing & (laps >= self.laps)
        flag_now = ~f.flag & jnp.any(leader_done)
        flag_laps = jnp.where(flag_now, laps, f.flag_laps)
        flag = f.flag | flag_now
        finished = f.finished | (racing & ((laps >= self.laps) | (f.flag & (laps > flag_laps))))
        finish_t = jnp.where(finished & ~f.finished, t * self.cfg.dt, f.finish_t)
        dry = (cars.car.fuel_used >= setup.fuel) & ~finished
        done = jnp.all(finished | cars.crashed | wrecked | dry) | (t * self.cfg.dt >= self.cfg.time_limit)
        out_s = jnp.where(cars.crashed & ~finished, f.out_s + self.cfg.dt, 0.0)
        return Field(cars, finished, finish_t, flag, flag_laps, bump, hit, damage, wrecked, setup, plan, box_call,
                     service, served, stops, t, done, f.burst_at, flat_m, out_s), reward

    def dry_out(self, f: Field) -> jax.Array:
        """(n_cars,) retired with an empty tank (step 9l, owner: a dry car crawling on the racing line at ~45 km/h caused
        two wrecks in one race; a real driver pulls off at once)."""
        return (f.cars.car.fuel_used >= f.setup.fuel) & ~f.finished

    def gone(self, f: Field) -> jax.Array:
        """(n_cars,) no longer on the circuit (not solid, not seen): out cars the marshals have recovered, and dry cars
        (they pull off)."""
        recovered = (f.out_s >= self.recover_s) if self.recover_s > 0 else jnp.zeros(self.n, bool)
        return recovered | self.dry_out(f)

    def pit(self, f: Field, cars: race.State, damage: race.Damage, racing):
        """Pit stops: a car that stops in its own box gets new tyres (and a nose if it's broken) and is held there for
        the work's time."""
        g, cfg = self.game, self.cfg
        in_box = jax.vmap(g.in_pit_lane)(cars) & (jnp.abs(self.to_box(cars)) < self.box_half)
        arrive = racing & in_box & (jnp.hypot(cars.car.vx, cars.car.vy) < 1.0) & ~f.served & (f.service <= 0)
        nose = arrive & (damage.wing_front > 0.05)
        service = jnp.where(arrive, self.tyre_s + nose * self.nose_s, jnp.maximum(f.service - cfg.dt, 0.0))
        served = (f.served | arrive) & in_box  # cleared once it drives out of the box
        fresh = arrive[:, None]
        car = cars.car._replace(wear=jnp.where(fresh, 0.0, cars.car.wear),
                                temp=jnp.where(fresh, cfg.tyre_start, cars.car.temp))  # off the blankets
        damage = damage._replace(wing_front=jnp.where(nose, 0.0, damage.wing_front),
                                 puncture=damage.puncture & ~fresh)
        setup = f.setup._replace(compound=jnp.where(arrive, f.plan, f.setup.compound))
        return (cars._replace(car=car), damage, setup, f.plan, f.box_call & ~arrive, service, served,
                f.stops + arrive)

    def to_box(self, cars: race.State) -> jax.Array:
        """(n_cars,) metres from each car's centre to its box mark along the lap, -L/2 to L/2 (positive: still ahead),
        from its exact lap position (the nearest centreline point is 2-4 m coarse, more than the box's +-2.5 m)."""
        L = self.tracks.length[cars.track]
        s = (cars.lap_pos - self.game.pit_at[cars.track] + L / 2) % L - L / 2  # from the pit lane's middle
        return (self.box_s - s + L / 2) % L - L / 2

    @staticmethod
    def box_sense(d) -> jax.Array:
        """What a driver reads of `d` metres to its mark: a square-root scale, sharp near the box (1 m 0.1, 4 m 0.2,
        25 m 0.5, 100 m and beyond 1), the way a driver sees the box board come up (step 8f: /100 m read 1 and 3 m as
        0.01 and 0.03, and it stopped short)."""
        return jnp.sign(d) * jnp.sqrt(jnp.minimum(jnp.abs(d) / 100.0, 1.0))

    def air(self, f: Field, moving) -> rc.Air:
        """The air each car drives into this decision: every car ahead's wake (a car parked at the wall or finished
        leaves none; a sliding wreck still does)."""
        c = f.cars.car
        speed = jnp.where(moving, jnp.hypot(c.vx, c.vy), 0.0)
        return race.wake(c.pos, c.yaw, speed, self.wake_cfg)

    def break_parts(self, damage: race.Damage, hit) -> tuple[race.Damage, jax.Array]:
        """Add this decision's hits (n, len(PARTS)) joules to the damage. Returns it and which cars are wrecked."""
        c, P = self.dmg, {k: i for i, k in enumerate(PARTS)}
        wheels = hit[:, P["wheel_fl"]:]
        toe_max = jnp.deg2rad(c.toe_max_deg)
        toe = damage.toe + jnp.deg2rad(c.toe_deg_per_kj) * (wheels[:, 0] - wheels[:, 1]) / 1000.0  # pulls to the hit side
        new = race.Damage(
            wing_front=jnp.minimum(damage.wing_front + hit[:, P["wing_front"]] / c.wing_front_j, 1.0),
            wing_rear=jnp.minimum(damage.wing_rear + hit[:, P["wing_rear"]] / c.wing_rear_j, 1.0),
            floor=jnp.minimum(damage.floor + hit[:, P["floor"]] / c.floor_j, 1.0),
            toe=jnp.clip(toe, -toe_max, toe_max),
            puncture=damage.puncture | (wheels >= c.puncture_j))
        wrecked = (hit.sum(1) >= c.retire_j) | jnp.any(wheels >= c.suspension_j, axis=1)
        return new, wrecked

    # ── contact ──────────────────────────────────────────────────────────────

    def boxes(self, cars: race.State, setup: race.Setup | None = None):
        """Each car's physics (with its fuel load) and its body as a box: (front, rear, half width) in metres from the
        centre of mass."""
        spec = self.game.spec
        p = (jax.vmap(lambda d: rc.derive(d, spec))(cars.design) if setup is None
             else jax.vmap(self.game.params)(cars.design, setup))
        tyre = jnp.maximum(cars.design[:, rc.D["tyre_front_mm"]], cars.design[:, rc.D["tyre_rear_mm"]]) / 1000.0
        return p, p.a + spec.overhang_front, p.b + spec.overhang_rear, (p.track_width + tyre) / 2

    @staticmethod
    def outline(c: rc.CarState, front, rear, half_w):
        """Box centres, forward and left unit vectors (n, 2), half lengths and the corners (n, 4, 2): front-left,
        front-right, rear-left, rear-right."""
        cy, sy = jnp.cos(c.yaw), jnp.sin(c.yaw)
        ex, ey = jnp.stack([cy, sy], -1), jnp.stack([-sy, cy], -1)  # (n, 2) each car's forward and left
        half_l = (front + rear) / 2
        centre = c.pos + ((front - rear) / 2)[:, None] * ex  # the box centre isn't the centre of mass
        corners = (centre[:, None, :] + jnp.array([1, 1, -1, -1.0])[None, :, None] * half_l[:, None, None] * ex[:, None]
                   + jnp.array([1, -1, 1, -1.0])[None, :, None] * half_w[:, None, None] * ey[:, None])  # (n, 4, 2)
        return centre, ex, ey, half_l, corners

    def collide(self, cars: race.State, solid, moving, setup: race.Setup | None = None
                ) -> tuple[race.State, jax.Array, jax.Array]:
        """Resolve contact between the cars' boxes. `solid`: (n,) takes part at all; `moving`: (n,) can be moved
        (an out car is solid but stuck). Returns the cars, each one's contact impulse (N s) and the joules each of
        its parts took (n, len(PARTS))."""
        p, front, rear, half_w = self.boxes(cars, setup)
        c = cars.car
        inv_m = jnp.where(moving, 1.0 / (p.mass - c.fuel_used), 0.0)
        inv_i = jnp.where(moving, 1.0 / p.yaw_inertia, 0.0)
        centre, ex, ey, half_l, corners = self.outline(c, front, rear, half_w)

        def inside(pts, k):
            """Which of `pts` (4, 2) are inside car k's box."""
            d = pts - centre[k]
            return (jnp.abs(d @ ex[k]) < half_l[k]) & (jnp.abs(d @ ey[k]) < half_w[k])

        def overlap(i, j):
            """Separating-axis test: penetration depth, normal (from i to j), contact point."""
            axes = jnp.stack([ex[i], ey[i], ex[j], ey[j]])
            rad = lambda k, u: half_l[k] * jnp.abs(u @ ex[k]) + half_w[k] * jnp.abs(u @ ey[k])
            dist = axes @ (centre[j] - centre[i])
            depth = jax.vmap(lambda u: rad(i, u) + rad(j, u))(axes) - jnp.abs(dist)
            a = jnp.argmin(depth)
            normal = axes[a] * jnp.where(dist[a] < 0, -1.0, 1.0)
            ins = jnp.concatenate([inside(corners[i], j), inside(corners[j], i)])
            pts = jnp.concatenate([corners[i], corners[j]])
            point = jnp.where(ins.any(), (pts * ins[:, None]).sum(0) / jnp.maximum(ins.sum(), 1),
                              (centre[i] + centre[j]) / 2)  # edges crossing with no corner inside: rare
            return depth[a], normal, point

        n = self.n
        ii, jj = jnp.triu_indices(n, 1)
        depth, normal, point = jax.vmap(overlap)(ii, jj)
        touching = (depth > 0) & solid[ii] & solid[jj] & (moving[ii] | moving[jj])

        cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        perp = lambda w, r: w[..., None] * jnp.stack([-r[..., 1], r[..., 0]], -1)  # w x r
        ra, rb = point - c.pos[ii], point - c.pos[jj]
        v = c.vx[:, None] * ex + c.vy[:, None] * ey  # world velocities
        w = c.r
        e, mu = self.restitution, self.body_friction

        def closing(v, w):
            rel = (v[jj] + perp(w[jj], rb)) - (v[ii] + perp(w[ii], ra))
            vn = jnp.sum(rel * normal, -1)
            return rel, vn

        rel0, vn0 = closing(v, w)
        slip0 = jnp.linalg.norm(rel0 - vn0[:, None] * normal, axis=-1)

        def solve(_, carry):
            v, w, total, jn_sum, jt_sum = carry
            rel, vn = closing(v, w)
            k_n = inv_m[ii] + inv_m[jj] + cross(ra, normal) ** 2 * inv_i[ii] + cross(rb, normal) ** 2 * inv_i[jj]
            jn = jnp.where(touching & (vn < 0), -(1 + e) * vn / jnp.maximum(k_n, 1e-9), 0.0)
            tang = rel - vn[:, None] * normal
            tn = jnp.linalg.norm(tang, axis=-1)
            t = tang / jnp.maximum(tn, 1e-6)[:, None]
            k_t = inv_m[ii] + inv_m[jj] + cross(ra, t) ** 2 * inv_i[ii] + cross(rb, t) ** 2 * inv_i[jj]
            jt = -jnp.minimum(tn / jnp.maximum(k_t, 1e-9), mu * jn)  # rubbing, up to Coulomb's limit
            imp = jn[:, None] * normal + jt[:, None] * t  # on car j; car i gets the opposite
            dv = (jnp.zeros_like(v).at[jj].add(imp * inv_m[jj, None]).at[ii].add(-imp * inv_m[ii, None]))
            dw = (jnp.zeros(n).at[jj].add(cross(rb, imp) * inv_i[jj]).at[ii].add(-cross(ra, imp) * inv_i[ii]))
            size = jnp.linalg.norm(imp, axis=-1)
            total = total.at[ii].add(size).at[jj].add(size)
            return v + dv, w + dw, total, jn_sum + jn, jt_sum + jnp.abs(jt)

        z = jnp.zeros_like(depth)  # zeros_like, not zeros: inside shard_map the loop's carry must vary like its inputs
        v, w, total, jn_sum, jt_sum = jax.lax.fori_loop(0, self.contact_iters, solve, (v, w, jnp.zeros_like(w), z, z))

        # the energy the hit destroyed: crushing (what the bounce doesn't give back) plus rubbing; half to each car
        energy = 0.5 * jn_sum * jnp.maximum(-vn0, 0.0) * (1 - e) + 0.5 * jt_sum * slip0
        a, b, half_t = p.a, p.b, p.track_width / 2
        reach = self.dmg.wheel_reach if self.dmg is not None else 0.45

        def part(k, pt):
            """Which part of car k a hit at world point `pt` lands on (index into PARTS)."""
            d = pt - c.pos[k]
            lx, ly = d @ ex[k], d @ ey[k]
            at_side = jnp.abs(ly) > half_t[k] - 0.3
            wheel = jnp.where(lx > 0, jnp.where(ly > 0, 3, 4), jnp.where(ly > 0, 5, 6))
            near_wheel = at_side & (jnp.minimum(jnp.abs(lx - a[k]), jnp.abs(lx + b[k])) < reach)
            return jnp.where(near_wheel, wheel, jnp.where(lx > a[k], 0, jnp.where(lx < -b[k], 1, 2)))

        hit = (jnp.zeros((n, len(PARTS)))
               .at[ii, jax.vmap(part)(ii, point)].add(energy / 2)
               .at[jj, jax.vmap(part)(jj, point)].add(energy / 2))

        # push the boxes apart (by mass share), leaving a hair of overlap so contact persists while pressed together
        push = jnp.where(touching, jnp.maximum(depth - 0.01, 0.0), 0.0)
        share = inv_m[ii] + inv_m[jj]
        si, sj = inv_m[ii] / jnp.maximum(share, 1e-9), inv_m[jj] / jnp.maximum(share, 1e-9)
        shift = (jnp.zeros_like(v).at[ii].add(-(push * si)[:, None] * normal)
                 .at[jj].add((push * sj)[:, None] * normal))
        pos = c.pos + shift

        car = c._replace(pos=pos, vx=jnp.sum(v * ex, -1), vy=jnp.sum(v * ey, -1), r=w)
        idx, lat, lap_pos = jax.vmap(self.game._locate)(cars.track, cars.idx, pos)
        L = self.tracks.length[cars.track]
        moved = (lap_pos - cars.lap_pos + L / 2) % L - L / 2
        return cars._replace(car=car, idx=idx, lat=lat, lap_pos=lap_pos, progress=cars.progress + moved), total, hit

    # ── what each driver senses ──────────────────────────────────────────────

    @property
    def obs_size(self) -> int:
        return self.cfg.obs_size + 4 * len(SLOTS) + 8 + len(rc.COMPOUNDS) + 1 + 5

    def car_lasers(self, f: Field) -> jax.Array:
        """(n, rays): metres along each car's lasers to the nearest other car's body (ray_range: none). Finished cars
        are ghosts and don't show, nor do recovered ones; cars out of the race are there until recovered."""
        _, front, rear, half_w = self.boxes(f.cars)
        *_, corners = self.outline(f.cars.car, front, rear, half_w)
        a = corners[:, jnp.array([0, 2, 0, 1])].reshape(-1, 2)  # (n * 4, 2) edges: front, rear, left, right
        b = corners[:, jnp.array([1, 3, 2, 3])].reshape(-1, 2)
        owner = jnp.repeat(jnp.arange(self.n), 4)
        shows = jnp.repeat(~(f.finished | self.gone(f)), 4)
        c = f.cars.car

        def one(i):
            ang = c.yaw[i] + self.game.rays
            d = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)[:, None, :]
            e, w = b - a, a - c.pos[i]
            den = race._cross(d, e)
            safe = jnp.where(jnp.abs(den) < 1e-9, 1e-9, den)
            t, u = race._cross(w, e) / safe, race._cross(w, d) / safe
            hit = (jnp.abs(den) >= 1e-9) & (t > 0) & (u >= 0) & (u <= 1) & (owner != i) & shows
            return jnp.minimum(jnp.min(jnp.where(hit, t, jnp.inf), axis=1), self.cfg.ray_range)
        return jax.vmap(one)(jnp.arange(self.n))

    def neighbours(self, f: Field) -> jax.Array:
        """(n, len(SLOTS), 4): the nearest cars ahead and behind along the track, nearest first. Each: there (0/1),
        gap along the track (/ sense_range, negative behind), how far to the side (/ half the track width, left
        positive) and how much faster than me along the track (m/s / 20). Lap counts don't matter: a car being
        lapped right in front is right in front."""
        cars, cfg = f.cars, self.cfg
        L = self.tracks.length[cars.track]
        gap = (cars.lap_pos[None, :] - cars.lap_pos[:, None] + L[:, None] / 2) % L[:, None] - L[:, None] / 2  # (i, j)
        h = self.tracks.heading[cars.track, cars.idx]
        along = cars.car.vx * jnp.cos(cars.car.yaw - h) - cars.car.vy * jnp.sin(cars.car.yaw - h)
        seen = (~jnp.eye(self.n, dtype=bool)) & ~(f.finished | self.gone(f))[None, :] & (jnp.abs(gap) < self.sense_range)
        feats = jnp.stack([jnp.ones_like(gap), gap / self.sense_range,
                           (cars.lat[None, :] - cars.lat[:, None]) / (cfg.width / 2),
                           (along[None, :] - along[:, None]) / 20.0], -1)

        def pick(ahead):
            ok = seen & ((gap >= 0) if ahead else (gap < 0))
            order = jnp.argsort(jnp.where(ok, jnp.abs(gap), jnp.inf), axis=1)[:, :SLOTS.count("ahead" if ahead else "behind")]
            got = jnp.take_along_axis(ok, order, 1)
            return jnp.where(got[..., None], jnp.take_along_axis(feats, order[..., None], 1), 0.0)
        return jnp.concatenate([pick(True), pick(False)], 1)

    def observe(self, f: Field) -> jax.Array:
        """(n_cars, obs_size): each car's own view, its lasers now stopped by other cars too, then the cars around it
        and what's broken on its own car (a driver feels a lost wing or a flat), which tyres it's on and how much of
        the race is left (its fuel left is in its own view), and the pits: called in, in the lane, how far to its box
        (`box_sense`), how far ahead the pit entry is (/ 1 km) and the stop's work left (/ 10 s)."""
        own = jax.vmap(self.game.observe)(f.cars, f.setup)
        rays = len(self.cfg.ray_angles)
        lasers = jnp.minimum(own[:, :rays], self.car_lasers(f) / self.cfg.ray_range)
        d = f.damage
        broken = jnp.concatenate([jnp.stack([d.wing_front, d.wing_rear, d.floor,
                                             d.toe / jnp.deg2rad(self.dmg.toe_max_deg if self.dmg else 5.0)], -1),
                                  d.puncture.astype(jnp.float32)], -1)
        tyres = jax.nn.one_hot(f.setup.compound, len(rc.COMPOUNDS))
        left = jnp.clip(1 - f.cars.progress / (self.laps * self.tracks.length[f.cars.track]), 0.0, 1.0)[:, None]
        g, cars = self.game, f.cars
        s = g.from_pit(cars.track, cars.idx)
        L = self.tracks.length[cars.track]
        pits = jnp.stack([f.box_call.astype(jnp.float32), jax.vmap(g.in_pit_lane)(cars).astype(jnp.float32),
                          self.box_sense(self.to_box(cars)),
                          jnp.clip(((-self.cfg.pit_entry - s) % L) / 1000.0, 0.0, 1.0), f.service / 10.0], -1)
        return jnp.concatenate([lasers, own[:, rays:], self.neighbours(f).reshape(self.n, -1), broken, tyres, left,
                                pits], -1)

    def order(self, f: Field) -> jax.Array:
        """Each car's position (0 = leading): finishers by finish time, then the rest by distance, the out at the back."""
        group = jnp.where(f.finished, 0, jnp.where(f.cars.crashed | f.wrecked | self.dry_out(f), 2, 1))
        within = jnp.where(f.finished, f.finish_t, -f.cars.progress)
        return jnp.argsort(jnp.lexsort((within, group)))
