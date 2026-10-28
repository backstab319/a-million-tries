"""Race day (episode 15), step 8: the race school, where a driver learns among other cars.

A race is ~25 minutes (~30k decisions a car) and a driver's reward looks ~10 s ahead, so the school doesn't run whole
races. It runs **stretches** of `stretch_s` seconds, each a random moment of a race on a random circuit:
- `grid_share` of them start on the grid (the start is where the pile-ups are);
- the rest start mid-race: the cars in a pack somewhere on the lap (random order, gaps, sides and speed), on a random
  lap, with tyres worn and hot, fuel burned to match the laps done, and now and then something broken. So the driver
  sees late-race states (worn tyres, a light car, a lost wing) without driving 25 minutes to get there.
Every car is a random design (driver school's parts box) on a random compound.

The team calls cars in at random moments (`call_rate_s`, and `called` of the cars at a stretch's start), with a random
compound to fit: the driver's job is to answer the call (`RaceField`'s stop payment), the strategist's (step 9) is when.
Pit lessons (`pit_share` of mid-race stretches) start called cars before their boxes, so the driver meets the
payment early (it learned to stay out of the lane without them: step 8d). The pit coach (`coach`, off by default) points
called cars at their marks with potential-based shaping (step 8f: lesson starts alone made almost no stops).

A car's run ends when it hits the wall, is wrecked, finishes, or the stretch ends; the stretch ends at `stretch_s`, when
the race is over, or when no car is racing. Pure functions over `race_field.Field`, nothing drawn here.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ailearns.games import race
from ailearns.games import race_car as rc
from ailearns.games import race_field


@dataclass(frozen=True)
class SchoolConfig:
    stretch_s: float = 60.0      # seconds a stretch runs (driver school's runs: 60 s)
    grid_share: float = 0.2      # stretches that start on the grid
    pack_gap: float = 20.0       # mid-race: mean metres between cars beyond `min_gap` (exponential: some nose to tail)
    min_gap: float = 7.0         # a car is ~5.6 m long: spawned boxes never overlap
    pack_side: float = 3.0       # each car up to this many metres either side of the centreline
    start_speed: float = 50.0    # the pack's speed: up to this (m/s), each car within `speed_spread` of it
    speed_spread: float = 5.0
    pace: tuple = (45.0, 65.0)   # m/s the leader has averaged so far (the stop payment is at each car's pace)
    worn: float = 0.9            # tyre wear up to this (the cliff starts at 0.8)
    temp_spread: float = 10.0    # C around the compound's best (a mid-race tyre is in its window, give or take)
    fuel_spread: float = 0.15    # fuel burned: the laps done's share of the tank, +- this share
    damaged: float = 0.15        # mid-race: share of cars with something broken
    called: float = 0.15         # share of cars already called in when a stretch starts
    call_rate_s: float = 120.0   # mean seconds between a car's random box calls
    # pit lessons (step 8d: a driver that has never been paid for a stop learns to stay out of the lane): in this share
    # of mid-race stretches each car with chance `pit_car` starts called in, up to `pit_before` metres before its box
    # (in the lane when that's past the entry), slow enough to stop at `pit_decel` m/s^2
    pit_share: float = 0.5
    pit_car: float = 0.5
    pit_before: float = 600.0
    pit_decel: float = 5.0
    pit_close: float = 3.0       # distance before the box = pit_before x uniform ** this
    # the turn-in (race-field-5 stopped in the lane but never turned in from the racing line): in this share of lessons
    # the whole pack starts up to `approach_before` metres before the pit entry and each car with chance `pit_car` is
    # called from where it is racing (nothing moved, so nothing spawns on top of anything)
    pit_approach: float = 0.5
    approach_before: float = 300.0
    # (step 8k) mid-race stretches start the leader this many laps from the flag (0: on a random lap)
    last_laps: int = 0
    # the pit coach (step 8f; owner OK'd coaching in CQB: "a soldier in training is shown the ropes"; say so on screen):
    # while a car is called in, a potential that's highest stopped on its mark, paid as gamma x phi(after) - phi(before)
    # (potential-based shaping: nothing to farm, looping in and out earns nothing, and the best driving is unchanged;
    # it only points the way). phi = -(coach_d x metres to its mark (capped at coach_reach)
    #   + coach_lat x metres left of the lane line, ramped in over the coach_turn_in metres before the pit entry
    #   + coach_v x speed, ramped in over the last coach_brake metres). 0 when not called, out, or the run ended.
    coach: float = 0.0           # its weight (0: off)
    coach_gamma: float = 0.995   # the trainer's gamma
    coach_d: float = 0.01        # per metre: as much as the progress reward
    coach_reach: float = 600.0
    coach_lat: float = 0.3       # per metre left of the lane line
    coach_lane: float = -8.5     # the lane: right of this (it's -7 to -14 m)
    coach_turn_in: float = 150.0
    coach_v: float = 0.1         # per m/s
    coach_brake: float = 60.0


class Lesson(NamedTuple):
    field: race_field.Field
    t0: jax.Array          # the field's decision count when this stretch started
    start: jax.Array       # (n_cars,) each car's progress then (metres driven this stretch = progress - start)


class RaceSchool:
    def __init__(self, field: race_field.RaceField, cfg: SchoolConfig = SchoolConfig()):
        self.field, self.cfg = field, cfg
        self.n = field.n

    def racing(self, f: race_field.Field) -> jax.Array:
        """(n_cars,) still in the race: not out, wrecked or finished."""
        return ~(f.cars.crashed | f.finished | f.wrecked)

    def start(self, key) -> Lesson:
        """A new stretch: a random circuit, designs and compounds, on the grid or mid-race."""
        F, c, n = self.field, self.cfg, self.n
        kt, kd, kc, kg, km, kw, kp, kb = jax.random.split(key, 8)
        track = jax.random.randint(kt, (), 0, F.tracks.length.shape[0])
        lo, span = jnp.asarray(rc.DESIGN_LOW), jnp.asarray(rc.DESIGN_HIGH - rc.DESIGN_LOW)
        designs = lo + jax.random.uniform(kd, (n, len(rc.DESIGN))) * span
        compounds = jax.random.randint(kc, (n,), 0, len(rc.COMPOUNDS))
        grid = F.grid(kg, track, designs, compounds)
        grid = grid._replace(plan=jax.random.randint(kp, (n,), 0, len(rc.COMPOUNDS)), cars=self.shuffle(kb, grid.cars))
        mid = self.midrace(km, grid, track)
        f = jax.tree.map(lambda a, b: jnp.where(jax.random.uniform(kw) < c.grid_share, a, b), grid, mid)
        return Lesson(f, f.t, f.cars.progress)

    def shuffle(self, key, cars: race.State) -> race.State:
        """The cars swap grid slots at random (a trainer that drives the first few cars mustn't always start at the
        front)."""
        o = jax.tree.map(lambda a: a[jax.random.permutation(key, self.n)], cars)
        return cars._replace(car=cars.car._replace(pos=o.car.pos, yaw=o.car.yaw), idx=o.idx, lat=o.lat,
                             lap_pos=o.lap_pos, progress=o.progress)

    def midrace(self, key, f: race_field.Field, track) -> race_field.Field:
        """The grid's cars moved to a random moment of the race: a pack on a random lap, worn, fuelled to match,
        some damaged, some called in."""
        F, c, n = self.field, self.cfg, self.n
        spec = F.game.spec
        k = jax.random.split(key, 16)
        L = F.tracks.length[track]
        ka, kb, kc, kd = jax.random.split(jax.random.fold_in(key, 3), 4)
        lesson_on = jax.random.uniform(ka) < c.pit_share
        approach = lesson_on & (jax.random.uniform(kb) < c.pit_approach)  # the pack before the pit entry
        lead = jnp.where(approach, F.game.pit_at[track] - F.cfg.pit_entry - jax.random.uniform(kc) * c.approach_before,
                         jax.random.uniform(k[0]) * L) % L
        gaps = c.min_gap + c.pack_gap * jax.random.exponential(k[1], (n,))
        behind = jnp.cumsum(gaps) - gaps[0]  # pack slot k this far behind the leader
        slot = jax.random.permutation(k[2], n)  # which car sits in which slot
        metres = lead - behind[slot]
        side = c.pack_side * jax.random.uniform(k[3], (n,), minval=-1.0, maxval=1.0)
        v = jnp.maximum(jax.random.uniform(k[4]) * c.start_speed
                        + c.speed_spread * jax.random.uniform(k[5], (n,), minval=-1.0, maxval=1.0), 0.0)
        cars = jax.vmap(lambda kk, d, m, s, vv: F.place(kk, track, d, m, s, vv))(
            jax.random.split(k[6], n), f.cars.design, metres, side, v)
        cars, metres, lesson = self.pit_lesson(jax.random.fold_in(key, 2), cars, track, lead, metres,
                                               lesson_on & ~approach)
        lesson = lesson | (approach & (jax.random.uniform(kd, (n,)) < c.pit_car))  # called from where they race
        lap = jax.random.randint(k[7], (), 0, F.laps)  # the leader's lap; cars behind the line are a lap down
        if c.last_laps:  # (step 8k) the end of a race: the leader on this many laps to go
            lap = jnp.asarray(F.laps - c.last_laps)
        progress = lap * L + metres  # from the metres placed, not the lap position (it wraps at the line)
        race_done = jnp.clip(progress / (F.laps * L), 0.0, 1.0)

        car = cars.car
        wear = (c.worn * jax.random.uniform(k[8], (n, 1)) * jax.random.uniform(k[9], (n, 4), minval=0.8, maxval=1.0))
        temp = (spec.temp_best + jnp.asarray(rc.COMPOUND_TEMP)[f.setup.compound][:, None]
                + c.temp_spread * jax.random.normal(k[10], (n, 4)))
        tank = f.setup.fuel
        burned = race_done * jax.random.uniform(k[11], (n,), minval=1 - c.fuel_spread, maxval=1 + c.fuel_spread)
        car = car._replace(wear=wear, temp=temp, fuel_used=tank * jnp.clip(burned, 0.0, 0.97))

        hurt = jax.random.uniform(k[12], (n,)) < c.damaged
        u = jax.random.uniform(k[13], (n, 6))
        toe_max = jnp.deg2rad(F.dmg.toe_max_deg if F.dmg is not None else 5.0)
        damage = race.Damage(
            wing_front=jnp.where(hurt & (u[:, 0] < 0.5), u[:, 1], 0.0),
            wing_rear=jnp.where(hurt & (u[:, 0] > 0.75), u[:, 2], 0.0),
            floor=jnp.where(hurt & (u[:, 3] < 0.4), 0.6 * u[:, 4], 0.0),
            toe=jnp.where(hurt & (u[:, 3] > 0.8), toe_max * (2 * u[:, 5] - 1), 0.0),
            puncture=(hurt & (u[:, 2] < 0.1))[:, None] & jax.nn.one_hot(jax.random.randint(k[14], (n,), 0, 4), 4,
                                                                        dtype=bool))
        # the clock: the leader has averaged a random pace so far (each car's pace is its own progress over this)
        pace = jax.random.uniform(k[15], minval=c.pace[0], maxval=c.pace[1])
        t = jnp.round(jnp.maximum(progress.max(), 0.0) / pace / F.cfg.dt).astype(f.t.dtype)
        cars = cars._replace(car=car, progress=progress, t=jnp.full(n, t, cars.t.dtype))
        call = (jax.random.uniform(jax.random.fold_in(key, 1), (n,)) < c.called) | lesson
        return f._replace(cars=cars, damage=damage, box_call=call, t=t)

    def pit_lesson(self, key, cars: race.State, track, lead, metres, on):
        """If `on`: each car with chance `pit_car` moved to before its box (on the track's right
        before the pit entry, in the lane after it), slow enough to stop on its mark, at least 8 m from the next one.
        Returns the cars, their metres along the lap (behind the leader at `lead`) and which cars are having the
        lesson (they're called in)."""
        F, c, n = self.field, self.cfg, self.n
        cfg = F.cfg
        k = jax.random.split(key, 5)
        L = F.tracks.length[track]
        lesson = (jax.random.uniform(k[0], (n,)) < c.pit_car) & on
        # most start close (a reverse curriculum: ~20% within 5 m of the mark, some paid on the spot), the rest spread
        # back to `pit_before` (race-field-1: spread evenly, 0 stops in 300M steps)
        s = F.box_s - jax.random.uniform(k[2], (n,)) ** c.pit_close * c.pit_before  # metres from the pit's middle
        order = jnp.argsort(-jnp.where(lesson, s, -jnp.inf))  # lesson cars first, furthest along first

        def space(prev, x):  # each at least 8 m behind the one ahead of it
            si, on = x
            out = jnp.where(on, jnp.minimum(si, prev - 8.0), si)
            return jnp.where(on, out, prev), out
        _, spaced = jax.lax.scan(space, jnp.full_like(s[0], jnp.inf), (s[order], lesson[order]))  # _like: shard_map
        s = jnp.zeros(n).at[order].set(spaced)
        d = jnp.maximum(F.box_s - s, 0.0)
        in_lane = s > -cfg.pit_entry
        lat = jnp.where(in_lane, (cfg.pit_wall_lat + cfg.pit_outer) / 2, -5.0)  # -5: right of the pack (+-3)
        cap = jnp.where(in_lane, cfg.pit_limit, c.start_speed)
        v = jax.random.uniform(k[3], (n,)) * jnp.minimum(cap, jnp.sqrt(2 * c.pit_decel * d))
        at = F.game.pit_at[track] + s
        m = lead - (lead - at) % L  # behind the leader on the lap
        moved = jax.vmap(lambda kk, dd, mm, ll, vv: F.place(kk, track, dd, mm, ll, vv))(
            jax.random.split(k[4], n), cars.design, m, lat, v)
        cars = jax.tree.map(lambda a, b: jnp.where(lesson.reshape(lesson.shape + (1,) * (a.ndim - 1)), a, b),
                            moved, cars)
        return cars, jnp.where(lesson, m, metres), lesson

    def step(self, s: Lesson, actions, key) -> tuple[Lesson, jax.Array, jax.Array, jax.Array, jax.Array]:
        """One decision for every car (actions (n_cars, 2 or 3)), after the team's random calls. Returns the lesson,
        each car's reward, which cars were racing before it (their step counts), which cars' runs just ended, and
        whether the stretch is over."""
        F, c = self.field, self.cfg
        f = s.field
        racing = self.racing(f)
        kc, kp = jax.random.split(key)
        call = racing & ~f.box_call & (jax.random.uniform(kc, (self.n,)) < F.cfg.dt / c.call_rate_s)
        f = f._replace(box_call=f.box_call | call,
                       plan=jnp.where(call, jax.random.randint(kp, (self.n,), 0, len(rc.COMPOUNDS)), f.plan))
        phi = self.potential(s.field)  # before this decision's calls: a new call is charged, so it all adds up
        f, reward = F.step(f, actions)
        still = self.racing(f)
        over = f.done | ((f.t - s.t0) * F.cfg.dt >= c.stretch_s) | ~jnp.any(still)
        ended = racing & (~still | over)
        if c.coach:
            reward = reward + c.coach * (c.coach_gamma * jnp.where(ended, 0.0, self.potential(f)) - phi)
        return s._replace(field=f), reward, racing, ended, over

    def potential(self, f: race_field.Field) -> jax.Array:
        """(n_cars,) the pit coach's potential (see SchoolConfig.coach): 0 unless called in and racing."""
        F, c, cfg = self.field, self.cfg, self.field.cfg
        cars = f.cars
        L = F.tracks.length[cars.track]
        ds = (F.to_box(cars) + F.box_half) % L - F.box_half  # metres to its mark ahead (just past it, in the window: < 0)
        entry = F.box_s + cfg.pit_entry  # its mark is this far past the lane's entry
        turn_in = jnp.clip(1 - (ds - entry) / c.coach_turn_in, 0.0, 1.0)
        brake = jnp.clip(1 - jnp.abs(ds) / c.coach_brake, 0.0, 1.0)
        v = jnp.hypot(cars.car.vx, cars.car.vy)
        phi = -(c.coach_d * jnp.minimum(jnp.abs(ds), c.coach_reach)
                + c.coach_lat * turn_in * jnp.maximum(cars.lat - c.coach_lane, 0.0) + c.coach_v * brake * v)
        return jnp.where(f.box_call & self.racing(f), phi, 0.0)

    def observe(self, s: Lesson) -> jax.Array:
        return self.field.observe(s.field)

    def reset_if(self, s: Lesson, over, key) -> Lesson:
        """A new stretch where this one is over."""
        fresh = self.start(key)
        return jax.tree.map(lambda a, b: jnp.where(over, a, b), fresh, s)
