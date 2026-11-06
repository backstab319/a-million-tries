"""Race control (episode 16): marshalling sectors and flags. Pure functions over a `race_field.Field`, no drawing.

Each circuit is cut into marshalling sectors of about `sector_m` metres (real circuits have ~20-30 marshal posts and
light panels). A sector is green, single yellow or double yellow (F1 B1.8.4, FIA Appendix H):
- an incident is a car stopped or crawling (under `crawl_v`) anywhere but the pit lane, or an out car not yet
  recovered (`RaceField.gone`);
- on the track: double yellow in its sector and single yellow in the sector before (the warning);
- off the track (`RaceField.off_track`): single yellow in its sector;
- cleared when the car is recovered, gets going again, or (an out car off the track, which stays all race) when the
  post judges it safe, `post_clear_s` after it stopped (~60 s: our estimate).
The flags need no memory of their own: everything comes from the field (an out car's `out_s` is its time stopped).

The race director (`direct`, a script with its rules written down, the same in every race; F1 B5.12-B5.13) reads the
out cars stopped on the track (at rest, not yet recovered, not off it): one at the side (the tarmac's edge or beside
it) -> the Virtual Safety Car; one on the racing line (its centre within `line_m` of the centreline), or two or more
anywhere on the track -> the Safety Car. The VSC ends `vsc_end_s` after the track is clear
("VSC ENDING", then green); the Safety Car comes in at the end of the lap after the track is clear ("SAFETY CAR IN THIS
LAP", once the track is clear and the field is queued behind the Safety Car, or it has led for `queue_wait_s`). Under either, each car has a delta against a minimum time (`min_pace`): it must not finish a marshalling sector
with the delta below 0 (B5.12.2: a time penalty, counted in `Control.breaks`; applied to the result in step 7).

The Safety Car (B5.13) is a real car on the circuit (`race_field.Bodies`: solid, seen by the lasers and the senses),
driven by the scripted test driver a little under the minimum pace (`sc_share`), so a car queued behind it never
breaks its delta. Deployed, it waits at the end of the pit lane and goes when the leader is `sc_release` metres behind
(joining ahead of it); "in this lap", it turns into the pit lane at its next pit entry, and the race is green when the
leader next crosses the line. It has an F1 car's body and physics (the real one is a road sports car: ours is ~1 m
longer). It doesn't take damage (our choice: nothing in our races should be able to take it out).

The red flag (B5.14, episode 16 step 6): three or more cars stopped on the track in one marshalling sector, or stopped
cars side by side (within `block_m` along the track) leaving no gap of `block_gap` across the tarmac. Every racing car
is then driven by a script (a procedure, not a driving skill): at the pit limiter's speed into the pit lane, queueing in
the fast lane in the order they arrive, `slot_m` apart, behind the closed pit exit (a car already past its place in the
lane goes round once more). The Safety Car leaves the circuit (our shortcut: it reappears at the head of the queue).
Once the track is clear and the field is queued (at least `red_min_s` under the red flag; or the track clear for
`red_wait_s`), every queued car gets its damage repaired and new tyres (`Field.plan`'s compound) for free
(B5.14.4), and the race resumes behind the Safety Car from the pit exit (a rolling resumption, B5.15): the script
drives each car out of the lane, then the Safety Car rules.

Pit entry closed (B1.6.4): while an out car is in the pit lane, waiting for its crew, the team can't call a car in
(`Field.box_call` is cleared for cars not yet in the lane), and a car that enters the lane anyway is counted in
`Control.closed_in` (a penalty). Our "essential repairs only" is "none".

Penalties (B5.12-B5.13, the stewards; episode 16 step 7), applied to the classification, not to any reward (the
rulebook, not reward shaping): each car's `penalty_s` is added to its race time, and `classify` orders the field
with them.
- overtaking under the VSC or the Safety Car (anywhere), or in a yellow sector: 5 s, unless the place is given back
  within one marshalling sector (our choice). A pass is on the road (lapping a car counts), within `pass_m`; passing a
  car that's out, crawling or in the pit lane is allowed (a car with an obvious problem; in and out of the pits), and so
  is passing in the pit lane. Passing the Safety Car on the track counts the same.
- not slowing in a double yellow: a whole sector driven under a double yellow (it was double as the car entered it)
  in under `dbl_factor` x the reference car's time for it: 5 s.
- breaking the VSC / Safety Car minimum time: 5 s per sector (`Control.breaks`).
- entering the pit lane while its entry is closed: 10 s (real stewards: usually a 10 s stop-and-go; we have no stop-and-
  go, so time).
- fewer than two dry compounds used (`two_compounds`; tyres fitted under a red flag count, as in F1): disqualified.
- the black flag (step 9h, owner): `black_flag_n` infringements of the above (each 5 or 10 s penalty counts one):
  disqualified, finished or not. Real stewards escalate (time, drive-through, stop-and-go) and black-flag a driver
  who keeps ignoring the flags; ours is a fixed count (our choice, 5: 25 s of penalties, very rare in F1). Our
  simplification: a black-flagged car keeps running to the end (a real one must retire to the pits within a lap).
Not penalised because it can't happen: reversing in the pit lane (no reverse there), passing under the red flag
(scripted).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.games import race_car as rc
from ailearns.games import race_checks, race_field

GREEN, YELLOW, DOUBLE = 0, 1, 2
N_SENSES = 5  # my sector: yellow, double; the next sector: yellow, double, metres to it / sector_m
RACING, VSC, SC, SC_IN, RED = 0, 1, 2, 3, 4  # the race's status
N_STATUS_SENSES = 5  # VSC, Safety Car, Safety Car in this lap, red flag (one-hot; all 0: racing); my delta / 5 s
# the driver's race control senses (episode 16 plan section 4), after `RaceField.observe`'s, in this order
DRIVER_SENSES = ("vsc", "safety_car", "sc_in_this_lap", "red_flag",            # the race's status, one-hot
                 "my_sector_yellow", "my_sector_double",                       # the light panel I'm passing
                 "next_sector_yellow", "next_sector_double", "next_sector_m",  # the next one, and metres to it / 250
                 "my_delta",                                                   # time banked / 5 s (VSC, Safety Car)
                 "safety_car_ahead",                                           # the car directly ahead is the Safety Car
                 "box_side",                                                   # my box's offset across the lane / 5 m
                 # (step 15n) the nearest car stopped on the track ahead, as a driver sees it (under `crawl_v`, on the
                 # tarmac or its edge, not in the pit lane; nothing from race control's records): there, metres to it /
                 # sense_range, its offset from me across the track / half the width (left positive)
                 "stopped_ahead", "stopped_ahead_m", "stopped_ahead_side")
# the pit wall's race control senses (episode 16 plan section 4), after `es_strategist.senses`' 25, in this order
STRATEGIST_SENSES = ("vsc", "safety_car", "red_flag",  # the race's status, one-hot (safety_car: in this lap too)
                     "laps_since",      # laps (the leader's) since this VSC / Safety Car / red flag began / 5, to 1
                     "two_compounds",   # it has raced on two dry compounds (one more is a disqualification)
                     "queue_place",     # its place in the Safety Car's queue / 8 (0: first behind it, or no SC)
                     "to_pit_entry",    # metres to the pit entry ahead / 1 km, to 1 (can it still make this lap's?)
                     "pit_closed")      # the pit entry is closed


class Control(NamedTuple):
    status: jax.Array    # RACING, VSC, SC, SC_IN or RED
    clear_s: jax.Array   # seconds the track has been clear of out cars (0 while one is there)
    in_lap: jax.Array    # SC_IN: the leader's laps done when the Safety Car left the track (green at its next line)
    armed: jax.Array     # (n_cars,) its delta is running (from its first sector line after the deployment)
    delta: jax.Array     # (n_cars,) seconds banked against the minimum time (VSC or Safety Car; 0 while racing)
    prog: jax.Array      # (n_cars,) progress at the last update (metres)
    sec: jax.Array       # (n_cars,) the marshalling sector each car was in at the last update
    breaks: jax.Array    # (n_cars,) marshalling sectors finished under the minimum time (5 s each, B5.12.2b)
    vsc_n: jax.Array     # VSCs deployed this race
    sc_n: jax.Array      # Safety Cars deployed this race
    sc: object           # race.State (1, ...): the Safety Car
    sc_setup: object     # race.Setup (1, ...)
    sc_on: jax.Array     # (1,) on the circuit
    sc_wait: jax.Array   # waiting at the end of the pit lane for the leader
    sc_lane: jax.Array   # on the pit lane's line (leaving the lane after its wait, or going in), easing in and out
    sc_s: jax.Array      # seconds since it left the pit lane to lead
    red_n: jax.Array     # red flags this race
    red_s: jax.Array     # seconds under the red flag (0 otherwise)
    slot: jax.Array      # (n_cars,) its place in the red flag's queue in the fast lane (-1: none; kept to the pit exit)
    n_slot: jax.Array    # the next free place in the queue
    pit_closed: jax.Array  # the pit entry is closed (an out car in the lane)
    lane_prev: jax.Array   # (n_cars,) in the pit lane at the last update
    closed_in: jax.Array   # (n_cars,) times it entered the pit lane while it was closed
    owe: jax.Array       # (n_cars, n_cars) metres car i has left to give back the place it took from car j (0: none)
    owe_sc: jax.Array    # (n_cars,) the same, for passing the Safety Car
    sc_prev: jax.Array   # the Safety Car's lap position at the last update
    passes: jax.Array    # (n_cars,) places taken under yellow, the VSC or the Safety Car and not given back (5 s each)
    sec_t: jax.Array     # (n_cars,) seconds in its current sector
    dbl_in: jax.Array    # (n_cars,) its current sector was double yellow as it entered it
    dbl: jax.Array       # (n_cars,) double yellow sectors driven too fast (5 s each)
    used: jax.Array      # (n_cars, len(rc.COMPOUNDS)) the compounds it has raced on
    since: jax.Array     # the leader's progress when this VSC / Safety Car / red flag began (kept through "in this
                         # lap"; metres)
    lane_hits: jax.Array  # (n_cars,) (step 13d) collisions it caused in the pit lane (`Field.fault`; 5 s each, as
                          # "causing a collision"; not in the red flag's scripted queue)


def reference_speeds(tracks, grip_g=1.4, brake_g=1.5, accel_g=0.6, v_max=90.0) -> np.ndarray:
    """(T, P) metres per second at each centreline point for the reference car: a speed profile (a lap simulator's
    lap: each corner at `grip_g`, braking for it at `brake_g`, accelerating out at `accel_g`). With the test driver's
    limits (`race_checks.test_driver`) it gives the test driver's lap times (61-112 s on the 8 test circuits, within
    a second) without its crashes (it doesn't finish 2 of them)."""
    curv, L = np.asarray(tracks.curvature), np.asarray(tracks.length)
    T, P = curv.shape
    ds, g = L / P, 9.81
    v = np.minimum(np.sqrt(grip_g * g / np.maximum(np.abs(curv), 1e-4)), v_max)
    for _ in range(2):  # braking: backwards round the loop (twice, for the wrap)
        for i in range(P - 1, -1, -1):
            v[:, i] = np.minimum(v[:, i], np.sqrt(v[:, (i + 1) % P] ** 2 + 2 * brake_g * g * ds))
    for _ in range(2):  # accelerating: forwards
        for i in range(P):
            v[:, i] = np.minimum(v[:, i], np.sqrt(v[:, i - 1] ** 2 + 2 * accel_g * g * ds))
    return v


class RaceControl:
    def __init__(self, field: race_field.RaceField, sector_m: float = 250.0, crawl_v: float = 5.0,
                 post_clear_s: float = 60.0, start_grace_s: float = 5.0, race_pace: float = 1.3,
                 vsc_factor: float = 1.4, vsc_end_s: float = 10.0, sc_share: float = 0.8,
                 sc_release: tuple = (200.0, 400.0), queue_lengths: float = 10.0, queue_wait_s: float = 600.0,
                 block_m: float = 15.0, block_gap: float = 3.0, body_half: float = 1.5, slot_m: float = 8.0,
                 red_min_s: float = 30.0, red_wait_s: float = 300.0, red_grip: float = 0.6,
                 red_brake: float = 0.08,
                 pass_m: float = 50.0, dbl_factor: float = 1.4, two_compounds: bool = True, line_m: float = 3.0,
                 black_flag_n: int = 5):
        self.field = field
        self.black_flag_n = black_flag_n  # infringements to the black flag (0: never)
        # (step 9a) a car stopped with its centre this close to the centreline is on the racing line: the Safety Car
        # (our estimate: its body then covers the middle of the 12 m tarmac); at the edge or on the verge, a VSC
        self.line_m = line_m
        self.sector_m, self.crawl_v, self.post_clear_s = sector_m, crawl_v, post_clear_s
        # a standing start: every car is under `crawl_v` for its first second or so; that's not an incident
        self.start_grace_s = start_grace_s
        lengths = np.asarray(field.tracks.length)
        self.sectors = jnp.asarray(np.maximum(np.round(lengths / sector_m), 1).astype(np.int32))  # (T,) per circuit
        self.max_sectors = int(self.sectors.max())  # fixed shape: the circuits with fewer leave the rest green
        # the minimum time under the VSC and the Safety Car, seconds per metre at each centreline point (one fixed table
        # per circuit, like the real ECU's): `vsc_factor` x race pace (real VSC: ~30-40% off it), race pace being
        # `race_pace` x the reference car's (episode 15's race driver, race-fin-7:50: median laps 1.21-1.39 x it on
        # the 8 test circuits). Both factors are our estimates, checked in step 9.
        ref = reference_speeds(field.tracks)
        self.min_pace = jnp.asarray(vsc_factor * race_pace / ref, jnp.float32)
        # the reference car's time for each marshalling sector (the double yellow's yardstick)
        T, P = ref.shape
        ref_s = np.zeros((T, self.max_sectors))
        for t in range(T):
            n = int(self.sectors[t])
            at = np.minimum((np.arange(P) * n / P).astype(int), n - 1)
            np.add.at(ref_s[t], at, lengths[t] / P / ref[t])
        self.ref_sector_s = jnp.asarray(ref_s, jnp.float32)
        self.pass_m, self.dbl_factor, self.two_compounds = pass_m, dbl_factor, two_compounds
        self.vsc_end_s = vsc_end_s
        # "in this lap" once the field is queued, or after `queue_wait_s` leading at most (step 9d: 120 s before; cars
        # 0.5-2.9 km back when it left the lane can't close up in that at the delta's speed, ~20% faster than it, and
        # real Safety Car spells run ~4 laps; the cap is for a car that never catches up)
        self.sc_share, self.sc_release, self.queue_wait_s = sc_share, sc_release, queue_wait_s
        self.queue_m = queue_lengths * 5.6  # B5.13: at most 10 car lengths apart (our cars: ~5.6 m)
        self.sc_design = jnp.asarray(rc.F1_LIKE[0])
        # the red flag: a stopped car covers `body_half` m either side of its centre (its width is 2 m, its length
        # 5.6 m: a crashed car is often turned); a gap of `block_gap` (a car's width and a metre) gets a car through
        self.block_m, self.block_gap, self.body_half = block_m, block_gap, body_half
        # the queue's head: 12 m behind the Safety Car's spot at the end of the fast lane (`sc_waiting`); real red flags
        # last 20 minutes or more, ours `red_min_s` at least (the work is the same, the waiting isn't footage), and
        # `red_wait_s` after the track is clear at most (a lap at the limiter's 80 km/h: ~225 s on a 5 km circuit)
        self.slot_m, self.head_s = slot_m, field.cfg.pit_exit - 110.0 - 12.0
        self.red_min_s, self.red_wait_s, self.red_grip = red_min_s, red_wait_s, red_grip
        self.red_brake = red_brake  # the script's firmest pedal slowing to the limiter: ~0.5 g (linear on 6.5 g; full: -1)

    def sector_len(self, track) -> jax.Array:
        return self.field.tracks.length[track] / self.sectors[track]

    def sector(self, cars) -> jax.Array:
        """(n_cars,) the marshalling sector each car is in (0 starts at the line)."""
        return jnp.minimum(jnp.floor(cars.lap_pos / self.sector_len(cars.track)).astype(jnp.int32),
                           self.sectors[cars.track] - 1)

    def incidents(self, f: race_field.Field) -> jax.Array:
        """(n_cars,) the flag each car's incident shows in its own sector: GREEN (none), YELLOW (off the track) or
        DOUBLE (on it)."""
        F, cars = self.field, f.cars
        out = (cars.crashed | f.wrecked | F.dry_out(f)) & ~f.finished & ~F.gone(f)
        speed = jnp.hypot(cars.car.vx, cars.car.vy)
        started = f.t * F.cfg.dt >= self.start_grace_s
        crawling = ~(cars.crashed | f.wrecked | f.finished) & (speed < self.crawl_v) & started
        lane = jax.vmap(F.game.in_pit_lane)(cars)
        off = F.off_track(cars)
        cleared = off & out & (f.out_s >= self.post_clear_s)  # off the track and out: the post calls it safe
        there = (out | crawling) & ~lane & ~cleared
        return jnp.where(there, jnp.where(off, YELLOW, DOUBLE), GREEN)

    def flags(self, f: race_field.Field) -> jax.Array:
        """(max_sectors,) each sector's flag for this race (all its cars are on one circuit)."""
        n = self.sectors[f.cars.track[0]]
        level, s = self.incidents(f), self.sector(f.cars)
        here = jax.nn.one_hot(s, self.max_sectors, dtype=jnp.int32) * level[:, None]
        warn = jax.nn.one_hot((s - 1) % n, self.max_sectors, dtype=jnp.int32) * (level == DOUBLE)[:, None] * YELLOW
        return jnp.maximum(here, warn).max(0)

    def senses(self, f: race_field.Field) -> jax.Array:
        """(n_cars, N_SENSES) what each driver sees of the flags: its own sector yellow, double yellow; the next
        sector's yellow, double yellow (its light panel is in sight ahead), and the metres to it / sector_m."""
        flags = self.flags(f)
        cars = f.cars
        s = self.sector(cars)
        nxt = (s + 1) % self.sectors[cars.track]
        here, ahead = flags[s], flags[nxt]
        to_next = (s + 1) * self.sector_len(cars.track) - cars.lap_pos
        return jnp.stack([here == YELLOW, here == DOUBLE, ahead == YELLOW, ahead == DOUBLE,
                          jnp.clip(to_next / self.sector_m, 0.0, 2.0)], -1).astype(jnp.float32)

    def init(self, f: race_field.Field) -> Control:
        z = jnp.zeros(self.field.n)
        sc = jax.tree.map(lambda a: a[None], self.sc_waiting(f.cars.track[0]))
        return Control(jnp.asarray(RACING), jnp.asarray(0.0), jnp.asarray(0.0), z > 0, z, f.cars.progress,
                       self.sector(f.cars), jnp.zeros(self.field.n, jnp.int32), jnp.asarray(0), jnp.asarray(0),
                       sc, jax.tree.map(lambda a: a[:1], f.setup), jnp.zeros(1, bool), jnp.asarray(False),
                       jnp.asarray(False), jnp.asarray(0.0), jnp.asarray(0), jnp.asarray(0.0),
                       jnp.full(self.field.n, -1, jnp.int32), jnp.asarray(0), jnp.asarray(False),
                       jax.vmap(self.field.game.in_pit_lane)(f.cars), jnp.zeros(self.field.n, jnp.int32),
                       jnp.zeros((self.field.n, self.field.n)), jnp.zeros(self.field.n), sc.lap_pos[0],
                       jnp.zeros(self.field.n, jnp.int32), jnp.zeros(self.field.n), jnp.zeros(self.field.n, bool),
                       jnp.zeros(self.field.n, jnp.int32),
                       jax.nn.one_hot(f.setup.compound, len(rc.COMPOUNDS)) > 0, jnp.asarray(0.0),
                       jnp.zeros(self.field.n, jnp.int32))

    def sc_waiting(self, track):
        """The Safety Car at the end of the pit lane (in the fast lane, before it bends back to the track)."""
        F = self.field
        at = F.game.pit_at[track] + F.cfg.pit_exit - 110.0
        return F.place(jax.random.PRNGKey(0), track, self.sc_design, at, F.cfg.drive_lat)

    def bodies(self, c: Control) -> race_field.Bodies:
        return race_field.Bodies(c.sc, c.sc_setup, c.sc_on)

    def leader(self, f: race_field.Field) -> jax.Array:
        """The leading car still racing (its index)."""
        F = self.field
        running = ~(f.cars.crashed | f.wrecked | f.finished | F.dry_out(f))
        # (ep16 step 9g) the furthest along, the lowest index among ties: `F.order`'s first running car, without its two
        # sorts every step
        return jnp.argmax(jnp.where(running, f.cars.progress, -jnp.inf))

    def queued(self, f: race_field.Field, c: Control) -> jax.Array:
        """Every car racing (out of the pit lane) is in one queue behind the Safety Car, each within `queue_m` of the
        car ahead (lapped cars stay where they are: our director doesn't wave them past, B5.13.4c); after a red flag,
        not before the whole field is out of the pit lane."""
        F, cars = self.field, f.cars
        L = F.tracks.length[cars.track]
        running = ~(cars.crashed | f.wrecked | f.finished | F.dry_out(f)) & ~jax.vmap(F.game.in_pit_lane)(cars)
        behind = jnp.where(running, (c.sc.lap_pos[0] - cars.lap_pos) % L, jnp.inf)
        # each car's gap to the one just ahead of it in the queue (the Safety Car for the first): as sorting `behind`
        # and differencing (ep16 step 9g: without the sort; ties: the lower index counts as ahead, as the sort had it)
        i = jnp.arange(F.n)
        before = (behind[None, :] < behind[:, None]) | ((behind[None, :] == behind[:, None]) & (i[None, :] < i[:, None]))
        ahead = jnp.max(jnp.where(before & jnp.isfinite(behind)[None, :], behind[None, :], 0.0), 1)
        gaps = behind - ahead
        resuming = jnp.any(self.running(f) & (c.slot >= 0))  # after a red flag: cars still to come out of the pit lane
        return c.sc_on[0] & ~resuming & jnp.all(jnp.where(jnp.isfinite(behind), gaps <= self.queue_m, True))

    def sc_action(self, f: race_field.Field, c: Control) -> jax.Array:
        """The Safety Car's inputs: the test driver, at `sc_share` of the minimum pace's speed; held at the end of the
        pit lane while it waits; "in this lap" it takes the pit lane at its next entry (the scripted pit stop's line)."""
        F, g, cfg = self.field, self.field.game, self.field.cfg
        st = jax.tree.map(lambda a: a[0], c.sc)
        s = g.from_pit(st.track, st.idx)
        in_lane = c.sc_lane & g.in_pit_zone(st.track, st.idx)
        ease = jnp.clip(jnp.minimum((s + cfg.pit_entry) / 60.0, (cfg.pit_exit - s) / 100.0), 0.0, 1.0)
        act = race_checks.test_driver(g, st, offset=jnp.where(in_lane, cfg.drive_lat * ease, 0.0))
        v = jnp.hypot(st.car.vx, st.car.vy)
        target = self.sc_share / self.min_pace[st.track, st.idx]
        pedal = jnp.minimum(act[1], jnp.clip(0.5 * (target - v), -0.5, 1.0))
        return jnp.stack([act[0], jnp.where(c.sc_wait, -1.0, pedal)])

    def drive_sc(self, f: race_field.Field, c: Control) -> Control:
        """The Safety Car moves one decision (before the cars collide with it in `RaceField.step_with`)."""
        F, g = self.field, self.field.game
        L = F.tracks.length[f.cars.track[0]]
        gap = (c.sc.lap_pos[0] - f.cars.lap_pos[self.leader(f)]) % L  # the leader behind it
        wait = c.sc_wait & ~((gap > self.sc_release[0]) & (gap < self.sc_release[1]))
        st = jax.tree.map(lambda a: a[0], c.sc)
        s = g.from_pit(st.track, st.idx)
        going_in = (c.status == SC_IN) & (s > -F.cfg.pit_entry) & (s < 0)
        lane = (c.sc_lane | going_in) & g.in_pit_zone(st.track, st.idx)
        c = c._replace(sc_wait=wait, sc_lane=lane, sc_s=jnp.where(c.sc_on[0] & ~wait, c.sc_s + F.cfg.dt, 0.0))
        dmg = jax.tree.map(lambda a: jnp.zeros_like(a[0]), f.damage)
        setup = jax.tree.map(lambda a: a[0], c.sc_setup)
        new, _ = g.step(st, self.sc_action(f, c), dmg, rc.CLEAN_AIR, setup)
        new = new._replace(car=new.car._replace(fuel_used=jnp.zeros_like(new.car.fuel_used)),  # never runs dry
                           crashed=jnp.asarray(False), done=jnp.asarray(False))
        moved = jax.tree.map(lambda a, b: jnp.where(c.sc_on[0], a, b), new, st)
        return c._replace(sc=jax.tree.map(lambda a: a[None], moved))

    def step(self, f: race_field.Field, c: Control, actions) -> tuple[race_field.Field, Control, jax.Array]:
        """One decision of a race with race control: the Safety Car moves, the cars move and collide (with it too),
        then the director. Returns the field, race control and each car's reward."""
        F = self.field
        lane = jax.vmap(F.game.in_pit_lane)(f.cars)
        f = f._replace(box_call=f.box_call & ~(c.pit_closed & ~lane))  # no calling a car in to a closed pit lane
        c = self.drive_sc(f, c)
        actions = jnp.asarray(actions, jnp.float32)
        red = self.red_action(f, c)
        red = jnp.concatenate([red, -jnp.ones((F.n, actions.shape[1] - 2))], 1)  # (no reverse)
        actions = jnp.where(self.scripted(f, c)[:, None], red, actions)
        f, reward, b = F.step_with(f, actions, self.bodies(c))
        new = self.direct(f, c._replace(sc=b.cars))
        f = self.repair(f, (c.status == RED) & (new.status != RED) & (c.slot >= 0))
        return f, new, reward

    def running(self, f: race_field.Field) -> jax.Array:
        return ~(f.cars.crashed | f.wrecked | f.finished | self.field.dry_out(f))

    def scripted(self, f: race_field.Field, c: Control) -> jax.Array:
        """(n_cars,) driven by the red flag's script: under the red flag, and resuming until it's out of the lane."""
        return self.running(f) & ((c.status == RED) | (c.slot >= 0))

    def red_action(self, f: race_field.Field, c: Control) -> jax.Array:
        """(n_cars, 2) the red flag's script: the test driver at the pit limiter's speed, into the pit lane at its next
        entry (down the fast lane), easing to a stop on its place in the queue behind the car ahead; resuming (the
        race no longer red), down the fast lane and out, following the car ahead."""
        F, g, cfg = self.field, self.field.game, self.field.cfg
        cars = f.cars
        s = jax.vmap(g.from_pit)(cars.track, cars.idx)
        lane = jax.vmap(g.in_pit_lane)(cars)
        take = lane | ((s > -cfg.pit_entry) & (s < -cfg.pit_wall_len - 20.0))  # not past the start of the pit wall
        ease = jnp.clip(jnp.minimum((s + cfg.pit_entry) / 60.0, (cfg.pit_exit - s) / 100.0), 0.0, 1.0)
        # cautious in the corners (`red_grip` g): tyres cool at 80 km/h, and a script cornering at the test driver's
        # 1.4 g understeered off at a hairpin after ~200 s of it (step 9c: at 0.8 g a car on cold, worn tyres, 45 C
        # and 57%, still slid off a 44 m hairpin)
        act = jax.vmap(lambda st, o: race_checks.test_driver(g, st, grip_g=self.red_grip, brake_g=self.red_grip,
                                                             offset=o))(cars, jnp.where(take, cfg.drive_lat * ease, 0.0))
        v = jnp.hypot(cars.car.vx, cars.car.vy)
        to = jnp.where((c.status == RED) & (c.slot >= 0), self.head_s - c.slot * self.slot_m - F.pit_s(cars), 1e4)
        target = jnp.minimum(cfg.pit_limit, jnp.sqrt(2 * 3.0 * jnp.maximum(to - 0.5, 0.0)))
        # behind any car in its path (a stopped one too: it waits behind it), every car on the circuit (the senses' two
        # nearest aren't enough: three stopped side by side, the two beside the line are the nearest)
        allc, there = F.on_circuit(f, self.bodies(c))
        L = F.tracks.length[cars.track[0]]
        gap = (allc.lap_pos[None, :] - cars.lap_pos[:, None]) % L  # (i, j) metres car j is ahead of car i
        in_line = (there[None, :] & ~jnp.eye(F.n, there.shape[0], dtype=bool) & (gap < 300.0)
                   & (jnp.abs(allc.lat[None, :] - cars.lat[:, None]) < 2.4))
        stand = self.slot_m - 1.0  # at a standstill, centre to centre (the queue's places are `slot_m` apart)
        # slowing at 3 m/s^2 behind it, from far enough back for the gentle pedal, and a second behind at speed (step
        # 9e: a car seen 40 m ahead at 80 km/h needed ~0.7 g; one 5 m behind another at 20 m/s hit it as it braked
        # for a corner)
        room = jnp.maximum(gap - stand - v[:, None] * 1.0, 0.0)
        follow = jnp.hypot(allc.car.vx, allc.car.vy)[None, :] + jnp.sqrt(2 * 3.0 * room)
        target = jnp.minimum(target, jnp.where(in_line, follow, 999.0).min(1))
        # (step 9c) easing off to the limiter, as a driver does at a red flag (braking for the corners as before): at
        # half pedal from 150 km/h, on tyres cooled behind the Safety Car, two cars slid off in the first 4 s; a quarter
        # pedal was still ~1.6 g (step 9e: one spun into the pit wall), now ~0.5 g
        pedal = jnp.minimum(act[:, 1], jnp.clip(0.5 * (target - v), -self.red_brake, 1.0))
        return jnp.stack([act[:, 0], jnp.where((target < 0.3) & (v < 1.0), -1.0, pedal)], 1)

    def repair(self, f: race_field.Field, mask) -> race_field.Field:
        """The red flag's free work (B5.14.4) on the `mask` cars: accident damage repaired, new tyres (`Field.plan`'s
        compound, off the blankets). Not a pit stop (no stop counted)."""
        F, m = self.field, mask
        shape = lambda a: m.reshape(m.shape + (1,) * (a.ndim - 1))  # noqa: E731
        damage = jax.tree.map(lambda a: jnp.where(shape(a), jnp.zeros_like(a), a), f.damage)
        car = f.cars.car._replace(wear=jnp.where(m[:, None], 0.0, f.cars.car.wear),
                                  temp=jnp.where(m[:, None], F.cfg.tyre_start, f.cars.car.temp))
        return f._replace(cars=f.cars._replace(car=car), damage=damage, flat_m=jnp.where(m[:, None], 0.0, f.flat_m),
                          setup=f.setup._replace(compound=jnp.where(m, f.plan, f.setup.compound)),
                          box_call=f.box_call & ~m)

    @property
    def obs_size(self) -> int:
        return self.field.obs_size + len(DRIVER_SENSES)

    def observe(self, f: race_field.Field, c: Control) -> jax.Array:
        """(n_cars, obs_size): `RaceField.observe` (the Safety Car seen as a car), then `DRIVER_SENSES`."""
        return jnp.concatenate([self.field.observe(f, self.bodies(c)), self.driver_senses(f, c)], -1)

    def driver_senses(self, f: race_field.Field, c: Control) -> jax.Array:
        """(n_cars, len(DRIVER_SENSES)), in `DRIVER_SENSES`' order."""
        F, cars, cfg = self.field, f.cars, self.field.cfg
        status = self.status_senses(c)
        allc, there = F.on_circuit(f, self.bodies(c))
        L = F.tracks.length[cars.track[0]]
        gap = (allc.lap_pos[None, :] - cars.lap_pos[:, None]) % L  # (i, j) metres j is ahead of i
        seen = there[None, :] & ~jnp.eye(F.n, there.shape[0], dtype=bool) & (gap < F.sense_range)
        nearest = jnp.argmin(jnp.where(seen, gap, jnp.inf), 1)
        sc_ahead = seen.any(1) & (nearest >= F.n)  # the bodies come after the racing cars
        lane = jax.vmap(F.game.in_pit_lane)(cars)
        box = jnp.where(lane, jnp.clip((cfg.box_lat - cars.lat) / 5.0, -1.0, 1.0), 0.0)
        # (step 15n) the nearest stopped car ahead: 15m's teacher, judging "stopped" by speed alone, got past wrecks
        # as well as with race control's records, but the brains never learned to read it from relative speeds
        n = F.n
        v = jnp.hypot(cars.car.vx, cars.car.vy)
        still = (there[:n] & (v < self.crawl_v) & ~lane & ~f.finished & (jnp.abs(cars.lat) < cfg.width / 2 + 2.0))
        sg = jnp.where(still[None, :] & ~jnp.eye(n, dtype=bool) & (gap[:, :n] < F.sense_range), gap[:, :n], jnp.inf)
        j = jnp.argmin(sg, 1)
        found = jnp.isfinite(sg[jnp.arange(n), j])
        stopped = jnp.stack([found.astype(jnp.float32), jnp.where(found, sg[jnp.arange(n), j] / F.sense_range, 0.0),
                             jnp.where(found, (cars.lat[j] - cars.lat) / (cfg.width / 2), 0.0)], -1)
        return jnp.concatenate([status[:, :4], self.senses(f), status[:, 4:],
                                sc_ahead[:, None].astype(jnp.float32), box[:, None], stopped], -1)

    def stopped_on_track(self, f: race_field.Field) -> tuple[jax.Array, jax.Array]:
        """(n_cars,) out cars stopped on the track waiting for the marshals, and (n_cars,) those of them on the racing
        line (the car's centre within `line_m` of the centreline: the middle of the tarmac). Stopped: parked, or under
        `crawl_v` (the director calls the VSC or the Safety Car on where a car ends up; the flags go out at once).
        (Step 9a: the whole tarmac counted as the racing line and a sliding wreck counted where it was, so every car
        stopped on the track brought out the Safety Car and no race had a VSC.)"""
        F, cars = self.field, f.cars
        at_rest = cars.crashed | (jnp.hypot(cars.car.vx, cars.car.vy) < self.crawl_v)  # (a crashed car's speed is
        out = (cars.crashed | f.wrecked | F.dry_out(f)) & at_rest & ~f.finished & ~F.gone(f)  # frozen, not zeroed)
        there = out & ~F.off_track(cars) & ~jax.vmap(F.game.in_pit_lane)(cars)
        return there, there & (jnp.abs(cars.lat) < self.line_m)

    def red_needed(self, f: race_field.Field) -> jax.Array:
        """Three or more cars stopped on the track in one marshalling sector, or stopped cars within `block_m` of each
        other along it leaving no gap of `block_gap` across the tarmac (the track blocked)."""
        F, cars = self.field, f.cars
        there, _ = self.stopped_on_track(f)
        sec = self.sector(cars)
        three = jnp.any(there & (((sec[:, None] == sec[None, :]) & there[None, :]).sum(1) >= 3))
        L = F.tracks.length[cars.track[0]]
        apart = (cars.lap_pos[:, None] - cars.lap_pos[None, :] + L / 2) % L - L / 2
        w = F.cfg.width / 2

        def widest(near):  # the widest gap across the tarmac between the cars `near` (a mask) and the edges
            # (ep16 step 9g: without sorting; a sweep over the cars sorted by their left side was 1.7 of 3.4 s of a
            # training step on a T4.) A free gap starts at the tarmac's edge or at a car's side no other car covers,
            # and runs to the next car's side beyond it (or the other edge). Where there's no gap the value differs
            # from the sweep's (it's below zero either way), and only `< block_gap` is used
            lo = jnp.where(near, cars.lat - self.body_half, jnp.inf)
            hi = jnp.where(near, cars.lat + self.body_half, -jnp.inf)
            x = jnp.concatenate([jnp.full(1, -w), jnp.maximum(hi, -w)])  # where a gap could start
            free = ~jnp.any((lo[None, :] < x[:, None]) & (x[:, None] < hi[None, :]), 1) & (x < w)
            to = jnp.minimum(w, jnp.min(jnp.where(lo[None, :] >= x[:, None], lo[None, :], jnp.inf), 1))
            return jnp.max(jnp.where(free, to - x, -jnp.inf))

        blocked = jax.vmap(widest)(there[None, :] & (jnp.abs(apart) < self.block_m)) < self.block_gap
        return three | jnp.any(there & blocked)

    def direct(self, f: race_field.Field, c: Control) -> Control:
        """The race director, once a decision (after `RaceField.step`): the race's status and every car's delta."""
        F, cars, dt = self.field, f.cars, self.field.cfg.dt
        there, tarmac = self.stopped_on_track(f)
        need_sc = (there.sum() >= 2) | tarmac.any()
        need_red = self.red_needed(f)
        clear = ~there.any()
        clear_s = jnp.where(clear, c.clear_s + dt, 0.0)
        running = self.running(f)
        lane = jax.vmap(F.game.in_pit_lane)(cars)
        v = jnp.hypot(cars.car.vx, cars.car.vy)
        queued_red = jnp.all(~running | ((c.slot >= 0) & lane & (v < 0.5)))
        resume = (c.red_s >= self.red_min_s) & clear & (queued_red | (clear_s >= self.red_wait_s))
        leader_laps = jnp.max(jnp.where(running | f.finished, F.laps_done(f), -1.0))  # finished: the flag counts
        st = c.status
        st_new = jnp.select(
            [need_red & (st != RED),                       # the red flag
             (st == RED) & resume,                         # resuming behind the Safety Car
             need_sc & (st != SC) & (st != RED),           # (back) to the Safety Car
             (st == RACING) & there.any(),                 # a VSC
             (st == VSC) & (clear_s >= self.vsc_end_s),    # VSC ending -> green
             (st == SC) & clear & ~c.sc_wait & (self.queued(f, c) | (c.sc_s >= self.queue_wait_s)),  # in this lap
             (st == SC_IN) & ~c.sc_on[0] & (leader_laps > c.in_lap)],  # it's in, the leader at the line: green
            [RED, SC, SC, VSC, RACING, SC_IN, RACING], st)
        # the Safety Car: out at the end of the pit lane when it's deployed (or back), off the circuit once it's in
        # the pit lane past its entry "in this lap" (and the leader's laps then: green at its next line)
        g, sc = F.game, jax.tree.map(lambda a: a[0], c.sc)
        spawn = (st_new == SC) & ~c.sc_on[0]
        gone_in = (st_new == SC_IN) & c.sc_on[0] & g.in_pit_lane(sc) & (g.from_pit(sc.track, sc.idx) < 0)
        in_lap = jnp.where(gone_in, leader_laps, c.in_lap)
        off = gone_in | (st_new == RACING) | (st_new == VSC) | (st_new == RED)
        sc_on = jnp.where(spawn, True, jnp.where(off, False, c.sc_on[0]))
        sc_state = jax.tree.map(lambda a, b: jnp.where(spawn, a[None], b),
                                self.sc_waiting(cars.track[0]), c.sc)
        # the delta: + the time taken, - the minimum time for the metres driven (outside the pit lane, still racing).
        # It starts at the car's next sector line after the deployment (our choice: a car caught mid-sector at racing
        # speed has no time to slow for the rest of it), and that sector is the first one judged
        neutral = (st_new == VSC) | (st_new == SC) | (st_new == SC_IN)
        was = (st == VSC) | (st == SC) | (st == SC_IN)
        sec = self.sector(cars)
        crossed = sec != c.sec
        armed = neutral & (c.armed | (was & crossed))
        counted = armed & running & ~jax.vmap(F.game.in_pit_lane)(cars)
        moved = cars.progress - c.prog
        took = jnp.where(counted, dt - moved * self.min_pace[cars.track, cars.idx], 0.0)
        delta = jnp.where(armed & c.armed, c.delta + took, 0.0)  # 0 at its first line, and while racing
        broke = c.armed & armed & running & crossed & (delta < 0)  # finished a sector under the minimum
        new_sc = (st_new == SC) & (st != SC) & (st != SC_IN) & (st != RED)
        # the red flag's queue: a car gets the next place as it enters the pit lane (in the order they arrive; cars
        # entering together, the one further along first), if it hasn't passed it already (then it goes round again)
        red = st_new == RED
        ps = F.pit_s(cars)
        cand = red & running & lane & (c.slot < 0)
        want = c.n_slot + (cand[None, :] & (ps[None, :] > ps[:, None])).sum(1)
        ok = cand & (ps < self.head_s - want * self.slot_m - 2.0)
        slot = jnp.where(ok, want, c.slot)
        zone = jax.vmap(F.game.in_pit_zone)(cars.track, cars.idx)  # (not the lane: its centre crosses the wall's
        slot = jnp.where(~red & ~zone, -1, slot)  # line before the exit) resumed: out of the zone, its own driving
        n_slot = jnp.where(red, jnp.maximum(c.n_slot, jnp.max(jnp.where(ok, want + 1, 0))), 0)
        # pit entry closed: an out car in the lane, waiting for its crew (not under the red flag: the lane is the queue)
        out_lane = ~running & ~f.finished & lane & ~F.gone(f)
        entered = lane & ~c.lane_prev & running & (slot < 0)
        owe, owe_sc, passes = self.overtakes(f, c, st_new, lane, running, moved, sc_state, sc_on)
        flags = self.flags(f)
        sec_t = jnp.where(crossed, 0.0, c.sec_t + dt)
        slow = c.sec_t + dt >= self.dbl_factor * self.ref_sector_s[cars.track, c.sec]
        dbl = c.dbl + (crossed & c.dbl_in & running & ~slow)
        dbl_in = jnp.where(crossed, (flags[sec] == DOUBLE) & ~lane, c.dbl_in)
        used = c.used | (jax.nn.one_hot(f.setup.compound, len(rc.COMPOUNDS)) > 0)
        began = (st_new != st) & ((st_new == VSC) | (st_new == SC) | (st_new == RED))
        since = jnp.where(began, cars.progress[self.leader(f)], c.since)
        return Control(st_new, clear_s, in_lap, armed, delta, cars.progress, sec, c.breaks + broke,
                       c.vsc_n + ((st_new == VSC) & (st != VSC)), c.sc_n + new_sc, sc_state, c.sc_setup,
                       sc_on[None], jnp.where(spawn, st != RED, c.sc_wait), jnp.where(spawn, True, c.sc_lane & sc_on),
                       c.sc_s, c.red_n + (red & (st != RED)), jnp.where(red & (st == RED), c.red_s + dt, 0.0), slot,
                       n_slot, out_lane.any() & ~red, lane, c.closed_in + (entered & c.pit_closed), owe, owe_sc,
                       sc_state.lap_pos[0], passes, sec_t, dbl_in, dbl, used, since,
                       c.lane_hits + (f.fault & lane & running & (slot < 0) & ~red))

    def overtakes(self, f, c, st_new, lane, running, moved, sc_state, sc_on):
        """Places taken under yellow, the VSC or the Safety Car (a pass on the road, within `pass_m`): owed back for a
        marshalling sector's metres; given back (the passed car ahead on the road again) it's cleared, not given back
        it's a penalty. Passing a car that's out, crawling or in the pit lane, or passing in the pit lane, is
        allowed."""
        F, cars = self.field, f.cars
        L = F.tracks.length[cars.track[0]]
        road = lambda a, b: (a[:, None] - b[None, :] + L / 2) % L - L / 2  # noqa: E731  (i, j): metres i is ahead of j
        now, before = road(cars.lap_pos, cars.lap_pos), road(c.prog % L, c.prog % L)
        speed = jnp.hypot(cars.car.vx, cars.car.vy)
        neutral = (st_new == VSC) | (st_new == SC) | (st_new == SC_IN)
        flagged = (neutral | (self.flags(f)[self.sector(cars)] != GREEN)) & (st_new != RED)  # where the passer is
        able = running & ~lane
        fair = able & (speed >= self.crawl_v)  # a car with an obvious problem may be passed
        passed = ((before < 0) & (before > -self.pass_m) & (now >= 0) & (now < self.pass_m)
                  & (able & flagged)[:, None] & fair[None, :])
        back = (now < 0) & (now > -self.pass_m)  # behind it again: the place given back
        owing = c.owe > 0
        passed = passed & ~owing.T  # taking back a place it was owed isn't an overtake
        left = jnp.where(owing, c.owe - jnp.maximum(moved, 0.0)[:, None], 0.0)
        cleared = owing & (back | ~running[None, :] | ~running[:, None])
        expired = owing & ~cleared & (left <= 0)
        owe = jnp.where(passed, self.sector_m, jnp.where(owing & ~cleared & ~expired, left, 0.0))
        # the Safety Car: passing it on the track (not while it's in the pit lane: waiting, or going in)
        sc = jax.tree.map(lambda a: a[0], sc_state)
        sc_track = sc_on & c.sc_on[0] & ~F.game.in_pit_lane(sc)
        now_sc = (cars.lap_pos - sc.lap_pos + L / 2) % L - L / 2
        before_sc = (c.prog % L - c.sc_prev + L / 2) % L - L / 2
        passed_sc = (sc_track & able & (before_sc < 0) & (before_sc > -self.pass_m) & (now_sc >= 0)
                     & (now_sc < self.pass_m))
        owing_sc = c.owe_sc > 0
        left_sc = jnp.where(owing_sc, c.owe_sc - jnp.maximum(moved, 0.0), 0.0)
        cleared_sc = owing_sc & (((now_sc < 0) & (now_sc > -self.pass_m)) | ~running | ~sc_on)
        expired_sc = owing_sc & ~cleared_sc & (left_sc <= 0)
        owe_sc = jnp.where(passed_sc, self.sector_m, jnp.where(owing_sc & ~cleared_sc & ~expired_sc, left_sc, 0.0))
        return owe, owe_sc, c.passes + expired.sum(1) + expired_sc

    def penalty_s(self, c: Control) -> jax.Array:
        """(n_cars,) seconds added to each car's race time by the stewards."""
        return 5.0 * (c.breaks + c.passes + c.dbl + c.lane_hits) + 10.0 * c.closed_in

    def infringements(self, c: Control) -> jax.Array:
        """(n_cars,) penalties so far, counted one each."""
        return c.breaks + c.passes + c.dbl + c.closed_in + c.lane_hits

    def black_flagged(self, c: Control) -> jax.Array:
        """(n_cars,) shown the black flag: `black_flag_n` infringements."""
        if self.black_flag_n <= 0:
            return jnp.zeros(c.breaks.shape, bool)
        return self.infringements(c) >= self.black_flag_n

    def disqualified(self, f: race_field.Field, c: Control) -> jax.Array:
        """(n_cars,) finished on fewer than two dry compounds (F1: every driver must use two in a dry race), or
        black-flagged."""
        one = f.finished & (c.used.sum(-1) < 2) if self.two_compounds else jnp.zeros(self.field.n, bool)
        return one | self.black_flagged(c)

    def classify(self, f: race_field.Field, c: Control) -> jax.Array:
        """Each car's position after the stewards (0 = winner): finishers by race time + penalties; then cars still
        running by distance, less their penalties at their own average speed; the out; the disqualified last."""
        F = self.field
        pen = self.penalty_s(c)
        pace = jnp.maximum(f.cars.progress, 0.0) / jnp.maximum(f.t * F.cfg.dt, 1.0)
        group = jnp.where(self.disqualified(f, c), 3, jnp.where(f.finished, 0, jnp.where(self.running(f), 1, 2)))
        within = jnp.where(f.finished, f.finish_t + pen, -(f.cars.progress - pen * pace))
        return jnp.argsort(jnp.lexsort((within, group)))

    def status_senses(self, c: Control) -> jax.Array:
        """(n_cars, N_STATUS_SENSES): the race's status (VSC, Safety Car, Safety Car in this lap, red flag; one-hot,
        all 0 while racing; on the dash and the light panels) and my delta / 5 s (clipped to -1..1)."""
        one = jnp.stack([c.status == VSC, c.status == SC, c.status == SC_IN, c.status == RED]).astype(jnp.float32)
        return jnp.concatenate([jnp.broadcast_to(one, (c.delta.shape[0], 4)),
                                jnp.clip(c.delta / 5.0, -1.0, 1.0)[:, None]], -1)

    def strategist_senses(self, f: race_field.Field, c: Control) -> jax.Array:
        """(n_cars, len(STRATEGIST_SENSES)), in `STRATEGIST_SENSES`' order: what the pit wall sees of race control."""
        F, cars, g = self.field, f.cars, self.field.game
        L = F.tracks.length[cars.track[0]]
        one = jnp.stack([c.status == VSC, (c.status == SC) | (c.status == SC_IN), c.status == RED])
        neutral = c.status != RACING
        laps = jnp.where(neutral, (cars.progress[self.leader(f)] - c.since) / L, 0.0)
        # the Safety Car's queue: the cars on the track behind it, nearest first (a car in the pit lane: the place it
        # would join at)
        behind = (c.sc.lap_pos[0] - cars.lap_pos) % L
        lane = jax.vmap(g.in_pit_lane)(cars)
        queued = self.running(f) & ~lane
        # (step 9g) worked out once: fused into the 8 x 8 count below, the compiler redid `running` and the pit lane
        # for every pair (3.2% of a T4 generation)
        behind, queued = jax.lax.optimization_barrier((behind, queued))
        place = (queued[None, :] & (behind[None, :] < behind[:, None])).sum(1)
        sc = c.sc_on[0] & ((c.status == SC) | (c.status == SC_IN))
        to_entry = (-F.cfg.pit_entry - g.from_pit(cars.track, cars.idx)) % L
        cols = [jnp.broadcast_to(one.astype(jnp.float32), (F.n, 3)),
                jnp.full((F.n, 1), jnp.clip(laps / 5.0, 0.0, 1.0)),
                (c.used.sum(-1) >= 2)[:, None].astype(jnp.float32),
                jnp.where(sc, place / 8.0, 0.0)[:, None],
                jnp.clip(to_entry / 1000.0, 0.0, 1.0)[:, None],
                jnp.full((F.n, 1), c.pit_closed.astype(jnp.float32))]
        return jnp.concatenate(cols, -1)
