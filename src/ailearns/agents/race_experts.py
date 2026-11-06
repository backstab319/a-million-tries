"""Race day 2 (episode 16), step 9d: the scripted experts for race control (plan section 4, D1). "We showed it how to
drive under the flags" (say so).

The same hybrid as episode 15's pit expert (`bc_field.expert`): the brain keeps steering and racing, and the script
only caps its pedal where a rule applies (its own pedal is kept when it's slower), so the expert never drives a corner
worse than the brain does. Speeds come from race control's minimum-time table (`RaceControl.min_pace`: the VSC's
minimum speed at each point, `v_min`); race pace is `vsc_factor` x it (the table is `vsc_factor` x race pace's time).

- **Yellow** (a car's own sector, or the next one when it's in braking distance): ~20% off race pace in a single
  yellow, ~40% in a double (`yellow_off`, `double_off`; the double yellow's rule needs 1.4 x the reference car's
  sector time, about 34% off race pace), and no overtaking.
- **VSC:** hold the delta at +`delta_s` (a speed around `v_min`, faster with time banked, slower without), no
  overtaking.
- **Safety Car** (and "in this lap", the restart's run to the line): the same delta, and close up behind the car
  ahead (the Safety Car too) to `sc_gap_m` (3-8 car lengths; ours 5), no overtaking. At green it's the brain's again.
- Following, everywhere: a target speed of the car ahead's speed plus `follow_k` x the gap past the wanted one, for any
  racing car (or the Safety Car) up to `look_m` ahead on the road, whatever its line (a pass on the road is counted
  wherever it happens). Cars out, crawling or in the pit lane may be passed (the stewards' rule), so they're not
  followed; the brain steers round them.
- **Round a stopped car** (any zone, and wherever one is): a car out or crawling on the track within `dodge_m` ahead
  is passed `dodge_lat` m to its side (the side the car is on if there's room, else the one with more tarmac; the
  script steers), at no more than `pass_v` beside it (braking for that at `decel`). Any stopped car on the tarmac,
  not only one in its line: the brain's line swings across the track in a corner, and checking "in my line" missed
  one that way. Traced (step 9d): a
  car stopped dead on the racing line was hit by 4 of the other 7 with the driver alone, and as many with the speed
  caps only (the brain doesn't steer round it).
- The pedal: `0.5 x (target - speed)`, braking at most ~0.5 g under the VSC and the Safety Car (the lower of it and the brain's in a yellow; the script's alone under the VSC,
  the Safety Car and going round a stopped car: the brain stopped behind wrecks and waited), braking at most `brake` (red flag lesson, step 9c: hard braking on tyres cooled
  behind the Safety Car slid cars off).

`experts` returns the inputs and each expert's zone; the box (`bc_field.expert`, in charge in its own zone) goes on top.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ailearns.agents import bc_field
from ailearns.games import race_control as rcm
from ailearns.games.race_checks import test_driver

YELLOW, VSC, SC, BOX, DODGE = 0, 1, 2, 3, 4  # the zones' columns
ZONES = ("yellow", "vsc", "safety_car", "box", "dodge")


@dataclass(frozen=True)
class ExpertConfig:
    yellow_off: float = 0.2    # share off race pace in a single yellow sector
    double_off: float = 0.4    # in a double yellow sector
    vsc_factor: float = 1.4    # RaceControl's: race pace = this x the minimum speed
    decel: float = 4.0         # m/s^2: slowing for a yellow sector ahead (its speed reached at its line)
    delta_s: float = 1.0       # seconds banked against the minimum time it holds (VSC, Safety Car)
    delta_gain: float = 0.15   # speed share per second of delta off `delta_s`
    sc_gap_m: float = 28.0     # behind the car ahead under the Safety Car (5 car lengths)
    follow_m: float = 15.0     # behind the car ahead in a yellow or a VSC (no overtaking)
    look_m: float = 60.0
    dodge_m: float = 150.0     # a stopped car this far ahead in its path: round it
    dodge_lat: float = 4.0     # metres to its side (a car's width, 2 m, and 2 m of air)
    pass_v: float = 15.0       # m/s beside it (54 km/h), from `pass_lead` m before it
    pass_lead: float = 20.0
    creep_v: float = 2.0       # (step 15g) m/s while still in line with a stopped car close ahead, plus `creep_k` x
    creep_k: float = 0.5       # the metres between my nose and its tail: steer out before going (from a standstill
                               # nose to tail, `pass_v` was full throttle into it: 64% of 15f's cases touched)
    car_len: float = 5.6
    clear_lat: float = 2.4     # sideways from its centre line to be clear of it (a car's width and 0.4 m)
    room_m: float = 9.0        # (step 15g) nose to tail: under this, in line and stopped, back up first (reverse,
                               # straight) to this much room, stop, then steer out (from 10 m centre to centre a car
                               # can't turn out in time: 15f touched in every case at 6 and 10 m, none at 20 m)
    back_clear_m: float = 12.0  # ... only with no car this close behind me in my line
    back_pedal: float = 0.25
    pass_decel: float = 2.5    # m/s^2 slowing for it (run 4: at 4 m/s^2 from the last moment a car still reached one
                               # at 30 m/s, swerved at that speed and overshot into the pit wall)
    follow_k: float = 0.3
    brake: float = 0.5         # the firmest pedal in a yellow or going round a stopped car
    neutral_brake: float = 0.08  # under the VSC and the Safety Car: ~0.5 g, easing off (the pedal is linear on the
                                 # brakes' 6.5 g: run 4's half pedal and run 5's quarter, ~1.6 g, both spun cars on
                                 # cool tyres as the Safety Car came out)
    blind: bool = False        # (step 15m) judge "out" only by what a driver can sense (a car on the track under
                               # `crawl_v`), not race control's record: is that enough to drive by?
    approach_m: float = 400.0  # the box expert's


def experts(ctl: rcm.RaceControl, f, c: rcm.Control, own, cfg: ExpertConfig = ExpertConfig()):
    """(n_cars, 3) the experts' inputs (the brain's own `own` (n_cars, 3) outside their zones) and (n_cars, 5) the
    zones (`ZONES`)."""
    F = ctl.field
    cars = f.cars
    tr, idx = cars.track, cars.idx
    v = jnp.hypot(cars.car.vx, cars.car.vy)
    v_min = 1.0 / ctl.min_pace[tr, idx]
    v_race = cfg.vsc_factor * v_min
    lane = jax.vmap(F.game.in_pit_lane)(cars)
    running = ctl.running(f) & ~lane

    # yellow: here, or the next sector's (reached at its line at `decel`)
    sense = ctl.senses(f)
    here = jnp.where(sense[:, 1] > 0, 1 - cfg.double_off, jnp.where(sense[:, 0] > 0, 1 - cfg.yellow_off, 1e3))
    ahead = jnp.where(sense[:, 3] > 0, 1 - cfg.double_off, jnp.where(sense[:, 2] > 0, 1 - cfg.yellow_off, 1e3))
    to_next = sense[:, 4] * ctl.sector_m
    v_ahead = jnp.sqrt((ahead * v_race) ** 2 + 2 * cfg.decel * to_next)
    v_yellow = jnp.minimum(here * v_race, v_ahead)
    yellow = running & (v_yellow < v + 5.0) & ((here < 1) | (ahead < 1))

    # the delta (VSC, Safety Car): v_min, faster with more banked
    neutral = (c.status == rcm.VSC) | (c.status == rcm.SC) | (c.status == rcm.SC_IN)
    v_delta = v_min * jnp.clip(1.0 + cfg.delta_gain * (c.delta - cfg.delta_s), 0.6, 1.4)
    is_sc = (c.status == rcm.SC) | (c.status == rcm.SC_IN)

    # the car ahead on the road (racing cars at speed, out of the pit lane, and the Safety Car on the track)
    allc, there = F.on_circuit(f, ctl.bodies(c))
    fair = jnp.concatenate([running & (v >= ctl.crawl_v), c.sc_on & ~F.game.in_pit_lane(
        jax.tree.map(lambda a: a[0], c.sc))[None]]) & there
    L = F.tracks.length[tr[0]]
    gap = (allc.lap_pos[None, :] - cars.lap_pos[:, None]) % L  # (i, j) metres j is ahead of i
    near = fair[None, :] & ~jnp.eye(F.n, fair.shape[0], dtype=bool) & (gap < cfg.look_m)
    want = jnp.where(is_sc, cfg.sc_gap_m, cfg.follow_m)
    follow = jnp.hypot(allc.car.vx, allc.car.vy)[None, :] + cfg.follow_k * (gap - want)
    v_follow = jnp.maximum(jnp.where(near, follow, 1e3).min(1), 0.0)

    # a stopped car in its path (out, or crawling, on the track): round it on the side with more tarmac
    half = F.cfg.width / 2
    out = (jnp.concatenate([(jnp.zeros_like(v, bool) if cfg.blind else ~ctl.running(f)) | (v < ctl.crawl_v),
                            jnp.zeros_like(c.sc_on)]) & there
           & ~jax.vmap(F.game.in_pit_lane)(allc))  # (not one in the pit lane: behind the wall)
    on_track = jnp.abs(allc.lat) < half + 2.0

    block = (out & on_track)[None, :] & (gap < cfg.dodge_m) & (gap > 0) & ~jnp.eye(F.n, fair.shape[0], dtype=bool)
    j = jnp.argmin(jnp.where(block, gap, jnp.inf), 1)
    dodging = running & block.any(1)
    lat_j = allc.lat[j]
    mine = jnp.where(cars.lat >= lat_j, 1.0, -1.0)
    room = jnp.where(mine > 0, half - lat_j, lat_j + half)  # tarmac beside it on my side
    side = jnp.where(room >= cfg.dodge_lat + 1.0, mine, jnp.where(half - lat_j > lat_j + half, 1.0, -1.0))
    to_lat = jnp.clip(lat_j + side * cfg.dodge_lat, -half + 1.0, half - 1.0)
    steer = jax.vmap(lambda st, o: test_driver(F.game, st, offset=o)[0])(cars, to_lat)
    # the script steers only when the car would pass too close; clear of it already, the brain keeps steering (run 8:
    # going round a stopped car was 15 of 31 cars out under the Safety Car, all solo: steering to a fixed distance
    # from any stopped car, on the verge or across the track too, often mid-corner)
    close = jnp.abs(cars.lat - lat_j) < cfg.dodge_lat
    v_dodge = jnp.sqrt(cfg.pass_v ** 2
                       + 2 * cfg.pass_decel * jnp.maximum(gap[jnp.arange(F.n), j] - cfg.pass_lead, 0.0))

    target = jnp.where(yellow, v_yellow, 1e3)
    in_line = jnp.abs(cars.lat - lat_j) < cfg.clear_lat
    v_creep = cfg.creep_v + cfg.creep_k * jnp.maximum(gap[jnp.arange(F.n), j] - cfg.car_len, 0.0)
    v_dodge = jnp.where(in_line, jnp.minimum(v_dodge, v_creep), v_dodge)
    nose = gap[jnp.arange(F.n), j] - cfg.car_len
    behind = jnp.where(there[:F.n, None] & ~jnp.eye(F.n, dtype=bool)
                       & (jnp.abs(cars.lat[:, None] - cars.lat[None, :]) < cfg.clear_lat), gap[:, :F.n], jnp.inf).min(0)
    reversing = cars.car.gear == 0
    back_up = dodging & in_line & (nose < cfg.room_m) & (behind > cfg.back_clear_m) & (reversing | (v < 1.0))
    # reversing with the room: brake to a stop (then forward); a stopped car asking for reverse gets it
    rev_act = jnp.stack([jnp.zeros(F.n), jnp.where(reversing & ~back_up, -1.0, jnp.where(reversing, cfg.back_pedal,
                                                                                        0.0)),
                         jnp.where(back_up, 1.0, -1.0)], 1)
    stopping = reversing & dodging & ~back_up   # backed up far enough: stop, then forward
    target = jnp.where(dodging, jnp.minimum(target, v_dodge), target)
    target = jnp.where(neutral & running, jnp.minimum(target, v_delta), target)
    zone_y = (yellow | dodging) & ~neutral
    zone_v = running & (c.status == rcm.VSC)
    zone_s = running & is_sc
    flagged = zone_y | zone_v | zone_s
    target = jnp.where(flagged, jnp.minimum(target, v_follow), target)
    firmest = jnp.where((zone_v | zone_s) & ~dodging, cfg.neutral_brake, cfg.brake)
    script = jnp.clip(0.5 * (target - v), -firmest, 1.0)
    # the script's pedal outright under the VSC and the Safety Car (well under race pace) and going round a stopped car:
    # the brain's own was kept when slower, and it stopped behind a wreck and waited there (step 9d: two cars stood
    # still for 34 and 92 s behind wrecks off their line); in a yellow at near race pace the brain's braking for a
    # corner still wins
    firm = zone_v | zone_s | dodging
    # (run 7: of 38 cars out under the Safety Car, 2 had contact and 31 were away from the pits: solo, with the
    # script's pedal capped at ~0.5 g of braking it couldn't slow for corners, and it gave full throttle out of them on
    # cool tyres.) So the brain's braking still counts near or above the target speed (a corner), not well below it
    # (stopping behind a wreck), and the script's throttle is capped near the brain's
    corner = (own[:, 1] < script) & (v > 0.7 * target)
    # (the floor of 0.3 throttle only for a car nearly stopped, under 10 m/s, the stopped-behind-a-wreck case; run 9: a
    # car the floor kept on the throttle while the brain lifted for a corner slid off)
    floor = jnp.where(v < 10.0, jnp.maximum(own[:, 1], 0.3), own[:, 1])
    neutral_pedal = jnp.where(corner, own[:, 1], jnp.where(script > 0, jnp.minimum(script, floor), script))
    pedal = jnp.where(dodging, script, jnp.where(firm, neutral_pedal, jnp.minimum(own[:, 1], script)))
    act = jnp.stack([jnp.where(dodging & close, steer, own[:, 0]), jnp.where(flagged, pedal, own[:, 1]), jnp.where(flagged, -1.0, own[:, 2])], 1)

    box, zone_b = bc_field.expert(F, f, cfg.approach_m, act)  # the box on top (it keeps `act` on its approach)
    # in the pit lane: behind whatever is ahead in its line (step 9d: a car rammed the Safety Car waiting at the end of
    # the lane at 80 km/h; now they queue behind it, as at a real pit exit)
    in_line = (there[None, :] & ~jnp.eye(F.n, fair.shape[0], dtype=bool) & (gap < 40.0)
               & (jnp.abs(allc.lat[None, :] - cars.lat[:, None]) < 2.4))
    queue = jnp.maximum(jnp.hypot(allc.car.vx, allc.car.vy)[None, :] + (gap - 8.0) * 0.3, 0.0)
    v_lane = jnp.where(in_line, queue, 1e3).min(1)
    box = box.at[:, 1].set(jnp.where(lane, jnp.minimum(box[:, 1], jnp.clip(0.5 * (v_lane - v), -1.0, 1.0)), box[:, 1]))
    act = jnp.where(zone_b[:, None], box, act)
    # going round a stopped car comes first outside the lane (run 3: a called car held on its line beside the pits
    # drove into one stopped on it)
    # and the flags' speeds hold for a called car on the track too (run 3: one held beside the pits at 52 m/s as the
    # Safety Car came out ran wide off the track)
    act = act.at[:, 1].set(jnp.where(zone_b & flagged & ~lane, jnp.minimum(act[:, 1], pedal), act[:, 1]))
    act = act.at[:, 0].set(jnp.where(dodging & close & ~lane, steer, act[:, 0]))
    act = act.at[:, 1].set(jnp.where(dodging & ~lane, jnp.minimum(act[:, 1], pedal), act[:, 1]))
    act = jnp.where(((back_up | stopping) & ~lane)[:, None], rev_act, act)
    return act, jnp.stack([zone_y, zone_v, zone_s, zone_b, dodging], 1)
