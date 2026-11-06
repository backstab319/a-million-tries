// Race control (episode 16), ported line by line from games/race_control.py: the marshalling sectors and their flags,
// the race director (VSC, Safety Car, "in this lap", green; each car's delta against the minimum time and the sectors
// it finished under it) and the Safety Car as a body on the circuit (driven by the scripted test driver, solid, seen);
// the pit entry closed while an out car waits in the lane; (step 10c) the red flag (three stopped in a sector or the
// track blocked; a scripted queue in the pit lane, free repairs and tyres, out behind the Safety Car) and the
// stewards (passes under yellow, the VSC or the Safety Car unless given back, double yellow speed, the closed pit
// entry, two compounds, the black flag; the classification). Settings: data.control (export_web.py field ep15
// control|red); field-check.mjs checks this against Python.

import { CLEAN_AIR } from "./race-engine.js";

export const GREEN = 0, YELLOW = 1, DOUBLE = 2;
export const RACING = 0, VSC = 1, SC = 2, SC_IN = 3, RED = 4;   // the race's status

const mod = (a, n) => ((a % n) + n) % n;
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const speed = (c) => Math.hypot(c.car.vx, c.car.vy);
const wrap = (d, L) => mod(d + L / 2, L) - L / 2;
const N_COMPOUNDS = 3;
// RaceControl's own defaults, for data exported before step 10c
const DEFAULTS = { block_m: 15, block_gap: 3, body_half: 1.5, slot_m: 8, red_min_s: 30, red_wait_s: 300, red_grip: 0.6,
                   red_brake: 0.08, pass_m: 50, dbl_factor: 1.4, two_compounds: true, black_flag_n: 5,
                   reach_m: 250, sc_wear: 0.3 };   // (the last two: es_strategist's, the pit wall's moment, step 10e)

// (P,) metres per second at each centreline point for the reference car: a lap simulator's speed profile (each corner
// at gripG, braking for it at brakeG, accelerating out at accelG), race_control.reference_speeds
export function referenceSpeeds(tr, P, G, gripG = 1.4, brakeG = 1.5, accelG = 0.6, vMax = 90) {
  const ds = tr.length / P, v = Array.from({ length: P }, (_, i) => Math.min(Math.sqrt(gripG * G / Math.max(Math.abs(tr.curvature[i]), 1e-4)), vMax));
  for (let r = 0; r < 2; r++) for (let i = P - 1; i >= 0; i--) v[i] = Math.min(v[i], Math.sqrt(v[(i + 1) % P] ** 2 + 2 * brakeG * G * ds));
  for (let r = 0; r < 2; r++) for (let i = 0; i < P; i++) v[i] = Math.min(v[i], Math.sqrt(v[mod(i - 1, P)] ** 2 + 2 * accelG * G * ds));
  return v;
}

export class RaceControl {
  constructor(F, settings) {
    this.F = F;
    const { sectors, min_pace, ref_sector_s, ...rules } = settings;   // (those three are Python's, for field-check.mjs)
    Object.assign(this, DEFAULTS, rules);   // sector_m, crawl_v, post_clear_s, ..., sc_design
    this.paces = new Map();          // circuit -> the minimum time, seconds per metre at each centreline point
    this.sectorS = new Map();        // circuit -> the reference car's time for each marshalling sector
    this.head_s = F.cfg.pit_exit - 110 - 12;   // the red flag queue's head: 12 m behind the Safety Car's spot
  }

  // ── the marshalling sectors and their flags ──
  sectors(tr) { return Math.max(Math.round(tr.length / this.sector_m), 1); }
  sectorLen(tr) { return tr.length / this.sectors(tr); }
  sector(c) { return Math.min(Math.floor(c.lap_pos / this.sectorLen(c.tr)), this.sectors(c.tr) - 1); }

  minPace(tr) {
    if (!this.paces.has(tr)) {
      const ref = referenceSpeeds(tr, this.F.cfg.points, this.F.game.car.G);
      this.paces.set(tr, ref.map((v) => this.vsc_factor * this.race_pace / v));
    }
    return this.paces.get(tr);
  }

  // the reference car's time for each marshalling sector (the double yellow's yardstick)
  refSectorS(tr) {
    if (!this.sectorS.has(tr)) {
      const P = this.F.cfg.points, n = this.sectors(tr), ref = referenceSpeeds(tr, P, this.F.game.car.G), out = Array(n).fill(0);
      for (let i = 0; i < P; i++) out[Math.min(Math.floor(i * n / P), n - 1)] += tr.length / P / ref[i];
      this.sectorS.set(tr, out);
    }
    return this.sectorS.get(tr);
  }

  running(f, i) { const c = f.cars[i]; return !(c.crashed || f.wrecked[i] || f.finished[i] || this.F.dryOut(f, i)); }

  // the flag car i's incident shows in its own sector: GREEN (none), YELLOW (off the track) or DOUBLE (on it)
  incident(f, i) {
    const F = this.F, c = f.cars[i];
    const out = (c.crashed || f.wrecked[i] || F.dryOut(f, i)) && !f.finished[i] && !F.gone(f, i);
    const started = f.t * F.cfg.dt >= this.start_grace_s;
    const crawling = !(c.crashed || f.wrecked[i] || f.finished[i]) && speed(c) < this.crawl_v && started;
    const off = F.offTrack(c), cleared = off && out && f.out_s[i] >= this.post_clear_s;   // the post calls it safe
    const there = (out || crawling) && !F.game.inPitLane(c) && !cleared;
    return there ? (off ? YELLOW : DOUBLE) : GREEN;
  }

  // each sector's flag (double yellow where a car stopped on the track, single yellow in the sector before)
  flags(f) {
    const tr = f.cars[0].tr, n = this.sectors(tr), out = Array(n).fill(GREEN);
    f.cars.forEach((c, i) => {
      const level = this.incident(f, i), s = this.sector(c);
      out[s] = Math.max(out[s], level);
      if (level === DOUBLE) out[mod(s - 1, n)] = Math.max(out[mod(s - 1, n)], YELLOW);
    });
    return out;
  }

  // ── the race's state ──
  init(f) {
    const n = this.F.n, g = this.F.game, sc = this.scWaiting(f.cars[0].tr);
    return { status: RACING, clear_s: 0, in_lap: 0, armed: Array(n).fill(false), delta: Array(n).fill(0),
             prog: f.cars.map((c) => c.progress), sec: f.cars.map((c) => this.sector(c)), breaks: Array(n).fill(0),
             vsc_n: 0, sc_n: 0, sc, sc_setup: { ...f.setup[0] }, sc_on: false,
             sc_wait: false, sc_lane: false, sc_s: 0, red_n: 0, red_s: 0, slot: Array(n).fill(-1), n_slot: 0,
             pit_closed: false, lane_prev: f.cars.map((c) => g.inPitLane(c)), closed_in: Array(n).fill(0),
             owe: Array.from({ length: n }, () => Array(n).fill(0)), owe_sc: Array(n).fill(0), sc_prev: sc.lap_pos,
             passes: Array(n).fill(0), sec_t: Array(n).fill(0), dbl_in: Array(n).fill(false), dbl: Array(n).fill(0),
             used: f.setup.map((s) => Array.from({ length: N_COMPOUNDS }, (_, k) => s.compound === k)), since: 0,
             lane_hits: Array(n).fill(0) };
  }

  // the Safety Car at the end of the pit lane (in the fast lane, before it bends back to the track)
  scWaiting(tr) {
    const F = this.F;
    return F.place(tr, this.sc_design, tr.pitAt + F.cfg.pit_exit - 110, F.driveLat);
  }

  bodies(c) { return { cars: [c.sc], setup: [c.sc_setup], on: [c.sc_on] }; }

  // the leading car still racing (the furthest along; the lowest index among ties)
  leader(f) {
    let best = -1, far = -Infinity;
    f.cars.forEach((c, i) => { if (this.running(f, i) && c.progress > far) { far = c.progress; best = i; } });
    return Math.max(best, 0);
  }

  // every car racing (out of the pit lane) in one queue behind the Safety Car, each within queue_m of the car ahead
  queued(f, c) {
    const F = this.F, L = f.cars[0].tr.length, n = F.n;
    const behind = f.cars.map((car, i) => (this.running(f, i) && !F.game.inPitLane(car) ? mod(c.sc.lap_pos - car.lap_pos, L) : Infinity));
    let all = true;
    for (let i = 0; i < n; i++) {
      if (!isFinite(behind[i])) continue;
      let ahead = 0;   // the car just ahead of it in the queue (ties: the lower index ahead); 0: the Safety Car
      for (let j = 0; j < n; j++)
        if (isFinite(behind[j]) && (behind[j] < behind[i] || (behind[j] === behind[i] && j < i))) ahead = Math.max(ahead, behind[j]);
      all &&= behind[i] - ahead <= this.queue_m;
    }
    const resuming = c.slot.some((s, i) => s >= 0 && this.running(f, i));   // after a red flag (10c)
    return c.sc_on && !resuming && all;
  }

  // ── the Safety Car ──
  // its inputs: the test driver at sc_share of the minimum pace's speed; held at the end of the pit lane while it
  // waits; "in this lap" it takes the pit lane at its next entry
  scAction(c) {
    const F = this.F, g = F.game, cfg = F.cfg, st = c.sc, s = st.tr.fromPit(st.idx, cfg);
    const inLane = c.sc_lane && g.inPitZone(st.tr, st.idx);
    const ease = clip(Math.min((s + cfg.pit_entry) / 60, (cfg.pit_exit - s) / 100), 0, 1);
    const act = F.testDriver(st, 1.4, 1.5, inLane ? F.driveLat * ease : 0);
    const target = this.sc_share / this.minPace(st.tr)[st.idx];
    const pedal = Math.min(act[1], clip(0.5 * (target - speed(st)), -0.5, 1));
    return [act[0], c.sc_wait ? -1 : pedal];
  }

  // the Safety Car moves one decision (before the cars collide with it)
  driveSc(f, c) {
    const F = this.F, g = F.game, st = c.sc, L = st.tr.length;
    const gap = mod(st.lap_pos - f.cars[this.leader(f)].lap_pos, L);   // the leader behind it
    const wait = c.sc_wait && !(gap > this.sc_release[0] && gap < this.sc_release[1]);
    const s = st.tr.fromPit(st.idx, F.cfg);
    const goingIn = c.status === SC_IN && s > -F.cfg.pit_entry && s < 0;
    const lane = (c.sc_lane || goingIn) && g.inPitZone(st.tr, st.idx);
    c = { ...c, sc_wait: wait, sc_lane: lane, sc_s: c.sc_on && !wait ? c.sc_s + F.cfg.dt : 0 };
    if (!c.sc_on) return c;
    const none = { wing_front: 0, wing_rear: 0, floor: 0, toe: 0, puncture: [false, false, false, false] };
    const moved = g.stepRace(st, this.scAction(c), none, CLEAN_AIR, c.sc_setup);
    return { ...c, sc: { ...moved, car: { ...moved.car, fuel_used: 0 }, crashed: false, done: false } };   // never runs dry
  }

  // ── one decision with race control: the Safety Car moves, the cars move and collide (with it too), the director ──
  step(f, c, actions) {
    const F = this.F, g = F.game;
    f = { ...f, box_call: f.box_call.map((b, i) => b && !(c.pit_closed && !g.inPitLane(f.cars[i]))) };   // closed
    c = this.driveSc(f, c);
    const red = this.redAction(f, c), scripted = this.scripted(f, c);   // the red flag's script (no reverse)
    actions = actions.map((a, i) => (scripted[i] ? [...red[i], ...Array(a.length - 2).fill(-1)] : a));
    const b = this.bodies(c), nf = F.step(f, actions, b);
    const nc = this.direct(nf, { ...c, sc: b.cars[0] });
    return [this.repair(nf, c.slot.map((s) => c.status === RED && nc.status !== RED && s >= 0)), nc];
  }

  // ── the red flag ──
  // driven by its script: under the red flag, and resuming until out of the lane
  scripted(f, c) { return f.cars.map((_, i) => this.running(f, i) && (c.status === RED || c.slot[i] >= 0)); }

  // the script: the test driver at the pit limiter's speed, into the pit lane at its next entry (down the fast lane),
  // easing to a stop on its place in the queue behind the car ahead; resuming, down the fast lane and out
  redAction(f, c) {
    const F = this.F, g = F.game, cfg = F.cfg, [all, there] = F.onCircuit(f, this.bodies(c));
    const stand = this.slot_m - 1;   // at a standstill, centre to centre
    return f.cars.map((car, i) => {
      const tr = car.tr, L = tr.length, s = tr.fromPit(car.idx, cfg), lane = g.inPitLane(car);
      const take = lane || (s > -cfg.pit_entry && s < -cfg.pit_wall_len - 20);   // not past the start of the pit wall
      const ease = clip(Math.min((s + cfg.pit_entry) / 60, (cfg.pit_exit - s) / 100), 0, 1);
      const act = F.testDriver(car, this.red_grip, this.red_grip, take ? F.driveLat * ease : 0);   // cautious
      const v = speed(car);
      const to = c.status === RED && c.slot[i] >= 0 ? this.head_s - c.slot[i] * this.slot_m - F.pitS(car) : 1e4;
      let target = Math.min(cfg.pit_limit, Math.sqrt(2 * 3 * Math.max(to - 0.5, 0)));
      all.forEach((o, j) => {   // behind any car in its path: slowing at 3 m/s^2, a second behind at speed
        const gap = mod(o.lap_pos - car.lap_pos, L);
        if (!there[j] || j === i || gap >= 300 || Math.abs(o.lat - car.lat) >= 2.4) return;
        target = Math.min(target, speed(o) + Math.sqrt(2 * 3 * Math.max(gap - stand - v, 0)));
      });
      const pedal = Math.min(act[1], clip(0.5 * (target - v), -this.red_brake, 1));
      return [act[0], target < 0.3 && v < 1 ? -1 : pedal];
    });
  }

  // three or more cars stopped on the track in one marshalling sector, or stopped cars within block_m of each other
  // leaving no gap of block_gap across the tarmac
  redNeeded(f) {
    const [there] = this.stoppedOnTrack(f), cars = f.cars, n = cars.length, L = cars[0].tr.length;
    const sec = cars.map((c) => this.sector(c)), w = this.F.cfg.width / 2, bh = this.body_half;
    for (let i = 0; i < n; i++) if (there[i] && sec.filter((s, j) => there[j] && s === sec[i]).length >= 3) return true;
    const widest = (near) => {   // the widest gap across the tarmac between the cars `near` and the edges
      const lo = cars.map((c, j) => (near[j] ? c.lat - bh : Infinity)), hi = cars.map((c, j) => (near[j] ? c.lat + bh : -Infinity));
      let best = -Infinity;
      for (const x of [-w, ...hi.map((h) => Math.max(h, -w))]) {   // where a gap could start
        if (x >= w || lo.some((l, j) => l < x && x < hi[j])) continue;
        const to = Math.min(w, ...lo.filter((l) => l >= x));
        best = Math.max(best, to - x);
      }
      return best;
    };
    return cars.some((ci, i) => there[i] && widest(cars.map((cj, j) => there[j] && Math.abs(wrap(ci.lap_pos - cj.lap_pos, L)) < this.block_m)) < this.block_gap);
  }

  // the red flag's free work on the `mask` cars: damage repaired, new tyres (the planned compound); not a stop
  repair(f, mask) {
    if (!mask.some(Boolean)) return f;
    const tyre = this.F.cfg.tyre_start;
    return { ...f,
      cars: f.cars.map((c, i) => (mask[i] ? { ...c, car: { ...c.car, wear: [0, 0, 0, 0], temp: [tyre, tyre, tyre, tyre] } } : c)),
      damage: f.damage.map((d, i) => (mask[i] ? { wing_front: 0, wing_rear: 0, floor: 0, toe: 0, puncture: [false, false, false, false] } : d)),
      flat_m: f.flat_m.map((m, i) => (mask[i] ? [0, 0, 0, 0] : m)),
      setup: f.setup.map((s, i) => (mask[i] ? { ...s, compound: f.plan[i] } : s)),
      box_call: f.box_call.map((b, i) => b && !mask[i]) };
  }

  // out cars stopped on the track waiting for the marshals, and those on the racing line (within line_m of the centre)
  stoppedOnTrack(f) {
    const F = this.F, there = [], line = [];
    f.cars.forEach((c, i) => {
      const atRest = c.crashed || speed(c) < this.crawl_v;
      const out = (c.crashed || f.wrecked[i] || F.dryOut(f, i)) && atRest && !f.finished[i] && !F.gone(f, i);
      there.push(out && !F.offTrack(c) && !F.game.inPitLane(c));
      line.push(there[i] && Math.abs(c.lat) < this.line_m);
    });
    return [there, line];
  }

  // the race director, once a decision (after the field's step): the race's status and every car's delta
  direct(f, c) {
    const F = this.F, g = F.game, dt = F.cfg.dt, n = F.n, cars = f.cars, tr = cars[0].tr;
    const [there, tarmac] = this.stoppedOnTrack(f);
    const stopped = there.filter(Boolean).length;
    const needSc = stopped >= 2 || tarmac.some(Boolean), needRed = this.redNeeded(f);
    const clear = stopped === 0, clear_s = clear ? c.clear_s + dt : 0;
    const running = cars.map((_, i) => this.running(f, i)), lane = cars.map((car) => g.inPitLane(car));
    const v = cars.map(speed), queuedRed = running.every((r, i) => !r || (c.slot[i] >= 0 && lane[i] && v[i] < 0.5));
    const resume = c.red_s >= this.red_min_s && clear && (queuedRed || clear_s >= this.red_wait_s);
    const leaderLaps = Math.max(...cars.map((car, i) => (running[i] || f.finished[i] ? Math.floor(car.progress / tr.length) : -1)));
    const st = c.status;
    const rules = [
      [needRed && st !== RED, RED],                                             // the red flag
      [st === RED && resume, SC],                                               // resuming behind the Safety Car
      [needSc && st !== SC && st !== RED, SC],                                  // (back) to the Safety Car
      [st === RACING && stopped > 0, VSC],                                      // a VSC
      [st === VSC && clear_s >= this.vsc_end_s, RACING],                        // VSC ending -> green
      [st === SC && clear && !c.sc_wait && (this.queued(f, c) || c.sc_s >= this.queue_wait_s), SC_IN],   // in this lap
      [st === SC_IN && !c.sc_on && leaderLaps > c.in_lap, RACING],              // it's in, the leader at the line: green
    ];
    const stNew = (rules.find(([when]) => when) ?? [true, st])[1];
    // the Safety Car: out at the end of the pit lane when it's deployed (or back), off the circuit once it's in the
    // pit lane past its entry "in this lap" (and the leader's laps then: green at its next line)
    const sc = c.sc, spawn = stNew === SC && !c.sc_on;
    const goneIn = stNew === SC_IN && c.sc_on && g.inPitLane(sc) && sc.tr.fromPit(sc.idx, F.cfg) < 0;
    const off = goneIn || stNew === RACING || stNew === VSC || stNew === RED;
    const scOn = spawn ? true : off ? false : c.sc_on;
    // the delta: + the time taken, - the minimum time for the metres driven (outside the pit lane, still racing),
    // from the car's next sector line after the deployment
    const neutral = (s) => s === VSC || s === SC || s === SC_IN;
    const pace = this.minPace(tr), armed = [], delta = [], breaks = [], sec = [];
    for (let i = 0; i < n; i++) {
      sec.push(this.sector(cars[i]));
      const crossed = sec[i] !== c.sec[i];
      armed.push(neutral(stNew) && (c.armed[i] || (neutral(st) && crossed)));
      const counted = armed[i] && running[i] && !lane[i];
      const took = counted ? dt - (cars[i].progress - c.prog[i]) * pace[cars[i].idx] : 0;
      delta.push(armed[i] && c.armed[i] ? c.delta[i] + took : 0);
      breaks.push(c.breaks[i] + (c.armed[i] && armed[i] && running[i] && crossed && delta[i] < 0));   // under the minimum
    }
    const newSc = stNew === SC && st !== SC && st !== SC_IN && st !== RED;
    // the red flag's queue: a car gets the next place as it enters the pit lane (in the order they arrive; together,
    // the one further along first), if it hasn't passed it already (then it goes round again)
    const red = stNew === RED, ps = cars.map((car) => F.pitS(car));
    const cand = cars.map((_, i) => red && running[i] && lane[i] && c.slot[i] < 0);
    const want = ps.map((p, i) => c.n_slot + cand.filter((x, j) => x && ps[j] > p).length);
    const ok = cand.map((x, i) => x && ps[i] < this.head_s - want[i] * this.slot_m - 2);
    const slot = cars.map((car, i) => (!red && !g.inPitZone(car.tr, car.idx) ? -1 : ok[i] ? want[i] : c.slot[i]));
    const nSlot = red ? Math.max(c.n_slot, ...want.map((w, i) => (ok[i] ? w + 1 : 0))) : 0;
    // the pit entry closed: an out car in the lane, waiting for its crew (not under the red flag)
    const outLane = cars.some((car, i) => !running[i] && !f.finished[i] && lane[i] && !F.gone(f, i));
    const scState = spawn ? this.scWaiting(tr) : sc;
    const moved = cars.map((car, i) => car.progress - c.prog[i]);
    const [owe, owe_sc, passes] = this.overtakes(f, c, stNew, lane, running, moved, scState, scOn);
    const flags = this.flags(f), ref = this.refSectorS(tr), sec_t = [], dbl = [], dbl_in = [];
    for (let i = 0; i < n; i++) {   // a whole sector under a double yellow in under dbl_factor x the reference's time
      const crossed = sec[i] !== c.sec[i], slow = c.sec_t[i] + dt >= this.dbl_factor * ref[c.sec[i]];
      sec_t.push(crossed ? 0 : c.sec_t[i] + dt);
      dbl.push(c.dbl[i] + (crossed && c.dbl_in[i] && running[i] && !slow));
      dbl_in.push(crossed ? flags[sec[i]] === DOUBLE && !lane[i] : c.dbl_in[i]);
    }
    const began = stNew !== st && (stNew === VSC || stNew === SC || stNew === RED);
    return { ...c, status: stNew, clear_s, in_lap: goneIn ? leaderLaps : c.in_lap, armed, delta,
             prog: cars.map((car) => car.progress), sec, breaks, vsc_n: c.vsc_n + (stNew === VSC && st !== VSC),
             sc_n: c.sc_n + newSc, sc: scState, sc_on: scOn,
             sc_wait: spawn ? st !== RED : c.sc_wait, sc_lane: spawn ? true : c.sc_lane && scOn,
             red_n: c.red_n + (red && st !== RED), red_s: red && st === RED ? c.red_s + dt : 0, slot, n_slot: nSlot,
             pit_closed: outLane && !red, lane_prev: lane,
             closed_in: c.closed_in.map((k, i) => k + (lane[i] && !c.lane_prev[i] && running[i] && slot[i] < 0 && c.pit_closed)),
             owe, owe_sc, sc_prev: scState.lap_pos, passes, sec_t, dbl_in, dbl,
             used: c.used.map((u, i) => u.map((x, k) => x || f.setup[i].compound === k)),
             since: began ? cars[this.leader(f)].progress : c.since,
             // (step 13d) collisions it caused in the pit lane (5 s each; not in the red flag's queue)
             lane_hits: (c.lane_hits ?? Array(this.F.n).fill(0)).map((k, i) => k + (!!f.fault?.[i] && lane[i] && running[i] && slot[i] < 0 && !red)) };
  }

  // ── the stewards ──
  // places taken under yellow, the VSC or the Safety Car (a pass on the road, within pass_m): owed back for a
  // marshalling sector's metres; given back it's cleared, not given back a penalty. Passing a car that's out,
  // crawling or in the pit lane, or passing in the pit lane, is allowed. Passing the Safety Car on the track counts.
  overtakes(f, c, stNew, lane, running, moved, sc, scOn) {
    const cars = f.cars, n = cars.length, L = cars[0].tr.length, flags = this.flags(f), pm = this.pass_m;
    const neutral = stNew === VSC || stNew === SC || stNew === SC_IN;
    const able = cars.map((_, i) => running[i] && !lane[i]), fair = cars.map((car, i) => able[i] && speed(car) >= this.crawl_v);
    const flagged = cars.map((car) => (neutral || flags[this.sector(car)] !== GREEN) && stNew !== RED);
    const now = (i, j) => wrap(cars[i].lap_pos - cars[j].lap_pos, L), before = (i, j) => wrap(mod(c.prog[i], L) - mod(c.prog[j], L), L);
    const owe = [], passes = [];
    for (let i = 0; i < n; i++) {
      owe.push([]);
      let expiredN = 0;
      for (let j = 0; j < n; j++) {
        const b = before(i, j), a = now(i, j), owing = c.owe[i][j] > 0;
        const passed = b < 0 && b > -pm && a >= 0 && a < pm && able[i] && flagged[i] && fair[j] && !(c.owe[j][i] > 0);
        const left = owing ? c.owe[i][j] - Math.max(moved[i], 0) : 0;
        const cleared = owing && ((a < 0 && a > -pm) || !running[j] || !running[i]);
        const expired = owing && !cleared && left <= 0;
        expiredN += expired;
        owe[i].push(passed ? this.sector_m : owing && !cleared && !expired ? left : 0);
      }
      passes.push(expiredN);
    }
    const scTrack = scOn && c.sc_on && !this.F.game.inPitLane(sc);
    const owe_sc = cars.map((car, i) => {
      const a = wrap(car.lap_pos - sc.lap_pos, L), b = wrap(mod(c.prog[i], L) - c.sc_prev, L), owing = c.owe_sc[i] > 0;
      const passed = scTrack && able[i] && b < 0 && b > -pm && a >= 0 && a < pm;
      const left = owing ? c.owe_sc[i] - Math.max(moved[i], 0) : 0;
      const cleared = owing && ((a < 0 && a > -pm) || !running[i] || !scOn);
      const expired = owing && !cleared && left <= 0;
      passes[i] += c.passes[i] + expired;
      return passed ? this.sector_m : owing && !cleared && !expired ? left : 0;
    });
    return [owe, owe_sc, passes];
  }

  // seconds added to each car's race time; penalties counted one each; the black flag; disqualified
  penaltyS(c) { return c.breaks.map((b, i) => 5 * (b + c.passes[i] + c.dbl[i] + (c.lane_hits?.[i] ?? 0)) + 10 * c.closed_in[i]); }
  infringements(c) { return c.breaks.map((b, i) => b + c.passes[i] + c.dbl[i] + c.closed_in[i] + (c.lane_hits?.[i] ?? 0)); }
  blackFlagged(c) { return this.infringements(c).map((k) => this.black_flag_n > 0 && k >= this.black_flag_n); }
  disqualified(f, c) {
    const black = this.blackFlagged(c);
    return f.cars.map((_, i) => (this.two_compounds && f.finished[i] && c.used[i].filter(Boolean).length < 2) || black[i]);
  }

  // each car's position after the stewards (0 = winner): finishers by race time + penalties; then cars still running
  // by distance, less their penalties at their own average speed; the out; the disqualified last
  classify(f, c) {
    const pen = this.penaltyS(c), dsq = this.disqualified(f, c), t = Math.max(f.t * this.F.cfg.dt, 1);
    const key = f.cars.map((car, i) => {
      const group = dsq[i] ? 3 : f.finished[i] ? 0 : this.running(f, i) ? 1 : 2;
      const pace = Math.max(car.progress, 0) / t;
      return [group, f.finished[i] ? f.finish_t[i] + pen[i] : -(car.progress - pen[i] * pace), i];
    });
    const pos = Array(key.length);
    key.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]).forEach((k, p) => { pos[k[2]] = p; });
    return pos;
  }

  // ── (step 10d) what the driver and the pit wall see of race control (DRIVER_SENSES, STRATEGIST_SENSES) ──
  // what each driver sees of the flags: its own sector yellow, double yellow; the next sector's yellow, double yellow,
  // and the metres to it / sector_m
  flagSenses(f) {
    const fl = this.flags(f);
    return f.cars.map((c) => {
      const s = this.sector(c), here = fl[s], ahead = fl[(s + 1) % this.sectors(c.tr)];
      const toNext = (s + 1) * this.sectorLen(c.tr) - c.lap_pos;
      return [here === YELLOW, here === DOUBLE, ahead === YELLOW, ahead === DOUBLE].map(Number)
        .concat(clip(toNext / this.sector_m, 0, 2));
    });
  }

  // the driver's 15, in DRIVER_SENSES' order: the status (VSC, Safety Car, "in this lap", red flag; one-hot), the
  // flags, my delta / 5 s, the car directly ahead is the Safety Car, my box's offset across the lane / 5 m, then
  // (step 15n) the nearest car stopped on the track ahead: seen, metres to it / sense range, its offset across / half
  // the width (stopped: under crawl_v, not in the pit lane or finished, on the tarmac or within 2 m of its edge)
  driverSenses(f, c) {
    const F = this.F, n = F.n, [all, there] = F.onCircuit(f, this.bodies(c)), L = f.cars[0].tr.length;
    const status = [VSC, SC, SC_IN, RED].map((s) => Number(c.status === s)), flags = this.flagSenses(f);
    const half = F.cfg.width / 2;
    const still = f.cars.map((car, j) => there[j] && speed(car) < this.crawl_v && !F.game.inPitLane(car) &&
      !f.finished[j] && Math.abs(car.lat) < half + 2);
    return f.cars.map((car, i) => {
      let sj = -1, sBest = Infinity;   // the nearest stopped car ahead in sight (ties: the lower index)
      for (let j = 0; j < n; j++) {
        const gap = mod(all[j].lap_pos - car.lap_pos, L);
        if (still[j] && j !== i && gap < F.sense_range && gap < sBest) { sBest = gap; sj = j; }
      }
      const stopped = sj < 0 ? [0, 0, 0] : [1, sBest / F.sense_range, (f.cars[sj].lat - car.lat) / half];
      let nearest = -1, best = Infinity;   // the nearest body ahead in sight (ties: the lower index)
      all.forEach((o, j) => {
        const gap = mod(o.lap_pos - car.lap_pos, L);
        if (there[j] && j !== i && gap < F.sense_range && gap < best) { best = gap; nearest = j; }
      });
      const box = F.game.inPitLane(car) ? clip((F.boxLat - car.lat) / 5, -1, 1) : 0;
      return [...status, ...flags[i], clip(c.delta[i] / 5, -1, 1), Number(nearest >= n), box, ...stopped];
    });
  }

  // the driver's whole input: RaceField.observe (the Safety Car seen as a car), then the 12
  observe(f, c) {
    const own = this.F.observe(f, this.bodies(c)), rc = this.driverSenses(f, c);
    return own.map((o, i) => [...o, ...rc[i]]);
  }

  // the pit wall's 8, in STRATEGIST_SENSES' order: the status (VSC, Safety Car or "in this lap", red flag), the
  // leader's laps since it began / 5, two dry compounds used, its place in the Safety Car's queue / 8, metres to the
  // pit entry / 1 km, the pit entry closed
  strategistSenses(f, c) {
    const F = this.F, g = F.game, cfg = F.cfg, L = f.cars[0].tr.length, st = c.status;
    const one = [st === VSC, st === SC || st === SC_IN, st === RED].map(Number);
    const laps = st !== RACING ? (f.cars[this.leader(f)].progress - c.since) / L : 0;
    // the Safety Car's queue: the cars on the track behind it, nearest first (a car in the pit lane: where it'd join)
    const behind = f.cars.map((car) => mod(c.sc.lap_pos - car.lap_pos, L));
    const queued = f.cars.map((car, i) => this.running(f, i) && !g.inPitLane(car));
    const sc = c.sc_on && (st === SC || st === SC_IN);
    return f.cars.map((car, i) => {
      const place = queued.filter((q, j) => q && behind[j] < behind[i]).length;
      const toEntry = mod(-cfg.pit_entry - car.tr.fromPit(car.idx, cfg), L);
      return [...one, clip(laps / 5, 0, 1), Number(c.used[i].filter(Boolean).length >= 2), sc ? place / 8 : 0,
              clip(toEntry / 1000, 0, 1), Number(c.pit_closed)];
    });
  }
}
