// Checks field-engine.js (live race day) against the Python race: node field-check.mjs
// field-data.json holds the whole Python race state at a few moments of the final race (the start, the first AI pit
// wall decision, just before the first stop, a late hit) and every car's motion for 10 s after each. From each
// snapshot the JS race runs on its own (its own driver, pit wall, contact, ...) and is compared with Python: the
// senses both computed from the same state, and how far the cars drift apart. Like Python re-run with a different
// batch size, the paths split after a few seconds (floats: doubles here, float32 there; contact amplifies it).
// (ep16 step 10) Each snapshot is taken just before an event (a stop, a pull-back, a held release, a pull-off, a
// recovery...), so the rules are checked over its first second too: the stop's work, wrecked, out, recovered. Two
// references: field-data.json's (the live game: episode 15's lane) and field-ref-ep16.json's (the new two-lane pit
// lane, the box expert driving called cars there). The sim replays Python's inputs (on its own the driver amplifies
// the tiniest sense difference near its box: the box sense is a square root, steepest at the mark); the driver is
// checked apart, on the same senses: its outputs against Python's. (Step 10b) A third reference, field-ref-control.json:
// episode 15's lane with race control (race-control.js): each sector's flag, the race's status, every car's delta and
// sectors under the minimum time, the Safety Car's path and the pit entry, from snapshots before each race control
// event; and the minimum time table itself. (Step 10c) field-ref-red.json: a pile-up injected at 600 s (a red flag),
// the queue, the restart and the stewards' counts at every step, and the classification from the race's final state.
import { readFileSync } from "node:fs";
import { decode } from "./engine.js";
import { Race, Strategist } from "./field-engine.js";
import { readdirSync } from "node:fs";

const load = (name) => JSON.parse(readFileSync(new URL(name, import.meta.url)));
const live = load("./field-data.json");
const refs = [["episode 15's lane, live game", live], ["episode 16's lane, the box expert's stops", { ...live, ...load("./field-ref-ep16.json") }]];
for (const name of readdirSync(new URL(".", import.meta.url)).filter((x) => /^field-ref-(control(-\d+)?|red)\.json$/.test(x)).sort())
  refs.push([`episode 15's lane, race control (${name})`, { ...live, ...load(`./${name}`), file: name }]);   // (-k: circuit k)
// (step 10d) Python's race control senses at each of those snapshots (export_web.py field_senses)
const senses = load("./field-ref-senses.json"), seenNonZero = { driver: new Set(), strat: new Set() };

// one car of a Python race.State (export_web._tree) as race-engine.js's
function carFrom(F, tr, cars, i) {
  const c = cars.car, design = cars.design[i];
  return {
    car: { x: c.pos[i][0], y: c.pos[i][1], yaw: c.yaw[i], vx: c.vx[i], vy: c.vy[i], r: c.r[i], steer: c.steer[i], ax: c.ax[i],
           ay: c.ay[i], gear: c.gear[i], shift: c.shift[i], ride: c.ride[i], temp: c.temp[i], wear: c.wear[i], fuel_used: c.fuel_used[i] },
    design, p: F.game.car.derive(design), tr, idx: cars.idx[i], lat: cars.lat[i], lap_pos: cars.lap_pos[i],
    progress: cars.progress[i], t: cars.t[i], last: cars.last[i], surface: cars.surface[i], dirt: cars.dirt[i],
    sliding: cars.sliding[i], plank: cars.plank[i], crashed: cars.crashed[i], done: cars.done[i], info: null };
}

// a Python Field (and race control's Control) as field-engine.js's race state
function fromPython(snap, race) {
  const F = race.F, p = snap.field, tr = race.tr, inf = (x) => (x >= 1e29 ? Infinity : x);
  const cars = p.cars;
  const field = {
    cars: cars.design.map((_, i) => carFrom(F, tr, cars, i)),
    finished: p.finished, finish_t: p.finish_t.map(inf), flag: p.flag, flag_laps: p.flag_laps, bump: p.bump, hit: p.hit,
    damage: p.damage.wing_front.map((_, i) => ({ wing_front: p.damage.wing_front[i], wing_rear: p.damage.wing_rear[i],
      floor: p.damage.floor[i], toe: p.damage.toe[i], puncture: p.damage.puncture[i] })),
    wrecked: p.wrecked, setup: p.setup.compound.map((k, i) => ({ compound: k, fuel: p.setup.fuel[i], wear: p.setup.wear[i] })),
    plan: p.plan, box_call: p.box_call, service: p.service, served: p.served, stops: p.stops, t: p.t, done: p.done,
    burst_at: p.burst_at, flat_m: p.flat_m, out_s: p.out_s, fault: p.fault ?? Array(p.out_s.length).fill(false), free_s: p.free_s ?? Array(p.out_s.length).fill(1e3), rub_s: p.rub_s ?? Array(p.out_s.length).fill(0),
  };
  const pc = snap.control, one = (x) => (Array.isArray(x) ? x[0] : x);
  const c = pc && { ...pc, sc: carFrom(F, tr, pc.sc, 0), sc_on: one(pc.sc_on), sc_wait: one(pc.sc_wait), sc_lane: one(pc.sc_lane),
    sc_setup: { compound: pc.sc_setup.compound[0], fuel: pc.sc_setup.fuel[0], wear: pc.sc_setup.wear[0] } };
  return { f: field, setFrom: snap.set_from, prev: snap.prev, ...(c ? { c, prevSt: snap.prev_st ?? c.status } : {}) };
}

let ok = true;
for (const [title, data] of refs) {
const ref = data.reference;
console.log(`${title}:`);
if (data.control) {   // race control's minimum time table and sectors, worked out in JS from the circuits
  const probe = new Race(data, ref.track, { state: {}, control: true }), mp = decode(data.control.min_pace);
  const rs = data.control.ref_sector_s && decode(data.control.ref_sector_s);
  let worst = 0, worstSec = 0, sectorsOk = true;
  probe.F.game.circuits.forEach((tr, t) => {
    const mine = probe.ctl.minPace(tr), P = mine.length;
    for (let i = 0; i < P; i++) worst = Math.max(worst, Math.abs(mine[i] / mp.data[t * P + i] - 1));
    sectorsOk &&= probe.ctl.sectors(tr) === data.control.sectors[t];
    if (rs) probe.ctl.refSectorS(tr).forEach((x, q) => { worstSec = Math.max(worstSec, Math.abs(x / rs.data[t * rs.shape[1] + q] - 1)); });
  });
  const good = worst < 1e-5 && worstSec < 1e-5 && sectorsOk;
  ok &&= good;
  console.log(`${good ? "ok " : "BAD"} the minimum time: off by up to ${worst.toExponential(1)} (relative)${rs ? `, sector reference times ${worstSec.toExponential(1)}` : ""}; sectors ${sectorsOk ? "same" : "differ"}`);
}
if (data.final) {   // the stewards' classification from the race's final state
  const race = new Race(data, ref.track, { state: {}, control: true });
  Object.assign(race, fromPython({ ...data.final, set_from: [], prev: [] }, race));
  const fin = data.final, pos = race.ctl.classify(race.f, race.c), pen = race.ctl.penaltyS(race.c);
  const dsq = race.ctl.disqualified(race.f, race.c), black = race.ctl.blackFlagged(race.c);
  const same = (a, b) => a.every((x, i) => Number(x) === Number(b[i]));
  const good = same(pos, fin.classify) && same(pen, fin.penalty) && same(dsq, fin.dsq) && same(black, fin.black);
  ok &&= good;
  console.log(`${good ? "ok " : "BAD"} the classification: ${pos.map((p) => p + 1).join(" ")} (Python ${fin.classify.map((p) => p + 1).join(" ")}); `
    + `penalties ${pen.join(" ")} s; disqualified ${dsq.map(Number).join("")}, black flag ${black.map(Number).join("")}`);
}
ref.snaps.forEach((snap, si) => {
  const race = new Race(data, ref.track, { state: {}, control: !!data.control });
  Object.assign(race, fromPython(snap, race));
  const pos = decode(snap.pos), obs = decode(snap.obs), sx = decode(snap.strat_x), steps = pos.shape[0], n = race.F.n;
  const act = decode(snap.action), service = decode(snap.service), A = act.shape[2];
  let rcErr = "";   // (step 10d) race control's 12 + 8 senses from the snapshot's state, against Python's
  const py = data.file && senses.refs[data.file]?.[si];
  if (py) for (const [kind, mine] of [["driver", race.ctl.driverSenses(race.f, race.c)], ["strat", race.ctl.strategistSenses(race.f, race.c)]])
    mine.forEach((row, i) => row.forEach((x, q) => {
      if (x !== 0) seenNonZero[kind].add(q);
      const e = Math.abs(x - py[kind][i][q]);
      if (e > 1e-4 && !rcErr) rcErr = `${kind === "driver" ? "driver" : "pit wall"} ${senses[kind][q]} car ${i + 1}: ${x.toFixed(4)} v ${py[kind][i][q].toFixed(4)}`;
    }));
  if (py?.driver_stopped) {   // (step 15n) the same moment with one car stopped dead: the stopped-car senses
    const body = race.f.cars[py.stopped_car].car, [vx, vy] = [body.vx, body.vy];
    body.vx = body.vy = 0;
    race.ctl.driverSenses(race.f, race.c).forEach((row, i) => row.forEach((x, q) => {
      if (x !== 0) seenNonZero.driver.add(q);
      const e = Math.abs(x - py.driver_stopped[i][q]);
      if (e > 1e-4 && !rcErr) rcErr = `driver ${senses.driver[q]} car ${i + 1} (car ${py.stopped_car + 1} stopped): ${x.toFixed(4)} v ${py.driver_stopped[i][q].toFixed(4)}`;
    }));
    [body.vx, body.vy] = [vx, vy];
  }
  let k = 0;
  race.inputs = () => Array.from({ length: n }, (_, i) => Array.from(act.data.subarray((k * n + i) * A, (k * n + i + 1) * A)));
  let obsErr = 0, obs0 = 0, sx0 = 0, sxErr = 0, worstSense = "", split = null, drift1 = 0, drift5 = 0, callsOk = true, stopsOk = true;
  let rules = "", drv0 = 0, drvErr = 0;   // the first rule state to differ in the first second; the driver's outputs
  let rc = "", deltaErr = 0, scDrift = 0, scDrift1 = 0;   // race control: what differs first over 10 s; the delta; the Safety Car
  const dl = snap.delta && decode(snap.delta), scp = snap.sc_pos && decode(snap.sc_pos);
  for (; k < steps; k++) {
    if (k % 20 === 0) {   // the senses, from (nearly) the same state
      const x = Strategist.senses(race.F, race.f, race.setFrom, 0);
      for (let i = 0; i < n; i++) for (let q = 0; q < 25; q++) sxErr = Math.max(sxErr, Math.abs(x[i][q] - sx.data[((k / 20) * n + i) * 25 + q]));
      if (k === 0) sx0 = sxErr;
    }
    race.step();
    if (k % 20 === 0 && !ref.replay) for (let i = 0; i < n; i++) {   // (the box expert's inputs aren't the driver's)
      const a = race.driver.act(race.lastObs[i]);
      for (let q = 0; q < A; q++) drvErr = Math.max(drvErr, Math.abs(a[q] - act.data[(k * n + i) * A + q]));
      if (k === 0) drv0 = drvErr;
    }
    if (k % 20 === 0) for (let i = 0; i < n; i++) for (let q = 0; q < 115; q++) {
      const e = Math.abs(race.lastObs[i][q] - obs.data[((k / 20) * n + i) * 115 + q]);
      if (e > obsErr) { obsErr = e; worstSense = `car ${i + 1} sense ${q} at step ${k}`; }
      if (k === 0) obs0 = obsErr;
    }
    let d = 0;
    for (let i = 0; i < n; i++) d = Math.max(d, Math.hypot(race.f.cars[i].car.x - pos.data[(k * n + i) * 2], race.f.cars[i].car.y - pos.data[(k * n + i) * 2 + 1]));
    if (k === 19) drift1 = d;
    if (k === 99) drift5 = d;
    if (split === null && d > 0.01) split = k;
    if (race.f.box_call.some((b, i) => b !== snap.box_call[k][i])) callsOk = false;
    if (race.f.stops.some((s, i) => s !== snap.stops[k][i])) stopsOk = false;
    if (race.c) {
      const c = race.c, at = `at step ${k}`, fl = race.ctl.flags(race.f);
      const differ = c.status !== snap.status[k] ? "status" : fl.some((x, q) => x !== snap.flags[k][q]) ? "flags"
        : c.sc_on !== snap.sc_on[k] ? "safety car on" : c.sc_wait !== snap.sc_wait[k] ? "safety car waiting"
        : c.pit_closed !== snap.pit_closed[k] ? "pit closed" : c.breaks.some((b, i) => b !== snap.breaks[k][i]) ? "breaks"
        : c.armed.some((a, i) => a !== snap.armed[k][i]) ? "delta armed"
        : snap.slot && c.slot.some((x, i) => x !== snap.slot[k][i]) ? "red flag queue"
        : snap.passes && c.passes.some((x, i) => x !== snap.passes[k][i]) ? "pass penalties"
        : snap.dbl && c.dbl.some((x, i) => x !== snap.dbl[k][i]) ? "double yellow penalties"
        : snap.closed_in && c.closed_in.some((x, i) => x !== snap.closed_in[k][i]) ? "closed entry penalties"
        : snap.lane_hits && c.lane_hits.some((x, i) => x !== snap.lane_hits[k][i]) ? "pit lane collisions"
        : snap.red_s && Math.abs(c.red_s - decode(snap.red_s).data[k]) > 1e-3 ? "red flag time" : "";
      if (differ && !rc) rc = `${differ} ${at}`;
      if (k < 20 && differ) rules ||= `${differ} ${at}`;
      for (let i = 0; i < n; i++) deltaErr = Math.max(deltaErr, Math.abs(c.delta[i] - dl.data[k * n + i]));
      if (c.sc_on) scDrift = Math.max(scDrift, Math.hypot(c.sc.car.x - scp.data[2 * k], c.sc.car.y - scp.data[2 * k + 1]));
      if (k === 19) scDrift1 = scDrift;
    }
    if (k < 20 && !rules) for (let i = 0; i < n; i++) {
      const f = race.f, at = `car ${i + 1} at step ${k}`;
      if (Math.abs(f.service[i] - service.data[k * n + i]) > 1e-3) rules ||= `stop's work ${at}`;
      if (f.wrecked[i] !== snap.wrecked[k][i]) rules ||= `wrecked ${at}`;
      if (f.cars[i].crashed !== snap.crashed[k][i]) rules ||= `out ${at}`;
      if (race.F.gone(f, i) !== snap.gone[k][i]) rules ||= `recovered ${at}`;
      if (f.stops[i] !== snap.stops[k][i]) rules ||= `stops ${at}`;
      if (f.box_call[i] !== snap.box_call[k][i]) rules ||= `box call ${at}`;   // (step 10e: the pit wall's moment)
    }
  }
  // from the same state: the same senses, (nearly) the same first second and the same rules in it; after that the
  // paths may split
  const good = obs0 < 2e-3 && sx0 < 1e-3 && drv0 < 2e-3 && drift1 < 0.05 && !rules && scDrift1 < 0.05 && !rcErr;
  ok &&= good;
  console.log(`${good ? "ok " : "BAD"} ${(snap.event ?? "").padEnd(12)} step ${String(snap.step).padStart(5)}: ${rules ? `rules differ (${rules}); ` : ""}senses off by ${obs0.toExponential(1)} (pit wall ${sx0.toExponential(1)}${ref.replay ? "" : `, driver ${drv0.toExponential(1)}`}); `
    + `drift ${(drift1 * 1000).toFixed(1)} mm in 1 s, ${(drift5 * 1000).toFixed(0)} mm in 5 s, paths split (1 cm) at ${split === null ? "never" : `${(split * 0.05).toFixed(2)} s`}; `
    + `over 10 s: senses off by up to ${obsErr.toExponential(1)} (${worstSense}), box calls ${callsOk ? "same" : "differ"}, stops ${stopsOk ? "same" : "differ"}`
    + (py ? `; race control senses ${rcErr ? `differ (${rcErr})` : "same"}` : "")
    + (race.c ? `; race control ${rc ? `differs first: ${rc}` : "same"}, delta off by up to ${deltaErr.toFixed(4)} s, Safety Car drift ${(scDrift1 * 1000).toFixed(1)} mm in 1 s, ${(scDrift * 1000).toFixed(0)} mm in 10 s` : ""));
});
}
for (const kind of ["driver", "strat"])   // which race control senses the snapshots ever set (the rest: never checked)
  console.log(`${kind === "driver" ? "driver" : "pit wall"} senses never non-zero in a snapshot: ${senses[kind].filter((_, q) => !seenNonZero[kind].has(q)).join(", ") || "none"}`);
console.log(ok ? "all snapshots match" : "MISMATCH");
process.exit(ok ? 0 : 1);
