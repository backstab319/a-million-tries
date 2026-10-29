// Checks field-engine.js (live race day) against the Python race: node field-check.mjs
// field-data.json holds the whole Python race state at a few moments of the final race (the start, the first AI pit
// wall decision, just before the first stop, a late hit) and every car's motion for 10 s after each. From each
// snapshot the JS race runs on its own (its own driver, pit wall, contact, ...) and is compared with Python: the
// senses both computed from the same state, and how far the cars drift apart. Like Python re-run with a different
// batch size, the paths split after a few seconds (floats: doubles here, float32 there; contact amplifies it).
import { readFileSync } from "node:fs";
import { decode } from "./engine.js";
import { Race, Strategist } from "./field-engine.js";

const data = JSON.parse(readFileSync(new URL("./field-data.json", import.meta.url)));
const ref = data.reference;

// a Python Field (export_web._tree) as field-engine.js's race state
function fromPython(snap, race) {
  const F = race.F, p = snap.field, tr = race.tr, inf = (x) => (x >= 1e29 ? Infinity : x);
  const cars = p.cars, c = cars.car;
  const field = {
    cars: cars.design.map((design, i) => ({
      car: { x: c.pos[i][0], y: c.pos[i][1], yaw: c.yaw[i], vx: c.vx[i], vy: c.vy[i], r: c.r[i], steer: c.steer[i], ax: c.ax[i],
             ay: c.ay[i], gear: c.gear[i], shift: c.shift[i], ride: c.ride[i], temp: c.temp[i], wear: c.wear[i], fuel_used: c.fuel_used[i] },
      design, p: F.game.car.derive(design), tr, idx: cars.idx[i], lat: cars.lat[i], lap_pos: cars.lap_pos[i],
      progress: cars.progress[i], t: cars.t[i], last: cars.last[i], surface: cars.surface[i], dirt: cars.dirt[i],
      sliding: cars.sliding[i], plank: cars.plank[i], crashed: cars.crashed[i], done: cars.done[i], info: null })),
    finished: p.finished, finish_t: p.finish_t.map(inf), flag: p.flag, flag_laps: p.flag_laps, bump: p.bump, hit: p.hit,
    damage: p.damage.wing_front.map((_, i) => ({ wing_front: p.damage.wing_front[i], wing_rear: p.damage.wing_rear[i],
      floor: p.damage.floor[i], toe: p.damage.toe[i], puncture: p.damage.puncture[i] })),
    wrecked: p.wrecked, setup: p.setup.compound.map((k, i) => ({ compound: k, fuel: p.setup.fuel[i], wear: p.setup.wear[i] })),
    plan: p.plan, box_call: p.box_call, service: p.service, served: p.served, stops: p.stops, t: p.t, done: p.done,
    burst_at: p.burst_at, flat_m: p.flat_m, out_s: p.out_s,
  };
  return { f: field, setFrom: snap.set_from, prev: snap.prev };
}

let ok = true;
for (const snap of ref.snaps) {
  const race = new Race(data, ref.track, { state: {} });
  Object.assign(race, fromPython(snap, race));
  const pos = decode(snap.pos), obs = decode(snap.obs), sx = decode(snap.strat_x), steps = pos.shape[0], n = race.F.n;
  let obsErr = 0, obs0 = 0, sx0 = 0, sxErr = 0, worstSense = "", split = null, drift1 = 0, drift5 = 0, callsOk = true, stopsOk = true;
  for (let k = 0; k < steps; k++) {
    if (k % 20 === 0) {   // the senses, from (nearly) the same state
      const x = Strategist.senses(race.F, race.f, race.setFrom, 0);
      for (let i = 0; i < n; i++) for (let q = 0; q < 25; q++) sxErr = Math.max(sxErr, Math.abs(x[i][q] - sx.data[((k / 20) * n + i) * 25 + q]));
      if (k === 0) sx0 = sxErr;
    }
    race.step();
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
  }
  // from the same state: the same senses and (nearly) the same first second; after that the paths may split
  const good = obs0 < 2e-3 && sx0 < 1e-3 && drift1 < 0.05;
  ok &&= good;
  console.log(`${good ? "ok " : "BAD"} step ${String(snap.step).padStart(5)}: senses off by ${obs0.toExponential(1)} (pit wall ${sx0.toExponential(1)}); `
    + `drift ${(drift1 * 1000).toFixed(1)} mm in 1 s, ${(drift5 * 1000).toFixed(0)} mm in 5 s, paths split (1 cm) at ${split === null ? "never" : `${(split * 0.05).toFixed(2)} s`}; `
    + `over 10 s: senses off by up to ${obsErr.toExponential(1)} (${worstSense}), box calls ${callsOk ? "same" : "differ"}, stops ${stopsOk ? "same" : "differ"}`);
}
console.log(ok ? "all snapshots match" : "MISMATCH");
process.exit(ok ? 0 : 1);
