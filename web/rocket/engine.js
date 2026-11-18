// The rocket booster from 70 km (episode 18), one booster, ported line by line from src/ailearns/games/rocket.py
// (Boosters.step / obs / reset) for the world "entry": the 1976 US Standard Atmosphere, drag through the sound
// barrier, an entry burn on three engines then a landing burn on one (with the half-second start sequence),
// heating, air load and the bending limit, body lift, grid fins, calm sea (no wind, no heaving deck, no faults).
// Same order of operations as the Python, in float64, so check.mjs can replay Python flights step for step.
// The pilot (the AI): the network from runs/rocket-ppo29 u1400 (data/pilot.json), mean action, as in the video.

export const FLYING = 0, LANDED = 1, CRASHED = 2, TIMEOUT = 3;
const G0 = 9.80665;
const SUTTON_GRAVES = 1.7415e-4;

// (11a) 1976 US Standard Atmosphere: height (m), density (kg/m^3), pressure (Pa), temperature (K)
const US76 = [
  [0, 1.225, 101325, 288.15], [5e3, 0.7364, 54048, 255.65], [10e3, 0.4135, 26500, 223.15],
  [11e3, 0.3639, 22632, 216.65], [15e3, 0.1948, 12111, 216.65], [20e3, 0.08891, 5529, 216.65],
  [25e3, 0.04008, 2549, 221.65], [30e3, 0.01841, 1197, 226.65], [32e3, 0.01322, 868.0, 228.65],
  [40e3, 0.003996, 287.1, 251.05], [47e3, 0.001426, 110.9, 270.65], [50e3, 0.001027, 79.78, 270.65],
  [51e3, 0.0008598, 66.94, 270.65], [60e3, 0.0003097, 21.96, 247.02], [70e3, 8.283e-5, 5.221, 219.59],
  [71e3, 6.420e-5, 3.956, 214.65], [80e3, 1.846e-5, 1.052, 196.65], [90e3, 3.40e-6, 0.1836, 186.87]];
const US_H = US76.map(r => r[0]);
const US_LRHO = US76.map(r => Math.log(r[1]));
const US_LP = US76.map(r => Math.log(r[2]));
const US_T = US76.map(r => r[3]);

// numpy.interp: linear, clamped to the end values outside the table
export function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  const n = xs.length;
  if (x >= xs[n - 1]) return ys[n - 1];
  let i = 1;
  while (xs[i] < x) i++;
  const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

export function us76(y) {  // -> [density, pressure / sea level, speed of sound]
  y = Math.min(Math.max(y, 0.0), US_H[US_H.length - 1]);
  const rho = Math.exp(interp(y, US_H, US_LRHO));
  const p = Math.exp(interp(y, US_H, US_LP)) / US76[0][2];
  return [rho, p, Math.sqrt(1.4 * 287.05 * interp(y, US_H, US_T))];
}

const RAD = Math.PI / 180;
// world("entry") = ship + fins + ENTRY (games/rocket.py); data/config.json is exported from Python and wins
export const ENTRY = {
  dry_mass: 22200, length: 42.6, diameter: 3.7, cg_height: 15.0, leg_span: 18.0,
  thrust_max: 845200, thrust_min: 482000, isp: 282.0, ignitions: 2, thrust_lag: 0.3,
  gimbal_max: 5 * RAD, gimbal_rate: 20 * RAD, rcs_force: 1000, gravity: 9.81,
  cd_axial: 0.8, cd_side: 1.2, fin_area: 3.6, fin_height: 40.0, fin_cn: 2.5, fin_max: 20 * RAD,
  fin_rate: 40 * RAD, fin_stall: 25 * RAD, drag_slices: 8,
  start_height: [68e3, 72e3], start_fall: [1600, 1900], start_x: [-29e3, -25e3], start_vx: [-3, 3],
  start_tilt: [-3 * RAD, 3 * RAD], start_spin: [-0.5 * RAD, 0.5 * RAD], start_fuel: [19e3, 21e3],
  start_fpa: [55 * RAD, 65 * RAD],
  land_vy: 3.0, land_vx: 1.0, land_tilt: 8 * RAD, land_spin: 5 * RAD, pad_radius: 17.0, dt: 0.05, max_time: 240,
  cd_mach: [[0.0, 1.0], [0.8, 1.0], [1.05, 1.4], [1.3, 1.5], [2.0, 1.3], [3.0, 1.15], [5.0, 1.05], [8.0, 1.0]],
  thrust_vac: 914e3, light_engines: [3, 1], heat_limit: 1.5e6, q_limit: 150e3, bend_limit: 800e3,
  heat_radius: 1.85, start_seq: 0.5, switch_band: 0.5, body_lift: true,
};

export class Booster {
  constructor(cfg = ENTRY) {
    this.c = cfg;
    this.machX = cfg.cd_mach.map(r => r[0]);
    this.machF = cfg.cd_mach.map(r => r[1]);
  }

  // the start of a flight: either Python's exact numbers (check.mjs) or drawn like Boosters.reset (the game)
  setStart(s) {
    Object.assign(this, {x: s.x, y: s.y, vx: s.vx, vy: s.vy, tilt: s.tilt, spin: s.spin, fuel: s.fuel});
    this.thrust = 0; this.gimbal = 0; this.fin = 0; this.heat = 0; this.hold = 0; this.peak_q = 0; this.bend = 0;
    this.time = 0; this.lit = false; this.ignitions = this.c.ignitions; this.outcome = FLYING;
    return this;
  }

  static draw(cfg, rand) {  // the same ranges as Boosters.reset, our own random numbers
    const u = ([a, b]) => a + (b - a) * rand();
    const s = {x: u(cfg.start_x), y: u(cfg.start_height), vx: u(cfg.start_vx), vy: -u(cfg.start_fall),
               tilt: u(cfg.start_tilt), spin: u(cfg.start_spin), fuel: u(cfg.start_fuel)};
    const path = u(cfg.start_fpa), speed = -s.vy;  // (11b) coming in along a slanted path, engine first
    s.vx = speed * Math.cos(path); s.vy = -speed * Math.sin(path);
    s.tilt = s.tilt - Math.atan2(s.vx, -s.vy);
    return s;
  }

  get mass() { return this.c.dry_mass + this.fuel; }
  get inertia() { const c = this.c; return this.mass * (c.length ** 2 / 12 + (c.length / 2 - c.cg_height) ** 2); }
  get exhaust() { return this.c.isp * G0; }
  engines() {  // how many engines the current (or last) light uses: entry burn 3, landing burn 1
    const c = this.c, k = Math.min(Math.max(c.ignitions - this.ignitions - 1, 0), c.light_engines.length - 1);
    return c.light_engines[k];
  }
  lowestFoot() { return this.y - this.c.leg_span / 2 * Math.abs(Math.sin(this.tilt)); }
  legWear() { const c = this.c; return 0.5 * ((this.vy / c.land_vy) ** 2 + (this.vx / c.land_vx) ** 2); }

  obs() {
    const c = this.c, [, , sound] = us76(this.y), sq = v => Math.sqrt(v);
    const mach = Math.hypot(this.vx, this.vy) / sound;
    const sgn = v => (v > 0) - (v < 0);
    return [
      this.x / 500.0, this.y / 2500.0, this.vx / 50.0, this.vy / 300.0,
      Math.sin(this.tilt), Math.cos(this.tilt), this.spin,
      this.fuel / c.start_fuel[1], this.thrust / c.thrust_max, this.gimbal / c.gimbal_max,
      this.lit ? 1 : 0, this.ignitions,
      sgn(this.x) * sq(Math.abs(this.x) / 500.0), sq(Math.max(this.y, 0.0) / 2500.0),
      sgn(this.vx) * sq(Math.abs(this.vx) / 50.0), sgn(this.vy) * sq(Math.abs(this.vy) / 300.0),
      0.0, 0.0,  // calm: no wind, no heaving deck
      this.fin / c.fin_max,
      this.y / 70e3, this.vy / 2000.0, this.vx / 1000.0, this.x / 30e3, mach / 6.0,
      this.peak_q / c.q_limit, this.heat / c.heat_limit, this.hold / Math.max(c.start_seq, 1e-9),
    ];
  }

  // one dt; action [engine, gimbal, thrusters, fins] each in -1..1. Returns the outcome.
  step(action) {
    const c = this.c, dt = c.dt;
    if (this.outcome !== FLYING) return this.outcome;
    const a = [0, 1, 2, 3].map(i => Math.min(Math.max(action[i] ?? 0, -1), 1));

    // engine: with a light in hand a command below -band shuts it, above +band lights it, held through the
    // start sequence (the pumps spin up while asked, back down while not)
    let shut = this.lit && this.ignitions > 0 && a[0] < -c.switch_band;
    let light = a[0] > c.switch_band && !this.lit && this.ignitions > 0 && this.fuel > 0;
    if (c.start_seq > 0) {
      this.hold = Math.min(Math.max(this.hold + ((shut || light) ? dt : -dt), 0.0), c.start_seq);
      const done = this.hold >= c.start_seq - 1e-9;
      shut = shut && done; light = light && done;
      if (shut || light) this.hold = 0.0;
    }
    if (shut) this.lit = false;
    if (light) this.ignitions -= 1;
    this.lit = (this.lit || light) && this.fuel > 0;
    let target = this.lit ? c.thrust_min + (c.thrust_max - c.thrust_min) * Math.max(a[0], 0.0) : 0.0;
    target *= this.engines();
    const lag = c.thrust_lag <= 0 ? 1.0 : 1.0 - Math.exp(-dt / c.thrust_lag);
    this.thrust = this.thrust + (target - this.thrust) * lag;
    const stepG = c.gimbal_rate * dt;
    this.gimbal = this.gimbal + Math.min(Math.max(a[1] * c.gimbal_max - this.gimbal, -stepG), stepG);
    const stepF = c.fin_rate * dt;
    this.fin = this.fin + Math.min(Math.max(a[3] * c.fin_max - this.fin, -stepF), stepF);

    // fuel burnt this step; the rocket equation over the step; thin air pushes harder
    const m0 = this.mass;
    const burn = Math.min(this.thrust / this.exhaust * dt, this.fuel);
    this.fuel = this.fuel - burn;
    const boost = c.thrust_vac / c.thrust_max - (c.thrust_vac / c.thrust_max - 1.0) * us76(this.y)[1];
    const dvThrust = this.exhaust * boost * Math.log(m0 / this.mass);
    const thrustUsed = burn > 0 ? this.thrust * boost : 0.0;
    if (this.fuel <= 0) { this.lit = false; this.thrust = 0.0; }

    const st = Math.sin(this.tilt), ct = Math.cos(this.tilt);
    const dirx = Math.sin(this.tilt - this.gimbal), diry = Math.cos(this.tilt - this.gimbal);

    // drag along and across the body (slice by slice, turning included), Mach, heating, air load, bending
    const [rho, , sound] = us76(this.y);
    const rvx = this.vx;  // calm
    const vAx = rvx * st + this.vy * ct;
    const vSide = rvx * ct - this.vy * st;
    const areaAx = Math.PI * (c.diameter / 2) ** 2;
    const speed = Math.hypot(rvx, this.vy);
    const cdAx = c.cd_axial * interp(speed / sound, this.machX, this.machF);
    let fAx = -0.5 * rho * cdAx * areaAx * Math.abs(vAx) * vAx;
    const q = 0.5 * rho * speed ** 2;
    this.heat = this.heat + SUTTON_GRAVES * Math.sqrt(rho / c.heat_radius) * speed ** 3 * dt;
    this.peak_q = Math.max(this.peak_q, q);
    this.bend = q * Math.atan2(Math.abs(vSide), Math.abs(vAx)) * 180 / Math.PI;
    const k = c.drag_slices;
    let fSide = 0, aeroTorque = 0;
    for (let j = 0; j < k; j++) {
      const arm = (j + 0.5) * c.length / k - c.cg_height;
      const vSlice = vSide + this.spin * arm;
      const fSlice = -0.5 * rho * c.cd_side * (c.length / k * c.diameter) * Math.abs(vSlice) * vSlice;
      fSide += fSlice; aeroTorque += fSlice * arm;
    }
    if (c.body_lift) {  // slender-body lift at the leading end
      const lift = -rho * areaAx * vSide * Math.abs(vAx);
      const lead = vAx < 0 ? -c.cg_height : c.length - c.cg_height;
      fSide = fSide + lift;
      aeroTorque = aeroTorque + lift * lead;
    }
    // grid fins: a plate at angle `fin` high up; force normal to it, up to the stall
    const armF = c.fin_height - c.cg_height;
    const vFin = vSide + this.spin * armF;
    const sf = Math.sin(this.fin), cf = Math.cos(this.fin);
    const wN = vAx * sf - vFin * cf, wT = -vAx * cf - vFin * sf;
    const aoa = Math.atan2(wN, Math.abs(wT));
    const fFin = 0.5 * rho * (vAx ** 2 + vFin ** 2) * c.fin_area * c.fin_cn
      * Math.min(Math.max(aoa, -c.fin_stall), c.fin_stall);
    fAx = fAx - fFin * sf;
    fSide = fSide + fFin * cf;
    aeroTorque = aeroTorque + fFin * cf * armF;
    const m = m0;
    const ax = (fAx * st + fSide * ct) / m;
    const ay = (fAx * ct - fSide * st) / m - c.gravity;

    // turning: gimbal, aero, thrusters at the top
    const torque = c.cg_height * thrustUsed * Math.sin(this.gimbal) + aeroTorque
      + a[2] * c.rcs_force * (c.length - c.cg_height);
    const spinNew = this.spin + torque / this.inertia * dt;  // inertia with the fuel after this step's burn
    const vxNew = this.vx + ax * dt + dvThrust * dirx;
    const vyNew = this.vy + ay * dt + dvThrust * diry;
    this.x = this.x + 0.5 * (this.vx + vxNew) * dt;
    this.y = this.y + 0.5 * (this.vy + vyNew) * dt;
    this.vx = vxNew; this.vy = vyNew;
    this.tilt = this.tilt + 0.5 * (this.spin + spinNew) * dt;
    this.spin = spinNew;
    this.time = this.time + dt;

    // touchdown, breakup, timeout
    const down = this.lowestFoot() <= 0.0;
    if (down) {
      const soft = Math.abs(this.vy) <= c.land_vy && Math.abs(this.vx) <= c.land_vx
        && Math.abs(this.tilt) <= c.land_tilt && Math.abs(this.spin) <= c.land_spin && Math.abs(this.x) <= c.pad_radius;
      this.outcome = soft ? LANDED : CRASHED;
      this.why = soft ? "landed" : this.whyCrash();
    } else if (this.heat > c.heat_limit) { this.outcome = CRASHED; this.why = "burnt up: too much heat on the engine end"; }
    else if (this.peak_q > c.q_limit) { this.outcome = CRASHED; this.why = "torn apart: air load past 150 kPa"; }
    else if (this.bend > c.bend_limit) { this.outcome = CRASHED; this.why = "buckled: swung across the airflow too hard"; }
    if (!down && this.time >= c.max_time) { this.outcome = TIMEOUT; this.why = "out of time"; }  // written last, as in Python
    if (this.outcome !== FLYING) { this.lit = false; this.thrust = 0.0; }
    return this.outcome;
  }

  whyCrash() {
    const c = this.c, bits = [];
    if (Math.abs(this.vy) > c.land_vy) bits.push(`hit at ${Math.abs(this.vy).toFixed(1)} m/s (limit ${c.land_vy})`);
    if (Math.abs(this.vx) > c.land_vx) bits.push(`sliding ${Math.abs(this.vx).toFixed(1)} m/s sideways (limit ${c.land_vx})`);
    if (Math.abs(this.tilt) > c.land_tilt) bits.push(`leaning ${(Math.abs(this.tilt) / RAD).toFixed(0)}° (limit 8)`);
    if (Math.abs(this.spin) > c.land_spin) bits.push(`turning ${(Math.abs(this.spin) / RAD).toFixed(0)}°/s`);
    if (Math.abs(this.x) > c.pad_radius) bits.push(`missed the deck by ${(Math.abs(this.x) - c.pad_radius).toFixed(0)} m`);
    return bits.join(", ") || "crashed";
  }
}

// the AI: tanh MLP, mean action (pi weights from data/pilot.json: {"pi0.w": [[...]], ...}, rows = inputs)
export class Pilot {
  constructor(w) { this.w = w; this.layers = Object.keys(w).filter(k => /^pi\d+\.w$/.test(k)).length; }
  act(obs) {
    let x = obs;
    const lin = (x, W, b) => b.map((bj, j) => { let s = bj; for (let i = 0; i < x.length; i++) s += x[i] * W[i][j]; return s; });
    for (let l = 0; l < this.layers; l++) x = lin(x, this.w[`pi${l}.w`], this.w[`pi${l}.b`]).map(Math.tanh);
    return lin(x, this.w["pi.out.w"], this.w["pi.out.b"]).map(v => Math.min(Math.max(v, -1), 1));
  }
}
