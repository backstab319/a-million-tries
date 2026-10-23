// "Beat the engineer" (episode 14): the racing sandbox's car, circuit and driver, ported line by line from
// games/race_car.py, games/race.py and agents/ppo_car.py. The car class, rules, circuits and the driver's weights
// come from race-data.json (export_web.py race). Works in the browser and in Node (race-check.mjs compares it
// with the Python game). Doubles here, float32 in JAX: tiny differences, checked by race-check.mjs.

import { decode } from "./engine.js";

const mod = (a, n) => ((a % n) + n) % n;           // Python/JAX-style modulo
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const cross = (ax, ay, bx, by) => ax * by - ay * bx;
const sign = (x) => (x > 0 ? 1 : x < 0 ? -1 : 0);

// ── The car (race_car.py) ────────────────────────────────────────────────────

export class Car {
  constructor(data) {
    this.spec = data.spec;
    this.G = data.G; this.RHO = data.RHO; this.PETROL = data.PETROL_J_PER_KG;
    this.names = data.design.map((d) => d.name);
    this.low = data.design.map((d) => d.low);
    this.high = data.design.map((d) => d.high);
    this.D = Object.fromEntries(this.names.map((n, i) => [n, i]));
    this.budget = data.budget_k;
    this.prices = data.prices;
  }

  mid() { return this.low.map((lo, i) => (lo + this.high[i]) / 2); }

  normalise(d) { return d.map((x, i) => 2 * (x - this.low[i]) / (this.high[i] - this.low[i]) - 1); }

  awd(d) {  // 1 if the car needs four-wheel-drive hardware (any real split), easing in over the first 10%
    const s = d[this.D.drive_front];
    return clip(10 * Math.min(s, 1 - s), 0, 1);
  }

  fuelStart(d) {
    const sp = this.spec;
    const burn = 1000 * d[this.D.power_kw] / (sp.thermal_eff * this.PETROL);
    return sp.fuel_kg + burn * sp.race_s * sp.full_throttle_share;
  }

  parts(d) {  // every part's mass (kg) and place (metres behind the front axle)
    const sp = this.spec, D = this.D, L = d[D.wheelbase];
    const wheel = (mm) => 2 * (sp.wheel_kg + sp.wheel_kg_per_mm * (mm - 185));
    return [
      [sp.chassis_kg, sp.chassis_x * L],
      [sp.driver_kg, sp.driver_x * L],
      [this.fuelStart(d), sp.fuel_x * L],
      [sp.engine_kg_base + sp.engine_kg_per_kw * d[D.power_kw], L * (-0.05 + 1.15 * d[D.engine_pos])],
      [sp.wing_front_kg * d[D.wing_front], -sp.overhang_front],
      [sp.wing_rear_kg * d[D.wing_rear], L + sp.overhang_rear],
      [wheel(d[D.tyre_front_mm]), 0],
      [wheel(d[D.tyre_rear_mm]), L],
      [sp.awd_kg * this.awd(d), 0.5 * L],
    ];
  }

  mass(d) { return this.parts(d).reduce((s, [m]) => s + m, 0); }

  weightFront(d) {
    const ps = this.parts(d), m = this.mass(d);
    return 1 - ps.reduce((s, [pm, px]) => s + pm * px, 0) / m / d[this.D.wheelbase];
  }

  costBreakdown(d) {  // what each part costs ($k)
    const P = this.prices, D = this.D, [e0, e1, e2] = P.engine, [t0, t1] = P.tyres, kw = d[D.power_kw];
    return {
      chassis: P.chassis + 40 * (d[D.wheelbase] - 2.3),
      engine: e0 + e1 * kw + e2 * kw ** 2,
      wings: P.wing_front * d[D.wing_front] + P.wing_rear * d[D.wing_rear],
      tyres: 2 * t0 + t1 * ((d[D.tyre_front_mm] - 185) + (d[D.tyre_rear_mm] - 185)),
      awd: P.awd * this.awd(d),
    };
  }

  cost(d) { return Object.values(this.costBreakdown(d)).reduce((a, b) => a + b, 0); }

  derive(d) {  // design numbers -> physics constants
    const sp = this.spec, D = this.D;
    const ps = this.parts(d);
    const m = ps.reduce((s, [pm]) => s + pm, 0);
    const L = d[D.wheelbase];
    const xcg = ps.reduce((s, [pm, px]) => s + pm * px, 0) / m;
    const a = clip(xcg, 0.15 * L, 0.85 * L), b = L - a;
    const length = L + sp.overhang_front + sp.overhang_rear;
    const inertia = ps.reduce((s, [pm, px]) => s + pm * (px - xcg) ** 2, 0)
      + sp.chassis_kg * length ** 2 / 12
      + (ps[6][0] + ps[7][0]) * (sp.track_width / 2) ** 2;
    const top = d[D.top_gear_kmh] / 3.6;
    const gearTop = [];
    for (let i = 0; i < sp.gears; i++) gearTop.push(top * sp.gear_spread ** (i / (sp.gears - 1) - 1));
    const k = d[D.spring_n_mm];
    const stiff = clip((k - sp.spring_soft) / (sp.spring_stiff - sp.spring_soft), 0, 1);
    const rh = d[D.ride_height_mm] / 1000;
    const wf = d[D.wing_front], wr = d[D.wing_rear], tf = d[D.tyre_front_mm], tr = d[D.tyre_rear_mm];
    const awd = this.awd(d);
    return {
      mass: m, yaw_inertia: inertia, a, b, cog_height: sp.cog_base + rh, track_width: sp.track_width,
      cla_floor: sp.cla_floor, floor_front: sp.floor_front,
      cla_front: sp.cla_wing_front * wf, cla_rear: sp.cla_wing_rear * wr,
      ground_gain: sp.ground_gain, ground_ref: sp.ground_ref, ground_peak: sp.ground_peak, ground_stall: sp.ground_stall,
      cda: sp.cda_base + sp.cda_wing_front * wf + sp.cda_wing_rear * wr
        + sp.cda_per_tyre_mm * ((tf - 185) + (tr - 185)) / 2,
      ride_height: rh, spring: k * 1000, settle: 0.05 * Math.sqrt(100 / k), bump_loss: stiff, plank_mu: sp.plank_mu,
      brake_front: d[D.brake_front], roll_front: d[D.roll_front], drive_front: d[D.drive_front],
      mu: sp.mu, mu_long: sp.mu_long, load_sens: sp.load_sens,
      fz_nominal: [tf, tf, tr, tr].map((w) => sp.fz_nominal_245 * w / 245),
      power: (sp.drivetrain - sp.awd_loss * awd) * 1000 * d[D.power_kw], gear_top: gearTop,
      shift_s: sp.shift_s, diff_bias: sp.diff_bias, brake_force: sp.brake_g * this.G * m,
      pb: sp.pacejka_b, pc: sp.pacejka_c, pe: sp.pacejka_e, slide: sp.slide,
      max_steer: sp.max_steer, steer_rate: sp.steer_rate, roll_resist: sp.roll_resist,
      spin_wheels: sp.wheel_spin_share * (ps[6][0] + ps[7][0]),
      spin_engine: sp.engine_inertia * (sp.rev_limit_rpm * 2 * Math.PI / 60) ** 2,
      engine_brake: sp.engine_brake, yaw_down_loss: sp.yaw_down_loss, cda_side: sp.cda_side,
      tyre_size: [tf, tf, tr, tr].map((w) => w / 245), temp_best: sp.temp_best, temp_window: sp.temp_window,
      temp_loss: sp.temp_loss, temp_floor: sp.temp_floor, temp_air: sp.temp_air, tyre_heat_cap: sp.tyre_heat_cap,
      tyre_cool: sp.tyre_cool, tyre_cool_v: sp.tyre_cool_v, heat_share: sp.heat_share, wear_rate: sp.wear_rate,
      wear_hot: sp.wear_hot, wear_loss: sp.wear_loss, wear_cliff: sp.wear_cliff,
      fuel_per_j: 1 / (sp.thermal_eff * this.PETROL * sp.drivetrain),
    };
  }

  carAt(x, y, yaw, speed, rideHeight, tyreTemp) {
    return { x, y, yaw, vx: speed, vy: 0, r: 0, steer: 0, ax: 0, ay: 0, gear: 1, shift: 0,
             ride: [rideHeight, rideHeight], temp: [tyreTemp, tyreTemp, tyreTemp, tyreTemp], wear: [0, 0, 0, 0],
             fuel_used: 0 };
  }
}

function tempGrip(p, t) {
  const off = (t - p.temp_best) / p.temp_window;
  return clip(1 - p.temp_loss * off ** 2, p.temp_floor, 1);
}

function wearGrip(p, w) { return 1 - p.wear_loss * w - Math.max(w - p.wear_cliff, 0); }

function engineForce(p, vx, gear) {  // push at the wheels at full throttle and the gear for it (1-based)
  const v = Math.max(vx, 0);
  const perGear = p.gear_top.map((gt) => (v <= gt ? p.power / Math.max(v, 0.8 * gt) : 0));
  let g = 0;
  for (let i = 1; i < perGear.length; i++) if (perGear[i] > perGear[g]) g = i;
  if (perGear[gear - 1] >= 0.95 * perGear[g]) g = gear - 1;
  return [perGear[g], g + 1];
}

function magic(p, alpha) {
  const bx = p.pb * alpha;
  return Math.sin(p.pc * Math.atan(bx - p.pe * (bx - Math.atan(bx))));
}

function axleSplit(total, capL, capR, bias) {  // a limited-slip differential: (left, right)
  const weak = Math.min(capL, capR);
  const weakAsk = total / 2 <= weak ? total / 2 : total / (1 + bias);
  const left = capL <= capR ? weakAsk : total - weakAsk;
  return [left, total - left];
}

// One physics step. steerIn, pedal in [-1, 1]. surf: {grip, rolling, rough, lift} (4 each); slope {gx, gy, load}.
// Returns [new state, info].
function substep(G, RHO, p, s, steerIn, pedal, surf, slope, dt) {
  const want = clip(steerIn, -1, 1) * p.max_steer;
  const steer = s.steer + clip(want - s.steer, -p.steer_rate * dt, p.steer_rate * dt);
  const speed = Math.hypot(s.vx, s.vy);
  const L = p.a + p.b, tw = p.track_width, h = p.cog_height, m = p.mass - s.fuel_used;
  const q = 0.5 * RHO * speed ** 2;

  // downforce: wings, and a floor that works harder near the ground (stalls once it's on the ground)
  const floorH = (s.ride[0] + s.ride[1]) / 2;
  let ground = clip(1 + p.ground_gain * (p.ground_ref - floorH) / (p.ground_ref - p.ground_peak), 0, 1 + p.ground_gain);
  const near = clip(floorH / p.ground_peak, 0, 1);
  if (floorH < p.ground_peak) ground = (1 + p.ground_gain) * (p.ground_stall + (1 - p.ground_stall) * near ** 2);
  const beta = Math.atan2(Math.abs(s.vy), Math.max(Math.abs(s.vx), 5));
  const yawKeep = clip(1 - p.yaw_down_loss * beta ** 2, 0, 1);
  const floor = q * p.cla_floor * ground * yawKeep;
  const downF = floor * p.floor_front + q * p.cla_front * yawKeep;   // clean air: the game has no wakes
  const downR = floor * (1 - p.floor_front) + q * p.cla_rear * yawKeep;

  // axle loads
  const staticF = m * G * p.b / L, staticR = m * G * p.a / L;
  let front = staticF * slope.load + downF - m * s.ax * h / L;
  let rear = staticR * slope.load + downR + m * s.ax * h / L;

  // suspension: past the ride height the plank takes the rest
  const lift = [Math.max(surf.lift[0], surf.lift[1]), Math.max(surf.lift[2], surf.lift[3])];
  const squash = [(front - staticF) / (2 * p.spring), (rear - staticR) / (2 * p.spring)];
  const ride = [p.ride_height - squash[0] - lift[0], p.ride_height - squash[1] - lift[1]];
  const plank = [Math.max(-ride[0], 0) * 2 * p.spring, Math.max(-ride[1], 0) * 2 * p.spring];
  front -= plank[0]; rear -= plank[1];

  const lat = m * s.ay * h / tw;
  const load = [front / 2 - p.roll_front * lat, front / 2 + p.roll_front * lat,
                rear / 2 - (1 - p.roll_front) * lat, rear / 2 + (1 - p.roll_front) * lat].map((x) => Math.max(x, 0));

  const fmax = load.map((ld, i) => {
    const mu = p.mu * clip(1 - p.load_sens * (ld / p.fz_nominal[i] - 1), 0.3, 1 + p.load_sens);
    return mu * ld * surf.grip[i] * (1 - surf.rough[i] * p.bump_loss) * tempGrip(p, s.temp[i]) * wearGrip(p, s.wear[i]) + 1e-3;
  });

  const wx = [p.a, p.a, -p.b, -p.b], wy = [tw / 2, -tw / 2, tw / 2, -tw / 2], delta = [steer, steer, 0, 0];
  const c = delta.map(Math.cos), sn = delta.map(Math.sin);
  const u = [], w = [], alpha = [];
  for (let i = 0; i < 4; i++) {
    const ux = s.vx - s.r * wy[i], uy = s.vy + s.r * wx[i];
    u.push(ux * c[i] + uy * sn[i]);
    w.push(-ux * sn[i] + uy * c[i]);
    alpha.push(Math.atan2(w[i], Math.max(Math.abs(u[i]), 1)));
  }

  // drive or brake
  const [push0, best] = engineForce(p, s.vx, s.gear);
  const shifting = best !== s.gear;
  const shift = shifting ? p.shift_s : Math.max(s.shift - dt, 0);
  const push = shift > 0 ? 0 : push0;
  const throttle = Math.max(pedal, 0), brake = Math.max(-pedal, 0);
  const gt = p.gear_top[best - 1];
  const engaged = shift <= 0 && s.vx > 0.8 * p.gear_top[0];
  const revs = clip(s.vx / gt, 0, 1);
  const engineDrag = engaged ? p.engine_brake * p.power / gt * revs : 0;
  const total = throttle * push - (1 - throttle) * engineDrag;
  const cap = fmax.map((f) => p.mu_long * f);
  const [fl, fr] = axleSplit(total * p.drive_front, cap[0], cap[1], p.diff_bias);
  const [rl, rr] = axleSplit(total * (1 - p.drive_front), cap[2], cap[3], p.diff_bias);
  const driveF = [fl, fr, rl, rr];
  const bshare = [p.brake_front / 2, p.brake_front / 2, (1 - p.brake_front) / 2, (1 - p.brake_front) / 2];
  const stopCap = m * Math.max(s.vx, 0) / dt;

  const sliding = [], fx = [], fy = [], temp = [], wear = [], rollCoef = [];
  let fxTot = 0, fyTot = 0, moment = 0, rollSum = 0;
  for (let i = 0; i < 4; i++) {
    const braking = brake * Math.min(p.brake_force, stopCap) * bshare[i];
    const fxAsk = driveF[i] - braking * sign(u[i] + 1e-6);
    const fyPure = -fmax[i] * magic(p, alpha[i]);
    const fmaxX = p.mu_long * fmax[i];
    const ratio = fxAsk / fmaxX;
    const sl = Math.abs(ratio) > 1;
    const gripFy = fyPure * Math.sqrt(Math.max(1 - ratio ** 2, 0));
    const locked = sl && fxAsk < 0;
    const vmag = Math.hypot(u[i], w[i]) + 1e-3;
    let X, Y;
    if (!sl) { X = fxAsk; Y = gripFy; }
    else if (locked) { X = -p.slide * fmaxX * u[i] / vmag; Y = -p.slide * fmax[i] * w[i] / vmag; }
    else { X = p.slide * fmaxX; Y = fyPure * Math.sqrt(1 - p.slide ** 2); }
    sliding.push(sl); fx.push(X); fy.push(Y);
    const bfx = X * c[i] - Y * sn[i], bfy = X * sn[i] + Y * c[i];
    fxTot += bfx; fyTot += bfy; moment += wx[i] * bfy - wy[i] * bfx;
    const rc = p.roll_resist * (1 + speed * 3.6 / 160) + surf.rolling[i];
    rollCoef.push(rc); rollSum += rc * load[i];

    // tyre temperature and wear
    const spinning = sl && !locked;
    const au = Math.abs(u[i]);
    const slipX = locked ? au : spinning ? 0.2 * au + 2 : 0.03 * Math.abs(ratio) * au;
    const work = Math.abs(Y) * Math.abs(w[i]) + Math.abs(X) * slipX;
    const heat = p.heat_share * work + rc * load[i] * speed;
    const cool = p.tyre_cool * p.tyre_size[i] * (1 + speed / p.tyre_cool_v) * (s.temp[i] - p.temp_air);
    temp.push(s.temp[i] + (heat - cool) / (p.tyre_heat_cap * p.tyre_size[i]) * dt);
    const hot = 2 ** (Math.max(s.temp[i] - p.temp_best, 0) / p.wear_hot);
    wear.push(Math.min(s.wear[i] + p.wear_rate * work * hot / p.tyre_size[i] * dt, 1));
  }
  const cda = p.cda + p.cda_side * Math.sin(beta) ** 2;
  const resist = rollSum + q * cda + p.plank_mu * (plank[0] + plank[1]);
  fxTot -= resist * s.vx / (speed + 0.5);
  fyTot -= resist * s.vy / (speed + 0.5);

  // integrate (semi-implicit Euler)
  const mLong = m + p.spin_wheels + (engaged ? p.spin_engine / gt ** 2 : 0);
  let vx = s.vx + ((fxTot + m * slope.gx) / mLong + s.r * s.vy) * dt;
  let vy = s.vy + ((fyTot + m * slope.gy) / m - s.r * s.vx) * dt;
  let r = s.r + moment / p.yaw_inertia * dt;
  if (brake > 0 && s.vx >= 0 && vx < 0) vx = 0;

  const blend = clip((Math.abs(vx) - 3) / 3, 0, 1);
  vy = blend * vy + (1 - blend) * vx * Math.tan(steer) * p.b / L;
  r = blend * r + (1 - blend) * vx * Math.tan(steer) / L;

  const yaw = s.yaw + r * dt;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const x = s.x + (vx * cy - vy * sy) * dt, y = s.y + (vx * sy + vy * cy) * dt;

  const feltX = fxTot / mLong, feltY = fyTot / m;
  const k = dt / (dt + p.settle);
  const burned = p.fuel_per_j * Math.max(total, 0) * Math.max(s.vx, 0) * dt;
  const ns = { x, y, yaw, vx, vy, r, steer, ax: s.ax + (feltX - s.ax) * k, ay: s.ay + (feltY - s.ay) * k, gear: best,
               shift, ride, temp, wear, fuel_used: s.fuel_used + burned };
  return [ns, { load, sliding, gear: best, ax: feltX, ay: feltY, plank, downforce: downF + downR }];
}

// ── The circuit and the game (race.py) ───────────────────────────────────────

export class Circuit {
  constructor(c, cfg) {
    this.seed = c.seed;
    this.length = c.length;
    this.center = decode(c.center).data;     // P x 2
    for (const k of ["heading", "curvature", "z", "grade", "side", "vcurv"]) this[k] = decode(c[k]).data;
    const P = (this.P = this.heading.length), edge = cfg.width / 2 + cfg.grass;
    // walls: left (inside, counter-clockwise) then right, segments a -> b (as race.make_tracks)
    this.wallA = new Float64Array(4 * P);
    this.wallB = new Float64Array(4 * P);
    const wall = (i, side) => {
      const h = this.heading[i];
      return [this.center[2 * i] - Math.sin(h) * edge * side, this.center[2 * i + 1] + Math.cos(h) * edge * side];
    };
    for (let i = 0; i < P; i++) {
      const j = (i + 1) % P;
      for (const [o, side] of [[0, 1], [P, -1]]) {
        const a = wall(i, side), b = wall(j, side);
        this.wallA[2 * (o + i)] = a[0]; this.wallA[2 * (o + i) + 1] = a[1];
        this.wallB[2 * (o + i)] = b[0]; this.wallB[2 * (o + i) + 1] = b[1];
      }
    }
  }
}

export class RaceGame {
  constructor(data) {
    this.cfg = data.cfg;
    this.car = new Car(data);
    this.circuits = data.circuits.map((c) => new Circuit(c, data.cfg));
    this.rays = data.cfg.ray_angles.map((a) => a * Math.PI / 180);
  }

  circuit(seed) { return this.circuits.find((c) => c.seed === seed); }

  // a car on the start line, standing, facing along the circuit (race.RaceGame.init with random_start=False)
  init(design, circuit) {
    const tr = typeof circuit === "number" ? this.circuit(circuit) : circuit;
    const p = this.car.derive(design);
    const car = this.car.carAt(tr.center[0], tr.center[1], tr.heading[0], 0,
                               design[this.car.D.ride_height_mm] / 1000, this.cfg.tyre_start);
    return { car, design, p, tr, idx: 0, lat: 0, lap_pos: 0, progress: 0, t: 0, last: [0, 0],
             surface: [1, 1, 1, 1], dirt: [0, 0, 0, 0], sliding: [false, false, false, false], plank: [0, 0],
             crashed: false, done: false, info: null };
  }

  locate(tr, idx, x, y) {  // nearest centreline point, signed offset, metres along the lap
    const P = tr.P, c = tr.center;
    let j = 0, bestD = Infinity;
    for (let o = -6; o < 12; o++) {
      const i = mod(idx + o, P), d = (c[2 * i] - x) ** 2 + (c[2 * i + 1] - y) ** 2;
      if (d < bestD) { bestD = d; j = i; }
    }
    const onSeg = (i0) => {
      const ia = mod(i0, P), ib = mod(i0 + 1, P);
      const ax = c[2 * ia], ay = c[2 * ia + 1], abx = c[2 * ib] - ax, aby = c[2 * ib + 1] - ay;
      const u = clip(((x - ax) * abx + (y - ay) * aby) / (abx * abx + aby * aby), 0, 1);
      const d = Math.hypot(ax + u * abx - x, ay + u * aby - y);
      return [d, sign(cross(abx, aby, x - ax, y - ay)) * d, ia + u];
    };
    const [d0, s0, f0] = onSeg(j - 1), [d1, s1, f1] = onSeg(j);
    const lat = d0 < d1 ? s0 : s1, frac = d0 < d1 ? f0 : f1;
    return [j, lat, mod(frac * tr.length / P, tr.length)];
  }

  surfaces(car, p, tr, idx, lat) {  // what's under each tyre, and how far out the furthest tyre is
    const cfg = this.cfg, na = tr.heading[idx] + Math.PI / 2, nx = Math.cos(na), ny = Math.sin(na);
    const wx = [p.a, p.a, -p.b, -p.b], wy = [p.track_width / 2, -p.track_width / 2, p.track_width / 2, -p.track_width / 2];
    const cy = Math.cos(car.yaw), sy = Math.sin(car.yaw), half = cfg.width / 2;
    const surf = { grip: [], rolling: [], rough: [], lift: [] };
    let furthest = 0;
    for (let i = 0; i < 4; i++) {
      const ox = wx[i] * cy - wy[i] * sy, oy = wx[i] * sy + wy[i] * cy;
      const wl = Math.abs(lat + ox * nx + oy * ny);
      furthest = Math.max(furthest, wl);
      const s = wl > half ? cfg.grass_surface : wl > half - cfg.kerb ? cfg.kerb_surface : cfg.tarmac_surface;
      surf.grip.push(s[0]); surf.rolling.push(s[1]); surf.rough.push(s[2]); surf.lift.push(s[3]);
    }
    return [surf, furthest];
  }

  slope(st) {  // gravity along the road (car frame) and the load factor
    const tr = st.tr, i = st.idx, G = this.car.G;
    const h = tr.heading[i], yaw = st.car.yaw, grade = tr.grade[i], side = tr.side[i];
    const upx = grade * Math.cos(h) + side * -Math.sin(h), upy = grade * Math.sin(h) + side * Math.cos(h);
    const tilt = Math.sqrt(1 + grade ** 2 + side ** 2);
    const gx = -G * upx / tilt, gy = -G * upy / tilt;
    const v2 = st.car.vx ** 2 + st.car.vy ** 2;
    const load = 1 / tilt + v2 * tr.vcurv[i] / G - v2 * tr.curvature[i] * side / Math.sqrt(1 + side ** 2) / G;
    return { gx: gx * Math.cos(yaw) + gy * Math.sin(yaw), gy: -gx * Math.sin(yaw) + gy * Math.cos(yaw),
             load: Math.max(load, 0) };
  }

  step(st, steerIn, pedalIn) {
    const cfg = this.cfg, steer = clip(steerIn, -1, 1), pedal = clip(pedalIn, -1, 1), p = st.p, tr = st.tr;
    const [surf] = this.surfaces(st.car, p, tr, st.idx, st.lat);
    surf.grip = surf.grip.map((g, i) => g * (1 - cfg.dirt_grip * st.dirt[i]));
    const slope = this.slope(st), dt = cfg.dt / cfg.substeps;
    let car = st.car, info;
    for (let k = 0; k < cfg.substeps; k++)
      [car, info] = substep(this.car.G, this.car.RHO, p, car, steer, pedal, surf, slope, dt);
    const [idx, lat, lapPos] = this.locate(tr, st.idx, car.x, car.y);
    const [surfNow, furthest] = this.surfaces(car, p, tr, idx, lat);
    const L = tr.length;
    const moved = mod(lapPos - st.lap_pos + L / 2, L) - L / 2;
    const crashed = furthest > cfg.width / 2 + cfg.grass - 0.3;
    const t = st.t + 1;
    const done = crashed || t * cfg.dt >= cfg.time_limit;
    const dirt = surfNow.grip.map((g, i) => (g <= cfg.grass_surface[0] ? 1 : Math.max(st.dirt[i] - cfg.dt / cfg.dirt_clear, 0)));
    return { car, design: st.design, p, tr, idx, lat, lap_pos: lapPos, progress: st.progress + moved, t,
             last: [steer, pedal], surface: surfNow.grip, dirt, sliding: info.sliding, plank: info.plank,
             crashed, done, info };
  }

  lasers(st) {  // metres to the nearest wall along each laser, capped at ray_range
    const tr = st.tr, P = tr.P, [lo, hi] = this.cfg.wall_window, out = [];
    const px = st.car.x, py = st.car.y;
    for (const ray of this.rays) {
      const dx = Math.cos(st.car.yaw + ray), dy = Math.sin(st.car.yaw + ray);
      let best = Infinity;
      for (let o = lo; o < hi; o++) {
        const n = mod(st.idx + o, P);
        for (const s of [n, n + P]) {
          const ax = tr.wallA[2 * s], ay = tr.wallA[2 * s + 1];
          const ex = tr.wallB[2 * s] - ax, ey = tr.wallB[2 * s + 1] - ay, wx = ax - px, wy = ay - py;
          const den = cross(dx, dy, ex, ey);
          if (Math.abs(den) < 1e-9) continue;
          const t = cross(wx, wy, ex, ey) / den, u = cross(wx, wy, dx, dy) / den;
          if (t > 0 && u >= 0 && u <= 1 && t < best) best = t;
        }
      }
      out.push(Math.min(best, this.cfg.ray_range));
    }
    return out;
  }

  ahead(st) {  // centreline points at cfg.ahead metres ahead
    const tr = st.tr, P = tr.P;
    return this.cfg.ahead.map((a) => mod(Math.trunc((st.lap_pos + a) / tr.length * P), P));
  }

  observe(st) {
    const cfg = this.cfg, car = st.car, tr = st.tr, sp = this.car.spec;
    const ks = this.ahead(st);
    const headingErr = mod(car.yaw - tr.heading[st.idx] + Math.PI, 2 * Math.PI) - Math.PI;
    return [
      ...this.lasers(st).map((d) => d / cfg.ray_range),
      ...ks.map((k) => clip(tr.curvature[k] * 30, -2, 2)),
      ...ks.map((k) => tr.grade[k] * 10), ...ks.map((k) => tr.side[k] * 10),
      car.vx / 100, car.vy / 10, car.r / 2, car.ax / 50, car.ay / 50, st.lat / (cfg.width / 2), headingErr,
      car.steer / sp.max_steer, st.last[1],
      ...st.surface.map((g, i) => g * (1 - cfg.dirt_grip * st.dirt[i])),
      ...st.sliding.map(Number),
      ...car.temp.map((t) => (t - sp.temp_best) / sp.temp_window),
      ...car.wear,
      1 - car.fuel_used / this.car.fuelStart(st.design),
      ...this.car.normalise(st.design),
    ];
  }
}

// ── The driver (ppo_car.forward: two tanh layers, the policy's mean) ─────────

export class Driver {
  constructor(brain) {
    this.L = ["body.0", "body.2", "policy"].map((n) => {
      const w = decode(brain[n + ".weight"]);
      return { w: w.data, out: w.shape[0], inp: w.shape[1], b: decode(brain[n + ".bias"]).data };
    });
  }

  act(obs) {
    let x = obs;
    this.L.forEach((l, li) => {
      const y = new Array(l.out);
      for (let o = 0; o < l.out; o++) {
        let s = l.b[o];
        const row = o * l.inp;
        for (let i = 0; i < l.inp; i++) s += l.w[row + i] * x[i];
        y[o] = li < 2 ? Math.tanh(s) : clip(s, -1, 1);
      }
      x = y;
    });
    return x;  // [steer, pedal]
  }
}
