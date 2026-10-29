// Race day, live (episode 15): eight cars racing on one circuit, run in the browser. Ported line by line from
// games/race_field.py (the field: grid, contact, damage, slipstream, pit stops, tyre failures, marshals, senses,
// order), agents/es_strategist.py (the AI pit wall: its senses and decision) and scripts/race_full.py (the race:
// the rule's calls, the AI pit wall's once-a-lap call, the driver in every car). The single car, the circuit and the
// driver come from race-engine.js. Data: field-data.json (export_web.py field); field-check.mjs checks this against
// the Python race. Doubles here, float32 in JAX: tiny differences that grow (even Python can't replay a race
// exactly when it's batched differently), so the check is short stretches from the same state.

import { RaceGame, Driver, CLEAN_AIR } from "./race-engine.js";
import { decode } from "./engine.js";

const mod = (a, n) => ((a % n) + n) % n;
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const cross = (ax, ay, bx, by) => ax * by - ay * bx;
const RAD = Math.PI / 180;
export const PARTS = ["wing_front", "wing_rear", "floor", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"];
const TYRE_SETS = 4;   // sets of tyres a car's failure points are drawn for (a set per stop; the last repeats)

export class RaceField {
  constructor(data) {
    this.data = data;
    this.game = new RaceGame(data);
    this.cfg = data.cfg;
    Object.assign(this, data.field);   // n, laps, grid_gap, ... (RaceField's settings)
    this.dmg = data.field.damage;
    this.wk = data.field.wake;
    this.rays = this.cfg.ray_angles.length;
  }

  // ── the grid (RaceField.grid / place) ──
  fuelAllowance(tr, design) {
    const sp = this.game.car.spec, raceS = this.laps * tr.length / this.fuel_speed;
    const burn = 1000 * design[this.game.car.D.power_kw] / (sp.thermal_eff * this.game.car.PETROL);
    return sp.fuel_kg + this.fuel_share * burn * sp.full_throttle_share * raceS;
  }

  place(tr, design, metres, side) {
    const P = tr.P, L = tr.length, s = mod(metres, L) / L * P, i = Math.floor(s) % P, u = s - Math.floor(s);
    const yaw = tr.heading[i], j = (i + 1) % P, c = tr.center;
    const x = c[2 * i] + u * (c[2 * j] - c[2 * i]) - Math.sin(yaw) * side;
    const y = c[2 * i + 1] + u * (c[2 * j + 1] - c[2 * i + 1]) + Math.cos(yaw) * side;
    const st = this.game.init(design, tr);
    st.car = this.game.car.carAt(x, y, yaw, 0, design[this.game.car.D.ride_height_mm] / 1000, this.cfg.tyre_start);
    const [idx, lat, lapPos] = this.game.locate(tr, i, x, y);
    return Object.assign(st, { idx, lat, lap_pos: lapPos, progress: lapPos });
  }

  // a new race on circuit `tr`: designs (n x 13), compounds (n), rand () -> [0, 1) for the tyres' failure points
  grid(tr, designs, compounds, rand = Math.random) {
    const n = this.n, L = tr.length;
    const cars = designs.map((d, k) => {
      const st = this.place(tr, d, L - (this.pole_back + this.grid_gap * k), k % 2 === 0 ? this.grid_side : -this.grid_side);
      st.progress = st.lap_pos - L;
      return st;
    });
    const wear = this.gp_km ? this.tyre_cal * this.gp_km * 1000 / (this.laps * L) : 1;
    const [lo, hi] = this.burst_wear;
    return {
      cars, finished: Array(n).fill(false), finish_t: Array(n).fill(Infinity), flag: false, flag_laps: Array(n).fill(0),
      bump: Array(n).fill(0), hit: designs.map(() => Array(PARTS.length).fill(0)),
      damage: designs.map(() => ({ wing_front: 0, wing_rear: 0, floor: 0, toe: 0, puncture: [false, false, false, false] })),
      wrecked: Array(n).fill(false),
      setup: designs.map((d, k) => ({ compound: compounds[k], fuel: this.fuelAllowance(tr, d), wear })),
      plan: compounds.slice(), box_call: Array(n).fill(false), service: Array(n).fill(0), served: Array(n).fill(false),
      stops: Array(n).fill(0), t: 0, done: false,
      burst_at: designs.map(() => Array.from({ length: TYRE_SETS }, () => [0, 0, 0, 0].map(() => lo + rand() * (hi - lo)))),
      flat_m: designs.map(() => [0, 0, 0, 0]), out_s: Array(n).fill(0),
    };
  }

  // ── one decision for every car (RaceField.step) ──
  dryOut(f, i) { return f.cars[i].car.fuel_used >= f.setup[i].fuel && !f.finished[i]; }

  gone(f, i) { return (this.recover_s > 0 && f.out_s[i] >= this.recover_s) || this.dryOut(f, i); }

  step(f, actions) {
    const n = this.n, cfg = this.cfg, g = this.game;
    const moving = f.cars.map((c, i) => !(c.crashed || f.finished[i] || this.dryOut(f, i)));
    const racing = moving.map((m, i) => m && !f.wrecked[i]);
    const air = this.air(f, moving);
    let cars = f.cars.map((c, i) => {
      if (!moving[i]) return c;
      const held = f.wrecked[i] || f.service[i] > 0;   // a wreck locks its brakes; a car in its box is held
      const a = held ? actions[i].map((_, k) => (k === 1 ? -1 : 0)) : actions[i];
      return g.stepRace(c, a, f.damage[i], air[i], f.setup[i]);
    });
    let bump = Array(n).fill(0), hit = cars.map(() => Array(PARTS.length).fill(0)), damage = f.damage;
    let wrecked = f.wrecked.slice();
    const solid = cars.map((_, i) => !(f.finished[i] || this.gone(f, i)));
    [cars, bump, hit] = this.collide(cars, solid, moving, f.setup);
    let wreckedNow;
    [damage, wreckedNow] = this.breakParts(f.damage, hit);
    wrecked = wrecked.map((w, i) => w || (wreckedNow[i] && racing[i]));
    const parked = wrecked.map((w, i) => w && Math.abs(cars[i].car.vx) < 0.5);
    const pit = this.pit(f, cars, damage, racing.map((r, i) => r && !wrecked[i]));
    cars = pit.cars; damage = pit.damage;
    const flat_m = [], out_s = [];
    for (let i = 0; i < n; i++) {
      const d = damage[i], limit = f.burst_at[i][Math.min(pit.stops[i], TYRE_SETS - 1)];
      d.puncture = d.puncture.map((p, k) => p || (cars[i].car.wear[k] >= limit[k] && racing[i]));
      const moved = Math.hypot(cars[i].car.vx, cars[i].car.vy) * cfg.dt;
      flat_m.push(d.puncture.map((p, k) => (p ? f.flat_m[i][k] + moved * racing[i] : 0)));
      const shredded = flat_m[i].filter((m, k) => m >= this.shred_m && f.flat_m[i][k] < this.shred_m).length;
      d.floor = Math.min(d.floor + this.shred_floor * shredded, 1);
      if (racing[i] && flat_m[i].some((m) => m >= this.shred_m + this.rim_m)) wrecked[i] = true;   // a wheel gone
      cars[i] = { ...cars[i], crashed: cars[i].crashed || parked[i], done: cars[i].done || wrecked[i] };
    }
    const t = f.t + 1, laps = cars.map((c) => Math.floor(c.progress / c.tr.length));
    const flagNow = !f.flag && laps.some((l, i) => racing[i] && l >= this.laps);
    const flag_laps = flagNow ? laps.slice() : f.flag_laps;
    const finished = f.finished.map((fin, i) => fin || (racing[i] && (laps[i] >= this.laps || (f.flag && laps[i] > f.flag_laps[i]))));
    const finish_t = finished.map((fin, i) => (fin && !f.finished[i] ? t * cfg.dt : f.finish_t[i]));
    let done = t * cfg.dt >= cfg.time_limit, all = true;
    for (let i = 0; i < n; i++) {
      const dry = cars[i].car.fuel_used >= pit.setup[i].fuel && !finished[i];
      all &&= finished[i] || cars[i].crashed || wrecked[i] || dry;
      out_s.push(cars[i].crashed && !finished[i] ? f.out_s[i] + cfg.dt : 0);
    }
    return { cars, finished, finish_t, flag: f.flag || flagNow, flag_laps, bump, hit, damage, wrecked, setup: pit.setup,
             plan: f.plan, box_call: pit.box_call, service: pit.service, served: pit.served, stops: pit.stops, t,
             done: done || all, burst_at: f.burst_at, flat_m, out_s };
  }

  // ── pit stops (RaceField.pit) ──
  toBox(c, i) {
    const L = c.tr.length, s = mod(c.lap_pos - c.tr.pitAt + L / 2, L) - L / 2;
    return mod(this.box_s[i] - s + L / 2, L) - L / 2;
  }

  static boxSense(d) { return Math.sign(d) * Math.sqrt(Math.min(Math.abs(d) / 100, 1)); }

  pit(f, cars, damage, racing) {
    const cfg = this.cfg, out = { cars: [], damage: [], setup: [], box_call: [], service: [], served: [], stops: [] };
    for (let i = 0; i < this.n; i++) {
      const c = cars[i], d = damage[i];
      const inBox = this.game.inPitLane(c) && Math.abs(this.toBox(c, i)) < this.box_half;
      const arrive = racing[i] && inBox && Math.hypot(c.car.vx, c.car.vy) < 1 && !f.served[i] && f.service[i] <= 0;
      const nose = arrive && d.wing_front > 0.05;
      out.service.push(arrive ? this.tyre_s + nose * this.nose_s : Math.max(f.service[i] - cfg.dt, 0));
      out.served.push((f.served[i] || arrive) && inBox);
      out.cars.push(arrive ? { ...c, car: { ...c.car, wear: [0, 0, 0, 0], temp: Array(4).fill(cfg.tyre_start) } } : c);
      out.damage.push({ ...d, wing_front: nose ? 0 : d.wing_front, puncture: d.puncture.map((p) => p && !arrive) });
      out.setup.push(arrive ? { ...f.setup[i], compound: f.plan[i] } : f.setup[i]);
      out.box_call.push(f.box_call[i] && !arrive);
      out.stops.push(f.stops[i] + arrive);
    }
    return out;
  }

  // ── damage (RaceField.break_parts) ──
  breakParts(damage, hit) {
    const c = this.dmg, toeMax = c.toe_max_deg * RAD, out = [], wrecked = [];
    for (let i = 0; i < this.n; i++) {
      const d = damage[i], h = hit[i], wheels = h.slice(3);
      const toe = d.toe + c.toe_deg_per_kj * RAD * (wheels[0] - wheels[1]) / 1000;   // pulls to the hit side
      out.push({ wing_front: Math.min(d.wing_front + h[0] / c.wing_front_j, 1), wing_rear: Math.min(d.wing_rear + h[1] / c.wing_rear_j, 1),
                 floor: Math.min(d.floor + h[2] / c.floor_j, 1), toe: clip(toe, -toeMax, toeMax),
                 puncture: d.puncture.map((p, k) => p || wheels[k] >= c.puncture_j) });
      wrecked.push(h.reduce((a, b) => a + b, 0) >= c.retire_j || wheels.some((w) => w >= c.suspension_j));
    }
    return [out, wrecked];
  }

  // ── slipstream (RaceField.air, race.wake) ──
  air(f, moving) {
    const n = this.n, w = this.wk, cs = f.cars.map((c) => c.car);
    const v = cs.map((c, i) => (moving[i] ? Math.hypot(c.vx, c.vy) : 0));
    return cs.map((ci, i) => {
      let keep = 1;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const cj = cs[j], fx = Math.cos(cj.yaw), fy = Math.sin(cj.yaw), rx = ci.x - cj.x, ry = ci.y - cj.y;
        const behind = -(rx * fx + ry * fy);
        if (!(behind > 0 && behind < w.reach)) continue;
        const side = Math.abs(rx * fy - ry * fx), half = w.width + w.spread * behind;
        const each = Math.exp(-behind / w.fade) * Math.exp(-((side / half) ** 2)) * clip(v[j] / Math.max(v[i], 1), 0, 1);
        keep *= 1 - each;   // wakes compound, never past full strength
      }
      const s = 1 - keep;
      return { drag: 1 - w.drag_cut * s, down_front: 1 - w.front_wing_cut * s, down_rear: 1 - w.rear_wing_cut * s,
               floor: 1 - w.floor_cut * s };
    });
  }

  // ── contact (RaceField.boxes / outline / collide) ──
  boxes(cars, setup) {
    const sp = this.game.car.spec, D = this.game.car.D;
    return cars.map((c, i) => {
      const p = setup ? this.game.params(c.design, setup[i]) : c.p;
      const tyre = Math.max(c.design[D.tyre_front_mm], c.design[D.tyre_rear_mm]) / 1000;
      return { p, front: p.a + sp.overhang_front, rear: p.b + sp.overhang_rear, halfW: (p.track_width + tyre) / 2 };
    });
  }

  static outline(car, b) {
    const ex = [Math.cos(car.yaw), Math.sin(car.yaw)], ey = [-ex[1], ex[0]], halfL = (b.front + b.rear) / 2;
    const off = (b.front - b.rear) / 2, centre = [car.x + off * ex[0], car.y + off * ex[1]];
    const corners = [[1, 1], [1, -1], [-1, 1], [-1, -1]].map(([l, w]) =>   // front-left, front-right, rear-left, rear-right
      [centre[0] + l * halfL * ex[0] + w * b.halfW * ey[0], centre[1] + l * halfL * ex[1] + w * b.halfW * ey[1]]);
    return { centre, ex, ey, halfL, halfW: b.halfW, corners };
  }

  collide(cars, solid, moving, setup) {
    const n = this.n, bx = this.boxes(cars, setup), cs = cars.map((c) => c.car);
    const invM = cs.map((c, i) => (moving[i] ? 1 / (bx[i].p.mass - c.fuel_used) : 0));
    const invI = cs.map((c, i) => (moving[i] ? 1 / bx[i].p.yaw_inertia : 0));
    const o = cs.map((c, i) => RaceField.outline(c, bx[i]));
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
    const inside = (pt, k) => {
      const d = [pt[0] - o[k].centre[0], pt[1] - o[k].centre[1]];
      return Math.abs(dot(d, o[k].ex)) < o[k].halfL && Math.abs(dot(d, o[k].ey)) < o[k].halfW;
    };
    const pairs = [];
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const A = o[i], B = o[j], axes = [A.ex, A.ey, B.ex, B.ey];
      const rad = (K, u) => K.halfL * Math.abs(dot(u, K.ex)) + K.halfW * Math.abs(dot(u, K.ey));
      const cc = [B.centre[0] - A.centre[0], B.centre[1] - A.centre[1]];
      let a = 0, depth = Infinity, dist = 0;
      axes.forEach((u, k) => {
        const ds = dot(u, cc), dp = rad(A, u) + rad(B, u) - Math.abs(ds);
        if (dp < depth) { depth = dp; a = k; dist = ds; }
      });
      const s = dist < 0 ? -1 : 1, normal = [axes[a][0] * s, axes[a][1] * s];
      const ins = [...A.corners.map((pt) => inside(pt, j)), ...B.corners.map((pt) => inside(pt, i))];
      const pts = [...A.corners, ...B.corners], m = ins.filter(Boolean).length;
      const point = m ? [0, 1].map((k) => pts.reduce((sum, pt, q) => sum + (ins[q] ? pt[k] : 0), 0) / m)
        : [(A.centre[0] + B.centre[0]) / 2, (A.centre[1] + B.centre[1]) / 2];
      const touching = depth > 0 && solid[i] && solid[j] && (moving[i] || moving[j]);
      pairs.push({ i, j, depth, normal, point, touching,
                   ra: [point[0] - cs[i].x, point[1] - cs[i].y], rb: [point[0] - cs[j].x, point[1] - cs[j].y] });
    }
    let v = cs.map((c, i) => [c.vx * o[i].ex[0] + c.vy * o[i].ey[0], c.vx * o[i].ex[1] + c.vy * o[i].ey[1]]);
    let w = cs.map((c) => c.r);
    const e = this.restitution, mu = this.body_friction;
    const closing = (P) => {
      const { i, j, ra, rb, normal } = P;
      const rel = [v[j][0] - w[j] * rb[1] - (v[i][0] - w[i] * ra[1]), v[j][1] + w[j] * rb[0] - (v[i][1] + w[i] * ra[0])];
      return [rel, dot(rel, normal)];
    };
    for (const P of pairs) {
      const [rel, vn] = closing(P);
      P.vn0 = vn; P.slip0 = Math.hypot(rel[0] - vn * P.normal[0], rel[1] - vn * P.normal[1]); P.jn = 0; P.jt = 0;
    }
    const total = Array(n).fill(0);
    for (let it = 0; it < this.contact_iters; it++) {   // all pairs at once from the same velocities (as the JAX loop)
      const dv = cs.map(() => [0, 0]), dw = Array(n).fill(0);
      for (const P of pairs) {
        const { i, j, ra, rb, normal } = P, [rel, vn] = closing(P);
        const kN = invM[i] + invM[j] + cross(...ra, ...normal) ** 2 * invI[i] + cross(...rb, ...normal) ** 2 * invI[j];
        const jn = P.touching && vn < 0 ? -(1 + e) * vn / Math.max(kN, 1e-9) : 0;
        const tang = [rel[0] - vn * normal[0], rel[1] - vn * normal[1]], tn = Math.hypot(...tang);
        const t = [tang[0] / Math.max(tn, 1e-6), tang[1] / Math.max(tn, 1e-6)];
        const kT = invM[i] + invM[j] + cross(...ra, ...t) ** 2 * invI[i] + cross(...rb, ...t) ** 2 * invI[j];
        const jt = -Math.min(tn / Math.max(kT, 1e-9), mu * jn);   // rubbing, up to Coulomb's limit
        const imp = [jn * normal[0] + jt * t[0], jn * normal[1] + jt * t[1]];   // on car j; car i gets the opposite
        dv[j][0] += imp[0] * invM[j]; dv[j][1] += imp[1] * invM[j];
        dv[i][0] -= imp[0] * invM[i]; dv[i][1] -= imp[1] * invM[i];
        dw[j] += cross(...rb, ...imp) * invI[j]; dw[i] -= cross(...ra, ...imp) * invI[i];
        const size = Math.hypot(...imp);
        total[i] += size; total[j] += size;
        P.jn += jn; P.jt += Math.abs(jt);
      }
      v = v.map((x, k) => [x[0] + dv[k][0], x[1] + dv[k][1]]);
      w = w.map((x, k) => x + dw[k]);
    }
    // the energy each hit destroyed (crushing + rubbing), half to each car, on the part it landed on
    const reach = this.dmg.wheel_reach, hit = cars.map(() => Array(PARTS.length).fill(0));
    const part = (k, pt) => {
      const d = [pt[0] - cs[k].x, pt[1] - cs[k].y], lx = dot(d, o[k].ex), ly = dot(d, o[k].ey), p = bx[k].p;
      const atSide = Math.abs(ly) > p.track_width / 2 - 0.3;
      const wheel = lx > 0 ? (ly > 0 ? 3 : 4) : (ly > 0 ? 5 : 6);
      if (atSide && Math.min(Math.abs(lx - p.a), Math.abs(lx + p.b)) < reach) return wheel;
      return lx > p.a ? 0 : lx < -p.b ? 1 : 2;
    };
    const shift = cs.map(() => [0, 0]);
    for (const P of pairs) {
      const energy = 0.5 * P.jn * Math.max(-P.vn0, 0) * (1 - e) + 0.5 * P.jt * P.slip0;
      hit[P.i][part(P.i, P.point)] += energy / 2;
      hit[P.j][part(P.j, P.point)] += energy / 2;
      // push the boxes apart (by mass share), leaving a hair of overlap so contact persists while pressed together
      const push = P.touching ? Math.max(P.depth - 0.01, 0) : 0, share = invM[P.i] + invM[P.j];
      const si = invM[P.i] / Math.max(share, 1e-9), sj = invM[P.j] / Math.max(share, 1e-9);
      shift[P.i][0] -= push * si * P.normal[0]; shift[P.i][1] -= push * si * P.normal[1];
      shift[P.j][0] += push * sj * P.normal[0]; shift[P.j][1] += push * sj * P.normal[1];
    }
    const out = cars.map((c, k) => {
      const x = c.car.x + shift[k][0], y = c.car.y + shift[k][1];
      const car = { ...c.car, x, y, vx: dot(v[k], o[k].ex), vy: dot(v[k], o[k].ey), r: w[k] };
      const [idx, lat, lapPos] = this.game.locate(c.tr, c.idx, x, y), L = c.tr.length;
      const moved = mod(lapPos - c.lap_pos + L / 2, L) - L / 2;
      return { ...c, car, idx, lat, lap_pos: lapPos, progress: c.progress + moved };
    });
    return [out, total, hit];
  }

  // ── what each driver senses (RaceField.car_lasers / neighbours / observe) ──
  carLasers(f) {
    const n = this.n, bx = this.boxes(f.cars, null), segs = [];
    f.cars.forEach((c, k) => {
      if (f.finished[k] || this.gone(f, k)) return;   // ghosts don't show
      const q = RaceField.outline(c.car, bx[k]).corners;
      for (const [a, b] of [[0, 1], [2, 3], [0, 2], [1, 3]]) segs.push([k, q[a], q[b]]);   // front, rear, left, right
    });
    return f.cars.map((c, i) => this.game.rays.map((ray) => {
      const ang = c.car.yaw + ray, dx = Math.cos(ang), dy = Math.sin(ang);
      let best = Infinity;
      for (const [k, a, b] of segs) {
        if (k === i) continue;
        const ex = b[0] - a[0], ey = b[1] - a[1], wx = a[0] - c.car.x, wy = a[1] - c.car.y, den = cross(dx, dy, ex, ey);
        if (Math.abs(den) < 1e-9) continue;
        const t = cross(wx, wy, ex, ey) / den, u = cross(wx, wy, dx, dy) / den;
        if (t > 0 && u >= 0 && u <= 1 && t < best) best = t;
      }
      return Math.min(best, this.cfg.ray_range);
    }));
  }

  neighbours(f) {
    const n = this.n, cs = f.cars, half = this.cfg.width / 2;
    const along = cs.map((c) => {
      const h = c.tr.heading[c.idx];
      return c.car.vx * Math.cos(c.car.yaw - h) - c.car.vy * Math.sin(c.car.yaw - h);
    });
    const shows = cs.map((_, j) => !(f.finished[j] || this.gone(f, j)));
    return cs.map((ci, i) => {
      const L = ci.tr.length, near = [];
      for (let j = 0; j < n; j++) {
        if (j === i || !shows[j]) continue;
        const gap = mod(cs[j].lap_pos - ci.lap_pos + L / 2, L) - L / 2;
        if (Math.abs(gap) < this.sense_range)
          near.push({ j, gap, f: [1, gap / this.sense_range, (cs[j].lat - ci.lat) / half, (along[j] - along[i]) / 20] });
      }
      const pick = (ahead) => {   // nearest first; equal gaps by car number (a stable sort, as argsort)
        const got = near.filter((x) => (ahead ? x.gap >= 0 : x.gap < 0)).sort((a, b) => Math.abs(a.gap) - Math.abs(b.gap) || a.j - b.j);
        return [0, 1].flatMap((k) => (got[k] ? got[k].f : [0, 0, 0, 0]));
      };
      return [...pick(true), ...pick(false)];
    });
  }

  observe(f) {
    const g = this.game, cfg = this.cfg, R = this.rays, carL = this.carLasers(f), nb = this.neighbours(f);
    const toeMax = this.dmg.toe_max_deg * RAD;
    return f.cars.map((c, i) => {
      const own = g.observe(c, f.setup[i]), d = f.damage[i], L = c.tr.length, s = c.tr.fromPit(c.idx, cfg);
      for (let k = 0; k < R; k++) own[k] = Math.min(own[k], carL[i][k] / cfg.ray_range);
      return [...own, ...nb[i], d.wing_front, d.wing_rear, d.floor, d.toe / toeMax, ...d.puncture.map(Number),
              ...[0, 1, 2].map((k) => Number(f.setup[i].compound === k)),
              clip(1 - c.progress / (this.laps * L), 0, 1),
              Number(f.box_call[i]), Number(g.inPitLane(c)), RaceField.boxSense(this.toBox(c, i)),
              clip(mod(-cfg.pit_entry - s, L) / 1000, 0, 1), f.service[i] / 10];
    });
  }

  // each car's position (0 = leading): finishers by finish time, then the rest by distance, the out at the back
  order(f) {
    const key = f.cars.map((c, i) => [f.finished[i] ? 0 : c.crashed || f.wrecked[i] || this.dryOut(f, i) ? 2 : 1,
                                      f.finished[i] ? f.finish_t[i] : -c.progress, i]);
    const rank = key.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]).map((k) => k[2]);
    const pos = Array(this.n);
    rank.forEach((car, p) => { pos[car] = p; });
    return pos;
  }
}

// ── the AI pit wall (es_strategist.decide / senses) ──
export class Strategist {
  constructor(w) {
    for (const k of ["w1", "b1", "w2", "b2", "lin"]) { const a = decode(w[k]); this[k] = a.data; this[k + "_shape"] = a.shape; }
  }

  decide(x) {   // senses (25) -> [pit now?, compound]
    const [I, H] = this.w1_shape, out = [...this.b2];
    const h = [];
    for (let k = 0; k < H; k++) { let s = this.b1[k]; for (let i = 0; i < I; i++) s += x[i] * this.w1[i * H + k]; h.push(Math.tanh(s)); }
    for (let o = 0; o < 4; o++) {
      for (let k = 0; k < H; k++) out[o] += h[k] * this.w2[k * 4 + o];
      for (let i = 0; i < I; i++) out[o] += x[i] * this.lin[i * 4 + o];
    }
    let c = 0;
    for (let k = 1; k < 3; k++) if (out[1 + k] > out[1 + c]) c = k;
    return [out[0] > 0, c];
  }

  static senses(F, f, setFrom, gridFlag) {
    const N = F.n, pos = F.order(f), byPos = Array(N);
    pos.forEach((p, c) => { byPos[p] = c; });
    return f.cars.map((c, i) => {
      const L = c.tr.length, left = clip(1 - c.progress / (F.laps * L), 0, 1), wear = c.car.wear;
      const ahead = byPos[clip(pos[i] - 1, 0, N - 1)], behind = byPos[clip(pos[i] + 1, 0, N - 1)];
      const gapA = pos[i] > 0 ? f.cars[ahead].progress - c.progress : 0;
      const gapB = pos[i] < N - 1 ? c.progress - f.cars[behind].progress : 0;
      const d = f.damage[i], worst = (k) => Math.max(...f.cars[k].car.wear);
      return [left, left * F.laps / 25, ...[0, 1, 2].map((k) => Number(f.setup[i].compound === k)), ...wear, Math.max(...wear),
              Math.max(c.progress - setFrom[i], 0) / L / 25, 1 - c.car.fuel_used / f.setup[i].fuel, f.stops[i] / 3,
              d.floor, d.wing_front, Number(d.puncture.some(Boolean)), pos[i] / (N - 1),
              clip(gapA / 1000, 0, 3), clip(gapB / 1000, 0, 3), worst(ahead), worst(behind),
              (f.stops[ahead] - f.stops[i]) / 3, (f.stops[behind] - f.stops[i]) / 3, L / 5000, gridFlag];
    });
  }
}

// ── the race (scripts/race_full.py --strategist): the driver in every car, the AI pit wall on `team` cars, the rule
// on the rest; opts.player: a car the player calls instead, asking opts.human(f) as it crosses the decision line:
// {pit, compound} (the rule never calls it) ──
export class Race {
  constructor(data, circuit, opts = {}) {
    this.F = new RaceField(data);
    this.driver = new Driver(data.driver);
    this.strat = new Strategist(data.strategist);
    this.team = opts.team ?? data.team;
    this.rule = data.rule; this.decideM = data.decide_m;
    this.player = opts.player ?? -1; this.human = opts.human ?? null;
    const F = this.F, tr = typeof circuit === "object" ? circuit : F.game.circuits[circuit ?? data.reference.track];
    this.tr = tr;
    if (opts.state) { Object.assign(this, opts.state); return; }   // a snapshot (field-check.mjs)
    const rand = opts.rand ?? Math.random, n = F.n, designs = opts.designs ?? data.designs;
    const comp = Array(n).fill(this.rule.start);
    let f = F.grid(tr, designs, comp, rand);
    // the AI pit wall picks its cars' starting tyres, then the grid slots are shuffled (as race_full.py)
    const start = Strategist.senses(F, f, f.cars.map((c) => c.progress), 1).map((x) => this.strat.decide(x)[1]);
    const first = opts.start ?? ((i) => null);
    const tyres = comp.map((c, i) => first(i) ?? (this.team[i] ? start[i] : c));
    const burst = f.burst_at;
    f = F.grid(tr, designs, tyres, rand);
    f.burst_at = burst;
    const perm = opts.perm ?? shuffle(n, rand), o = perm.map((k) => f.cars[k]);
    f.cars = f.cars.map((c, i) => ({ ...c, car: { ...c.car, x: o[i].car.x, y: o[i].car.y, yaw: o[i].car.yaw }, idx: o[i].idx,
                                      lat: o[i].lat, lap_pos: o[i].lap_pos, progress: o[i].progress }));
    this.f = f;
    this.setFrom = f.cars.map((c) => c.progress);
    this.prev = f.cars.map((c) => this.toEntry(c));
  }

  toEntry(c) { return mod(-this.F.cfg.pit_entry - c.tr.fromPit(c.idx, this.F.cfg), c.tr.length); }

  racing(i) { const f = this.f; return !(f.cars[i].crashed || f.finished[i] || f.wrecked[i]); }

  // car i crosses its decision line (decide_m before the pit entry) in the next step: its once-a-lap call
  atLine(i) {
    const f = this.f, c = f.cars[i];
    return !f.done && this.racing(i) && !f.box_call[i] && !this.F.game.inPitLane(c) && this.prev[i] > this.decideM
      && this.toEntry(c) <= this.decideM;
  }

  // one decision (0.05 s). Returns what happened: {line: cars at the decision line, call: cars called in}
  step() {
    const F = this.F, f = this.f, n = F.n;
    if (f.done) return { line: [], call: [] };
    const racing = f.cars.map((_, i) => this.racing(i));
    const now = f.cars.map((c) => this.toEntry(c));
    const atLine = racing.map((_, i) => this.atLine(i));
    let x = null;
    const call = Array(n).fill(false), plan = f.plan.slice(), line = [];
    for (let i = 0; i < n; i++) {
      if (atLine[i]) line.push(i);
      if (i === this.player) {
        const h = atLine[i] ? this.human(f) : null;
        if (h && h.pit) { call[i] = true; plan[i] = h.compound; }
      } else if (this.team[i]) {
        if (atLine[i]) {
          x ??= Strategist.senses(F, f, this.setFrom, 0);
          const [pit, c] = this.strat.decide(x[i]);
          if (pit) { call[i] = true; plan[i] = c; }
        }
      } else if (racing[i] && !f.box_call[i] && Math.max(...f.cars[i].car.wear) >= this.rule.call_wear) {
        call[i] = true; plan[i] = this.rule.plan;
      }
    }
    const g = { ...f, box_call: f.box_call.map((b, i) => b || call[i]), plan: plan.map((p, i) => (call[i] ? p : f.plan[i])) };
    const obs = F.observe(g);
    const acts = obs.map((o) => this.driver.act(o));
    const nf = F.step(g, acts);
    this.setFrom = this.setFrom.map((s, i) => (nf.stops[i] > g.stops[i] ? nf.cars[i].progress : s));
    this.prev = now;
    this.f = nf;
    this.lastObs = obs; this.lastActs = acts;
    return { line, call: call.map((c, i) => (c ? i : -1)).filter((i) => i >= 0) };
  }
}

function shuffle(n, rand) {
  const a = [...Array(n).keys()];
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}
