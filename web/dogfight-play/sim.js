// Two F-16s, guns only (episode 21), ported line by line from src/ailearns/games/dogfight.py (Fighters.step,
// nose, observe, init): a 3-D point-mass flight model from public F-16 figures, the 1976 US Standard Atmosphere,
// the M61 cannon's shells as packets with drag and gravity. Same order of operations as the Python, so check.mjs
// can replay Python fights step for step. Axes: x, y, z-up (metres). data/config.json is exported from Python.

export const ALIVE = 0, SHOT = 1, CRASHED = 2;
export const G = 9.80665;

// (episode 18) 1976 US Standard Atmosphere: height (m), density (kg/m^3), pressure (Pa), temperature (K)
const US76 = [
  [0, 1.225, 101325, 288.15], [5e3, 0.7364, 54048, 255.65], [10e3, 0.4135, 26500, 223.15],
  [11e3, 0.3639, 22632, 216.65], [15e3, 0.1948, 12111, 216.65], [20e3, 0.08891, 5529, 216.65],
  [25e3, 0.04008, 2549, 221.65], [30e3, 0.01841, 1197, 226.65], [32e3, 0.01322, 868.0, 228.65],
  [40e3, 0.003996, 287.1, 251.05], [47e3, 0.001426, 110.9, 270.65], [50e3, 0.001027, 79.78, 270.65],
  [51e3, 0.0008598, 66.94, 270.65], [60e3, 0.0003097, 21.96, 247.02], [70e3, 8.283e-5, 5.221, 219.59],
  [71e3, 6.420e-5, 3.956, 214.65], [80e3, 1.846e-5, 1.052, 196.65], [90e3, 3.40e-6, 0.1836, 186.87]];
const US_H = US76.map((r) => r[0]), US_LRHO = US76.map((r) => Math.log(r[1])), US_T = US76.map((r) => r[3]);

function interp(x, xs, ys) {   // numpy.interp
  if (x <= xs[0]) return ys[0];
  const n = xs.length;
  if (x >= xs[n - 1]) return ys[n - 1];
  let i = 1;
  while (xs[i] < x) i++;
  const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}
export function us76(y) {   // -> [density, speed of sound]
  y = Math.min(Math.max(y, 0.0), US_H[US_H.length - 1]);
  return [Math.exp(interp(y, US_H, US_LRHO)), Math.sqrt(1.4 * 287.05 * interp(y, US_H, US_T))];
}

// small vector helpers on [x, y, z] arrays
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm = (a) => Math.sqrt(dot(a, a));
export const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const unit = (a, eps = 1e-6) => mul(a, 1 / Math.max(norm(a), eps));
const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
const smooth = (x, a, b) => { const s = clip((x - a) / (b - a), 0, 1); return s * s * (3 - 2 * s); };
const rotate = (u, k, ang) => add(mul(u, Math.cos(ang)), mul(cross(k, u), Math.sin(ang)));   // u about unit k
const Z = [0, 0, 1];

export class Fighters {
  constructor(cfg) {
    this.c = cfg;
    this.K = cfg.n_packets;
  }

  cd0(mach) { const c = this.c; return c.cd0 * (1 + 2.2 * smooth(mach, 0.85, 1.1) - 0.8 * smooth(mach, 1.2, 2.0)); }
  k(mach) { return this.c.k_induced * (1 + 1.0 * smooth(mach, 0.9, 1.4)); }
  thrust(power, rho, mach) {
    const c = this.c, sigma = rho / 1.225;
    const dry = c.thrust_idle + (c.thrust_mil - c.thrust_idle) * clip(power / 0.75, 0, 1);
    const wet = (c.thrust_max - c.thrust_mil) * clip((power - 0.75) / 0.25, 0, 1);
    return dry * sigma ** 0.8 * (1 + 0.15 * mach) + wet * sigma ** 0.7 * (1 + 0.3 * mach);
  }
  loadLimits(q) {
    const c = this.c, qs = (q * c.wing_area) / c.weight;
    return [Math.max(c.g_min, qs * c.cl_alpha * c.alpha_min), Math.min(c.g_max, qs * c.cl_alpha * c.alpha_max)];
  }

  // a fight: plane 1 level at alt, plane 0 anywhere within ~4 km at level 1 (Fighters.init with level = 1; our own
  // random numbers, so not the same starts as Python's)
  init(rand = Math.random) {
    const c = this.c, u = (lo = 0, hi = 1) => lo + (hi - lo) * rand();
    const alt = u(c.alt_lo, c.alt_hi), psi = u(-Math.PI, Math.PI);
    const fwd1 = [Math.cos(psi), Math.sin(psi), 0], v1 = u(200, 260);
    const rng = 600 + u(0, 3400), off = u(-Math.PI, Math.PI);
    const tail = [-Math.cos(psi + off), -Math.sin(psi + off), 0];
    const p0 = add(add([0, 0, alt], mul(tail, rng)), [0, 0, u(-1000, 1000)]);
    const head0 = psi + u(-Math.PI, Math.PI);
    const fwd0 = [Math.cos(head0), Math.sin(head0), 0], v0 = u(200, 280);
    return this.fromState({
      pos: [p0, [0, 0, alt]], vel: [mul(fwd0, v0), mul(fwd1, v1)], up: [Z.slice(), Z.slice()],
      load: [1, 1], roll: [0, 0], power: [0.75, 0.75], health: [1, 1], ammo: [c.rounds, c.rounds], cause: [0, 0],
      time: 0, n: 0,
    });
  }

  // a state from plain arrays (check.mjs loads Python's start state this way); empty gun packets
  fromState(s) {
    const K = this.K, z3 = () => Array.from({ length: K }, () => [0, 0, 0]);
    return {
      pos: s.pos.map((p) => p.slice()), vel: s.vel.map((p) => p.slice()), up: s.up.map((p) => p.slice()),
      load: s.load.slice(), roll: s.roll.slice(), power: s.power.slice(), health: s.health.slice(),
      ammo: s.ammo.slice(), cause: s.cause.slice(),
      ppos: s.ppos ? s.ppos.map((r) => r.map((p) => p.slice())) : [z3(), z3()],
      pvel: s.pvel ? s.pvel.map((r) => r.map((p) => p.slice())) : [z3(), z3()],
      prounds: s.prounds ? s.prounds.map((r) => r.slice()) : [new Array(K).fill(0), new Array(K).fill(0)],
      page: s.page ? s.page.map((r) => r.slice()) : [new Array(K).fill(0), new Array(K).fill(0)],
      prho: s.prho ? s.prho.map((r) => r.slice()) : [new Array(K).fill(1), new Array(K).fill(1)],
      time: s.time, n: s.n,
    };
  }

  // where each jet points: the velocity tilted up by the angle of attack -> [[nose0, nose1], [alpha0, alpha1]]
  nose(st) {
    const c = this.c, noses = [], alphas = [];
    for (let i = 0; i < 2; i++) {
      const [rho] = us76(st.pos[i][2]);
      const q = 0.5 * rho * dot(st.vel[i], st.vel[i]);
      const alpha = clip((st.load[i] * c.weight) / (Math.max(q, 1) * c.wing_area * c.cl_alpha), c.alpha_min, c.alpha_max);
      const vhat = unit(st.vel[i]);
      noses.push(add(mul(vhat, Math.cos(alpha)), mul(st.up[i], Math.sin(alpha))));
      alphas.push(alpha);
    }
    return [noses, alphas];
  }

  // action [[g, roll, throttle, trigger] x 2] in [-1, 1] -> the new state (st is not changed)
  step(st, action) {
    const c = this.c, dt = c.dt, K = this.K;
    const a = action.map((r) => r.map((x) => clip(x, -1, 1)));
    const alive = st.cause.map((x) => x === ALIVE);
    const out = { pos: [], vel: [], up: [], load: [], roll: [], power: [], noseUsed: [], rho: [] };
    for (let i = 0; i < 2; i++) {
      // 1. air, speed, what the wings can do
      const alt = st.pos[i][2], [rho, sound] = us76(alt);
      const speed = norm(st.vel[i]), vhat = unit(st.vel[i]);
      const q = 0.5 * rho * speed ** 2, mach = speed / sound;
      const [glo, ghi] = this.loadLimits(q);
      // 2. the flight computer follows the stick, the engine spools
      const gcmd = a[i][0] >= 0 ? a[i][0] * c.g_max : -a[i][0] * c.g_min;
      let load = st.load[i] + (clip(gcmd, glo, ghi) - st.load[i]) * (1 - Math.exp(-dt / c.g_lag));
      load = clip(load, glo, ghi);
      const rollCmd = a[i][1] * c.roll_rate_max * clip(q / c.roll_q, 0, 1);
      const roll = st.roll[i] + (rollCmd - st.roll[i]) * (1 - Math.exp(-dt / c.roll_lag));
      const power = st.power[i] + ((a[i][2] + 1) / 2 - st.power[i]) * (1 - Math.exp(-dt / c.spool));
      // 3. forces: lift along up, drag against the airflow, thrust along the nose, gravity
      const lift = load * c.weight;
      const cl = lift / (Math.max(q, 1) * c.wing_area);
      const alpha = clip(cl / c.cl_alpha, c.alpha_min, c.alpha_max);
      const drag = q * c.wing_area * (this.cd0(mach) + this.k(mach) * cl ** 2);
      const thrust = this.thrust(power, rho, mach);
      const nose = add(mul(vhat, Math.cos(alpha)), mul(st.up[i], Math.sin(alpha)));
      const force = add(sub(mul(st.up[i], lift), mul(vhat, drag)), mul(nose, thrust));
      const acc = sub(mul(force, 1 / c.mass), mul(Z, G));
      const vel = add(st.vel[i], mul(acc, dt));
      const pos = add(st.pos[i], mul(vel, dt));
      // 4. roll the lift direction about the flight path, keep it perpendicular to the new velocity
      let up = rotate(st.up[i], vhat, roll * dt);
      const vn = unit(vel);
      const upNew = unit(sub(up, mul(vn, dot(up, vn))));
      up = norm(vel) > 1 ? upNew : st.up[i];
      Object.entries({ pos, vel, up, load, roll, power, noseUsed: nose, rho }).forEach(([k, v]) => out[k].push(v));
    }
    // 5. the gun: fire a packet into this step's slot of the ring; move the packets already flying
    const slot = st.n % K;
    const ppos = st.ppos.map((r) => r.map((p) => p.slice())), pvel = st.pvel.map((r) => r.map((p) => p.slice()));
    const prounds = st.prounds.map((r) => r.slice()), page = st.page.map((r) => r.slice()), prho = st.prho.map((r) => r.slice());
    const shot = [0, 0], damageTo = [0, 0];
    for (let i = 0; i < 2; i++) {
      const fire = alive[i] && a[i][3] > 0 && st.ammo[i] > 0;
      shot[i] = fire ? Math.min(st.ammo[i], c.rate * dt) : 0;
      if (fire) {
        ppos[i][slot] = add(st.pos[i], mul(out.noseUsed[i], 8.0));
        pvel[i][slot] = add(st.vel[i], mul(out.noseUsed[i], c.muzzle_speed));
      }
      prounds[i][slot] = shot[i];
      page[i][slot] = 0;
      prho[i][slot] = out.rho[i];
    }
    for (let i = 0; i < 2; i++) {   // hits: each packet's closest approach to the OTHER jet during this step
      const j = 1 - i;
      let hits = 0;
      for (let k = 0; k < K; k++) {
        const r0 = sub(ppos[i][k], st.pos[j]), w = sub(pvel[i][k], st.vel[j]);
        const tc = -dot(r0, w) / Math.max(dot(w, w), 1e-6);
        const passing = tc >= 0 && tc < dt && prounds[i][k] > 0;
        if (!passing) continue;
        const miss = norm(add(r0, mul(w, clip(tc, 0, dt))));
        const spread = c.dispersion * c.muzzle_speed * (page[i][k] + tc);
        const s2 = c.target_radius ** 2 + spread ** 2;
        hits += prounds[i][k] * (c.target_radius ** 2 / s2) * Math.exp(-(miss ** 2) / (2 * s2));
        prounds[i][k] = 0;
      }
      damageTo[j] = hits / c.hits_to_kill;
    }
    for (let i = 0; i < 2; i++) {   // shells fly on: drag and gravity
      for (let k = 0; k < K; k++) {
        const bs = norm(pvel[i][k]);
        pvel[i][k] = sub(sub(pvel[i][k], mul(pvel[i][k], ((c.bullet_drag * prho[i][k]) / 1.225) * bs * dt)), mul(Z, G * dt));
        ppos[i][k] = add(ppos[i][k], mul(pvel[i][k], dt));
        page[i][k] += dt;
        if (page[i][k] > c.bullet_life) prounds[i][k] = 0;
      }
    }
    // 6. damage, the ground, the end
    const health = [], cause = [];
    for (let i = 0; i < 2; i++) {
      let h = alive[i] ? Math.max(st.health[i] - damageTo[i], 0) : st.health[i];
      const crashed = alive[i] && out.pos[i][2] <= 0;
      if (crashed) h = 0;
      health.push(h);
      cause.push(!alive[i] ? st.cause[i] : crashed ? CRASHED : h <= 0 ? SHOT : ALIVE);
    }
    const time = st.time + dt;
    const done = cause.some((x) => x !== ALIVE) || time >= c.time_limit - 1e-6;
    return {
      st: { pos: out.pos, vel: out.vel, up: out.up, load: out.load, roll: out.roll, power: out.power, health,
        ammo: [st.ammo[0] - shot[0], st.ammo[1] - shot[1]], cause, ppos, pvel, prounds, page, prho, time, n: st.n + 1 },
      done,
    };
  }

  // plane i's senses (38 numbers), in its own body frame (nose f, right wing r, canopy u)
  observe(st, i) {
    const c = this.c, j = 1 - i;
    const [noses, alphas] = this.nose(st);
    const f = noses[i];
    const u = unit(sub(st.up[i], mul(f, dot(st.up[i], f))));
    const r = cross(f, u);
    const frame = (v) => [dot(v, f), dot(v, r), dot(v, u)];
    const p = st.pos[i], v = st.vel[i], speed = norm(v);
    const [, sound] = us76(p[2]);
    const d = sub(st.pos[j], p), dist = norm(d), dhat = mul(d, 1 / Math.max(dist, 1));
    const w = sub(st.vel[j], v);
    const bv = sub(add(v, mul(f, c.muzzle_speed)), st.vel[j]);
    const tof = clip(dot(d, bv) / Math.max(dot(bv, bv), 1), 0, c.bullet_life);
    const pip = frame(sub(mul(bv, tof), d));
    const own = [speed / 300, speed / sound, p[2] / 5000, v[2] / 100, f[2], u[2], r[2], st.load[i] / 9,
      alphas[i] / c.alpha_max, st.power[i], st.ammo[i] / c.rounds, st.health[i], (p[2] + speed ** 2 / (2 * G)) / 8000];
    const other = [...mul(frame(d), 1 / 1000), dist / 1000, Math.log1p(dist / 100), ...frame(dhat), ...mul(frame(w), 1 / 300),
      ...frame(noses[j]), norm(st.vel[j]) / 300, d[2] / 1000, -dot(noses[j], dhat), -dot(w, dhat) / 300, st.health[j],
      st.ammo[j] / c.rounds, ...mul(pip, 1 / 50), tof, 1 - st.time / c.time_limit];
    return [...own, ...other];
  }
}

// the AI pilot: the policy network (data/pilot.json), flown as in training: the stick's mean plus its trained
// spread (normal noise), the trigger drawn from its firing probability
export class Pilot {
  constructor(w) { this.w = w; }
  mean(obs) {
    const w = this.w;
    let x = obs;
    for (let l = 0; l < w.layers; l++) {
      const W = w[`pi${l}.w`], b = w[`pi${l}.b`], y = new Array(b.length);
      for (let o = 0; o < b.length; o++) {
        let s = b[o];
        for (let k = 0; k < x.length; k++) s += x[k] * W[k][o];
        y[o] = Math.tanh(s);
      }
      x = y;
    }
    const W = w["pi.out.w"], b = w["pi.out.b"];
    return b.map((bo, o) => x.reduce((s, xk, k) => s + xk * W[k][o], bo));
  }
  act(obs, rand = Math.random) {
    const m = this.mean(obs), s = this.w.std;
    const gauss = () => Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());
    const fire = rand() < 1 / (1 + Math.exp(-m[3]));
    return [m[0] + s[0] * gauss(), m[1] + s[1] * gauss(), m[2] + s[2] * gauss(), fire ? 1 : -1];
  }
}
