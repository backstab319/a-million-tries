// The race car game and the AI's brain, ported line by line from games/car.py and
// agents/ppo_car.py. Track, constants and weights come from data.json (export_web.py).
// Works in the browser and in Node (check.mjs compares it with the Python game).

export function decode({ shape, b64 }) {
  const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return { shape, data: new Float32Array(bytes.buffer) };
}

const mod = (a, n) => ((a % n) + n) % n;           // Python/JAX-style modulo
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const cross = (ax, ay, bx, by) => ax * by - ay * bx;

export class Game {
  constructor(data) {
    this.cfg = data.cfg;
    const t = data.track;
    this.center = decode(t.center).data;           // P x 2
    this.headings = decode(t.heading).data;        // P
    this.wallA = decode(t.wall_a).data;            // S x 2
    this.wallB = decode(t.wall_b).data;
    this.length = t.length;
    this.P = this.headings.length;
    this.S = this.wallA.length / 2;
    this.rays = this.cfg.ray_angles.map((d) => (d * Math.PI) / 180);
  }

  init() {  // standing still on the start line, facing along the track
    return { x: this.center[0], y: this.center[1], heading: this.headings[0], speed: 0, idx: 0, lapPos: 0,
             progress: 0, t: 0, last: [0, 0], crashed: false };
  }

  locate(st, x, y) {
    const { P, center: c } = this;
    let j = 0, best = Infinity;
    for (let k = -6; k < 15; k++) {
      const i = mod(st.idx + k, P);
      const d = (c[2 * i] - x) ** 2 + (c[2 * i + 1] - y) ** 2;
      if (d < best) { best = d; j = i; }
    }
    const onSegment = (i0) => {
      const a = mod(i0, P), b = mod(i0 + 1, P);
      const ax = c[2 * a], ay = c[2 * a + 1], abx = c[2 * b] - ax, aby = c[2 * b + 1] - ay;
      const u = clip(((x - ax) * abx + (y - ay) * aby) / (abx * abx + aby * aby), 0, 1);
      return [Math.hypot(ax + u * abx - x, ay + u * aby - y), a + u];
    };
    const [d0, f0] = onSegment(j - 1), [d1, f1] = onSegment(j);
    const frac = d0 < d1 ? f0 : f1;
    return { idx: j, lateral: Math.min(d0, d1), lapPos: mod(frac * (this.length / P), this.length) };
  }

  step(st, steerIn, throttleIn) {
    const cfg = this.cfg;
    const steer = clip(steerIn, -1, 1), throttle = clip(throttleIn, -1, 1);
    const drag = cfg.accel / cfg.max_speed ** 2;
    const acc = (throttle >= 0 ? throttle * cfg.accel : throttle * cfg.brake) - drag * st.speed ** 2;
    const v = clip(st.speed + acc * cfg.dt, 0, cfg.max_speed * 1.05);
    const want = Math.tan(steer * cfg.max_steer) / cfg.wheelbase;   // curvature the wheels ask for
    const can = cfg.grip / Math.max(v * v, 1e-3);                   // curvature the tyres can hold
    const heading = st.heading + v * clip(want, -can, can) * cfg.dt;
    const x = st.x + v * cfg.dt * Math.cos(heading), y = st.y + v * cfg.dt * Math.sin(heading);
    const loc = this.locate(st, x, y);
    const L = this.length;
    const moved = mod(loc.lapPos - st.lapPos + L / 2, L) - L / 2;
    return { x, y, heading, speed: v, idx: loc.idx, lapPos: loc.lapPos, progress: st.progress + moved,
             t: st.t + 1, last: [steer, throttle], crashed: loc.lateral > cfg.width / 2 - cfg.car_half_width };
  }

  lasers(st) {  // distance to the nearest wall along each laser, capped at ray_range
    const { wallA: A, wallB: B, S } = this, out = [];
    for (const r of this.rays) {
      const dx = Math.cos(st.heading + r), dy = Math.sin(st.heading + r);
      let best = Infinity;
      for (let s = 0; s < S; s++) {
        const ex = B[2 * s] - A[2 * s], ey = B[2 * s + 1] - A[2 * s + 1];
        const wx = A[2 * s] - st.x, wy = A[2 * s + 1] - st.y;
        const den = cross(dx, dy, ex, ey);
        if (Math.abs(den) < 1e-9) continue;
        const t = cross(wx, wy, ex, ey) / den, u = cross(wx, wy, dx, dy) / den;
        if (t > 0 && u >= 0 && u <= 1 && t < best) best = t;
      }
      out.push(Math.min(best, this.cfg.ray_range));
    }
    return out;
  }

  observe(st, lasers = this.lasers(st)) {
    return [...lasers.map((d) => d / this.cfg.ray_range), st.speed / this.cfg.max_speed, ...st.last];
  }

  laps(st) { return Math.floor(st.progress / this.length); }
}

export class Brain {  // two tanh layers, then steering + pedal (the policy's mean: best-guess driving)
  constructor(b) {
    this.w0 = decode(b["body.0.weight"]); this.b0 = decode(b["body.0.bias"]).data;
    this.w2 = decode(b["body.2.weight"]); this.b2 = decode(b["body.2.bias"]).data;
    this.wp = decode(b["policy.weight"]); this.bp = decode(b["policy.bias"]).data;
  }

  static dense(w, bias, x, act) {
    const [nOut, nIn] = w.shape, y = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = bias[o];
      for (let i = 0; i < nIn; i++) s += w.data[o * nIn + i] * x[i];
      y[o] = act ? Math.tanh(s) : s;
    }
    return y;
  }

  act(obs) {
    const h1 = Brain.dense(this.w0, this.b0, obs, true);
    const h2 = Brain.dense(this.w2, this.b2, h1, true);
    const m = Brain.dense(this.wp, this.bp, h2, false);
    return { steer: clip(m[0], -1, 1), throttle: clip(m[1], -1, 1), h1, h2 };
  }
}
