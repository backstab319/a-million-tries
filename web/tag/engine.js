// Tag and the AIs' brains, ported line by line from games/tag.py and agents/ppo_car.py.
// Arenas, settings and weights come from tag.json + weights.bin (export_web.py tag).
// Works in the browser and in Node (check.mjs compares it with the Python game).

export const CHASER = 0, RUNNER = 1;
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

// Ray p + t*d against boxes: entry and exit t per box (hit if tIn <= tOut).
function slab(px, py, dx, dy, box) {
  const ix = 1 / (Math.abs(dx) < 1e-9 ? 1e-9 : dx), iy = 1 / (Math.abs(dy) < 1e-9 ? 1e-9 : dy);
  const t1x = (box[0] - px) * ix, t2x = (box[2] - px) * ix, t1y = (box[1] - py) * iy, t2y = (box[3] - py) * iy;
  return [Math.max(Math.min(t1x, t2x), Math.min(t1y, t2y)), Math.min(Math.max(t1x, t2x), Math.max(t1y, t2y))];
}

export class Arena {
  constructor(a) {  // boxes as [x0, y0, x1, y1]
    this.seed = a.seed;
    this.boxes = a.lo.map((lo, k) => [lo[0], lo[1], a.hi[k][0], a.hi[k][1]]);
    this.spawn = a.spawn;
  }
}

export class Game {
  constructor(cfg) {
    this.cfg = cfg;
    this.rays = [...Array(cfg.n_rays)].map((_, k) => {
      const a = (k * 2 * Math.PI) / cfg.n_rays;
      return [Math.cos(a), Math.sin(a)];
    });
    this.speed = [cfg.chaser_speed, cfg.runner_speed];
    this.accel = [cfg.chaser_accel, cfg.runner_accel];
  }

  // Both standing still; positions given (check.mjs) or picked like the Python game:
  // the chaser on a random free spot, the runner on the first of 16 random spots far enough away.
  init(arena, pos = null, rand = Math.random) {
    if (!pos) {
      const pick = () => arena.spawn[Math.floor(rand() * arena.spawn.length)];
      const c = pick();
      const cand = [...Array(16)].map(pick);
      const dist = cand.map((r) => Math.hypot(r[0] - c[0], r[1] - c[1]));
      let k = dist.findIndex((d) => d >= this.cfg.spawn_dist);
      if (k < 0) k = dist.indexOf(Math.max(...dist));
      pos = [c.slice(), cand[k].slice()];
    }
    pos = pos.map((p) => p.slice());
    const seen = this.visible(arena, pos[0], pos[1]);
    return { pos, vel: [[0, 0], [0, 0]], mem: [pos[1].slice(), pos[0].slice()], memVel: [[0, 0], [0, 0]],
             seen, since: 0, t: 0, caught: false, done: false };
  }

  visible(arena, a, b) {  // no obstacle on the straight line from a to b
    const dx = b[0] - a[0], dy = b[1] - a[1];
    for (const box of arena.boxes) {
      const [tIn, tOut] = slab(a[0], a[1], dx, dy, box);
      if (tIn <= tOut && tOut >= 0 && tIn <= 1) return false;
    }
    return true;
  }

  lasers(arena, p) {  // distance to the nearest wall or obstacle along each laser, capped at ray_range
    const S = this.cfg.size;
    return this.rays.map(([dx, dy]) => {
      let edge = Infinity;
      if (Math.abs(dx) >= 1e-9) edge = Math.min(edge, ((dx > 0 ? S : 0) - p[0]) / dx);
      if (Math.abs(dy) >= 1e-9) edge = Math.min(edge, ((dy > 0 ? S : 0) - p[1]) / dy);
      let box = Infinity;
      for (const b of arena.boxes) {
        const [tIn, tOut] = slab(p[0], p[1], dx, dy, b);
        if (tIn <= tOut && tOut > 0) box = Math.min(box, Math.max(tIn, 0));
      }
      return Math.min(edge, box, this.cfg.ray_range);
    });
  }

  // Move a disc out of the obstacles it overlaps and stop its motion into them.
  pushOut(arena, p, v) {
    const r = this.cfg.radius;
    let ox = 0, oy = 0, vx = 0, vy = 0;
    for (const [x0, y0, x1, y1] of arena.boxes) {
      const qx = clip(p[0], x0, x1), qy = clip(p[1], y0, y1);
      const ddx = p[0] - qx, ddy = p[1] - qy, dist = Math.hypot(ddx, ddy);
      let nx, ny, depth;
      if (dist < 1e-6) {  // the centre got into the box itself: leave by the nearest face
        const pen = [p[0] - x0, p[1] - y0, x1 - p[0], y1 - p[1]];
        const f = pen.indexOf(Math.min(...pen));
        [nx, ny] = [[-1, 0], [0, -1], [1, 0], [0, 1]][f];
        depth = pen[f] + r;
      } else {
        const m = Math.max(dist, 1e-6);
        nx = ddx / m; ny = ddy / m; depth = r - dist;
      }
      if (depth > 0) {
        ox += nx * depth; oy += ny * depth;
        const into = Math.min(nx * v[0] + ny * v[1], 0);
        vx += nx * into; vy += ny * into;
      }
    }
    return [[p[0] + ox, p[1] + oy], [v[0] - vx, v[1] - vy]];
  }

  // actions: [[ax, ay] chaser, [ax, ay] runner], each an acceleration, length capped at 1
  step(arena, st, actions) {
    const cfg = this.cfg, r = cfg.radius, S = cfg.size;
    const pos = [], vel = [];
    for (let w = 0; w < 2; w++) {
      let ax = clip(actions[w][0], -1, 1), ay = clip(actions[w][1], -1, 1);
      const n = Math.max(Math.hypot(ax, ay), 1);
      ax /= n; ay /= n;
      const drag = this.accel[w] / this.speed[w];
      let vx = st.vel[w][0] + (ax * this.accel[w] - st.vel[w][0] * drag) * cfg.dt;
      let vy = st.vel[w][1] + (ay * this.accel[w] - st.vel[w][1] * drag) * cfg.dt;
      let px = st.pos[w][0] + vx * cfg.dt, py = st.pos[w][1] + vy * cfg.dt;
      const cx = clip(px, r, S - r), cy = clip(py, r, S - r);  // the arena's edge
      if (cx !== px) vx = 0;
      if (cy !== py) vy = 0;
      const [p, v] = this.pushOut(arena, [cx, cy], [vx, vy]);
      pos.push(p); vel.push(v);
    }
    // caught: the closest the two came at any moment during this step, not just at its end
    const r0x = st.pos[1][0] - st.pos[0][0], r0y = st.pos[1][1] - st.pos[0][1];
    const r1x = pos[1][0] - pos[0][0], r1y = pos[1][1] - pos[0][1];
    const drx = r1x - r0x, dry = r1y - r0y;
    const u = clip(-(r0x * drx + r0y * dry) / Math.max(drx * drx + dry * dry, 1e-9), 0, 1);
    const caught = Math.hypot(r0x + u * drx, r0y + u * dry) < cfg.catch_dist;
    const seen = this.visible(arena, pos[0], pos[1]);
    const t = st.t + 1;
    const timeout = !caught && t * cfg.dt >= cfg.time_limit - 1e-6;
    return {
      pos, vel, seen, t, caught, done: caught || timeout,
      mem: seen ? [pos[1].slice(), pos[0].slice()] : st.mem,
      memVel: seen ? [vel[1].slice(), vel[0].slice()] : st.memVel,
      since: seen ? 0 : st.since + 1,
    };
  }

  observe(arena, st, who) {  // what player `who` knows, scaled to about [-1, 1] (27 numbers)
    const cfg = this.cfg, me = who, other = 1 - who, p = st.pos[me], S = cfg.size, v = cfg.chaser_speed;
    const otherVel = st.seen ? st.vel[other] : (cfg.remember_heading ? st.memVel[me] : [0, 0]);
    return [
      ...this.lasers(arena, p).map((d) => d / cfg.ray_range),
      st.vel[me][0] / v, st.vel[me][1] / v,
      (p[0] / S) * 2 - 1, (p[1] / S) * 2 - 1,
      st.seen ? 1 : 0,
      (st.mem[me][0] - p[0]) / S, (st.mem[me][1] - p[1]) / S,
      otherVel[0] / v, otherVel[1] / v,
      Math.min((st.since * cfg.dt) / cfg.time_limit, 1),
      1 - (st.t * cfg.dt) / cfg.time_limit,
    ];
  }
}

export class Brain {  // two tanh layers, then the move (the policy's mean: its best guess, as in the videos)
  constructor(layers, weights) {
    this.l = {};
    for (const { name, shape, offset } of layers) {
      const n = shape.reduce((a, b) => a * b, 1);
      this.l[name] = { shape, data: weights.subarray(offset, offset + n) };
    }
  }

  static dense(w, b, x, act) {
    const [nOut, nIn] = w.shape, y = new Float64Array(nOut);
    for (let o = 0; o < nOut; o++) {
      let s = b.data[o];
      for (let i = 0; i < nIn; i++) s += w.data[o * nIn + i] * x[i];
      y[o] = act ? Math.tanh(s) : s;
    }
    return y;
  }

  act(obs) {
    const L = this.l;
    const h1 = Brain.dense(L["body.0.weight"], L["body.0.bias"], obs, true);
    const h2 = Brain.dense(L["body.2.weight"], L["body.2.bias"], h1, true);
    const m = Brain.dense(L["policy.weight"], L["policy.bias"], h2, false);
    return [clip(m[0], -1, 1), clip(m[1], -1, 1)];
  }
}
