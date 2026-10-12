// CQB ("the kill house") and its AIs' brains, ported line by line from games/cqb.py and agents/ppo_cqb.py.
// Houses, rules and weights come from cqb.json + weights.bin (export_web.py cqb).
// Works in the browser and in Node (check.mjs compares it with the Python game, dice included).

export const ASSAULT = 0, DEFENSE = 1;
export const FWD = 0, SIDE = 1, TURN = 2, FIRE = 3, INTERACT = 4, RUN = 5, THROW = 6, RELOAD = 7;
const FAR = -1000;
const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const norm = (x, y) => Math.sqrt(x * x + y * y);
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const rot = (v, a) => { const c = Math.cos(a), s = Math.sin(a); return [c * v[0] - s * v[1], s * v[0] + c * v[1]]; };
const roundEven = (x) => {   // jnp.round: halves go to the even neighbour
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r;
};

// Ray p + t*d against a box [lx, ly, hx, hy]: entry and exit t (hit if tIn <= tOut).
function slab(px, py, dx, dy, b) {
  const ix = 1 / (Math.abs(dx) < 1e-9 ? 1e-9 : dx), iy = 1 / (Math.abs(dy) < 1e-9 ? 1e-9 : dy);
  const x1 = (b[0] - px) * ix, x2 = (b[2] - px) * ix, y1 = (b[1] - py) * iy, y2 = (b[3] - py) * iy;
  return [Math.max(Math.min(x1, x2), Math.min(y1, y2)), Math.min(Math.max(x1, x2), Math.max(y1, y2))];
}

function clear(p, q, boxes) {  // the straight line from p to q is free of the boxes
  const dx = q[0] - p[0], dy = q[1] - p[1];
  for (const b of boxes) {
    const [tIn, tOut] = slab(p[0], p[1], dx, dy, b);
    if (tIn <= tOut && tOut >= 0 && tIn <= 1) return false;
  }
  return true;
}

function pushOut(p, v, boxes, r) {  // games/cqb.py _push_out: deepest box first, three times
  const normals = [[-1, 0], [0, -1], [1, 0], [0, 1]];
  p = p.slice(); v = v.slice();
  for (let it = 0; it < 3; it++) {
    let best = -Infinity, n = null;
    for (const b of boxes) {
      const dx = p[0] - clip(p[0], b[0], b[2]), dy = p[1] - clip(p[1], b[1], b[3]);
      const dist = norm(dx, dy), inside = dist < 1e-6;
      const pen = [p[0] - b[0], p[1] - b[1], b[2] - p[0], b[3] - p[1]];
      const depth = inside ? Math.min(...pen) + r : r - dist;
      if (depth > best) {
        best = depth;
        n = inside ? normals[pen.indexOf(Math.min(...pen))] : [dx / Math.max(dist, 1e-6), dy / Math.max(dist, 1e-6)];
      }
    }
    if (best > 0) {
      p = [p[0] + n[0] * best, p[1] + n[1] * best];
      const dot = Math.min(n[0] * v[0] + n[1] * v[1], 0);
      v = [v[0] - n[0] * dot, v[1] - n[1] * dot];
    }
  }
  return [p, v];
}

export class House {
  constructor(h) {
    const box = (lo, hi) => lo.map((l, k) => [l[0], l[1], hi[k][0], hi[k][1]]);
    const real = (b) => b[0] !== FAR || b[2] !== FAR;
    this.seed = h.seed;
    this.walls = box(h.wall_lo, h.wall_hi).filter(real);
    this.cover = box(h.cover_lo, h.cover_hi).filter(real);
    this.doors = box(h.door_lo, h.door_hi);                 // all slots (indices matter); valid says which are doors
    this.doorValid = h.door_valid; this.doorOutside = h.door_outside; this.centerFed = h.door_center_fed;
    this.entries = h.entry.slice(0, h.n_entries); this.entryFace = h.entry_face.slice(0, h.n_entries);
    this.spawn = h.spawn; this.footLo = h.foot_lo; this.footHi = h.foot_hi; this.cells = h.cell_point;
  }
}

export class Game {
  constructor(cfg) {
    this.cfg = cfg;
    const steps = (s) => Math.round(s / cfg.dt);
    this.doorSteps = steps(cfg.door_time); this.T = steps(cfg.time_limit);
    this.cosHalf = Math.cos((cfg.fov / 2) * Math.PI / 180); this.cosAim = Math.cos(cfg.aim * Math.PI / 180);
    this.flashSteps = steps(cfg.flash); this.soundSteps = steps(cfg.sound_memory);
    this.rayDirs = [...Array(cfg.n_rays)].map((_, k) => [Math.cos(k * 2 * Math.PI / cfg.n_rays), Math.sin(k * 2 * Math.PI / cfg.n_rays)]);
    const half = (cfg.fov / 2) * Math.PI / 180;
    this.cone = [...Array(cfg.cone_rays)].map((_, k) => -half + (2 * half * k) / (cfg.cone_rays - 1));
    this.bangsOn = cfg.flashbangs.some((x) => x > 0);
    this.fuseSteps = steps(cfg.fuse); this.blindSteps = steps(cfg.blind_time);
    this.gunsOn = cfg.magazine > 0; this.reloadSteps = steps(cfg.reload_time);
    this.healthOn = (cfg.health || 0) > 0; this.reactOn = (cfg.react || 0) > 0;
    this.reactSteps = Math.max(steps(cfg.react || 0), 1);
  }

  // A round's start: given (check.mjs: pos, face, door_open as the Python game had them) or drawn like init().
  start(h, given = null, rand = Math.random) {
    const cfg = this.cfg;
    if (!given) {
      const e = Math.floor(rand() * h.entries.length);
      const d = h.spawn[Math.floor(rand() * cfg.spawn_points)];
      given = { pos: [h.entries[e].slice(), d.slice()], face: [h.entryFace[e], (rand() * 2 - 1) * Math.PI],
                door_open: h.doorValid.map((v, k) => v && !h.doorOutside[k] && rand() < cfg.inside_open) };
    }
    const pos = given.pos.map((p) => p.slice()), z2 = () => [0, 0];
    const st = {
      pos, vel: [z2(), z2()], face: given.face.slice(), turn: [0, 0], doorOpen: given.door_open.slice(),
      doorTimer: h.doors.map(() => 0), doorEvent: [false, false], doorAt: pos.map((p) => p.slice()),
      seen: [false, false], mem: [z2(), z2()], memVel: [z2(), z2()], since: [this.T, this.T],
      cleared: [h.cells.map(() => false), h.cells.map(() => false)],
      sound: [0, 1].map(() => [0, 1, 2].map(() => [0, 0, 0])), soundAge: [0, 1].map(() => [0, 1, 2].map(() => this.soundSteps)),
      alive: [true, true], fired: [false, false], flash: [0, 0], flashSeen: [false, false],
      bangs: cfg.flashbangs.slice(), bangPos: pos.map((p) => p.slice()), bangTimer: [0, 0], bangNow: [false, false],
      blind: [0, 0], bangHeard: [[0, 0, 0], [0, 0, 0]], bangAge: [this.soundSteps, this.soundSteps],
      mag: [cfg.magazine, cfg.magazine], reserve: [cfg.magazine * cfg.spare_mags, cfg.magazine * cfg.spare_mags],
      reloadTimer: [0, 0], reloadNow: [false, false], kick: [0, 0], reloadHeard: [[0, 0, 0], [0, 0, 0]],
      reloadAge: [this.soundSteps, this.soundSteps], hp: [cfg.health || 0, cfg.health || 0],
      landed: [this.soundSteps, this.soundSteps], aim: [0, 0], t: 0, done: false, outcome: 0,
    };
    return this.look(h, st);
  }

  boxes(h, st, doorOpen = st.doorOpen) {   // walls, cover, shut doors (in this order: pushOut's ties follow it)
    return [...h.walls, ...h.cover, ...h.doors.filter((_, k) => h.doorValid[k] && !doorOpen[k])];
  }

  wallsOnly(h, st) {                        // walls and shut doors: what stops sound, a throw and a bang
    return [...h.walls, ...h.doors.filter((_, k) => h.doorValid[k] && !st.doorOpen[k])];
  }

  cast(p, dirs, boxes) {  // distance along each direction to the first box or the yard's edge, capped at sight
    const cfg = this.cfg, yard = [cfg.yard_w, cfg.yard_h];
    return dirs.map((d) => {
      let edge = Infinity;
      for (const ax of [0, 1]) {
        if (Math.abs(d[ax]) < 1e-9) continue;
        edge = Math.min(edge, (d[ax] > 0 ? yard[ax] - p[ax] : -p[ax]) / d[ax]);
      }
      let box = Infinity;
      for (const b of boxes) {
        const [tIn, tOut] = slab(p[0], p[1], d[0], d[1], b);
        if (tIn <= tOut && tOut > 0) box = Math.min(box, Math.max(tIn, 0));
      }
      return Math.min(edge, box, cfg.sight);
    });
  }

  inCone(p, face, q) {
    const dx = q[0] - p[0], dy = q[1] - p[1], dist = norm(dx, dy);
    return dx * Math.cos(face) + dy * Math.sin(face) >= this.cosHalf * dist && dist <= this.cfg.sight;
  }

  body(p, q) {
    const dx = q[0] - p[0], dy = q[1] - p[1], k = this.cfg.radius / Math.max(norm(dx, dy), 1e-6);
    return [q, [q[0] - dy * k, q[1] + dx * k], [q[0] + dy * k, q[1] - dx * k]];
  }

  sees(st, who, boxes) {
    const p = st.pos[who];
    return st.blind[who] === 0 && this.body(p, st.pos[1 - who]).some((q) => this.inCone(p, st.face[who], q) && clear(p, q, boxes));
  }

  light(p, face, boxes) {
    return this.cast(p, this.cone.map((a) => [Math.cos(face + a), Math.sin(face + a)]), boxes);
  }

  lit(p, face, qs, boxes) {
    const reach = this.light(p, face, boxes), c = this.cone, step = (c[c.length - 1] - c[0]) / (this.cfg.cone_rays - 1);
    return qs.map((q) => {
      const dx = q[0] - p[0], dy = q[1] - p[1], dist = norm(dx, dy), rel = wrap(Math.atan2(dy, dx) - face);
      const k = clip(roundEven((rel - c[0]) / step), 0, this.cfg.cone_rays - 1);
      return Math.abs(rel) <= c[c.length - 1] + 1e-6 && dist <= reach[k];
    });
  }

  look(h, st) {
    const boxes = this.boxes(h, st);
    const seen = [0, 1].map((w) => this.sees(st, w, boxes));
    st.mem = [0, 1].map((w) => (seen[w] ? st.pos[1 - w].slice() : st.mem[w]));
    st.memVel = [0, 1].map((w) => (seen[w] ? st.vel[1 - w].slice() : st.memVel[w]));
    st.since = [0, 1].map((w) => (seen[w] ? 0 : Math.min(st.since[w] + 1, this.T)));
    st.cleared = [0, 1].map((w) => {
      const look = this.lit(st.pos[w], st.face[w], h.cells, boxes);
      return st.cleared[w].map((c, k) => c || (look[k] && st.blind[w] === 0));
    });
    st.seen = seen;
    if (this.reactOn) st.aim = [0, 1].map((w) => (seen[w] ? Math.min(st.aim[w] + 1, this.reactSteps) : Math.max(st.aim[w] - 1, 0)));
    return st;
  }

  move(h, st, a) {
    const cfg = this.cfg, r = cfg.radius;
    const face = [0, 1].map((w) => wrap(st.face[w] + clip(a[w][TURN], -1, 1) * (cfg.turn_rate * Math.PI / 180) * cfg.dt));
    let pos = [], vel = [];
    for (const w of [0, 1]) {
      let m = [clip(a[w][FWD], -1, 1), clip(a[w][SIDE], -1, 1)];
      const n = Math.max(norm(m[0], m[1]), 1);
      m = [m[0] / n, m[1] / n];
      m = [m[0] * (m[0] > 0 ? 1 : cfg.sideways), m[1] * cfg.sideways];
      let speed = a[w][RUN] > 0 ? cfg.run : cfg.walk;
      if (this.healthOn) speed *= st.hp[w] < cfg.wounded_below ? cfg.wounded_speed : 1;
      const want = rot(m, face[w]).map((x) => x * speed);
      let dv = [want[0] - st.vel[w][0], want[1] - st.vel[w][1]];
      const k = Math.min(1, (cfg.accel * cfg.dt) / Math.max(norm(dv[0], dv[1]), 1e-9));
      const v = [st.vel[w][0] + dv[0] * k, st.vel[w][1] + dv[1] * k];
      vel.push(v);
      pos.push([st.pos[w][0] + v[0] * cfg.dt, st.pos[w][1] + v[1] * cfg.dt]);
    }
    const gx = pos[1][0] - pos[0][0], gy = pos[1][1] - pos[0][1], dist = norm(gx, gy);
    const over = Math.max(2 * r - dist, 0), px = (gx / Math.max(dist, 1e-6)) * over / 2, py = (gy / Math.max(dist, 1e-6)) * over / 2;
    pos = [[pos[0][0] - px, pos[0][1] - py], [pos[1][0] + px, pos[1][1] + py]];
    const yard = [cfg.yard_w, cfg.yard_h], boxes = this.boxes(h, st);
    for (const w of [0, 1]) {
      const c = [0, 1].map((ax) => clip(pos[w][ax], r, yard[ax] - r));
      vel[w] = vel[w].map((v, ax) => (c[ax] !== pos[w][ax] ? 0 : v));
      [pos[w], vel[w]] = pushOut(c, vel[w], boxes, r);
    }
    return [pos, vel, face];
  }

  doors(h, st, a, pos) {
    const cfg = this.cfg, D = h.doors.length;
    const gap = (p, b) => norm(p[0] - clip(p[0], b[0], b[2]), p[1] - clip(p[1], b[1], b[3]));
    const near = [0, 0], start = [false, false], hitAny = new Array(D).fill(false);
    for (const w of [0, 1]) {
      let best = Infinity, k = 0;
      h.doors.forEach((b, d) => { const g = h.doorValid[d] ? gap(pos[w], b) : Infinity; if (g < best) { best = g; k = d; } });
      near[w] = k;
      start[w] = a[w][INTERACT] > 0 && best <= cfg.reach && st.doorTimer[k] === 0;
      if (start[w]) hitAny[k] = true;
    }
    const timer = h.doors.map((_, d) => Math.max((hitAny[d] ? this.doorSteps : st.doorTimer[d]) - 1, 0));
    const open = h.doors.map((b, d) => {
      const finishing = st.doorTimer[d] === 1 || (hitAny[d] && this.doorSteps === 1);
      const blocked = pos.some((p) => gap(p, b) < cfg.radius);
      return st.doorOpen[d] !== (finishing && !(st.doorOpen[d] && blocked));
    });
    const at = near.map((k) => [(h.doors[k][0] + h.doors[k][2]) / 2, (h.doors[k][1] + h.doors[k][3]) / 2]);
    return [open, timer, start, at];
  }

  hitChance(st, who, boxes) {
    const cfg = this.cfg, o = 1 - who;
    const dx = st.pos[o][0] - st.pos[who][0], dy = st.pos[o][1] - st.pos[who][1], dist = norm(dx, dy);
    const onAim = dx * Math.cos(st.face[who]) + dy * Math.sin(st.face[who]) >= this.cosAim * dist;
    let p = clip(cfg.hit_close - cfg.hit_falloff * dist, 0.05, 1.0);
    p *= 1 - cfg.move_penalty * Math.min(norm(...st.vel[who]) / cfg.walk, 1.0);
    p *= 1 - cfg.turn_penalty * Math.abs(st.turn[who]);
    if (this.healthOn) p *= st.hp[who] < cfg.wounded_below ? cfg.wounded_aim : 1;
    if (this.reactOn) p *= st.aim[who] / this.reactSteps;
    return onAim && this.sees(st, who, boxes) ? p : 0;
  }

  reload(st, a) {
    const cfg = this.cfg;
    for (const w of [0, 1]) {
      const timer = Math.max(st.reloadTimer[w] - 1, 0);
      if (st.reloadTimer[w] === 1) {
        const take = Math.min(cfg.magazine - st.mag[w], st.reserve[w]);
        st.mag[w] += take; st.reserve[w] -= take;
      }
      const want = a[w][RELOAD] > 0;
      const go = (want || st.mag[w] === 0) && st.mag[w] < cfg.magazine && st.reserve[w] > 0 && timer === 0 && st.alive[w] && !st.done;
      st.reloadTimer[w] = go ? this.reloadSteps : timer;
      st.reloadNow[w] = go;
      st.kick[w] = Math.max(st.kick[w] - cfg.dt / cfg.recoil_recover, 0);
    }
    return st;
  }

  shoot(h, st, a, dice, where) {
    const cfg = this.cfg, boxes = this.boxes(h, st);
    const fire = [0, 1].map((w) => a[w][FIRE] > 0 && a[w][RUN] <= 0 && st.alive[w] && !st.done
                                   && (!this.gunsOn || (st.mag[w] > 0 && st.reloadTimer[w] === 0)));
    const p = [0, 1].map((w) => this.hitChance(st, w, boxes) * (this.gunsOn ? 1 - cfg.recoil * st.kick[w] : 1));
    const hits = [0, 1].map((w) => fire[w] && dice[w] < p[w]);   // hits[w]: w hit the other one
    let alive;
    if (this.healthOn) {
      const dmg = where.map((u) => (u < cfg.head_share ? cfg.head_damage : u < cfg.head_share + cfg.torso_share ? cfg.torso_damage : cfg.limb_damage));
      const hp = [0, 1].map((w) => st.hp[w] - (hits[1 - w] ? dmg[1 - w] : 0));
      alive = [0, 1].map((w) => st.alive[w] && hp[w] > 0);
      st.hp = hp.map((x) => Math.max(x, 0));
      st.landed = [0, 1].map((w) => (hits[w] ? 0 : Math.min(st.landed[w] + 1, this.soundSteps)));
    } else {
      alive = [0, 1].map((w) => st.alive[w] && !hits[1 - w]);
    }
    const flash = [0, 1].map((w) => (fire[w] ? this.flashSteps : Math.max(st.flash[w] - 1, 0)));
    const seesFlash = [0, 1].map((w) => flash[1 - w] > 0 && clear(st.pos[w], st.pos[1 - w], boxes) && st.blind[w] === 0);
    st.mem = [0, 1].map((w) => (seesFlash[w] ? st.pos[1 - w].slice() : st.mem[w]));
    st.since = [0, 1].map((w) => (seesFlash[w] ? 0 : st.since[w]));
    Object.assign(st, { alive, fired: fire, flash, flashSeen: seesFlash, hits });
    if (this.gunsOn) {
      st.mag = st.mag.map((m, w) => m - (fire[w] ? 1 : 0));
      st.kick = st.kick.map((k, w) => (fire[w] ? 1 : k));
    }
    return st;
  }

  wallsBetween(walls, p, q) {
    let n = 0;
    for (const b of walls) {
      const [tIn, tOut] = slab(p[0], p[1], q[0] - p[0], q[1] - p[1], b);
      if (tIn <= tOut && tOut >= 0 && tIn <= 1) n++;
    }
    return n;
  }

  heard(me, q, range, walls) {  // [direction x, y (rounded to sound_dirs), loudness]
    const cfg = this.cfg, k = (2 * Math.PI) / cfg.sound_dirs, dx = q[0] - me[0], dy = q[1] - me[1];
    const reach = range * cfg.through_wall ** this.wallsBetween(walls, me, q);
    const loud = reach > 0 ? clip(1 - norm(dx, dy) / Math.max(reach, 1e-6), 0, 1) : 0;
    const ang = roundEven(Math.atan2(dy, dx) / k) * k;
    return [Math.cos(ang), Math.sin(ang), loud];
  }

  hear(h, st) {
    const cfg = this.cfg, walls = this.wallsOnly(h, st);
    const made = [0, 1].map((w) => {
      const speed = norm(...st.vel[w]);
      const steps = speed > cfg.walk + 0.1 ? cfg.hear_run : speed > cfg.still ? cfg.hear_walk : 0;
      return [steps, st.doorEvent[w] ? cfg.hear_door : 0, st.fired[w] ? cfg.hear_shot : 0].map((x) => x * (st.alive[w] ? 1 : 0));
    });
    for (const w of [0, 1]) {
      const me = st.pos[w], o = 1 - w;
      const d0 = st.doorAt[o], dd = Math.max(norm(me[0] - d0[0], me[1] - d0[1]), 1e-6);
      const door = [d0[0] + ((me[0] - d0[0]) / dd) * cfg.wall, d0[1] + ((me[1] - d0[1]) / dd) * cfg.wall];
      [st.pos[o], door, st.pos[o]].forEach((src, kind) => {
        const nu = this.heard(me, src, made[o][kind], walls);
        const age = Math.min(st.soundAge[w][kind] + 1, this.soundSteps);
        const old = age < this.soundSteps ? st.sound[w][kind] : [0, 0, 0];
        const take = nu[2] > 0 && nu[2] >= old[2];
        st.sound[w][kind] = take ? nu : old;
        st.soundAge[w][kind] = take ? 0 : age;
      });
    }
    if (!this.bangsOn) return st;
    const nb = [0, 1].map((w) => this.heard(st.pos[w], st.bangPos[1 - w], st.bangNow[1 - w] ? cfg.hear_bang : 0, walls));
    for (const w of [0, 1]) {
      const age = Math.min(st.bangAge[w] + 1, this.soundSteps), old = age < this.soundSteps ? st.bangHeard[w] : [0, 0, 0];
      st.bangHeard[w] = nb[w][2] > 0 ? nb[w] : old;
      st.bangAge[w] = nb[w][2] > 0 ? 0 : age;
    }
    return st;
  }

  hearReload(h, st) {
    const walls = this.wallsOnly(h, st);
    const nr = [0, 1].map((w) => this.heard(st.pos[w], st.pos[1 - w], st.reloadNow[1 - w] && st.alive[1 - w] ? this.cfg.hear_reload : 0, walls));
    for (const w of [0, 1]) {
      const age = Math.min(st.reloadAge[w] + 1, this.soundSteps), old = age < this.soundSteps ? st.reloadHeard[w] : [0, 0, 0];
      st.reloadHeard[w] = nr[w][2] > 0 ? nr[w] : old;
      st.reloadAge[w] = nr[w][2] > 0 ? 0 : age;
    }
    return st;
  }

  flashbangs(h, st, a) {
    const cfg = this.cfg, walls = this.wallsOnly(h, st);
    const timer = st.bangTimer.map((x) => Math.max(x - 1, 0)), now = st.bangTimer.map((x) => x === 1);
    let blind = st.blind.map((x) => Math.max(x - 1, 0));
    for (const src of [0, 1]) {
      const q = st.bangPos[src];
      blind = blind.map((b, w) => {
        const d = norm(st.pos[w][0] - q[0], st.pos[w][1] - q[1]);
        const steps = this.inCone(st.pos[w], st.face[w], q) ? this.blindSteps : Math.round(this.blindSteps * cfg.blind_away);
        return now[src] && d <= cfg.bang_range && clear(q, st.pos[w], walls) && st.alive[w] ? Math.max(b, steps) : b;
      });
    }
    for (const w of [0, 1]) {
      const thrown = a[w][THROW] > 0 && st.bangs[w] > 0 && timer[w] === 0 && st.alive[w] && !st.done;
      if (thrown) {
        const ahead = [Math.cos(st.face[w]), Math.sin(st.face[w])];
        const room = this.cast(st.pos[w], [ahead], walls)[0], dist = Math.max(Math.min(cfg.throw, room - 0.3), 0);
        st.bangs[w] -= 1;
        st.bangPos[w] = [st.pos[w][0] + ahead[0] * dist, st.pos[w][1] + ahead[1] * dist];
        timer[w] = this.fuseSteps;
      }
    }
    Object.assign(st, { bangTimer: timer, bangNow: now, blind });
    return st;
  }

  // One decision. actions: [assault, defense], 8 numbers each (see FWD..RELOAD). dice: two uniforms for the hits
  // (check.mjs passes the Python game's; the browser rolls them), where: two more for where hits land (health on).
  step(h, st, actions, dice = [Math.random(), Math.random()], where = [Math.random(), Math.random()]) {
    if (st.done) return st;
    const a = actions.map((x) => [...x, ...Array(8 - x.length).fill(-1)]);
    const prev = st;
    st = structuredClone(st);
    let [pos, vel, face] = this.move(h, prev, a);
    pos = pos.map((p, w) => (prev.alive[w] ? p : prev.pos[w].slice()));
    vel = vel.map((v, w) => (prev.alive[w] ? v : [0, 0]));
    const live = a.map((x, w) => x.map((v) => v * (prev.alive[w] ? 1 : 0)));
    const [open, timer, event, at] = this.doors(h, prev, live, pos);
    Object.assign(st, { pos, vel, face: face.map((f, w) => (prev.alive[w] ? f : prev.face[w])),
      turn: a.map((x, w) => clip(x[TURN], -1, 1) * (prev.alive[w] ? 1 : 0)), doorOpen: open, doorTimer: timer,
      doorEvent: event, doorAt: at.map((p, w) => (event[w] ? p : prev.doorAt[w])), t: prev.t + 1 });
    if (this.bangsOn) st = this.flashbangs(h, st, a);
    st = this.look(h, st);
    if (this.gunsOn) st = this.reload(st, a);
    st = this.hear(h, this.shoot(h, st, a, dice, where));
    if (this.gunsOn) st = this.hearReload(h, st);
    // the Python game keeps its state in float32: storing it the same way keeps long rounds together (check.mjs)
    const f = Math.fround;
    st.pos = st.pos.map((p) => p.map(f)); st.vel = st.vel.map((v) => v.map(f)); st.face = st.face.map(f);
    st.kick = st.kick.map(f); st.hp = st.hp.map(f);
    const aDown = !st.alive[ASSAULT], dDown = !st.alive[DEFENSE], timeout = st.t >= this.T && !aDown && !dDown;
    st.outcome = dDown && !aDown ? 1 : aDown && !dDown ? -1 : timeout ? -1 : 0;
    st.done = aDown || dDown || timeout;
    return st;
  }

  // What operator `who` knows (games/cqb.py observe): about -1..1, in its own frame. Nothing it can't see or hear.
  observe(h, st, who) {
    const cfg = this.cfg, me = who, o = 1 - who, p = st.pos[me], face = st.face[me];
    const own = (v) => rot(v, -face), sub = (a, b) => [a[0] - b[0], a[1] - b[1]], sc = (v, k) => [v[0] * k, v[1] * k];
    const boxes = this.boxes(h, st), obs = [];
    const lasers = this.cast(p, this.rayDirs.map((d) => rot(d, face)), boxes);
    obs.push(...lasers.map((x) => x / cfg.sight));
    const seen = st.seen[me], never = st.since[me] >= this.T, rel = sub(st.pos[o], p), dn = Math.max(norm(...rel), 1e-6);
    obs.push(seen ? 1 : 0, ...(seen ? sc(own(rel), 1 / cfg.sight) : [0, 0]), ...(seen ? sc(own(st.vel[o]), 1 / cfg.run) : [0, 0]),
             ...(never ? [0, 0] : sc(own(sub(st.mem[me], p)), 1 / cfg.sight)), Math.min(st.since[me] / this.T, 1),
             ...(st.flashSeen[me] ? own(sc(rel, 1 / dn)) : [0, 0]));
    for (const s of st.sound[me]) obs.push(...(s[2] > 0 ? own([s[0], s[1]]) : [0, 0]), s[2]);
    const flo = h.footLo, fhi = h.footHi;
    obs.push(Math.sin(face), Math.cos(face), ...sc(own(st.vel[me]), 1 / cfg.run),
             ...[0, 1].map((ax) => clip(((p[ax] - flo[ax]) / (fhi[ax] - flo[ax])) * 2 - 1, -2, 2)), st.turn[me]);
    const mid = h.doors.map((b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]);
    const dist = mid.map((m, d) => (h.doorValid[d] ? norm(m[0] - p[0], m[1] - p[1]) : Infinity));
    const near = dist.map((d, k) => [d, k]).sort((x, y) => x[0] - y[0] || x[1] - y[1]).slice(0, 2).map((x) => x[1]);
    for (const k of near) {
      const b = h.doors[k], axis = b[2] - b[0] <= b[3] - b[1] ? 0 : 1;   // doors are thin across their wall
      const side = p[axis] > mid[k][axis] ? 1 : 0;
      obs.push(...sc(own(sub(mid[k], p)), 0.1), st.doorOpen[k] ? 1 : 0, h.centerFed[k][side] ? 1 : 0);
    }
    obs.push(...st.cleared[me].map((c) => (c ? 1 : 0)), 1 - st.t / this.T);
    if (this.bangsOn) {
      const live = st.bangTimer[me] > 0, hb = st.bangHeard[me];
      obs.push(st.bangs[me] / Math.max(cfg.flashbangs[me], 1), st.blind[me] / this.blindSteps,
               ...(live ? sc(own(sub(st.bangPos[me], p)), 1 / cfg.sight) : [0, 0]), st.bangTimer[me] / this.fuseSteps,
               ...(hb[2] > 0 ? own([hb[0], hb[1]]) : [0, 0]), hb[2]);
    }
    if (this.gunsOn) {
      const rh = st.reloadHeard[me];
      obs.push(st.mag[me] / cfg.magazine, st.reserve[me] / Math.max(cfg.magazine * cfg.spare_mags, 1),
               st.reloadTimer[me] / this.reloadSteps, st.kick[me], ...(rh[2] > 0 ? own([rh[0], rh[1]]) : [0, 0]), rh[2]);
    }
    if (this.healthOn) obs.push(st.hp[me] / cfg.health, st.landed[me] < this.soundSteps ? 1 : 0);
    if (this.reactOn) obs.push(st.aim[me] / this.reactSteps);
    return obs;
  }

  // For drawing: how far each ray of `who`'s light fan reaches.
  lightOf(h, st, who) { return this.light(st.pos[who], st.face[who], this.boxes(h, st)); }
}

export class Brain {  // two tanh layers, then its best guess: 3 moves (clipped) and 5 switches (on above 0)
  constructor(layers, weights) {
    this.l = {};
    for (const { name, shape, offset } of layers) {
      const n = shape.reduce((a, b) => a * b, 1);
      this.l[name] = { shape, data: weights.subarray(offset, offset + n) };
    }
    this.nIn = this.l["body.0.weight"].shape[1];
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
    const h1 = Brain.dense(L["body.0.weight"], L["body.0.bias"], obs.slice(0, this.nIn), true);
    const h2 = Brain.dense(L["body.2.weight"], L["body.2.bias"], h1, true);
    const out = Brain.dense(L["policy.weight"], L["policy.bias"], h2, false);
    return [...[0, 1, 2].map((k) => clip(out[k], -1, 1)), ...[...out].slice(3).map((x) => (x > 0 ? 1 : -1))];
  }
}
