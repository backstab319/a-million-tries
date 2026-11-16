// The ecosystem of episode 17, ported line by line from src/ailearns/games/ecosystem.py (the land, the plants, the
// weather), ecosystem_life.py (bodies, brains, births, deaths) and ecosystem_space.py (who sees whom).
// check.mjs replays Python's own dice through this file and compares the worlds tick by tick; keep them in step.
// No drawing here: app.js draws.

export const WATER = 0, DESERT = 1, GRASS = 2, FOREST = 3;
export const N_IN = 33, N_HID = 8, N_OUT = 8;
export const N_W = N_IN * N_HID + N_HID + N_HID * N_OUT + N_OUT;   // 344
export const DAY_REAL_S = 86400.0;
const PI = Math.PI, TWO_PI = 2 * Math.PI, DEG = Math.PI / 180.0;
const SECTORS_DEG = [-60.0, -20.0, 20.0, 60.0];

export const WORLD = {
  size: 128, cell_m: 2.0, dt_days: 0.25, year_days: 360.0,
  plant_kj_m2: 5000.0, growth_per_day: 0.05, seed_per_day: 0.02, dieback_per_day: 0.01,
  water_frac: 0.08, desert_frac: 0.15, forest_frac: 0.25,
  capacity: [0.0, 0.1, 1.0, 2.0], root_ratio: [0.0, 1.0, 2.0, 0.25],
  regrow_per_day: 0.05, root_loss_per_day: 0.002, stubble: 0.1,
  dead_quality: 0.65, dead_decay_per_day: 0.004, start_fill: 0.5, patch_m: 64.0,
  temp_winter_c: 0.0, temp_summer_c: 20.0,
  fruit_trees_ha: [0.0, 0.0, 1.0, 10.0], fruit_kj: 220000.0, fruit_days: [200.0, 300.0], crown_m: 4.0,
  fruit_rot_per_day: 0.15, rot_q10: 2.5,
  rain: true, storms_ha_day: 0.1, storm_m: 30.0, storm_rain: 0.8, belt_share: 0.8, belt_m: 40.0, belt_loop: 0.3,
  belt_per_year: 2.0, dry_days: 15.0, wet_full: 0.4, drought_per_day: 0.02,
};

export const LIFE = {
  day_s: 60.0, dt: 1.0, max_creatures: 6000, ceiling: 100000,
  bmr_kj_day: 293.0, cot_j_m: 10.7, aerobic_scope: 10.0, body_kj_kg: 7000.0, build_overhead: 0.3, fat_kj_kg: 37000.0,
  digest_plant: 0.5, plant_curve: 2.0,
  cold: true, body_c: 37.0, conduct_ml_o2: 1.02, conduct_exp: -0.505, j_per_ml_o2: 20.1,
  cold_breath: true, breath_water_per_kj: 0.03e-3,
  gut: true, gut_contents: 0.4, gut_contents_dig: 0.9, gut_fibre_max: 0.75, gut_fibre_k_h: 0.05, gut_mrt_h: 60.0,
  gut_ref_kg: 600.0, gut_me_de: 0.82,
  digest_meat: 0.85, digest_meat_grazer: 0.6, meat_rot_per_day: 0.2, rot_q10: 2.5, reach_m: 1.5,
  sexual: true, mate_gap: 0.25, alone_wait_days: 1.0, alone: [0.0, 1.0], sigma_alone: 0.03,
  call_bmr: 2.0, sound_cell_m: 8.0, digest_fruit: 0.8, digest_fruit_hunter: 0.5, fruit_sense_kj: 2000.0, ear_m: 4.0,
  weapon: 1.0, milk: true, wean_frac: 0.4, milk_eff: 0.7, milk_floor: 0.3, carry: true,
  thirst: true, water_per_kj: 0.2e-3, water_green: 0.18e-3, water_dead: 0.007e-3, water_meat: 0.1e-3,
  water_fruit: 0.39e-3, water_milk: 0.3e-3, dry_max_l_kg: 0.15, puddle_wet: 0.6,
  fight_per_s: 0.5, growth_a: 1.0, life_days: 300.0, eat_bmr_day: 12.0, turn_rad_s: 2.0, sight_m: 6.0,
  sight_deg: 40.0, grow_above: 0.3, mass_kg: [0.5, 8.0], birth_frac: [0.05, 0.3], fat_frac: [0.05, 0.4],
  diet: [0.0, 1.0], sigma_diet: 0.03, start_fill: 0.5, sigma_w: 0.1, sigma_gene: 0.05,
};

// numpy's float remainder (np.remainder / Python %): the sign of the divisor
export function pymod(x, m) {
  let r = x % m;
  if (r !== 0) { if ((m < 0) !== (r < 0)) r += m; } else r = m < 0 ? -0 : 0;
  return r;
}
const sum = (a) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k]; return s; };

// ---- dice ------------------------------------------------------------------------------------------------------------
// A seeded generator with numpy's calls (not numpy's numbers): xoshiro128** seeded by splitmix32.
export class Rng {
  constructor(seed) {
    const seeds = Array.isArray(seed) ? seed : [seed];
    let x = 0x9e3779b9;
    for (const s of seeds) x = (x ^ Math.imul((s | 0) + 0x7f4a7c15, 0x85ebca6b)) >>> 0;
    const sm = () => { x = (x + 0x9e3779b9) >>> 0; let z = x; z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35); return (z ^ (z >>> 16)) >>> 0; };
    this.s = [sm(), sm(), sm(), sm()];
    this.spare = null;
  }
  u32() {
    const s = this.s, r = Math.imul(((Math.imul(s[1], 5) << 7) | (Math.imul(s[1], 5) >>> 25)), 9) >>> 0;
    const t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = (s[3] << 11) | (s[3] >>> 21);
    return r;
  }
  one() { return ((this.u32() >>> 5) * 67108864 + (this.u32() >>> 6)) / 9007199254740992; }
  gauss() {
    if (this.spare !== null) { const g = this.spare; this.spare = null; return g; }
    let u, v, s;
    do { u = 2 * this.one() - 1; v = 2 * this.one() - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const f = Math.sqrt(-2 * Math.log(s) / s);
    this.spare = v * f;
    return u * f;
  }
  random(k) { const a = new Float64Array(k); for (let j = 0; j < k; j++) a[j] = this.one(); return a; }
  normal(loc, scale, k) { const a = new Float64Array(k); for (let j = 0; j < k; j++) a[j] = loc + scale * this.gauss(); return a; }
  uniform(lo, hi, k) { const a = new Float64Array(k); for (let j = 0; j < k; j++) a[j] = lo + (hi - lo) * this.one(); return a; }
  poisson(lam) { const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= this.one(); } while (p > L); return k - 1; }
  choice(values, k, p = null) {
    const a = new Float64Array(k);
    let cum = null;
    if (p) { cum = new Float64Array(p.length); let c = 0; for (let j = 0; j < p.length; j++) { c += p[j]; cum[j] = c; } }
    for (let j = 0; j < k; j++) {
      let at;
      if (cum) { const r = this.one() * cum[cum.length - 1]; at = 0; while (at < cum.length - 1 && cum[at] <= r) at++; }
      else at = Math.floor(this.one() * values.length);
      a[j] = values[at];
    }
    return a;
  }
  permutation(values) {
    const a = Float64Array.from(values);
    for (let j = a.length - 1; j > 0; j--) { const r = Math.floor(this.one() * (j + 1)); const t = a[j]; a[j] = a[r]; a[r] = t; }
    return a;
  }
}

// Python's dice, replayed in order (check.mjs): each call must ask for the same thing Python asked for.
export class ReplayRng {
  constructor(entries, buf) { this.entries = entries; this.buf = buf; this.at = 0; }
  take(name, k) {
    const e = this.entries[this.at++];
    if (!e) throw new Error(`replay: ran out of dice (wanted ${name} x${k})`);
    const [, nm, off, len] = e;
    if (nm !== name || (k !== null && len !== k)) throw new Error(`replay: wanted ${name} x${k}, Python drew ${nm} x${len} (#${this.at - 1})`);
    return this.buf.subarray(off, off + len);
  }
  random(k) { return this.take("random", k); }
  normal(loc, scale, k) { return this.take("normal", k); }
  uniform(lo, hi, k) { return this.take("uniform", k); }
  poisson() { return this.take("poisson", 1)[0]; }
  choice(values, k) { return this.take("choice", k); }
  permutation(values) { return this.take("permutation", values.length); }
}

// ---- the land, the plants, the weather (ecosystem.py) ----------------------------------------------------------------
function smoothNoise(rng, n, scales, weights) {
  const out = new Float64Array(n * n);
  for (let q = 0; q < scales.length; q++) {
    const s = scales[q], w = weights[q], g = rng.random((s + 1) * (s + 1));
    const ix = new Int32Array(n), fx = new Float64Array(n);
    for (let j = 0; j < n; j++) { const x = j * (s / n); ix[j] = Math.trunc(x); fx[j] = x - ix[j]; }
    const rows = new Float64Array(n * (s + 1));
    for (let r = 0; r < n; r++) for (let c = 0; c <= s; c++)
      rows[r * (s + 1) + c] = g[ix[r] * (s + 1) + c] * (1 - fx[r]) + g[(ix[r] + 1) * (s + 1) + c] * fx[r];
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++)
      out[r * n + c] += w * (rows[r * (s + 1) + ix[c]] * (1 - fx[c]) + rows[r * (s + 1) + ix[c] + 1] * fx[c]);
  }
  const order = Array.from(out.keys()).sort((a, b) => out[a] - out[b] || a - b);
  const ranks = new Float64Array(n * n);
  for (let k = 0; k < order.length; k++) ranks[order[k]] = k / (n * n);
  return ranks;
}

function quantile(sorted, q) {   // numpy's linear method (with its lerp)
  const v = q * (sorted.length - 1), lo = Math.floor(v), t = v - lo;
  const a = sorted[lo], b = sorted[Math.min(lo + 1, sorted.length - 1)], d = b - a;
  return t >= 0.5 ? b - d * (1 - t) : a + d * t;
}

export class World {
  constructor(cfg, g) {
    this.cfg = cfg;
    this.n = cfg.size;
    const nn = this.n * this.n;
    this.biome = g.biome; this.capacity = g.capacity; this.plants = g.plants; this.roots = g.roots;
    this.rootCap = g.root_capacity;
    this.dead = new Float64Array(nn); this.fruit = new Float64Array(nn); this.wet = new Float64Array(nn).fill(0.5);
    this.trees = g.trees; this.rng = g.rng; this.day = 0.0;
    this.ledger = { start: sum(this.plants) + sum(this.roots), sun_in: 0.0, eaten: 0.0, dieback: 0.0 };
    this._crown = null;
  }

  // rngs: { land, trees, weather } (each an Rng or a ReplayRng); by default seeded from `seed`
  static make(seed, cfg = WORLD, rngs = null) {
    rngs = rngs || { land: new Rng([seed, 1]), trees: new Rng([seed, 7]), weather: new Rng([seed, 9]) };
    const n = cfg.size, nn = n * n, rng = rngs.land;
    const coarse = Math.max(Math.round(n * cfg.cell_m / cfg.patch_m), 1);
    const scales = [coarse, 2 * coarse, 4 * coarse], weights = [0.6, 0.3, 0.1];
    const height = smoothNoise(rng, n, scales, weights), moisture = smoothNoise(rng, n, scales, weights);
    const soil = smoothNoise(rng, n, scales, weights);
    const biome = new Int8Array(nn).fill(GRASS);
    const landM = [];
    for (let c = 0; c < nn; c++) if (height[c] >= cfg.water_frac) landM.push(moisture[c]);
    landM.sort((a, b) => a - b);
    const lo = quantile(landM, cfg.desert_frac), hi = quantile(landM, 1 - cfg.forest_frac);
    for (let c = 0; c < nn; c++) {
      const land = height[c] >= cfg.water_frac;
      if (land && moisture[c] < lo) biome[c] = DESERT;
      if (land && moisture[c] >= hi) biome[c] = FOREST;
      if (!land) biome[c] = WATER;
    }
    const capacity = new Float64Array(nn), plants = new Float64Array(nn), rc = new Float64Array(nn), roots = new Float64Array(nn);
    for (let c = 0; c < nn; c++) {
      const fert = 0.5 + 0.5 * soil[c];
      capacity[c] = cfg.plant_kj_m2 * cfg.cell_m ** 2 * fert * cfg.capacity[biome[c]];
      plants[c] = capacity[c] * cfg.start_fill;
      rc[c] = capacity[c] * cfg.root_ratio[biome[c]];
      roots[c] = rc[c] * cfg.start_fill;
    }
    const ha = (cfg.cell_m ** 2) / 10000, t = rngs.trees.random(nn), trees = [];
    for (let c = 0; c < nn; c++) if (t[c] < cfg.fruit_trees_ha[biome[c]] * ha) trees.push([Math.floor(c / n), c % n]);
    return new World(cfg, { biome, capacity, plants, roots, root_capacity: rc, trees, rng: rngs.weather });
  }

  totalPlants() { return sum(this.plants) + sum(this.roots) + sum(this.dead) + sum(this.fruit); }
  season() { return 0.5 - 0.5 * Math.cos(2 * PI * this.day / this.cfg.year_days); }
  temperature() { const c = this.cfg; return c.temp_winter_c + (c.temp_summer_c - c.temp_winter_c) * this.season(); }

  crown() {
    if (!this._crown) {
      const cfg = this.cfg, n = this.n, r = Math.ceil(cfg.crown_m / cfg.cell_m), off = [];
      for (let a = -r; a <= r; a++) for (let b = -r; b <= r; b++)
        if ((a * a + b * b) * cfg.cell_m ** 2 <= cfg.crown_m ** 2) off.push([a, b]);
      const cells = [], share = [];
      for (const [tr, tc] of this.trees) {
        const mine = [];
        for (const [a, b] of off) {
          const rr = tr + a, cc = tc + b;
          if (rr >= 0 && rr < n && cc >= 0 && cc < n && this.biome[rr * n + cc] !== WATER) mine.push(rr * n + cc);
        }
        for (const c of mine) { cells.push(c); share.push(1.0 / mine.length); }
      }
      this._crown = [Int32Array.from(cells), Float64Array.from(share)];
    }
    return this._crown;
  }

  crownCells() {   // fruit only ever falls here: the crown cells, sorted and unique
    if (!this._crownCells) this._crownCells = Int32Array.from(new Set(this.crown()[0])).sort();
    return this._crownCells;
  }

  fruitFall(day, dt) {
    const [a, b] = this.cfg.fruit_days, y = pymod(day, this.cfg.year_days);
    if (!(a <= y && y < b)) return 0.0;
    return this.cfg.fruit_kj * PI / (2 * (b - a)) * Math.sin(PI * (y - a) / (b - a)) * dt;
  }

  belt() {
    const cfg = this.cfg, L = cfg.size * cfg.cell_m, a = 2 * PI * cfg.belt_per_year * this.day / cfg.year_days;
    return [L / 2 + cfg.belt_loop * L * Math.cos(a), L / 2 + cfg.belt_loop * L * Math.sin(a)];
  }

  // the soil dries; storms fall and wet it -> how wet each cell grows, 0..1
  weather(dt) {
    const cfg = this.cfg, n = this.n, L = n * cfg.cell_m, wet = this.wet, f = Math.exp(-dt / cfg.dry_days);
    for (let c = 0; c < wet.length; c++) wet[c] *= f;
    const k = this.rng.poisson(cfg.storms_ha_day * L * L / 10000 * dt);
    this.storms = [];
    if (k) {
      const [bx, by] = this.belt();
      const front = this.rng.random(k), xn = this.rng.normal(bx, cfg.belt_m, k), xu = this.rng.uniform(0, L, k);
      const yn = this.rng.normal(by, cfg.belt_m, k), yu = this.rng.uniform(0, L, k);
      const gx = new Float64Array(n), gy = new Float64Array(n);
      for (let s = 0; s < k; s++) {
        const fr = front[s] < cfg.belt_share, cx = fr ? xn[s] : xu[s], cy = fr ? yn[s] : yu[s];
        this.storms.push([cx, cy]);
        for (let j = 0; j < n; j++) {
          const c = (j + 0.5) * cfg.cell_m, ux = (c - cx) / cfg.storm_m, uy = (c - cy) / cfg.storm_m;
          gx[j] = Math.exp(-(ux * ux) / 2); gy[j] = Math.exp(-(uy * uy) / 2);
        }
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) wet[r * n + c] += cfg.storm_rain * (gx[r] * gy[c]);
      }
      for (let c = 0; c < wet.length; c++) wet[c] = Math.min(wet[c], 1.0);
    }
    const g = new Float64Array(wet.length);
    for (let c = 0; c < wet.length; c++) g[c] = Math.min(wet[c] / cfg.wet_full, 1.0);
    return g;
  }

  step() {
    const cfg = this.cfg, n = this.n, B = this.plants, K = this.capacity, R = this.roots, RK = this.rootCap;
    const s = this.season(), dt = cfg.dt_days;
    const g = cfg.rain ? this.weather(dt) : null;
    const nb = new Float64Array(n * n);
    for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
      const up = r > 0 ? B[(r - 1) * n + c] : 0, dn = r < n - 1 ? B[(r + 1) * n + c] : 0;
      const lf = c > 0 ? B[r * n + c - 1] : 0, rt = c < n - 1 ? B[r * n + c + 1] : 0;
      nb[r * n + c] = (up + dn + lf + rt) / 4;
    }
    let sunIn = 0, rootDie = 0, rotSum = 0;
    for (let c = 0; c < n * n; c++) {
      const gc = g ? g[c] : 1.0, k = K[c], b = B[c], rr = R[c], rk = RK[c];
      const room = k > 0 ? (k - b) / k : 0, rootRoom = rk > 0 ? (rk - rr) / rk : 0;
      const both = Math.max(room, rootRoom);
      let grow = gc * s * dt * (cfg.growth_per_day * b + cfg.seed_per_day * nb[c]) * both;
      let toRoot = (k + rk > 0 ? rk / (k + rk) : 0) * grow;
      toRoot = Math.min(toRoot, rk - rr);
      const toShoot = Math.min(grow - toRoot, k - b);
      toRoot = Math.min(grow - toShoot, rk - rr);
      grow = toShoot + toRoot;
      const up = Math.min(gc * s * dt * cfg.regrow_per_day * rr * room, k - b - toShoot);
      let die = (1 - s) * dt * cfg.dieback_per_day * b;
      if (cfg.rain) die = Math.min(die + (1 - gc) * dt * cfg.drought_per_day * b, b);
      const rd = (2 - s) / 1.5 * dt * cfg.root_loss_per_day * rr;
      const rot = this.dead[c] * (1 - Math.exp(-cfg.dead_decay_per_day * dt));
      B[c] = b + toShoot + up - die;
      R[c] = rr + toRoot - up - rd;
      this.dead[c] = this.dead[c] + die - rot;
      sunIn += grow; rootDie += rd; rotSum += rot;
    }
    this.ledger.sun_in += sunIn;
    this.ledger.dieback += rootDie + rotSum;
    if (this.trees.length) {
      const drop = this.fruitFall(this.day, dt);
      if (drop > 0) {
        const [cells, share] = this.crown();
        for (let j = 0; j < cells.length; j++) this.fruit[cells[j]] += drop * share[j];
        this.ledger.sun_in += drop * this.trees.length;
      }
      const rate = cfg.fruit_rot_per_day * cfg.rot_q10 ** ((this.temperature() - cfg.temp_summer_c) / 10);
      const f = 1 - Math.exp(-rate * dt);
      let gone = 0;
      for (let c = 0; c < n * n; c++) { const x = this.fruit[c] * f; this.fruit[c] -= x; gone += x; }
      this.ledger.dieback += gone;
    }
    this.day += dt;
  }

  // eaters at flat cells each try to eat `want` kJ: green above the stubble first, then dead grass -> [green, dead]
  graze(cells, want) {
    const P = this.plants, K = this.capacity, st = this.cfg.stubble, D = this.dead;
    const green = take(P, (c) => Math.max(P[c] - st * K[c], 0.0), cells, want);
    const rest = new Float64Array(want.length);
    for (let j = 0; j < want.length; j++) rest[j] = want[j] - green[j];
    const dead = take(D, (c) => D[c], cells, rest);
    this.ledger.eaten += sum(green) + sum(dead);
    return [green, dead];
  }

  eatFruit(cells, want) {
    const F = this.fruit, got = take(F, (c) => F[c], cells, want);
    this.ledger.eaten += sum(got);
    return got;
  }
}

// take up to reach(c) from `pool` (in place) for eaters asking `want` on `cells`, shared pro rata. Only the cells
// asked from are touched: numpy clamps every cell at 0 on each take, but nothing else ever goes below 0 (checked over
// 200 days), so that's the same; reach(c) is read before the cell is taken from (as numpy's copy).
const ASKED = { a: new Float64Array(0), share: new Float64Array(0) };
function take(pool, reach, cells, want) {
  if (ASKED.a.length < pool.length) { ASKED.a = new Float64Array(pool.length); ASKED.share = new Float64Array(pool.length); }
  const asked = ASKED.a, share = ASKED.share, touched = [];
  for (let j = 0; j < cells.length; j++) { const c = cells[j]; if (asked[c] === 0 && share[c] !== -1) { touched.push(c); share[c] = -1; } asked[c] += want[j]; }
  for (const c of touched) {
    const a = asked[c];
    share[c] = a > 0 ? Math.min(reach(c), a) / a : 0;
    if (a > 0) pool[c] -= a * share[c];
    pool[c] = Math.max(pool[c], 0.0);
  }
  const got = new Float64Array(cells.length);
  for (let j = 0; j < cells.length; j++) got[j] = want[j] * share[cells[j]];
  for (const c of touched) { asked[c] = 0; share[c] = 0; }
  return got;
}

// ---- who sees whom (ecosystem_space.py, the compiled version) -------------------------------------------------------
export function nearestBySector(px, py, hd, radius, size, edges = SECTORS_DEG) {
  const n = px.length, nb = Math.max(Math.ceil(size / radius), 1);
  const bx = new Int32Array(n), by = new Int32Array(n), count = new Int32Array(nb * nb + 1);
  for (let k = 0; k < n; k++) {
    const x = Math.min(Math.max(Math.floor(px[k] / radius), 0), nb - 1);
    const y = Math.min(Math.max(Math.floor(py[k] / radius), 0), nb - 1);
    bx[k] = x; by[k] = y; count[x * nb + y + 1]++;
  }
  for (let c = 0; c < nb * nb; c++) count[c + 1] += count[c];
  const fill = count.slice(0, nb * nb), member = new Int32Array(n);
  for (let k = 0; k < n; k++) { const c = bx[k] * nb + by[k]; member[fill[c]++] = k; }
  const near = new Int32Array(n).fill(-1), nearD = new Float64Array(n).fill(Infinity);
  const secB = new Int32Array(3 * n).fill(-1), secD = new Float64Array(3 * n).fill(Infinity);
  const r2 = radius * radius;
  for (let a = 0; a < n; a++) {
    const xa = px[a], ya = py[a];
    for (let ox = bx[a] - 1; ox <= bx[a] + 1; ox++) {
      if (ox < 0 || ox >= nb) continue;
      for (let oy = by[a] - 1; oy <= by[a] + 1; oy++) {
        if (oy < 0 || oy >= nb) continue;
        const c = ox * nb + oy;
        for (let m = count[c]; m < count[c + 1]; m++) {
          const b = member[m];
          if (b === a) continue;
          const dx = px[b] - xa, dy = py[b] - ya, d2 = dx * dx + dy * dy;
          if (d2 >= r2) continue;
          const d = Math.sqrt(d2);
          if (d < nearD[a] || (d === nearD[a] && b < near[a])) { near[a] = b; nearD[a] = d; }
          const rel = (pymod(Math.atan2(dy, dx) - hd[a] + PI, TWO_PI) - PI) * (180 / PI);
          let s = -1;
          for (let e = 0; e < 3; e++) if (edges[e] <= rel && rel < edges[e + 1]) s = e;
          if (s >= 0 && (d < secD[3 * a + s] || (d === secD[3 * a + s] && b < secB[3 * a + s]))) { secB[3 * a + s] = b; secD[3 * a + s] = d; }
        }
      }
    }
  }
  return { near, nearD, secB, secD };
}

// ---- bodies and brains (ecosystem_life.py) --------------------------------------------------------------------------
export function plantKeep(cfg, diet, mass) {
  const curve = 1 - diet ** cfg.plant_curve;
  if (!cfg.gut) return cfg.digest_plant * curve;
  const hours = cfg.gut_mrt_h * (mass / cfg.gut_ref_kg) ** 0.25;
  const fibre = cfg.gut_fibre_max * (1 - Math.exp(-cfg.gut_fibre_k_h * hours));
  return curve * cfg.gut_me_de * (cfg.gut_contents * cfg.gut_contents_dig + (1 - cfg.gut_contents) * fibre);
}

export function keepWarmS(cfg, mass, airC, restS) {
  if (!cfg.cold) return 0.0 * restS;
  const g = mass * 1000;
  const cDay = cfg.conduct_ml_o2 * g ** (1 + cfg.conduct_exp) * cfg.j_per_ml_o2 * 24 / 1000;
  return Math.max(cDay * (cfg.body_c - airC) / cfg.day_s - restS, 0.0);
}

const PER = { alive: Uint8Array, px: Float64Array, py: Float64Array, heading: Float64Array, mass: Float64Array,
  adult: Float64Array, birth: Float64Array, fat: Float64Array, diet: Float64Array, reserve: Float64Array,
  age: Float64Array, gen: Int32Array, uid: Float64Array, parent: Float64Array, calling: Uint8Array,
  mem0: Float64Array, mem1: Float64Array, parent2: Float64Array, alone: Float64Array, dry: Float64Array,
  drank: Uint8Array };

export class Life {
  // rng: an Rng or a ReplayRng
  constructor(world, n, rng, cfg = LIFE) {
    this.world = world; this.cfg = cfg; this.rng = rng;
    this.realSPerS = DAY_REAL_S / cfg.day_s;
    this.t = 0.0; this.ticks = 0;
    this.plantEvery = Math.round(world.cfg.dt_days * cfg.day_s / cfg.dt);
    this.cap = 0;
    this._alloc(cfg.max_creatures);
    this.meat = new Float64Array(world.n * world.n);
    this.nextUid = 0;
    this.books = { start: 0, heat: 0, dung: 0, decay: 0, kills: 0, births: 0, births_alone: 0, deaths: 0, full: 0 };
    this.causes = { killed: 0, thirst: 0, starved: 0, old: 0 };   // (the page's count; not in the Python books)
    this._shore = null;
    if (n > 0) this.add(n);
  }

  _alloc(m) {
    const old = this.cap;
    for (const [k, T] of Object.entries(PER)) {
      const a = new T(m);
      if (["uid", "parent", "parent2"].includes(k)) a.fill(-1);
      if (old) a.set(this[k]);
      this[k] = a;
    }
    const w = new Float64Array(m * N_W);
    if (old) w.set(this.w);
    this.w = w;
    this.cap = m;
  }

  count() { let c = 0; for (let k = 0; k < this.cap; k++) c += this.alive[k]; return c; }
  bmrS(m) { return this.cfg.bmr_kj_day * m ** 0.75 / this.cfg.day_s; }
  walkKjM(c) { return this.cfg.cot_j_m * c ** 0.684 / 1000; }
  sprintKjS(lean) { return this.cfg.aerobic_scope * this.bmrS(lean) / this.realSPerS; }
  topSpeed(lean, carried) { return this.sprintKjS(lean) / this.walkKjM(carried); }
  reserveCap(lean, fat) { return fat * this.cfg.fat_kj_kg * lean; }
  carried(k) { return this.mass[k] + this.reserve[k] / this.cfg.fat_kj_kg; }
  lifespan(adult) { return this.cfg.life_days * (adult * 1000) ** 0.25 / 10; }
  energy() { let e = 0; for (let k = 0; k < this.cap; k++) if (this.alive[k]) e += this.reserve[k] + this.cfg.body_kj_kg * this.mass[k]; return e; }

  _grow(need) {
    const m = this.cap, free = m - this.count();
    let nw = m;
    while (free + nw - m < need && nw < this.cfg.ceiling) nw = Math.min(2 * nw, this.cfg.ceiling);
    if (nw !== m) this._alloc(nw);
  }
  _free() { const f = []; for (let k = 0; k < this.cap; k++) if (!this.alive[k]) f.push(k); return f; }

  _cell(x, y) {   // -> the flat cell (clipped to the map); this._in: whether the point is on the map
    const n = this.world.n, cm = this.world.cfg.cell_m;
    let r = Math.floor(x / cm), c = Math.floor(y / cm);
    this._in = r >= 0 && r < n && c >= 0 && c < n;
    r = Math.min(Math.max(r, 0), n - 1); c = Math.min(Math.max(c, 0), n - 1);
    return r * n + c;
  }

  genes(idx) {
    const g = { adult: [], birth: [], fat: [], alone: [], diet: [], w: new Float64Array(idx.length * N_W) };
    idx.forEach((k, j) => {
      for (const f of ["adult", "birth", "fat", "alone", "diet"]) g[f].push(this[f][k]);
      g.w.set(this.w.subarray(k * N_W, (k + 1) * N_W), j * N_W);
    });
    return g;
  }

  // k new creatures on random land: random ones (diet range `diet`), or (`like`, a genes() dict) mutated copies
  add(k, diet = null, like = null, p = null) {
    const cfg = this.cfg, rng = this.rng, w = this.world, n = w.n;
    this._grow(k);
    const idx = this._free().slice(0, k);
    k = idx.length;
    const lo = Math.log(cfg.mass_kg[0]), hi = Math.log(cfg.mass_kg[1]);
    const land = [];
    for (let c = 0; c < n * n; c++) if (w.biome[c] !== WATER) land.push(c);
    const c = rng.choice(land, k), u = rng.random(2 * k);
    const hd = rng.uniform(-PI, PI, k), ad = rng.uniform(lo, hi, k);
    const bf = rng.uniform(cfg.birth_frac[0], cfg.birth_frac[1], k), ff = rng.uniform(cfg.fat_frac[0], cfg.fat_frac[1], k);
    const al = rng.uniform(cfg.alone[0], cfg.alone[1], k);
    const dr = diet || cfg.diet, di = rng.uniform(dr[0], dr[1], k);
    const ws = rng.normal(0, 1, k * N_W);
    idx.forEach((s, j) => {
      this.alive[s] = 1;
      this.px[s] = (Math.floor(c[j] / n) + u[2 * j]) * w.cfg.cell_m;
      this.py[s] = (c[j] % n + u[2 * j + 1]) * w.cfg.cell_m;
      this.heading[s] = hd[j]; this.adult[s] = Math.exp(ad[j]); this.mass[s] = this.adult[s];
      this.birth[s] = bf[j]; this.fat[s] = ff[j]; this.alone[s] = al[j]; this.diet[s] = di[j];
      this.gen[s] = 0; this.parent[s] = -1;
      this.reserve[s] = cfg.start_fill * this.reserveCap(this.mass[s], this.fat[s]);
      this.age[s] = 0.0;
      this.w.set(ws.subarray(j * N_W, (j + 1) * N_W), s * N_W);
      this.calling[s] = 0; this.mem0[s] = 0; this.mem1[s] = 0; this.parent2[s] = -1; this.dry[s] = 0.0;
    });
    if (like) {
      const src = rng.choice(Array.from(like.diet.keys()), k, p);
      const e1 = rng.normal(0, cfg.sigma_gene, k), e2 = rng.normal(0, cfg.sigma_gene, k), e3 = rng.normal(0, cfg.sigma_gene, k);
      const na = rng.normal(0, cfg.sigma_alone, k), nd = rng.normal(0, cfg.sigma_diet, k);
      const nw = rng.normal(0, cfg.sigma_w, k * N_W);
      const clip = (x, a, b) => Math.min(Math.max(x, a), b);
      idx.forEach((s, j) => {
        const q = src[j];
        this.adult[s] = like.adult[q] * Math.exp(e1[j]); this.mass[s] = this.adult[s];
        this.birth[s] = clip(like.birth[q] * Math.exp(e2[j]), 0.01, 0.9);
        this.fat[s] = clip(like.fat[q] * Math.exp(e3[j]), 0.01, 1.0);
        this.alone[s] = clip(like.alone[q] + na[j], 0.0, 1.0);
        this.diet[s] = clip(like.diet[q] + nd[j], 0.0, 1.0);
        for (let o = 0; o < N_W; o++) this.w[s * N_W + o] = like.w[q * N_W + o] + nw[j * N_W + o];
        this.reserve[s] = cfg.start_fill * this.reserveCap(this.mass[s], this.fat[s]);
      });
    }
    let e = 0;
    idx.forEach((s, j) => { this.uid[s] = this.nextUid + j; e += this.reserve[s] + cfg.body_kj_kg * this.mass[s]; });
    this.nextUid += k;
    this.books.start += e;
    return idx;
  }

  // (5a, 7b) something in the air, on the coarse grid, softened `passes` times
  _air(xs, ys, amount, passes) {
    const cfg = this.cfg, w = this.world, n = Math.max(Math.round(w.n * w.cfg.cell_m / cfg.sound_cell_m), 1);
    let f = new Float64Array(n * n);
    if (xs.length) {
      for (let j = 0; j < xs.length; j++) {
        const r = Math.min(Math.max(Math.trunc(xs[j] / cfg.sound_cell_m), 0), n - 1);
        const c = Math.min(Math.max(Math.trunc(ys[j] / cfg.sound_cell_m), 0), n - 1);
        f[r * n + c] += typeof amount === "number" ? amount : amount[j];
      }
      for (let p = 0; p < passes; p++) {
        let g = new Float64Array(n * n);
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++)
          g[r * n + c] = ((r > 0 ? f[(r - 1) * n + c] : 0) + 2 * f[r * n + c] + (r < n - 1 ? f[(r + 1) * n + c] : 0)) / 4;
        f = g; g = new Float64Array(n * n);
        for (let r = 0; r < n; r++) for (let c = 0; c < n; c++)
          g[r * n + c] = ((c > 0 ? f[r * n + c - 1] : 0) + 2 * f[r * n + c] + (c < n - 1 ? f[r * n + c + 1] : 0)) / 4;
        f = g;
      }
    }
    f.n = n;
    return f;
  }

  _ears(i, field, scale, x, col) {
    const cfg = this.cfg, n = field.n;
    let any = false;
    for (let c = 0; c < field.length; c++) if (field[c]) { any = true; break; }
    if (!any) { for (let r = 0; r < i.length; r++) { x[r * N_IN + col] = 0; x[r * N_IN + col + 1] = 0; } return; }
    const at = (xx, yy) => field[Math.min(Math.max(Math.trunc(xx / cfg.sound_cell_m), 0), n - 1) * n
      + Math.min(Math.max(Math.trunc(yy / cfg.sound_cell_m), 0), n - 1)];
    for (let r = 0; r < i.length; r++) {
      const k = i[r], side = this.heading[k] + PI / 2, ex = cfg.ear_m * Math.cos(side), ey = cfg.ear_m * Math.sin(side);
      const left = at(this.px[k] + ex, this.py[k] + ey), right = at(this.px[k] - ex, this.py[k] - ey);
      x[r * N_IN + col] = 1 - Math.exp(-(left + right) / 2 / scale);
      x[r * N_IN + col + 1] = right + left > 0 ? (right - left) / (right + left) : 0;
    }
  }

  _riders(i) {   // (8h) young under wean_frac whose mother is alive: [young, mother], indexes into i
    const cfg = this.cfg, ry = [], rm = [];
    if (!cfg.carry) return [ry, rm];
    const at = new Map();
    for (let r = 0; r < i.length; r++) at.set(this.uid[i[r]], r);
    for (let r = 0; r < i.length; r++) {
      const k = i[r];
      if (!(this.mass[k] < cfg.wean_frac * this.adult[k] && this.parent[k] >= 0)) continue;
      const m = at.get(this.parent[k]);
      if (m !== undefined && m !== r) { ry.push(r); rm.push(m); }
    }
    return [ry, rm];
  }

  drinkable() {
    const w = this.world, n = w.n;
    if (!this._shore) {
      this._shore = new Uint8Array(n * n);
      for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
        if (w.biome[r * n + c] === WATER) continue;
        const wt = (rr, cc) => rr >= 0 && rr < n && cc >= 0 && cc < n && w.biome[rr * n + cc] === WATER;
        this._shore[r * n + c] = wt(r - 1, c) || wt(r + 1, c) || wt(r, c - 1) || wt(r, c + 1) ? 1 : 0;
      }
    }
    const shore = this._shore, wet = w.wet, pw = this.cfg.puddle_wet, rain = w.cfg.rain;
    return (c) => shore[c] === 1 || (rain && wet[c] > pw);
  }

  senses(i) {
    const cfg = this.cfg, w = this.world, n = w.n, ni = i.length;
    const perCell = w.cfg.plant_kj_m2 * w.cfg.cell_m ** 2;
    const x = new Float64Array(ni * N_IN);
    const here = new Int32Array(ni);
    // sound and fruit scent
    const cx = [], cy = [];
    for (let k = 0; k < this.cap; k++) if (this.alive[k] && this.calling[k]) { cx.push(this.px[k]); cy.push(this.py[k]); }
    const sound = this._air(cx, cy, 1.0, 2);
    const fx = [], fy = [], fa = [];
    for (const c of w.crownCells()) if (w.fruit[c] > 0) {
      fx.push((Math.floor(c / n) + 0.5) * w.cfg.cell_m); fy.push((c % n + 0.5) * w.cfg.cell_m); fa.push(w.fruit[c]);
    }
    const scent = this._air(fx, fy, fa, 3);
    const eyes = [-cfg.sight_deg * DEG, 0.0 * DEG, cfg.sight_deg * DEG];
    for (let r = 0; r < ni; r++) {
      const k = i[r], o = r * N_IN;
      const h = this._cell(this.px[k], this.py[k]);
      here[r] = h;
      x[o] = (w.plants[h] + w.dead[h]) / perCell;
      for (let e = 0; e < 3; e++) {
        const a = this.heading[k] + eyes[e];
        const cell = this._cell(this.px[k] + cfg.sight_m * Math.cos(a), this.py[k] + cfg.sight_m * Math.sin(a));
        const blocked = !this._in || w.biome[cell] === WATER;
        x[o + 1 + e] = blocked ? 0.0 : (w.plants[cell] + w.dead[cell]) / perCell;
        x[o + 4 + e] = blocked ? 1 : 0;
        x[o + 17 + e] = blocked ? 0.0 : Math.min(this.meat[cell] / (5 * cfg.body_kj_kg), 1.0);
        x[o + 27 + e] = blocked || w.fruit[cell] === 0 ? 0.0 : 1 - Math.exp(-w.fruit[cell] / cfg.fruit_sense_kj);
      }
      x[o + 7] = this.reserve[k] / this.reserveCap(this.mass[k], this.fat[k]);
      x[o + 8] = this.mass[k] / this.adult[k];
      x[o + 9] = 1.0;
      x[o + 10] = Math.min(this.meat[h] / (5 * cfg.body_kj_kg), 1.0);
      x[o + 24] = this.mem0[k]; x[o + 25] = this.mem1[k];
      x[o + 26] = w.fruit[h] === 0 ? 0.0 : 1 - Math.exp(-w.fruit[h] / cfg.fruit_sense_kj);
      x[o + 32] = this.dry[k] / (cfg.dry_max_l_kg * this.mass[k]);
    }
    this._ears(i, sound, 1.0, x, 22);
    this._ears(i, scent, cfg.fruit_sense_kj, x, 30);
    // other creatures: the nearest in each eye's sector; carried young neither see nor are seen
    this._ahead = new Int32Array(ni).fill(-1); this._aheadD = new Float64Array(ni).fill(Infinity);
    this._near = new Int32Array(ni).fill(-1); this._nearD = new Float64Array(ni).fill(Infinity);
    this._ride = this._riders(i);
    const free = new Uint8Array(ni).fill(1);
    for (const r of this._ride[0]) free[r] = 0;
    const f = [];
    for (let r = 0; r < ni; r++) if (free[r]) f.push(r);
    const px = new Float64Array(f.length), py = new Float64Array(f.length), hd = new Float64Array(f.length);
    f.forEach((r, j) => { px[j] = this.px[i[r]]; py[j] = this.py[i[r]]; hd[j] = this.heading[i[r]]; });
    const s = nearestBySector(px, py, hd, cfg.sight_m, w.n * w.cfg.cell_m);
    f.forEach((r, j) => { this._near[r] = s.near[j] >= 0 ? f[s.near[j]] : -1; this._nearD[r] = s.nearD[j]; });
    for (let j = 0; j < f.length; j++) for (let e = 0; e < 3; e++) {
      const bj = s.secB[3 * j + e];
      if (bj < 0) continue;
      const a = f[j], b = f[bj], d = s.secD[3 * j + e];
      x[a * N_IN + 11 + e] = 1 - d / cfg.sight_m;
      x[a * N_IN + 14 + e] = Math.min(Math.max(Math.log(this.carried(i[b]) / this.carried(i[a])), -3), 3) / 3;
      if (e === 1) {
        this._ahead[a] = b; this._aheadD[a] = d;
        const ka = i[a], kb = i[b], ua = this.uid[ka], ub = this.uid[kb], pa = this.parent[ka], pb = this.parent[kb];
        const qa = this.parent2[ka], qb = this.parent2[kb];
        x[a * N_IN + 20] = 1 - Math.abs(this.diet[ka] - this.diet[kb]);
        x[a * N_IN + 21] = (pb === ua || qb === ua || pa === ub || qa === ub || (pa === pb && pa >= 0) || (qa === qb && qa >= 0)) ? 1 : 0;
      }
    }
    this._here = here;
    return x;
  }

  think(i, x) {
    const out = new Float64Array(i.length * N_OUT), h = new Float64Array(N_HID);
    const ob1 = N_IN * N_HID, ow2 = ob1 + N_HID, ob2 = ow2 + N_HID * N_OUT, w = this.w;
    for (let r = 0; r < i.length; r++) {
      const base = i[r] * N_W;
      for (let j = 0; j < N_HID; j++) h[j] = w[base + ob1 + j];
      for (let q = 0; q < N_IN; q++) {
        const xq = x[r * N_IN + q], wq = base + q * N_HID;
        for (let j = 0; j < N_HID; j++) h[j] += xq * w[wq + j];
      }
      for (let j = 0; j < N_HID; j++) h[j] = Math.tanh(h[j]);
      for (let o = 0; o < N_OUT; o++) {
        let s = w[base + ob2 + o];
        for (let j = 0; j < N_HID; j++) s += h[j] * w[base + ow2 + j * N_OUT + o];
        out[r * N_OUT + o] = s;
      }
    }
    return out;
  }

  step() {
    const cfg = this.cfg, w = this.world, b = this.books, dt = cfg.dt, day = dt / cfg.day_s;
    const i = [];
    for (let k = 0; k < this.cap; k++) if (this.alive[k]) i.push(k);
    const ni = i.length;
    this.lastI = i;
    if (ni) {
      const x = this.senses(i), out = this.think(i, x);
      this.lastOut = out;
      const turn = new Float64Array(ni), speed = new Float64Array(ni), eat = new Uint8Array(ni), split = new Uint8Array(ni);
      const attack = new Uint8Array(ni), call = new Uint8Array(ni);
      const m = new Float64Array(ni), fat = new Float64Array(ni), diet = new Float64Array(ni), carried = new Float64Array(ni);
      const bmr = new Float64Array(ni);
      for (let r = 0; r < ni; r++) {
        const o = r * N_OUT, k = i[r];
        turn[r] = Math.tanh(out[o]); speed[r] = 1 / (1 + Math.exp(-out[o + 1]));
        eat[r] = out[o + 2] > 0; split[r] = out[o + 3] > 0; attack[r] = out[o + 4] > 0; call[r] = out[o + 5] > 0;
        m[r] = this.mass[k]; fat[r] = this.fat[k]; diet[r] = this.diet[k]; carried[r] = this.carried(k);
        bmr[r] = this.bmrS(m[r]);
      }
      const [ry, rm] = this._ride;
      const load = Float64Array.from(carried);
      for (let j = 0; j < ry.length; j++) load[rm[j]] += carried[ry[j]];
      for (const r of ry) { speed[r] = 0.0; attack[r] = 0; }

      // move; water and the map's edge stop you (you pay only for what you covered)
      const walked = new Float64Array(ni);
      for (let r = 0; r < ni; r++) {
        const k = i[r];
        this.heading[k] = pymod(this.heading[k] + turn[r] * cfg.turn_rad_s * dt + PI, TWO_PI) - PI;
        const v = speed[r] * (cfg.aerobic_scope * bmr[r] / this.realSPerS / this.walkKjM(load[r]));
        const sx = v * dt * Math.cos(this.heading[k]), sy = v * dt * Math.sin(this.heading[k]);
        const cell = this._cell(this.px[k] + sx, this.py[k] + sy);
        const ok = this._in && w.biome[cell] !== WATER;
        this.px[k] += ok ? sx : 0.0; this.py[k] += ok ? sy : 0.0;
        walked[r] = ok ? v * dt : 0.0;
      }
      for (let j = 0; j < ry.length; j++) { this.px[i[ry[j]]] = this.px[i[rm[j]]]; this.py[i[ry[j]]] = this.py[i[rm[j]]]; }

      // attack the nearest straight ahead, if within reach
      const tgt = this._ahead, td = this._aheadD, killed = new Uint8Array(ni), att = [];
      for (let r = 0; r < ni; r++) if (attack[r] && tgt[r] >= 0 && td[r] < cfg.reach_m) att.push(r);
      this.fights = att.map((r) => [i[r], i[tgt[r]]]);
      if (att.length) {
        const settled = this.rng.random(att.length), won = this.rng.random(att.length);
        att.forEach((a, j) => {
          const t = tgt[a];
          const ca = carried[a] * (1 + cfg.weapon * this.diet[i[a]]), ct = carried[t] * (1 + cfg.weapon * this.diet[i[t]]);
          const st = settled[j] < cfg.fight_per_s * dt, wn = won[j] < ca ** 2 / (ca ** 2 + ct ** 2);
          if (st && wn) killed[t] = 1;
          if (st && !wn) killed[a] = 1;
        });
      }

      // pay for resting, keeping warm, walking and attacking
      const temp = w.temperature(), cost = new Float64Array(ni), warm = new Float64Array(ni);
      let heat = 0;
      for (let r = 0; r < ni; r++) {
        const effort = attack[r] ? cfg.aerobic_scope * bmr[r] / this.realSPerS * dt : 0.0;
        const rest = bmr[r] * (1 + cfg.call_bmr * call[r]);
        warm[r] = keepWarmS(cfg, m[r], temp, rest);
        cost[r] = (rest + warm[r]) * dt + this.walkKjM(load[r]) * walked[r] + effort;
        this.reserve[i[r]] -= cost[r];
        heat += cost[r];
      }
      b.heat += heat;

      // eat from the cell you're on: fruit, then carcass, then plants
      const here = new Int32Array(ni), want = new Float64Array(ni), room = new Float64Array(ni);
      const kM = new Float64Array(ni), kP = new Float64Array(ni), kF = new Float64Array(ni), askF = new Float64Array(ni);
      for (let r = 0; r < ni; r++) {
        const k = i[r];
        kM[r] = cfg.digest_meat_grazer + (cfg.digest_meat - cfg.digest_meat_grazer) * diet[r];
        kP[r] = plantKeep(cfg, diet[r], m[r]);
        room[r] = Math.max(this.reserveCap(m[r], fat[r]) - this.reserve[k], 0.0);
        want[r] = eat[r] * (1 - speed[r]) * cfg.eat_bmr_day * bmr[r] * dt;
        here[r] = this._cell(this.px[k], this.py[k]);
        kF[r] = cfg.digest_fruit + (cfg.digest_fruit_hunter - cfg.digest_fruit) * diet[r];
        askF[r] = Math.min(want[r], room[r] / kF[r]);
      }
      const gotF = w.eatFruit(here, askF), wantM = new Float64Array(ni);
      for (let r = 0; r < ni; r++) {
        room[r] -= kF[r] * gotF[r];
        want[r] = want[r] - gotF[r];
        wantM[r] = kM[r] > 0.01 ? Math.min(want[r], room[r] / Math.max(kM[r], 1e-9)) : 0.0;
      }
      const gotM = this._takeMeat(here, wantM), wantP = new Float64Array(ni);
      for (let r = 0; r < ni; r++) {
        room[r] -= kM[r] * gotM[r];
        wantP[r] = Math.max(kP[r] > 0.01 ? Math.min(want[r] - gotM[r], room[r] / Math.max(kP[r], 1e-9)) : 0.0, 0.0);
      }
      const [green, dead] = w.graze(here, wantP);
      let dung = 0;
      for (let r = 0; r < ni; r++) {
        const gotP = kP[r] * (green[r] + w.cfg.dead_quality * dead[r]);
        this.reserve[i[r]] += kM[r] * gotM[r] + gotP + kF[r] * gotF[r];
        dung += (1 - kM[r]) * gotM[r] + green[r] + dead[r] - gotP + (1 - kF[r]) * gotF[r];
      }
      b.dung += dung;
      const milk = cfg.milk ? this._nurse(i, m, fat, killed) : new Float64Array(ni);
      if (cfg.thirst) {
        const dr = this.drinkable();
        for (let r = 0; r < ni; r++) {
          const k = i[r];
          const wetCost = cfg.cold_breath ? cost[r] - (1 - cfg.breath_water_per_kj / cfg.water_per_kj) * warm[r] * dt : cost[r];
          this.dry[k] += cfg.water_per_kj * wetCost;
          this.dry[k] -= (cfg.water_green * green[r] + cfg.water_dead * dead[r] + cfg.water_meat * gotM[r]
            + cfg.water_fruit * gotF[r] + cfg.water_milk * milk[r]);
          this.drank[k] = eat[r] && dr(here[r]) ? 1 : 0;
          if (this.drank[k]) this.dry[k] = 0.0;
          this.dry[k] = Math.max(this.dry[k], 0.0);
        }
      }

      // grow towards adult mass while the reserve is comfortable, paying for the tissue
      const perKg = cfg.body_kj_kg * (1 + cfg.build_overhead);
      let built = 0;
      for (let r = 0; r < ni; r++) {
        const k = i[r];
        let g = cfg.growth_a * (m[r] * 1000) ** 0.75 * (1 - (m[r] / this.adult[k]) ** 0.25) * day / 1000;
        const spare = Math.max(this.reserve[k] - cfg.grow_above * this.reserveCap(m[r], fat[r]), 0.0);
        g = Math.max(Math.min(g, spare / perKg), 0.0);
        this.mass[k] += g;
        this.reserve[k] -= g * perKg;
        built += g * cfg.body_kj_kg * cfg.build_overhead;
        this.age[k] += day;
        this.calling[k] = call[r];
        this.mem0[k] = Math.tanh(out[r * N_OUT + 6]); this.mem1[k] = Math.tanh(out[r * N_OUT + 7]);
      }
      b.heat += built;

      const ready = new Uint8Array(ni);
      for (let r = 0; r < ni; r++) ready[r] = split[r] && this.mass[i[r]] >= 0.95 * this.adult[i[r]] && !killed[r] ? 1 : 0;
      if (cfg.sexual) {
        const near = this._near, nd = this._nearD;
        let cand = [];
        for (let r = 0; r < ni; r++) if (ready[r] && near[r] >= 0 && nd[r] < cfg.reach_m) cand.push(r);
        cand = cand.filter((r) => ready[near[r]]);
        cand = cand.filter((r) => Math.abs(this.diet[i[r]] - this.diet[i[near[r]]]) <= cfg.mate_gap);
        const mated = new Uint8Array(ni);
        if (cand.length) {
          const used = new Set(), ma = [], pa = [];
          for (const a of this.rng.permutation(cand)) {
            const b_ = near[a];
            if (!used.has(a) && !used.has(b_)) { used.add(a); used.add(b_); ma.push(a); pa.push(b_); }
          }
          this._births(ma.map((r) => i[r]), pa.map((r) => i[r]));
          for (const r of ma) mated[r] = 1;
          for (const r of pa) mated[r] = 1;
        }
        let anyWait = false;
        const wait = new Uint8Array(ni);
        for (let r = 0; r < ni; r++) { wait[r] = ready[r] && !mated[r] && !killed[r] ? 1 : 0; anyWait ||= wait[r]; }
        if (anyWait) {
          const u = this.rng.random(ni), solo = [];
          for (let r = 0; r < ni; r++) {
            const p_ = this.alone[i[r]] * dt / (cfg.alone_wait_days * cfg.day_s);
            if (wait[r] && u[r] < p_) solo.push(i[r]);
          }
          if (solo.length) { const n0 = b.births; this._births(solo); b.births_alone += b.births - n0; }
        }
      } else {
        this._births(i.filter((k, r) => ready[r]));
      }
      this._deaths(i, killed);
    }

    this.ticks += 1;
    this.t += dt;
    if (this.ticks % this.plantEvery === 0) {
      const rate = cfg.meat_rot_per_day * cfg.rot_q10 ** ((w.temperature() - w.cfg.temp_summer_c) / 10);
      const f = 1 - Math.exp(-rate * w.cfg.dt_days);
      let rot = 0;
      for (let c = 0; c < this.meat.length; c++) { const x = this.meat[c] * f; this.meat[c] -= x; rot += x; }
      b.decay += rot;
      w.step();
    }
  }

  parched(k) { return this.dry[k] > this.cfg.dry_max_l_kg * this.mass[k]; }

  _nurse(i, m, fat, killed) {   // (8f) young near their mother drink her milk -> kJ each got
    const cfg = this.cfg, dt = cfg.dt, ni = i.length, got = new Float64Array(ni);
    const young = [];
    for (let r = 0; r < ni; r++) if (m[r] < cfg.wean_frac * this.adult[i[r]] && this.parent[i[r]] >= 0 && !killed[r]) young.push(r);
    if (!young.length) return got;
    const at = new Map();
    for (let r = 0; r < ni; r++) at.set(this.uid[i[r]], r);
    const ys = [], ms = [];
    for (const y of young) {
      const mo = at.get(this.parent[i[y]]);
      if (mo === undefined || killed[mo]) continue;
      if (Math.hypot(this.px[i[y]] - this.px[i[mo]], this.py[i[y]] - this.py[i[mo]]) < cfg.reach_m) { ys.push(y); ms.push(mo); }
    }
    if (!ys.length) return got;
    const ask = ys.map((y) => Math.min(cfg.eat_bmr_day * this.bmrS(m[y]) * dt,
      Math.max(this.reserveCap(m[y], fat[y]) - this.reserve[i[y]], 0.0) / cfg.milk_eff));
    const spare = new Float64Array(ni);
    for (let r = 0; r < ni; r++) spare[r] = Math.max(this.reserve[i[r]] - cfg.milk_floor * this.reserveCap(m[r], fat[r]), 0.0);
    const asked = new Float64Array(ni);
    ms.forEach((mo, j) => { asked[mo] += ask[j]; });
    const give = ys.map((y, j) => ask[j] * Math.min(1.0, asked[ms[j]] > 0 ? spare[ms[j]] / asked[ms[j]] : 0));
    ms.forEach((mo, j) => { this.reserve[i[mo]] -= give[j]; });
    let lost = 0;
    ys.forEach((y, j) => { this.reserve[i[y]] += cfg.milk_eff * give[j]; lost += (1 - cfg.milk_eff) * give[j]; got[y] = cfg.milk_eff * give[j]; });
    this.books.heat += lost;
    return got;
  }

  _takeMeat(cells, want) { const M = this.meat; return take(M, (c) => M[c], cells, want); }

  // a parent splits off one child (or, with partners, each pair has one, each paying half)
  _births(parents, partners = null) {
    const cfg = this.cfg, b = this.books, rng = this.rng;
    let k = parents.length;
    if (!k) return;
    const q = partners || parents;
    const pick = (a) => {
      if (!partners) return parents.map((p) => a[p]);
      const u = rng.random(k);
      return parents.map((p, j) => (u[j] < 0.5 ? a[p] : a[q[j]]));
    };
    const pa = pick(this.adult), ea = rng.normal(0, cfg.sigma_gene, k);
    const pb = pick(this.birth), eb = rng.normal(0, cfg.sigma_gene, k);
    const pf = pick(this.fat), ef = rng.normal(0, cfg.sigma_gene, k);
    const pd = pick(this.diet), ed = rng.normal(0, cfg.sigma_diet, k);
    const pl = pick(this.alone), el = rng.normal(0, cfg.sigma_alone, k);
    const uw = partners ? rng.random(k * N_W) : null;
    const clip = (x, a, c) => Math.min(Math.max(x, a), c);
    let g = parents.map((p, j) => {
      const adult = pa[j] * Math.exp(ea[j]), frac = clip(pb[j] * Math.exp(eb[j]), 0.01, 0.9);
      const fat = clip(pf[j] * Math.exp(ef[j]), 0.01, 1.0), diet = clip(pd[j] + ed[j], 0.0, 1.0);
      const alone = clip(pl[j] + el[j], 0.0, 1.0), m0 = frac * adult, r0 = 0.5 * this.reserveCap(m0, fat);
      const cost = m0 * cfg.body_kj_kg * (1 + cfg.build_overhead) + r0;
      return { p, q: q[j], j, adult, frac, fat, diet, alone, m0, r0, cost };
    });
    const share = partners ? 0.5 : 1.0;
    g = g.filter((c) => this.reserve[c.p] >= share * c.cost && this.reserve[c.q] >= share * c.cost);
    if (this.cap - this.count() < g.length) this._grow(g.length);
    const free = this._free();
    if (free.length < g.length) { b.full += g.length - free.length; g = g.slice(0, free.length); }
    const kids = free.slice(0, g.length);
    let heat = 0;
    g.forEach((c) => { this.reserve[c.p] -= share * c.cost; });
    if (partners) g.forEach((c) => { this.reserve[c.q] -= share * c.cost; });
    g.forEach((c) => { heat += c.m0 * cfg.body_kj_kg * cfg.build_overhead; });
    b.heat += heat;
    const hd = rng.uniform(-PI, PI, kids.length), nw = rng.normal(0, cfg.sigma_w, kids.length * N_W);
    g.forEach((c, j) => {
      const s = kids[j];
      this.alive[s] = 1; this.px[s] = this.px[c.p]; this.py[s] = this.py[c.p]; this.heading[s] = hd[j];
      this.adult[s] = c.adult; this.birth[s] = c.frac; this.mass[s] = c.m0; this.reserve[s] = c.r0;
      this.fat[s] = c.fat; this.diet[s] = c.diet; this.alone[s] = c.alone; this.age[s] = 0.0; this.dry[s] = 0.0;
      this.gen[s] = Math.max(this.gen[c.p], this.gen[c.q]) + 1;
      this.parent[s] = this.uid[c.p]; this.parent2[s] = partners ? this.uid[c.q] : -1;
      this.uid[s] = this.nextUid + j;
      for (let o = 0; o < N_W; o++) {
        const from = partners && uw[c.j * N_W + o] >= 0.5 ? c.q : c.p;
        this.w[s * N_W + o] = this.w[from * N_W + o] + nw[j * N_W + o];
      }
      this.calling[s] = 0; this.mem0[s] = 0; this.mem1[s] = 0;
    });
    this.nextUid += kids.length;
    b.births += kids.length;
  }

  // killed, starved, old or parched: the body (and any fat left) becomes a carcass where it fell
  _deaths(i, killed) {
    const cfg = this.cfg, b = this.books;
    let nk = 0;
    for (let r = 0; r < i.length; r++) {
      const k = i[r];
      nk += killed[r];
      const starved = this.reserve[k] < 0, old = this.age[k] > this.lifespan(this.adult[k]), dry = this.parched(k);
      if (!(killed[r] || starved || old || dry)) continue;
      this.causes[killed[r] ? "killed" : dry ? "thirst" : starved ? "starved" : "old"]++;
      const cell = this._cell(this.px[k], this.py[k]);
      this.meat[cell] += this.reserve[k] + cfg.body_kj_kg * this.mass[k];
      this.alive[k] = 0; this.reserve[k] = 0.0; this.mass[k] = 0.0;
      b.deaths++;
    }
    b.kills += nk;
  }

  error() {
    const wb = this.world.ledger, b = this.books, entered = wb.start + wb.sun_in + b.start;
    const expected = entered - wb.dieback - b.heat - b.dung - b.decay;
    return (expected - this.world.totalPlants() - this.energy() - sum(this.meat)) / entered;
  }
}
