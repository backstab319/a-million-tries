// node check.mjs [ref dir]: the JS port against the Python ecosystem, on Python's own dice.
// The references come from `uv run python scripts/eco_web_ref.py` (out/eco-live-ref/, not shipped with the Space).
import { readFileSync } from "node:fs";
import { Life, LIFE, ReplayRng, World, WORLD, N_W } from "./engine.js";

const dir = process.argv[2] || new URL("../../out/eco-live-ref/", import.meta.url).pathname;
let allOk = true;
for (const name of ["winter", "autumn"]) {
  const ref = JSON.parse(readFileSync(`${dir}/${name}.json`));
  const raw = readFileSync(`${dir}/${name}.dice.bin`);
  const buf = new Float64Array(raw.buffer, raw.byteOffset, raw.byteLength / 8);
  const stream = (s) => new ReplayRng(ref.dice.filter((e) => e[0] === s), buf);
  const cfg = { ...WORLD, size: ref.size, rain: true };
  const world = World.make(ref.seed, cfg, { land: stream("land"), trees: stream("trees"), weather: stream("weather") });
  const st = ref.start;
  const gap = (a, b) => { let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k])); return m; };
  const land = st.biome.every((v, k) => v === world.biome[k]) && gap(world.capacity, st.capacity) < 1e-9
    && gap(world.plants, st.plants) < 1e-9 && gap(world.roots, st.roots) < 1e-9
    && JSON.stringify(world.trees) === JSON.stringify(st.trees);
  console.log(`${name}: the land ${land ? "matches ✓" : "DIFFERS ✗"} (${world.trees.length} fruit trees)`);
  allOk &&= land;
  world.day = ref.day;
  const life = new Life(world, 0, stream("life"), LIFE);
  life.add(ref.n, ref.diet);
  if (ref.n_like) {
    const L = ref.like, like = { ...L, w: Float64Array.from(L.w.flat()) };
    if (like.w.length !== L.diet.length * N_W) throw new Error("like.w");
    life.add(ref.n_like, null, like);
  }
  let worst = { pos: 0, mass: 0, reserve: 0, dry: 0, plants: 0, meat: 0 }, broke = null, si = 0;
  const cmp = (s) => {
    const slots = [];
    for (let k = 0; k < life.cap; k++) if (life.alive[k]) slots.push(k);
    if (slots.length !== s.slots.length || slots.some((v, k) => v !== s.slots[k])) return `alive slots differ (JS ${slots.length}, Python ${s.slots.length})`;
    if (life.books.births !== s.births || life.books.deaths !== s.deaths || life.books.kills !== s.kills)
      return `books differ: births ${life.books.births}/${s.births} deaths ${life.books.deaths}/${s.deaths} kills ${life.books.kills}/${s.kills}`;
    slots.forEach((k, j) => {
      worst.pos = Math.max(worst.pos, Math.abs(life.px[k] - s.pos[j][0]), Math.abs(life.py[k] - s.pos[j][1]));
      worst.mass = Math.max(worst.mass, Math.abs(life.mass[k] - s.mass[j]) / s.mass[j]);
      worst.reserve = Math.max(worst.reserve, Math.abs(life.reserve[k] - s.reserve[j]) / Math.max(Math.abs(s.reserve[j]), 1));
      worst.dry = Math.max(worst.dry, Math.abs(life.dry[k] - s.dry[j]));
    });
    const tot = (a) => a.reduce((x, y) => x + y, 0);
    worst.plants = Math.max(worst.plants, Math.abs(tot(world.plants) - s.plants) / s.plants);
    worst.meat = Math.max(worst.meat, Math.abs(tot(life.meat) - s.meat) / Math.max(s.meat, 1));
    return null;
  };
  // every `sync` ticks, take Python's state for everything rounding moves (positions, energy, water, the plants...):
  // chaos (random brains feed their memory back) grows a 1e-15 gap ~10x every 10 ticks, so a whole day can't stay
  // in step on different maths libraries; between syncs every birth, death, kill and draw of the dice must agree
  const streams = { life: life.rng, weather: world.rng };
  const load = (y) => {
    const slots = [];
    for (let k = 0; k < life.cap; k++) if (life.alive[k]) slots.push(k);
    if (slots.length !== y.slots.length || slots.some((v, k) => v !== y.slots[k])) return "alive slots differ at a sync";
    for (const [nm, r] of Object.entries(streams)) if (r.at !== y.dice[nm]) return `${nm} dice out of step (${r.at} v ${y.dice[nm]})`;
    y.slots.forEach((k, j) => { for (const f of Object.keys(y.life)) life[f][k] = y.life[f][j]; });
    for (const f of Object.keys(y.world)) world[f].set(y.world[f]);
    life.meat.set(y.meat);
    world.day = y.day;
    Object.assign(world.ledger, y.ledger);
    Object.assign(life.books, y.books);
    return null;
  };
  let syncs = 0;
  try {
    broke = cmp(ref.snaps[si++]) && "tick 0: " + cmp(ref.snaps[0]);
    for (let t = 1; t <= ref.ticks && !broke; t++) {
      if ((t - 1) % ref.sync === 0) { const y = ref.syncs.find((q) => q.tick === t - 1); const e = load(y); if (e) { broke = `tick ${t - 1}: ${e}`; break; } syncs++; }
      life.step();
      if (si < ref.snaps.length && ref.snaps[si].tick === t) { const e = cmp(ref.snaps[si++]); if (e) broke = `tick ${t}: ${e}`; }
    }
  } catch (e) { broke = `tick ${life.ticks + 1}: ${e.message}`; }
  const s = ref.snaps[ref.snaps.length - 1];
  const fmt = (v) => v.toExponential(1);
  console.log(`${name}: ${broke ? "BROKE at " + broke + " ✗" : `${ref.ticks} ticks (${syncs} syncs), the same ${s.slots.length} alive, births ${s.births}, deaths ${s.deaths}, kills ${s.kills} ✓`}`);
  console.log(`  largest gaps: position ${fmt(worst.pos)} m, mass ${fmt(worst.mass)}, reserve ${fmt(worst.reserve)}, ` +
    `water ${fmt(worst.dry)} l, plants ${fmt(worst.plants)}, carcasses ${fmt(worst.meat)} (relative); JS books error ${fmt(life.error())}`);
  allOk &&= !broke;
}
console.log(allOk ? "all match" : "MISMATCH");
process.exit(allOk ? 0 : 1);
