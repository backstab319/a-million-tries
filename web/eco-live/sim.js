// The world runs here, in a worker so the page stays smooth (or, where a worker can't start, e.g. some sandboxed
// iframes, on the page itself: app.js then calls connect()); app.js draws what it sends.
import { Life, LIFE, N_W, N_OUT, Rng, World, WORLD } from "./engine.js";

let life = null, world = null, rng = null, running = true, daysPerSec = 1, survivors = null;
let send = (m, t) => postMessage(m, t);   // in a worker; connect() swaps it
let lastGrid = -1, selected = -1, budgetMs = 30, debt = 0, lastT = 0;

async function loadSurvivors() {
  if (survivors) return survivors;
  const meta = await (await fetch("survivors.json")).json();
  const buf = new Float32Array(await (await fetch("survivors.bin")).arrayBuffer());
  const n = meta.count, per = 5 + N_W, g = { adult: [], birth: [], fat: [], alone: [], diet: [], w: new Float64Array(n * N_W) };
  for (let k = 0; k < n; k++) {
    const o = k * per;
    g.adult.push(buf[o]); g.birth.push(buf[o + 1]); g.fat.push(buf[o + 2]); g.alone.push(buf[o + 3]); g.diet.push(buf[o + 4]);
    for (let j = 0; j < N_W; j++) g.w[k * N_W + j] = buf[o + 5 + j];
  }
  survivors = { g, meta };
  return survivors;
}

async function start(m) {
  // build the whole new world first, then swap it in (the loop keeps running the old one meanwhile)
  const s = m.start === "survivors" ? await loadSurvivors() : null;
  const w = World.make(m.seed, { ...WORLD }), r = new Rng([m.seed, 2]), l = new Life(w, 0, r, { ...LIFE });
  if (s) l.add(s.meta.count, null, s.g);
  else l.add(m.n || 1000, [0.0, 0.1]);
  world = w; rng = r; life = l; running = true;
  lastGrid = -1; selected = -1; debt = 0;
  send({ type: "land", n: world.n, cell_m: world.cfg.cell_m, biome: world.biome, capacity: Float32Array.from(world.capacity),
    trees: world.trees, seed: m.seed, start: m.start });
  post(true);
}

function grids() {
  const nn = world.n * world.n, food = new Float32Array(nn), dead = new Float32Array(nn), wet = new Float32Array(nn);
  const fruit = new Float32Array(nn), meat = new Float32Array(nn);
  for (let c = 0; c < nn; c++) {
    food[c] = world.plants[c]; dead[c] = world.dead[c]; wet[c] = world.wet[c]; fruit[c] = world.fruit[c]; meat[c] = life.meat[c];
  }
  return { food, dead, wet, fruit, meat };
}

function post(withGrid) {
  const ids = [];
  for (let k = 0; k < life.cap; k++) if (life.alive[k]) ids.push(k);
  const n = ids.length, A = new Float32Array(n * 9);
  let massSum = 0, dietSum = 0, genMax = 0, dryMax = 0;
  // outputs of the last step, by slot (for eat / attack / call marks)
  const act = new Map();
  if (life.lastOut && life.lastI) life.lastI.forEach((k, r) => act.set(k, r));
  ids.forEach((k, j) => {
    const r = act.get(k), o = r === undefined ? null : r * N_OUT;
    const young = life.mass[k] < life.cfg.wean_frac * life.adult[k] ? 1 : 0;
    const flags = (life.drank[k] ? 1 : 0) | (life.calling[k] ? 2 : 0) | (o !== null && life.lastOut[o + 4] > 0 ? 4 : 0)
      | (o !== null && life.lastOut[o + 2] > 0 ? 8 : 0) | (young ? 16 : 0);
    A.set([life.px[k], life.py[k], life.heading[k], life.mass[k], life.diet[k], flags,
      life.dry[k] / (life.cfg.dry_max_l_kg * life.mass[k]), life.reserve[k] / life.reserveCap(life.mass[k], life.fat[k]), life.uid[k]], j * 9);
    massSum += life.mass[k]; dietSum += life.diet[k]; genMax = Math.max(genMax, life.gen[k]);
    dryMax = Math.max(dryMax, life.dry[k] / (life.cfg.dry_max_l_kg * life.mass[k]));
  });
  let sel = null;
  if (selected >= 0) {
    const k = ids.find((q) => life.uid[q] === selected);
    if (k === undefined) { sel = { gone: true, uid: selected }; selected = -1; }
    else {
      const r = act.get(k), o = r === undefined ? null : r * N_OUT;
      sel = { uid: life.uid[k], mass: life.mass[k], adult: life.adult[k], age: life.age[k], life: life.lifespan(life.adult[k]),
        diet: life.diet[k], gen: life.gen[k], alone: life.alone[k], thirst: life.dry[k] / (life.cfg.dry_max_l_kg * life.mass[k]),
        reserve: life.reserve[k] / life.reserveCap(life.mass[k], life.fat[k]), parent: life.parent[k],
        out: o === null ? null : Array.from(life.lastOut.subarray(o, o + N_OUT)) };
    }
  }
  const msg = { type: "frame", A, n, day: world.day + (life.ticks % life.plantEvery) / life.cfg.day_s, temp: world.temperature(),
    season: world.season(), books: { ...life.books }, causes: { ...life.causes }, massMean: n ? massSum / n : 0,
    dietMean: n ? dietSum / n : 0, genMax, error: life.error(), storms: world.storms || [], sel, running, daysPerSec };
  const transfer = [A.buffer];
  if (withGrid) { msg.grid = grids(); transfer.push(...Object.values(msg.grid).map((g) => g.buffer)); }
  send(msg, transfer);
}

function loop() {
  const now = performance.now();
  if (life && running) {
    debt = Math.min(debt + (now - lastT) / 1000 * daysPerSec * life.cfg.day_s, life.cfg.day_s * daysPerSec * 0.5 + 1);
    const t0 = performance.now();
    while (debt >= 1 && performance.now() - t0 < budgetMs) { life.step(); debt -= 1; if (life.count() === 0) break; }
    if (performance.now() - t0 >= budgetMs) debt = Math.min(debt, 1);   // can't keep up: run flat out, don't pile up
    const day = Math.floor(world.day * 4);
    post(day !== lastGrid);
    lastGrid = day;
    if (life.count() === 0) { running = false; post(true); send({ type: "extinct", day: world.day }); }
  }
  lastT = now;
  setTimeout(loop, 16);
}

async function receive(m) {
  if (m.type === "start") await start(m);
  else if (m.type === "run") running = m.on;
  else if (m.type === "speed") daysPerSec = m.daysPerSec;
  else if (m.type === "select") {
    let best = -1, bd = (m.r || 4) ** 2;
    for (let k = 0; k < life.cap; k++) if (life.alive[k]) {
      const d = (life.px[k] - m.x) ** 2 + (life.py[k] - m.y) ** 2;
      if (d < bd) { bd = d; best = k; }
    }
    selected = best >= 0 ? life.uid[best] : -1;
    post(false);
  } else if (m.type === "scramble") {   // the episode's control: every brain swapped for a random one
    for (let k = 0; k < life.cap; k++) if (life.alive[k]) {
      const w = rng.normal(0, 1, N_W);
      life.w.set(w, k * N_W);
      life.mem0[k] = 0; life.mem1[k] = 0;
    }
    post(false);
  } else if (m.type === "add") {
    if (m.what === "survivors") { const s = await loadSurvivors(); life.add(s.meta.count, null, s.g); }
    else life.add(m.n, [0.0, 0.1]);
    if (!running && life.count()) running = true;
    post(true);
  }
}

// on the page: messages go straight to `onFrame`; returns the function to send messages to the sim
export function connect(onFrame) { send = (m) => onFrame(m); return receive; }
if (typeof WorkerGlobalScope !== "undefined" && self instanceof WorkerGlobalScope) onmessage = (e) => receive(e.data);
lastT = performance.now();
loop();
