// One whole live race in Node (no drawing): node field-race.mjs <circuit 0-7> <seed> [random]
// (random: every car a new random design, as scripts/strat_compare.py; else the video race's 8 cars)
// Prints a JSON summary (finished, wrecked, out, dry, stops, positions of the AI pit wall's cars and the rule's),
// to compare the JS port with the Python race's statistics (scripts/strat_compare.py, race_full.py).
import { readFileSync } from "node:fs";
import { Race } from "./field-engine.js";

const data = JSON.parse(readFileSync(new URL("./field-data.json", import.meta.url)));
const [circuit = 3, seed = 1] = process.argv.slice(2, 4).map(Number), random = process.argv[4] === "random";
let s = seed >>> 0;   // mulberry32: a plain multiplicative generator seeded 1..8 gave car 1 the least power every race
const rand = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const designs = random ? data.designs.map(() => data.design.map((p) => p.low + rand() * (p.high - p.low))) : data.designs;
const race = new Race(data, circuit, { rand, designs });
const t0 = performance.now();
while (!race.f.done) race.step();
const f = race.f, F = race.F, pos = F.order(f), n = F.n;
const how = (i) => (f.finished[i] ? "finished" : f.wrecked[i] ? "wrecked" : f.cars[i].crashed ? "wall" : F.dryOut(f, i) ? "dry" : "running");
const avg = (team) => { const p = pos.filter((_, i) => race.team[i] === team); return p.reduce((a, b) => a + b + 1, 0) / p.length; };
console.log(JSON.stringify({ circuit: race.tr.seed, seed, minutes: +(f.t * F.cfg.dt / 60).toFixed(1), wall_s: +((performance.now() - t0) / 1000).toFixed(0),
  how: [...Array(n).keys()].map(how), stops: f.stops, compound: f.setup.map((x) => x.compound), pos, ai: avg(true), rule: avg(false) }));
