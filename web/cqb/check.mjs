// node check.mjs: the JS port against the Python game. Each reference round (cqb.json, from export_web.py cqb)
// starts where the Python round started, uses the same brains, rules and dice, and must end the same way.
import { readFileSync } from "node:fs";
import { ASSAULT, Brain, DEFENSE, Game, House } from "./engine.js";

const data = JSON.parse(readFileSync(new URL("./cqb.json", import.meta.url)));
const buf = readFileSync(new URL("./weights.bin", import.meta.url));
const weights = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const brain = (spec) => new Brain(data.brains.find((b) => b.spec === spec).layers, weights);
const name = (spec) => spec.split("/")[1] || spec;
let same = 0, shortMiss = 0;
for (const ref of data.reference) {
  const game = new Game(data.rules[ref.rules]);
  const house = new House(data.houses.find((h) => h.seed === ref.house));
  const scripted = ref.attacker_moves.length > 0;   // a script's recorded moves (it throws flashbangs)
  const a = scripted ? null : brain(ref.attacker), d = brain(ref.defender);
  let st = game.start(house, ref.start), drift5 = 0, drift = 0;
  for (let k = 0; k < ref.steps && !st.done; k++) {
    const acts = [scripted ? ref.attacker_moves[k] : a.act(game.observe(house, st, ASSAULT)), d.act(game.observe(house, st, DEFENSE))];
    st = game.step(house, st, acts, ref.dice[k], ref.where[k] || [0, 0]);
    const e = Math.max(...[0, 1].map((w) => Math.hypot(st.pos[w][0] - ref.pos[k + 1][w][0], st.pos[w][1] - ref.pos[k + 1][w][1])));
    drift = Math.max(drift, e);
    if (k < 50) drift5 = drift;
  }
  const ok = st.outcome === ref.outcome && st.t === ref.steps;
  same += ok;
  if (!ok && ref.steps < 300) shortMiss++;
  const res = (o) => (o === 1 ? "attacker wins" : "defender wins");
  console.log(`${name(ref.attacker)} vs ${name(ref.defender)} house ${ref.house}: JS ${res(st.outcome)} at ${(st.t / 10).toFixed(1)} s, ` +
    `Python ${res(ref.outcome)} at ${(ref.steps / 10).toFixed(1)} s ${ok ? "✓" : "✗"}; ` +
    `drift ${drift5.toFixed(4)} m in 5 s, ${drift.toFixed(4)} m overall` + (ref.bangs ? `; ${ref.bangs} flashbang(s)` : ""));
}
console.log(`${same}/${data.reference.length} rounds end the same way`);
// Python computes in float32, JS in float64: tiny differences grow, and over 30 s or more a close decision can flip
// (the same rounds also come out differently on the Python game's GPU and CPU). A round under 30 s must match.
console.log(shortMiss ? `${shortMiss} round(s) under 30 s differ: a porting bug` : "every round under 30 s matches");
process.exit(shortMiss ? 1 : 0);
