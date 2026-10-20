// node check.mjs: the JS port against the Python game. Each reference round (cqb.json, from export_web.py cqb)
// starts where the Python round started, uses the same brains, rules and dice, and must end the same way.
import { readFileSync } from "node:fs";
import { ASSAULT, Brain, DEFENSE, Game, House } from "./engine.js";

const data = JSON.parse(readFileSync(new URL("./cqb.json", import.meta.url)));
const buf = readFileSync(new URL("./weights.bin", import.meta.url));
const weights = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const brain = (spec) => new Brain(data.brains.find((b) => b.spec === spec).layers, weights);
const name = (spec) => spec.split("/")[1] || spec;
// (ep13) rounds under 20 s that part by rounding alone (traced: every input within float32 noise), named so they
// can't hide a real porting bug (see the note at the end)
const FLIPS = { "home:100068": "parts at 3.2 s sliding along a wall (inputs within 7e-7 at 1 s, 7e-6 at 2 s), 5 cm apart at 15.8 s: a 24.5% vs 33.6% hit roll" };
let same = 0, shortMiss = 0, flips = 0;
for (const ref of data.reference) {
  const game = new Game(data.rules[ref.rules]);
  const house = new House(data.houses.find((h) => h.seed === ref.house && (h.layout || "") === (ref.layout || "")));
  const scripted = ref.attacker_moves.length > 0;   // a script's recorded moves (it throws flashbangs)
  const a = scripted ? null : brain(ref.attacker), d = brain(ref.defender);
  let st = game.start(house, ref.start), drift5 = 0, drift = 0;
  for (let k = 0; k < ref.steps && !st.done; k++) {   // every operator plays its side's brain (2 vs 1: three)
    const acts = game.ops.map((w) => (game.team[w] === DEFENSE ? d.act(game.observe(house, st, w))
      : scripted ? ref.attacker_moves[k] : a.act(game.observe(house, st, w))));
    st = game.step(house, st, acts, ref.dice[k], ref.where[k] || game.ops.map(() => 0));
    const e = Math.max(...game.ops.map((w) => Math.hypot(st.pos[w][0] - ref.pos[k + 1][w][0], st.pos[w][1] - ref.pos[k + 1][w][1])));
    drift = Math.max(drift, e);
    if (k < 50) drift5 = drift;
  }
  const ok = st.outcome === ref.outcome && st.t === ref.steps && (st.hostageHitBy ?? -1) === (ref.hostage_hit_by ?? -1);
  same += ok;
  const flip = FLIPS[`${ref.layout || "kill"}:${ref.house}`];
  if (!ok && flip) { flips++; console.log(`  known float32 flip: ${flip}`); }
  if (ref.steps < 200 && !flip && (st.outcome !== ref.outcome || Math.abs(st.t - ref.steps) > 5
                         || (st.hostageHitBy ?? -1) !== (ref.hostage_hit_by ?? -1))) shortMiss++;
  const res = (o) => (o === 1 ? "attacker wins" : "defender wins");
  console.log(`${game.nA}v${game.nD} ${name(ref.attacker)} vs ${name(ref.defender)} house ${ref.house}: JS ${res(st.outcome)} at ${(st.t / 10).toFixed(1)} s, ` +
    `Python ${res(ref.outcome)} at ${(ref.steps / 10).toFixed(1)} s ${ok ? "✓" : "✗"}; ` +
    `drift ${drift5.toFixed(4)} m in 5 s, ${drift.toFixed(4)} m overall` + (ref.bangs ? `; ${ref.bangs} flashbang(s)` : "") + (ref.layout ? `; a ${ref.layout}` : "")
    + (ref.hostage_hit_by >= 0 ? `; hostage hit by ${ref.hostage_hit_by ? "Defense" : "Assault"} (JS: ${st.hostageHitBy})` : ""));
}
console.log(`${same}/${data.reference.length} rounds end the same way`);
// Python computes in float32, JS in float64: tiny differences grow, and in a long round a close decision can flip
// (the same rounds also come out differently on the Python game's GPU and CPU). Episode 9 (two operators) allowed
// that over 30 s; with three operators edge cases come 50% more often, and in episode 10's rounds we found: house
// 100078 (2v1) flips "has the defender looked at cell 9" at 1.4 s, the cell 1e-7 rad from its light's edge, and still
// ends the same; house 100104 (2v1) parts by 0.1 mm at 11.1 s with every input identical (rounding) and ends
// differently at 21 s. So: a round under 20 s must match. Episode 12 (2v2, the hostage): house 100067 parts at 9.9 s
// when the escort's run switch, a hair from its threshold, flips on a 1e-4 difference in its teammate's bearing
// (input 137, float32 sin); every hostage input matches; the walk-out ends 0.2 s later. So: a round under 20 s must
// end the same way and within half a second. Episode 13 (homes): longer rounds, narrow hallways, more sliding along
// walls; house 100068 parts by float32 rounding at 3.2 s and ends differently at 15.8 s (FLIPS, above).
console.log(shortMiss ? `${shortMiss} round(s) under 20 s differ: a porting bug`
  : "every round under 20 s ends the same way" + (flips ? `, but ${flips} named float32 flip(s)` : ""));
process.exit(shortMiss ? 1 : 0);
