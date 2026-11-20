// Check the JS sim (sim.js) against the Python one: replay data/reference.json's fights (Python's start state and
// actions) and compare every step's state and both jets' senses; and
// the pilot's network: its outputs on Python's senses against Python's (every 20th step).
//   node web/dogfight-play/check.mjs
import { readFileSync } from "node:fs";
import { Fighters, Pilot, norm, sub } from "./sim.js";

const here = new URL(".", import.meta.url).pathname;
const cfg = JSON.parse(readFileSync(here + "data/config.json"));
const fights = JSON.parse(readFileSync(here + "data/reference.json"));
const sim = new Fighters(cfg);
const pilot = new Pilot(JSON.parse(readFileSync(here + "data/pilot.json")));
let worstNet = 0;
let worst = { pos: 0, obs: 0, health: 0 }, bad = 0;
for (const f of fights) {
  let st = sim.fromState(f.start);
  f.steps.forEach((ref, k) => {
    const obs = [sim.observe(st, 0), sim.observe(st, 1)];
    const dObs = Math.max(...obs.flatMap((o, i) => o.map((x, m) => Math.abs(x - ref.obs[i][m]))));
    if (k % 20 === 0) for (const i of [0, 1]) worstNet = Math.max(worstNet, ...pilot.mean(ref.obs[i]).map((x, m) => Math.abs(x - ref.m[i][m])));
    st = sim.step(st, ref.a).st;
    const dPos = Math.max(norm(sub(st.pos[0], ref.pos[0])), norm(sub(st.pos[1], ref.pos[1])));
    const dH = Math.max(Math.abs(st.health[0] - ref.health[0]), Math.abs(st.health[1] - ref.health[1]));
    worst = { pos: Math.max(worst.pos, dPos), obs: Math.max(worst.obs, dObs), health: Math.max(worst.health, dH) };
    if ((dPos > 1.0 || dObs > 1e-2 || dH > 1e-3 || st.cause.join() !== ref.cause.join()) && bad++ < 5)
      console.log(`fight ${f.seed} step ${k}: pos off ${dPos.toFixed(3)} m, senses off ${dObs.toExponential(2)}, health off ${dH.toExponential(2)}, cause ${st.cause} v ${ref.cause}`);
  });
}
console.log(`${fights.length} fights, ${fights.reduce((s, f) => s + f.steps.length, 0)} steps: worst position ${worst.pos.toFixed(3)} m, senses ${worst.obs.toExponential(2)}, health ${worst.health.toExponential(2)}; ${bad} steps over tolerance`);
console.log(`pilot network: worst output off ${worstNet.toExponential(2)}`);
process.exit(bad || worstNet > 1e-3 ? 1 : 0);
