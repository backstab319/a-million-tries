// Checks the JS booster (engine.js) against the Python sim: node web/rocket/check.mjs
// 1. the senses at the start; 2. Python's flights replayed action for action (open loop), state for state;
// 3. the AI's network, JS v Python, on the same senses; 4. the AI flying live in JS: Python's starts, then 200 new
// starts drawn in JS (its land rate should be near the video's 98%).
import {readFileSync} from "fs";
import {Booster, Pilot, LANDED, FLYING} from "./engine.js";

const here = new URL(".", import.meta.url).pathname;
const cfg = JSON.parse(readFileSync(here + "data/config.json"));
const pilotData = JSON.parse(readFileSync(here + "data/pilot.json"));
const refs = JSON.parse(readFileSync(here + "data/ref.json"));
const pilot = new Pilot(pilotData.weights);
const FIELDS = ["x", "y", "vx", "vy", "tilt", "spin", "fuel", "thrust", "gimbal", "fin", "heat"];
let ok = true;
const fail = msg => { ok = false; console.log("FAIL", msg); };

for (const r of refs) {
  const b = new Booster(cfg).setStart(r.start);
  const o = b.obs();
  const dObs = Math.max(...o.map((v, i) => Math.abs(v - r.obs0[i])));
  if (dObs > 1e-12) fail(`seed ${r.seed}: start senses differ by ${dObs}`);
  // open-loop replay: how many steps agree to 1e-9 (relative), and where they first part
  const n = r.actions.length - 1;
  // (the dynamics amplify last-bit rounding, so the two part slowly late in the flight, as the JAX port's did:
  // playbook 7. Agreeing to 1e-9 for the first ~60 s and ending the same way is the bar.)
  let agree = -1, end = 0;
  for (let i = 0; i < n && b.outcome === FLYING; i++) {
    b.step(r.actions[i]);
    let d = 0;
    for (const f of FIELDS) {
      const want = r.states[f][i + 1];
      d = Math.max(d, Math.abs(b[f] - want) / Math.max(1, Math.abs(want)));
    }
    if (d >= 1e-9 && agree < 0) agree = i;
    end = d;
  }
  if (agree < 0) agree = n;
  const outcome = ["flying", "landed", "crashed", "timed out"][b.outcome];
  console.log(`seed ${r.seed}: start senses ${dObs.toExponential(1)}; replay agrees to 1e-9 for ${agree} of ${n} steps` +
              ` (${(agree * cfg.dt).toFixed(0)} s), ${end.toExponential(1)} apart at the end; ended ${outcome} (Python: ${r.outcome})`);
  if (agree * cfg.dt < 60) fail(`seed ${r.seed}: replay parted at step ${agree}`);
  if (end > 1e-3) fail(`seed ${r.seed}: ${end} apart at the end`);
  if (outcome !== r.outcome) fail(`seed ${r.seed}: outcome ${outcome} v ${r.outcome}`);
  const a = pilot.act(r.obs0), dA = Math.max(...a.map((v, i) => Math.abs(v - r.actions[0][i])));
  console.log(`  the AI's first action, JS v Python (float32): ${dA.toExponential(1)}`);
  if (dA > 1e-4) fail(`seed ${r.seed}: pilot differs by ${dA}`);
}

function fly(start) {  // the AI flying live in JS
  const b = new Booster(cfg).setStart(start);
  while (b.outcome === FLYING) b.step(pilot.act(b.obs()));
  return b;
}
for (const r of refs) {
  const b = fly(r.start);
  console.log(`seed ${r.seed}, AI flying live in JS: ${b.why} at ${b.time.toFixed(1)} s (Python: ${r.outcome})`);
}
let seed = 12345;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
let landed = 0;
const N = 200;
for (let k = 0; k < N; k++) if (fly(Booster.draw(cfg, rand)).outcome === LANDED) landed++;
console.log(`AI on ${N} new starts drawn in JS: ${(100 * landed / N).toFixed(1)}% landed (Python, 4000 flights: 98.0%)`);
if (landed / N < 0.9) fail("the AI lands much less often in JS");
console.log(ok ? "OK" : "FAILED");
process.exit(ok ? 0 : 1);
