// node race-check.mjs: the JS port of the racing sandbox against the Python game (reference runs in race-data.json).
// 1. Python's inputs replayed: does the car follow the same path, and does the driver see the same numbers?
// 2. The JS driver drives itself: does it get as far in 60 s as it did in Python? (the general driver, and each
//    finalist with its own driver, as in the video's final race)
import { readFileSync } from "node:fs";
import { decode } from "./engine.js";
import { RaceGame, Driver } from "./race-engine.js";

const data = JSON.parse(readFileSync(new URL("./race-data.json", import.meta.url)));
const game = new RaceGame(data);
const drivers = Object.fromEntries(Object.entries(data.drivers).map(([n, b]) => [n, new Driver(b)]));
let bad = 0;

for (const ref of data.reference) {
  const driver = drivers[ref.driver], design = data.cars[ref.car], pos = decode(ref.pos).data, act = decode(ref.action).data;
  const obs = decode(ref.obs), nObs = obs.shape[1];

  // 1. open loop
  // float32 (JAX) vs doubles (JS): at the grip limit a millimetre grows (a tyre on the edge of locking under braking
  // goes one way or the other), so open-loop paths split in the end, as early as ~9 s; the senses are compared while
  // the paths agree within 1 mm, the drift is judged over the first 5 s
  let st = game.init(design, ref.circuit), err5 = 0, err = 0, obsErr = 0, split = null, close = true;
  for (let k = 0; k < ref.steps; k++) {
    if (k % 50 === 0 && close) {
      const o = game.observe(st);
      for (let i = 0; i < nObs; i++) obsErr = Math.max(obsErr, Math.abs(o[i] - obs.data[(k / 50) * nObs + i]));
    }
    st = game.step(st, act[2 * k], act[2 * k + 1]);
    const e = Math.hypot(st.car.x - pos[2 * (k + 1)], st.car.y - pos[2 * (k + 1) + 1]);
    err = Math.max(err, e);
    if (k < 100) err5 = err;
    if (e > 0.001) close = false;
    if (split === null && e > 0.01) split = (k + 1) * data.cfg.dt;
  }

  // 2. closed loop
  st = game.init(design, ref.circuit);
  while (!st.done) {
    const [s, p] = driver.act(game.observe(st));
    st = game.step(st, s, p);
  }
  const gap = st.progress - ref.progress;
  const ok = err5 < 0.005 && obsErr < 1e-3 && st.crashed === ref.crashed && Math.abs(gap) < 0.02 * ref.progress;
  if (!ok) bad++;
  console.log(`${ok ? "ok " : "BAD"} ${`${ref.car} (${ref.driver === "general" ? "general" : "own"})`.padEnd(20)} on ${ref.circuit}: replay drift ${(err5 * 1000).toFixed(2)} mm in 5 s, ` +
    `paths split (1 cm) at ${split === null ? "never" : split.toFixed(1) + " s"}, senses off by ${obsErr.toExponential(1)}; ` +
    `JS driver ${st.progress.toFixed(0)} m vs python ${ref.progress.toFixed(0)} (${gap >= 0 ? "+" : ""}${gap.toFixed(1)}), ` +
    `crashed ${st.crashed} (python ${ref.crashed})`);
}
console.log(bad ? `${bad} of ${data.reference.length} runs differ` : `all ${data.reference.length} runs match`);
process.exit(bad ? 1 : 0);
