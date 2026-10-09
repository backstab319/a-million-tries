// node check.mjs: the JS port against the Python game (reference drives in data.json).
import { readFileSync } from "node:fs";
import { Game, Brain, decode } from "./engine.js";

const data = JSON.parse(readFileSync(new URL("./data.json", import.meta.url)));
const game = new Game(data), ref = data.reference;
const lapTime = (states) => {
  const k = states.findIndex((s) => s.progress >= game.length);
  return k < 0 ? null : Math.round((k + 1) * data.cfg.dt * 10) / 10;
};

// 1. the owner's lap: same inputs -> same path?
const act = decode(ref.me_action).data, mePos = decode(ref.me_pos).data;
let st = game.init(), states = [], err = 0;
for (let k = 0; k < act.length / 2; k++) {
  st = game.step(st, act[2 * k], act[2 * k + 1]);
  states.push(st);
  err = Math.max(err, Math.hypot(st.x - mePos[2 * (k + 1)], st.y - mePos[2 * (k + 1) + 1]));
}
console.log(`owner replay: lap ${lapTime(states)} s (python ${ref.me_lap}), max position error ${err.toFixed(4)} m, crashed ${states.some((s) => s.crashed)}`);

// 2. the AI driving itself in JS
const brain = new Brain(data.brain), aiPos = decode(ref.ai_pos).data;
st = game.init(); states = []; err = 0;
let errAt10 = null;
for (let k = 0; k < 400 && !st.crashed; k++) {
  const a = brain.act(game.observe(st));
  st = game.step(st, a.steer, a.throttle);
  states.push(st);
  if (2 * (k + 1) < aiPos.length) err = Math.max(err, Math.hypot(st.x - aiPos[2 * (k + 1)], st.y - aiPos[2 * (k + 1) + 1]));
  if (k === 99) errAt10 = err;
}
console.log(`AI in JS: lap ${lapTime(states)} s (python ${ref.ai_lap}), max drift ${errAt10.toFixed(3)} m in 10 s, ${err.toFixed(3)} m in 40 s, crashed ${st.crashed}`);
