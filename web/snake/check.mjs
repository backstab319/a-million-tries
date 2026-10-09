// node check.mjs: replay the Python AI's game (snake.json) in the JS game + brain.
import { readFileSync } from "node:fs";
import { Snake, SnakeBrain } from "./engine.js";

const data = JSON.parse(readFileSync(new URL("./snake.json", import.meta.url)));
const buf = readFileSync(new URL("./weights.bin", import.meta.url));
const brain = new SnakeBrain(data.layers, data.body, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const game = new Snake(data.cfg), ref = data.reference;
let st = game.init(ref.food[0]), maxDiff = 0, same = 0, t0 = performance.now();
for (let k = 0; k < ref.action.length; k++) {
  const s = brain.scores(game.observe(st));
  maxDiff = Math.max(maxDiff, ...s.map((v, i) => Math.abs(v - ref.logits[k][i])));
  if (s.indexOf(Math.max(...s)) === ref.action[k]) same++;
  st = game.step(st, ref.action[k], ref.food[k + 1]);
  if (st.length !== ref.length[k]) { console.log(`length differs at move ${k}`); break; }
}
const ms = (performance.now() - t0) / ref.action.length;
console.log(`moves ${ref.action.length}: same choice ${same}/${ref.action.length}, max score difference ${maxDiff.toExponential(2)}, ` +
            `final length ${st.length} (python ${ref.length.at(-1)}), won ${st.won} (python ${ref.result}), ${ms.toFixed(1)} ms per move`);
