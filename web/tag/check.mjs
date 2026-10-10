// node check.mjs: the JS port against the Python game (AI vs AI reference rounds in tag.json).
import { readFileSync } from "node:fs";
import { Arena, Brain, Game, CHASER, RUNNER } from "./engine.js";

const data = JSON.parse(readFileSync(new URL("./tag.json", import.meta.url)));
const buf = readFileSync(new URL("./weights.bin", import.meta.url));
const weights = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const game = new Game(data.cfg);
const brain = (spec) => new Brain(data.brains.find((b) => b.spec === spec).layers, weights);
let same = 0;
for (const ref of data.reference) {
  const arena = new Arena(data.arenas.find((a) => a.seed === ref.arena));
  const chaser = brain(ref.chaser), runner = brain(ref.runner);
  let st = game.init(arena, ref.pos[0]), err5 = 0, err = 0;
  for (let k = 0; k < ref.steps && !st.done; k++) {
    st = game.step(arena, st, [chaser.act(game.observe(arena, st, CHASER)), runner.act(game.observe(arena, st, RUNNER))]);
    const e = Math.max(...[0, 1].map((w) => Math.hypot(st.pos[w][0] - ref.pos[k + 1][w][0], st.pos[w][1] - ref.pos[k + 1][w][1])));
    err = Math.max(err, e);
    if (k < 50) err5 = err;
  }
  const ok = st.caught === ref.caught && st.t === ref.steps;
  same += ok;
  console.log(`${ref.chaser.split("/")[1]} vs ${ref.runner.split("/")[1]} arena ${ref.arena}: ` +
    `JS ${st.caught ? "caught" : "escaped"} at ${(st.t / 10).toFixed(1)} s, Python ${ref.caught ? "caught" : "escaped"} at ` +
    `${(ref.steps / 10).toFixed(1)} s ${ok ? "✓" : "✗"}; max drift ${err5.toFixed(4)} m in 5 s, ${err.toFixed(4)} m overall`);
}
console.log(`${same}/${data.reference.length} rounds end the same way`);
