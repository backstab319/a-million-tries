// Plays the AI's game in the background and streams its moves to the page. The AI's game
// doesn't depend on yours, so it can think ahead of what's shown.
import { Snake, SnakeBrain } from "./engine.js";

let game, brain, run = 0;
const ready = (async () => {
  const data = await (await fetch("snake.json")).json();
  const buf = await (await fetch("weights.bin")).arrayBuffer();
  game = new Snake(data.cfg); brain = new SnakeBrain(data.layers, data.body, buf);
})();

onmessage = async (e) => {
  if (e.data.type !== "start") return;
  await ready;
  const id = ++run;
  let st = game.init();
  postMessage({ id, st });
  while (id === run && !st.dead && !st.won && !st.starved) {
    st = game.step(st, brain.act(game.observe(st)));
    postMessage({ id, st });
    await new Promise((r) => setTimeout(r, 0));      // let a restart message in
  }
};
