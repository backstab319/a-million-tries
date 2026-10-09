// Beat my AI at Snake: your board on one side, the AI's on the other, same rules, same
// speed. The AI is the real trained brain (5 conv layers) running in your browser.
import { Snake, TURN_LEFT, TURN_RIGHT, STRAIGHT } from "./engine.js";

const C = { bg: "#0d0f16", panel: "#161923", grid: "#1c2030", edge: "#262b3a", text: "#e8eaf0", muted: "#787f94",
            green: "#4ade80", amber: "#fbbf24", red: "#f43f5e" };
const canvas = document.getElementById("c"), g = canvas.getContext("2d");
const data = await (await fetch("snake.json")).json();
const game = new Snake(data.cfg), S = data.cfg.size, CELLS = S * S;
const SPEEDS = [5, 7, 10], worker = new Worker("worker.js", { type: "module" });

let me, ai, aiQueue = [], aiFast = false, wanted = [], phase, countdown, acc = 0, speedIx = 1, runId = 0;
let best = Number(localStorage.getItem("bms-best")) || 0;
worker.onmessage = (e) => { if (e.data.id === runId) aiQueue.push(e.data.st); };

function restart() {
  me = game.init(); ai = null; aiQueue = []; aiFast = false; wanted = []; phase = "countdown"; countdown = 3; acc = 0;
  runId += 1; worker.postMessage({ type: "start" });
}
restart();
const over = (st) => st && (st.dead || st.won || st.starved);

// ── input: arrows / WASD / swipes give a direction; we turn it into straight / right / left ──
const dirKeys = { ArrowUp: 0, KeyW: 0, ArrowRight: 1, KeyD: 1, ArrowDown: 2, KeyS: 2, ArrowLeft: 3, KeyA: 3 };
const want = (d) => { if (wanted.length < 2) wanted.push(d); };
addEventListener("keydown", (e) => {
  if (e.code in dirKeys) { want(dirKeys[e.code]); e.preventDefault(); }
  if (e.code === "KeyR" || e.code === "Enter") restart();
  if (e.code === "KeyF") aiFast = !aiFast;
  if (e.code === "Digit1" || e.code === "Digit2" || e.code === "Digit3") speedIx = Number(e.code.at(-1)) - 1;
});
let touch = null;
canvas.addEventListener("pointerdown", (e) => { touch = [e.clientX, e.clientY]; });
canvas.addEventListener("pointerup", (e) => {
  if (!touch) return;
  const dx = e.clientX - touch[0], dy = e.clientY - touch[1]; touch = null;
  if (Math.hypot(dx, dy) < 20) { if (over(me) && phase !== "countdown") { if (over(ai)) restart(); else aiFast = !aiFast; } return; }
  want(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : (dy > 0 ? 2 : 0));
});
function myMove() {
  while (wanted.length) {
    const d = wanted.shift();
    if (d === me.dir) return STRAIGHT;
    if (d === (me.dir + 1) % 4) return TURN_RIGHT;
    if (d === (me.dir + 3) % 4) return TURN_LEFT;      // a U-turn isn't a move: try the next key
  }
  return STRAIGHT;
}

function tick() {
  if (!over(me)) {
    me = game.step(me, myMove());
    if (me.length - 3 > best) { best = me.length - 3; localStorage.setItem("bms-best", best); }
  }
  const n = over(me) && aiFast ? 6 : 1;               // after you're out, the AI can play on fast
  for (let k = 0; k < n && !over(ai) && aiQueue.length; k++) ai = aiQueue.shift();
}

// ── drawing ──
let W, H;
function resize() { const d = devicePixelRatio || 1; W = innerWidth; H = innerHeight;
  canvas.width = W * d; canvas.height = H * d; g.setTransform(d, 0, 0, d, 0, 0); }
addEventListener("resize", resize); resize();
const text = (s, x, y, size, color, align = "left", weight = 700) => {
  g.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  g.fillStyle = color; g.textAlign = align; g.textBaseline = "top"; g.fillText(s, x, y);
};

function drawBoard(st, x, y, size, color, label) {
  const cell = size / S;
  g.fillStyle = C.panel; g.fillRect(x, y, size, size);
  for (let r = 0; r < S; r++) for (let c = 0; c < S; c++)
    if ((r + c) % 2) { g.fillStyle = C.grid; g.fillRect(x + c * cell, y + r * cell, cell, cell); }
  g.strokeStyle = C.edge; g.lineWidth = 2; g.strokeRect(x, y, size, size);
  if (!st) { text("loading…", x + size / 2, y + size / 2 - 10, 18, C.muted, "center", 600); return; }
  const [fr, fc] = st.food;
  if (!st.won) { g.fillStyle = C.red; g.beginPath(); g.arc(x + (fc + 0.5) * cell, y + (fr + 0.5) * cell, cell * 0.32, 0, 7); g.fill(); }
  for (let i = 0; i < CELLS; i++) {
    const v = st.board[i]; if (!v) continue;
    const r = Math.floor(i / S), c = i % S, head = r === st.head[0] && c === st.head[1];
    g.globalAlpha = head ? 1 : 0.45 + 0.55 * (v / st.length);
    g.fillStyle = st.dead ? C.red : color;
    const m = head ? cell * 0.06 : cell * 0.12;
    g.beginPath(); g.roundRect(x + c * cell + m, y + r * cell + m, cell - 2 * m, cell - 2 * m, cell * 0.22); g.fill();
  }
  g.globalAlpha = 1;
  const status = st.won ? "FILLED THE BOARD!" : st.dead ? "CRASHED" : st.starved ? "STARVED" : "";
  text(label, x, y - 34, 24, color);
  text(`${st.length - 3} apples`, x + size, y - 34, 24, C.text, "right");
  if (status) {
    g.fillStyle = "#0d0f16b0"; g.fillRect(x, y + size / 2 - 34, size, 68);
    text(status, x + size / 2, y + size / 2 - 16, 30, st.won ? C.green : C.red, "center");
  }
}

function draw() {
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  const wide = W >= H, top = 90;
  const size = wide ? Math.min((W - 120) / 2, H - top - 110) : Math.min(W - 40, (H - top - 170) / 2);
  const [x1, y1, x2, y2] = wide ? [W / 2 - size - 30, top + 40, W / 2 + 30, top + 40]
                                : [(W - size) / 2, top + 40, (W - size) / 2, top + 40 + size + 60];
  text("Beat my AI: Snake", 20, 18, 28, C.text);
  text(`The AI fills the whole board in 92% of its games. Your best: ${best} apples (99 fills it).`, 20, 54, 15, C.muted, "left", 500);
  drawBoard(me, x1, y1, size, C.amber, "YOU");
  drawBoard(ai, x2, y2, size, C.green, "THE AI");
  const help = over(me)
    ? (over(ai) ? "R or tap: play again" : `F or tap: ${aiFast ? "normal speed" : "fast-forward the AI"} · R: play again`)
    : `arrows / WASD / swipe · speed ${SPEEDS[speedIx]} moves/s (1-3) · R: restart`;
  text(help, W / 2, H - 40, 16, C.muted, "center", 500);
  if (phase === "countdown") text(String(Math.ceil(countdown)), W / 2, H / 2 - 60, 120, C.amber, "center");
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.25, (now - last) / 1000); last = now;
  if (phase === "countdown") { countdown -= dt; if (countdown <= 0 && aiQueue.length) { phase = "playing"; ai = aiQueue.shift(); } }
  else { acc += dt; const step = 1 / SPEEDS[speedIx]; while (acc >= step) { acc -= step; tick(); } }
  draw(); requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
