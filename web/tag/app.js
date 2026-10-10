// Tag vs the AI: you play the robot (chase) or the blob (run) against any generation from the videos.
// Same game, same rules, ten decisions a second for both of you, and you only see what your
// character sees. The AI's moves come from its real trained brain, running right here in your
// browser (engine.js, checked against the Python game: node check.mjs).
import { Arena, Brain, Game, CHASER, RUNNER } from "./engine.js";

const C = { bg: "#0d0f16", panel: "#161923", edge: "#262b3a", floorA: "#242a3a", floorB: "#282e3f", block: "#969eb2",
            blockSide: "#5c6378", fog: "#080a10", text: "#e8eaf0", muted: "#787f94", green: "#4ade80", amber: "#fbbf24",
            red: "#f43f5e" };
const canvas = document.getElementById("c"), g = canvas.getContext("2d");
const [data, buf] = await Promise.all([fetch("tag.json").then((r) => r.json()),
                                       fetch("weights.bin").then((r) => r.arrayBuffer())]);
const weights = new Float32Array(buf), game = new Game(data.cfg), cfg = data.cfg;
const arenas = data.arenas.map((a) => new Arena(a));
const brains = data.brains.map((b) => ({ ...b, brain: new Brain(b.layers, weights) }));
const LOOK = 1.9;  // characters are drawn bigger than their collision disc (its true size is the shadow)

// ── menu ──
const menu = document.getElementById("menu");
let side = localStorage.getItem("tag-side") || "robot", gen = Number(localStorage.getItem("tag-gen")) || 1;
const opponent = () => brains.find((b) => b.side !== side && b.gen === gen);
const score = (o) => JSON.parse(localStorage.getItem(`tag-score-${side}-${o.spec}`) || "[0,0]");
function renderMenu() {
  const o = opponent(), [won, played] = score(o), them = side === "robot" ? "blob" : "robot";
  for (const b of menu.querySelectorAll("[data-side]")) b.classList.toggle("on", b.dataset.side === side);
  for (const b of menu.querySelectorAll("[data-gen]")) {
    b.classList.toggle("on", Number(b.dataset.gen) === gen);
    b.textContent = `${them} ${b.dataset.gen}`;
  }
  document.getElementById("goal").textContent = side === "robot"
    ? "You chase. Touch the blob within 20 seconds." : "You run. Don't get touched for 20 seconds.";
  document.getElementById("info").innerHTML =
    `<b>${them} ${gen}</b> practised <b>${o.tries.toLocaleString()}</b> rounds. ` +
    `Its creator ${side === "robot" ? "caught it" : "got away from it"} <b>${o.creator} of 10</b> times.` +
    (played ? ` You: <b>${won} of ${played}</b>.` : "");
}
menu.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.side) side = b.dataset.side;
  if (b.dataset.gen) gen = Number(b.dataset.gen);
  if (b.id === "play") return start();
  localStorage.setItem("tag-side", side); localStorage.setItem("tag-gen", gen);
  renderMenu();
});

// ── input: arrows / WASD, or drag anywhere on a touch screen (a joystick from where you touched) ──
const keys = {};
const keyMap = { ArrowLeft: "l", KeyA: "l", ArrowRight: "r", KeyD: "r", ArrowUp: "u", KeyW: "u", ArrowDown: "d", KeyS: "d" };
addEventListener("keydown", (e) => {
  if (keyMap[e.code]) { keys[keyMap[e.code]] = true; e.preventDefault(); }
  if (phase === "menu" && e.code === "Enter") start();
  else if (phase === "over" && (e.code === "Space" || e.code === "Enter")) next();
  else if (phase !== "menu" && e.code === "Escape") showMenu();
});
addEventListener("keyup", (e) => { if (keyMap[e.code]) keys[keyMap[e.code]] = false; });
let stick = null;
canvas.addEventListener("pointerdown", (e) => {
  if (phase === "over") return next();
  stick = { x0: e.clientX, y0: e.clientY, x: 0, y: 0 };
});
canvas.addEventListener("pointermove", (e) => {
  if (!stick) return;
  const dx = e.clientX - stick.x0, dy = stick.y0 - e.clientY, n = Math.hypot(dx, dy);
  [stick.x, stick.y] = n < 12 ? [0, 0] : [dx / n, dy / n];
});
for (const ev of ["pointerup", "pointercancel", "pointerleave"]) canvas.addEventListener(ev, () => { stick = null; });
const myMove = () => stick ? [stick.x, stick.y]
  : [(keys.r ? 1 : 0) - (keys.l ? 1 : 0), (keys.u ? 1 : 0) - (keys.d ? 1 : 0)];  // world y is up

// ── a round ──
let phase = "menu", arena, st, prev, acc, countdown, trail, ai, me, them, result, tries = Number(localStorage.getItem("tag-tries")) || 0;
function showMenu() { phase = "menu"; menu.hidden = false; renderMenu(); }
function start() { menu.hidden = true; next(); }
function next() {
  ai = opponent(); me = side === "robot" ? CHASER : RUNNER; them = 1 - me;
  arena = arenas[Math.floor(Math.random() * arenas.length)];
  st = prev = game.init(arena); acc = 0; countdown = 3; trail = []; result = null; phase = "countdown";
  tries += 1; localStorage.setItem("tag-tries", tries);
}
function tick() {  // one decision each (10 per second)
  const acts = [];
  acts[me] = myMove();
  acts[them] = ai.brain.act(game.observe(arena, st, them));
  prev = st; st = game.step(arena, st, acts);
  trail.push(st.pos[me].slice()); if (trail.length > 30) trail.shift();
  if (st.done) {
    const won = me === CHASER ? st.caught : !st.caught, [w, p] = score(ai);
    localStorage.setItem(`tag-score-${side}-${ai.spec}`, JSON.stringify([w + won, p + 1]));
    result = { won, at: performance.now() }; phase = "over";
  }
}

// ── drawing ──
let W, H, dpr;
function resize() { dpr = devicePixelRatio || 1; W = innerWidth; H = innerHeight;
  canvas.width = W * dpr; canvas.height = H * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0); }
addEventListener("resize", resize); resize();
const fogCanvas = document.createElement("canvas"), fg = fogCanvas.getContext("2d");
const text = (s, x, y, size, color, align = "left", weight = 700) => {
  g.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  g.fillStyle = color; g.textAlign = align; g.textBaseline = "top"; g.fillText(s, x, y);
};

function drawArena(tf, s) {
  const S = cfg.size, [x0, y0] = tf(0, S);
  g.fillStyle = C.blockSide; g.beginPath(); g.roundRect(x0 - 0.8 * s, y0 - 0.8 * s, (S + 1.6) * s, (S + 1.6) * s, 0.4 * s); g.fill();
  for (let i = 0; i < S / 2; i++) for (let j = 0; j < S / 2; j++) {   // checkered floor, 2 m tiles
    g.fillStyle = (i + j) % 2 ? C.floorA : C.floorB;
    const [x, y] = tf(i * 2, (j + 1) * 2); g.fillRect(x, y, 2 * s + 0.5, 2 * s + 0.5);
  }
}
function drawBlocks(tf, s) {
  for (const [a, b, c, d] of arena.boxes) {
    const [x, y] = tf(a, d), w = (c - a) * s, h = (d - b) * s, side = Math.min(0.45 * s, h * 0.4);
    g.fillStyle = C.blockSide; g.beginPath(); g.roundRect(x, y, w, h, s * 0.25); g.fill();
    g.fillStyle = C.block; g.beginPath(); g.roundRect(x, y, w, h - side, s * 0.25); g.fill();
  }
}
function drawFog(tf, eye) {  // everything the player can't see from `eye` (behind obstacles), soft-edged
  fogCanvas.width = canvas.width; fogCanvas.height = canvas.height;
  fg.setTransform(dpr, 0, 0, dpr, 0, 0); fg.fillStyle = C.fog;
  for (const [a, b, c, d] of arena.boxes) {
    const corners = [[a, b], [c, b], [c, d], [a, d]];
    for (let k = 0; k < 4; k++) {   // each edge, pushed away from the eye: together they make the shadow
      const p = corners[k], q = corners[(k + 1) % 4];
      const far = (v) => { const dx = v[0] - eye[0], dy = v[1] - eye[1], n = Math.hypot(dx, dy) || 1;
                           return [v[0] + (dx / n) * 120, v[1] + (dy / n) * 120]; };
      fg.beginPath(); fg.moveTo(...tf(...p)); fg.lineTo(...tf(...q)); fg.lineTo(...tf(...far(q))); fg.lineTo(...tf(...far(p)));
      fg.closePath(); fg.fill();
    }
  }
  const [x0, y0] = tf(0, cfg.size), [x1, y1] = tf(cfg.size, 0);
  g.save(); g.beginPath(); g.rect(x0, y0, x1 - x0, y1 - y0); g.clip();   // the arena only
  g.globalAlpha = 0.72; g.filter = "blur(3px)"; g.drawImage(fogCanvas, 0, 0, W, H); g.restore();
}
function shadow(tf, s, p) {
  const [x, y] = tf(...p); g.fillStyle = "#00000070";
  g.beginPath(); g.ellipse(x, y + 0.25 * s, cfg.radius * 1.3 * s, cfg.radius * 0.7 * s, 0, 0, 2 * Math.PI); g.fill();
}
function eyes(x, y, r, look, pupil) {
  const n = Math.hypot(...look) || 1, lx = look[0] / n, ly = -look[1] / n;
  for (const k of [-1, 1]) {
    const ex = x + k * r * 0.42 + lx * r * 0.12, ey = y - r * 0.08 + ly * r * 0.12;
    g.fillStyle = "#fff"; g.beginPath(); g.arc(ex, ey, r * 0.27, 0, 2 * Math.PI); g.fill();
    g.fillStyle = pupil; g.beginPath(); g.arc(ex + lx * r * 0.12, ey + ly * r * 0.12, r * 0.13, 0, 2 * Math.PI); g.fill();
  }
}
function robot(tf, s, p, look) {
  const [x, y] = tf(...p), r = cfg.radius * LOOK * s;
  g.fillStyle = C.green; g.beginPath(); g.roundRect(x - r, y - r, 2 * r, 2 * r, r * 0.45); g.fill();
  g.fillStyle = "#0b1a12"; g.beginPath(); g.roundRect(x - r * 0.75, y - r * 0.55, r * 1.5, r * 1.05, r * 0.3); g.fill();
  eyes(x, y, r, look, C.green);
}
function blob(tf, s, p, look) {
  const [x, y] = tf(...p), r = cfg.radius * LOOK * s;
  g.fillStyle = C.amber; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill();
  g.strokeStyle = "#a16207"; g.lineWidth = r * 0.12; g.stroke();
  eyes(x, y, r, look, "#111");
}

function draw(alpha) {
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  if (phase === "menu") return;
  const hud = Math.min(150, H * 0.2), side_ = Math.min(W - 24, H - hud - 24), s = side_ / (cfg.size + 1.6);
  const ox = (W - side_) / 2 + 0.8 * s, oy = hud + 12 + 0.8 * s;
  const tf = (x, y) => [ox + x * s, oy + (cfg.size - y) * s];
  const a = phase === "playing" ? alpha : 1;
  const pos = [0, 1].map((w) => [prev.pos[w][0] + (st.pos[w][0] - prev.pos[w][0]) * a,
                                prev.pos[w][1] + (st.pos[w][1] - prev.pos[w][1]) * a]);
  const over = phase === "over", seen = st.seen || over;
  drawArena(tf, s);
  if (!over) drawFog(tf, pos[me]);
  drawBlocks(tf, s);
  if (trail.length > 1) {
    g.strokeStyle = me === CHASER ? C.green : C.amber; g.globalAlpha = 0.45; g.lineWidth = Math.max(2, s * 0.15);
    g.beginPath(); trail.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, ...tf(...p))); g.stroke(); g.globalAlpha = 1;
  }
  if (!seen) {  // where I last saw it
    const [x, y] = tf(...st.mem[me]), r = cfg.radius * LOOK * s * 0.6;
    g.setLineDash([r * 0.5, r * 0.4]); g.strokeStyle = them === CHASER ? C.green : C.amber; g.lineWidth = Math.max(2, r * 0.18);
    g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
    text("?", x, y - r * 0.7, r * 1.4, them === CHASER ? C.green : C.amber, "center");
  }
  const toward = (w) => [pos[1 - w][0] - pos[w][0], pos[1 - w][1] - pos[w][1]];
  for (const w of [RUNNER, CHASER]) {
    if (w === them && !seen) continue;
    shadow(tf, s, pos[w]);
    (w === CHASER ? robot : blob)(tf, s, pos[w], seen ? toward(w) : st.vel[w]);
  }
  if (over && st.caught) {
    const age = (performance.now() - result.at) / 1000, [x, y] = tf(...st.pos[RUNNER]);
    if (age < 1.2) { g.strokeStyle = C.green; g.globalAlpha = 1 - age / 1.2; g.lineWidth = 6;
      g.beginPath(); g.arc(x, y, s * (1 + age * 6), 0, 2 * Math.PI); g.stroke(); g.globalAlpha = 1; }
  }

  // HUD
  const pad = 16, small = Math.max(13, Math.min(W, H) / 48), t = st.t * cfg.dt, left = Math.max(0, cfg.time_limit - t);
  const them_ = side === "robot" ? "blob" : "robot", [won, played] = score(ai);
  text(`You: the ${side} · vs ${them_} ${gen} (AI)`, pad, pad, small * 1.5, me === CHASER ? C.green : C.amber);
  text(`you ${won} of ${played} vs this AI · its creator ${ai.creator} of 10 · try #${tries}`, pad, pad + small * 2.1,
       small, C.muted, "left", 600);
  const bw = Math.min(W - 2 * pad - small * 9, 520), by = pad + small * 3.8;   // room for the time on its right
  g.fillStyle = C.panel; g.beginPath(); g.roundRect(pad, by, bw, small * 0.9, small * 0.45); g.fill();
  g.fillStyle = left < 5 ? C.red : C.amber; g.beginPath(); g.roundRect(pad, by, (bw * left) / cfg.time_limit, small * 0.9, small * 0.45); g.fill();
  text(over ? (st.caught ? `TAGGED in ${t.toFixed(1)} s` : "ESCAPED!") : `${left.toFixed(1)} s left`, pad + bw + 12, by - small * 0.15,
       small * 1.1, over ? (st.caught ? C.green : C.amber) : C.text, "left");
  if (W >= 900) text("arrows / WASD (or drag) · Esc: change opponent", W - pad, pad, small * 0.85, C.muted, "right", 500);
  if (phase === "countdown") text(String(Math.ceil(countdown)), W / 2, H / 2 - small * 4, small * 8, C.text, "center");
  if (over) {
    g.fillStyle = "#0d0f16cc"; g.fillRect(0, H * 0.38, W, H * 0.24);
    text(result.won ? "YOU WIN" : "AI WINS", W / 2, H * 0.41, small * 4, result.won ? C.green : C.red, "center");
    text("Space or tap: next round · Esc: change opponent", W / 2, H * 0.41 + small * 5, small * 1.1, C.text, "center", 600);
  }
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (phase === "countdown") { countdown -= dt; if (countdown <= 0) phase = "playing"; }
  else if (phase === "playing") { acc += dt; while (acc >= cfg.dt && phase === "playing") { acc -= cfg.dt; tick(); } }
  draw(Math.min(1, acc / cfg.dt));
  requestAnimationFrame(frame);
}
showMenu();
requestAnimationFrame(frame);
