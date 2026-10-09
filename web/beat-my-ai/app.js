// Beat my AI: you and the AI start together on the same track. Same game, same physics,
// ten decisions a second for both of you. The AI's moves come from its real trained brain,
// running right here in your browser (engine.js, checked against the Python game).
import { Game, Brain } from "./engine.js";

const C = { bg: "#0d0f16", panel: "#161923", edge: "#262b3a", asphalt: "#222634", kerb: "#9a9eb2", text: "#e8eaf0",
            muted: "#787f94", green: "#4ade80", amber: "#fbbf24", red: "#f43f5e" };
const CREATOR_LAP = 34.7;                 // the owner's best keyboard lap (episode 5)
const canvas = document.getElementById("c"), g = canvas.getContext("2d");
const data = await (await fetch("data.json")).json();
const game = new Game(data), brain = new Brain(data.brain), cfg = data.cfg;
const AI_LAP = data.reference.ai_lap;

// ── input: keyboard is on/off, so the wheel eases towards full lock (like the desktop game) ──
const keys = { left: false, right: false, gas: false, brake: false };
const keyMap = { ArrowLeft: "left", KeyA: "left", ArrowRight: "right", KeyD: "right",
                 ArrowUp: "gas", KeyW: "gas", ArrowDown: "brake", KeyS: "brake" };
addEventListener("keydown", (e) => {
  if (keyMap[e.code]) { keys[keyMap[e.code]] = true; e.preventDefault(); }
  if (e.code === "KeyR" || e.code === "Space" || e.code === "Enter") restart();
  if (e.code === "KeyL") showLasers = !showLasers;
});
addEventListener("keyup", (e) => { if (keyMap[e.code]) keys[keyMap[e.code]] = false; });
for (const b of document.querySelectorAll(".touch button")) {
  const set = (v) => (e) => { keys[b.dataset.k] = v; b.classList.toggle("on", v); e.preventDefault(); };
  b.addEventListener("pointerdown", set(true)); b.addEventListener("pointerup", set(false));
  b.addEventListener("pointerleave", set(false)); b.addEventListener("pointercancel", set(false));
}
canvas.addEventListener("pointerdown", () => { if (phase === "crashed" || phase === "finished") restart(); });

// ── race state ──
let me, mePrev, ai, aiPrev, aiLasers, steer, acc, phase, countdown, meLap, aiLapDone, showLasers = true;
let best = Number(localStorage.getItem("bma-best")) || null;
let tries = Number(localStorage.getItem("bma-tries")) || 0;

function restart() {
  me = mePrev = game.init(); ai = aiPrev = game.init(); aiLasers = game.lasers(ai);
  steer = 0; acc = 0; phase = "countdown"; countdown = 3; meLap = null; aiLapDone = null;
  tries += 1; localStorage.setItem("bma-tries", tries);
}
restart();

function tick() {  // one decision for each driver (10 per second)
  const want = (keys.left ? 1 : 0) - (keys.right ? 1 : 0);     // + = left
  steer += Math.max(-0.35, Math.min(0.35, want - steer));
  const throttle = (keys.gas ? 1 : 0) - (keys.brake ? 1 : 0);
  mePrev = me; me = game.step(me, steer, throttle);
  if (aiLapDone === null || ai.t * cfg.dt < AI_LAP + 3) {     // the AI keeps going a little past the line
    const a = brain.act(game.observe(ai, aiLasers));
    aiPrev = ai; ai = game.step(ai, a.steer, a.throttle); aiLasers = game.lasers(ai);
    if (aiLapDone === null && ai.progress >= game.length) aiLapDone = ai.t * cfg.dt;
  } else aiPrev = ai;
  if (me.crashed) phase = "crashed";
  else if (me.progress >= game.length) {
    phase = "finished"; meLap = Math.round(me.t * cfg.dt * 10) / 10;
    if (best === null || meLap < best) { best = meLap; localStorage.setItem("bma-best", best); }
  }
}

// ── drawing ──
let W, H, dpr;
function resize() { dpr = devicePixelRatio || 1; W = innerWidth; H = innerHeight;
  canvas.width = W * dpr; canvas.height = H * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0); }
addEventListener("resize", resize); resize();

const P = game.P, A = game.wallA, B = game.wallB;
function wallPath(path, off, tf) {
  path.moveTo(...tf(A[2 * off], A[2 * off + 1]));
  for (let s = off; s < off + P; s++) path.lineTo(...tf(B[2 * s], B[2 * s + 1]));
  path.closePath();
}
function drawTrack(tf, scale, detail) {
  const road = new Path2D(); wallPath(road, 0, tf); wallPath(road, P, tf);
  g.fillStyle = C.asphalt; g.fill(road, "evenodd");
  g.strokeStyle = C.kerb; g.lineWidth = Math.max(1, scale * (detail ? 0.7 : 0.25)); g.stroke(road);
  if (!detail) return;
  const c = game.center, h = game.headings[0], nx = -Math.sin(h), ny = Math.cos(h), half = cfg.width / 2;
  for (let k = 0; k < 8; k++) {   // chequered start line
    const u0 = -half + (k * cfg.width) / 8, u1 = u0 + cfg.width / 8;
    g.strokeStyle = k % 2 ? "#05060a" : C.text; g.lineWidth = scale * 1.2; g.beginPath();
    g.moveTo(...tf(c[0] + nx * u0, c[1] + ny * u0)); g.lineTo(...tf(c[0] + nx * u1, c[1] + ny * u1)); g.stroke();
  }
}
function drawCar(tf, scale, s, color, alpha, outline) {
  const [x, y] = tf(s.x, s.y);
  g.save(); g.globalAlpha = alpha; g.translate(x, y); g.rotate(-s.heading);
  const l = Math.max(4.6 * scale, 10), w = Math.max(2 * scale, 5);
  g.fillStyle = color; g.beginPath(); g.roundRect(-l / 2, -w / 2, l, w, w * 0.3); g.fill();
  if (outline) { g.strokeStyle = C.bg; g.lineWidth = 2; g.stroke(); }
  g.fillStyle = "#05060a"; g.fillRect(l * 0.05, -w * 0.32, l * 0.18, w * 0.64);   // windscreen
  g.restore();
}
const lerp = (a, b, t) => a + (b - a) * t;
const blend = (p, q, t) => ({ ...q, x: lerp(p.x, q.x, t), y: lerp(p.y, q.y, t),
                               heading: lerp(p.heading, q.heading, t) });
const text = (s, x, y, size, color, align = "left", weight = 700) => {
  g.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  g.fillStyle = color; g.textAlign = align; g.textBaseline = "top"; g.fillText(s, x, y);
};
const secs = (st) => (st.t * cfg.dt).toFixed(1);

function draw(alpha) {
  const m = blend(mePrev, me, phase === "racing" ? alpha : 1), a = blend(aiPrev, ai, phase === "racing" ? alpha : 1);
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  const scale = Math.min(W, H) / 70;                                 // pixels per metre
  const tf = (x, y) => [W / 2 + (x - m.x) * scale, H / 2 - (y - m.y) * scale];
  drawTrack(tf, scale, true);
  if (showLasers && phase !== "countdown") {                        // what the AI sees
    g.strokeStyle = C.green; g.globalAlpha = 0.35; g.lineWidth = 1.5;
    game.rays.forEach((r, i) => {
      const d = aiLasers[i], h = ai.heading + r;
      g.beginPath(); g.moveTo(...tf(ai.x, ai.y)); g.lineTo(...tf(ai.x + d * Math.cos(h), ai.y + d * Math.sin(h))); g.stroke();
    });
    g.globalAlpha = 1;
  }
  drawCar(tf, scale, a, C.green, 0.75, false);
  drawCar(tf, scale, m, phase === "crashed" ? C.red : C.amber, 1, true);

  // HUD
  const pad = 20, small = Math.max(14, Math.min(W, H) / 40), big = small * 2.3;
  text(`${Math.round(me.speed * 3.6)} km/h`, pad, pad, big, C.text);
  text(`you ${secs(me)} s`, pad, pad + big * 1.15, small * 1.3, C.amber);
  text(`AI ${aiLapDone ? aiLapDone.toFixed(1) + " s, finished" : secs(ai) + " s"}`, pad, pad + big * 1.15 + small * 1.7,
       small * 1.3, C.green);
  text(best ? `your best ${best.toFixed(1)} s` : "your best: -", pad, pad + big * 1.15 + small * 3.4, small, C.muted, "left", 600);
  text(`to beat: AI ${AI_LAP.toFixed(1)} s · its creator ${CREATOR_LAP.toFixed(1)} s`, W - pad, pad, small, C.text, "right");
  text("arrows / WASD · L: the AI's lasers · R: restart", W - pad, pad + small * 1.6, small * 0.8, C.muted, "right", 500);

  // minimap
  const mm = Math.min(W, H) * 0.26, mx = W - mm - pad, my = H - mm - pad - (matchMedia("(pointer: coarse)").matches ? 110 : 0);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < P; i++) { x0 = Math.min(x0, game.center[2 * i]); x1 = Math.max(x1, game.center[2 * i]);
    y0 = Math.min(y0, game.center[2 * i + 1]); y1 = Math.max(y1, game.center[2 * i + 1]); }
  const ms = (mm * 0.9) / Math.max(x1 - x0, y1 - y0);
  const mtf = (x, y) => [mx + mm / 2 + (x - (x0 + x1) / 2) * ms, my + mm / 2 - (y - (y0 + y1) / 2) * ms];
  drawTrack(mtf, ms, false);
  for (const [s, col] of [[a, C.green], [m, C.amber]]) {
    g.fillStyle = col; g.beginPath(); g.arc(...mtf(s.x, s.y), 5, 0, 2 * Math.PI); g.fill();
  }

  // overlays
  const mid = (title, color, lines) => {
    g.fillStyle = "#0d0f16cc"; g.fillRect(0, H * 0.3, W, H * 0.4);
    text(title, W / 2, H * 0.34, big * 1.4, color, "center");
    lines.forEach((l, i) => text(l, W / 2, H * 0.34 + big * 1.8 + i * small * 1.6, small * 1.15, C.text, "center", 600));
  };
  if (phase === "countdown") text(String(Math.ceil(countdown)), W / 2, H * 0.28, big * 2.5, C.amber, "center");
  if (phase === "crashed") mid("CRASHED", C.red, [`try #${tries}`, "press R or tap to go again"]);
  if (phase === "finished") {
    const gap = meLap - AI_LAP;
    mid(gap < 0 ? "YOU BEAT THE AI!" : `${meLap.toFixed(1)} s`, gap < 0 ? C.green : C.amber, [
      gap < 0 ? `${meLap.toFixed(1)} s, ${(-gap).toFixed(1)} s faster than the AI` : `${gap.toFixed(1)} s behind the AI (${AI_LAP.toFixed(1)} s)`,
      meLap < CREATOR_LAP ? "and faster than its creator" : `its creator did ${CREATOR_LAP.toFixed(1)} s`,
      "press R or tap to go again"]);
  }
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (phase === "countdown") { countdown -= dt; if (countdown <= 0) phase = "racing"; }
  else if (phase === "racing") { acc += dt; while (acc >= cfg.dt && phase === "racing") { acc -= cfg.dt; tick(); } }
  draw(Math.min(1, acc / cfg.dt));
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
