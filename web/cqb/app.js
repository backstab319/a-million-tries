// The Kill House vs the AI: you clear a dark house against episode 9's trained defender, or hold a room against
// its trained attacker. Same game, same rules, ten decisions a second for both of you; you see only what your light
// shows. The AI's moves come from its real trained brain, running here (engine.js, checked against the Python game:
// node check.mjs).
import { ASSAULT, Brain, DEFENSE, Game, House } from "./engine.js";

const C = { bg: "#0d0f16", yard: "#11141c", floor: "#1d2230", wall: "#9aa3b8", door: "#fbbf24", cover: "#6b5d4a",
            dark: "rgba(5,6,10,0.88)", text: "#e8eaf0", muted: "#787f94", blue: "#60a5fa", red: "#f43f5e",
            amber: "#fbbf24", light: "rgba(255,240,200,0.22)", panel: "#161923" };
const canvas = document.getElementById("c"), g = canvas.getContext("2d");
const [data, buf] = await Promise.all([fetch("cqb.json").then((r) => r.json()), fetch("weights.bin").then((r) => r.arrayBuffer())]);
const weights = new Float32Array(buf);
const houses = data.houses.map((h) => new House(h));
const brainOf = (spec) => { const b = data.brains.find((x) => x.spec === spec); return { ...b, brain: new Brain(b.layers, weights) }; };
const MODES = {
  attack: { me: ASSAULT, ai: brainOf("runs/cqb-d5-guns:900"), rules: "runs/cqb-d5-guns:900", color: C.blue,
            goal: "Find the defender somewhere in the house and take it down within 45 seconds. You carry a rifle " +
                  "(30 rounds, 3 spare magazines) and two flashbangs: the thing my AI never learned to use.",
            ai_line: "The AI defender practised for {n} decisions. Under these rules my AI attacker beats it 13% of " +
                     "the time; my hand-written flash-and-enter script, about 22%." },
  defend: { me: DEFENSE, ai: brainOf("runs/cqb-a12-react:300"), rules: "runs/cqb-a12-react:300", color: C.red,
            goal: "Hold your room until the 45 seconds run out. The AI attacker starts outside, in the dark. Health " +
                  "is on (hits wound, they don't always kill), and nobody can aim the instant they see someone.",
            ai_line: "The AI attacker is the one from the end of episode 9 ({n} decisions in its last training run). " +
                     "It beats my best AI defender 25% of the time." },
};
const NAMES = ["attacker", "defender"];

// ── sound (the episode's own effects) ──
let audio = null;
const SFX = ["gunshot", "door", "reload", "flashbang", "hit"], sfx = {};
async function initAudio() {
  if (audio) return;
  audio = new AudioContext();
  await Promise.all(SFX.map(async (n) => { sfx[n] = await audio.decodeAudioData(await (await fetch(`sfx/${n}.mp3`)).arrayBuffer()); }));
}
function play(name, vol) {
  if (!audio || !sfx[name] || vol <= 0.02) return;
  const s = audio.createBufferSource(), v = audio.createGain();
  s.buffer = sfx[name]; v.gain.value = Math.min(1, vol) * 0.6; s.connect(v).connect(audio.destination); s.start();
}

// ── menu ──
const menu = document.getElementById("menu");
let mode = localStorage.getItem("cqb-mode") || "attack";
let steer = localStorage.getItem("cqb-steer") || "screen";   // "screen": W is up; "face": W is where you aim
const score = () => JSON.parse(localStorage.getItem(`cqb-score-${mode}`) || "[0,0]");
function renderMenu() {
  const M = MODES[mode], [won, played] = score();
  for (const b of menu.querySelectorAll("[data-mode]")) b.classList.toggle("on", b.dataset.mode === mode);
  for (const b of menu.querySelectorAll("[data-steer]")) b.classList.toggle("on", b.dataset.steer === steer);
  document.getElementById("goal").textContent = M.goal;
  document.getElementById("info").innerHTML = M.ai_line.replace("{n}", `<b>${(M.ai.decisions / 1e6).toFixed(0)} million</b>`) +
    (played ? ` You: <b>${won} of ${played}</b>.` : "");
}
menu.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.mode) { mode = b.dataset.mode; localStorage.setItem("cqb-mode", mode); }
  if (b.dataset.steer) { steer = b.dataset.steer; localStorage.setItem("cqb-steer", steer); }
  if (b.id === "play") return start();
  renderMenu();
});

// ── input ──
const keys = {}, pressed = {};
let mouse = null, mouseDown = false;
addEventListener("keydown", (e) => {
  if (!keys[e.code]) pressed[e.code] = true;
  keys[e.code] = true;
  if (["Space", "ShiftLeft", "ShiftRight"].includes(e.code)) e.preventDefault();
  if (phase === "menu" && e.code === "Enter") start();
  else if (phase === "over" && (e.code === "Space" || e.code === "Enter")) next();
  else if (phase !== "menu" && e.code === "Escape") showMenu();
});
addEventListener("keyup", (e) => { keys[e.code] = false; });
canvas.addEventListener("mousemove", (e) => { mouse = [e.clientX, e.clientY]; });
canvas.addEventListener("mousedown", (e) => { if (phase === "over") return next(); if (e.button === 0) mouseDown = true; });
addEventListener("mouseup", () => { mouseDown = false; });
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
const once = (code) => { const p = pressed[code]; pressed[code] = false; return p; };

// ── a round ──
let phase = "menu", game, house, st, prev, acc, countdown, M, me, ai, result, view = null;
let tries = Number(localStorage.getItem("cqb-tries")) || 0;
globalThis.cqbState = () => ({ st, house, phase, me });   // for automated checks (read only)
function showMenu() { phase = "menu"; menu.hidden = false; renderMenu(); }
function start() { menu.hidden = true; initAudio(); next(); }
function next() {
  M = MODES[mode]; me = M.me; ai = 1 - me;
  game = new Game(data.rules[M.rules]);
  house = houses[Math.floor(Math.random() * houses.length)];
  st = prev = game.start(house); acc = 0; countdown = 3; result = null; phase = "countdown";
  tries += 1; localStorage.setItem("cqb-tries", tries);
  for (const k in pressed) pressed[k] = false;
}

function myAction() {
  const k = (c) => (keys[c] ? 1 : 0), cfg = game.cfg, face = st.face[me];
  let turn = k("KeyQ") - k("KeyE");
  if (!turn && mouse && view) {   // turn towards the mouse, as fast as the game allows
    const [wx, wy] = view.inv(...mouse), want = Math.atan2(wy - st.pos[me][1], wx - st.pos[me][0]);
    const diff = Math.atan2(Math.sin(want - face), Math.cos(want - face));
    turn = Math.max(-1, Math.min(1, diff / ((cfg.turn_rate * Math.PI) / 180 * cfg.dt)));
    if (Math.abs(diff) < 0.02) turn = 0;
  }
  const on = (b) => (b ? 1 : -1);
  let fwd = k("KeyW") - k("KeyS"), side = k("KeyA") - k("KeyD");
  if (steer === "screen") {   // W is up the screen whichever way you face: turned into the game's own forward/sideways
    const x = k("KeyD") - k("KeyA"), y = k("KeyW") - k("KeyS"), c = Math.cos(-face), s_ = Math.sin(-face);
    [fwd, side] = [c * x - s_ * y, s_ * x + c * y];
  }
  return [fwd, side, turn, on(mouseDown || keys.Space), on(once("KeyF")),
          on(keys.ShiftLeft || keys.ShiftRight), on(once("KeyG")), on(once("KeyR"))];
}

function sounds(a, b) {   // what you hear this decision: your own noise, and the other's if it reached you
  const heard = (kind) => b.sound[me][kind][2];
  if (b.fired[me]) play("gunshot", 1);
  if (b.fired[ai]) play("gunshot", 0.25 + 0.75 * heard(2));
  if (b.doorEvent[me]) play("door", 0.7);
  if (b.doorEvent[ai] && heard(1) > 0) play("door", 0.3 + 0.7 * heard(1));
  if (b.reloadNow[me]) play("reload", 0.8);
  if (b.reloadNow[ai] && b.reloadHeard[me][2] > 0) play("reload", 0.3 + 0.7 * b.reloadHeard[me][2]);
  for (const w of [0, 1]) if (b.bangNow[w]) play("flashbang", w === me ? 1 : 0.3 + 0.7 * b.bangHeard[me][2]);
  for (const w of [0, 1]) if ((a.alive[w] && !b.alive[w]) || b.hp[w] < a.hp[w]) play("hit", w === me ? 1 : 0.7);
}

function tick() {  // one decision each (10 per second)
  const acts = [];
  acts[me] = myAction();
  acts[ai] = M.ai.brain.act(game.observe(house, st, ai));
  prev = st; st = game.step(house, st, acts);
  sounds(prev, st);
  if (st.done) {
    const won = me === ASSAULT ? st.outcome === 1 : st.outcome === -1, [w, p] = score();
    localStorage.setItem(`cqb-score-${mode}`, JSON.stringify([w + won, p + 1]));
    const how = !st.alive[me] && !st.alive[ai] ? "both down" : st.alive[me] && st.alive[ai] ? "time ran out"
      : st.alive[me] ? `you took the ${NAMES[ai]} down` : `the ${NAMES[ai]} took you down`;
    result = { won, how, at: performance.now() }; phase = "over";
  }
}

// ── drawing ──
let W, H, dpr;
function resize() { dpr = devicePixelRatio || 1; W = innerWidth; H = innerHeight;
  canvas.width = W * dpr; canvas.height = H * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0); }
addEventListener("resize", resize); resize();
const darkCanvas = document.createElement("canvas"), dg = darkCanvas.getContext("2d");
const text = (s, x, y, size, color, align = "left", weight = 700) => {
  g.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  g.fillStyle = color; g.textAlign = align; g.textBaseline = "top"; g.fillText(s, x, y);
};
const lerp = (a, b, t) => a + (b - a) * t;
const lerpAngle = (a, b, t) => a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;

function cone(ctx, tf, p, face, reach) {   // the lit fan: from p along each of the light's rays
  ctx.beginPath(); ctx.moveTo(...tf(...p));
  game.cone.forEach((a, k) => ctx.lineTo(...tf(p[0] + Math.cos(face + a) * reach[k], p[1] + Math.sin(face + a) * reach[k])));
  ctx.closePath();
}
function operator(tf, s, p, face, color) {
  const [x, y] = tf(...p), r = game.cfg.radius * 1.6 * s;
  g.strokeStyle = C.text; g.lineWidth = Math.max(2, r * 0.3);
  g.beginPath(); g.moveTo(x, y); g.lineTo(x + Math.cos(face) * r * 2.2, y - Math.sin(face) * r * 2.2); g.stroke();
  g.fillStyle = color; g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill();
}
function marks(tf, s, w, p) {   // as in the video: an amber ring filling while reloading, a health bar under it
  const [x, y] = tf(...p);
  if (game.gunsOn && st.reloadTimer[w] > 0) {
    const done = 1 - st.reloadTimer[w] / game.reloadSteps, r = 0.75 * s;
    g.lineWidth = Math.max(3, 0.12 * s); g.strokeStyle = C.muted;
    g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.stroke();
    g.strokeStyle = C.amber; g.beginPath(); g.arc(x, y, r, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * done); g.stroke();
  }
  if (game.healthOn) {
    const frac = Math.max(st.hp[w], 0) / game.cfg.health, bw = 1.1 * s, bh = Math.max(4, 0.16 * s), by = y + 0.85 * s;
    g.fillStyle = "#262b3a"; g.fillRect(x - bw / 2, by, bw, bh);
    g.fillStyle = frac >= 0.5 ? "#4ade80" : frac >= 0.25 ? C.amber : C.red; g.fillRect(x - bw / 2, by, bw * frac, bh);
  }
}
function glow(tf, s, p, radius, color) {
  const [x, y] = tf(...p), gr = g.createRadialGradient(x, y, 0, x, y, radius * s);
  gr.addColorStop(0, color); gr.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = gr; g.beginPath(); g.arc(x, y, radius * s, 0, 2 * Math.PI); g.fill();
}

function draw(alpha) {
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  if (phase === "menu") return;
  const cfg = game.cfg, hud = Math.min(110, H * 0.16), m = 5;   // the house and 5 m around it (the entries are 2.5 m out)
  const lo = [Math.max(0, house.footLo[0] - m), Math.max(0, house.footLo[1] - m)];
  const hi = [Math.min(cfg.yard_w, house.footHi[0] + m), Math.min(cfg.yard_h, house.footHi[1] + m)];
  const s = Math.min((W - 24) / (hi[0] - lo[0]), (H - hud - 24) / (hi[1] - lo[1]));
  const ox = (W - (hi[0] - lo[0]) * s) / 2 - lo[0] * s, oy = hud + 12 + ((H - hud - 24) - (hi[1] - lo[1]) * s) / 2 - (cfg.yard_h - hi[1]) * s;
  const tf = (x, y) => [ox + x * s, oy + (cfg.yard_h - y) * s];
  view = { inv: (px, py) => [(px - ox) / s, cfg.yard_h - (py - oy) / s] };
  const a = phase === "playing" ? alpha : 1, over = phase === "over";
  const pos = [0, 1].map((w) => [lerp(prev.pos[w][0], st.pos[w][0], a), lerp(prev.pos[w][1], st.pos[w][1], a)]);
  const face = [0, 1].map((w) => lerpAngle(prev.face[w], st.face[w], a));

  g.save(); g.beginPath(); g.rect(0, hud, W, H - hud); g.clip();   // the world stays under the HUD
  // the yard, the house, cover, doors
  g.fillStyle = C.yard; g.fillRect(ox, oy, cfg.yard_w * s, cfg.yard_h * s);
  const [fx, fy] = tf(house.footLo[0], house.footHi[1]);
  g.fillStyle = C.floor; g.fillRect(fx, fy, (house.footHi[0] - house.footLo[0]) * s, (house.footHi[1] - house.footLo[1]) * s);
  const box = (b, color) => { const [x, y] = tf(b[0], b[3]); g.fillStyle = color; g.fillRect(x, y, (b[2] - b[0]) * s, (b[3] - b[1]) * s); };
  for (const b of house.cover) box(b, C.cover);
  for (const b of house.walls) box(b, C.wall);
  house.doors.forEach((b, k) => { if (house.doorValid[k]) box(b, st.doorOpen[k] ? "rgba(251,191,36,0.25)" : C.door); });

  // lights: yours always; the other's while you can see it
  const showAI = over || st.seen[me];
  const reach = [0, 1].map((w) => game.lightOf(house, st, w));
  g.fillStyle = C.light;
  for (const w of [me, ...(showAI ? [ai] : [])]) { cone(g, (x, y) => tf(x, y), pos[w], face[w], reach[w]); g.fill(); }

  // the dark: everything but your light (and a little around you)
  if (!over) {
    darkCanvas.width = canvas.width; darkCanvas.height = canvas.height;
    dg.setTransform(dpr, 0, 0, dpr, 0, 0); dg.clearRect(0, 0, W, H); dg.fillStyle = C.dark; dg.fillRect(0, hud, W, H - hud);
    dg.globalCompositeOperation = "destination-out";
    cone(dg, tf, pos[me], face[me], reach[me]); dg.fill();
    const [mx, my] = tf(...pos[me]); dg.beginPath(); dg.arc(mx, my, 1.2 * s, 0, 2 * Math.PI); dg.fill();
    dg.globalCompositeOperation = "source-over";
    g.drawImage(darkCanvas, 0, 0, W, H);
    g.globalAlpha = 0.3;   // the layout stays faintly visible in the dark (the AI's lasers feel the walls around it too)
    for (const b of house.walls) box(b, C.wall);
    house.doors.forEach((b, k) => { if (house.doorValid[k] && !st.doorOpen[k]) box(b, C.door); });
    g.globalAlpha = 1;
  }

  // flashbangs: yours lying there, and the bang
  for (const w of [0, 1]) {
    if (st.bangTimer[w] > 0 && (w === me || over)) { const [x, y] = tf(...st.bangPos[w]); g.fillStyle = C.amber;
      g.beginPath(); g.arc(x, y, 0.18 * s, 0, 2 * Math.PI); g.fill(); }
    if (st.bangNow[w] || prev.bangNow?.[w]) glow(tf, s, st.bangPos[w], cfg.bang_range, "rgba(255,255,255,0.85)");
  }
  // muzzle flashes you can see, and where you last saw or heard the other
  for (const w of [0, 1]) if (st.fired[w] && (w === me || st.flashSeen[me] || showAI)) glow(tf, s, pos[w], 1.2, "rgba(255,210,120,0.9)");
  if (!showAI && st.since[me] < game.T) {
    const [x, y] = tf(...st.mem[me]), r = 0.5 * s;
    g.setLineDash([r * 0.5, r * 0.4]); g.strokeStyle = MODES[mode] === MODES.attack ? C.red : C.blue; g.lineWidth = 2;
    g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
    text("?", x, y - r * 0.6, r * 1.2, g.strokeStyle, "center");
  }
  operator(tf, s, pos[me], face[me], M.color); marks(tf, s, me, pos[me]);
  if (showAI || (st.flashSeen[me] && st.fired[ai])) {
    operator(tf, s, pos[ai], face[ai], me === ASSAULT ? C.red : C.blue);
    if (showAI) marks(tf, s, ai, pos[ai]);
  }

  // what you hear: a mark on a ring around you, towards the sound (steps grey, door amber, shot red)
  const [mx, my] = tf(...pos[me]);
  const heardMarks = [...st.sound[me].map((x, k) => [x, ["#c3c8d4", C.amber, C.red][k]]),
                      [st.reloadHeard[me], "#c3c8d4"], [st.bangHeard[me], "#ffffff"]];
  for (const [[dx, dy, loud], color] of heardMarks) {
    if (loud <= 0 || over) continue;
    const ang = Math.atan2(dy, dx);
    g.strokeStyle = color; g.globalAlpha = 0.35 + 0.65 * loud; g.lineWidth = Math.max(3, 0.25 * s);
    g.beginPath(); g.arc(mx, my, 2.2 * s, -ang - 0.25, -ang + 0.25); g.stroke(); g.globalAlpha = 1;
  }
  g.restore();
  if (st.blind[me] > 0 && !over) { g.fillStyle = `rgba(255,255,255,${Math.min(0.95, st.blind[me] / game.blindSteps)})`; g.fillRect(0, 0, W, H); }

  // HUD
  const pad = 16, small = Math.max(13, Math.min(W, H) / 52), left = Math.max(0, cfg.time_limit - st.t * cfg.dt), [won, played] = score();
  text(`You: the ${NAMES[me]} · vs the AI ${NAMES[ai]} (episode 9)`, pad, pad, small * 1.5, M.color);
  text(`you ${won} of ${played} vs this AI · try #${tries}`, pad, pad + small * 2.1, small, C.muted, "left", 600);
  const stats = [`${left.toFixed(1)} s left`];
  if (game.gunsOn) stats.push(st.reloadTimer[me] > 0 ? "reloading…" : `${st.mag[me]} | ${st.reserve[me]} rounds`);
  if (game.bangsOn && cfg.flashbangs[me]) stats.push(`${st.bangs[me]} flashbang${st.bangs[me] === 1 ? "" : "s"}`);
  if (game.healthOn) stats.push(`health ${Math.round(st.hp[me])}`);
  text(stats.join("   ·   "), pad, pad + small * 3.6, small * 1.25, left < 5 ? C.red : C.text, "left", 700);
  if (W >= 1000) text("WASD · mouse aim · click fire · F door · Shift run · G flashbang · R reload · Esc menu",
                      W - pad, pad, small * 0.85, C.muted, "right", 500);
  if (phase === "countdown") text(String(Math.ceil(countdown)), W / 2, H / 2 - small * 4, small * 8, C.text, "center");
  if (over) {
    const y0 = H - small * 9;   // at the bottom, so the revealed round stays in view
    g.fillStyle = "#0d0f16dd"; g.fillRect(0, y0 - small, W, small * 10);
    text(result.won ? "YOU WIN" : "AI WINS", W / 2, y0, small * 3.2, result.won ? M.color : C.red, "center");
    text(`${result.how} · ${(st.t * cfg.dt).toFixed(1)} s · everything revealed`, W / 2, y0 + small * 3.9,
         small * 1.1, C.text, "center", 600);
    text("Space or click: next house · Esc: menu", W / 2, y0 + small * 5.7, small, C.muted, "center", 600);
  }
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (phase === "countdown") { countdown -= dt; if (countdown <= 0) phase = "playing"; }
  else if (phase === "playing") { acc += dt; while (acc >= game.cfg.dt && phase === "playing") { acc -= game.cfg.dt; tick(); } }
  draw(Math.min(1, acc / (game ? game.cfg.dt : 0.1)));
  requestAnimationFrame(frame);
}
showMenu();
requestAnimationFrame(frame);
