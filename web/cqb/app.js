// The Kill House vs the AI: you clear a dark house against episode 9's trained defender, or hold a room against
// its trained attacker; or, from episode 10, hold it against the AI pair, or attack with an AI partner; or, from
// episode 12, two against two with a hostage: walk it out with an AI partner, or hold it with one; or, from episode 13,
// the same in a real home (furniture, windows, drywall bullets go through, a hostage that can be hit). Same game,
// same rules, ten decisions a second for everyone; you see what your side sees (your light, your partner's). The AI's
// moves come from its real trained brains, running here (engine.js, checked against the Python game: node check.mjs).
import { ASSAULT, Brain, DEFENSE, Game, House } from "./engine.js";
import { Sound } from "./sound.js";
// The look: "3d" (view3d.js: the house at night, the operators animated; three.js loads only if picked) or "2d" (the
// top-down view from the videos). Same game either way: the look only draws engine.js's state.

const C = { bg: "#0d0f16", yard: "#11141c", floor: "#1d2230", wall: "#9aa3b8", door: "#fbbf24", cover: "#6b5d4a",
            drywall: "#6c7489", low: "#4b4954", window: "#7dd3fc",   // (ep13) homes
            dark: "rgba(5,6,10,0.88)", text: "#e8eaf0", muted: "#787f94", blue: "#60a5fa", red: "#f43f5e",
            amber: "#fbbf24", light: "rgba(255,240,200,0.22)", panel: "#161923", cyan: "#67e8f9" };
const canvas = document.getElementById("c"), g = canvas.getContext("2d"), glCanvas = document.getElementById("gl");
// a build without the 3D look (the public one: the 3D client stays private) has no Look buttons: top-down only
const has3d = !!document.querySelector("[data-look]");
let look = has3d ? localStorage.getItem("cqb-look") || "3d" : "2d", v3d = null, v3dLoading = null;
async function load3d() {
  if (v3d) return v3d;
  v3dLoading ??= (async () => {
    const { View } = await import("./view3d.js");
    const v = new View(glCanvas);
    await v.load(["models/assault.glb", "models/defense.glb"]);   // by team: Assault, Defense
    v.resize();
    return (v3d = v);
  })();
  return v3dLoading;
}
const in3d = () => look === "3d" && v3d;
const [data, buf] = await Promise.all([fetch("cqb.json").then((r) => r.json()), fetch("weights.bin").then((r) => r.arrayBuffer())]);
const weights = new Float32Array(buf);
const houses = data.houses.filter((h) => !h.layout).map((h) => new House(h));
const homes = data.houses.filter((h) => h.layout === "home").map((h) => new House(h));   // (ep13) the home modes'
const brainOf = (spec) => { const b = data.brains.find((x) => x.spec === spec); return { ...b, brain: new Brain(b.layers, weights) }; };
const MODES = {
  attack: { me: 0, ai: brainOf("runs/cqb-d5-guns:900"), rules: "runs/cqb-d5-guns:900", color: C.blue,
            goal: "Find the defender somewhere in the house and take it down within 45 seconds. You carry a rifle " +
                  "(30 rounds, 3 spare magazines) and two flashbangs: the thing my AI never learned to use.",
            ai_line: "The AI defender practised for {n} decisions. Under these rules my AI attacker beats it 13% of " +
                     "the time; my hand-written flash-and-enter script, about 22%." },
  defend: { me: 1, ai: brainOf("runs/cqb-a12-react:300"), rules: "runs/cqb-a12-react:300", color: C.red,
            goal: "Hold your room until the 45 seconds run out. The AI attacker starts outside, in the dark. Health " +
                  "is on (hits wound, they don't always kill), and nobody can aim the instant they see someone.",
            ai_line: "The AI attacker is the one from the end of episode 9 ({n} decisions in its last training run). " +
                     "It beats my best AI defender 25% of the time." },
  // episode 10: two attackers, one brain; under the pair's rules (crashes are loud and stun you, the rifle comes up
  // slowly after running)
  defend2: { me: 2, ai: brainOf("runs/cqb-p4-stun:650"), rules: "runs/cqb-p4-stun:650", color: C.red, ep: 10,
             goal: "Hold your room against two AI attackers for 45 seconds. They share what either one sees, and " +
                   "they trade: if one goes down, the other is usually right behind. Crashing into a wall is loud and " +
                   "stuns you for half a second.",
             ai_line: "The AI pair from the end of episode 10 ({n} decisions in its last run). It beats my best AI " +
                      "defenders about 55% of the time." },
  attack2: { me: 0, ai: brainOf("runs/cqb-d8-vs-pair:350"), partner: brainOf("runs/cqb-p4-stun:650"),
             rules: "runs/cqb-p4-stun:650", color: C.blue, ep: 10,
             goal: "You're number one of a pair; the AI pair's brain plays number two. You see what either of you " +
                   "sees. Friendly fire is on, and crashing into a wall is loud and stuns you for half a second.",
             ai_line: "The AI defender trained against pairs ({n} decisions). Two AIs together beat it about 55% of " +
                      "the time." },
  // episode 12: two vs two and a hostage to walk out, under the defenders' rules (they hear it being moved)
  rescue: { me: 0, ai: brainOf("runs/cqb-d23-2v2-extract-noise:300"), partner: brainOf("runs/cqb-e14-2v2-extract:300"),
            rules: "runs/cqb-d23-2v2-extract-noise:300", color: C.blue, ep: 12,
            goal: "Get the hostage out, with an AI partner. Stand beside it for 3 seconds with a clear line to secure " +
                  "it, then walk it out of the house: you're slower while you lead it, and if you go down it stays " +
                  "where it is. Two AI defenders hold the house, and they can hear it being moved.",
            ai_line: "The AI defenders from the end of episode 12. Against my best AI attackers they hold the house " +
                     "about 62% of the time." },
  hold: { me: 2, ai: brainOf("runs/cqb-e14-2v2-extract:300"), partner: brainOf("runs/cqb-d23-2v2-extract-noise:300"),
          rules: "runs/cqb-d23-2v2-extract-noise:300", color: C.red, ep: 12,
          goal: "Two AI attackers come for the hostage. Hold the house with an AI partner for 45 seconds. You know " +
                "where the hostage is until they move it; after that only while you see it, or hear it.",
          ai_line: "The AI attackers from the end of episode 12, evolved over 21 million practice rounds. Against my " +
                   "best AI defenders they win about 38% of the time, and walk the hostage out in about one in five." },
  // episode 13: the same two vs two in a real home, under its rules: low furniture and windows stop bodies, not sight
  // or shots; bullets go through drywall (inside walls) as a guess (x unseen_hit x through_drywall per wall: a
// quarter through one); hit the hostage and your side loses.
  home: { me: 0, ai: brainOf("runs/cqb-d24-2v2-home:300"), partner: brainOf("runs/cqb-e17-2v2-home-doors:150"),
          rules: "runs/cqb-d24-2v2-home:300", color: C.blue, ep: 13, homes: true,
          goal: "The same rescue in a real home. Beds, sofas, counters and windows stop you but not sight or bullets; " +
                "bullets go through the thin inside walls (the lighter ones), but blind: a quarter of the usual chance " +
                "through one wall. Brick outside walls stop them. Hit the hostage and your side loses.",
          ai_line: "The AI defenders from episode 13, trained in homes. Against my best AI attackers they hold the " +
                   "house about 38% of the time." },
  homehold: { me: 2, ai: brainOf("runs/cqb-e17-2v2-home-doors:150"), partner: brainOf("runs/cqb-d24-2v2-home:300"),
              rules: "runs/cqb-d24-2v2-home:300", color: C.red, ep: 13, homes: true,
              goal: "Hold a real home with an AI partner for 45 seconds. Furniture and windows stop bodies, not sight " +
                    "or bullets; the lighter inside walls let bullets through, blind, at a quarter of the chance. Hit the hostage " +
                    "and your side loses.",
              ai_line: "The AI attackers from episode 13, trained in homes. Against my best AI defenders they win " +
                       "about 62% of the time, and walk the hostage out in more than two rounds in five." },
};
const NAMES = ["attacker", "defender"];

const sound = new Sound();   // the effects and the radio (sound.js)

// ── menu ──
const menu = document.getElementById("menu");
let mode = localStorage.getItem("cqb-mode") || "attack";
let steer = localStorage.getItem("cqb-steer") || "screen";   // "screen": W is up; "face": W is where you aim
const score = () => JSON.parse(localStorage.getItem(`cqb-score-${mode}`) || "[0,0]");
function renderMenu() {
  const M = MODES[mode], [won, played] = score();
  for (const b of menu.querySelectorAll("[data-mode]")) b.classList.toggle("on", b.dataset.mode === mode);
  for (const b of menu.querySelectorAll("[data-steer]")) b.classList.toggle("on", b.dataset.steer === steer);
  for (const b of menu.querySelectorAll("[data-look]")) b.classList.toggle("on", b.dataset.look === look);
  document.getElementById("goal").textContent = M.goal;
  document.getElementById("info").innerHTML = M.ai_line.replace("{n}", `<b>${(M.ai.decisions / 1e6).toFixed(0)} million</b>`) +
    (played ? ` You: <b>${won} of ${played}</b>.` : "");
}
menu.addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.mode) { mode = b.dataset.mode; localStorage.setItem("cqb-mode", mode); }
  if (b.dataset.steer) { steer = b.dataset.steer; localStorage.setItem("cqb-steer", steer); }
  if (b.dataset.look) { look = b.dataset.look; localStorage.setItem("cqb-look", look); }
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
let phase = "menu", game, house, st, prev, acc, countdown, M, me, side, team, foes, brains, result, view = null;
let tries = Number(localStorage.getItem("cqb-tries")) || 0;
globalThis.cqbState = () => ({ st, house, phase, me });   // for automated checks (read only)
Object.defineProperty(globalThis, "cqbView", { get: () => v3d });   // likewise: the 3D view
function showMenu() { phase = "menu"; menu.hidden = false; renderMenu(); }
async function start() {
  if (look === "3d" && !v3d) {
    const p = document.getElementById("play"), was = p.textContent;
    p.textContent = "Loading the 3D house…"; p.disabled = true;
    try { await load3d(); } catch (e) { console.error(e); look = "2d"; }   // no WebGL: the top-down view
    p.textContent = was; p.disabled = false;
  }
  menu.hidden = true; sound.init(); next();
}
function next() {
  M = MODES[mode]; me = M.me;
  game = new Game(data.rules[M.rules]);
  side = game.team[me]; team = game.sideOps(side); foes = game.sideOps(1 - side);
  brains = game.ops.map((w) => (w === me ? null : game.team[w] === side ? M.partner.brain : M.ai.brain));
  const pool = M.homes ? homes : houses;
  house = pool[Math.floor(Math.random() * pool.length)];
  st = prev = game.start(house); acc = 0; countdown = 3; result = null; phase = "countdown";
  glCanvas.hidden = !in3d();
  if (in3d()) {
    v3d.buildHouse(house, game.cfg); v3d.clearOperators(); v3d.followAt = null; v3d.lit = 0;
    for (const w of game.ops) v3d.addOperator(game.team[w], game.cfg);
    v3d.bangs(game, st, () => true, 0);
    v3d.warmUp();
  }
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
    let x = k("KeyD") - k("KeyA"), y = k("KeyW") - k("KeyS");
    if (in3d()) {   // the 3D camera looks from the south-east: up the screen is north-west in the game
      const { right, up } = v3d.screenAxes();
      [x, y] = [x * right[0] + y * up[0], x * right[1] + y * up[1]];
    }
    const c = Math.cos(-face), s_ = Math.sin(-face);
    [fwd, side] = [c * x - s_ * y, s_ * x + c * y];
  }
  return [fwd, side, turn, on(mouseDown || keys.Space), on(once("KeyF")),
          on(keys.ShiftLeft || keys.ShiftRight), on(once("KeyG")), on(once("KeyR"))];
}

function tick() {  // one decision each (10 per second)
  const acts = game.ops.map((w) => (w === me ? myAction() : brains[w].act(game.observe(house, st, w))));
  prev = st; st = game.step(house, st, acts);
  sound.hear(game, prev, st, me, team, foes);
  sound.at(house, st.pos[me]);
  if (st.done) {
    const won = side === ASSAULT ? st.outcome === 1 : st.outcome === -1, [w, p] = score();
    localStorage.setItem(`cqb-score-${mode}`, JSON.stringify([w + won, p + 1]));
    const up = (ops) => ops.some((k) => st.alive[k]), foe = foes.length > 1 ? `the AI ${NAMES[1 - side]}s` : `the AI ${NAMES[1 - side]}`;
    const out = game.extractOn && st.outcome === 1 && up(team) && up(foes);   // (ep12) walked out
    const hit = st.hostageHitBy >= 0 ? (st.hostageHitBy === side ? "your side hit the hostage" : "they hit the hostage") : null;
    const how = hit ?? (out ? (side === ASSAULT ? "the hostage is out" : "they walked the hostage out")
      : !up(team) && !up(foes) ? "all down" : up(team) && up(foes) ? "time ran out"
      : up(team) ? (st.alive[me] ? `you took ${foe} down` : `you went down; your partner finished it`)
      : `${foe} took ${team.length > 1 ? "you both" : "you"} down`);
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

function draw(alpha, dt = 0) {
  const three = in3d() && phase !== "menu";
  if (three) g.clearRect(0, 0, W, H); else { g.fillStyle = C.bg; g.fillRect(0, 0, W, H); }
  if (phase === "menu") return;
  const cfg = game.cfg, hud = Math.min(110, H * 0.16), m = 5;   // the house and 5 m around it (the entries are 2.5 m out)
  const a = phase === "playing" ? alpha : 1, over = phase === "over";
  const pos = game.ops.map((w) => [lerp(prev.pos[w][0], st.pos[w][0], a), lerp(prev.pos[w][1], st.pos[w][1], a)]);
  const face = game.ops.map((w) => lerpAngle(prev.face[w], st.face[w], a));
  // your side sees what any of it sees (and always knows where its own are)
  const shown = (w) => over || team.includes(w) || team.some((k) => st.seen[k][w]);
  const lights = team.filter((k) => st.alive[k]);
  let tf, s, ox, oy;
  if (three) {   // the 3D house; the marks below are drawn over it, where each point lands on screen
    v3d.update(dt, game, st, pos, face, game.ops.map(shown));
    v3d.bangs(game, st, (w) => team.includes(w) || over, dt);
    v3d.reveal(over ? 1 : 0, dt);
    const who = st.alive[me] ? me : (team.find((k) => st.alive[k]) ?? me);
    v3d.follow(pos[who], 13, dt, v3d.followAt == null);
    v3d.render();
    tf = (x, y) => v3d.toScreen([x, y]); s = v3d.pxPerMetre;
    view = { inv: (px, py) => v3d.toGround(px, py, 1.0) };
  } else {
    const lo = [Math.max(0, house.footLo[0] - m), Math.max(0, house.footLo[1] - m)];
    const hi = [Math.min(cfg.yard_w, house.footHi[0] + m), Math.min(cfg.yard_h, house.footHi[1] + m)];
    s = Math.min((W - 24) / (hi[0] - lo[0]), (H - hud - 24) / (hi[1] - lo[1]));
    ox = (W - (hi[0] - lo[0]) * s) / 2 - lo[0] * s; oy = hud + 12 + ((H - hud - 24) - (hi[1] - lo[1]) * s) / 2 - (cfg.yard_h - hi[1]) * s;
    tf = (x, y) => [ox + x * s, oy + (cfg.yard_h - y) * s];
    view = { inv: (px, py) => [(px - ox) / s, cfg.yard_h - (py - oy) / s] };
  }

  g.save(); g.beginPath(); g.rect(0, hud, W, H - hud); g.clip();   // the world stays under the HUD
  if (!three) {
  // the yard, the house, cover, doors
  g.fillStyle = C.yard; g.fillRect(ox, oy, cfg.yard_w * s, cfg.yard_h * s);
  const [fx, fy] = tf(house.footLo[0], house.footHi[1]);
  g.fillStyle = C.floor; g.fillRect(fx, fy, (house.footHi[0] - house.footLo[0]) * s, (house.footHi[1] - house.footLo[1]) * s);
  const box = (b, color) => { const [x, y] = tf(b[0], b[3]); g.fillStyle = color; g.fillRect(x, y, (b[2] - b[0]) * s, (b[3] - b[1]) * s); };
  for (const b of house.low) box(b, C.low);   // (ep13) low furniture: stops bodies, not sight or shots
  for (const b of house.cover) box(b, C.cover);
  house.walls.forEach((b, k) => box(b, house.wallSoft[k] ? C.drywall : C.wall));   // drywall: bullets go through
  for (const b of house.win) box(b, C.window);
  house.doors.forEach((b, k) => { if (house.doorValid[k]) box(b, st.doorOpen[k] ? "rgba(251,191,36,0.25)" : C.door); });

  // lights: your side's always; an enemy's while your side can see it
  const reach = game.ops.map((w) => game.lightOf(house, st, w));
  g.fillStyle = C.light;
  for (const w of game.ops) if (st.alive[w] && shown(w)) { cone(g, (x, y) => tf(x, y), pos[w], face[w], reach[w]); g.fill(); }

  // the dark: everything but your light (and a little around you)
  if (!over) {
    darkCanvas.width = canvas.width; darkCanvas.height = canvas.height;
    dg.setTransform(dpr, 0, 0, dpr, 0, 0); dg.clearRect(0, 0, W, H); dg.fillStyle = C.dark; dg.fillRect(0, hud, W, H - hud);
    dg.globalCompositeOperation = "destination-out";
    for (const k of lights) {
      cone(dg, tf, pos[k], face[k], reach[k]); dg.fill();
      const [kx, ky] = tf(...pos[k]); dg.beginPath(); dg.arc(kx, ky, 1.2 * s, 0, 2 * Math.PI); dg.fill();
    }
    dg.globalCompositeOperation = "source-over";
    g.drawImage(darkCanvas, 0, 0, W, H);
    g.globalAlpha = 0.3;   // the layout stays faintly visible in the dark (the AI's lasers feel the walls around it too)
    house.walls.forEach((b, k) => box(b, house.wallSoft[k] ? C.drywall : C.wall));
    for (const b of house.low) box(b, C.low);
    for (const b of house.win) box(b, C.window);
    house.doors.forEach((b, k) => { if (house.doorValid[k] && !st.doorOpen[k]) box(b, C.door); });
    g.globalAlpha = 1;
  }

  // flashbangs: yours lying there, and the bang
  for (const w of game.ops) {
    if (st.bangTimer[w] > 0 && (team.includes(w) || over)) { const [x, y] = tf(...st.bangPos[w]); g.fillStyle = C.amber;
      g.beginPath(); g.arc(x, y, 0.18 * s, 0, 2 * Math.PI); g.fill(); }
    if (st.bangNow[w] || prev.bangNow?.[w]) glow(tf, s, st.bangPos[w], cfg.bang_range, "rgba(255,255,255,0.85)");
  }
  }
  // muzzle flashes your side can see (in 3D: their light), and where your side last saw or heard each enemy
  const flashed = (w) => team.some((k) => st.flashSeen[k][w]);
  if (!three) for (const w of game.ops) if (st.fired[w] && (shown(w) || flashed(w))) glow(tf, s, pos[w], 1.2, "rgba(255,210,120,0.9)");
  const foeColor = side === ASSAULT ? C.red : C.blue;
  for (const j of foes) {
    if (shown(j) || st.since[me][j] >= game.T || (st.knownDown && st.knownDown[me][j])) continue;
    const [x, y] = tf(...st.mem[me][j]), r = 0.5 * s;
    g.setLineDash([r * 0.5, r * 0.4]); g.strokeStyle = foeColor; g.lineWidth = 2;
    g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
    text("?", x, y - r * 0.6, r * 1.2, g.strokeStyle, "center");
  }
  for (const w of game.ops) {
    const ours = team.includes(w), color = ours ? M.color : foeColor;
    if (!shown(w) && !(flashed(w) && st.fired[w])) continue;
    if (three) {   // the operator itself is in the 3D scene: here only its marks and the "you" / "AI" labels
      if (st.alive[w] && shown(w)) marks(tf, s, w, pos[w]);
      if (st.alive[w] && ours && team.length > 1) {
        const [x, y] = tf(...pos[w]);
        text(w === me ? "you" : "AI", x + 0.55 * s, y - 2.3 * s, Math.max(12, 0.45 * s), color, "left", 700);
      }
      continue;
    }
    if (!st.alive[w]) {   // down: a cross
      const [x, y] = tf(...pos[w]), r = 0.35 * s; g.strokeStyle = color; g.lineWidth = Math.max(2, 0.12 * s);
      g.beginPath(); g.moveTo(x - r, y - r); g.lineTo(x + r, y + r); g.moveTo(x - r, y + r); g.lineTo(x + r, y - r); g.stroke();
      continue;
    }
    operator(tf, s, pos[w], face[w], color); if (shown(w)) marks(tf, s, w, pos[w]);
    if (ours && team.length > 1) {   // you, and your partner
      const [x, y] = tf(...pos[w]);
      text(w === me ? "you" : "AI", x + 0.55 * s, y - 0.9 * s, Math.max(12, 0.45 * s), color, "left", 700);
    }
  }
  if (game.hostageOn) {   // (ep12) where your side knows it is: always for Assault; for Defense until it's moved,
    // then only while one of you sees its escort (as the AI is told)
    const e = st.escort, known = over || side === ASSAULT || e < 0 || team.some((k) => st.since[k][e] === 0);
    if (known) {
      const hp = e >= 0 ? [st.hostage[0] - Math.cos(st.face[e]) * 0.45, st.hostage[1] - Math.sin(st.face[e]) * 0.45]
        : st.hostage, [x, y] = tf(...hp), r = game.cfg.radius * 1.4 * s;
      if (e < 0) {   // the securing ring, and how far it's secured
        const [cx, cy] = tf(...st.hostage), R = game.cfg.secure_radius * s;
        g.setLineDash([6, 6]); g.strokeStyle = "rgba(103,232,249,0.35)"; g.lineWidth = 2;
        g.beginPath(); g.arc(cx, cy, R, 0, 2 * Math.PI); g.stroke(); g.setLineDash([]);
        if (st.secure > 0) { g.strokeStyle = C.cyan; g.lineWidth = Math.max(4, 0.15 * s);
          g.beginPath(); g.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * st.secure / game.secureSteps); g.stroke(); }
      }
      g.fillStyle = "#f4f4f5"; g.strokeStyle = C.cyan; g.lineWidth = Math.max(2, 0.1 * s);
      g.beginPath(); g.arc(x, y, r, 0, 2 * Math.PI); g.fill(); g.stroke();
      if (!three) text("hostage", x + r * 1.3, y - r, Math.max(11, 0.38 * s), C.cyan, "left", 700);
    }
  }
  if (st.stunT && st.stunT[me] > 0 && !over) text("STUNNED", ...tf(...pos[me]).map((v, i) => v + (i ? 0.7 * s : 0)), Math.max(14, 0.5 * s), C.amber, "center");

  // what you hear: a mark on a ring around you, towards the sound (steps grey, door amber, shot red)
  const [mx, my] = tf(...pos[me]);
  const heardMarks = [...st.sound[me].map((x, k) => [x, ["#c3c8d4", C.amber, C.red][k]]),
                      [st.reloadHeard[me], "#c3c8d4"], [st.bangHeard[me], "#ffffff"],
                      ...(game.noiseOn ? [[st.hostageHeard[me], C.cyan]] : [])];   // (ep12) the hostage being moved
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
  const vs = foes.length > 1 ? `the AI ${NAMES[1 - side]}s` : `the AI ${NAMES[1 - side]}`;
  text(`You: the ${NAMES[side]}${team.length > 1 ? " + an AI partner" : ""} · vs ${vs} (episode ${M.ep || 9})` +
       (st.alive[me] || over ? "" : " · you're down: watching your partner"), pad, pad, small * 1.5, M.color);
  text(`you ${won} of ${played} vs this AI · try #${tries}`, pad, pad + small * 2.1, small, C.muted, "left", 600);
  const stats = [`${left.toFixed(1)} s left`];
  if (game.gunsOn) stats.push(st.reloadTimer[me] > 0 ? "reloading…" : `${st.mag[me]} | ${st.reserve[me]} rounds`);
  if (game.bangsOn && cfg.flashbangs[side]) stats.push(`${st.bangs[me]} flashbang${st.bangs[me] === 1 ? "" : "s"}`);
  if (game.healthOn) stats.push(`health ${Math.round(st.hp[me])}`);
  if (game.hostageOn) {   // (ep12) where it stands with the hostage (both sides' AIs are told this too)
    const e = st.escort;
    stats.push(e === me ? "leading the hostage out (slower)" : e >= 0 ? (side === ASSAULT ? "your partner has the hostage"
      : "they're moving the hostage") : st.secure > 0 ? `securing the hostage ${(st.secure * cfg.dt).toFixed(1)} / ${cfg.secure_time} s`
      : "hostage: inside");
  }
  text(stats.join("   ·   "), pad, pad + small * 3.6, small * 1.25, left < 5 ? C.red : C.text, "left", 700);
  if (W >= 1000) text("WASD · mouse aim · click fire · F door · Shift run · G flashbang · R reload · Esc menu",   // (under
                      W - pad, W >= 1700 ? pad : pad + small * 3.8, small * 0.85, C.muted, "right", 500);   // the title if narrow)
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
  draw(Math.min(1, acc / (game ? game.cfg.dt : 0.1)), dt);
  requestAnimationFrame(frame);
}
showMenu();
requestAnimationFrame(frame);
