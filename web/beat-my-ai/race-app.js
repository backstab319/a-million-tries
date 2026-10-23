// Beat the engineer (episode 14): set the 13 dials of a racing car under the $1M price list, then your car and the
// AI engineer's race for 60 s from a standing start on a test circuit. Both are driven by the same AI driver (the
// general driver, which learned on random cars of every design and never on yours), so the car alone decides.
// The physics and the driver run here in your browser (race-engine.js, checked against the Python game).
import { RaceGame, Driver } from "./race-engine.js";

const C = { bg: "#0d0f16", panel: "#161923", edge: "#262b3a", asphalt: "#222634", kerb: "#9a9eb2", grass: "#16251c",
            wall: "#4a5068", text: "#e8eaf0", muted: "#787f94", green: "#4ade80", amber: "#fbbf24", red: "#f43f5e" };
const data = await (await fetch("race-data.json")).json();
const game = new RaceGame(data), car = game.car, cfg = data.cfg, D = car.D;
const ENGINEER = data.cars["eng-v4-2"];
const drivers = Object.fromEntries(Object.entries(data.drivers).map(([n, b]) => [n, new Driver(b)]));

// ── page ──
const css = document.createElement("style");
css.textContent = `
  #garage { position: fixed; inset: 0; overflow-y: auto; touch-action: pan-y; padding: 18px 16px 120px; box-sizing: border-box; }
  #garage .wrap { max-width: 760px; margin: 0 auto; }
  #garage h1 { margin: 0; font-size: clamp(24px, 5vw, 36px); }
  #garage p { color: ${C.muted}; margin: 6px 0 14px; line-height: 1.4; }
  .budget { position: sticky; top: -18px; background: ${C.bg}; padding: 10px 0; z-index: 2; }
  .bar { height: 14px; border-radius: 7px; background: ${C.edge}; overflow: hidden; display: flex; }
  .bar div { height: 100%; }
  .legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 13px; color: ${C.muted}; margin-top: 6px; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; }
  .total { font-size: 20px; font-weight: 700; display: flex; justify-content: space-between; margin-bottom: 6px; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: .08em; color: ${C.muted}; margin: 22px 0 6px; }
  .dial { display: grid; grid-template-columns: 1fr auto; gap: 2px 10px; padding: 9px 0; border-bottom: 1px solid ${C.edge}; }
  .dial label { font-weight: 600; } .dial .v { font-variant-numeric: tabular-nums; text-align: right; }
  .dial small { grid-column: 1 / 3; color: ${C.muted}; }
  .dial input { grid-column: 1 / 3; width: 100%; accent-color: ${C.amber}; height: 28px; }
  .facts { color: ${C.muted}; font-size: 14px; margin-top: 12px; }
  .go { position: fixed; left: 0; right: 0; bottom: 0; padding: 12px 16px; background: ${C.bg}ee; border-top: 1px solid ${C.edge};
    display: flex; gap: 10px; justify-content: center; z-index: 3; }
  button.btn { padding: 14px 22px; border-radius: 14px; border: 2px solid ${C.edge}; background: ${C.panel}; color: ${C.text};
    font: inherit; font-weight: 700; font-size: 17px; cursor: pointer; }
  button.btn.main { background: ${C.amber}; color: #111; border-color: ${C.amber}; }
  button.btn:disabled { opacity: .4; cursor: default; }
  .circ { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
  .circ canvas { width: 120px; height: 90px; background: ${C.panel}; border-radius: 10px; }
  #race-ui { position: fixed; left: 0; right: 0; bottom: 14px; display: none; gap: 8px; justify-content: center; flex-wrap: wrap; padding: 0 8px; }
  #result { position: fixed; inset: 0; display: none; overflow-y: auto; touch-action: pan-y; background: ${C.bg}e6; padding: 24px 16px 40px; box-sizing: border-box; }
  #result .wrap { max-width: 640px; margin: 0 auto; text-align: center; }
  #result h1 { font-size: clamp(30px, 7vw, 52px); margin: 10px 0; }
  #result table { width: 100%; border-collapse: collapse; margin: 18px 0; text-align: left; font-variant-numeric: tabular-nums; }
  #result td, #result th { padding: 6px 8px; border-bottom: 1px solid ${C.edge}; }
  #result th { color: ${C.muted}; font-weight: 600; }
  #result .row { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }`;
document.head.append(css);
document.body.insertAdjacentHTML("beforeend", `<canvas id="c" style="display:none"></canvas><div id="garage"></div>
  <div id="race-ui"><button class="btn" id="cam">Camera: leader</button><button class="btn" id="speed">Speed 1x</button><button class="btn" id="skip">Skip to the end</button></div>
  <div id="result"></div>`);
const canvas = document.getElementById("c"), g = canvas.getContext("2d");
const garage = document.getElementById("garage"), raceUI = document.getElementById("race-ui"), result = document.getElementById("result");

// ── the dials ──
const ENGINE_POS = (v) => (v < 0.15 ? "front, ahead of the axle" : v < 0.45 ? "front, behind the axle" : v < 0.75 ? "middle, behind the driver" : "rear, over the back axle");
const DRIVE = (v) => (v < 0.02 ? "rear-wheel drive" : v > 0.98 ? "front-wheel drive" : `four-wheel drive, ${Math.round(v * 100)}% to the front`);
const DIALS = {
  parts: [
    ["power_kw", "Engine power", (v) => `${Math.round(v)} kW (${Math.round(v * 1.341)} hp)`, "More push, but big engines get dear fast and burn more fuel"],
    ["wing_front", "Front wing", (v) => (v < 0.01 ? "none" : `${Math.round(v * 100)}% size`), "Downforce on the front axle, and drag"],
    ["wing_rear", "Rear wing", (v) => (v < 0.01 ? "none" : `${Math.round(v * 100)}% size`), "Downforce on the rear axle, and drag"],
    ["tyre_front_mm", "Front tyres", (v) => `${Math.round(v)} mm wide`, "Wider: more grip and a steadier temperature, more weight and drag"],
    ["tyre_rear_mm", "Rear tyres", (v) => `${Math.round(v)} mm wide`, "Wider: more grip and a steadier temperature, more weight and drag"],
    ["wheelbase", "Wheelbase", (v) => `${v.toFixed(2)} m`, "Longer: steadier, slower to turn in, a bigger chassis"],
    ["drive_front", "Drive", DRIVE, "Which wheels push. Any split needs four-wheel-drive hardware: $100k, 70 kg"],
  ],
  setup: [
    ["engine_pos", "Engine position", ENGINE_POS, "Where the weight sits"],
    ["brake_front", "Brake balance", (v) => `${Math.round(v * 100)}% front`, "Too far either way locks one axle first"],
    ["roll_front", "Roll balance", (v) => `${Math.round(v * 100)}% front`, "Which axle takes the weight in corners: more front = more understeer"],
    ["top_gear_kmh", "Gearing", (v) => `${Math.round(v)} km/h top speed`, "Short gears pull harder, long gears go faster"],
    ["ride_height_mm", "Ride height", (v) => `${Math.round(v)} mm`, "Lower: the floor makes more downforce, until it hits the road"],
    ["spring_n_mm", "Springs", (v) => `${Math.round(v)} N/mm`, "Stiff holds the floor up; soft grips better over kerbs and bumps"],
  ],
};
const SLICE = { chassis: "#5b6478", engine: C.red, wings: "#60a5fa", tyres: C.green, awd: "#c084fc" };

let mine = JSON.parse(localStorage.getItem("bte-car") || "null") || car.mid();
let circuitIdx = Number(localStorage.getItem("bte-circuit")) || 0;
let races = Number(localStorage.getItem("bte-races")) || 0;
let wins = Number(localStorage.getItem("bte-wins")) || 0;

function buildGarage() {
  const row = ([k, label, fmt, hint]) => {
    const i = D[k], lo = car.low[i], hi = car.high[i];
    return `<div class="dial"><label for="d-${k}">${label}</label><span class="v" id="v-${k}">${fmt(mine[i])}</span>
      <input type="range" id="d-${k}" min="0" max="1000" value="${Math.round((mine[i] - lo) / (hi - lo) * 1000)}">
      <small>${hint}</small></div>`;
  };
  garage.innerHTML = `<div class="wrap">
    <h1>Beat the engineer</h1>
    <p>In episode 14 an AI engineer designed a racing car under a $1M price list. Set the same 13 dials yourself,
      then your car and the engineer's race for 60 seconds on a circuit neither of you has seen. The same AI driver
      drives both (it learned on random cars of every design, never on yours or the engineer's), so the car alone decides.</p>
    <div class="budget"><div class="total"><span>Budget</span><span id="total"></span></div>
      <div class="bar" id="bar"></div><div class="legend" id="legend"></div></div>
    <h2>Parts · cost money</h2>${DIALS.parts.map(row).join("")}
    <h2>Setup · free</h2>${DIALS.setup.map(row).join("")}
    <div class="facts" id="facts"></div>
    <h2>Circuit</h2>
    <div class="circ"><button class="btn" id="prev">◀</button><canvas id="preview" width="240" height="180"></canvas>
      <button class="btn" id="next">▶</button><span id="cname"></span></div>
    <p>${races ? `You've raced ${races} time${races > 1 ? "s" : ""} and won ${wins}.` : ""}</p>
  </div>
  <div class="go"><button class="btn" id="watchbtn">Watch the final</button><button class="btn" id="reset">Start over</button><button class="btn main" id="race">Race the engineer</button></div>`;
  for (const [k, , fmt] of [...DIALS.parts, ...DIALS.setup]) {
    const i = D[k], el = document.getElementById(`d-${k}`);
    el.oninput = () => {
      mine[i] = car.low[i] + (el.value / 1000) * (car.high[i] - car.low[i]);
      document.getElementById(`v-${k}`).textContent = fmt(mine[i]);
      update();
    };
  }
  document.getElementById("watchbtn").onclick = () => { race = null; buildWatch(); };
  document.getElementById("reset").onclick = () => { mine = car.mid(); buildGarage(); };
  document.getElementById("race").onclick = () => startRace(duel(), "duel");
  document.getElementById("prev").onclick = () => pickCircuit(-1);
  document.getElementById("next").onclick = () => pickCircuit(1);
  pickCircuit(0);
  update();
}

function pickCircuit(step) {
  circuitIdx = (circuitIdx + step + data.circuits.length) % data.circuits.length;
  localStorage.setItem("bte-circuit", circuitIdx);
  const tr = game.circuits[circuitIdx], cv = document.getElementById("preview"), x = cv.getContext("2d");
  x.setTransform(1, 0, 0, 1, 0, 0); x.clearRect(0, 0, cv.width, cv.height);   // clear in pixels, not the last circuit's zoom
  const [x0, x1, y0, y1] = bounds(tr), s = 0.85 * Math.min(cv.width / (x1 - x0), cv.height / (y1 - y0));
  x.setTransform(s, 0, 0, -s, cv.width / 2 - s * (x0 + x1) / 2, cv.height / 2 + s * (y0 + y1) / 2);
  x.strokeStyle = C.kerb; x.lineWidth = 14; x.lineJoin = "round"; x.stroke(centrePath(tr));
  document.getElementById("cname").textContent = `Test circuit ${circuitIdx + 1} of ${data.circuits.length} · ${(tr.length / 1000).toFixed(2)} km`;
}

function update() {
  localStorage.setItem("bte-car", JSON.stringify(mine));
  const parts = car.costBreakdown(mine), total = car.cost(mine), over = total > car.budget + 1e-6;
  document.getElementById("total").innerHTML = `<span style="color:${over ? C.red : C.text}">$${Math.round(total)}k</span> of $${car.budget}k`;
  const scale = Math.max(total, car.budget);
  document.getElementById("bar").innerHTML = Object.entries(parts).map(([k, v]) =>
    `<div style="width:${(v / scale) * 100}%;background:${SLICE[k]}"></div>`).join("");
  document.getElementById("legend").innerHTML = Object.entries(parts).filter(([, v]) => v > 0.5).map(([k, v]) =>
    `<span><i style="background:${SLICE[k]}"></i>${k === "awd" ? "4WD" : k} $${Math.round(v)}k</span>`).join("")
    + (over ? `<span style="color:${C.red}">over budget by $${Math.round(total - car.budget)}k</span>` : `<span>$${Math.round(car.budget - total)}k left</span>`);
  const p = car.derive(mine);
  document.getElementById("facts").textContent = `Weight ${Math.round(p.mass)} kg with fuel for the race · ` +
    `${Math.round(car.weightFront(mine) * 100)}% on the front wheels · downforce at 250 km/h about ` +
    `${Math.round(downforce(p, 250 / 3.6) / 9.81)} kg`;
  const btn = document.getElementById("race");
  btn.disabled = over;
  btn.textContent = over ? "Over budget" : "Race the engineer";
}

function downforce(p, v) {  // clean air, the floor at its static ride height (the same formula as the physics, roughly)
  const q = 0.5 * car.RHO * v * v;
  const ground = Math.min(Math.max(1 + p.ground_gain * (p.ground_ref - p.ride_height) / (p.ground_ref - p.ground_peak), 0), 1 + p.ground_gain);
  return q * (p.cla_floor * ground + p.cla_front + p.cla_rear);
}

// ── circuits (world coordinates, y up) ──
const pathCache = new Map();
function centrePath(tr) {
  if (!pathCache.has(tr)) {
    const p = new Path2D(), c = tr.center;
    p.moveTo(c[0], c[1]);
    for (let i = 1; i < tr.P; i++) p.lineTo(c[2 * i], c[2 * i + 1]);
    p.closePath();
    const walls = new Path2D();
    for (const off of [0, tr.P]) {
      walls.moveTo(tr.wallA[2 * off], tr.wallA[2 * off + 1]);
      for (let s = off; s < off + tr.P; s++) walls.lineTo(tr.wallB[2 * s], tr.wallB[2 * s + 1]);
    }
    pathCache.set(tr, p);
    pathCache.set(tr.seed, walls);
  }
  return pathCache.get(tr);
}
function bounds(tr) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < tr.P; i++) {
    x0 = Math.min(x0, tr.center[2 * i]); x1 = Math.max(x1, tr.center[2 * i]);
    y0 = Math.min(y0, tr.center[2 * i + 1]); y1 = Math.max(y1, tr.center[2 * i + 1]);
  }
  return [x0, x1, y0, y1];
}
function drawCircuit(tr, detail) {
  const p = centrePath(tr), edge = cfg.width / 2 + cfg.grass;
  g.lineJoin = "round"; g.lineCap = "round";
  if (detail) { g.strokeStyle = C.grass; g.lineWidth = 2 * edge; g.stroke(p); }
  g.strokeStyle = C.kerb; g.lineWidth = cfg.width; g.stroke(p);
  g.strokeStyle = C.asphalt; g.lineWidth = cfg.width - 2 * cfg.kerb; g.stroke(p);
  if (!detail) return;
  g.strokeStyle = C.wall; g.lineWidth = 0.6; g.stroke(pathCache.get(tr.seed));
  const c = tr.center, h = tr.heading[0], nx = -Math.sin(h), ny = Math.cos(h), half = cfg.width / 2;
  for (let k = 0; k < 8; k++) {   // chequered start line
    const u0 = -half + (k * cfg.width) / 8, u1 = u0 + cfg.width / 8;
    g.strokeStyle = k % 2 ? "#05060a" : C.text; g.lineWidth = 1.2; g.lineCap = "butt"; g.beginPath();
    g.moveTo(c[0] + nx * u0, c[1] + ny * u0); g.lineTo(c[0] + nx * u1, c[1] + ny * u1); g.stroke();
  }
}

// the car seen from above, built from its dials (metres, x forward from the centre of mass)
function drawCar(s, d, p, color, grow) {
  const sp = car.spec, tw = sp.track_width, f = p.a, r = -p.b;
  g.save(); g.translate(s.car.x, s.car.y); g.rotate(s.car.yaw); g.scale(grow, grow);
  g.fillStyle = "#05060a";
  const tyre = (x, y, mm, steer) => {
    g.save(); g.translate(x, y); g.rotate(steer); g.fillRect(-0.33, -mm / 2000, 0.66, mm / 1000); g.restore();
  };
  for (const side of [1, -1]) {
    tyre(f, side * tw / 2, d[D.tyre_front_mm], s.car.steer);
    tyre(r, side * tw / 2, d[D.tyre_rear_mm], 0);
  }
  g.fillStyle = color;
  g.beginPath(); g.roundRect(r - sp.overhang_rear + 0.15, -0.45, f - r + sp.overhang_rear + sp.overhang_front - 0.3, 0.9, 0.3); g.fill();
  g.beginPath(); g.roundRect(r + 0.3, -0.7, (f - r) * 0.6, 1.4, 0.35); g.fill();   // sidepods
  if (d[D.wing_front] > 0.02) g.fillRect(f + sp.overhang_front - 0.25, -0.85, 0.12 + 0.2 * d[D.wing_front], 1.7);
  if (d[D.wing_rear] > 0.02) g.fillRect(r - sp.overhang_rear, -0.5, 0.1 + 0.25 * d[D.wing_rear], 1.0);
  g.fillStyle = "#05060a"; g.beginPath(); g.ellipse(f - (f - r) * 0.3, 0, 0.35, 0.22, 0, 0, 2 * Math.PI); g.fill();  // cockpit
  g.restore();
}

// ── the race: any number of cars, each with its driver (the duel: you and the engineer, both the general driver) ──
let race = null, speed = 1, acc = 0, W, H, dpr, cam = { mode: "leader", x: 0, y: 0 };
function resize() {
  dpr = devicePixelRatio || 1; W = innerWidth; H = innerHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
}
addEventListener("resize", resize); resize();

const duel = () => [{ who: "you", label: "you", d: mine.slice(), color: C.amber, driver: drivers.general },
                    { who: "engineer", label: "engineer", d: ENGINEER, color: C.green, driver: drivers.general }];

function startRace(entries = race ? race.entries : duel(), kind = race ? race.kind : "duel") {
  const tr = game.circuits[circuitIdx];
  race = { tr, kind, entries, cars: entries.map((e) => ({ ...e, st: game.init(e.d, tr) })), prev: null, countdown: 2, over: false };
  race.prev = race.cars.map((c) => c.st);
  garage.style.display = "none"; result.style.display = "none";
  canvas.style.display = "block"; raceUI.style.display = "flex";
  speed = 1; document.getElementById("speed").textContent = "Speed 1x"; acc = 0;
  document.getElementById("skip").textContent = "Skip to the end";
  cam = { mode: "leader", x: tr.center[0], y: tr.center[1], who: undefined };
  const camBtn = document.getElementById("cam");
  camBtn.style.display = kind === "watch" ? "" : "none";
  camBtn.textContent = "Camera: leader";
}
document.getElementById("speed").onclick = () => {
  speed = speed === 1 ? 4 : speed === 4 ? 16 : 1;
  document.getElementById("speed").textContent = `Speed ${speed}x`;
};
document.getElementById("cam").onclick = () => {  // leader -> each car -> the whole circuit -> leader
  const n = race.cars.length, order = ["leader", ...race.cars.map((_, i) => i), "whole"];
  cam.mode = order[(order.indexOf(cam.mode) + 1) % (n + 2)];
  document.getElementById("cam").textContent = `Camera: ${cam.mode === "leader" ? "leader" : cam.mode === "whole" ? "whole circuit" : race.cars[cam.mode].label}`;
};
document.getElementById("skip").onclick = () => {  // the rest of the race, in chunks so the page stays alive
  const btn = document.getElementById("skip");
  if (race.skipping) return;
  race.skipping = true; race.countdown = 0;
  const chunk = () => {
    if (!race || race.over) return;
    for (let k = 0; k < 40 && !race.over; k++) tick();
    btn.textContent = `Racing… ${Math.round(race.cars[0].st.t * cfg.dt)} s`;
    if (!race.over) setTimeout(chunk, 0);
  };
  chunk();
};

function tick() {  // one decision for each car (20 per second, as in training)
  race.prev = race.cars.map((c) => c.st);
  for (const c of race.cars) {
    if (c.st.done) continue;
    const [s, p] = c.driver.act(game.observe(c.st));
    c.st = game.step(c.st, s, p);
  }
  if (race.cars.every((c) => c.st.done)) { race.over = true; race.kind === "watch" ? finishWatch() : finish(); }
}

const km = (m) => `${Math.round(m).toLocaleString("en")} m`;
function backTo(build) { canvas.style.display = "none"; result.style.display = "none"; race = null; build(); garage.style.display = "block"; }

function finish() {
  const [me, eng] = race.cars.map((c) => c.st);
  const gap = me.progress - eng.progress;
  races += 1; localStorage.setItem("bte-races", races);
  const won = !me.crashed && gap > 0;
  if (won) { wins += 1; localStorage.setItem("bte-wins", wins); }
  const title = me.crashed ? "YOUR CAR CRASHED" : won ? "YOU BEAT THE ENGINEER!" : Math.abs(gap) < 5 ? "DEAD HEAT" : "THE ENGINEER WINS";
  const color = me.crashed ? C.red : won ? C.green : C.amber;
  const fmt = Object.fromEntries([...DIALS.parts, ...DIALS.setup].map(([k, label, f]) => [k, [label, f]]));
  const rows = car.names.map((k, i) => `<tr><td>${fmt[k][0]}</td><td style="color:${C.amber}">${fmt[k][1](race.cars[0].d[i])}</td>` +
    `<td style="color:${C.green}">${fmt[k][1](ENGINEER[i])}</td></tr>`).join("");
  result.innerHTML = `<div class="wrap">
    <h1 style="color:${color}">${title}</h1>
    <p style="font-size:19px">In 60 seconds your car got <b style="color:${C.amber}">${km(me.progress)}</b>${me.crashed ? ` (crashed at ${(me.t * cfg.dt).toFixed(1)} s)` : ""},
      the engineer's <b style="color:${C.green}">${km(eng.progress)}</b>${eng.crashed ? " (crashed)" : ""}:
      ${Math.abs(gap) < 0.5 ? "level." : `${km(Math.abs(gap))} ${gap > 0 ? "ahead" : "behind"}.`}</p>
    <p style="color:${C.muted}">Same driver, same circuit, same start. Your car cost $${Math.round(car.cost(race.cars[0].d))}k,
      the engineer's $${Math.round(car.cost(ENGINEER))}k.</p>
    <table><tr><th></th><th style="color:${C.amber}">yours</th><th style="color:${C.green}">the engineer's</th></tr>${rows}</table>
    <div class="row"><button class="btn" id="again">Watch again</button><button class="btn" id="tweak">Change my car</button>
      <button class="btn main" id="another">Next circuit</button></div>
    <p style="color:${C.muted};margin-top:20px">The engineer's car is from episode 14 of
      <a href="https://www.youtube.com/@amilliontries" target="_blank">A Million Tries</a>.
      Won ${wins} of ${races} race${races > 1 ? "s" : ""}.</p></div>`;
  result.style.display = "block"; raceUI.style.display = "none";
  document.getElementById("again").onclick = () => startRace();
  document.getElementById("tweak").onclick = () => backTo(buildGarage);
  document.getElementById("another").onclick = () => { pickCircuit(1); startRace(); };
}

// ── watch the final race (episode 14): the finalists, each with its own driver ──
const FIELD = [
  { key: "eng-v4-2", label: "The engineer's car", color: C.green, driver: "eng-v4-2", on: true },
  { key: "sweep-v4", label: "The sweep's best", color: "#60a5fa", driver: "sweep-v4", on: true },
  { key: "f1", label: "F1-like car", color: "#c084fc", driver: "f1", on: true, note: "over the budget" },
  { key: "mid", label: "Mid-range car", color: "#9a9eb2", driver: "general", on: false, note: "every dial in the middle" },
  { key: "you", label: "Your car", color: C.amber, driver: "general", on: false, note: "from the garage" },
];
const designOf = (f) => (f.key === "you" ? mine.slice() : data.cars[f.key]);

function buildWatch() {
  const row = (f, i) => `<label class="dial" style="grid-template-columns:auto 1fr auto;align-items:center;cursor:pointer">
      <input type="checkbox" data-i="${i}" ${f.on ? "checked" : ""} style="width:22px;height:22px;accent-color:${f.color};grid-column:auto">
      <span><b style="color:${f.color}">${f.label}</b><br><small style="color:${C.muted}">$${Math.round(car.cost(designOf(f))).toLocaleString("en")}k${f.note ? " · " + f.note : ""} ·
        ${f.driver === "general" ? "the general driver (it has no driver of its own)" : "its own driver"}</small></span><span></span></label>`;
  garage.innerHTML = `<div class="wrap">
    <h1>Watch the final race</h1>
    <p>Episode 14's finalists race for 60 seconds from a standing start, each with its own AI driver (trained on cars
      like it), as in the video's final race. Pick the field and a circuit: none of them has seen it.</p>
    <h2>The field</h2>${FIELD.map(row).join("")}
    <p style="font-size:14px">Your car and the mid-range car get the general driver, which learned on random cars of
      every design; the finalists' own drivers are a little better on their own cars, so it isn't a fair fight for yours
      (for that, race the engineer in "Build a car").</p>
    <h2>Circuit</h2>
    <div class="circ"><button class="btn" id="prev">◀</button><canvas id="preview" width="240" height="180"></canvas>
      <button class="btn" id="next">▶</button><span id="cname"></span></div>
  </div>
  <div class="go"><button class="btn" id="build">Build a car</button><button class="btn main" id="race">Start the race</button></div>`;
  for (const el of garage.querySelectorAll("input[type=checkbox]")) el.onchange = () => {
    FIELD[el.dataset.i].on = el.checked;
    document.getElementById("race").disabled = !FIELD.some((f) => f.on);
  };
  document.getElementById("prev").onclick = () => pickCircuit(-1);
  document.getElementById("next").onclick = () => pickCircuit(1);
  document.getElementById("build").onclick = () => { race = null; buildGarage(); };
  document.getElementById("race").onclick = () => startRace(FIELD.filter((f) => f.on).map((f) => ({
    who: f.key, label: f.label, d: designOf(f), color: f.color, driver: drivers[f.driver], own: f.driver !== "general" })), "watch");
  pickCircuit(0);
}

function finishWatch() {
  const ranked = [...race.cars].sort((a, b) => (a.st.crashed - b.st.crashed) || (b.st.progress - a.st.progress));
  const top = ranked[0];
  const rows = ranked.map((c, i) => `<tr><td>${i + 1}</td><td style="color:${c.color}"><b>${c.label}</b></td>
    <td>${km(c.st.progress)}${c.st.crashed ? ` <span style="color:${C.red}">crashed at ${(c.st.t * cfg.dt).toFixed(1)} s</span>` : ""}</td>
    <td>${i ? "−" + km(top.st.progress - c.st.progress) : ""}</td><td>$${Math.round(car.cost(c.d)).toLocaleString("en")}k</td></tr>`).join("");
  result.innerHTML = `<div class="wrap">
    <h1 style="color:${top.color}">${top.label.toUpperCase()} WINS</h1>
    <p style="font-size:19px">${km(top.st.progress)} in 60 seconds on test circuit ${circuitIdx + 1}.</p>
    <table><tr><th></th><th>car</th><th>distance</th><th>gap</th><th>cost</th></tr>${rows}</table>
    <p style="color:${C.muted}">In the video's final race, over 56 test circuits: the engineer's car 3,554 m on average, the
      F1-like car 3,545 m (a tie, for $393k more), the sweep's best 3,390 m. One circuit can go either way.</p>
    <div class="row"><button class="btn" id="again">Watch again</button><button class="btn" id="field">Change the field</button>
      <button class="btn main" id="another">Next circuit</button></div></div>`;
  result.style.display = "block"; raceUI.style.display = "none";
  document.getElementById("again").onclick = () => startRace();
  document.getElementById("field").onclick = () => backTo(buildWatch);
  document.getElementById("another").onclick = () => { pickCircuit(1); startRace(); };
}

const lerp = (a, b, t) => a + (b - a) * t;
const text = (s, x, y, size, color, align = "left", weight = 700) => {
  g.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  g.fillStyle = color; g.textAlign = align; g.textBaseline = "top"; g.fillText(s, x, y);
};

function draw(alpha, dt) {
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
  if (!race) return;
  const cars = race.cars.map((c, i) => {
    const a = race.prev[i].car, b = c.st.car, t = race.over ? 1 : alpha;
    return { ...c, view: { ...c.st, car: { ...b, x: lerp(a.x, b.x, t), y: lerp(a.y, b.y, t), yaw: lerp(a.yaw, b.yaw, t) } } };
  });
  const order = [...cars].sort((a, b) => (a.st.crashed - b.st.crashed) || (b.st.progress - a.st.progress));
  const watch = race.kind === "watch";
  let scale, cx, cy;
  if (watch && cam.mode === "whole") {
    const [x0, x1, y0, y1] = bounds(race.tr);
    scale = 0.85 * Math.min(W / (x1 - x0), H / (y1 - y0)); cx = (x0 + x1) / 2; cy = (y0 + y1) / 2;
  } else {
    const target = !watch ? cars[0] : cam.mode === "leader" ? order[0] : cars[cam.mode];
    if (target.who !== cam.who) {   // a new car to follow: the camera keeps its place and eases the jump away
      if (cam.who !== undefined) { cam.ox = cam.x - target.view.car.x; cam.oy = cam.y - target.view.car.y; }
      cam.who = target.who;
    }
    const k = Math.exp(-5 * dt);
    cam.ox = (cam.ox || 0) * k; cam.oy = (cam.oy || 0) * k;
    cam.x = target.view.car.x + cam.ox; cam.y = target.view.car.y + cam.oy;
    scale = Math.min(W, H) / 130; cx = cam.x; cy = cam.y;            // pixels per metre
  }
  g.setTransform(dpr * scale, 0, 0, -dpr * scale, dpr * (W / 2 - cx * scale), dpr * (H / 2 + cy * scale));
  drawCircuit(race.tr, true);
  const grow = Math.max(1, 6 / scale);   // the first car solid, the others see-through on top: all show while together
  cars.forEach((c, i) => {
    g.globalAlpha = i ? 0.6 : 1;
    drawCar(c.view, c.d, c.st.p, c.st.crashed ? C.red : c.color, grow);
  });
  g.globalAlpha = 1;

  // HUD
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const pad = 18, small = Math.max(14, Math.min(W, H) / 42), big = small * 2.1;
  const t = race.cars[0].st.t * cfg.dt;
  text(`${t.toFixed(1)} s`, pad, pad, big, C.text);
  text("of 60", pad, pad + big * 1.05, small, C.muted, "left", 600);
  if (watch) {   // the order board
    order.forEach((c, i) => {
      const y = pad + big * 1.05 + small * 1.6 + i * small * 1.5, st = c.st;
      const gap = i ? `−${Math.round(order[0].st.progress - st.progress)} m` : `${Math.round(st.progress)} m`;
      text(`${i + 1}  ${c.label}  ${st.crashed ? "CRASHED" : gap}`, pad, y, small * 1.05, st.crashed ? C.red : c.color);
    });
  } else {
    cars.forEach((c, i) => {
      const y = pad + big * 1.05 + small * 1.6 + i * small * 1.5;
      const st = c.st, kmh = Math.round(Math.hypot(st.car.vx, st.car.vy) * 3.6);
      text(`${c.label}  ${Math.round(st.progress)} m  ${kmh} km/h${st.crashed ? "  CRASHED" : ""}`,
           pad, y, small * 1.15, st.crashed ? C.red : c.color);
    });
    const gap = race.cars[0].st.progress - race.cars[1].st.progress;
    text(`${gap >= 0 ? "+" : "−"}${Math.abs(gap).toFixed(0)} m`, W - pad, pad, big, gap >= 0 ? C.amber : C.green, "right");
    text(gap >= 0 ? "you lead" : "the engineer leads", W - pad, pad + big * 1.05, small, C.muted, "right", 600);
  }
  if (watch && cam.mode !== "whole") {   // the followed car's speed
    const c = cam.mode === "leader" ? order[0] : cars[cam.mode];
    text(`${Math.round(Math.hypot(c.st.car.vx, c.st.car.vy) * 3.6)} km/h`, W - pad, pad, big, c.color, "right");
    text(c.label, W - pad, pad + big * 1.05, small, C.muted, "right", 600);
  }

  // minimap
  if (!(watch && cam.mode === "whole")) {
    const mm = Math.min(W, H) * 0.3, mx = W - mm - pad, my = H - mm - pad - (W < 600 ? 150 : 70);   // above the buttons
    const [x0, x1, y0, y1] = bounds(race.tr), ms = (mm * 0.9) / Math.max(x1 - x0, y1 - y0);
    g.setTransform(dpr * ms, 0, 0, -dpr * ms, dpr * (mx + mm / 2 - ms * (x0 + x1) / 2), dpr * (my + mm / 2 + ms * (y0 + y1) / 2));
    g.globalAlpha = 0.8; drawCircuit(race.tr, false); g.globalAlpha = 1;
    for (const c of [...cars].reverse()) {
      g.fillStyle = c.color; g.beginPath(); g.arc(c.view.car.x, c.view.car.y, 6 / ms, 0, 2 * Math.PI); g.fill();
    }
  }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (race.countdown > 0) text(String(Math.ceil(race.countdown)), W / 2, H * 0.3, big * 2.5, C.amber, "center");
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (race && !race.over && !race.skipping) {
    if (race.countdown > 0) race.countdown -= dt;
    else { acc += dt * speed; while (acc >= cfg.dt && !race.over) { acc -= cfg.dt; tick(); } }
  }
  draw(race && race.countdown <= 0 ? Math.min(1, acc / cfg.dt) : 1, dt);
  requestAnimationFrame(frame);
}
(window.raceMode === "watch" ? buildWatch : buildGarage)();
requestAnimationFrame(frame);
