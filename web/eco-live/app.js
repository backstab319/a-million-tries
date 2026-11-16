// Draws the world the worker (sim.js) runs, and the panel. Map: x (the sim's first axis) runs down, y across.
import { ECO, THEME, mix, rgb } from "./theme.js";

const $ = (id) => document.getElementById(id);
const map = $("map"), ctx = map.getContext("2d"), chart = $("chart"), cctx = chart.getContext("2d");
// the sim in a worker; if one can't start here, on the page (same code, a little less smooth)
let worker = null, onPage = null;
const send = (m) => { if (worker) worker.postMessage(m); else if (onPage) onPage(m); else pending.push(m); };
async function simOnPage(why) {
  if (onPage) return;
  console.warn("eco-live: running the sim on the page:", why);
  worker = null;
  const sim = await import("./sim.js");
  onPage = sim.connect((m) => handle(m));
  for (const m of pending.splice(0)) onPage(m);
}
const pending = [];
try {
  if (new URLSearchParams(location.search).has("noworker")) throw new Error("?noworker");
  worker = new Worker(new URL("./sim.js", import.meta.url), { type: "module" });
  worker.onerror = (e) => { e.preventDefault?.(); simOnPage(e.message || "worker error"); };
  worker.onmessage = (e) => handle(e.data);
} catch (e) { simOnPage(e.message); }
window.addEventListener("error", (e) => toast(`Something broke: ${e.message}`));
window.addEventListener("unhandledrejection", (e) => toast(`Something broke: ${e.reason}`));
const C = Object.fromEntries(Object.entries(ECO).map(([k, v]) => [k, Array.isArray(v) ? v.map(rgb) : rgb(v)]));

let land = null, frame = null, grid = null, img = null, off = null, history = [], selected = -1, storms = [];
let running = true, speed = 5, startMode = "random", seed = 1 + Math.floor(Math.random() * 9999);
const DPR = Math.min(window.devicePixelRatio || 1, 2);

function toast(t) { const el = $("toast"); el.textContent = t; el.style.opacity = 1; clearTimeout(toast.h); toast.h = setTimeout(() => (el.style.opacity = 0), 2600); }

function begin(mode) {
  startMode = mode; history = []; selected = -1; $("selcard").hidden = true;
  send({ type: "start", seed, start: mode });
  send({ type: "speed", daysPerSec: speed });
  send({ type: "run", on: (running = true) });
  $("play").textContent = "❚❚ Pause";
  $("intro").hidden = true;
}
document.querySelectorAll("[data-start]").forEach((b) => (b.onclick = () => begin(b.dataset.start)));
$("play").onclick = () => { running = !running; send({ type: "run", on: running }); $("play").textContent = running ? "❚❚ Pause" : "▶ Play"; };
document.querySelectorAll("[data-speed]").forEach((b) => (b.onclick = () => {
  speed = +b.dataset.speed; send({ type: "speed", daysPerSec: speed });
  document.querySelectorAll("[data-speed]").forEach((q) => q.classList.toggle("on", q === b));
}));
$("scramble").onclick = () => { send({ type: "scramble" }); toast("Every brain swapped for a random one"); };
$("addsurv").onclick = () => { send({ type: "add", what: "survivors" }); toast("The 134 survivors join"); };
$("addrand").onclick = () => { send({ type: "add", n: 200 }); toast("200 random grass-eaters join"); };
$("new").onclick = () => { seed = 1 + Math.floor(Math.random() * 9999); $("intro").hidden = false; };
$("unsel").onclick = () => { selected = -1; send({ type: "select", x: -1e9, y: -1e9 }); $("selcard").hidden = true; };

map.addEventListener("click", (e) => {
  if (!land) return;
  const r = map.getBoundingClientRect(), L = land.n * land.cell_m;
  const y = (e.clientX - r.left) / r.width * L, x = (e.clientY - r.top) / r.height * L;
  send({ type: "select", x, y, r: Math.max(4, L / 60) });
});

function handle(m) {
  if (m.type === "land") {
    land = m; off = new OffscreenCanvas(m.n, m.n); img = new ImageData(m.n, m.n);
    $("chartnote").textContent = `world ${m.seed}; alive (log scale), shaded: winter`;
  } else if (m.type === "frame") {
    frame = m;
    if (m.grid) { grid = m.grid; paintLand(); }
    if (m.storms.length) storms = m.storms.map(([x, y]) => ({ x, y, t: performance.now() }));
    const last = history[history.length - 1];
    if (!last || m.day - last.day >= 0.25) history.push({ day: m.day, n: m.n });
    if (m.sel && !m.sel.gone) selected = m.sel.uid;
    if (m.sel && m.sel.gone) { toast("The one you followed has died"); selected = -1; $("selcard").hidden = true; }
    panel(m);
  } else if (m.type === "extinct") {
    toast(`Extinct on day ${Math.floor(m.day)}`);
    running = false; $("play").textContent = "▶ Play";
  }
};

function paintLand() {
  const n = land.n, d = img.data, perCell = 5000 * land.cell_m ** 2;
  for (let c = 0; c < n * n; c++) {
    const b = land.biome[c];
    let col;
    if (b === 0) col = C.water;
    else {
      col = C.soil[b];
      col = mix(col, C.wetSoil, Math.min(grid.wet[c], 1) * 0.45);
      const green = Math.min(grid.food[c] / (perCell * 1.2), 1), dead = Math.min(grid.dead[c] / (perCell * 0.6), 1);
      col = mix(col, C.straw, dead * 0.85);
      col = mix(col, b === 3 ? C.greenForest : C.green, Math.sqrt(green));
      if (grid.meat[c] > 5000) col = mix(col, C.carcass, Math.min(grid.meat[c] / 40000, 1) * 0.6);
      if (grid.fruit[c] > 200) col = mix(col, C.fruit, Math.min(grid.fruit[c] / 4000, 1) * 0.8);
    }
    d[4 * c] = col[0]; d[4 * c + 1] = col[1]; d[4 * c + 2] = col[2]; d[4 * c + 3] = 255;
  }
  off.getContext("2d").putImageData(img, 0, 0);   // image rows are the sim's x (cell = row * n + col)
}

function draw() {
  requestAnimationFrame(draw);
  const W = Math.round(map.clientWidth * DPR);
  if (map.width !== W) { map.width = W; map.height = W; }
  ctx.fillStyle = THEME.panel; ctx.fillRect(0, 0, W, W);
  if (!land || !frame) return;
  const L = land.n * land.cell_m, s = W / L;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, W);
  // fruit trees (crowns)
  ctx.fillStyle = ECO.crown; ctx.globalAlpha = 0.55;
  for (const [r, c] of land.trees) { ctx.beginPath(); ctx.arc((c + 0.5) * land.cell_m * s, (r + 0.5) * land.cell_m * s, 4 * s, 0, 7); ctx.fill(); }
  ctx.globalAlpha = 1;
  // storms: a soft cloud where rain fell, fading
  const now = performance.now();
  for (const st of storms) {
    const a = 1 - (now - st.t) / 2500;
    if (a <= 0) continue;
    const g = ctx.createRadialGradient(st.y * s, st.x * s, 0, st.y * s, st.x * s, 30 * s);
    g.addColorStop(0, `rgba(${C.storm.join(",")},${0.35 * a})`); g.addColorStop(1, `rgba(${C.storm.join(",")},0)`);
    ctx.fillStyle = g; ctx.fillRect(st.y * s - 30 * s, st.x * s - 30 * s, 60 * s, 60 * s);
  }
  // animals, drawn bigger than life (a 2 kg animal is ~0.3 m; here ~3 m) so they show
  const A = frame.A, base = Math.max(3 * DPR, 1.5 * s);
  let selXY = null;
  for (let j = 0; j < frame.n; j++) {
    const o = 9 * j, x = A[o], y = A[o + 1], h = A[o + 2], m = A[o + 3], diet = A[o + 4], f = A[o + 5], uid = A[o + 8];
    const X = y * s, Y = x * s, r = base * Math.cbrt(m / 2);
    if (f & 2) { ctx.strokeStyle = ECO.call; ctx.globalAlpha = 0.5; ctx.lineWidth = DPR; ctx.beginPath(); ctx.arc(X, Y, r * 2.4, 0, 7); ctx.stroke(); ctx.globalAlpha = 1; }
    const col = f & 16 ? C.furYoung : mix(C.fur[0], C.fur[1], diet);
    ctx.fillStyle = `rgb(${col.map(Math.round).join(",")})`;
    ctx.beginPath(); ctx.arc(X, Y, r, 0, 7); ctx.fill();
    ctx.strokeStyle = f & 4 ? ECO.fight : ECO.outline; ctx.lineWidth = (f & 4 ? 1.6 : 0.8) * DPR;
    ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X + Math.sin(h) * r * 1.7, Y + Math.cos(h) * r * 1.7); ctx.stroke();
    if (f & 1) { ctx.fillStyle = ECO.drink; ctx.beginPath(); ctx.arc(X, Y, r * 0.45, 0, 7); ctx.fill(); }
    if (uid === selected) selXY = [X, Y, r];
  }
  if (selXY) {
    ctx.strokeStyle = THEME.text; ctx.lineWidth = 2 * DPR;
    ctx.beginPath(); ctx.arc(selXY[0], selXY[1], selXY[2] + 7 * DPR, 0, 7); ctx.stroke();
  }
}
requestAnimationFrame(draw);

const SEASON = (d) => { const y = ((d % 360) + 360) % 360; return y < 45 || y >= 315 ? "winter" : y < 135 ? "spring" : y < 225 ? "summer" : "autumn"; };
const fmtKg = (m) => (m >= 1 ? `${m.toFixed(1)} kg` : `${Math.round(m * 1000)} g`);

function panel(m) {
  $("when").textContent = `Year ${Math.floor(m.day / 360) + 1} · day ${Math.floor(m.day % 360)} · ${SEASON(m.day)} · ${m.temp.toFixed(0)} °C`;
  $("alive").textContent = m.n.toLocaleString();
  $("born").textContent = m.books.births.toLocaleString();
  $("gen").textContent = m.genMax;
  $("mass").textContent = fmtKg(m.massMean);
  $("diet").textContent = m.dietMean < 0.33 ? `grass ${(m.dietMean).toFixed(2)}` : m.dietMean < 0.66 ? `mixed ${m.dietMean.toFixed(2)}` : `meat ${m.dietMean.toFixed(2)}`;
  const cs = m.causes, tot = Math.max(cs.thirst + cs.starved + cs.killed + cs.old, 1);
  const parts = [["thirst", "thirst"], ["starved", "starved"], ["killed", "killed"], ["old", "old age"]];
  $("deaths").innerHTML = parts.map(([k]) => `<div style="width:${100 * cs[k] / tot}%;background:${ECO[k]}"></div>`).join("");
  $("legend").innerHTML = parts.map(([k, t]) => `<span><i style="background:${ECO[k]}"></i>${t} ${cs[k].toLocaleString()}</span>`).join("");
  drawChart();
  if (m.sel && !m.sel.gone) {
    const q = m.sel;
    $("selcard").hidden = false;
    $("seltitle").textContent = `#${q.uid}`;
    $("selthirst").style.width = `${Math.min(q.thirst, 1) * 100}%`; $("selthirst").style.background = ECO.thirst;
    $("selres").style.width = `${Math.max(Math.min(q.reserve, 1), 0) * 100}%`; $("selres").style.background = THEME.green;
    $("selgrid").innerHTML = `<span class="muted">mass</span><b>${fmtKg(q.mass)} of ${fmtKg(q.adult)}</b>
      <span class="muted">age</span><b>${q.age.toFixed(0)} of ~${q.life.toFixed(0)} days</b>
      <span class="muted">generation</span><b>${q.gen}</b><span class="muted">diet</span><b>${q.diet.toFixed(2)}</b>
      <span class="muted">breeds alone</span><b>${(q.alone * 100).toFixed(0)}%</b><span class="muted">parent</span><b>${q.parent < 0 ? "none (placed in the world)" : "#" + q.parent}</b>`;
    if (q.out) {
      const o = q.out, doing = [];
      if (o[2] > 0) doing.push(o[1] > 0 ? "eating (slowly: a mouth can't eat at a run)" : "eating");
      if (o[4] > 0) doing.push("attacking");
      if (o[5] > 0) doing.push("calling");
      if (o[3] > 0) doing.push("ready to breed");
      $("seldoing").textContent = `Its brain says: turn ${o[0] > 0 ? "right" : "left"}, ${(100 / (1 + Math.exp(-o[1]))).toFixed(0)}% speed` + (doing.length ? `; ${doing.join(", ")}` : "");
    }
  }
}

function drawChart() {
  const W = chart.width, H = chart.height, g = cctx;
  g.clearRect(0, 0, W, H);
  if (history.length < 2) return;
  const d0 = history[0].day, d1 = Math.max(history[history.length - 1].day, d0 + 5);
  const X = (d) => ((d - d0) / (d1 - d0)) * (W - 8) + 4;
  let top = 10;
  for (const p of history) top = Math.max(top, p.n);
  const Y = (n) => H - 6 - (Math.log10(1 + n) / Math.log10(1 + top * 1.3)) * (H - 14);
  g.fillStyle = "rgba(111,211,255,0.08)";   // winters
  for (let y = Math.floor(d0 / 360) * 360 - 45; y < d1; y += 360) { const a = X(Math.max(y, d0)), b = X(Math.min(y + 90, d1)); if (b > a) g.fillRect(a, 0, b - a, H); }
  g.strokeStyle = THEME.green; g.lineWidth = 2.5; g.beginPath();
  history.forEach((p, k) => (k ? g.lineTo(X(p.day), Y(p.n)) : g.moveTo(X(p.day), Y(p.n))));
  g.stroke();
  g.fillStyle = THEME.muted; g.font = `22px ${THEME.font}`;
  g.fillText(top.toLocaleString(), 6, 24);
}
