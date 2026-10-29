// Race day (episode 15) in the browser, drawn like scripts/race_tv.py (the map, a battle cam, the timing tower).
// Eight cars, 25 laps, one AI driver (race-day-driver-ep15) in every car; the pit calls are the contest.
//   window.racedayMode = "replay": the video's final race (circuit 100003), replayed: every car every 0.2 s from the
//     Python race (raceday-data.json, `python -m ailearns.export_web raceday`). Nothing is simulated.
//   window.racedayMode = "live": the race run here (field-engine.js, a port of the Python race, checked by
//     field-check.mjs) on any of 8 test circuits. Be the pit wall for car 1 (the AI pit wall, race-pit-wall-ep15, runs
//     cars 2-4, the rule cars 5-8), or just watch the AI pit wall (cars 1-4) take on the rule (cars 5-8).

const C = { bg: "#0d0f16", panel: "#161923", edge: "#262b3a", tarmac: "#282c3a", grass: "#163626", wall: "#969eb2",
            text: "#e8eaf0", muted: "#787f94", green: "#4ade80", amber: "#fbbf24", red: "#f43f5e", blue: "#60a5fa" };
const CARS = ["#4ade80", "#f43f5e", "#60a5fa", "#fbbf24", "#a78bfa", "#fb923c", "#2dd4bf", "#f472b6"];
const TYRE = [C.red, C.amber, C.text];  // soft, medium, hard (TV's colours)
const TYRES = ["soft", "medium", "hard"];
const LIVE = window.racedayMode === "live";
const AI_AVG = 4.08, RULE_AVG = 4.92;   // over 128 races on test circuits (scripts/strat_compare.py, devlog 9l)

// ── page ──
const css = document.createElement("style");
css.textContent = `
  #rd-ui { position: fixed; left: 0; right: 0; bottom: 0; display: flex; gap: 8px; align-items: center; padding: 8px 10px;
    background: ${C.bg}ee; border-top: 1px solid ${C.edge}; box-sizing: border-box; }
  #rd-ui button, .rd-card button { padding: 10px 14px; border-radius: 12px; border: 2px solid ${C.edge}; background: ${C.panel};
    color: ${C.text}; font: inherit; font-weight: 700; font-size: 16px; cursor: pointer; white-space: nowrap; }
  #rd-ui input { flex: 1; min-width: 60px; accent-color: ${C.amber}; height: 28px; }
  #rd-ui .grow { flex: 1; }
  @media (max-width: 600px) { #rd-ui { flex-wrap: wrap; justify-content: center; } #rd-ui input { order: -1; flex-basis: 100%; } }
  #rd-wall { position: fixed; left: 0; right: 0; display: none; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: center;
    padding: 8px 10px; background: ${C.panel}; border-top: 1px solid ${C.edge}; box-sizing: border-box; font-size: 15px; }
  #rd-wall b { color: ${C.amber}; } #rd-wall .m { color: ${C.muted}; }
  @media (max-width: 600px) { #rd-wall { font-size: 13px; padding: 6px; gap: 4px; } #rd-wall .wide { display: none; }
    #rd-wall button { padding: 6px 9px; } }
  #rd-wall button { padding: 8px 12px; border-radius: 10px; border: 2px solid ${C.edge}; background: ${C.bg}; color: ${C.text};
    font: inherit; font-weight: 700; cursor: pointer; }
  #rd-wall button.on { border-color: ${C.amber}; color: ${C.amber}; }
  .rd-over { position: fixed; inset: 0; display: none; overflow-y: auto; touch-action: pan-y; background: ${C.bg}f7;
    padding: 24px 16px 40px; box-sizing: border-box; z-index: 5; }
  .rd-over .wrap { max-width: 620px; margin: 0 auto; text-align: center; }
  .rd-over h1 { font-size: clamp(28px, 7vw, 48px); margin: 10px 0; }
  .rd-over p { color: ${C.muted}; line-height: 1.45; }
  .rd-over b { color: ${C.text}; }
  .rd-over table { width: 100%; border-collapse: collapse; margin: 18px 0; text-align: left; font-variant-numeric: tabular-nums; }
  .rd-over td, .rd-over th { padding: 6px 8px; border-bottom: 1px solid ${C.edge}; }
  .rd-over th { color: ${C.muted}; font-weight: 600; }
  .rd-over button { padding: 14px 22px; border-radius: 14px; border: 2px solid ${C.edge}; background: ${C.panel}; color: ${C.text};
    font: inherit; font-weight: 700; font-size: 17px; cursor: pointer; margin: 4px; }
  .rd-over button.main { background: ${C.amber}; color: #111; border-color: ${C.amber}; }
  .rd-over button.pick { border-color: ${C.amber}; color: ${C.amber}; }
  .rd-over canvas { width: 220px; height: 160px; background: ${C.panel}; border-radius: 12px; }
  .rd-over label { color: ${C.muted}; }
  .rd-card { position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%); width: min(460px, 92vw); display: none; z-index: 4;
    background: ${C.panel}ee; border: 2px solid ${C.amber}; border-radius: 18px; padding: 16px 18px; box-sizing: border-box; text-align: center;
    max-height: calc(100% - 16px); overflow-y: auto; touch-action: pan-y; }   /* a short sideways phone: scroll, don't clip */
  .rd-card h2 { margin: 0 0 6px; font-size: 22px; } .rd-card p { margin: 6px 0; color: ${C.muted}; line-height: 1.4; }
  .rd-card b { color: ${C.text}; }
  .rd-card .row { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 10px; }`;
document.head.append(css);
document.body.insertAdjacentHTML("beforeend", `<canvas id="c"></canvas>
  <div id="rd-wall"></div>
  <div id="rd-ui"><button id="play">Pause</button><button id="speed">20x</button><input id="scrub" type="range" min="0" max="1" step="0.1" value="0"><span class="grow" id="gap" style="display:none"></span><button id="camb">Camera: auto</button></div>
  <div class="rd-card" id="rd-ask"></div>
  <div class="rd-over" id="rd-end"></div><div class="rd-over" id="rd-setup"></div>`);
const canvas = document.getElementById("c"), g = canvas.getContext("2d");
const ui = document.getElementById("rd-ui"), scrub = document.getElementById("scrub"), endBox = document.getElementById("rd-end");
const wallBar = document.getElementById("rd-wall"), askCard = document.getElementById("rd-ask"), setupBox = document.getElementById("rd-setup");
if (LIVE) { scrub.style.display = "none"; document.getElementById("gap").style.display = ""; }

// ── where the race comes from: a recording, or the race itself ──
const TYPES = { int16: Int16Array, uint8: Uint8Array, float32: Float32Array };
const arr = (o) => { const b = Uint8Array.from(atob(o.b64), (c) => c.charCodeAt(0)); return new TYPES[o.dtype || "float32"](b.buffer); };

async function replaySource() {
  const d = await (await fetch("raceday-data.json")).json();
  const FL = { finished: 1, walled: 2, wrecked: 4, dry: 8, pit: 16, bump: 32, gone: 64, call: 128 };  // export_web RACEDAY_FLAGS
  const pos = arr(d.pos), yaw = arr(d.yaw), prog = arr(d.progress), order = arr(d.order), state = arr(d.state);
  const comp = arr(d.compound), stops = arr(d.stops), wear = arr(d.wear), fuel = arr(d.fuel), speedK = arr(d.speed);
  const N = d.team.length, dt = d.dt, has = (j, c, f) => (state[j * N + c] & f) !== 0;
  const isOut = (j, c) => has(j, c, FL.walled) || has(j, c, FL.wrecked) || (has(j, c, FL.dry) && !has(j, c, FL.finished));
  let n = d.order.shape[0];   // the replay ends a few seconds after the last car is home or out
  for (let j = 0; j < n; j++) {
    let all = true;
    for (let c = 0; c < N; c++) all &&= has(j, c, FL.finished) || isOut(j, c);
    if (all) { n = Math.min(n, j + Math.round(4 / dt)); break; }
  }
  let lastJ = 0;
  const src = {
    N, team: d.team, labels: d.team.map((t) => (t ? "AI" : "RULE")), L: d.length, laps: d.laps, seed: d.seed, cfg: d.cfg,
    pitAt: d.pit_at, centre: arr(d.center), normal: arr(d.normal), end: (n - 1) * dt, t: 0, live: false,
    driver: d.driver, strategist: d.strategist,
    seek(t) { src.t = Math.max(0, Math.min(src.end, t)); lastJ = Math.floor(src.t / dt); },
    advance(raceS) { src.t = Math.max(0, Math.min(src.end, src.t + raceS)); },   // (seek() would skip the events)
    view() {
      const i = Math.min(src.t / dt, n - 1), j = Math.floor(i), u = i - j, j2 = Math.min(j + 1, n - 1), v = blank(N);
      v.t = j * dt;
      for (let c = 0; c < N; c++) {
        const a = (j * N + c) * 2, b = (j2 * N + c) * 2, k = j * N + c;
        let dy = (yaw[j2 * N + c] - yaw[k]) / 1e4;
        dy = ((dy + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
        v.list.push({ c, x: (pos[a] * (1 - u) + pos[b] * u) / 10, y: (pos[a + 1] * (1 - u) + pos[b + 1] * u) / 10, h: yaw[k] / 1e4 + dy * u });
        v.order[c] = order[k]; v.fin[c] = has(j, c, FL.finished); v.out[c] = isOut(j, c); v.wrecked[c] = has(j, c, FL.wrecked);
        v.walled[c] = has(j, c, FL.walled); v.dry[c] = has(j, c, FL.dry); v.pit[c] = has(j, c, FL.pit); v.gone[c] = has(j, c, FL.gone);
        v.comp[c] = comp[k]; v.stops[c] = stops[k]; v.wear[c] = wear[k] / 255; v.fuel[c] = fuel[k] / 255; v.prog[c] = prog[k];
        v.speed[c] = speedK[k] * 2;
      }
      for (let jj = lastJ + 1; jj <= j; jj++)   // events in the frames skipped since the last draw
        for (let c = 0; c < N; c++) { if (has(jj, c, FL.call)) v.calls.push(c); if (has(jj, c, FL.bump)) v.bumps.push(c); }
      lastJ = j;
      return v;
    },
  };
  return src;
}

function blank(N) {
  const a = () => Array(N).fill(0);
  return { t: 0, list: [], order: a(), fin: a(), out: a(), wrecked: a(), walled: a(), dry: a(), pit: a(), gone: a(), comp: a(),
           stops: a(), wear: a(), fuel: a(), prog: a(), speed: a(), calls: [], bumps: [] };
}

let fieldData = null, engine = null;
async function liveSource({ circuit, player, startTyre, videoCars, rand = Math.random }) {
  fieldData ??= await (await fetch("field-data.json")).json();
  engine ??= await import("./field-engine.js");
  const d = fieldData, team = d.team.slice();
  // every car a random design, every race (as the 128-race comparison: in any one field some cars are simply faster;
  // in the video's field cars 1-4 are the slower ones), or the video race's 8 cars
  const designs = videoCars ? d.designs : d.designs.map(() => d.design.map((p) => p.low + rand() * (p.high - p.low)));
  const race = new engine.Race(d, circuit, {
    rand, designs, player: player ? 0 : -1, human: () => wall.choice(),
    team: player ? team.map((t, i) => t && i !== 0) : team, start: (i) => (player && i === 0 ? startTyre : null) });
  const F = race.F, N = F.n, tr = race.tr, dt = F.cfg.dt;
  const centre = tr.center, normal = new Float64Array(2 * tr.P);
  for (let i = 0; i < tr.P; i++) { normal[2 * i] = -Math.sin(tr.heading[i]); normal[2 * i + 1] = Math.cos(tr.heading[i]); }
  const snap = () => race.f.cars.map((c) => [c.car.x, c.car.y, c.car.yaw]);
  let prev = snap(), owed = 0, calls = [], bumps = [], flagT = null;
  const tyres = race.f.setup.map((s) => [s.compound]);   // each car's sets this race
  const src = {
    N, team: race.team, labels: race.team.map((t, i) => (player && i === 0 ? "YOU" : t ? "AI" : "RULE")), L: tr.length,
    laps: F.laps, seed: tr.seed, cfg: F.cfg, pitAt: tr.pitAt, centre, normal, end: F.cfg.time_limit, t: 0, live: true,
    race, tyres, player, driver: d.names.driver, strategist: d.names.strategist,
    // (as the chequered flag works: once the winner is home the rest finish as they cross the line; a car that never
    // does, say stuck in a pit-lane tangle (the Python race has those too), is classified where it is. The race used
    // to run on to its 60-minute cap for it; positions don't change, the wait does)
    get done() { return race.f.done || (flagT !== null && race.f.t * dt - flagT > 180); },
    // run the race on by up to raceS seconds, within `budgetMs` of compute; stops before the player's call
    advance(raceS, budgetMs = 12) {
      owed += raceS;
      const t0 = performance.now();
      while (owed >= dt && !src.done) {
        if (player && race.atLine(0) && !wall.answered) { wall.ask(src); owed = 0; break; }
        prev = snap();
        const stops = race.f.stops.slice(), out = race.step();
        wall.answered = false;
        calls.push(...out.call);
        race.f.bump.forEach((b, c) => { if (b > 0) bumps.push(c); });
        race.f.stops.forEach((s, c) => { if (s > stops[c]) tyres[c].push(race.f.setup[c].compound); });
        owed -= dt;
        if (race.f.flag && flagT === null) flagT = race.f.t * dt;
        if (performance.now() - t0 > budgetMs) { owed = Math.min(owed, dt); break; }   // the machine's top speed
      }
      src.t = race.f.t * dt;
    },
    view() {
      const f = race.f, v = blank(N), u = Math.min(owed / dt, 1), order = F.order(f);
      v.t = f.t * dt;
      f.cars.forEach((c, i) => {
        const [x0, y0, h0] = prev[i];
        let dh = c.car.yaw - h0;
        dh = ((dh + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
        v.list.push({ c: i, x: x0 + (c.car.x - x0) * u, y: y0 + (c.car.y - y0) * u, h: h0 + dh * u });
        const dry = F.dryOut(f, i);
        v.order[i] = order[i]; v.fin[i] = f.finished[i]; v.wrecked[i] = f.wrecked[i]; v.walled[i] = c.crashed && !f.wrecked[i];
        v.dry[i] = dry; v.out[i] = c.crashed || f.wrecked[i] || dry; v.pit[i] = F.game.inPitLane(c); v.gone[i] = F.gone(f, i);
        v.comp[i] = f.setup[i].compound; v.stops[i] = f.stops[i]; v.wear[i] = c.car.wear.reduce((a, b) => a + b, 0) / 4;
        v.fuel[i] = 1 - c.car.fuel_used / f.setup[i].fuel; v.prog[i] = c.progress; v.speed[i] = Math.round(Math.hypot(c.car.vx, c.car.vy) * 3.6);
      });
      v.calls = calls; v.bumps = bumps; calls = []; bumps = [];
      return v;
    },
  };
  return src;
}

// ── the player's pit wall (live): a standing order, and the call at each decision line ──
const wall = {
  box: false, compound: 2, ask: null, answered: false, askEvery: true, pending: false,
  choice() { const c = { pit: wall.box, compound: wall.compound }; wall.box = false; wall.draw(); return c; },
  draw() {
    if (!src?.player) return;
    const r = src.race, f = r.f, c = f.cars[0], worst = Math.max(...c.car.wear), toLine = r.toEntry(c) - r.decideM;
    const lap = Math.min(Math.floor(Math.max(c.progress, 0) / src.L) + 1, src.laps), set = src.tyres[0].length;
    wallBar.innerHTML = `<span><b>CAR 1</b><span class="wide"> · ${TYRES[f.setup[0].compound]}s</span>, worst tyre ${Math.round(worst * 100)}%
      · fuel ${Math.round((1 - c.car.fuel_used / f.setup[0].fuel) * 100)}% · <span class="m">${f.box_call[0] ? "called in: box this lap" :
      r.racing(0) ? `next call in ${(mod(toLine, src.L) / 1000).toFixed(1)} km` : "out of the race"}</span></span>
      <span><button id="w-box" class="${wall.box ? "on" : ""}">${wall.box ? "Box: YES" : "Box: no"}</button>
      ${[0, 1, 2].map((k) => `<button data-k="${k}" class="${wall.compound === k ? "on" : ""}">${TYRES[k][0].toUpperCase()}</button>`).join("")}
      <button id="w-ask" class="${wall.askEvery ? "on" : ""}">${wall.askEvery ? "Ask each lap" : "Auto"}</button></span>`;
    document.getElementById("w-box").onclick = () => { wall.box = !wall.box; wall.draw(); };
    document.getElementById("w-ask").onclick = () => { wall.askEvery = !wall.askEvery; wall.draw(); };
    for (const b of wallBar.querySelectorAll("button[data-k]")) b.onclick = () => { wall.compound = +b.dataset.k; wall.draw(); };
  },
  // the car is at its decision line: ask (pausing the race), or use the standing order
  ask(s) {
    if (!wall.askEvery) { wall.answered = true; return; }
    const r = s.race, f = r.f, c = f.cars[0], order = r.F.order(f), lap = Math.floor(Math.max(c.progress, 0) / s.L) + 1;
    // the nearest cars still racing (a gap to a car that's out or finished means nothing)
    const near = (dir) => { let b = -1; for (let k = 1; k < f.cars.length; k++) if (r.racing(k) && (order[k] - order[0]) * dir > 0
      && (b < 0 || (order[k] - order[b]) * dir < 0)) b = k; return b; };
    const ahead = near(-1), behind = near(1), gap = (k) => Math.abs(f.cars[k].progress - c.progress);
    const worst = Math.max(...c.car.wear), laps = Math.max(0, s.laps - c.progress / s.L);
    wall.pending = true; paused = true;
    askCard.innerHTML = `<h2>Car 1, lap ${lap} of ${s.laps}: box or stay out?</h2>
      <p>Car 1 is ${r.decideM} m from the pit entry. <b>P${order[0] + 1}</b>${ahead >= 0 ? `, ${Math.round(gap(ahead))} m behind car ${ahead + 1}` : ""}${behind >= 0 ? `, ${Math.round(gap(behind))} m ahead of car ${behind + 1}` : ""}.</p>
      <p>On <b>${TYRES[f.setup[0].compound]}s</b>, worst tyre <b>${Math.round(worst * 100)}% worn</b> (they fail past ~90%) · fuel <b>${Math.round((1 - c.car.fuel_used / f.setup[0].fuel) * 100)}%</b> · <b>${laps.toFixed(1)} laps</b> to go · ${f.stops[0]} stop${f.stops[0] === 1 ? "" : "s"} so far</p>
      <div class="row"><button data-a="out">Stay out</button>${[0, 1, 2].map((k) => `<button data-a="${k}" style="color:${TYRE[k]}">Box for ${TYRES[k]}s</button>`).join("")}</div>
      <p style="font-size:13px">A stop costs about 20 s (the pit lane's 80 km/h limit plus the tyre change).</p>`;
    askCard.style.display = "block";
    for (const b of askCard.querySelectorAll("button")) b.onclick = () => {
      wall.box = b.dataset.a !== "out";
      if (wall.box) wall.compound = +b.dataset.a;
      wall.answered = true; wall.pending = false; paused = false; askCard.style.display = "none"; wall.draw();
    };
  },
};
const mod = (a, n) => ((a % n) + n) % n;

// ── layout ──
let W, H, dpr, lay;
function resize() {
  dpr = devicePixelRatio || 1; W = innerWidth;
  wallBar.style.bottom = `${ui.offsetHeight}px`;
  H = innerHeight - ui.offsetHeight - (wallBar.style.display === "flex" ? wallBar.offsetHeight : 0);
  canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.height = `${H}px`;
  const m = 8;
  if (W > H * 1.15) {   // landscape: the map and the battle cam on the left, the tower on the right
    const tw = Math.min(460, W * 0.38), lw = W - tw - 3 * m;
    lay = { map: [m, m, lw, (H - 3 * m) * 0.68], battle: [m, 2 * m + (H - 3 * m) * 0.68, lw, (H - 3 * m) * 0.32],
            tower: [2 * m + lw, m, tw, H - 2 * m] };
  } else {              // portrait (a phone): map, battle cam, tower, top to bottom
    const mh = Math.min(W * 0.8, H * 0.36), bh = Math.min(W * 0.4, H * 0.18);
    lay = { map: [m, m, W - 2 * m, mh], battle: [m, 2 * m + mh, W - 2 * m, bh], tower: [m, 3 * m + mh + bh, W - 2 * m, H - 4 * m - mh - bh] };
  }
}
addEventListener("resize", resize);

// ── the circuit (world metres, y up), built for the current race ──
let geo = null;
function buildCircuit(s) {
  const { centre, normal, cfg, L } = s, P = centre.length / 2, mid = s.pitAt;
  const at = (m, lat) => {   // metres along the circuit, metres left of the centreline -> world
    const i = ((Math.round((m / L) * P) % P) + P) % P;
    return [centre[2 * i] + normal[2 * i] * lat, centre[2 * i + 1] + normal[2 * i + 1] * lat];
  };
  const loop = new Path2D(); loop.moveTo(centre[0], centre[1]);
  for (let i = 1; i < P; i++) loop.lineTo(centre[2 * i], centre[2 * i + 1]);
  loop.closePath();
  const edges = new Path2D();
  for (const side of [1, -1]) {
    const lat = side * cfg.width / 2;
    edges.moveTo(centre[0] + normal[0] * lat, centre[1] + normal[1] * lat);
    for (let i = 1; i <= P; i++) { const k = i % P; edges.lineTo(centre[2 * k] + normal[2 * k] * lat, centre[2 * k + 1] + normal[2 * k + 1] * lat); }
  }
  const lane = new Path2D(), pitWall = new Path2D();
  for (let q = 0; q <= 60; q++) { const p = at(mid - cfg.pit_entry + (q / 60) * (cfg.pit_entry + cfg.pit_exit), -cfg.width / 2); q ? lane.lineTo(...p) : lane.moveTo(...p); }
  for (let q = 60; q >= 0; q--) lane.lineTo(...at(mid - cfg.pit_entry + (q / 60) * (cfg.pit_entry + cfg.pit_exit), cfg.pit_outer));
  lane.closePath();
  for (let q = 0; q <= 30; q++) { const p = at(mid - cfg.pit_wall_len + (q / 30) * 2 * cfg.pit_wall_len, cfg.pit_wall_lat); q ? pitWall.lineTo(...p) : pitWall.moveTo(...p); }
  const boxes = CARS.map((_, b) => at(mid + (b - (s.N - 1) / 2) * 12, cfg.pit_outer + 1));
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < P; i++) {
    for (const lat of [cfg.width / 2 + cfg.grass, -(cfg.width / 2 + cfg.grass) + Math.min(0, cfg.pit_outer + cfg.width / 2)]) {
      const x = centre[2 * i] + normal[2 * i] * lat, y = centre[2 * i + 1] + normal[2 * i + 1] * lat;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
  }
  return { loop, edges, lane, pitWall, boxes, x0, x1, y0, y1, centre, cfg };
}

function view([rx, ry, rw, rh], cx, cy, s) {   // world -> this panel, s px per metre (CSS px)
  g.setTransform(dpr * s, 0, 0, -dpr * s, dpr * (rx + rw / 2 - cx * s), dpr * (ry + rh / 2 + cy * s));
}
function drawCircuit(s) {
  const { loop, edges, lane, pitWall, boxes, centre, cfg } = geo;
  g.lineJoin = "round"; g.lineCap = "round";
  g.strokeStyle = C.grass; g.lineWidth = cfg.width + 2 * cfg.grass; g.stroke(loop);
  g.fillStyle = C.tarmac; g.fill(lane);
  g.strokeStyle = C.amber; g.lineWidth = 1.5 / s; g.stroke(lane);   // outlined: on the map the lane is a few px wide
  g.strokeStyle = C.tarmac; g.lineWidth = cfg.width; g.stroke(loop);
  g.strokeStyle = C.wall; g.lineWidth = Math.max(0.3, 0.8 / s); g.stroke(edges);
  g.strokeStyle = C.wall; g.lineWidth = Math.max(0.5, 1.5 / s); g.stroke(pitWall);
  boxes.forEach(([x, y], b) => { g.fillStyle = CARS[b]; g.beginPath(); g.arc(x, y, Math.max(1, 2.5 / s), 0, 2 * Math.PI); g.fill(); });
  const h = Math.atan2(centre[3] - centre[1], centre[2] - centre[0]), nx = -Math.sin(h), ny = Math.cos(h), half = cfg.width / 2;
  g.lineCap = "butt";
  for (let k = 0; k < 8; k++) {   // chequered start line
    const u0 = -half + (k * cfg.width) / 8, u1 = u0 + cfg.width / 8;
    g.strokeStyle = k % 2 ? "#05060a" : C.text; g.lineWidth = Math.max(1.2, 2 / s); g.beginPath();
    g.moveTo(centre[0] + nx * u0, centre[1] + ny * u0); g.lineTo(centre[0] + nx * u1, centre[1] + ny * u1); g.stroke();
  }
}
// an F1-like car from above (metres, x forward), grown to at least `minPx` long on screen
function drawCar(x, y, h, color, s, minPx) {
  const k = Math.max(1, minPx / (5.4 * s));
  g.save(); g.translate(x, y); g.rotate(h); g.scale(k, k);
  g.fillStyle = "#05060a";
  for (const [tx, ty] of [[1.6, 0.8], [1.6, -0.8], [-1.6, 0.8], [-1.6, -0.8]]) g.fillRect(tx - 0.33, ty - 0.2, 0.66, 0.4);
  g.fillStyle = color;
  g.beginPath(); g.roundRect(-2.4, -0.42, 4.9, 0.84, 0.3); g.fill();
  g.beginPath(); g.roundRect(-1.4, -0.68, 2.0, 1.36, 0.35); g.fill();   // sidepods
  g.fillRect(2.35, -0.88, 0.3, 1.76); g.fillRect(-2.7, -0.5, 0.35, 1.0);   // wings
  g.fillStyle = "#05060a"; g.beginPath(); g.ellipse(0.5, 0, 0.4, 0.24, 0, 0, 2 * Math.PI); g.fill();
  g.restore();
}

// ── playback ──
let src = null, speed = 20, paused = false, cam = { follow: null, pair: null, zoom: null }, ended = false, V = null;
let flash = [];   // real ms of each car's last box call / hit
let measured = { since: performance.now(), raceT: 0, x: 0 };
const SPEEDS = LIVE ? [1, 5, 10, 20, 60] : [1, 5, 20, 60];
const btnPlay = document.getElementById("play"), btnSpeed = document.getElementById("speed"), btnCam = document.getElementById("camb");
btnPlay.onclick = () => {
  if (wall.pending) return;
  if (!LIVE && src.t >= src.end) { src.seek(0); ended = false; }
  if (LIVE && src.done) { showSetup(); return; }
  paused = !paused; btnPlay.textContent = paused ? "Play" : "Pause";
};
btnSpeed.onclick = () => { speed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length]; btnSpeed.textContent = `${speed}x`; };
const camName = () => (cam.follow === null ? "Camera: auto" : `Camera: car ${cam.follow + 1}`);
btnCam.onclick = () => { cam.follow = cam.follow === null ? 0 : cam.follow + 1 < src.N ? cam.follow + 1 : null; btnCam.textContent = camName(); };
scrub.oninput = () => { src.seek(+scrub.value); ended = false; };
canvas.onclick = (e) => {   // a row of the tower: follow that car (again: back to auto)
  const [tx, ty, tw] = lay.tower, rh = rowH();
  if (!V || e.clientX < tx || e.clientX > tx + tw || e.clientY < ty + head()) return;
  const r = Math.floor((e.clientY - ty - head()) / rh);
  for (let c = 0; c < src.N; c++) if (V.order[c] === r) cam.follow = cam.follow === c ? null : c;
  btnCam.textContent = camName();
};

// ── drawing ──
const text = (s, x, y, size, color, align = "left", weight = 700, base = "alphabetic") => {
  g.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  g.fillStyle = color; g.textAlign = align; g.textBaseline = base; g.fillText(s, x, y);
};
const panel = ([x, y, w, h]) => { g.fillStyle = C.panel; g.strokeStyle = C.edge; g.lineWidth = 1.5; g.beginPath(); g.roundRect(x, y, w, h, 12); g.fill(); g.stroke(); };
const bar = (x, y, w, h, v, color) => {
  g.fillStyle = C.edge; g.beginPath(); g.roundRect(x, y, w, h, h / 2); g.fill();
  g.fillStyle = color; g.beginPath(); g.roundRect(x, y, Math.max(2, w * Math.min(1, Math.max(0, v))), h, h / 2); g.fill();
};
const head = () => Math.min(70, lay.tower[3] * 0.11);
const rowH = () => (lay.tower[3] - head() - 6) / src.N;
const mix = (hex, a) => {   // the car's colour faded towards the background (out or finished)
  const p = (h, i) => parseInt(h.slice(i, i + 2), 16), b = C.bg;
  return `rgb(${[1, 3, 5].map((i) => Math.round(p(hex, i) * (1 - a) + p(b, i) * a)).join(",")})`;
};
const byOrder = (v) => v.list.slice().sort((a, b) => v.order[b.c] - v.order[a.c]);   // the leader drawn last (on top)

function drawMap(now, v) {
  const r = lay.map, s = Math.min(r[2] / (geo.x1 - geo.x0), r[3] / (geo.y1 - geo.y0)) * 0.96;
  panel(r);
  g.save(); g.beginPath(); g.rect(r[0], r[1], r[2], r[3]); g.clip();
  view(r, (geo.x0 + geo.x1) / 2, (geo.y0 + geo.y1) / 2, s);
  drawCircuit(s);
  for (const k of byOrder(v)) {
    if (v.gone[k.c]) continue;
    if (now - flash[k.c].bump < 400) { g.fillStyle = "#e8eaf055"; g.beginPath(); g.arc(k.x, k.y, 14 / s, 0, 2 * Math.PI); g.fill(); }
    drawCar(k.x, k.y, k.h, v.out[k.c] || v.fin[k.c] ? mix(CARS[k.c], 0.6) : CARS[k.c], s, Math.min(22, r[2] / 30));
  }
  g.restore();
}

function drawBattle(v) {
  const r = lay.battle, run = v.list.filter((k) => !(v.out[k.c] || v.fin[k.c]));
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y), byC = (c) => v.list[c];
  let target, label;
  const lead = v.list.find((k) => v.order[k.c] === 0);
  if (cam.follow !== null) {
    target = byC(cam.follow); label = `FOLLOWING · car ${cam.follow + 1}`; cam.pair = null;
  } else {   // the closest fight (kept while within 90 m), else the leader (race_tv.py's battle cam)
    let pair = cam.pair;
    if (!pair || !pair.every((c) => run.some((k) => k.c === c)) || dist(byC(pair[0]), byC(pair[1])) > 90) {
      let best = 80; pair = null;
      for (const a of run) for (const b of run) if (a.c < b.c && dist(a, b) < best) { best = dist(a, b); pair = [a.c, b.c]; }
      cam.pair = pair;
    }
    if (pair) {
      const [a, b] = pair.map(byC);
      target = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, sx: Math.abs(a.x - b.x), sy: Math.abs(a.y - b.y) };
      label = `BATTLE · car ${pair[0] + 1} v car ${pair[1] + 1}`;
    } else { target = lead; label = `LEADER · car ${lead.c + 1}`; }
  }
  const zMax = r[2] / 100, zMin = r[2] / 380;   // ~100 m wide, out to ~380 m to fit a pair
  const fit = Math.min(zMax, Math.max(zMin, Math.min(r[2] / ((target.sx || 0) + 30), r[3] / ((target.sy || 0) + 30))));
  const z = cam.zoom || fit;
  cam.zoom = Math.min(fit, z + (fit - z) * 0.1);   // out at once (both cars stay in), back in gently
  panel(r);
  g.save(); g.beginPath(); g.rect(r[0], r[1], r[2], r[3]); g.clip();
  view(r, target.x, target.y, cam.zoom);
  drawCircuit(cam.zoom);
  for (const k of byOrder(v)) {
    if (v.gone[k.c]) continue;
    drawCar(k.x, k.y, k.h, v.out[k.c] || v.fin[k.c] ? mix(CARS[k.c], 0.6) : CARS[k.c], cam.zoom, 0);
  }
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  for (const k of v.list) {
    if (v.gone[k.c]) continue;
    const px = r[0] + r[2] / 2 + (k.x - target.x) * cam.zoom, py = r[1] + r[3] / 2 - (k.y - target.y) * cam.zoom;
    if (px > r[0] && px < r[0] + r[2] && py > r[1] && py < r[1] + r[3]) text(`${k.c + 1}`, px, py - 14, 14, CARS[k.c], "center");
  }
  g.restore();
  text(label, r[0] + 12, r[1] + 22, 15, C.text);
  const fc = cam.follow ?? (cam.pair ? null : lead.c);
  if (fc !== null) text(`${v.speed[fc]} km/h`, r[0] + r[2] - 12, r[1] + 22, 15, C.muted, "right");
}

function drawTower(now, v) {
  const [x, y, w] = lay.tower, rh = rowH(), hd = head(), N = src.N, L = src.L;
  panel(lay.tower);
  const lead = Math.max(...v.prog), fin = v.fin.some(Boolean);
  const lap = Math.min(Math.floor(lead / L) + 1, src.laps), tt = v.t;
  text(src.player ? "YOUR PIT WALL" : "RACE DAY", x + 14, y + hd * 0.5, Math.min(30, hd * 0.42), C.text, "left", 800);
  text(fin ? "FINISHED" : `LAP ${Math.max(lap, 1)}/${src.laps}`, x + w - 14, y + hd * 0.5, Math.min(24, hd * 0.34), fin ? C.amber : C.text, "right");
  const sp = src.live && !paused ? ` · ${measured.x < speed * 0.9 ? `${speed}x (this device: ${measured.x.toFixed(0)}x)` : `${speed}x`}` : ` · ${speed}x`;
  text(`${Math.floor(tt / 60)}:${String(Math.floor(tt % 60)).padStart(2, "0")} · circuit ${src.seed} · ${(L / 1000).toFixed(1)} km${sp}`,
       x + 14, y + hd * 0.85, Math.min(15, hd * 0.21), C.muted, "left", 600);
  const fs = Math.min(20, rh * 0.36), ss = Math.min(15, rh * 0.26);
  for (let c = 0; c < N; c++) {
    const r = v.order[c], ry = y + hd + r * rh, o = v.out[c], f = v.fin[c];
    if (cam.follow === c) { g.fillStyle = C.edge; g.fillRect(x + 4, ry + 2, w - 8, rh - 4); }
    text(`${r + 1}`, x + w * 0.05, ry + rh * 0.62, fs * 1.3, C.text, "center", 800);
    g.fillStyle = CARS[c]; g.beginPath(); g.roundRect(x + w * 0.1, ry + rh * 0.18, 6, rh * 0.64, 3); g.fill();
    text(`Car ${c + 1}`, x + w * 0.1 + 12, ry + rh * 0.45, fs, C.text);
    const lb = src.labels[c];
    text(lb, x + w * 0.1 + 12 + fs * 3.3, ry + rh * 0.45, ss, lb === "YOU" ? C.amber : lb === "AI" ? C.green : C.muted);
    const gap = lead - v.prog[c], pit = v.pit[c];
    const st = f ? "FINISHED" : v.wrecked[c] ? "WRECKED" : v.walled[c] ? "OUT" : v.dry[c] ? "NO FUEL"
      : now - flash[c].call < 2500 && !pit ? "BOX, BOX" : pit ? "IN PIT" : r === 0 ? "leader"
      : gap >= L ? `+${(gap / L).toFixed(1)} laps` : `+${Math.round(gap).toLocaleString("en")} m`;
    const sc = f ? C.green : o ? C.red : pit || st === "BOX, BOX" ? C.amber : C.muted;
    text(st, x + w * 0.1 + 12, ry + rh * 0.8, ss, sc, "left", 600);
    text(TYRES[v.comp[c]][0].toUpperCase(), x + w * 0.5, ry + rh * 0.6, fs * 1.2, TYRE[v.comp[c]], "center", 800);
    bar(x + w * 0.55, ry + rh * 0.3, w * 0.17, Math.max(5, rh * 0.13), 1 - v.wear[c], TYRE[v.comp[c]]);
    bar(x + w * 0.75, ry + rh * 0.3, w * 0.12, Math.max(5, rh * 0.13), v.fuel[c], v.fuel[c] > 0.1 ? C.blue : C.red);
    if (r === 0) {
      text("tyres", x + w * 0.55, ry + rh * 0.75, ss * 0.85, C.muted, "left", 600);
      text("fuel", x + w * 0.75, ry + rh * 0.75, ss * 0.85, C.muted, "left", 600);
      text("stops", x + w * 0.945, ry + rh * 0.9, ss * 0.85, C.muted, "center", 600);
    }
    text(`${v.stops[c]}`, x + w * 0.945, ry + rh * 0.6, fs * 1.2, C.text, "center", 800);
  }
}

// ── the end ──
function showEnd() {
  const v = V, N = src.N, rows = [...Array(N).keys()].sort((a, b) => v.order[a] - v.order[b]);
  const avg = (lb) => { const p = rows.map((c, i) => [c, i + 1]).filter(([c]) => src.labels[c] === lb); return p.reduce((s, [, q]) => s + q, 0) / p.length; };
  const st = (c) => (v.fin[c] ? "finished" : v.wrecked[c] ? "wrecked" : v.walled[c] ? "hit the wall" : v.dry[c] ? "out of fuel" : "running");
  const sets = (c) => (src.tyres ? src.tyres[c].map((k) => `<span style="color:${TYRE[k]}">${TYRES[k][0].toUpperCase()}</span>`).join(" → ") : "");
  const you = src.player ? rows.indexOf(0) + 1 : 0, ai = avg("AI"), rule = avg("RULE");
  const head = src.player ? (you < ai ? "You beat the AI pit wall" : you === Math.round(ai * 100) / 100 ? "Level with the AI pit wall" : "The AI pit wall wins this one")
    : src.live ? "Chequered flag" : "Chequered flag";
  endBox.innerHTML = `<div class="wrap"><h1>${head}</h1>
    ${src.player ? `<p>You called car 1 home <b>P${you}</b>${v.fin[0] ? "" : ` (${st(0)})`}. The AI pit wall's cars averaged <b>P${ai.toFixed(2)}</b>, the rule's <b>P${rule.toFixed(2)}</b>.</p>`
      : `<p>Pit wall on cars 1-4: <b>the AI</b> (it learned when to stop and which tyre to fit). Cars 5-8: <b>the rule</b> (box when the worst tyre is 60% worn, fit hards).</p>`}
    <table><tr><th>Pos</th><th>Car</th><th>Pit wall</th><th>Stops</th>${src.tyres ? "<th>Tyres</th>" : ""}<th></th></tr>
    ${rows.map((c, i) => `<tr><td>${i + 1}</td><td style="color:${CARS[c]};font-weight:700">Car ${c + 1}</td><td>${src.labels[c] === "YOU" ? "<b>you</b>" : src.labels[c] === "AI" ? "AI" : "rule"}</td><td>${v.stops[c]}</td>${src.tyres ? `<td>${sets(c)}</td>` : ""}<td>${st(c)}</td></tr>`).join("")}</table>
    ${src.player ? "" : `<p>This race: the AI's cars averaged <b>P${ai.toFixed(2)}</b>, the rule's <b>P${rule.toFixed(2)}</b>.</p>`}
    <p>One race is noisy (a crash ahead of you, a puncture). Over 128 races on circuits it never trained on, the AI pit wall's cars averaged <b>P${AI_AVG}</b> and the rule's <b>P${RULE_AVG}</b>.${src.player ? " Can you beat P4.08 on average?" : ""}</p>
    <p>Same AI driver in every car: ${src.driver}. AI pit wall: ${src.strategist}.${src.live ? " This race ran in your browser: the episode's simulator, ported to JavaScript." : ""}</p>
    ${src.live ? `<button class="main" id="again">${src.player ? "Race again" : "Another race"}</button>` : `<button id="again">Watch again</button>`}<button id="close">Back to the race</button></div>`;
  endBox.style.display = "block";
  document.getElementById("again").onclick = () => {
    endBox.style.display = "none";
    if (src.live) { showSetup(); return; }
    src.seek(0); ended = false; paused = false; btnPlay.textContent = "Pause";
  };
  document.getElementById("close").onclick = () => { endBox.style.display = "none"; };
}

// ── live: pick a circuit and your starting tyre ──
const setup = { circuit: 3, tyre: 1, videoCars: false };
async function showSetup() {
  fieldData ??= await (await fetch("field-data.json")).json();
  // (circuit 100001, 5.3 km: 25 laps don't fit the race's 60-minute cap, in Python as here: nobody finishes)
  const ok = fieldData.circuits.map((c, k) => k).filter((k) => fieldData.circuits[k].length * 25 < 120000);
  const circuits = fieldData.circuits, step = (d) => { setup.circuit = ok[(ok.indexOf(setup.circuit) + d + ok.length) % ok.length]; drawMini(); };
  paused = true;
  setupBox.innerHTML = `<div class="wrap"><h1>Be the pit wall</h1>
    <p>Episode 15's race, run live in your browser: 8 cars, 25 laps, tyre wear, a fuel limit, contact and damage.
    The same AI driver drives every car. <b>You</b> make car 1's strategy: its starting tyres, and once a lap, as it nears
    the pits, <b>box or stay out</b> and which tyres to fit. <b>The AI pit wall</b> runs cars 2-4, <b>the rule</b> (box at
    60% worn, fit hards) cars 5-8. Grid slots are drawn at random.</p>
    <p><canvas id="mini"></canvas></p>
    <p><button id="prev">◀</button> <b id="cname"></b> <button id="next">▶</button></p>
    <p><label>Car 1 starts on</label><br>${[0, 1, 2].map((k) => `<button data-t="${k}" style="color:${TYRE[k]}">${TYRES[k]}s</button>`).join("")}</p>
    <p><label>The cars</label><br><button data-v="0">new random cars</button><button data-v="1">the video's 8 cars</button></p>
    <p style="font-size:14px">Every car is a different design (power, wings, tyres...), so in one race some cars are just
    faster. With new random cars each race, it evens out over a few races.</p>
    <p><button class="main" id="go">Start the race</button><button id="watch">Just watch: AI pit wall v the rule</button></p>
    <p style="font-size:14px">On a slower device the race may not reach the top speeds: it runs every car's physics and
    AI 20 times a second of race time. A 25-lap race is about 45 minutes of race time.</p></div>`;
  setupBox.style.display = "block";
  const mini = document.getElementById("mini"), mg = mini.getContext("2d");
  const drawMini = () => {
    const c = circuits[setup.circuit], P = c.center.shape[0], xy = decodeF32(c.center);
    mini.width = 440; mini.height = 320; mg.clearRect(0, 0, 440, 320);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < P; i++) { x0 = Math.min(x0, xy[2 * i]); x1 = Math.max(x1, xy[2 * i]); y0 = Math.min(y0, xy[2 * i + 1]); y1 = Math.max(y1, xy[2 * i + 1]); }
    const s = Math.min(400 / (x1 - x0), 280 / (y1 - y0));
    mg.strokeStyle = C.text; mg.lineWidth = 4; mg.lineJoin = "round"; mg.beginPath();
    for (let i = 0; i <= P; i++) { const k = i % P, px = 220 + (xy[2 * k] - (x0 + x1) / 2) * s, py = 160 - (xy[2 * k + 1] - (y0 + y1) / 2) * s; i ? mg.lineTo(px, py) : mg.moveTo(px, py); }
    mg.stroke();
    document.getElementById("cname").textContent = `circuit ${c.seed} · ${(c.length / 1000).toFixed(1)} km${setup.circuit === 3 ? " (the video's)" : ""}`;
    for (const b of setupBox.querySelectorAll("button[data-t]")) b.classList.toggle("pick", +b.dataset.t === setup.tyre);
    for (const b of setupBox.querySelectorAll("button[data-v]")) b.classList.toggle("pick", !!+b.dataset.v === setup.videoCars);
  };
  drawMini();
  document.getElementById("prev").onclick = () => step(-1);
  document.getElementById("next").onclick = () => step(1);
  for (const b of setupBox.querySelectorAll("button[data-t]")) b.onclick = () => { setup.tyre = +b.dataset.t; drawMini(); };
  for (const b of setupBox.querySelectorAll("button[data-v]")) b.onclick = () => { setup.videoCars = !!+b.dataset.v; drawMini(); };
  document.getElementById("go").onclick = () => startLive(true);
  document.getElementById("watch").onclick = () => startLive(false);
}
const decodeF32 = (o) => { const b = Uint8Array.from(atob(o.b64), (c) => c.charCodeAt(0)); return new Float32Array(b.buffer); };

async function startLive(player) {
  setupBox.style.display = "none"; endBox.style.display = "none";
  await begin(await liveSource({ circuit: setup.circuit, player, startTyre: setup.tyre, videoCars: setup.videoCars }));
  wallBar.style.display = player ? "flex" : "none";
  wall.box = false; wall.compound = 2; wall.answered = false; wall.pending = false;
  speed = player ? 10 : 20; btnSpeed.textContent = `${speed}x`;
  wall.draw(); resize();
}

async function begin(s) {
  src = s; geo = buildCircuit(s); flash = Array.from({ length: s.N }, () => ({ call: -1e9, bump: -1e9 }));
  cam = { follow: null, pair: null, zoom: null }; ended = false; paused = false; btnPlay.textContent = "Pause";
  scrub.max = s.end; measured = { since: performance.now(), raceT: 0, x: speed };
  V = s.view();
}

// ── the loop ──
let prev = performance.now(), wallTick = 0;
function frame(now) {
  const real = Math.max(0, Math.min(0.1, (now - prev) / 1000)); prev = now;   // (the first frame can be stamped before the script ran)
  if (src) {
    const t0 = src.t;
    if (!paused && !ended) src.advance(real * speed);
    if (now - measured.since > 1000) {   // the speed this device really runs at
      measured.x = (src.t - measured.raceT) / ((now - measured.since) / 1000) || measured.x;
      measured.since = now; measured.raceT = src.t;
    } else if (src.t < t0) measured.raceT = src.t;
    V = src.view();
    for (const c of V.calls) flash[c].call = now;
    for (const c of V.bumps) flash[c].bump = now;
    if (!src.live && !paused) scrub.value = src.t;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
    drawMap(now, V);
    drawBattle(V);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawTower(now, V);
    if (src.live) {
      document.getElementById("gap").textContent = "";
      if (src.player && now - wallTick > 250 && !wall.pending) { wall.draw(); wallTick = now; }
    }
    const over = src.live ? src.done : src.t >= src.end;
    if (over && !ended) { ended = true; paused = true; btnPlay.textContent = src.live ? "New race" : "Replay"; showEnd(); }
  }
  requestAnimationFrame(frame);
}

resize();
if (LIVE) showSetup(); else await begin(await replaySource());
requestAnimationFrame(frame);
