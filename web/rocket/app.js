// Land it yourself (episode 18): you fly the booster on the left, the AI (the video's Try 29) flies the same start
// on the right, both in the real sim (engine.js, checked against Python by check.mjs). Two starts: from 70 km (the
// whole flight, as in the video) or "take over at 3 km" (the AI flies the entry and the fall, you land it).
import {Booster, Pilot, FLYING, LANDED, us76} from "./engine.js";
import {THEME as T, SCENE as S, mix, css} from "./theme.js";

const $ = (id) => document.getElementById(id);
const cfg = await (await fetch("data/config.json")).json();
const pilot = new Pilot((await (await fetch("data/pilot.json")).json()).weights);
const DT = cfg.dt, RAD = Math.PI / 180;

// ── input: steer (←/→), throttle (↑/↓), engine on/off (Space); the same on the touch buttons ──────────────────
const input = {left: false, right: false, up: false, down: false, finL: false, finR: false, want: false, throttle: 0.4,
               copilot: true};
const keyMap = {ArrowLeft: "left", ArrowRight: "right", ArrowUp: "up", ArrowDown: "down", KeyA: "finL", KeyD: "finR",
                KeyW: "up", KeyS: "down"};
addEventListener("keydown", (e) => {
  if (keyMap[e.code]) { input[keyMap[e.code]] = true; e.preventDefault(); }
  if (e.code === "Space") { input.want = !input.want; e.preventDefault(); }
  if (e.code === "Digit1") setWarp(1);
  if (e.code === "Digit2") setWarp(4);
  if (e.code === "Digit3") setWarp(10);
});
addEventListener("keyup", (e) => { if (keyMap[e.code]) input[keyMap[e.code]] = false; });
for (const btn of document.querySelectorAll("[data-hold]")) {
  const k = btn.dataset.hold;
  const on = (e) => { input[k] = true; btn.classList.add("on"); e.preventDefault(); };
  const off = () => { input[k] = false; btn.classList.remove("on"); };
  btn.addEventListener("pointerdown", on); btn.addEventListener("pointerup", off);
  btn.addEventListener("pointerleave", off); btn.addEventListener("pointercancel", off);
}
$("engine").addEventListener("click", () => { input.want = !input.want; });
$("copilot").addEventListener("change", (e) => { input.copilot = e.target.checked; $("finbtns").hidden = input.copilot; });

// the player's controls -> the sim's four: engine (switch, then throttle), engine angle (the small thrusters move
// with it), grid fins. Right = lean right. Measured (devlog 12w): the landing needs the engine angle and the fins
// moved separately (the AI with one shared steer: 0 of 100; engine angle + fins: 96), so the fins are their own
// control: by default an AI copilot flies them (the same network, on your booster's senses), or you do (A/D).
function playerAction(b) {
  const steer = (input.right ? 1 : 0) - (input.left ? 1 : 0);
  let a0;
  if (input.want) a0 = b.lit ? input.throttle : 1.0;      // held above +0.5 for the 0.5 s start sequence: lights
  else a0 = b.lit ? -1.0 : 0.0;                          // held below -0.5 for 0.5 s: shuts down
  const fins = input.copilot ? pilot.act(b.obs())[3] : (input.finL ? 1 : 0) - (input.finR ? 1 : 0);  // fins + = lean left
  return [a0, steer, steer, fins];
}

// ── a game ───────────────────────────────────────────────────────────────────────────────────────────────────
let game = null, warp = 1;
function setWarp(w) {
  warp = w;
  for (const b of document.querySelectorAll("[data-warp]")) b.classList.toggle("on", +b.dataset.warp === w);
}
for (const b of document.querySelectorAll("[data-warp]")) b.addEventListener("click", () => setWarp(+b.dataset.warp));

function newStart() {
  let s = (Math.random() * 2 ** 31) | 0;
  const rand = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
  return Booster.draw(cfg, rand);
}

function start(mode, startState) {
  const st = startState || newStart();
  const you = new Booster(cfg).setStart(st), ai = new Booster(cfg).setStart(st);
  let handover = null;
  if (mode === "takeover") {  // the AI flies until 3 km up; you take over from exactly where it was
    while (ai.outcome === FLYING && ai.y > 3000) ai.step(pilot.act(ai.obs()));
    Object.assign(you, structuredClone({...ai}));
    you.c = cfg;
    handover = ai.time;
    input.want = you.lit; input.throttle = 0.4;
  } else { input.want = false; input.throttle = 0.4; }
  game = {mode, start: st, you, ai, trails: [[], []], acc: 0, done: false, handover};
  setWarp(mode === "full" ? 4 : 1);
  $("intro").hidden = true; $("result").hidden = true;
}

// ── drawing (the episode's 2D view, simplified) ──────────────────────────────────────────────────────────────
function view(canvas, b, trail, label, colour) {
  const ctx = canvas.getContext("2d"), dpr = devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const span = Math.min(Math.max(110 + b.y * 0.03, 110), 260);  // metres across: close in near the deck
  const m = span / W;                                            // metres per pixel
  const cx = b.x, cy = Math.max(b.y + cfg.length * 0.45, H * 0.36 * m);
  const P = (x, y) => [W / 2 + (x - cx) / m, H / 2 - (y - cy) / m];
  // sky: space up high, blue near the sea
  const low = Math.min(Math.max(1 - b.y / 40000, 0), 1);
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, css(mix(S.space, S.skyLow, low * 0.6)));
  g.addColorStop(1, css(mix(S.space, S.skyLow, low)));
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
  if (b.y > 15000) {  // stars, fading as the air thickens
    ctx.fillStyle = `rgba(232,234,240,${Math.min((b.y - 15000) / 20000, 0.8)})`;
    for (let k = 0; k < 60; k++) ctx.fillRect((k * 97.3) % W, (k * 57.1) % H, 1.2, 1.2);
  }
  // sea and the drone ship's deck (52 m across; the legs must stay on it)
  const [, seaY] = P(0, 0);
  if (seaY < H + 5) {
    ctx.fillStyle = S.sea; ctx.fillRect(0, seaY, W, H - seaY);
    ctx.strokeStyle = S.seaLine; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(0, seaY); ctx.lineTo(W, seaY); ctx.stroke();
    const [dl, dt] = P(-26, 0), [dr, db] = P(26, -4);
    ctx.fillStyle = S.deck; ctx.fillRect(dl, dt, dr - dl, db - dt);
    ctx.fillStyle = S.deckEdge; ctx.fillRect(dl, dt - 2, dr - dl, 3);
  }
  const [deckX] = P(0, 0);
  if (deckX < 0 || deckX > W) {  // the ship is off screen: say where, and how far
    const right = deckX > W, d = Math.hypot(b.x, b.y);
    ctx.fillStyle = T.text; ctx.font = `700 15px ${T.font}`; ctx.textAlign = right ? "right" : "left";
    ctx.fillText(`${right ? "" : "← "}ship ${d >= 1000 ? (d / 1000).toFixed(1) + " km" : d.toFixed(0) + " m"}${right ? " →" : ""}`,
                 right ? W - 10 : 10, H * 0.5);
  }
  // trail
  ctx.fillStyle = S.trail;
  for (const [x, y] of trail) { const [px, py] = P(x, y); ctx.beginPath(); ctx.arc(px, py, 2.2, 0, 7); ctx.fill(); }
  if (b.outcome === FLYING || b.outcome === LANDED) booster(ctx, b, P, m); else boom(ctx, b, P, m);
  // label
  ctx.fillStyle = colour; ctx.font = `800 17px ${T.font}`; ctx.textAlign = "left"; ctx.fillText(label, 10, 22);
}

function booster(ctx, b, P, m) {
  const L = cfg.length, D = cfg.diameter, t = b.tilt;
  const up = [Math.sin(t), Math.cos(t)], side = [Math.cos(t), -Math.sin(t)];
  const at = (a, s) => P(b.x + up[0] * a + side[0] * s, b.y + up[1] * a + side[1] * s);
  const poly = (pts, fill) => { ctx.fillStyle = fill; ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(...p) : ctx.moveTo(...p)); ctx.fill(); };
  if (b.lit && b.thrust > 0) {  // flame, behind the body; it balloons in thin air
    const n = b.engines(), power = Math.min(Math.max(b.thrust / (845200 * n), 0.3), 1.2);
    const thin = Math.min(Math.max(b.y / 50000, 0), 1);
    const len = (11 + 14 * power) * (1 + 0.6 * (n > 1)) * (1 + 0.25 * Math.sin(performance.now() / 21));
    const spread = D * (n === 1 ? 0.32 : 0.5) * (1 + 2.5 * thin), gim = b.gimbal;
    const down = [-(up[0] * Math.cos(gim) + side[0] * Math.sin(gim)), -(up[1] * Math.cos(gim) + side[1] * Math.sin(gim))];
    for (const [k, col] of [[1, T.orange], [0.55, T.amber], [0.25, S.flameCore]]) {
      const tip = P(b.x + down[0] * len * k, b.y + down[1] * len * k);
      poly([at(0, spread * k), tip, at(0, -spread * k)], col);
    }
  }
  const open = Math.min(Math.max((400 - b.y) / 250, 0), 1);  // legs swing out in the last few hundred metres
  ctx.strokeStyle = S.legs; ctx.lineWidth = Math.max(2, 1.1 / m);
  for (const k of [-1, 1]) {
    const hinge = at(8, D / 2 * k), stowed = at(1, D / 2 * k), spread = at(-0.6, (D / 2 + 4.6) * k);
    const foot = [stowed[0] + (spread[0] - stowed[0]) * open, stowed[1] + (spread[1] - stowed[1]) * open];
    ctx.beginPath(); ctx.moveTo(...hinge); ctx.lineTo(...foot); ctx.stroke();
  }
  const body = (a0, a1, fill) => poly([at(a0, D / 2), at(a1, D / 2), at(a1, -D / 2), at(a0, -D / 2)], fill);
  body(0, L, S.body);
  const soot = Math.min(b.heat / 6e5, 1);
  if (soot > 0) body(0, 3 + 9 * soot, css(mix([236, 238, 242], [58, 54, 52], 0.7 * soot)));
  body(L - 4.5, L, S.interstage);
  for (const k of [-1, 1]) {  // grid fins, swung by `fin`
    const f = b.fin * k, base = [cfg.fin_height, k * D / 2];
    const p0 = at(base[0], base[1]), p1 = at(base[0] + 1.5 * Math.sin(f), base[1] + k * 1.6 * Math.cos(f));
    ctx.strokeStyle = S.interstage; ctx.lineWidth = Math.max(2, 1.4 / m); ctx.beginPath(); ctx.moveTo(...p0); ctx.lineTo(...p1); ctx.stroke();
  }
  const hot = Math.min(Math.max(us76(b.y)[0] * Math.hypot(b.vx, b.vy) ** 3 / 2e8, 0), 1);  // entry glow
  if (hot > 0.05) { ctx.fillStyle = `rgba(251,146,60,${0.25 + 0.5 * hot})`; ctx.beginPath(); ctx.arc(...at(-1, 0), (6 + 12 * hot) / (m / 0.2), 0, 7); ctx.fill(); }
}

function boom(ctx, b, P, m) {
  const [x, y] = P(b.x, Math.max(b.y, 0) + 8), r = 22 / (m / 0.2);
  for (const [k, col] of [[1, T.red], [0.7, T.orange], [0.4, T.amber]]) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, r * k, 0, 7); ctx.fill(); }
}

function hud(el, b) {
  const speed = Math.hypot(b.vx, b.vy), [rho] = us76(b.y);
  const q = 0.5 * rho * speed * speed;
  const alpha = Math.atan2(Math.abs(b.vx * Math.cos(b.tilt) - b.vy * Math.sin(b.tilt)),
                           Math.abs(b.vx * Math.sin(b.tilt) + b.vy * Math.cos(b.tilt))) / RAD;
  const bar = (v, lim, warn = true) =>  // heat, air load, bending go red near their limit; fuel doesn't
    `<i style="width:${Math.min(100, 100 * v / lim).toFixed(0)}%;background:${warn && v > 0.85 * lim ? T.red : ""}"></i>`;
  el.innerHTML = `
    <div class="grid"><span>height</span><b>${b.y >= 1000 ? (b.y / 1000).toFixed(1) + " km" : Math.max(b.y, 0).toFixed(0) + " m"}</b>
    <span>speed</span><b>${speed >= 10 ? speed.toFixed(0) : speed.toFixed(1)} m/s</b>
    <span>engines</span><b>${b.lit ? b.engines() + " lit" : "off"} · ${b.ignitions} light${b.ignitions === 1 ? "" : "s"} left</b>
    <span>off the airflow</span><b>${speed > 30 ? alpha.toFixed(0) + "°" : "–"}</b></div>
    <div class="bars"><span>fuel</span><div class="bar fuel">${bar(b.fuel, cfg.start_fuel[1], false)}</div>
    <span>heat</span><div class="bar heat">${bar(b.heat, cfg.heat_limit)}</div>
    <span>air load</span><div class="bar q">${bar(q, cfg.q_limit)}</div>
    <span>bending</span><div class="bar bend">${bar(b.bend, cfg.bend_limit)}</div></div>`;
}

function result(b) {
  if (b.outcome === LANDED)
    return `<b class="ok">Landed</b> at ${Math.abs(b.vy).toFixed(1)} m/s, ${Math.abs(b.x).toFixed(1)} m off centre,
            leg wear ${(100 * b.legWear()).toFixed(0)}% <span class="muted">(${b.time.toFixed(1)} s)</span>`;
  return `<b class="bad">${b.outcome === 3 ? "Timed out" : "Crashed"}</b>: ${b.why} <span class="muted">(${b.time.toFixed(1)} s)</span>`;
}

// ── the loop ─────────────────────────────────────────────────────────────────────────────────────────────────
let last = performance.now();
function frame(now) {
  const real = Math.min((now - last) / 1000, 0.1); last = now;
  if (game && !game.done) {
    if (input.up) input.throttle = Math.min(1, input.throttle + real * 0.8);
    if (input.down) input.throttle = Math.max(0, input.throttle - real * 0.8);
    game.acc += real * warp;
    for (let n = 0; game.acc >= DT && n < 400; n++, game.acc -= DT) {
      const {you, ai} = game;
      if (you.outcome === FLYING) you.step(playerAction(you));
      if (ai.outcome === FLYING) ai.step(pilot.act(ai.obs()));
      for (const [k, b] of [you, ai].entries())
        if (b.outcome === FLYING && Math.round(b.time / DT) % 10 === 0) game.trails[k].push([b.x, b.y]);
    }
    if (game.you.outcome !== FLYING && game.ai.outcome !== FLYING) {
      game.done = true;
      $("res-you").innerHTML = result(game.you); $("res-ai").innerHTML = result(game.ai);
      $("res-title").textContent = game.you.outcome === LANDED
        ? (game.ai.outcome === LANDED && game.ai.legWear() < game.you.legWear() ? "You landed. The AI landed softer." : "You landed it!")
        : "The sea wins this one.";
      $("result").hidden = false;
    }
  }
  if (game) {
    view($("you"), game.you, game.trails[0], "YOU", T.green);
    view($("ai"), game.ai, game.trails[1], "AI (TRY 29)", T.blue);
    hud($("hud-you"), game.you); hud($("hud-ai"), game.ai);
    $("engine").classList.toggle("on", input.want);
    $("engine").textContent = input.want ? (game.you.lit ? "Engine: ON" : "Engine: starting…") : (game.you.lit ? "Engine: stopping…" : "Engine: OFF");
    $("thr").textContent = `${(57 + 43 * input.throttle).toFixed(0)}%`;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

$("go-full").addEventListener("click", () => start("full"));
$("go-takeover").addEventListener("click", () => start("takeover"));
$("again").addEventListener("click", () => start(game.mode, game.start));
$("new").addEventListener("click", () => start(game.mode));
$("menu").addEventListener("click", () => { $("result").hidden = true; $("intro").hidden = false; });
window.__game = () => game;  // for the headless check
