// Dogfight my AI (episode 21): you fly the blue F-16, the AI flies the red one. The sim (sim.js) and the pilot are
// the episode's, checked against the Python (check.mjs); the look (look.js) is the episode's replay page.
//   ?seed=N   a fixed start (otherwise a new one each fight)
import { THREE, v3, renderer, scene, camera, puffs, jets, smoke, vapour, sPts, sHalf, sAlpha, vPts, vHalf, vAlpha,
  fillRibbon, burners, tracers, spawn, explode, stepFx, trailN, vapN } from "./look.js";
import { theme } from "./theme.js";
import { ALIVE, SHOT, CRASHED, Fighters, Pilot, G, add, mul, sub, norm, unit, dot } from "./sim.js";

const q = new URLSearchParams(location.search);
const cfg = await (await fetch("data/config.json")).json();
const pilot = new Pilot(await (await fetch("data/pilot.json")).json());
const sim = new Fighters(cfg);
const DT = cfg.dt;

// ── controls: the real jet's fly-by-wire stick (a g command and a roll rate), throttle, trigger ────────────────
const keys = new Set();
addEventListener("keydown", (e) => {
  // (owner, 11-20) Ctrl was the throttle, and Ctrl+S with the pull key opened the browser's "Save page" over and over:
  // no modifier keys in the controls, and while flying none of their combinations reach the browser
  if (running && (e.ctrlKey || e.metaKey || e.altKey)) { e.preventDefault(); return; }
  keys.add(e.code);
  if (["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.code)) e.preventDefault();
  if (e.code === "Enter" && !running) start();
  if (e.code === "KeyP" && running) paused = !paused;
});
addEventListener("keyup", (e) => keys.delete(e.code));
addEventListener("blur", () => keys.clear());
const held = (...c) => c.some((k) => keys.has(k));
let pull = 0, push = 0, throttle = 0.5;   // throttle: the sim's action (-1 idle .. 0.5 full military .. 1 afterburner)
function playerAction(sec) {
  const p = held("KeyS", "ArrowDown"), u = held("KeyW", "ArrowUp");
  pull = p ? Math.min(1, pull + sec * 1.2) : Math.max(0, pull - sec * 4);
  push = u ? Math.min(1, push + sec * 1.5) : Math.max(0, push - sec * 4);
  if (held("KeyE", "KeyR")) throttle = Math.min(1, throttle + sec * 0.8);
  if (held("KeyQ", "KeyF")) throttle = Math.max(-1, throttle - sec * 0.8);
  const roll = (held("KeyD", "ArrowRight") ? 1 : 0) - (held("KeyA", "ArrowLeft") ? 1 : 0);
  const g = push > 0 ? -push : 1 / cfg.g_max + pull * (1 - 1 / cfg.g_max);   // hands off: 1 g
  return [g, roll, throttle, held("Space") ? 1 : -1];
}

// ── a fight ───────────────────────────────────────────────────────────────
let seed = parseInt(q.get("seed") || "0", 10) || 0;
const rnd = () => (seed ? ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646 : Math.random());
let st, prev, acc = 0, running = false, paused = false, dead = [false, false], hist = [[], []], over = 0;
const tally = JSON.parse(localStorage.getItem("dogfight-tally") || '{"won":0,"lost":0,"draw":0}');
function start() {
  st = sim.init(rnd);
  prev = st;
  acc = 0;
  over = 0;
  dead = [false, false];
  hist = [[], []];
  pull = push = 0;
  throttle = 0.5;
  jets.forEach((j) => (j.visible = true));
  burners.forEach((b) => (b.visible = true));
  running = true;
  paused = false;
  overlay.style.display = "none";
  audio.start();
}
const overlay = document.getElementById("overlay");
document.getElementById("go").onclick = () => start();

function finish() {
  const c = st.cause;
  let head, sub_;
  if (c[1] !== ALIVE && c[0] === ALIVE) { tally.won++; head = c[1] === SHOT ? "You shot it down" : "It flew into the ground"; sub_ = "You win."; }
  else if (c[0] !== ALIVE && c[1] === ALIVE) { tally.lost++; head = c[0] === SHOT ? "It shot you down" : "You flew into the ground"; sub_ = "The AI wins."; }
  else if (c[0] !== ALIVE) { tally.draw++; head = "Both down"; sub_ = "A draw."; }
  else { tally.draw++; head = "Time's up (2 minutes)"; sub_ = "A draw."; }
  localStorage.setItem("dogfight-tally", JSON.stringify(tally));
  overlay.querySelector(".box").innerHTML = `<h2>${head}</h2><div>${sub_}</div>
    <div class="small">You ${tally.won} · the AI ${tally.lost} · draws ${tally.draw}<br>
    In the episode, this AI beat its own previous version 57% to 14%.</div><div class="go" id="go">Fly again (Enter)</div>`;
  document.getElementById("go").onclick = () => start();
  overlay.style.display = "flex";
}

// ── sound: the engine (yours, by throttle), the wind (by speed), guns, hits, the explosion ─────────────────
const audio = (() => {
  let ctx = null, buf = {}, loops = {};
  const names = ["jet", "wind", "gun", "hit", "boom"];
  async function load() {
    ctx = new AudioContext();
    await Promise.all(names.map(async (n) => (buf[n] = await ctx.decodeAudioData(await (await fetch(`sfx/${n}.mp3`)).arrayBuffer()))));
    for (const n of ["jet", "wind", "gun"]) {
      const s = ctx.createBufferSource(), g = ctx.createGain();
      s.buffer = buf[n]; s.loop = true; g.gain.value = 0; s.connect(g).connect(ctx.destination); s.start();
      loops[n] = g;
    }
  }
  return {
    start() { if (!ctx) load().catch(() => {}); else ctx.resume(); },
    set(n, v) { if (loops[n]) loops[n].gain.setTargetAtTime(v, ctx.currentTime, 0.05); },
    once(n, v) {
      if (!ctx || !buf[n]) return;
      const s = ctx.createBufferSource(), g = ctx.createGain();
      s.buffer = buf[n]; g.gain.value = v; s.connect(g).connect(ctx.destination); s.start();
    },
    quiet() { for (const n in loops) this.set(n, 0); },
  };
})();

// ── one sim step: you and the AI decide, the sim moves, the effects ────────────────────────────────────────
function simStep(sec) {
  const mine = playerAction(sec);
  const ai = pilot.act(sim.observe(st, 1));
  prev = st;
  const r = sim.step(st, [mine, ai]);
  st = r.st;
  for (const i of [0, 1]) {
    hist[i].unshift(v3(st.pos[i]));
    if (hist[i].length > trailN) hist[i].pop();
    if (st.health[i] < prev.health[i] - 1e-9 && st.cause[i] === ALIVE) {
      for (let n = 0; n < 6; n++) spawn("spark", v3(st.pos[i]), new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(60), 0.35, 4, 10, theme.spark);
      audio.once("hit", i === 0 ? 0.9 : 0.5);
    }
    if (st.cause[i] !== ALIVE && !dead[i]) {
      dead[i] = true;
      explode(v3(st.pos[i]), v3(st.vel[i]), st.cause[i] === CRASHED);
      jets[i].visible = false;
      burners[i].visible = false;
      audio.once("boom", 1.0);
    }
  }
  if (r.done) running = false;
}

// ── drawing ─────────────────────────────────────────────────────────────
const hud = [document.getElementById("hud0"), document.getElementById("hud1")], bottom = document.getElementById("bottom");
const mk = (css) => { const e = document.createElement("div"); e.className = "mark"; e.style.cssText += css; document.body.appendChild(e); return e; };
const redCol = `#${theme.jets[1].toString(16).padStart(6, "0")}`, blueCol = `#${theme.jets[0].toString(16).padStart(6, "0")}`;
const diamond = mk(`width:2.4vh;height:2.4vh;margin:-1.2vh 0 0 -1.2vh;border:0.3vh solid ${redCol};transform:rotate(45deg)`);
const arrow = mk(`width:0;height:0;margin:-1.6vh 0 0 -1.6vh;border:1.6vh solid transparent;border-left:2.6vh solid ${redCol}`);
const lead = mk(`width:4vh;height:4vh;margin:-2vh 0 0 -2vh;border:0.35vh solid #ffd27a;border-radius:50%`);
const cross_ = mk(`width:1.6vh;height:1.6vh;margin:-0.8vh 0 0 -0.8vh;border-left:0.25vh solid #f2f4f8;border-top:0.25vh solid #f2f4f8;transform:rotate(45deg)`);
for (const t of tracers) { t.geometry.attributes.position.array.fill(-1e5); t.geometry.attributes.position.needsUpdate = true; }   // 64 slots, 40 used
const proj = new THREE.Vector3(), camPos = new THREE.Vector3(), camLook = new THREE.Vector3(), camUp = new THREE.Vector3(0, 1, 0);
let camReady = false;
const basis = new THREE.Matrix4();
const lerpV = (a, b, f) => v3(a).lerp(v3(b), f);
function place(el, worldPos, onScreenOnly = true) {
  proj.copy(worldPos).project(camera);
  const on = proj.z < 1 && Math.abs(proj.x) < 1 && Math.abs(proj.y) < 1;
  el.style.display = on || !onScreenOnly ? "block" : "none";
  if (on) { el.style.left = `${(proj.x + 1) * 50}vw`; el.style.top = `${(1 - proj.y) * 50}vh`; }
  return on;
}

function draw(sec) {
  if (running && !paused) {
    acc += sec;
    while (acc >= DT && running) { simStep(DT); acc -= DT; }
  } else if (!running && st) over += sec;
  if (!st) { renderer.render(scene, camera); return; }
  const f = running ? acc / DT : 1;
  const [noses] = sim.nose(st), [noses0] = sim.nose(prev);
  const P = [0, 1].map((i) => {
    const pos = lerpV(prev.pos[i], st.pos[i], f);
    if (!running && !dead[i]) pos.add(v3(st.vel[i]).multiplyScalar(over));   // after the end, the survivor flies on
    const nose = lerpV(noses0[i], noses[i], f).normalize();
    let up = lerpV(prev.up[i], st.up[i], f);
    up.sub(nose.clone().multiplyScalar(up.dot(nose))).normalize();
    return { pos, nose, up };
  });
  P.forEach((p, i) => {
    const right = new THREE.Vector3().crossVectors(p.nose, p.up);
    basis.makeBasis(p.nose, p.up, right);
    jets[i].position.copy(p.pos);
    jets[i].quaternion.setFromRotationMatrix(basis);
    // smoke: the last trailSeconds of positions, wider and fainter with age
    for (let j = 0; j < trailN; j++) {
      const h = hist[i][Math.min(j, hist[i].length - 1)] || p.pos, age = j / (trailN - 1);
      if (j === 0) sPts[0].copy(p.pos); else sPts[j].copy(h);
      sHalf[j] = 0.5 * (theme.smokeWidth[0] + (theme.smokeWidth[1] - theme.smokeWidth[0]) * Math.sqrt(age));
      sAlpha[j] = j < hist[i].length ? theme.smokeAlpha * (1 - age) * (1 - age) * Math.min(1, j / 3)
        * THREE.MathUtils.smoothstep(sPts[j].distanceTo(camera.position), theme.smokeNear[0], theme.smokeNear[1]) : 0;
    }
    fillRibbon(smoke[i], sPts, sHalf, sAlpha);
    // wingtip vapour while pulling hard
    const gk = st.load[i];
    for (const side of [1, -1]) {
      for (let j = 0; j < vapN; j++) {
        const h = hist[i][Math.min(j, hist[i].length - 1)] || p.pos;
        vPts[j].copy(j === 0 ? p.pos : h).addScaledVector(right.clone().normalize(), side * 4.62 * theme.jetScale);
        vAlpha[j] = dead[i] || j >= hist[i].length ? 0 : theme.vapourAlpha * THREE.MathUtils.smoothstep(gk, theme.vapourG[0], theme.vapourG[1]) * (1 - j / vapN);
      }
      fillRibbon(vapour[i][side > 0 ? 0 : 1], vPts, vHalf, vAlpha);
    }
    burners[i].position.copy(p.pos).addScaledVector(p.nose, -8.2 * theme.jetScale);
    burners[i].scale.setScalar((1.2 + 2.0 * Math.max(0, st.power[i] - 0.5)) * theme.jetScale);
    // tracers in flight
    const ta = tracers[i].geometry.attributes.position;
    for (let k = 0; k < Math.min(64, cfg.n_packets); k++) {
      const s = st.ppos[i][k];
      ta.array.set(st.prounds[i][k] > 0 && running ? [s[0], s[2], -s[1]] : [0, -1e5, 0], 3 * k);
    }
    ta.needsUpdate = true;
  });
  stepFx(sec);

  // the camera: behind and above your jet, banking with it
  const me = P[0];
  const want = me.pos.clone().addScaledVector(me.nose, -150).addScaledVector(me.up, 34);
  const look = me.pos.clone().addScaledVector(me.nose, 260);
  const a = camReady ? 1 - Math.exp(-sec * 8) : 1;
  camPos.lerp(want, a); camLook.lerp(look, a); camUp.lerp(me.up, a).normalize();
  camReady = true;
  camera.position.copy(camPos);
  camera.up.copy(camUp);
  camera.lookAt(camLook);
  camera.updateMatrixWorld();
  for (const s of puffs) s.material.opacity = 0.85 * THREE.MathUtils.smoothstep(s.position.distanceTo(camera.position), s.scale.x * 0.6, s.scale.x * 1.6);

  // the red jet's marker, or an arrow at the edge pointing to it
  const red = P[1].pos, onScreen = !dead[1] && place(diamond, red);
  if (!dead[1] && !onScreen) {
    const rel = red.clone().sub(camera.position).applyQuaternion(camera.quaternion.clone().invert());
    const ang = Math.atan2(-rel.y, rel.x);
    arrow.style.display = "block";
    arrow.style.left = `${50 + 44 * Math.cos(ang)}vw`; arrow.style.top = `${50 + 42 * Math.sin(ang)}vh`;
    arrow.style.transform = `rotate(${ang}rad)`;
  } else arrow.style.display = "none";
  // the gunsight: the nose cross, and the lead circle: where your shells will be when they reach the red jet's range
  place(cross_, me.pos.clone().addScaledVector(me.nose, 600));
  if (running && !dead[1]) {
    const fwd = noses[0], v = st.vel[0], d = sub(st.pos[1], st.pos[0]);
    const bv = add(v, mul(fwd, cfg.muzzle_speed));
    const tof = Math.min(Math.max(dot(d, sub(bv, st.vel[1])) / Math.max(dot(sub(bv, st.vel[1]), sub(bv, st.vel[1])), 1), 0), cfg.bullet_life);
    const at = add(add(st.pos[0], mul(bv, tof)), [0, 0, -0.5 * G * tof * tof]);
    if (tof > 0.05 && tof < cfg.bullet_life && norm(d) < 2500) place(lead, v3(at)); else lead.style.display = "none";   // only when it's ahead, in range
  } else lead.style.display = "none";

  // HUD
  const names = ["BLUE · you", "RED · the AI"], status = ["", "shot down", "crashed"];
  [0, 1].forEach((i) => {
    const hp = Math.max(0, st.health[i]), col = i ? redCol : blueCol;
    hud[i].innerHTML = `${names[i]}${st.cause[i] ? `<br>${status[st.cause[i]]}` : ""}<br>${Math.round(norm(st.vel[i]) * 1.944)} kt · ` +
      `${st.load[i].toFixed(1)} g · ${Math.round(st.pos[i][2]).toLocaleString()} m<br><span class="bar"><i style="width:${100 * hp}%;background:${col}"></i></span>`;
  });
  const thr = st.power[0] > 0.75 + 1e-3 ? "AFTERBURNER" : `${Math.round((st.power[0] / 0.75) * 100)}%`;
  bottom.textContent = `${Math.round(norm(sub(st.pos[1], st.pos[0]))).toLocaleString()} m apart · throttle ${thr} · ` +
    `${Math.round(st.ammo[0])} rounds · ${Math.max(0, cfg.time_limit - st.time).toFixed(0)} s` + (paused ? " · PAUSED (P)" : "");

  // sound
  if (running && !paused) {
    audio.set("jet", 0.25 + 0.35 * st.power[0]);
    audio.set("wind", 0.25 * Math.min(1, norm(st.vel[0]) / 400));
    const mineFiring = held("Space") && st.ammo[0] > 0, aiFiring = st.prounds[1][(st.n - 1 + cfg.n_packets) % cfg.n_packets] > 0;
    const dAi = norm(sub(st.pos[1], st.pos[0]));
    audio.set("gun", mineFiring ? 0.7 : aiFiring ? 0.6 / (1 + (dAi / 400) ** 2) : 0);
  } else audio.quiet();
  renderer.render(scene, camera);
  if (!running && overlay.style.display === "none" && over > 2.5) finish();
}

let last = null;
const loop = (now) => {
  draw(Math.min(Math.max(last === null ? 0 : (now - last) / 1000, 0), 0.1));
  last = now;
  requestAnimationFrame(loop);
};
requestAnimationFrame(loop);
// a fight to look at behind the start screen: the AI against itself, frozen
st = prev = sim.init(rnd);
globalThis.dfPlay = { get state() { return st; }, start, keys };
