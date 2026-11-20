// The dogfight's look (ep21), shared by the game: the same scene as the episode's replay page (web/dogfight/app.js):
// sky and sun, flat farmland (the sim's ground is flat), clouds that fade near the camera, F-16-shaped jets drawn
// jetScale x real size, smoke ribbons, wingtip vapour, engine glow, tracers, sparks, fireballs and debris.
// Sim axes are x, y, z-up (metres); three.js is y-up, so (x, y, z) -> (x, z, -y).
import * as THREE from "three";
import { theme } from "./theme.js";

const dt = 0.05;   // the sim's step: the trails keep one point per step
const v3 = (a) => new THREE.Vector3(a[0], a[2], -a[1]);

// ── scene ────────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById("c"), antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(theme.fog, 8000, 45000);
const camera = new THREE.PerspectiveCamera(55, 1, 2, 80000);
const resize = () => {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
};
addEventListener("resize", resize);
resize();

const sunDir = new THREE.Vector3(...theme.sunDir).normalize();
const fogC = new THREE.Color(theme.fog);
// a small seeded random, so every capture of a replay gets the same clouds
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;

// sky: a big sphere with a vertical gradient and a sun
{
  const g = new THREE.SphereGeometry(70000, 32, 16);
  const m = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { top: { value: new THREE.Color(theme.skyTop) }, bottom: { value: new THREE.Color(theme.skyBottom) },
      sunC: { value: new THREE.Color(theme.sun) }, sunDir: { value: sunDir } },
    vertexShader: "varying vec3 d; void main(){ d = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
    fragmentShader: `uniform vec3 top; uniform vec3 bottom; uniform vec3 sunC; uniform vec3 sunDir; varying vec3 d;
      void main(){ vec3 c = mix(bottom, top, smoothstep(0.0, 0.5, d.y));
        float s = max(dot(normalize(d), sunDir), 0.0);
        c += sunC * (pow(s, 900.0) * 1.5 + pow(s, 12.0) * 0.25);
        gl_FragColor = vec4(c, 1.0); }`,
  });
  scene.add(new THREE.Mesh(g, m));
}
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x4a5040, 1.6));
const sun = new THREE.DirectionalLight(theme.sun, 2.0);
sun.position.copy(sunDir).multiplyScalar(10000);
scene.add(sun);

// the ground: flat at 0 m like the sim's. Farm fields about fieldSize across (they replace the old 1 km grid as the
// sense of speed and height), hedges between them, darker woods; detail fades with distance so it doesn't shimmer.
{
  const m = new THREE.ShaderMaterial({
    uniforms: {
      fields: { value: theme.fields.map((c) => new THREE.Color(c)) }, woods: { value: new THREE.Color(theme.woods) },
      size: { value: theme.fieldSize }, fogC: { value: fogC }, fogNear: { value: 8000 }, fogFar: { value: 45000 },
    },
    vertexShader: "varying vec3 w; void main(){ vec4 p = modelMatrix * vec4(position,1.0); w = p.xyz; gl_Position = projectionMatrix * viewMatrix * p; }",
    fragmentShader: `uniform vec3 fields[6]; uniform vec3 woods; uniform float size; uniform vec3 fogC; uniform float fogNear; uniform float fogFar;
      varying vec3 w;
      float h(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      float n(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
        return mix(mix(h(i), h(i+vec2(1,0)), f.x), mix(h(i+vec2(0,1)), h(i+vec2(1,1)), f.x), f.y); }
      float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int k = 0; k < 4; k++){ s += a * n(p); p *= 2.03; a *= 0.5; } return s; }
      void main(){
        vec2 p = w.xz;
        float dist = distance(cameraPosition, w);
        float detail = 1.0 - smoothstep(3000.0, 20000.0, dist);
        vec2 r = mat2(0.94, 0.34, -0.34, 0.94) * p;                                   // fields not lined up with the view
        vec2 q = r / size + (vec2(fbm(p / 2500.0), fbm(p / 2500.0 + 17.0)) - 0.5) * 1.4;  // bend the field edges
        vec2 cell = floor(q), f = fract(q);
        if (h(cell + 3.1) > 0.55) { float k = floor(f.x * 3.0); cell += vec2(k * 0.31, 0.0); f.x = fract(f.x * 3.0); }  // strips
        int idx = int(h(cell) * 5.999);
        vec3 c = fields[0];
        for (int k = 1; k < 6; k++) if (k == idx) c = fields[k];
        c *= 0.92 + 0.16 * n(p / 40.0) * detail;                                     // texture inside a field
        float edge = min(min(f.x, 1.0 - f.x), min(f.y, 1.0 - f.y));
        c = mix(c * 0.72, c, mix(0.6, 1.0, smoothstep(0.0, 0.025, edge)) * detail + (1.0 - detail));   // hedges
        float wood = smoothstep(0.60, 0.66, fbm(p / 1800.0 + 5.0));
        c = mix(c, woods * (0.85 + 0.3 * n(p / 25.0)), wood);
        c = mix(c, fogC, smoothstep(fogNear, fogFar, dist));
        gl_FragColor = vec4(c, 1.0); }`,
  });
  const ground = new THREE.Mesh(new THREE.PlaneGeometry(200000, 200000), m);
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);
}

const puffs = [];
// clouds: fair-weather cumulus at about cloudBase, each a cluster of soft billboard puffs
{
  const cv = document.createElement("canvas");
  cv.width = cv.height = 128;
  const cx = cv.getContext("2d");
  const grad = cx.createRadialGradient(64, 64, 4, 64, 64, 64);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.55, "rgba(255,255,255,0.55)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  cx.fillStyle = grad;
  cx.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(cv);
  // each puff gets its own material so it can fade out when the camera flies close (a puff filling the frame hid
  // the fight: 6f)
  const lit = new THREE.SpriteMaterial({ map: tex, color: theme.cloud, transparent: true, opacity: 0.85, depthWrite: false });
  const shade = new THREE.SpriteMaterial({ map: tex, color: theme.cloudShade, transparent: true, opacity: 0.85, depthWrite: false });
  for (let c = 0; c < theme.cloudCount; c++) {
    const x = (rand() - 0.5) * 2 * theme.cloudSpread, z = (rand() - 0.5) * 2 * theme.cloudSpread;
    const y = theme.cloudBase + rand() * 400, w = 500 + rand() * 900;
    for (let k = 0; k < 7; k++) {
      const top = k > 2;
      const s = new THREE.Sprite((top ? lit : shade).clone());
      const r = w * (0.5 + rand() * 0.5);
      s.position.set(x + (rand() - 0.5) * w * 1.4, y + (top ? w * 0.25 + rand() * w * 0.3 : rand() * w * 0.1), z + (rand() - 0.5) * w * 1.4);
      s.scale.set(r * 1.6, r, 1);
      scene.add(s);
      puffs.push(s);
    }
  }
}

// ── the jets: an F-16 shape at real size (15.0 m long, 9.45 m span, 4.9 m high), nose +x, canopy +y, wings along z.
//    Grey airframe; the team colour on the fin and wingtips. Drawn jetScale x real size.
function jetMesh(color) {
  const grey = new THREE.MeshStandardMaterial({ color: theme.airframe, roughness: 0.6, metalness: 0.25, side: THREE.DoubleSide });
  const team = new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.2, side: THREE.DoubleSide });
  const glass = new THREE.MeshStandardMaterial({ color: theme.canopy, roughness: 0.1, metalness: 0.6 });
  const dark = new THREE.MeshStandardMaterial({ color: theme.nozzle, roughness: 0.8, metalness: 0.4, side: THREE.DoubleSide });
  const g = new THREE.Group();
  // fuselage: a lathe along x (radius, x), slightly flattened
  const prof = [[0, 7.6], [0.3, 6.9], [0.55, 5.8], [0.72, 4.2], [0.8, 2.5], [0.82, 0], [0.78, -3.5], [0.66, -6.2], [0.55, -7.4]]
    .map(([r, x]) => new THREE.Vector2(r, x));
  const body = new THREE.Mesh(new THREE.LatheGeometry(prof, 16), grey);
  body.geometry.rotateZ(-Math.PI / 2);
  body.scale.set(1, 0.9, 1.05);
  g.add(body);
  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.42, 0.9, 14, 1, true), dark);
  nozzle.rotation.z = Math.PI / 2;
  nozzle.position.x = -7.75;
  g.add(nozzle);
  const canopy = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 10), glass);
  canopy.scale.set(1.9, 0.5, 0.45);
  canopy.position.set(4.3, 0.6, 0);
  g.add(canopy);
  const intake = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.75, 1.1), grey);   // the chin intake
  intake.position.set(1.6, -0.75, 0);
  g.add(intake);
  const mouth = new THREE.Mesh(new THREE.PlaneGeometry(0.75, 1.1), dark);
  mouth.rotation.y = Math.PI / 2;
  mouth.position.set(3.31, -0.75, 0);
  g.add(mouth);
  // flat parts from (x, z) outlines, both sides; y is the height they sit at
  const flat = (pts, y, mat) => {
    const all = [...pts, ...pts.map(([x, z]) => [x, -z])];
    const tris = [];
    for (const side of [1, -1]) {
      const P = pts.map(([x, z]) => [x, y, side * z]);
      for (let k = 1; k < P.length - 1; k++) tris.push(P[0], P[k], P[k + 1]);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(tris.flat(), 3));
    geo.computeVertexNormals();
    g.add(new THREE.Mesh(geo, mat));
    return all;
  };
  flat([[2.2, 0.6], [-2.6, 4.55], [-3.7, 4.55], [-4.3, 0.6]], 0, grey);       // wings, 40° leading edge
  flat([[5.4, 0.55], [2.2, 1.15], [2.2, 0.55]], 0.05, grey);                 // the strakes (LERX) along the nose
  flat([[-5.3, 0.6], [-6.9, 2.75], [-7.5, 2.75], [-7.5, 0.6]], -0.1, grey);  // tailplanes
  for (const side of [1, -1]) {                                              // wingtip rails in the team colour
    const rail = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.16, 0.16), team);
    rail.position.set(-2.9, 0, side * 4.62);
    g.add(rail);
  }
  const fin = new THREE.BufferGeometry();                                    // the fin, in the team colour
  fin.setAttribute("position", new THREE.Float32BufferAttribute([
    [-3.6, 0.6, 0], [-6.5, 4.1, 0], [-7.3, 4.1, 0], [-3.6, 0.6, 0], [-7.3, 4.1, 0], [-7.5, 0.6, 0]].flat(), 3));
  fin.computeVertexNormals();
  g.add(new THREE.Mesh(fin, team));
  g.scale.setScalar(theme.jetScale);
  scene.add(g);
  return g;
}
const jets = [0, 1].map((i) => jetMesh(theme.jets[i]));

// trails: camera-facing ribbons; smoke widens and fades with age, wingtip vapour only shows in hard pulls
const trailN = Math.round(theme.trailSeconds / dt), vapN = Math.round(theme.vapourSeconds / dt);
function ribbon(n, color) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 2 * 3), 3));
  geo.setAttribute("alpha", new THREE.BufferAttribute(new Float32Array(n * 2), 1));
  geo.setAttribute("edge", new THREE.BufferAttribute(new Float32Array(n * 2).map((_, v) => (v % 2 ? -1 : 1)), 1));
  const idx = [];
  for (let k = 0; k < n - 1; k++) idx.push(2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 1, 2 * k + 3, 2 * k + 2);
  geo.setIndex(idx);
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    uniforms: { color: { value: new THREE.Color(color) } },
    vertexShader: "attribute float alpha; attribute float edge; varying float a; varying float e; void main(){ a = alpha; e = edge; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
    fragmentShader: "uniform vec3 color; varying float a; varying float e; void main(){ gl_FragColor = vec4(color, a * (1.0 - e * e)); }",   // soft edges
  });
  const m = new THREE.Mesh(geo, mat);
  m.frustumCulled = false;
  scene.add(m);
  return { m, n, pos: geo.attributes.position, alpha: geo.attributes.alpha };
}
// fill a ribbon from n points (THREE.Vector3), half-widths and alphas, facing the camera
const _t = new THREE.Vector3(), _c = new THREE.Vector3(), _s = new THREE.Vector3();
function fillRibbon(r, pts, half, alpha) {
  for (let k = 0; k < r.n; k++) {
    const a = pts[Math.max(0, k - 1)], b = pts[Math.min(r.n - 1, k + 1)];
    _t.subVectors(a, b);
    if (_t.lengthSq() < 1e-6) _t.set(1, 0, 0);
    _c.subVectors(camera.position, pts[k]);
    _s.crossVectors(_t, _c).normalize().multiplyScalar(half[k]);
    r.pos.setXYZ(2 * k, pts[k].x + _s.x, pts[k].y + _s.y, pts[k].z + _s.z);
    r.pos.setXYZ(2 * k + 1, pts[k].x - _s.x, pts[k].y - _s.y, pts[k].z - _s.z);
    r.alpha.setX(2 * k, alpha[k]);
    r.alpha.setX(2 * k + 1, alpha[k]);
  }
  r.pos.needsUpdate = true;
  r.alpha.needsUpdate = true;
}
const smoke = [0, 1].map((i) => ribbon(trailN, theme.smoke[i]));
const vapour = [0, 1].map(() => [ribbon(vapN, 0xffffff), ribbon(vapN, 0xffffff)]);
const sPts = Array.from({ length: trailN }, () => new THREE.Vector3());
const sHalf = new Float32Array(trailN), sAlpha = new Float32Array(trailN);
const vPts = Array.from({ length: vapN }, () => new THREE.Vector3());
const vHalf = new Float32Array(vapN).fill(theme.vapourWidth * theme.jetScale), vAlpha = new Float32Array(vapN);
// the engine: an additive glow at the nozzle
const glowTex = (() => {
  const cv = document.createElement("canvas");
  cv.width = cv.height = 64;
  const cx = cv.getContext("2d"), gr = cx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.3, "rgba(255,255,255,0.6)"); gr.addColorStop(1, "rgba(255,255,255,0)");
  cx.fillStyle = gr; cx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(cv);
})();
const burners = [0, 1].map(() => {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: theme.burner, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  s.scale.setScalar(2.5 * theme.jetScale);
  scene.add(s);
  return s;
});
const tracers = [0, 1].map((i) => {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({ color: theme.tracer[i], size: 5, sizeAttenuation: false, fog: false }));
  pts.frustumCulled = false;
  scene.add(pts);
  return pts;
});

// ── hits and kills: a small pool of sprites (sparks, fire, smoke) and debris, on their own clock so they play on
//    after the replay's last frame (a kill ends the recording)
const puffTex = (() => {
  const cv = document.createElement("canvas");
  cv.width = cv.height = 64;
  const cx = cv.getContext("2d"), gr = cx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.5, "rgba(255,255,255,0.5)"); gr.addColorStop(1, "rgba(255,255,255,0)");
  cx.fillStyle = gr; cx.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(cv);
})();
const fx = [];
function spawn(kind, at, vel, life, size0, size1, color) {
  const add = kind !== "smoke";
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: puffTex, color, transparent: true, depthWrite: false,
    blending: add ? THREE.AdditiveBlending : THREE.NormalBlending }));
  s.position.copy(at);
  scene.add(s);
  fx.push({ s, vel: vel.clone(), life, age: 0, size0, size1, kind, rise: kind === "smoke" ? 6 : 0 });
}
const debris = [];
function explode(at, vel, ground) {
  const v = vel.clone().multiplyScalar(ground ? 0 : 0.6);
  for (let k = 0; k < 14; k++) {
    const d = new THREE.Vector3(rand() - 0.5, rand() - 0.3, rand() - 0.5).normalize().multiplyScalar(20 + rand() * 50);
    spawn("fire", at, v.clone().add(d), 0.8 + rand() * 0.9, 15, 70 + rand() * 50, theme.fire[k % 2]);
  }
  for (let k = 0; k < 22; k++) {
    const d = new THREE.Vector3(rand() - 0.5, rand() * 0.6, rand() - 0.5).multiplyScalar(30);
    spawn("smoke", at, v.clone().multiplyScalar(0.5).add(d), 3 + rand() * 2.5, 15, 55 + rand() * 35, theme.deathSmoke);
  }
  if (ground) return;
  const mat = new THREE.MeshStandardMaterial({ color: theme.debris, roughness: 0.8 });
  for (let k = 0; k < 12; k++) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(1 + rand() * 3, 0.3, 0.6 + rand() * 2).scale(theme.jetScale, theme.jetScale, theme.jetScale), mat);
    m.position.copy(at);
    scene.add(m);
    debris.push({ m, vel: v.clone().add(new THREE.Vector3(rand() - 0.5, rand() - 0.2, rand() - 0.5).multiplyScalar(80)),
      spin: new THREE.Vector3(rand(), rand(), rand()).multiplyScalar(6), smokeT: 0 });
  }
}
function stepFx(sec) {
  for (let k = fx.length - 1; k >= 0; k--) {
    const f = fx[k];
    f.age += sec;
    if (f.age >= f.life) { scene.remove(f.s); f.s.material.dispose(); fx.splice(k, 1); continue; }
    const u = f.age / f.life;
    f.vel.multiplyScalar(Math.exp(-sec * 1.5));
    f.vel.y += f.rise * sec;
    f.s.position.addScaledVector(f.vel, sec);
    f.s.scale.setScalar(f.size0 + (f.size1 - f.size0) * Math.sqrt(u));
    f.s.material.opacity = f.kind === "smoke" ? 0.75 * (1 - u) : (1 - u) * (1 - u);
  }
  for (const d of debris) {
    if (d.m.position.y <= 0) continue;
    d.vel.y -= 9.81 * sec;
    d.m.position.addScaledVector(d.vel, sec);
    d.m.rotation.x += d.spin.x * sec; d.m.rotation.y += d.spin.y * sec; d.m.rotation.z += d.spin.z * sec;
    d.smokeT += sec;
    if (d.smokeT > 0.12) { d.smokeT = 0; spawn("smoke", d.m.position, new THREE.Vector3(), 2.5, 6, 30, theme.deathSmoke); }
  }
}
export { THREE, v3, renderer, scene, camera, puffs, jets, smoke, vapour, sPts, sHalf, sAlpha, vPts, vHalf, vAlpha,
  fillRibbon, burners, tracers, spawn, explode, stepFx, trailN, vapN, rand };
