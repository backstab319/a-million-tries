// The dogfight page's look (ep21). Colours and sizes live here, nowhere else.
export const theme = {
  skyTop: 0x2c4f7c, skyBottom: 0xb9cde0, fog: 0xb9cde0,
  ground: 0x5d6b4f, grid: 0x7a876a,
  // farmland seen from 3 km: fields of a few greens and browns, darker woods (the sim's ground is flat, so is this)
  fields: [0x6b7a4a, 0x7f8a52, 0x8f8659, 0x5a6b40, 0x9a9466, 0x6f7f55],
  woods: 0x4f5e3b, fieldSize: 450,     // metres
  sun: 0xfff3d6, sunDir: [0.45, 0.55, -0.7],
  cloud: 0xffffff, cloudShade: 0xb8c4d4, cloudBase: 2200, cloudCount: 140, cloudSpread: 30000,
  jets: [0x4aa3ff, 0xff6b4a],          // learner blue, target / other red (fin, wingtips, markers)
  airframe: 0x8a929b, canopy: 0x4a5a6e, nozzle: 0x3a3a3e,   // F-16 air-superiority grey
  smoke: [0xdfe9f5, 0xf5e2dc],
  tracer: [0xfff2a0, 0xffc08a],
  hudFont: "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
  hudText: "#f2f4f8", hudDim: "#c9cfdc",
  jetScale: 3,                          // drawn 3x real size so they read from far away (say so in the video)
  trailSeconds: 8,
  smokeWidth: [4, 22], smokeAlpha: 0.4, smokeNear: [150, 450],   // fades out this close to the camera (m);  // metres wide at the jet and after trailSeconds; alpha at the jet
  vapourG: [5.5, 8], vapourSeconds: 1.2, vapourAlpha: 0.45, vapourWidth: 0.35,   // wingtip vapour fades in between these g
  burner: 0xffb070,
  arc: 0x5fd38a,                        // (ep21) the pull-out path it could have flown (?arc=)
  spark: 0xffe28a, fire: [0xffd27a, 0xff7a2a], deathSmoke: 0x2b2b2e, debris: 0x5a5f66,
};
