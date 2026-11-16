// The page's look: the channel's colours (the same values as src/ailearns/render/theme.py, via web/cqb/theme.js)
// plus the land's own, from the episode's 3D view (web/eco/theme.js), flattened for a map. Colours live here only.
export const THEME = {
  bg: "#0d0f16", panel: "#161923", panelEdge: "#262b3a", text: "#e8eaf0", muted: "#787f94",
  blue: "#60a5fa", red: "#f43f5e", amber: "#fbbf24", green: "#4ade80", purple: "#a78bfa",
  font: "Inter, system-ui, sans-serif", mono: "'JetBrains Mono', ui-monospace, monospace",
};
export const ECO = {
  soil: ["#3d5a66", "#c9ad7c", "#7d6848", "#5a4a36"],   // bare ground by biome: lake bed, desert, grassland, forest
  wetSoil: "#3a2f24",       // soaked ground (blends in as the soil gets wet)
  green: "#5d9c3a",         // growing grass
  greenForest: "#3f7d33",   // ... in the forest
  straw: "#c8b06a",         // standing dead grass
  water: "#1f5a66", shore: "#2d6e78",
  fur: ["#c9a26b", "#2b1210"],   // an animal by diet: grass-eater ... meat-eater
  furYoung: "#e2c79a",      // the unweaned
  outline: "#231a14",
  crown: "#2f5e2a",         // a fruit tree's crown, from above
  fruit: "#d8323c",
  carcass: "#7a1f2b",
  call: "#fbbf24", fight: "#f43f5e", drink: "#6fd3ff", storm: "#b4bcc8",
  // causes of death, in the panel
  thirst: "#6fd3ff", starved: "#c8b06a", killed: "#f43f5e", old: "#787f94",
};
export const rgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
export const mix = (a, b, t) => a.map((v, k) => v + (b[k] - v) * t);
