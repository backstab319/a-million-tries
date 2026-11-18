// The page's look: the channel's colours (the same values as src/ailearns/render/theme.py) plus the booster scene's
// own, from the episode's 2D view (render/rocket_view.py). Colours live here only.
export const THEME = {
  bg: "#0d0f16", panel: "#161923", panelEdge: "#262b3a", text: "#e8eaf0", muted: "#787f94",
  blue: "#60a5fa", red: "#f43f5e", amber: "#fbbf24", green: "#4ade80", purple: "#a78bfa", orange: "#fb923c",
  font: "Inter, system-ui, sans-serif", mono: "'JetBrains Mono', ui-monospace, monospace",
};
export const SCENE = {
  skyLow: [30, 52, 84], space: [13, 15, 22], sea: "#10243a", seaLine: "#284a6e", deck: "#464c58",
  deckEdge: "#fbbf24", body: "#eceef2", soot: "#3a3634", interstage: "#282a32", legs: "#969caa",
  flameCore: "#ffecbe", trail: "#3a4152",
};
export const mix = (a, b, t) => a.map((v, k) => Math.round(v + (b[k] - v) * t));
export const css = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;
