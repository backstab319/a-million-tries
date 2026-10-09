// Snake and the AI's brain, ported from games/snake_jax.py and agents/ppo.py (SnakeNet,
// arch "cnn-deep", view "ego2"). Weights: weights.bin + the layer table in snake.json.
// Works in the browser and in Node (check.mjs replays a real AI game move for move).

export const DELTAS = [[-1, 0], [0, 1], [1, 0], [0, -1]];   // up, right, down, left
export const STRAIGHT = 0, TURN_RIGHT = 1, TURN_LEFT = 2;

export class Snake {
  constructor(cfg, rand = Math.random) {
    this.s = cfg.size; this.L0 = cfg.start_length; this.cfg = cfg; this.rand = rand;
    const s = this.s, w = 2 * s - 1;
    // snake's-eye view: where each cell of the 19x19 view comes from, per heading (np.rot90)
    this.rot = [0, 1, 2, 3].map((d) => {
      const src = new Int32Array(2 * w * w);
      for (let i = 0; i < w; i++) for (let j = 0; j < w; j++) {
        let a = i, b = j;
        for (let k = 0; k < d; k++) [a, b] = [b, w - 1 - a];
        src[2 * (i * w + j)] = a; src[2 * (i * w + j) + 1] = b;
      }
      return src;
    });
  }

  init(food = null) {
    const s = this.s, board = new Int32Array(s * s);
    for (let k = 0; k < this.L0; k++) board[(s >> 1) * s + (s >> 1) - k] = this.L0 - k;
    const st = { board, head: [s >> 1, s >> 1], dir: 1, food: null, length: this.L0, hunger: 0, steps: 0,
                 dead: false, won: false, starved: false };
    st.food = food ?? this.randomEmpty(board);
    return st;
  }

  randomEmpty(board) {
    const empty = [];
    for (let i = 0; i < board.length; i++) if (board[i] === 0) empty.push(i);
    const c = empty[Math.floor(this.rand() * empty.length)];
    return [Math.floor(c / this.s), c % this.s];
  }

  // One move (relative: straight / right / left). food: where the next apple appears if this
  // move eats one (replays pass the recorded one; play leaves it random).
  step(st, action, food = null) {
    const s = this.s, cfg = this.cfg;
    const dir = (st.dir + (action === TURN_RIGHT ? 1 : action === TURN_LEFT ? 3 : 0)) % 4;
    const r = st.head[0] + DELTAS[dir][0], c = st.head[1] + DELTAS[dir][1];
    const hitWall = r < 0 || r >= s || c < 0 || c >= s;
    const ate = !hitWall && r === st.food[0] && c === st.food[1];
    const hitSelf = !hitWall && st.board[r * s + c] > 1;   // the tail (1) moves out this turn
    const dead = hitWall || hitSelf;
    const board = st.board.slice();
    let length = st.length;
    if (!dead) {
      if (!ate) for (let i = 0; i < board.length; i++) board[i] = Math.max(board[i] - 1, 0);
      else length += 1;
      board[r * s + c] = length;
    }
    const won = ate && length === s * s;
    const hunger = ate ? 0 : st.hunger + 1;
    const starved = !dead && !won && hunger > cfg.hunger_base + cfg.hunger_per_length * length;
    return { board, head: dead ? st.head : [r, c], dir, length, hunger, steps: st.steps + 1,
             food: ate && !won ? (food ?? this.randomEmpty(board)) : st.food, dead, won, starved };
  }

  observe(st) {  // 4 planes x 19 x 19, centred on the head and turned to face its heading
    const s = this.s, w = 2 * s - 1, pad = s - 1, W = 3 * s - 2;
    const [hr, hc] = st.head, obs = new Float32Array(4 * w * w);
    const src = this.rot[st.dir];
    for (let k = 0; k < w * w; k++) {
      const wr = hr + src[2 * k], wc = hc + src[2 * k + 1];          // in the padded world (W x W)
      const br = wr - pad, bc = wc - pad;
      if (br < 0 || br >= s || bc < 0 || bc >= s) { obs[3 * w * w + k] = 1; continue; }   // wall
      const v = st.board[br * s + bc], isHead = br === hr && bc === hc;
      obs[k] = v > 0 && !isHead ? 1 : 0;                              // body here
      obs[w * w + k] = isHead ? 0 : v / st.length;                    // how long it stays
      obs[2 * w * w + k] = br === st.food[0] && bc === st.food[1] ? 1 : 0;
    }
    return obs;
  }
}

export class SnakeBrain {
  constructor(layers, body, buffer) {
    const all = new Float32Array(buffer);
    this.p = {};
    for (const l of layers) {
      const n = l.shape.reduce((a, b) => a * b, 1);
      this.p[l.name] = { shape: l.shape, data: all.subarray(l.offset, l.offset + n) };
    }
    this.body = body;                       // [["Conv2d", stride, pad], ["ReLU"], ..., ["Flatten"], ["Linear"], ["ReLU"]]
  }

  static conv(x, C, H, Wd, w, b, stride, pad) {
    // pad once, then for each weight sweep the whole output plane (no bounds checks inside)
    const [O, , K] = w.shape, Ho = Math.floor((H + 2 * pad - K) / stride) + 1, Wo = Math.floor((Wd + 2 * pad - K) / stride) + 1;
    const Hp = H + 2 * pad, Wp = Wd + 2 * pad, xp = new Float32Array(C * Hp * Wp);
    for (let c = 0; c < C; c++) for (let i = 0; i < H; i++)
      xp.set(x.subarray ? x.subarray((c * H + i) * Wd, (c * H + i + 1) * Wd) : x.slice((c * H + i) * Wd, (c * H + i + 1) * Wd),
             (c * Hp + i + pad) * Wp + pad);
    const y = new Float32Array(O * Ho * Wo), wd = w.data;
    for (let o = 0; o < O; o++) {
      const yo = o * Ho * Wo;
      y.fill(b.data[o], yo, yo + Ho * Wo);
      for (let c = 0; c < C; c++) for (let ki = 0; ki < K; ki++) for (let kj = 0; kj < K; kj++) {
        const wv = wd[((o * C + c) * K + ki) * K + kj];
        if (wv === 0) continue;
        for (let i = 0; i < Ho; i++) {
          const row = (c * Hp + i * stride + ki) * Wp + kj, out = yo + i * Wo;
          for (let j = 0; j < Wo; j++) y[out + j] += wv * xp[row + j * stride];
        }
      }
    }
    return [y, O, Ho, Wo];
  }

  static linear(x, w, b) {
    const [O, I] = w.shape, y = new Float32Array(O);
    for (let o = 0; o < O; o++) { let s = b.data[o]; for (let i = 0; i < I; i++) s += w.data[o * I + i] * x[i]; y[o] = s; }
    return y;
  }

  scores(obs, C = 4, H = 19, W = 19) {   // the three moves' scores (logits): straight, right, left
    let x = obs;
    this.body.forEach(([kind, stride, pad], i) => {
      const w = this.p[`body.${i}.weight`], b = this.p[`body.${i}.bias`];
      if (kind === "Conv2d") [x, C, H, W] = SnakeBrain.conv(x, C, H, W, w, b, stride, pad);
      else if (kind === "Linear") x = SnakeBrain.linear(x, w, b);
      else if (kind === "ReLU") x = x.map((v) => (v > 0 ? v : 0));
    });
    return Array.from(SnakeBrain.linear(x, this.p["policy.weight"], this.p["policy.bias"]));
  }

  act(obs) { const s = this.scores(obs); return s.indexOf(Math.max(...s)); }
}
