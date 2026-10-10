"""Tag, in JAX: a chaser and a runner in a square arena with obstacles, thousands of rounds at once.

Both players are discs that can accelerate in any direction. The chaser has the
higher top speed; the runner accelerates (and so changes direction) faster. The
chaser wins by touching the runner; the runner wins by lasting until time runs out.
Bumping into a wall or an obstacle just stops you.

Obstacles block vision. Each player only knows where the other one is while it
can see them; otherwise it knows where it *last* saw them, and how long ago.

What a player sees is everything in `observe`: distance "lasers" in a full circle,
its own velocity and position, the other player (or where it was last seen, and,
with `remember_heading`, how it was moving then) and the time left. Player 0 is always the chaser, player 1 the runner.

Obstacles are axis-aligned rectangles (square pillars and thin walls), which keeps
collisions, lasers and line of sight simple and exact. All functions here are pure,
so they work under jax.jit and jax.vmap; `Arenas` is plain arrays.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

CHASER, RUNNER = 0, 1


@dataclass(frozen=True)
class TagConfig:
    size: float = 40.0            # arena is size x size metres
    radius: float = 0.5           # players are discs this big
    catch_dist: float = 1.2       # centres this close = caught (touching, plus an arm's reach)
    dt: float = 0.1               # seconds per decision
    time_limit: float = 20.0      # seconds per round
    chaser_speed: float = 6.0     # top speeds, m/s
    runner_speed: float = 5.5
    chaser_accel: float = 8.0     # m/s^2 from standing; how quickly you can change direction
    runner_accel: float = 16.0
    n_rays: int = 16              # lasers in a full circle, starting east, counter-clockwise
    ray_range: float = 30.0
    max_obstacles: int = 8
    min_gap: float = 2.5          # obstacles stay this far from each other and the edge (no dead ends)
    spawn_points: int = 256       # free spots per arena to start from
    spawn_dist: float = 12.0      # players start at least this far apart
    catch_reward: float = 1.0     # rewards are the chaser's; the runner gets the negative
    timeout_penalty: float = 1.0
    time_penalty: float = 0.025   # per second, so a quick catch beats a slow one
    closer_reward: float = 0.0    # per metre the gap shrinks (a hint for training; 0 = catches only)
    remember_heading: bool = False  # out of sight: still know which way the other was moving when last seen

    @property
    def obs_size(self) -> int:
        return self.n_rays + 11  # lasers, velocity, position, seen?, other (where, velocity), since seen, time left


# ── Arenas (numpy: generated once, then handed to JAX as arrays) ─────────────

def _gap(lo1, hi1, lo2, hi2) -> float:
    """Distance between two rectangles (0 if they overlap)."""
    d = np.maximum(0, np.maximum(lo2 - hi1, lo1 - hi2))
    return float(np.hypot(*d))


def make_arena(seed: int, cfg: TagConfig = TagConfig()) -> dict:
    """3-7 obstacles: square pillars and thin walls, kept apart (and off the edge) by at
    least `min_gap`, so every free spot can reach every other: no sealed rooms."""
    rng = np.random.default_rng(seed)
    S, g = cfg.size, cfg.min_gap
    want = int(rng.integers(3, min(7, cfg.max_obstacles) + 1))
    los, his = [], []
    for _ in range(500):
        if len(los) == want:
            break
        if rng.random() < 0.5:  # pillar
            wh = np.full(2, rng.uniform(2.0, 5.0))
        else:                   # wall, horizontal or vertical
            wh = np.array([rng.uniform(6.0, 14.0), 1.0])[:: rng.choice([1, -1])]
        lo = rng.uniform(g, S - g - wh)
        hi = lo + wh
        if all(_gap(lo, hi, a, b) >= g for a, b in zip(los, his)):
            los.append(lo)
            his.append(hi)
    lo, hi = np.array(los).reshape(-1, 2), np.array(his).reshape(-1, 2)

    # free spots to start from: a 1 m grid, away from walls and obstacles
    xs = np.arange(1.0, S, 1.0)
    pts = np.stack(np.meshgrid(xs, xs), -1).reshape(-1, 2)
    clear = cfg.radius + 0.5
    closest = np.clip(pts[:, None], lo[None], hi[None])
    free = (np.linalg.norm(pts[:, None] - closest, axis=2) > clear).all(axis=1)
    pts = pts[free]
    spawn = pts[rng.choice(len(pts), cfg.spawn_points, replace=len(pts) < cfg.spawn_points)]
    return {"lo": lo, "hi": hi, "spawn": spawn, "seed": seed}


class Arenas(NamedTuple):
    lo: jax.Array      # (A, K, 2) obstacle corners; unused slots are empty boxes far outside
    hi: jax.Array      # (A, K, 2)
    spawn: jax.Array   # (A, N, 2) free starting spots


TRAIN_SEEDS = range(0, 1024)          # arenas it learns in
TEST_SEEDS = range(100_000, 100_256)  # arenas it has never seen (final numbers come from these)


def make_arenas(seeds, cfg: TagConfig = TagConfig()) -> Arenas:
    K, far = cfg.max_obstacles, -1000.0
    los, his, spawns = [], [], []
    for s in seeds:
        a = make_arena(s, cfg)
        pad = np.full((K - len(a["lo"]), 2), far)
        los.append(np.vstack([a["lo"], pad]))
        his.append(np.vstack([a["hi"], pad]))
        spawns.append(a["spawn"])
    f = lambda x: jnp.asarray(np.stack(x), jnp.float32)
    return Arenas(f(los), f(his), f(spawns))


# ── Geometry ─────────────────────────────────────────────────────────────────

def _slab(p, d, lo, hi):
    """Ray p + t*d against boxes: entry and exit t per box (hit if t_in <= t_out)."""
    inv = 1.0 / jnp.where(jnp.abs(d) < 1e-9, 1e-9, d)
    t1, t2 = (lo - p) * inv, (hi - p) * inv
    return jnp.max(jnp.minimum(t1, t2), axis=-1), jnp.min(jnp.maximum(t1, t2), axis=-1)


def _push_out(p, v, lo, hi, r):
    """Move a disc out of the obstacles it overlaps and stop its motion into them.
    Obstacles are further apart than a disc is wide, so it can touch at most one."""
    q = jnp.clip(p, lo, hi)                                   # (K, 2) closest point on each box
    d = p - q
    dist = jnp.linalg.norm(d, axis=1)
    inside = dist < 1e-6                                      # centre got into the box itself
    # inside: leave by the nearest face
    pen = jnp.concatenate([p - lo, hi - p], axis=1)           # (K, 4) distance to faces -x -y +x +y
    face = jnp.argmin(pen, axis=1)
    normals = jnp.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    n_in = normals[face]
    n_out = d / jnp.maximum(dist, 1e-6)[:, None]
    n = jnp.where(inside[:, None], n_in, n_out)
    depth = jnp.where(inside, jnp.min(pen, axis=1) + r, r - dist)
    hit = depth > 0
    p = p + jnp.sum(jnp.where(hit[:, None], n * depth[:, None], 0.0), axis=0)
    into = jnp.where(hit, jnp.minimum(jnp.sum(n * v, axis=1), 0.0), 0.0)
    v = v - jnp.sum(n * into[:, None], axis=0)
    return p, v


# ── The game (JAX) ───────────────────────────────────────────────────────────

class State(NamedTuple):
    arena: jax.Array    # which arena
    pos: jax.Array      # (2, 2) metres; row 0 chaser, row 1 runner
    vel: jax.Array      # (2, 2) m/s
    mem: jax.Array      # (2, 2) where each player last saw the other
    mem_vel: jax.Array  # (2, 2) and how it was moving then
    seen: jax.Array     # can they see each other right now? (it's mutual)
    since: jax.Array    # decisions since they last saw each other
    t: jax.Array        # decisions taken
    caught: jax.Array
    done: jax.Array


class TagGame:
    def __init__(self, arenas: Arenas, cfg: TagConfig = TagConfig()):
        self.arenas, self.cfg = arenas, cfg
        ang = jnp.arange(cfg.n_rays) * 2 * jnp.pi / cfg.n_rays
        self.ray_dirs = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)
        self.speed = jnp.array([cfg.chaser_speed, cfg.runner_speed])
        self.accel = jnp.array([cfg.chaser_accel, cfg.runner_accel])

    def init(self, key, arena=None) -> State:
        """Both players standing still on free spots at least `spawn_dist` apart
        (in `arena`, else a random one). They start out knowing where the other is."""
        ka, kc, kr = jax.random.split(key, 3)
        cfg, N = self.cfg, self.cfg.spawn_points
        ar = jax.random.randint(ka, (), 0, self.arenas.lo.shape[0]) if arena is None else jnp.asarray(arena)
        pts = self.arenas.spawn[ar]
        c = pts[jax.random.randint(kc, (), 0, N)]
        cand = pts[jax.random.randint(kr, (16,), 0, N)]
        dist = jnp.linalg.norm(cand - c, axis=1)
        far_enough = dist >= cfg.spawn_dist
        r = cand[jnp.where(far_enough.any(), jnp.argmax(far_enough), jnp.argmax(dist))]
        pos = jnp.stack([c, r])
        seen = self.visible(ar, c, r)
        z = jnp.asarray(0)
        return State(ar, pos, jnp.zeros((2, 2)), pos[::-1], jnp.zeros((2, 2)), seen, z, z, jnp.asarray(False),
                     jnp.asarray(False))

    def visible(self, arena, a, b) -> jax.Array:
        """True if no obstacle blocks the straight line from a to b."""
        t_in, t_out = _slab(a, b - a, self.arenas.lo[arena], self.arenas.hi[arena])
        blocked = (t_in <= t_out) & (t_out >= 0) & (t_in <= 1)
        return ~blocked.any()

    def lasers(self, arena, p) -> jax.Array:
        """Distance (metres) to the nearest wall or obstacle along each laser, capped at ray_range."""
        d = self.ray_dirs                                                   # (R, 2)
        S = self.cfg.size
        edge = jnp.where(d > 0, S - p, -p) / jnp.where(jnp.abs(d) < 1e-9, 1e-9, d)
        edge = jnp.min(jnp.where(jnp.abs(d) < 1e-9, jnp.inf, edge), axis=1)  # (R,)
        t_in, t_out = _slab(p, d[:, None, :], self.arenas.lo[arena], self.arenas.hi[arena])  # (R, K)
        hit = (t_in <= t_out) & (t_out > 0)
        box = jnp.min(jnp.where(hit, jnp.maximum(t_in, 0.0), jnp.inf), axis=1)
        return jnp.minimum(jnp.minimum(edge, box), self.cfg.ray_range)

    def step(self, st: State, actions) -> tuple[State, jax.Array]:
        """actions: (2, 2), one acceleration per player (length capped at 1).
        Returns the new state and the chaser's reward (the runner's is its negative)."""
        cfg = self.cfg
        a = jnp.clip(actions, -1, 1)
        a = a / jnp.maximum(jnp.linalg.norm(a, axis=1, keepdims=True), 1.0)
        acc = a * self.accel[:, None] - st.vel * (self.accel / self.speed)[:, None]  # drag: top speed is where they cancel
        vel = st.vel + acc * cfg.dt
        pos = st.pos + vel * cfg.dt

        r, S = cfg.radius, cfg.size  # the arena's edge
        clipped = jnp.clip(pos, r, S - r)
        vel = jnp.where(clipped != pos, 0.0, vel)
        pos = clipped
        lo, hi = self.arenas.lo[st.arena], self.arenas.hi[st.arena]
        pos, vel = jax.vmap(_push_out, in_axes=(0, 0, None, None, None))(pos, vel, lo, hi, r)

        # caught: closest the two came at any moment during this step, not just at its end
        rel0, rel1 = st.pos[RUNNER] - st.pos[CHASER], pos[RUNNER] - pos[CHASER]
        dr = rel1 - rel0
        u = jnp.clip(-jnp.dot(rel0, dr) / jnp.maximum(jnp.dot(dr, dr), 1e-9), 0, 1)
        caught = jnp.linalg.norm(rel0 + u * dr) < cfg.catch_dist

        seen = self.visible(st.arena, pos[CHASER], pos[RUNNER])
        mem = jnp.where(seen, pos[::-1], st.mem)
        mem_vel = jnp.where(seen, vel[::-1], st.mem_vel)
        since = jnp.where(seen, 0, st.since + 1)
        t = st.t + 1
        timeout = ~caught & (t * cfg.dt >= cfg.time_limit - 1e-6)
        closer = jnp.linalg.norm(rel0) - jnp.linalg.norm(rel1)
        reward = (caught * cfg.catch_reward - timeout * cfg.timeout_penalty - cfg.time_penalty * cfg.dt
                  + cfg.closer_reward * closer)
        new = State(st.arena, pos, vel, mem, mem_vel, seen, since, t, caught, caught | timeout)
        return new, reward

    def observe(self, st: State, who: int) -> jax.Array:
        """What player `who` (CHASER or RUNNER) knows, scaled to about [-1, 1]."""
        cfg, me, other = self.cfg, who, 1 - who
        p = st.pos[me]
        other_vel = jnp.where(st.seen, st.vel[other], st.mem_vel[me] if cfg.remember_heading else 0.0)
        return jnp.concatenate([
            self.lasers(st.arena, p) / cfg.ray_range,
            st.vel[me] / cfg.chaser_speed,
            p / cfg.size * 2 - 1,
            st.seen[None].astype(jnp.float32),
            (st.mem[me] - p) / cfg.size,
            other_vel / cfg.chaser_speed,
            jnp.minimum(st.since * cfg.dt / cfg.time_limit, 1.0)[None],
            (1 - st.t * cfg.dt / cfg.time_limit)[None],
        ])

    def reset_if(self, st: State, key) -> State:
        """Finished rounds start again in a random arena (for training)."""
        fresh = self.init(key)
        return jax.tree.map(lambda a, b: jnp.where(st.done, a, b), fresh, st)


# ── Scripted players (opponents and baselines; they only use what they're allowed to know) ──
# Each is (game, state, key) -> action.

def chaser_naive(game: TagGame, st: State, key=None) -> jax.Array:
    """Run straight at the runner, or at where it was last seen. If it isn't there any more,
    head for the middle of the arena."""
    p, target = st.pos[CHASER], st.mem[CHASER]
    lost = ~st.seen & (jnp.linalg.norm(target - p) < 1.5)
    target = jnp.where(lost, jnp.full(2, game.cfg.size / 2), target)
    d = target - p
    return d / jnp.maximum(jnp.linalg.norm(d), 1e-6)


def runner_flee(game: TagGame, st: State, key=None, danger: float = 4.0, dodge_at: float = 4.0) -> jax.Array:
    """Run directly away from the chaser (or where it was last seen), get pushed off walls and
    obstacles by its lasers, and side-step towards the more open side when the chaser gets close."""
    p, threat = st.pos[RUNNER], st.mem[RUNNER]
    away = p - threat
    dist = jnp.linalg.norm(away)
    away = away / jnp.maximum(dist, 1e-6)
    L = game.lasers(st.arena, p)
    w = jnp.clip((danger - L) / danger, 0, 1) ** 2
    repel = -jnp.sum(game.ray_dirs * w[:, None], axis=0)
    perp = jnp.array([-away[1], away[0]])
    room = jnp.sum(jnp.dot(game.ray_dirs, perp) * L)  # > 0: more room on the perp side
    dodge = perp * jnp.where(room >= 0, 1.0, -1.0) * (st.seen & (dist < dodge_at))
    want = away + 2.0 * repel + 1.5 * dodge
    return want / jnp.maximum(jnp.linalg.norm(want), 1e-6)


def runner_smart(game: TagGame, st: State, key=None, look: float = 6.0, edge_weight: float = 0.8,
                 hide_weight: float = 3.0, dodge_at: float = 3.0, temp: float = 0.5) -> jax.Array:
    """Like `runner_flee`, but it looks before it runs: for each of its 16 laser directions it
    asks "if I go that way, how far from the chaser (where it's heading) do I end up, how
    much room is there, and how far from the arena's edge am I?" and runs the best way.
    It also likes ways that put an obstacle between them. So it stays off the walls and
    out of corners, and uses obstacles to break line of sight."""
    cfg, p, threat = game.cfg, st.pos[RUNNER], st.mem[RUNNER]
    ahead = threat + jnp.where(st.seen, st.vel[CHASER], 0.0) * 0.5  # where the chaser will be
    L = game.lasers(st.arena, p)
    reach = jnp.minimum(L - cfg.radius, look)
    q = p + game.ray_dirs * reach[:, None]                          # (R, 2) where each way leads
    gap = jnp.linalg.norm(q - ahead, axis=1)
    edge = jnp.minimum(jnp.min(jnp.minimum(q, cfg.size - q), axis=1), 8.0)
    hidden = ~jax.vmap(lambda x: game.visible(st.arena, ahead, x))(q)  # an obstacle would be in the way
    score = gap + edge_weight * edge + 0.5 * reach + hide_weight * hidden
    w = jax.nn.softmax(score / temp)
    want = jnp.sum(game.ray_dirs * w[:, None], axis=0)
    away = p - threat
    dist = jnp.linalg.norm(away)
    away = away / jnp.maximum(dist, 1e-6)
    perp = jnp.array([-away[1], away[0]])
    room = jnp.sum(jnp.dot(game.ray_dirs, perp) * L)
    dodge = perp * jnp.where(room >= 0, 1.0, -1.0) * (st.seen & (dist < dodge_at))
    want = want / jnp.maximum(jnp.linalg.norm(want), 1e-6) + 1.5 * dodge
    return want / jnp.maximum(jnp.linalg.norm(want), 1e-6)


def runner_still(game: TagGame, st: State, key=None) -> jax.Array:
    return jnp.zeros(2)


def chaser_random(game: TagGame, st: State, key) -> jax.Array:
    return jax.random.uniform(key, (2,), minval=-1, maxval=1)


def play_rounds(game: TagGame, chaser, runner, n: int, seed: int = 0, arenas=None) -> dict:
    """n rounds of two scripted players, one round per
    arena in turn (or in random arenas). Returns catch rate, average time to catch, and the
    rounds' paths."""
    cfg = game.cfg
    T = int(round(cfg.time_limit / cfg.dt))
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    A = game.arenas.lo.shape[0]
    which = jnp.arange(n) % A if arenas is None else jnp.asarray(arenas)

    def one(key, arena):
        st = game.init(key, arena)

        def tick(st, k):
            kc, kr = jax.random.split(k)
            act = jnp.stack([chaser(game, st, kc), runner(game, st, kr)])
            new, _ = game.step(st, act)
            new = jax.tree.map(lambda a, b: jnp.where(st.done, b, a), new, st)  # freeze once over
            return new, (new.pos, new.seen)

        st, (path, seen) = jax.lax.scan(tick, st, jax.random.split(key, T))
        return st, path, seen

    st, path, seen = jax.jit(jax.vmap(one))(keys, which)
    caught = np.asarray(st.caught)
    times = np.asarray(st.t) * cfg.dt
    return {"catch_rate": float(caught.mean()), "catch_time": float(times[caught].mean()) if caught.any() else None,
            "caught": caught, "times": times, "paths": np.asarray(path), "seen": np.asarray(seen),
            "arena": np.asarray(which)}
