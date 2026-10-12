"""Scripted CQB players: opponents and baselines for the AI, and early footage.

Each is (game, state, who, key) -> action (7 numbers, see cqb.FWD..THROW), pure JAX.
What they may use: their own side of the state (what they see, remember, hear, their own "seen"
map) and the floor plan (rooms, doors, the room graph: like a briefing). Never where the other one
is, unless they see or hear it. The AI gets less than this: only its 107 numbers.

- dummy: stands still, never shoots (checks that Assault can find someone at all).
- random: random actions (what an untrained brain does, roughly).
- camper (Defense): stays where it starts, faces the doorway it'd be entered from (the first door
  on the way out of its room); turns to what it sees, hears or last saw; fires on sight.
- wanderer (Defense): walks somewhere new in the building every 12 s, looking where it walks;
  fires on sight.
- walker (Assault): goes to the nearest place it hasn't looked at yet (a "seen" map cell), again
  and again, looking where it walks; fires on sight. No slicing the pie, no checking corners
  from the doorway: the plain way in, to compare the AI against.
- walker_flash (Assault, games with flashbangs): the walker, plus the textbook use of a flashbang. Before a
  room it hasn't looked at, it stands beside the doorway (out of line with the room), opens the door from
  there, throws one in through the opening (about 2 m in: in front of whoever covers the door), steps further
  along the wall until it goes off, and goes in. One per room, while it has them; if no
  throw from beside the door lands in the room, it goes in like the walker.

Moving around uses the building's waypoint graph (cqb.nav_graph): each decision a script walks
towards the waypoint in plain view that makes its route shortest, and opens shut doors in its way.
"""

import jax
import jax.numpy as jnp

from ailearns.games.cqb import (ACTION_SIZE, FAR, FIRE, FWD, INTERACT, RELOAD, RUN, SIDE, THROW, TURN, CQBGame, State,
                                _cast, _clear, _rot, _slab, _wrap, full_action)


def _aim(game: CQBGame, st: State, who: int, target):
    """Turn towards a point: -> (turn, angle still to go)."""
    d = target - st.pos[who]
    diff = _wrap(jnp.arctan2(d[1], d[0]) - st.face[who])
    step = jnp.deg2rad(game.cfg.turn_rate) * game.cfg.dt
    return jnp.clip(diff / step, -1, 1), diff


def _act(fwd=0.0, side=0.0, turn=0.0, fire=False, interact=False, throw=False):
    a = jnp.zeros(ACTION_SIZE)
    on = lambda x: jnp.where(x, 1.0, -1.0)  # noqa: E731
    return (a.at[FWD].set(fwd).at[SIDE].set(side).at[TURN].set(turn).at[FIRE].set(on(fire))
            .at[INTERACT].set(on(interact)).at[RUN].set(-1.0).at[THROW].set(on(throw)).at[RELOAD].set(-1.0))


def _reloading(game: CQBGame, st: State, who: int, act):
    """With magazines: reload in a quiet moment (half empty, nothing seen for 3 s); empty ones reload by themselves."""
    if not game.guns_on:
        return act
    quiet = ~st.seen[who] & (st.since[who] > 30) & (st.mag[who] <= game.cfg.magazine // 2)
    return act.at[RELOAD].set(jnp.where(quiet, 1.0, -1.0))


def room_of(game: CQBGame, st: State, p):
    """The room p is in (max_rooms: outside, or on a wall line)."""
    b, bi, R = game.b, st.building, game.cfg.max_rooms
    inside = (p > b.room_lo[bi]).all(1) & (p < b.room_hi[bi]).all(1)
    return jnp.where(inside.any(), jnp.argmax(inside), R)


def _shoot_on_sight(game: CQBGame, st: State, who: int):
    """Stand still, turn onto it, fire once it's in the sights."""
    turn, diff = _aim(game, st, who, st.pos[1 - who])
    return _act(turn=turn, fire=jnp.abs(diff) < jnp.deg2rad(game.cfg.aim) * 0.8)


def _walk(game: CQBGame, st: State, who: int, target, interact=False, look=None):
    """Walk towards a point, looking where it walks (or at `look`); turn first when it's way off."""
    p, face = st.pos[who], st.face[who]
    d = target - p
    dist = jnp.linalg.norm(d)
    want = d / jnp.maximum(dist, 1e-6)
    local = _rot(want * jnp.minimum(dist / 0.3, 1.0), -face)
    turn, diff = _aim(game, st, who, p + want if look is None else look)
    slow = jnp.clip(1.0 - jnp.abs(diff) / jnp.pi, 0.3, 1.0)
    return _act(fwd=local[0] * slow, side=local[1] * slow / game.cfg.sideways, turn=turn, interact=interact)


def _routes(game: CQBGame, st: State, who: int):
    """From where I stand: which waypoints I can walk straight to, and the walking distance from me
    to every waypoint (via the visible one that makes it shortest). -> (visible (K,), dist (K,), via (K,))"""
    b, bi = game.b, st.building
    p = st.pos[who]
    nodes = b.nav_nodes[bi]
    k = game.cfg.radius - 0.05                     # a bit under my radius: pressed against a wall I'm 0.3 m from it
    lo = jnp.concatenate([b.wall_lo[bi], b.cover_lo[bi]]) - k
    hi = jnp.concatenate([b.wall_hi[bi], b.cover_hi[bi]]) + k
    vis = _clear(p, nodes, lo, hi) & (nodes[:, 0] > FAR / 2)
    first = jnp.where(vis, jnp.linalg.norm(nodes - p, axis=1), 1e5)          # (K,) me -> visible waypoint
    total = first[:, None] + b.nav_dist[bi]                                  # (K via, K to)
    return vis, total.min(0), total.argmin(0)


def _nav(game: CQBGame, st: State, who: int, goal):
    """Walk to waypoint `goal` along the waypoint graph, opening shut doors that are in the way."""
    b, bi = game.b, st.building
    p = st.pos[who]
    nodes = b.nav_nodes[bi]
    vis, _, via = _routes(game, st, who)
    n = via[goal]
    n = jnp.where(jnp.linalg.norm(nodes[n] - p) < 0.25, b.nav_next[bi, n, goal], n)   # there: on to the next
    after = b.nav_next[bi, n, goal]
    n = jnp.where(vis[after] & (after >= 0), after, n)                       # the one after is in view: cut the corner
    target = jnp.where(jnp.linalg.norm(nodes[goal] - p) < 0.25, p, nodes[jnp.maximum(n, 0)])
    # a shut door on the line to the target, within reach: open it
    t_in, t_out = _slab(p, (target - p)[None, :], b.door_lo[bi], b.door_hi[bi])
    crosses = (t_in <= t_out) & (t_out >= 0) & (t_in <= 1) & b.door_valid[bi]
    reach = jnp.linalg.norm(p - jnp.clip(p, b.door_lo[bi], b.door_hi[bi]), axis=1) <= game.cfg.reach * 0.9
    opening = (crosses & ~st.door_open & (st.door_timer == 0) & reach).any()
    return _walk(game, st, who, target, interact=opening)


# ── The players ──────────────────────────────────────────────────────────────

def dummy(game, st, who, key=None):
    return _act()


def random(game, st, who, key):
    a = jax.random.uniform(key, (6,), minval=-1, maxval=1)       # the first 6 drawn as before flashbangs
    return jnp.concatenate([a, jax.random.uniform(jax.random.fold_in(key, 1), (1,), minval=-1, maxval=1)])


def camper(game: CQBGame, st: State, who: int, key=None):
    b, bi, R = game.b, st.building, game.cfg.max_rooms
    p = st.pos[who]
    here = room_of(game, st, p)
    k = jnp.maximum(b.next_door[bi, here, R], 0)                           # the first door on the way out
    watch = (b.door_lo[bi, k] + b.door_hi[bi, k]) / 2
    snd = st.sound[who]
    loud = jnp.argmax(snd[:, 2])
    heard = snd[loud, 2] > 0
    target = jnp.where(st.since[who] < 30, st.mem[who],                    # just saw it (or its flash): there
                       jnp.where(heard, p + snd[loud, :2], watch))         # heard it: that way; else the door
    turn, _ = _aim(game, st, who, target)
    return _reloading(game, st, who, jnp.where(st.seen[who], _shoot_on_sight(game, st, who), _act(turn=turn)))


def wanderer(game: CQBGame, st: State, who: int, key=None):
    cells = game.cfg.seen_grid[0] * game.cfg.seen_grid[1]
    goal = (st.building * 7 + (st.t // 120) * 13) % cells                    # somewhere new every 12 s
    return _reloading(game, st, who, jnp.where(st.seen[who], _shoot_on_sight(game, st, who), _nav(game, st, who, goal)))


def _goal(game: CQBGame, st: State, who: int):
    """The walker's goal: the nearest "seen" map cell not looked at yet (all looked at: somewhere, again)."""
    cells = game.cfg.seen_grid[0] * game.cfg.seen_grid[1]
    _, dist, _ = _routes(game, st, who)
    todo = ~st.cleared[who] & (dist[:cells] < 1e4)
    goal = jnp.argmin(jnp.where(todo, dist[:cells], jnp.inf))
    return jnp.where(todo.any(), goal, (st.building * 7 + (st.t // 80) * 13) % cells)


def walker(game: CQBGame, st: State, who: int, key=None):
    lost = (st.since[who] < 30) & ~st.seen[who]                              # saw (or heard a shot) a moment ago
    turn_mem, _ = _aim(game, st, who, st.mem[who])
    act = jnp.where(lost, _act(turn=turn_mem), _nav(game, st, who, _goal(game, st, who)))
    return _reloading(game, st, who, jnp.where(st.seen[who], _shoot_on_sight(game, st, who), act))


def _rooms_of(game: CQBGame, st: State, q):
    """room_of for several points (N, 2) -> (N,)."""
    return jax.vmap(lambda x: room_of(game, st, x))(q)


def _clear_each(q, p, lo, hi):
    """Is the line from each point q (N, 2) to p clear of the boxes? -> (N,)"""
    return jax.vmap(lambda x: _clear(x, p[None], lo, hi)[0])(q)


def walker_flash(game: CQBGame, st: State, who: int, key=None):
    b, bi, R, cfg = game.b, st.building, game.cfg.max_rooms, game.cfg
    p = st.pos[who]
    here = room_of(game, st, p)
    goal_room = b.cell_room[bi, _goal(game, st, who)]
    k = jnp.maximum(b.next_door[bi, here, jnp.where(goal_room < 0, here, goal_room)], 0)   # the next door on my way
    dlo, dhi = b.door_lo[bi, k], b.door_hi[bi, k]
    mid = (dlo + dhi) / 2
    axis = jnp.argmin(dhi - dlo)                                           # doors are thin across their wall
    sides = b.door_rooms[bi, k]                                            # the room on the -normal, +normal side
    s = jnp.where(sides[1] == here, -1.0, 1.0)
    beyond = jnp.where(s > 0, sides[1], sides[0])
    into = jnp.zeros(2).at[axis].set(s)                                    # from the doorway into that room
    across = jnp.stack([-into[1], into[0]])
    unseen = ((b.cell_room[bi] == beyond) & ~st.cleared[who]).any()
    flashed = (st.bangs[who] < cfg.flashbangs[who]) & (room_of(game, st, st.bang_pos[who]) == beyond)
    heading = (goal_room >= 0) & (goal_room != here)
    near = jnp.linalg.norm(p - jnp.clip(p, dlo, dhi)) <= 2.0
    want = heading & near & unseen & ~flashed & (beyond < R) & (st.bangs[who] > 0)
    live = st.bang_timer[who] > 0
    lo, hi = game.boxes(st)
    olo, ohi = game.walls(st._replace(door_open=st.door_open.at[k].set(True)))   # as if the door were open
    # two stack spots, one each side of the doorway (0.5 m out, 0.9 m along the wall), each with a spot further
    # along to wait in; from each, throws aimed at points across the opening that land in the room with a wall
    # between where they land and the waiting spot. (All from the floor plan, so the plan doesn't change as I walk.)
    back = mid - into * 0.5
    sdir = jnp.stack([across, -across])                                    # (2 sides, 2)
    free = _cast(back, sdir, lo, hi, cfg)
    spots = back[None] + sdir * jnp.minimum(0.9, free - cfg.radius - 0.05)[:, None]
    waits = back[None] + sdir * jnp.minimum(1.6, free - cfg.radius - 0.05)[:, None]
    aims = mid[None] + jnp.linspace(-0.3, 0.3, 7)[:, None] * across[None]  # (7, 2), clear of the door frame

    def landing(src, dirs, lo, hi):
        return src + dirs * jnp.clip(jnp.minimum(cfg.throw, _cast(src, dirs, lo, hi, cfg) - 0.3), 0.0, None)[:, None]

    def plan(spot, wait_at):
        dirs = (aims - spot) / jnp.maximum(jnp.linalg.norm(aims - spot, axis=1, keepdims=True), 1e-6)
        land = landing(spot, dirs, olo, ohi)
        ok = (_rooms_of(game, st, land) == beyond) & ~_clear_each(land, wait_at, olo, ohi)
        return jnp.where(ok, (land - mid) @ into, -jnp.inf)                # how deep it lands
    # about 2 m in: in front of whoever is covering the doorway (in their light, so blinded the longest)
    score = lambda depth: jnp.where(depth > 1.0, -jnp.abs(depth - 2.0), -jnp.inf)  # noqa: E731
    sc = score(jax.vmap(plan)(spots, waits))                               # (2, 7)
    mine = jnp.where(jnp.dot(p - mid, across) >= 0, 0, 1)                  # the side I'm on: no crossing the doorway
    sc = sc.at[1 - mine].set(jnp.where(jnp.isfinite(sc[mine]).any(), -jnp.inf, sc[1 - mine]))
    pick = jnp.argmax(sc.reshape(-1))
    side_i = pick // 7                                                     # (the aim is re-planned at the spot)
    can = jnp.isfinite(sc.reshape(-1)[pick])
    spot, wait_at = spots[side_i], waits[side_i]
    there = jnp.linalg.norm(p - spot) < 0.2
    # at the spot: aim from where I really stand, and let go only if a throw the way I face lands in the room
    here_sc = score(plan(p, wait_at))
    turn, diff = _aim(game, st, who, aims[jnp.argmax(here_sc)])
    face = jnp.stack([jnp.cos(st.face[who]), jnp.sin(st.face[who])])
    wlo, whi = game.walls(st)
    now = landing(p, face[None], wlo, whi)[0]
    lands_in = ((room_of(game, st, now) == beyond) & ((now - mid) @ into > 1.0)
                & ~_clear_each(now[None], wait_at, wlo, whi)[0])
    door_shut = ~st.door_open[k]
    at_spot = jnp.where(door_shut, _act(turn=_aim(game, st, who, mid)[0], interact=st.door_timer[k] == 0),
                        _act(turn=turn, throw=(jnp.abs(diff) < jnp.deg2rad(2.0)) & lands_in))
    ready = jnp.where(there, at_spot, _walk(game, st, who, spot, look=mid))
    # thrown: wait further along the wall, facing the doorway, until it goes off (stepping off sooner puts me
    # back in line with it); then in
    dist = jnp.where(b.door_valid[bi], jnp.linalg.norm(p - jnp.clip(p, b.door_lo[bi], b.door_hi[bi]), axis=1), jnp.inf)
    w = jnp.argmin(dist)                                                   # the door it went through (the goal may
    wmid = (b.door_lo[bi, w] + b.door_hi[bi, w]) / 2                       # have changed since)
    wn = jnp.zeros(2).at[jnp.argmin(b.door_hi[bi, w] - b.door_lo[bi, w])].set(1.0)
    wn = jnp.where(jnp.dot(st.bang_pos[who] - wmid, wn) > 0, wn, -wn)     # towards the flashbang
    wa = jnp.stack([-wn[1], wn[0]])
    wa = jnp.where(jnp.dot(p - wmid, wa) >= 0, wa, -wa)                    # the side I'm on
    wfree = _cast(wmid - wn * 0.5, wa[None], lo, hi, cfg)[0]
    hold = _walk(game, st, who, wmid - wn * 0.5 + wa * jnp.minimum(1.6, wfree - cfg.radius - 0.05), look=wmid)
    act = jnp.where(live, hold, walker(game, st, who, key))
    act = jnp.where(want & ~live & can & (~there | door_shut | jnp.isfinite(here_sc).any()), ready, act)
    return _reloading(game, st, who, jnp.where(st.seen[who], _shoot_on_sight(game, st, who), act))


SCRIPTS = {"dummy": dummy, "random": random, "camper": camper, "wanderer": wanderer, "walker": walker,
           "walker_flash": walker_flash}


def play_rounds(game: CQBGame, assault, defense, buildings, seed: int = 0, record: bool = False,
                stacked: bool = False) -> dict:
    """One round per building index in `buildings` (all at once). Players are SCRIPTS names or
    functions. Returns outcomes, round lengths, final positions, when each side first saw the
    other (decision, -1 never) and (record) every state. stacked: drill starts (CQBGame.init_stacked)."""
    A = SCRIPTS.get(assault, assault)
    D = SCRIPTS.get(defense, defense)
    last, first, hist = jax.jit(lambda b: rounds(game, A, D, b, seed, record, stacked))(jnp.asarray(buildings))
    out = {"outcome": jax.device_get(last.outcome), "t": jax.device_get(last.t), "alive": jax.device_get(last.alive),
           "pos": jax.device_get(last.pos), "first_seen": jax.device_get(first)}
    if record:
        out["history"] = jax.device_get(hist)
    return out


def rounds(game: CQBGame, A, D, buildings, seed: int = 0, record: bool = False, stacked: bool = False):
    """play_rounds' rounds, as a pure function of traced values (players are functions): so a caller can compile it
    once with a brain's weights as an input and play many checkpoints through it. -> (last states, first seen,
    history or None)."""
    keys = jax.vmap(lambda b: jax.random.fold_in(jax.random.PRNGKey(seed), b))(buildings)

    def act(P, st, who, k, h):
        """A player's action; a brain with memory (ppo_cqb.brain) also takes and returns its memory h."""
        if getattr(P, "memory", 0):
            a, h = P(game, st, who, k, h)
            return full_action(a), h
        return full_action(P(game, st, who, k)), h

    def one(key, bi):
        st = (game.init_stacked if stacked else game.init)(key, bi)

        def tick(carry, k):
            st, first, ha, hd = carry
            ka, kd = jax.random.split(k)
            (aa, ha), (ad, hd) = act(A, st, 0, ka, ha), act(D, st, 1, kd, hd)
            st, _ = game.step(st, jnp.stack([aa, ad]))
            first = jnp.where((first < 0) & st.seen, st.t, first)        # when each first saw the other
            return (st, first, ha, hd), (st if record else None)

        first0 = jnp.where(st.seen, 0, -1)
        h0 = (jnp.zeros(getattr(A, "memory", 0)), jnp.zeros(getattr(D, "memory", 0)))   # each round from nothing
        (last, first, _, _), hist = jax.lax.scan(tick, (st, first0, *h0),
                                                 jax.random.split(jax.random.fold_in(key, 1), game.T))
        if record:
            hist = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), st, hist)
        return last, first, hist

    return jax.vmap(one)(keys, buildings)
