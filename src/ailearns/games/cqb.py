"""CQB, in JAX: one Assault operator clears a dark kill house, one Defense operator holds it.

The building is generated from a seed: a footprint in a yard (about 22-40 m^2 per room),
split by walls into 3-6 rooms of 3.5-10 m a side, every wall between two rooms with one
doorway, and 2-3 doorways to the outside.
Walls, doors and cover are axis-aligned boxes (like tag's obstacles), which keeps
collisions, lasers and line of sight simple and exact.

Coordinates are metres; the yard is `yard_w` x `yard_h`, x east, y north. A wall is
drawn centred on its line, `wall` thick. Rooms are given by their wall lines.

Every doorway has a door (closed or open). For each side of a door we record whether
it is *center-fed* (the doorway is away from the room's corners: hidden corners on both
sides of the entry) or *corner-fed* (the doorway is next to a corner: one hidden corner),
the two situations people are taught to enter differently (docs/cqb-plan.md 2b).

All functions on the game itself are pure, so they work under jax.jit and jax.vmap;
`Buildings` is plain arrays, padded to fixed sizes so nothing recompiles.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

ASSAULT, DEFENSE = 0, 1
FAR = -1000.0  # where unused (or open) boxes are parked
# an action is 8 numbers per operator: forward/back, sideways (left +), turn (left +), and five
# on/off choices (on when > 0). Older players give 6 or 7: a missing THROW or RELOAD reads as off
FWD, SIDE, TURN, FIRE, INTERACT, RUN, THROW, RELOAD = range(8)
ACTION_SIZE = 8
BANG_OBS = 8  # extra observation numbers when the game has flashbangs (see CQBGame.observe)
GUN_OBS = 7   # and when it has magazines
HEALTH_OBS = 2  # and when it has health
REACT_OBS = 1   # and when it has a reaction time
ENEMY_OBS = 10  # numbers about one enemy (with teams: one slot per enemy, nearest known first)
MATE_OBS = 6    # numbers about one teammate
READY_OBS = 1   # and, last, how far my rifle is up (with `ready`)
STUN_OBS = 1    # and, after it, how long I'm still stunned (with `stun`)


def full_action(a):
    """Pad actions (..., 6 or 7) from players that predate flashbangs or reloading to (..., 8), the rest off."""
    if a.shape[-1] == ACTION_SIZE:
        return a
    return jnp.concatenate([a, -jnp.ones(a.shape[:-1] + (ACTION_SIZE - a.shape[-1],), a.dtype)], -1)


@dataclass(frozen=True)
class CQBConfig:
    # the building
    yard_w: float = 40.0
    yard_h: float = 30.0
    rooms: tuple = (3, 6)         # how many rooms
    room_area: tuple = (22.0, 40.0)  # the footprint is about this many m^2 per room
    aspect: tuple = (1.2, 1.7)    # footprint width / depth
    width: tuple = (12.0, 28.0)   # footprint limits, metres (east-west)
    depth: tuple = (8.0, 20.0)    # (north-south)
    min_room: float = 3.5         # a room is at least this big a side (wall line to wall line)
    max_room: float = 10.0        # and at most this big (split further, up to max_rooms)
    wall: float = 0.2             # wall thickness
    door: float = 1.0             # doorway width
    jamb: float = 0.3             # least wall between a doorway and a corner
    corner_fed: float = 0.75      # a doorway this close to a room's corner is corner-fed for that room
    outside_doors: tuple = (2, 3)
    cover: tuple = (0, 4)         # tables, crates, lockers
    cover_size: tuple = (0.6, 1.5)
    cover_gap: float = 0.9        # cover stays this far from walls and other cover (no dead ends)
    entry_dist: float = 2.5       # Assault starts this far outside its door
    spawn_points: int = 64        # Defense start points per building
    # padding (fixed shapes for JAX)
    max_walls: int = 20
    max_doors: int = 8
    max_cover: int = 4
    max_rooms: int = 6
    max_entries: int = 3
    # an operator
    radius: float = 0.3
    walk: float = 1.4             # m/s, quiet
    run: float = 3.5              # m/s, loud
    sideways: float = 0.7         # strafing and walking backwards: this much of the speed
    accel: float = 7.0            # m/s^2: full walking speed in 0.2 s
    turn_rate: float = 240.0      # degrees per second
    reach: float = 1.0            # interact with a door this close (to the doorway)
    door_time: float = 0.5        # seconds to open or close a door
    inside_open: float = 0.5      # chance an inside door starts open (outside doors start closed)
    # seeing (the building is dark: you see what your weapon light shows)
    fov: float = 90.0             # degrees, the light's cone
    sight: float = 20.0           # metres
    seen_grid: tuple = (8, 6)     # the "seen" map: the footprint in cols x rows cells
    cone_rays: int = 16           # the light is a fan of this many rays; a cell is seen if the ray towards it reaches it
    n_rays: int = 24              # distance lasers in a full circle, starting straight ahead
    # shooting (one hit and you're down)
    aim: float = 10.0             # degrees: the target must be this close to where you aim
    hit_close: float = 0.9        # chance per shot at point-blank...
    hit_falloff: float = 0.02     # ...minus this per metre
    move_penalty: float = 0.6     # x(1 - this) while moving at walking speed (and you can't fire running)
    turn_penalty: float = 0.5     # x(1 - this) while turning at full rate
    flash: float = 0.2            # seconds a muzzle flash shows the shooter to anyone with a clear line
    # sound (heard through walls: each wall or shut door in the way cuts the range by a third)
    hear_run: float = 10.0        # metres: running footsteps
    hear_walk: float = 4.0        # walking footsteps (standing still is silent)
    hear_door: float = 12.0       # a door starting to open or close
    hear_shot: float = 100.0      # a gunshot: the whole building
    through_wall: float = 2 / 3   # range kept per wall in the way
    still: float = 0.3            # m/s: slower than this is silent
    sound_memory: float = 0.5     # seconds a sound stays in mind
    sound_dirs: int = 16          # you hear roughly where it came from: one of this many directions
    # flashbangs (ep9 3t): thrown the way you face, they go off after a fuse and blind everyone close by with no
    # wall or shut door in between (the light fills the room: cover doesn't shield you, and a throw arcs over it)
    flashbangs: tuple = (0, 0)    # per round: (Assault, Defense). (0, 0) = the game before 3t, exactly
    throw: float = 6.0            # metres: how far one flies (stopping short of the first wall or shut door)
    fuse: float = 1.5             # seconds from the throw to the bang
    bang_range: float = 7.0       # metres
    blind_time: float = 2.0       # seconds you can't see anything (the owner's spec: anyone in line of sight)...
    blind_away: float = 1.0       # ...times this if it wasn't in your light's cone (1: facing away doesn't help)
    hear_bang: float = 100.0      # metres: the whole building
    # weapons (ep9 3am): one carbine each
    magazine: int = 0             # rounds per magazine; 0 = no magazines, no reloading, no recoil: the game before 3am
    spare_mags: int = 3           # rounds carried besides the loaded magazine: spare_mags x magazine
    reload_time: float = 2.5      # seconds; can't shoot meanwhile (can walk). Starts by itself when the magazine's empty
    recoil: float = 0.5           # a shot right after a shot hits x(1 - this); recovers over...
    recoil_recover: float = 0.3   # ...this many seconds (aimed shots at this pace keep full accuracy)
    hear_reload: float = 4.0      # metres: a magazine change, like walking
    # health (ep9 3ar): a hit takes health by where it lands, instead of one hit = down
    health: float = 0.0           # 0 = one hit and you're down: the game before 3ar
    head_share: float = 0.10      # where a hit lands: the head...
    torso_share: float = 0.55     # ...the torso, else an arm or a leg
    head_damage: float = 100.0
    torso_damage: float = 40.0
    limb_damage: float = 25.0
    wounded_below: float = 50.0   # under this much health: wounded...
    wounded_speed: float = 0.75   # ...moves at this much speed...
    wounded_aim: float = 0.75     # ...and hits this much as often
    # reaction time (ep9 3aw): a shooter who has just spotted the enemy isn't on target yet. The aim settles over
    # `react` seconds of seeing them (a shot hits x 1/3, 2/3, then fully at 0.3 s) and unsettles as fast out of sight
    react: float = 0.0            # seconds; 0 = no reaction time: the game before 3aw
    # teams (ep10): operators per side, (Assault, Defense). Operators are numbered Assault first, then Defense.
    # (1, 1) = the game before ep10, exactly. Assault pairs start stacked at one door, `stack_gap` apart
    team_sizes: tuple = (1, 1)
    stack_gap: float = 0.8
    # habits (ep10 step 9, owner OK): sprinting into walls was free and silent; real operators keep off walls and slow
    # down to shoot. Both off by default (the game before, exactly)
    bump: float = 0.0             # metres a collision is heard: hitting a wall, door or crate faster than walking
                                  # (speed straight into it), heard like footsteps (a third less per wall); 0 = off
    ready: float = 0.0            # seconds to bring the rifle up after running or a collision: shots hit x 1/3,
                                  # 2/3, then fully (at 0.3); 0 = off
    stun: float = 0.0             # seconds a hard collision (the same as a loud one) leaves you unable to move, turn or
                                  # fire (ep10 9f, owner OK: slamming into a wall with your kit on hurts); 0 = off
    # a round
    dt: float = 0.1               # seconds per decision
    time_limit: float = 45.0
    win_reward: float = 1.0       # rewards are Assault's; Defense gets the negative
    time_cost: float = 0.0        # Assault, per second. Without it Assault learns to wait outside: with
                                  # discounting, a loss at the time limit weighs ~0.1 of being shot now (ep9 3c)

    @property
    def obs_size(self) -> int:
        # lasers, the enemy (10), sounds (3 x 3), me (7), 2 nearest doors (2 x 4), the seen map, time left,
        # and with flashbangs 8 more at the end (so brains from before can be given the first 107)
        base = self.n_rays + 10 + 9 + 7 + 8 + self.seen_grid[0] * self.seen_grid[1] + 1
        return (base + (BANG_OBS if any(self.flashbangs) else 0) + (GUN_OBS if self.magazine else 0)
                + (HEALTH_OBS if self.health else 0) + (REACT_OBS if self.react else 0) + self.team_obs
                + (READY_OBS if self.ready else 0) + (STUN_OBS if self.stun else 0))

    @property
    def team_obs(self) -> int:
        """With teams: one more enemy slot per extra enemy and a slot per teammate (the biggest team decides, so both
        sides read the same size), and which operator of my team I am. Last, so earlier brains read theirs."""
        if tuple(self.team_sizes) == (1, 1):
            return 0
        extra = max(self.team_sizes) - 1
        return extra * (ENEMY_OBS + MATE_OBS) + 1


# ── Buildings (numpy: generated once, then handed to JAX as arrays) ──────────

def _split_rooms(rng, lo, hi, n, cfg):
    """Recursive splits: keep splitting a big room across its longer side until there are n
    rooms and none is over max_room a side (or there are max_rooms, or nothing can be split).
    Returns the rooms and the split lines (axis, c, a, b): axis 0 = a vertical wall at x=c from y=a to y=b, axis 1 = horizontal."""
    rooms, lines = [(lo.copy(), hi.copy())], []
    m = cfg.min_room
    while True:
        size = np.array([h - l for l, h in rooms])
        can = (size >= 2 * m).any(axis=1)
        too_big = can & (size.max(axis=1) > cfg.max_room)
        if not can.any() or (len(rooms) >= n and not too_big.any()) or len(rooms) >= cfg.max_rooms:
            break
        if too_big.any():                                   # the biggest room first
            i = int(np.argmax(np.where(too_big, size.max(axis=1), -1)))
        else:
            area = size.prod(axis=1) * can
            i = int(rng.choice(len(rooms), p=area / area.sum()))
        l, h = rooms.pop(i)
        sz = h - l
        ok = sz >= 2 * m
        axis = int(np.argmax(np.where(ok, sz, -1)))       # split the longer side that can be split
        c = l[axis] + np.round(rng.uniform(m, sz[axis] - m) * 2) / 2
        h1, l2 = h.copy(), l.copy()
        h1[axis], l2[axis] = c, c
        rooms += [(l, h1), (l2, h)]
        other = 1 - axis
        lines.append((axis, float(c), float(l[other]), float(h[other])))
    return rooms, lines


def _segments(axis, c, a, b, lines, eps=1e-6):
    """The pieces of the wall line (axis, c, a..b) between the walls that meet it."""
    cuts = [a, b]
    for ax2, c2, a2, b2 in lines:
        if ax2 != axis and a - eps <= c2 <= b + eps and (abs(a2 - c) < eps or abs(b2 - c) < eps):
            cuts.append(c2)   # a perpendicular wall ends on this line
    cuts = sorted(set(round(x, 6) for x in cuts))
    return [(s, e) for s, e in zip(cuts[:-1], cuts[1:]) if e - s > eps]


def _place_door(rng, segs, cfg):
    """A doorway on one of the segments: next to a corner or away from them, at random.
    Returns the doorway's (start, end) along the line."""
    w, j = cfg.door, cfg.jamb
    fits = [(s, e) for s, e in segs if e - s >= w + 2 * j]
    if not fits:                                            # never seen; keep the doorway anyway
        s, e = max(segs, key=lambda se: se[1] - se[0])
        d0 = (s + e - w) / 2
        return d0, d0 + w
    s, e = fits[int(rng.integers(len(fits)))]
    if rng.random() < 0.5:                                  # next to a corner
        d0 = s + j if rng.random() < 0.5 else e - j - w
    else:                                                   # away from the corners
        lo, hi = s + 1.2, e - 1.2 - w
        d0 = rng.uniform(lo, hi) if hi > lo else (s + e - w) / 2
    d0 = float(np.round(d0 * 10) / 10)
    return d0, d0 + w


def _wall_boxes(axis, c, a, b, gaps, t):
    """Boxes for the wall line (axis, c, a..b) with doorway gaps cut out. Lines are extended
    by half a wall at both ends so corners are filled."""
    pts = [a - t / 2] + [x for g in sorted(gaps) for x in g] + [b + t / 2]
    boxes = []
    for s, e in zip(pts[0::2], pts[1::2]):
        if e - s > 1e-6:
            boxes.append(_box(axis, c, s, e, t))
    return boxes


def _box(axis, c, s, e, t):
    """The box of a wall (or door) piece on line (axis, c) from s to e along the line."""
    if axis == 0:   # vertical line x = c
        return np.array([c - t / 2, s]), np.array([c + t / 2, e])
    return np.array([s, c - t / 2]), np.array([e, c + t / 2])


def _room_at(rooms, p):
    for k, (l, h) in enumerate(rooms):
        if (p > l).all() and (p < h).all():
            return k
    return -1


def make_building(seed: int, cfg: CQBConfig = CQBConfig()) -> dict:
    """A kill house: rooms, wall boxes, doors (with which rooms they join and whether each side
    is center- or corner-fed), cover, Assault's entry points and Defense's start points."""
    rng = np.random.default_rng(seed)
    t = cfg.wall
    n = int(rng.integers(cfg.rooms[0], cfg.rooms[1] + 1))
    area, aspect = n * rng.uniform(*cfg.room_area), rng.uniform(*cfg.aspect)
    size = np.array([np.sqrt(area * aspect), np.sqrt(area / aspect)])
    size = np.clip(np.round(size * 2) / 2, [cfg.width[0], cfg.depth[0]], [cfg.width[1], cfg.depth[1]])
    yard = np.array([cfg.yard_w, cfg.yard_h])
    slack = yard - size
    room_outside = cfg.entry_dist + 1.5                     # Assault's start stays inside the yard
    lo = np.round((slack / 2 + rng.uniform(-1, 1, 2) * np.maximum(slack / 2 - room_outside, 0)) * 2) / 2
    hi = lo + size

    rooms, lines = _split_rooms(rng, lo, hi, n, cfg)
    perimeter = [(0, lo[0], lo[1], hi[1]), (0, hi[0], lo[1], hi[1]),
                 (1, lo[1], lo[0], hi[0]), (1, hi[1], lo[0], hi[0])]
    every = lines + perimeter

    doors = []   # (axis, c, d0, d1, outside)
    gaps = {i: [] for i in range(len(every))}
    for i, (axis, c, a, b) in enumerate(lines):             # one doorway in every inside wall
        d0, d1 = _place_door(rng, _segments(axis, c, a, b, every), cfg)
        gaps[i].append((d0, d1))
        doors.append((axis, c, d0, d1, False))
    k = int(rng.integers(cfg.outside_doors[0], cfg.outside_doors[1] + 1))
    sides = rng.permutation(4)[:k]                          # outside doorways on different sides
    for sd in sides:
        i = len(lines) + int(sd)
        axis, c, a, b = every[i]
        d0, d1 = _place_door(rng, _segments(axis, c, a, b, every), cfg)
        gaps[i].append((d0, d1))
        doors.append((axis, c, d0, d1, True))

    wall_lo, wall_hi = [], []
    for i, (axis, c, a, b) in enumerate(every):
        for bl, bh in _wall_boxes(axis, c, a, b, gaps[i], t):
            wall_lo.append(bl)
            wall_hi.append(bh)

    door_info = []
    for axis, c, d0, d1, outside in doors:
        bl, bh = _box(axis, c, d0, d1, t)
        mid = (bl + bh) / 2
        normal = np.eye(2)[axis]
        sides_ = []
        for sgn in (-1, 1):                                 # the room on each side, and how it's fed
            r = _room_at(rooms, mid + sgn * normal * 0.5)
            if r < 0:
                sides_.append((-1, False))
                continue
            rl, rh = rooms[r]
            along = 1 - axis
            gap = min(d0 - rl[along], rh[along] - d1) - t / 2   # wall left between doorway and corner
            sides_.append((r, bool(gap < cfg.corner_fed)))
        door_info.append({"lo": bl, "hi": bh, "center": mid, "axis": axis, "outside": outside,
                          "rooms": (sides_[0][0], sides_[1][0]), "corner_fed": (sides_[0][1], sides_[1][1])})

    # cover: away from walls and from each other, so it never closes a way through
    cover_lo, cover_hi = [], []
    want = int(rng.integers(cfg.cover[0], cfg.cover[1] + 1))
    g = cfg.cover_gap + t / 2
    for _ in range(200):
        if len(cover_lo) == want:
            break
        rl, rh = rooms[int(rng.integers(len(rooms)))]
        wh = rng.uniform(*cfg.cover_size, 2)
        free = (rh - rl) - 2 * g - wh
        if (free < 0).any():
            continue
        bl = np.round((rl + g + rng.uniform(0, 1, 2) * free) * 10) / 10
        bh = bl + wh
        if all(np.hypot(*np.maximum(0, np.maximum(ol - bh, bl - oh))) >= cfg.cover_gap
               for ol, oh in zip(cover_lo, cover_hi)):
            cover_lo.append(bl)
            cover_hi.append(bh)

    # Assault: outside each outside door, facing it
    entries = []
    for d in door_info:
        if d["outside"]:
            r_in = d["rooms"][0] if d["rooms"][0] >= 0 else d["rooms"][1]
            out_sgn = -1 if d["rooms"][0] < 0 else 1
            normal = np.eye(2)[d["axis"]] * out_sgn
            p = d["center"] + normal * cfg.entry_dist
            entries.append((p, float(np.arctan2(-normal[1], -normal[0])), r_in))

    # Defense: a random room each time, a free spot in it
    spawn, spawn_room = [], []
    clear = 0.6
    order = rng.permutation(np.arange(cfg.spawn_points) % len(rooms))
    for r in order:
        rl, rh = rooms[r]
        for _ in range(100):
            p = rng.uniform(rl + t / 2 + clear, rh - t / 2 - clear)
            if all(np.hypot(*(p - np.clip(p, cl, ch))) > clear for cl, ch in zip(cover_lo, cover_hi)):
                break
        spawn.append(p)
        spawn_room.append(int(r))

    return {"seed": seed, "lo": lo, "hi": hi, "rooms": rooms, "lines": lines,
            "wall_lo": np.array(wall_lo), "wall_hi": np.array(wall_hi), "doors": door_info,
            "cover_lo": np.array(cover_lo).reshape(-1, 2), "cover_hi": np.array(cover_hi).reshape(-1, 2),
            "entries": entries, "spawn": np.array(spawn), "spawn_room": np.array(spawn_room)}


def cell_fracs(cfg: CQBConfig) -> np.ndarray:
    """Centres of the "seen" map's cells as fractions of the footprint, (cols * rows, 2), row by row."""
    gx, gy = cfg.seen_grid
    u = np.stack(np.meshgrid(np.arange(gx), np.arange(gy), indexing="xy"), -1).reshape(-1, 2) + 0.5
    return u / np.array([gx, gy])


def cell_points(b: dict, cfg: CQBConfig = CQBConfig(), clear: float = 0.45) -> np.ndarray:
    """The point of each "seen" map cell that has to be lit for the cell to count as seen: its
    centre, or, if that's inside (or hard against) a wall or cover, the nearest point of the cell
    where someone could stand (otherwise the cell could never be seen, or only from on top of a
    crate). (cells, 2) metres."""
    lo, hi = b["lo"], b["hi"]
    size = (hi - lo) / np.array(cfg.seen_grid)
    centres = lo + cell_fracs(cfg) * (hi - lo)
    blo = np.vstack([b["wall_lo"], b["cover_lo"]])
    bhi = np.vstack([b["wall_hi"], b["cover_hi"]])
    gap = lambda q: np.linalg.norm(q[:, None] - np.clip(q[:, None], blo[None], bhi[None]), axis=2).min(1)
    out = centres.copy()
    u = np.linspace(-0.45, 0.45, 19)
    offsets = np.stack(np.meshgrid(u, u), -1).reshape(-1, 2)
    for i in np.where(gap(centres) < clear)[0]:
        for reach in (1.0, 1.6):                            # in the cell, then a bit around it
            cand = np.clip(centres[i] + offsets * size * reach, lo + 0.05, hi - 0.05)
            g = gap(cand)
            if (g >= clear).any():
                ok = g >= clear
                out[i] = cand[ok][np.argmin(np.linalg.norm(cand[ok] - centres[i], axis=1))]
                break
        else:
            out[i] = cand[np.argmax(g)]                      # nowhere that clear: the most open spot
    return out


def room_graph(b: dict, cfg: CQBConfig = CQBConfig(), cells=None) -> dict:
    """For finding the way (scripts and analysis; the AI isn't given this). Nodes are the rooms and
    the outside (index max_rooms). next_door[i, j]: the door to go through from i to get to j
    (fewest doors; -1 if i == j); hops[i, j]: how many doors that is. cell_room: the room each "seen"
    map cell's centre is in (-1: outside the rooms)."""
    R = cfg.max_rooms
    n = len(b["rooms"])
    node = lambda r: R if r < 0 else r
    nbrs = {i: [] for i in list(range(n)) + [R]}
    for k, d in enumerate(b["doors"]):
        a, c = map(node, d["rooms"])
        nbrs[a].append((c, k))
        nbrs[c].append((a, k))
    next_door = np.full((R + 1, R + 1), -1, np.int32)
    hops = np.full((R + 1, R + 1), 99, np.int32)
    for src in nbrs:                                        # breadth-first from every node
        hops[src, src] = 0
        first = {src: -1}
        queue = [src]
        while queue:
            u = queue.pop(0)
            for v, k in nbrs[u]:
                if v not in first:
                    first[v] = k if u == src else first[u]
                    hops[src, v] = hops[src, u] + 1
                    queue.append(v)
        for v, k in first.items():
            next_door[src, v] = k
    cells = cell_points(b, cfg) if cells is None else cells
    cell_room = np.array([_room_at(b["rooms"], c) for c in cells], np.int32)
    return {"next_door": next_door, "hops": hops, "cell_room": cell_room}


def nav_graph(b: dict, cfg: CQBConfig = CQBConfig(), body: float = 0.32, cells=None) -> dict:
    """Waypoints for scripts (the AI isn't given this). Nodes, in this order: the "seen" map's cell
    points, a point 0.55 m out from each doorway on both sides (on its centre line), and a point off
    each corner of every piece of cover. Two nodes are joined when a body can walk straight from one
    to the other (doors count as open: scripts open them on the way). Returns nodes (N, 2), the
    door each doorway node belongs to (-1 for others), shortest walking distances (N, N) and the
    next node on each shortest path (N, N)."""
    D, C = cfg.max_doors, cfg.max_cover
    cells = cell_points(b, cfg) if cells is None else cells
    door_pts, door_of = [], []
    out = cfg.wall / 2 + cfg.cover_gap / 2                  # halfway to where cover may start
    for k in range(D):
        if k < len(b["doors"]):
            d = b["doors"][k]
            n = np.eye(2)[d["axis"]]
            door_pts += [d["center"] - n * out, d["center"] + n * out]
        else:
            door_pts += [np.full(2, FAR), np.full(2, FAR)]
        door_of += [k, k]
    m = cfg.cover_gap / 2                                   # halfway to a wall: the gap beside a crate has waypoints too
    corner_pts = []
    for c in range(C):
        if c < len(b["cover_lo"]):
            lo, hi = b["cover_lo"][c] - m, b["cover_hi"][c] + m
            corner_pts += [lo, hi, np.array([lo[0], hi[1]]), np.array([hi[0], lo[1]])]
        else:
            corner_pts += [np.full(2, FAR)] * 4
    nodes = np.vstack([cells, np.array(door_pts), np.array(corner_pts)])
    N = len(nodes)
    real = nodes[:, 0] > FAR / 2
    lo = np.vstack([b["wall_lo"], b["cover_lo"]]) - body
    hi = np.vstack([b["wall_hi"], b["cover_hi"]]) + body
    yard = np.array([cfg.yard_w, cfg.yard_h])
    real &= (nodes > cfg.radius).all(1) & (nodes < yard - cfg.radius).all(1)
    real &= ~(((nodes[:, None] > lo[None]) & (nodes[:, None] < hi[None])).all(2)).any(1)  # not in a wall or crate
    # straight-line clearance between every pair (slab test against the widened boxes)
    p, q = nodes[:, None, None, :], nodes[None, :, None, :]
    d = q - p
    inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    t1, t2 = (lo[None, None] - p) * inv, (hi[None, None] - p) * inv
    t_in, t_out = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
    blocked = ((t_in <= t_out) & (t_out >= 0) & (t_in <= 1)).any(-1)
    length = np.linalg.norm(nodes[:, None] - nodes[None], axis=2)
    dist = np.where(~blocked & real[:, None] & real[None], length, np.inf)
    np.fill_diagonal(dist, 0.0)
    nxt = np.where(np.isfinite(dist), np.arange(N)[None, :], -1)
    for k in range(N):                                      # Floyd-Warshall
        via = dist[:, k:k + 1] + dist[k:k + 1, :]
        better = via < dist
        dist = np.where(better, via, dist)
        nxt = np.where(better, nxt[:, k:k + 1], nxt)
    door_of = np.array([-1] * len(cells) + door_of + [-1] * len(corner_pts), np.int32)
    return {"nodes": nodes, "door_of": door_of, "dist": dist, "next": nxt.astype(np.int32), "real": real}


class Buildings(NamedTuple):
    wall_lo: jnp.ndarray      # (B, W, 2) wall boxes; unused slots are empty boxes far outside
    wall_hi: jnp.ndarray
    door_lo: jnp.ndarray      # (B, D, 2) door boxes (the doorway's gap in the wall)
    door_hi: jnp.ndarray
    door_valid: jnp.ndarray   # (B, D) is there a door in this slot
    door_center_fed: jnp.ndarray  # (B, D, 2) per side (-normal, +normal): center-fed?
    door_outside: jnp.ndarray     # (B, D)
    cover_lo: jnp.ndarray     # (B, C, 2)
    cover_hi: jnp.ndarray
    entry: jnp.ndarray        # (B, E, 2) where Assault starts
    entry_face: jnp.ndarray   # (B, E) facing the door (radians)
    n_entries: jnp.ndarray    # (B,)
    spawn: jnp.ndarray        # (B, N, 2) where Defense can start
    foot_lo: jnp.ndarray      # (B, 2) the building's footprint (for the "seen" map)
    foot_hi: jnp.ndarray
    cell_point: jnp.ndarray   # (B, cells, 2) the point of each map cell that must be lit (cell_points)
    # the floor plan, for scripts (the AI isn't given these; the game never reads them)
    room_lo: jnp.ndarray = None      # (B, R, 2) rooms by their wall lines; unused slots far outside
    room_hi: jnp.ndarray = None
    door_rooms: jnp.ndarray = None   # (B, D, 2) the room on each side (max_rooms = outside)
    next_door: jnp.ndarray = None    # (B, R + 1, R + 1) see room_graph
    hops: jnp.ndarray = None         # (B, R + 1, R + 1)
    cell_room: jnp.ndarray = None    # (B, cells)
    spawn_room: jnp.ndarray = None   # (B, N) the room each Defense start spot is in
    nav_nodes: jnp.ndarray = None    # (B, K, 2) waypoints (nav_graph): map cells, then doorways, then cover corners
    nav_door: jnp.ndarray = None     # (B, K) the door a doorway waypoint belongs to (-1 for others)
    nav_dist: jnp.ndarray = None     # (B, K, K) walking distance between waypoints (1e4: no way)
    nav_next: jnp.ndarray = None     # (B, K, K) next waypoint on the way


TRAIN_SEEDS = range(0, 1024)          # buildings it learns in
TEST_SEEDS = range(100_000, 100_256)  # buildings it has never seen: choose on the first 64, report on the rest


def _building_arrays(s: int, cfg: CQBConfig) -> dict:
    """One building, as the padded arrays make_buildings stacks (numpy; a top-level function so it can run in other
    processes)."""
    far = -1000.0

    def pad(x, n, fill=far, w=2):
        x = np.asarray(x, float).reshape(len(x), w)
        return np.vstack([x, np.full((n - len(x), x.shape[1]), fill)]) if len(x) < n else x

    b = make_building(s, cfg)
    D, E, R = cfg.max_doors, cfg.max_entries, cfg.max_rooms
    cells = cell_points(b, cfg)                                              # worked out once, used three times
    g, nav = room_graph(b, cfg, cells), nav_graph(b, cfg, cells=cells)
    dr = np.full((D, 2), R, np.int32)
    for k, d in enumerate(b["doors"]):
        dr[k] = [R if r < 0 else r for r in d["rooms"]]
    return {
        "wall_lo": pad(b["wall_lo"], cfg.max_walls), "wall_hi": pad(b["wall_hi"], cfg.max_walls),
        "door_lo": pad([d["lo"] for d in b["doors"]], D), "door_hi": pad([d["hi"] for d in b["doors"]], D),
        "door_valid": np.arange(D) < len(b["doors"]),
        "door_center_fed": pad([[not f for f in d["corner_fed"]] for d in b["doors"]], D, 0)[:, :2],
        "door_outside": np.array([d["outside"] for d in b["doors"]] + [False] * (D - len(b["doors"]))),
        "cover_lo": pad(b["cover_lo"], cfg.max_cover), "cover_hi": pad(b["cover_hi"], cfg.max_cover),
        "entry": pad([e[0] for e in b["entries"]], E, 0),
        "entry_face": np.array([e[1] for e in b["entries"]] + [0.0] * (E - len(b["entries"]))),
        "n_entries": len(b["entries"]), "spawn": b["spawn"], "foot_lo": b["lo"], "foot_hi": b["hi"],
        "cell_point": cells,
        "room_lo": pad([r[0] for r in b["rooms"]], R), "room_hi": pad([r[1] for r in b["rooms"]], R),
        "door_rooms": dr, "next_door": g["next_door"], "hops": g["hops"], "cell_room": g["cell_room"],
        "spawn_room": np.asarray(b["spawn_room"], np.int32),
        "nav_nodes": nav["nodes"], "nav_door": nav["door_of"],
        "nav_dist": np.where(np.isfinite(nav["dist"]), nav["dist"], 1e4), "nav_next": nav["next"],
    }


def buildings_key(seeds, cfg: CQBConfig = CQBConfig()) -> str:
    """A fingerprint of a set of buildings: the generator's own code, the settings it reads, and the seeds. A cached
    set is used only if this matches, so changing the generator or its settings never loads stale buildings."""
    import hashlib
    import inspect
    import re
    fns = [_split_rooms, _segments, _place_door, _wall_boxes, _box, _room_at, make_building, cell_fracs, cell_points,
           room_graph, nav_graph, _building_arrays]
    src = "".join(inspect.getsource(f) for f in fns)
    used = sorted(set(re.findall(r"cfg\.(\w+)", src)))
    blob = src + repr([(k, getattr(cfg, k)) for k in used]) + repr([int(x) for x in seeds])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_dirs() -> list:
    """Where cached building sets are looked for (the first one is written to): $AILEARNS_BUILDINGS (paths joined by
    ":"; Kaggle jobs point it at the cache dataset), else ~/.cache/ailearns/buildings."""
    import os
    from pathlib import Path
    env = os.environ.get("AILEARNS_BUILDINGS")
    return [Path(d) for d in env.split(":") if d] if env else [Path.home() / ".cache" / "ailearns" / "buildings"]


CACHE_MIN = 64   # sets smaller than this are quick to make: never cached


def make_buildings(seeds, cfg: CQBConfig = CQBConfig(), cache: bool = True) -> Buildings:
    """The buildings for `seeds`, as padded arrays. Sets of CACHE_MIN or more are cached (buildings_key: a cached set
    is the same as a fresh one) and, when made fresh, made in parallel on every CPU."""
    seeds = [int(x) for x in seeds]
    path = None
    if cache and len(seeds) >= CACHE_MIN:
        name = f"buildings-{buildings_key(seeds, cfg)}.npz"
        hit = next((d / name for d in _cache_dirs() if (d / name).exists()), None)
        if hit is not None:
            with np.load(hit) as z:
                return _to_buildings({k: z[k] for k in Buildings._fields})
        path = _cache_dirs()[0] / name
    if len(seeds) >= 128:
        import multiprocessing as mp
        import os
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(os.cpu_count(), mp_context=mp.get_context("spawn")) as ex:
            per = list(ex.map(_building_arrays, seeds, [cfg] * len(seeds), chunksize=16))
    else:
        per = [_building_arrays(s, cfg) for s in seeds]
    out = {k: np.stack([np.asarray(b[k]) for b in per]) for k in Buildings._fields}
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **out)
            tmp.replace(path)
        except OSError:
            pass                                                             # a read-only place: just don't cache
    return _to_buildings(out)


def _to_buildings(out: dict) -> Buildings:
    f = {k: jnp.asarray(v) for k, v in out.items()}
    f = {k: v.astype(jnp.float32) if v.dtype.kind == "f" else v for k, v in f.items()}
    f["door_center_fed"] = f["door_center_fed"] > 0.5
    return Buildings(**f)


# ── Geometry ─────────────────────────────────────────────────────────────────

def _push_out(p, v, lo, hi, r, iters: int = 3):
    """Move a disc out of the boxes it overlaps, deepest first, and stop its motion into them.
    Boxes meet at wall junctions, so a disc can touch two at once; resolving one at a time
    (instead of tag's sum) never pushes twice along the same face."""
    normals = jnp.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    lx, ly, hx, hy = lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1]      # x and y apart: the boxes innermost
    for _ in range(iters):
        dx, dy = p[0] - jnp.clip(p[0], lx, hx), p[1] - jnp.clip(p[1], ly, hy)  # from the closest point on each box
        dist = jnp.sqrt(dx * dx + dy * dy)
        inside = dist < 1e-6
        pen = jnp.stack([p[0] - lx, p[1] - ly, hx - p[0], hy - p[1]])        # (4, K) to faces -x -y +x +y
        depth = jnp.where(inside, jnp.min(pen, axis=0) + r, r - dist)
        k = jnp.argmax(depth)
        n = jnp.where(inside[k], normals[jnp.argmin(pen[:, k])],
                      jnp.stack([dx[k], dy[k]]) / jnp.maximum(dist[k], 1e-6))
        hit = depth[k] > 0
        p = p + jnp.where(hit, n * depth[k], 0.0)
        v = v - jnp.where(hit, n * jnp.minimum(jnp.dot(n, v), 0.0), 0.0)
    return p, v


def _slab(p, d, lo, hi):
    """Ray p + t*d against boxes: entry and exit t per box (hit if t_in <= t_out).
    x and y are done separately so the innermost axis is the boxes (much faster on a CPU)."""
    inv = 1.0 / jnp.where(jnp.abs(d) < 1e-9, 1e-9, d)
    ix, iy = inv[..., 0], inv[..., 1]
    x1, x2 = (lo[..., 0] - p[..., 0]) * ix, (hi[..., 0] - p[..., 0]) * ix
    y1, y2 = (lo[..., 1] - p[..., 1]) * iy, (hi[..., 1] - p[..., 1]) * iy
    return (jnp.maximum(jnp.minimum(x1, x2), jnp.minimum(y1, y2)),
            jnp.minimum(jnp.maximum(x1, x2), jnp.maximum(y1, y2)))


def _clear(p, q, lo, hi):
    """Is the straight line from p to each point q (N, 2) free of the boxes? -> (N,)"""
    t_in, t_out = _slab(p, (q - p)[:, None, :], lo, hi)
    return ~((t_in <= t_out) & (t_out >= 0) & (t_in <= 1)).any(-1)


def _cast(p, d, lo, hi, cfg):
    """Distance from p along each direction d (R, 2) to the first box or the yard's edge, capped at sight."""
    yard = jnp.array([cfg.yard_w, cfg.yard_h])
    safe = jnp.where(jnp.abs(d) < 1e-9, 1e-9, d)
    edge = jnp.min(jnp.where(jnp.abs(d) < 1e-9, jnp.inf, jnp.where(d > 0, yard - p, -p) / safe), axis=1)
    t_in, t_out = _slab(p, d[:, None, :], lo, hi)
    hit = (t_in <= t_out) & (t_out > 0)
    box = jnp.min(jnp.where(hit, jnp.maximum(t_in, 0.0), jnp.inf), axis=1)
    return jnp.minimum(jnp.minimum(edge, box), cfg.sight)


def _rot(v, a):
    """Rotate vector v by angle a (counter-clockwise)."""
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def _wrap(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


# ── The game (JAX) ───────────────────────────────────────────────────────────
# N operators, Assault first (operators 0..nA-1), then Defense. Tables "i, j" are about operator j as operator i
# knows it (sight, memory, aim); with teams, memory is the team's (a sighting by anyone on it is called out).

class State(NamedTuple):
    building: jax.Array   # which building
    pos: jax.Array        # (N, 2) metres
    vel: jax.Array        # (N, 2) m/s
    face: jax.Array       # (N,) radians, where the weapon (and its light) points
    turn: jax.Array       # (N,) last turn, -1..1
    door_open: jax.Array  # (D,)
    door_timer: jax.Array  # (D,) decisions until a door being moved finishes (0 = still)
    door_event: jax.Array  # (N,) did this operator start moving a door this decision (a sound)
    door_at: jax.Array    # (N, 2) the middle of the door it started moving (where that sound comes from)
    seen: jax.Array       # (N, N) does i see j right now (j alive, i not blinded)
    mem: jax.Array        # (N, N, 2) where i's team last saw enemy j
    mem_vel: jax.Array    # (N, N, 2) and how it was moving then
    since: jax.Array      # (N, N) decisions since then (T = never)
    cleared: jax.Array    # (N, cells) the "seen" map: which cells of the building each has looked at
    sound: jax.Array      # (N, 3, 3) per listener, per kind (steps, door, shot): direction (x, y) and loudness 0..1
    sound_age: jax.Array  # (N, 3) decisions since that sound was heard
    alive: jax.Array      # (N,)
    fired: jax.Array      # (N,) fired this decision (a sound)
    flash: jax.Array      # (N,) decisions left of this operator's muzzle flash
    flash_seen: jax.Array  # (N, N) i sees enemy j's muzzle flash right now
    bangs: jax.Array      # (N,) flashbangs left
    bang_pos: jax.Array   # (N, 2) where each one's live flashbang lies
    bang_timer: jax.Array  # (N,) decisions until it goes off (0 = none live)
    bang_now: jax.Array   # (N,) this one's flashbang went off this decision
    blind: jax.Array      # (N,) decisions left that this operator can't see
    bang_heard: jax.Array  # (N, 3) an enemy's last bang: direction (x, y) and loudness 0..1
    bang_age: jax.Array   # (N,) decisions since it was heard
    mag: jax.Array        # (N,) rounds in the magazine
    reserve: jax.Array    # (N,) rounds carried besides it
    reload_timer: jax.Array  # (N,) decisions until the reload is done (0 = not reloading)
    reload_now: jax.Array  # (N,) started reloading this decision (a sound)
    kick: jax.Array       # (N,) recoil left from the last shot, 1 -> 0
    reload_heard: jax.Array  # (N, 3) an enemy's last reload: direction (x, y) and loudness 0..1
    reload_age: jax.Array  # (N,) decisions since it was heard
    hp: jax.Array         # (N,) health left
    landed: jax.Array     # (N,) decisions since this operator's last shot hit (sound_steps or more: not lately)
    aim: jax.Array        # (N, N) how settled i's aim on j is: +1 per decision in sight, -1 out of it,
                          # 0..react_steps (always 0 without a reaction time)
    known_down: jax.Array  # (N, N) i's team knows enemy j is down (saw it go down, or hit it)
    hit_by: jax.Array     # (N,) who last hit this operator (-1: nobody yet); for counting and footage, not observed
    bumped: jax.Array     # (N,) ran into a wall, door or crate this decision (with `bump`: a sound)
    ready_t: jax.Array    # (N,) decisions since this operator last ran or bumped, up to ready_steps (with `ready`)
    stun_t: jax.Array     # (N,) decisions of stun left (with `stun`)
    key: jax.Array        # randomness for hits (in the state, so a recorded round replays exactly)
    t: jax.Array          # decisions taken
    done: jax.Array
    outcome: jax.Array    # +1 Assault won, -1 Defense won, 0 not over (or both down at once)


class CQBGame:
    def __init__(self, buildings: Buildings, cfg: CQBConfig = CQBConfig()):
        self.b, self.cfg = buildings, cfg
        self.door_steps = int(round(cfg.door_time / cfg.dt))
        self.T = int(round(cfg.time_limit / cfg.dt))
        self.cell_frac = jnp.asarray(cell_fracs(cfg), jnp.float32)          # cell centres, 0..1 of the footprint
        self.cos_half = float(np.cos(np.deg2rad(cfg.fov / 2)))
        self.cos_aim = float(np.cos(np.deg2rad(cfg.aim)))
        self.flash_steps = int(round(cfg.flash / cfg.dt))
        ang = jnp.arange(cfg.n_rays) * 2 * jnp.pi / cfg.n_rays
        self.ray_dirs = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)            # in the operator's own frame
        half = np.deg2rad(cfg.fov / 2)
        self.cone_angles = jnp.linspace(-half, half, cfg.cone_rays)            # the light's fan, own frame
        self.sound_steps = int(round(cfg.sound_memory / cfg.dt))
        self.bangs_on = any(cfg.flashbangs)     # without them none of the flashbang work is even compiled
        self.fuse_steps = int(round(cfg.fuse / cfg.dt))
        self.blind_steps = int(round(cfg.blind_time / cfg.dt))
        self.guns_on = cfg.magazine > 0         # likewise magazines, reloading and recoil
        self.reload_steps = int(round(cfg.reload_time / cfg.dt))
        self.health_on = cfg.health > 0         # likewise health, damage and wounds
        self.react_on = cfg.react > 0           # likewise the reaction time
        self.react_steps = max(int(round(cfg.react / cfg.dt)), 1)
        self.bump_on = cfg.bump > 0             # likewise loud collisions...
        self.ready_on = cfg.ready > 0           # ...and the rifle coming up after running
        self.ready_steps = max(int(round(cfg.ready / cfg.dt)), 1)
        self.stun_on = cfg.stun > 0
        self.stun_steps = max(int(round(cfg.stun / cfg.dt)), 1)
        # teams (ep10): who's who, all static
        self.nA, self.nD = (int(n) for n in cfg.team_sizes)
        self.N = self.nA + self.nD
        self.teams_on = (self.nA, self.nD) != (1, 1)
        self.team = np.array([ASSAULT] * self.nA + [DEFENSE] * self.nD)     # each operator's side
        self.enemies = [[j for j in range(self.N) if self.team[j] != self.team[i]] for i in range(self.N)]
        self.mates = [[j for j in range(self.N) if self.team[j] == self.team[i] and j != i] for i in range(self.N)]
        self.is_enemy = jnp.asarray(self.team[:, None] != self.team[None, :])
        self.same_team = jnp.asarray(self.team[:, None] == self.team[None, :])
        self.slots = max(self.nA, self.nD)                                   # enemy (and teammate + 1) slots

    def side_ops(self, side: int) -> list:
        """The operators on a side (ASSAULT or DEFENSE)."""
        return list(range(self.nA)) if side == ASSAULT else list(range(self.nA, self.N))

    def sides_out(self, alive):
        """alive (..., N) -> is Assault out, is Defense out (...): all of a side down."""
        return ~alive[..., :self.nA].any(-1), ~alive[..., self.nA:].any(-1)

    def init(self, key, building=None) -> State:
        """Assault outside a random outside door, facing it (a team stacked one behind the other); Defense on random
        spots inside, facing random ways. Outside doors closed; inside doors open or closed at random."""
        kb, ke, ks, kf, kd, key = jax.random.split(key, 6)
        cfg, b, N = self.cfg, self.b, self.N
        bi = jax.random.randint(kb, (), 0, b.wall_lo.shape[0]) if building is None else jnp.asarray(building)
        e = jax.random.randint(ke, (), 0, b.n_entries[bi])
        face_a = b.entry_face[bi, e]
        back = -jnp.stack([jnp.cos(face_a), jnp.sin(face_a)]) * cfg.stack_gap
        a_pos = [b.entry[bi, e] + back * k if k else b.entry[bi, e] for k in range(self.nA)]
        if self.nD == 1:                                                     # the draws of the game before teams
            d_pos = [b.spawn[bi, jax.random.randint(ks, (), 0, cfg.spawn_points)]]
            d_face = [jax.random.uniform(kf, (), minval=-jnp.pi, maxval=jnp.pi)]
        else:
            d_pos = list(b.spawn[bi, jax.random.choice(ks, cfg.spawn_points, (self.nD,), replace=False)])
            d_face = list(jax.random.uniform(kf, (self.nD,), minval=-jnp.pi, maxval=jnp.pi))
        pos = jnp.stack(a_pos + d_pos)
        if self.nA > 1:                                                      # the stack stays in the yard
            pos = jnp.clip(pos, cfg.radius, jnp.array([cfg.yard_w, cfg.yard_h]) - cfg.radius)
        face = jnp.stack([face_a] * self.nA + d_face)
        D = b.door_lo.shape[1]
        door_open = (jax.random.uniform(kd, (D,)) < cfg.inside_open) & ~b.door_outside[bi] & b.door_valid[bi]
        z = jnp.asarray(0)
        cells = self.cell_frac.shape[0]
        n1 = jnp.zeros(N, jnp.int32)
        st = State(building=bi, pos=pos, vel=jnp.zeros((N, 2)), face=face, turn=jnp.zeros(N), door_open=door_open,
                   door_timer=jnp.zeros(D, jnp.int32), door_event=jnp.zeros(N, bool), door_at=pos,
                   seen=jnp.zeros((N, N), bool), mem=jnp.zeros((N, N, 2)), mem_vel=jnp.zeros((N, N, 2)),
                   since=jnp.full((N, N), self.T), cleared=jnp.zeros((N, cells), bool), sound=jnp.zeros((N, 3, 3)),
                   sound_age=jnp.full((N, 3), self.sound_steps, jnp.int32), alive=jnp.ones(N, bool),
                   fired=jnp.zeros(N, bool), flash=n1, flash_seen=jnp.zeros((N, N), bool),
                   bangs=jnp.asarray([cfg.flashbangs[s] for s in self.team], jnp.int32), bang_pos=pos, bang_timer=n1,
                   bang_now=jnp.zeros(N, bool), blind=n1, bang_heard=jnp.zeros((N, 3)),
                   bang_age=jnp.full(N, self.sound_steps, jnp.int32),
                   mag=jnp.full(N, cfg.magazine, jnp.int32), reserve=jnp.full(N, cfg.magazine * cfg.spare_mags, jnp.int32),
                   reload_timer=n1, reload_now=jnp.zeros(N, bool), kick=jnp.zeros(N), reload_heard=jnp.zeros((N, 3)),
                   reload_age=jnp.full(N, self.sound_steps, jnp.int32), hp=jnp.full(N, cfg.health),
                   landed=jnp.full(N, self.sound_steps, jnp.int32), aim=jnp.zeros((N, N), jnp.int32),
                   known_down=jnp.zeros((N, N), bool), hit_by=jnp.full(N, -1, jnp.int32), bumped=jnp.zeros(N, bool),
                   ready_t=jnp.full(N, self.ready_steps, jnp.int32), stun_t=jnp.zeros(N, jnp.int32), key=key, t=z,
                   done=jnp.asarray(False), outcome=z)
        return self._look(st)

    def init_stacked(self, key, building=None) -> State:
        """A drill start (ep9 3y, training only): a normal start, then Assault is moved to beside the doorway of
        Defense's room (the first door on the way out of it, the one a camper watches), on the side with room to
        stand, 0.5 m out from the wall and 0.9 m along it (out of line with most of the room), facing the doorway
        (a team: stacked behind the first, away from the doorway); that door open half the time; the rest of the
        house counted as already looked at. Assault's job from here: the entry. Uses the floor plan (like the
        scripts); the game itself never calls it."""
        k_init, k_door = jax.random.split(key)
        st = self.init(k_init, building)
        cfg, b, bi, R = self.cfg, self.b, st.building, self.cfg.max_rooms
        d = st.pos[self.nA]                                                     # (the first) Defense
        inside = (d > b.room_lo[bi]).all(1) & (d < b.room_hi[bi]).all(1)
        r = jnp.argmax(inside)                                                  # Defense's room
        k = jnp.maximum(b.next_door[bi, r, R], 0)
        dlo, dhi = b.door_lo[bi, k], b.door_hi[bi, k]
        mid = (dlo + dhi) / 2
        axis = jnp.argmin(dhi - dlo)
        into = jnp.zeros(2).at[axis].set(jnp.where(b.door_rooms[bi, k, 1] == r, 1.0, -1.0))   # into Defense's room
        across = jnp.stack([-into[1], into[0]])
        back = mid - into * 0.5
        lo, hi = self.boxes(st)
        free = _cast(back, jnp.stack([across, -across]), lo, hi, cfg)
        side = jnp.where(free[0] >= free[1], across, -across)
        spot = back + side * jnp.clip(free.max() - cfg.radius - 0.05, 0.0, 0.9)
        spot, _ = _push_out(spot, jnp.zeros(2), lo, hi, cfg.radius)            # tight corners: out of the walls
        to_door = mid - spot
        open_ = jax.random.uniform(k_door) < 0.5
        pos, door_at, face, cleared = st.pos, st.door_at, st.face, st.cleared
        away = -to_door / jnp.maximum(jnp.linalg.norm(to_door), 1e-6)
        for n in range(self.nA):
            s = spot if n == 0 else _push_out(spot + away * cfg.stack_gap * n, jnp.zeros(2), lo, hi, cfg.radius)[0]
            to = mid - s
            pos, door_at = pos.at[n].set(s), door_at.at[n].set(s)
            face = face.at[n].set(jnp.arctan2(to[1], to[0]))
            if b.cell_room is not None:
                cleared = cleared.at[n].set(b.cell_room[bi] != r)
        N = self.N
        st = st._replace(pos=pos, door_at=door_at, face=face, door_open=st.door_open.at[k].set(open_), cleared=cleared)
        return self._look(st._replace(seen=jnp.zeros((N, N), bool), since=jnp.full((N, N), self.T),
                                      aim=jnp.zeros((N, N), jnp.int32)))

    def boxes(self, st: State, door_open=None):
        """Everything solid right now: walls, cover and closed doors (open doors parked far away)."""
        b, bi = self.b, st.building
        shut = (~(st.door_open if door_open is None else door_open) & b.door_valid[bi])[:, None]
        lo = jnp.concatenate([b.wall_lo[bi], b.cover_lo[bi], jnp.where(shut, b.door_lo[bi], FAR)])
        hi = jnp.concatenate([b.wall_hi[bi], b.cover_hi[bi], jnp.where(shut, b.door_hi[bi], FAR)])
        return lo, hi

    def in_cone(self, p, face, q):
        """Is each point q (N, 2) inside the light cone from p facing `face` (ignoring walls)?"""
        d = q - p
        dist = jnp.linalg.norm(d, axis=-1)
        ahead = d @ jnp.stack([jnp.cos(face), jnp.sin(face)]) >= self.cos_half * dist
        return ahead & (dist <= self.cfg.sight)

    def body(self, p, q):
        """3 points across the body at q, as seen from p (centre and both edges)."""
        d = q - p
        n = jnp.stack([-d[1], d[0]]) / jnp.maximum(jnp.linalg.norm(d), 1e-6) * self.cfg.radius
        return jnp.stack([q, q + n, q - n])

    def _other(self, who: int, target):
        if target is not None:
            return target
        assert self.N == 2, "with teams, say which operator"
        return 1 - who

    def sees(self, st: State, who: int, lo=None, hi=None, target=None):
        """Does `who` see `target` (1v1: the other one): any of 3 points across its body in the cone and in the clear
        (and `who` isn't blinded by a flashbang)."""
        if lo is None:
            lo, hi = self.boxes(st)
        j = self._other(who, target)
        p, pts = st.pos[who], self.body(st.pos[who], st.pos[j])
        return (self.in_cone(p, st.face[who], pts) & _clear(p, pts, lo, hi)).any() & (st.blind[who] == 0)

    def cells(self, building):
        """The points of the "seen" map's cells that must be lit (cell centres, moved out of walls and cover)."""
        return self.b.cell_point[building]

    def _team_any(self, x):
        """(N, N) bools -> (N, N): does anyone on i's team have it about j."""
        return (self.same_team[:, :, None] & x[None, :, :]).any(1) if self.teams_on else x

    def _sight(self, st: State, lo, hi):
        """(N, N): who sees whom (only the living see, and are seen; nobody sees themselves)."""
        rows = []
        for i in range(self.N):
            row = [self.sees(st, i, lo, hi, j) if j != i else jnp.asarray(False) for j in range(self.N)]
            rows.append(jnp.stack(row))
        seen = jnp.stack(rows)
        return seen & st.alive[:, None] & st.alive[None, :] if self.teams_on else seen

    def _look(self, st: State) -> State:
        """Who sees whom, the team's memory of each enemy, and the cells each has now looked at."""
        lo, hi = self.boxes(st)
        seen = self._sight(st, lo, hi)
        spotted = self._team_any(seen) & self.is_enemy                          # (N, N) someone on my team sees j
        mem = jnp.where(spotted[..., None], st.pos[None], st.mem)
        mem_vel = jnp.where(spotted[..., None], st.vel[None], st.mem_vel)
        since = jnp.where(spotted, 0, jnp.minimum(st.since + 1, self.T))
        c = self.cells(st.building)
        look = jax.vmap(lambda p, f: self.lit(p, f, c, lo, hi))(st.pos, st.face) & (st.blind == 0)[:, None]
        st = st._replace(seen=seen, mem=mem, mem_vel=mem_vel, since=since, cleared=st.cleared | look)
        if self.react_on:
            st = st._replace(aim=jnp.where(seen, jnp.minimum(st.aim + 1, self.react_steps), jnp.maximum(st.aim - 1, 0)))
        return st

    def light(self, p, face, lo, hi):
        """How far each ray of the light's fan reaches (metres, capped at `sight`): the shape of
        the lit area, used for the "seen" map (and for drawing the fog)."""
        ang = face + self.cone_angles
        return _cast(p, jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1), lo, hi, self.cfg)

    def lit(self, p, face, q, lo, hi):
        """Is each point q (N, 2) lit: in the cone and no further than the fan's ray towards it?"""
        reach = self.light(p, face, lo, hi)
        d = q - p
        dist = jnp.linalg.norm(d, axis=-1)
        rel = _wrap(jnp.arctan2(d[:, 1], d[:, 0]) - face)
        step = (self.cone_angles[-1] - self.cone_angles[0]) / (self.cfg.cone_rays - 1)
        k = jnp.clip(jnp.round((rel - self.cone_angles[0]) / step).astype(jnp.int32), 0, self.cfg.cone_rays - 1)
        return (jnp.abs(rel) <= self.cone_angles[-1] + 1e-6) & (dist <= reach[k])

    def _move(self, st: State, a):
        """Turn, accelerate towards the asked-for velocity, collide. -> position, velocity, facing, and how hard each
        hit a wall, door or crate (m/s)."""
        cfg, N = self.cfg, self.N
        face = _wrap(st.face + jnp.clip(a[:, TURN], -1, 1) * jnp.deg2rad(cfg.turn_rate) * cfg.dt)
        m = jnp.clip(a[:, :2], -1, 1)
        m = m / jnp.maximum(jnp.linalg.norm(m, axis=1, keepdims=True), 1.0)
        m = m * jnp.stack([jnp.where(m[:, 0] > 0, 1.0, cfg.sideways), jnp.full(N, cfg.sideways)], 1)
        speed = jnp.where(a[:, RUN] > 0, cfg.run, cfg.walk)
        if self.health_on:
            speed = speed * jnp.where(st.hp < cfg.wounded_below, cfg.wounded_speed, 1.0)
        want = _rot(m, face) * speed[:, None]
        dv = want - st.vel
        dv = dv * jnp.minimum(1.0, cfg.accel * cfg.dt / jnp.maximum(jnp.linalg.norm(dv, axis=1, keepdims=True), 1e-9))
        vel = st.vel + dv
        pos = st.pos + vel * cfg.dt

        # operators can't overlap: each overlapping pair is pushed apart, half each (the fallen don't count)
        moves = [[] for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                gap = pos[j] - pos[i]
                dist = jnp.linalg.norm(gap)
                overlap = jnp.maximum(2 * cfg.radius - dist, 0.0)
                if self.teams_on:
                    overlap = jnp.where(st.alive[i] & st.alive[j], overlap, 0.0)
                push = gap / jnp.maximum(dist, 1e-6) * overlap / 2
                moves[i].append(-push)
                moves[j].append(push)
        pos = pos + jnp.stack([sum(m[1:], m[0]) for m in moves])

        r = cfg.radius
        yard = jnp.array([cfg.yard_w, cfg.yard_h])
        clipped = jnp.clip(pos, r, yard - r)
        vel = jnp.where(clipped != pos, 0.0, vel)
        lo, hi = self.boxes(st)
        before = vel
        pos, vel = jax.vmap(_push_out, in_axes=(0, 0, None, None, None))(clipped, vel, lo, hi, r)
        impact = jnp.linalg.norm(before - vel, axis=1)   # the speed walls took away (straight into them, not along)
        return pos, vel, face, impact

    def _doors(self, st: State, a, pos):
        """Interact: start opening (or closing) the nearest door within reach. A door finishes
        moving door_time later; a door can't close on someone standing in the doorway."""
        cfg, b, bi = self.cfg, self.b, st.building
        dlo, dhi = b.door_lo[bi], b.door_hi[bi]                                 # (D, 2)
        dist = jnp.linalg.norm(pos[:, None] - jnp.clip(pos[:, None], dlo[None], dhi[None]), axis=-1)  # (N, D)
        dist = jnp.where(b.door_valid[bi][None], dist, jnp.inf)
        near = jnp.argmin(dist, axis=1)
        wants = (a[:, INTERACT] > 0) & (dist[jnp.arange(self.N), near] <= cfg.reach)
        free = st.door_timer[near] == 0
        start = wants & free                                                    # (N,)
        hit = jax.nn.one_hot(near, dlo.shape[0], dtype=bool) & start[:, None]   # (N, D)
        timer = jnp.where(hit.any(0), self.door_steps, st.door_timer)
        timer = jnp.maximum(timer - 1, 0)
        finishing = (st.door_timer == 1) | (hit.any(0) & (self.door_steps == 1))
        inway = jnp.linalg.norm(pos[:, None] - jnp.clip(pos[:, None], dlo[None], dhi[None]), axis=-1) < cfg.radius
        blocked = (inway & st.alive[:, None]).any(0) if self.teams_on else inway.any(0)   # someone in the doorway
        flip = finishing & ~(st.door_open & blocked)
        return st.door_open ^ flip, timer, start, (dlo[near] + dhi[near]) / 2

    def on_aim(self, st: State, who: int, target):
        """Is `target` within the aim angle of where `who` points (ignoring walls)?"""
        d = st.pos[target] - st.pos[who]
        dist = jnp.linalg.norm(d)
        return d @ jnp.stack([jnp.cos(st.face[who]), jnp.sin(st.face[who])]) >= self.cos_aim * dist

    def hit_chance(self, st: State, who: int, lo=None, hi=None, target=None):
        """The chance a shot from `who` right now hits `target` (1v1: the other one; 0 if it can't hit)."""
        cfg = self.cfg
        j = self._other(who, target)
        d = st.pos[j] - st.pos[who]
        dist = jnp.linalg.norm(d)
        on_aim = d @ jnp.stack([jnp.cos(st.face[who]), jnp.sin(st.face[who])]) >= self.cos_aim * dist
        p = jnp.clip(cfg.hit_close - cfg.hit_falloff * dist, 0.05, 1.0)
        p = p * (1 - cfg.move_penalty * jnp.minimum(jnp.linalg.norm(st.vel[who]) / cfg.walk, 1.0))
        p = p * (1 - cfg.turn_penalty * jnp.abs(st.turn[who]))
        if self.health_on:
            p = p * jnp.where(st.hp[who] < cfg.wounded_below, cfg.wounded_aim, 1.0)
        if self.react_on:
            p = p * st.aim[who, j] / self.react_steps
        if self.ready_on:
            p = p * st.ready_t[who] / self.ready_steps
        return jnp.where(on_aim & self.sees(st, who, lo, hi, j), p, 0.0)

    def targets(self, st: State, lo=None, hi=None):
        """Where each operator's shot would go: the first body on its aim line, the nearest living operator within the
        aim angle that it sees, teammates included (friendly fire). -> (target (N,), hit chance (N,)); with nobody
        there the chance is 0. 1v1: always the other one."""
        if lo is None:
            lo, hi = self.boxes(st)
        if not self.teams_on:
            return jnp.asarray([1, 0]), jnp.stack([self.hit_chance(st, w, lo, hi) for w in (ASSAULT, DEFENSE)])
        tgt, chance = [], []
        for i in range(self.N):
            others = [j for j in range(self.N) if j != i]
            dist = jnp.stack([jnp.linalg.norm(st.pos[j] - st.pos[i]) for j in others])
            line = jnp.stack([self.on_aim(st, i, j) & st.seen[i, j] for j in others])
            k = jnp.argmin(jnp.where(line, dist, jnp.inf))
            p = jnp.stack([self.hit_chance(st, i, lo, hi, j) for j in others])
            tgt.append(jnp.asarray(others)[k])
            chance.append(jnp.where(line.any(), p[k], 0.0))
        return jnp.stack(tgt), jnp.stack(chance)

    def _reload(self, st: State, a) -> State:
        """A reload in progress ticks down and refills the magazine from the rounds carried when it's done; a new one
        starts on the reload switch (if the magazine isn't full) or by itself when it's empty, if there are rounds
        left. Recoil wears off."""
        cfg = self.cfg
        timer = jnp.maximum(st.reload_timer - 1, 0)
        done = st.reload_timer == 1
        take = jnp.where(done, jnp.minimum(cfg.magazine - st.mag, st.reserve), 0)
        mag, reserve = st.mag + take, st.reserve - take
        want = a[:, RELOAD] > 0 if a.shape[-1] > RELOAD else jnp.zeros(self.N, bool)
        start = (want | (mag == 0)) & (mag < cfg.magazine) & (reserve > 0) & (timer == 0) & st.alive & ~st.done
        return st._replace(mag=mag, reserve=reserve, reload_timer=jnp.where(start, self.reload_steps, timer),
                           reload_now=start,
                           kick=jnp.maximum(st.kick - cfg.dt / cfg.recoil_recover, 0.0))

    def _shoot(self, st: State, a) -> State:
        """Everyone may fire in the same decision (several can go down). A shot lights up the shooter. With magazines:
        only with a round loaded and not reloading; each shot uses one and kicks (the next shot soon after hits less).
        A shot goes to the first body on the aim line (targets), teammates included."""
        N = self.N
        lo, hi = self.boxes(st)
        fire = (a[:, FIRE] > 0) & (a[:, RUN] <= 0) & st.alive & ~st.done
        if self.guns_on:
            fire = fire & (st.mag > 0) & (st.reload_timer == 0)
        key, k = jax.random.split(st.key)
        tgt, p = self.targets(st, lo, hi)
        if self.guns_on:
            p = p * (1 - self.cfg.recoil * st.kick)
        hits = fire & (jax.random.uniform(k, (N,)) < p)                 # hits[i]: i hit its target
        H = jax.nn.one_hot(tgt, N, dtype=bool) & hits[:, None]          # (N shooter, N hit)
        if self.health_on:   # where each hit lands, and what it takes
            cfg = self.cfg
            key, kw = jax.random.split(key)
            where = jax.random.uniform(kw, (N,))
            damage = jnp.where(where < cfg.head_share, cfg.head_damage,
                               jnp.where(where < cfg.head_share + cfg.torso_share, cfg.torso_damage, cfg.limb_damage))
            hp = st.hp - jnp.where(H, damage[:, None], 0.0).sum(0)
            alive = st.alive & (hp > 0)
            st = st._replace(hp=jnp.maximum(hp, 0.0), landed=jnp.where(hits, 0, jnp.minimum(st.landed + 1, self.sound_steps)))
        else:
            alive = st.alive & ~H.any(0)
        flash = jnp.where(fire, self.flash_steps, jnp.maximum(st.flash - 1, 0))
        # who sees an enemy's flash: a clear line to it, light cone or not
        rows = []
        for w in range(N):
            rows.append(jnp.stack([(flash[j] > 0) & _clear(st.pos[w], st.pos[j][None], lo, hi)[0] & (st.blind[w] == 0)
                                   & (st.alive[w] if self.teams_on else True)
                                   if j in self.enemies[w] else jnp.asarray(False) for j in range(N)]))
        sees_flash = jnp.stack(rows)
        spotted = self._team_any(sees_flash)
        mem = jnp.where(spotted[..., None], st.pos[None], st.mem)
        since = jnp.where(spotted, 0, st.since)
        if self.teams_on:   # who's down, as each team knows it: it went down in sight of the team, or to its shot
            fell = st.alive & ~alive
            news = (self._team_any(st.seen) | self._team_any(H)) & fell[None, :] & self.is_enemy
            st = st._replace(known_down=st.known_down | news)
        hit_by = jnp.where(H.any(0), jnp.argmax(H, axis=0), st.hit_by)
        st = st._replace(alive=alive, fired=fire, flash=flash, flash_seen=sees_flash, mem=mem, since=since, key=key,
                         hit_by=hit_by)
        if self.guns_on:
            st = st._replace(mag=st.mag - fire, kick=jnp.where(fire, 1.0, st.kick))
        return st

    def lasers(self, st: State, who: int, lo=None, hi=None):
        """Distance (metres) to the nearest wall, shut door, cover or yard edge along each laser,
        starting straight ahead and going round to the left; capped at `sight`."""
        if lo is None:
            lo, hi = self.boxes(st)
        return _cast(st.pos[who], _rot(self.ray_dirs, st.face[who]), lo, hi, self.cfg)

    def _enemy_obs(self, st: State, me: int, j, own, p):
        """The 10 numbers about enemy j as `me` knows it: seen?, where and how it moves (if seen), where the team last
        saw it, how long ago, the direction of its muzzle flash (if seen)."""
        cfg = self.cfg
        seen = st.seen[me, j]
        never = st.since[me, j] >= self.T
        return jnp.concatenate([
            seen[None].astype(jnp.float32),
            jnp.where(seen, own(st.pos[j] - p) / cfg.sight, 0.0),
            jnp.where(seen, own(st.vel[j]) / cfg.run, 0.0),
            jnp.where(never, 0.0, own(st.mem[me, j] - p) / cfg.sight),
            jnp.minimum(st.since[me, j] / self.T, 1.0)[None],
            jnp.where(st.flash_seen[me, j], own((st.pos[j] - p) / jnp.maximum(jnp.linalg.norm(st.pos[j] - p),
                                                                              1e-6)), 0.0),
        ])

    def enemy_order(self, st: State, me: int):
        """`me`'s enemies, nearest known first: seen now (by distance), then remembered (by distance to where they were
        last seen), then never seen; those the team knows are down go last. -> (enemies,) operator numbers."""
        en = jnp.asarray(self.enemies[me])
        p = st.pos[me]
        dist = jnp.linalg.norm(st.mem[me, en] - p, axis=1)
        key = jnp.where(st.seen[me, en], jnp.linalg.norm(st.pos[en] - p, axis=1),
                        jnp.where(st.since[me, en] < self.T, 1e3 + dist, 1e6))
        key = key + jnp.where(st.known_down[me, en], 1e7, 0.0) + jnp.arange(len(self.enemies[me])) * 1e-3
        return en[jnp.argsort(key)]

    def observe(self, st: State, who: int) -> jax.Array:
        """What operator `who` knows, about -1..1, positions and directions in its own frame
        (x straight ahead, y to its left). Nothing here comes from what it can't see or hear."""
        cfg, b, bi = self.cfg, self.b, st.building
        me = who
        p, face = st.pos[me], st.face[me]
        own = lambda v: _rot(v, -face)                                          # world -> my frame
        if self.teams_on:
            order = self.enemy_order(st, me)
            down = st.known_down[me, order]
            enemy = jnp.where(down[0], 0.0, self._enemy_obs(st, me, order[0], own, p))
        else:
            enemy = self._enemy_obs(st, me, 1 - who, own, p)
        snd = st.sound[me]
        sounds = jnp.concatenate([own(snd[:, :2]) * (snd[:, 2:] > 0), snd[:, 2:]], 1).reshape(-1)
        flo, fhi = b.foot_lo[bi], b.foot_hi[bi]
        mine = jnp.concatenate([
            jnp.stack([jnp.sin(face), jnp.cos(face)]),
            own(st.vel[me]) / cfg.run,
            jnp.clip((p - flo) / (fhi - flo) * 2 - 1, -2, 2),
            st.turn[me][None],
        ])
        # the two nearest doors: where (my frame), open?, center-fed on my side?
        dlo, dhi = b.door_lo[bi], b.door_hi[bi]
        mid = (dlo + dhi) / 2
        dist = jnp.where(b.door_valid[bi], jnp.linalg.norm(mid - p, axis=1), jnp.inf)
        near = jnp.argsort(dist)[:2]
        axis = jnp.argmin((dhi - dlo)[near], axis=1)                            # doors are thin across their wall
        side = (p[axis] > mid[near, axis]).astype(jnp.int32)
        doors = jnp.concatenate([
            own(mid[near] - p) / 10.0,
            st.door_open[near][:, None].astype(jnp.float32),
            b.door_center_fed[bi, near, side][:, None].astype(jnp.float32),
        ], 1).reshape(-1)
        obs = [self.lasers(st, me) / cfg.sight, enemy, sounds, mine, doors,
               st.cleared[me].astype(jnp.float32), (1 - st.t / self.T)[None]]
        mine_side = int(self.team[me])
        if self.bangs_on:   # flashbangs left, how blind I am, my live flashbang (where, fuse left), an enemy's bang
            live = st.bang_timer[me] > 0
            hb = st.bang_heard[me]
            obs.append(jnp.concatenate([
                (st.bangs[me] / max(cfg.flashbangs[mine_side], 1))[None], (st.blind[me] / self.blind_steps)[None],
                jnp.where(live, own(st.bang_pos[me] - p) / cfg.sight, 0.0), (st.bang_timer[me] / self.fuse_steps)[None],
                own(hb[:2]) * (hb[2] > 0), hb[2:]]))
        if self.guns_on:    # rounds loaded, rounds carried, reloading, recoil, an enemy's reload heard
            rh = st.reload_heard[me]
            obs.append(jnp.concatenate([
                (st.mag[me] / cfg.magazine)[None], (st.reserve[me] / max(cfg.magazine * cfg.spare_mags, 1))[None],
                (st.reload_timer[me] / self.reload_steps)[None], st.kick[me][None],
                own(rh[:2]) * (rh[2] > 0), rh[2:]]))
        if self.health_on:  # (last: older brains read the numbers before) my health; did my shot just hit (you can tell when you hit someone)
            obs.append(jnp.stack([st.hp[me] / cfg.health, (st.landed[me] < self.sound_steps).astype(jnp.float32)]))
        if self.react_on:   # how settled my aim is (on the nearest enemy I know of)
            j = order[0] if self.teams_on else 1 - who
            obs.append((st.aim[me, j] / self.react_steps)[None].astype(jnp.float32))
        if self.teams_on:   # (last) the other enemies, my teammates, which one of my team I am
            extra = self.slots - 1
            more = [jnp.where(down[k], 0.0, self._enemy_obs(st, me, order[k], own, p)) for k in range(1, len(order))]
            more += [jnp.zeros(ENEMY_OBS)] * (extra - len(more))
            mates = []
            for j in self.mates[me]:
                rel = st.face[j] - face
                mates.append(jnp.concatenate([
                    own(st.pos[j] - p) / cfg.sight, jnp.stack([jnp.sin(rel), jnp.cos(rel)]),
                    (st.hp[j] / cfg.health if self.health_on else st.alive[j].astype(jnp.float32))[None],
                    st.alive[j][None].astype(jnp.float32)]))
            mates += [jnp.zeros(MATE_OBS)] * (extra - len(mates))
            k = (self.side_ops(mine_side)).index(me)
            obs += more + mates + [jnp.asarray([k / extra], jnp.float32)]
        if self.ready_on:   # (last of all) how far my rifle is up
            obs.append((st.ready_t[me] / self.ready_steps)[None].astype(jnp.float32))
        if self.stun_on:    # (after it) how long I'm still stunned
            obs.append((st.stun_t[me] / self.stun_steps)[None].astype(jnp.float32))
        return jnp.concatenate(obs)

    def reset_if(self, st: State, key) -> State:
        """Finished rounds start again in a random building (for training)."""
        fresh = self.init(key)
        return jax.tree.map(lambda a, b: jnp.where(st.done, a, b), fresh, st)

    def walls(self, st: State):
        """Walls and shut doors (no cover): what stops sound getting through, a throw, and a bang's light."""
        b, bi = self.b, st.building
        shut = (~st.door_open & b.door_valid[bi])[:, None]
        lo = jnp.concatenate([b.wall_lo[bi], jnp.where(shut, b.door_lo[bi], FAR)])
        hi = jnp.concatenate([b.wall_hi[bi], jnp.where(shut, b.door_hi[bi], FAR)])
        return lo, hi

    def walls_between(self, st: State, p, q):
        """How many walls and shut doors the straight line from p to q goes through."""
        lo, hi = self.walls(st)
        t_in, t_out = _slab(p, q - p, lo, hi)
        return jnp.sum((t_in <= t_out) & (t_out >= 0) & (t_in <= 1))

    def _flashbangs(self, st: State, a) -> State:
        """Fuses burn down; a bang blinds whoever is within bang_range with no wall or shut door in between
        for blind_time (x blind_away if it wasn't in their light's cone); then a throw: the flashbang
        lands `throw` metres ahead, or just short of the first wall or shut door (at your feet if you throw
        into one). One live flashbang per operator at a time."""
        cfg = self.cfg
        lo, hi = self.walls(st)
        timer = jnp.maximum(st.bang_timer - 1, 0)
        now = st.bang_timer == 1                                                # (N,) whose went off
        blind = jnp.maximum(st.blind - 1, 0)
        for src in range(self.N):
            q = st.bang_pos[src]
            d = jnp.linalg.norm(st.pos - q, axis=1)                             # (N,) each operator to it
            clear = _clear(q, st.pos, lo, hi)
            facing = jax.vmap(lambda p, f: self.in_cone(p, f, q[None])[0])(st.pos, st.face)
            steps = jnp.where(facing, self.blind_steps, int(round(self.blind_steps * cfg.blind_away)))
            hit = now[src] & (d <= cfg.bang_range) & clear & st.alive
            blind = jnp.where(hit, jnp.maximum(blind, steps), blind)
        want = a[:, THROW] > 0 if a.shape[-1] > THROW else jnp.zeros(self.N, bool)
        throw = want & (st.bangs > 0) & (timer == 0) & st.alive & ~st.done
        ahead = jnp.stack([jnp.cos(st.face), jnp.sin(st.face)], 1)              # (N, 2)
        room = jax.vmap(lambda p, d: _cast(p, d[None], lo, hi, cfg)[0])(st.pos, ahead)
        land = st.pos + ahead * jnp.clip(jnp.minimum(cfg.throw, room - 0.3), 0.0, None)[:, None]
        return st._replace(bangs=st.bangs - throw, bang_pos=jnp.where(throw[:, None], land, st.bang_pos),
                           bang_timer=jnp.where(throw, self.fuse_steps, timer), bang_now=now, blind=blind)

    @staticmethod
    def _loudest(heard: list):
        """Several enemies' sounds of one kind, each (..., 3) (direction, loudness) -> the loudest of them."""
        if len(heard) == 1:
            return heard[0]
        x = jnp.stack(heard)                                                    # (enemies, ..., 3)
        k = jnp.argmax(x[..., 2], axis=0)
        return jnp.take_along_axis(x, k[None, ..., None], axis=0)[0]

    def _hear(self, st: State) -> State:
        """Each operator hears its enemies' footsteps, doors and shots made this decision (the loudest of each kind)."""
        cfg = self.cfg
        speed = jnp.linalg.norm(st.vel, axis=1)
        steps = jnp.where(speed > cfg.walk + 0.1, cfg.hear_run, jnp.where(speed > cfg.still, cfg.hear_walk, 0.0))
        if self.bump_on:
            steps = jnp.where(st.bumped, jnp.maximum(steps, cfg.bump), steps)
        made = jnp.stack([steps, jnp.where(st.door_event, cfg.hear_door, 0.0),
                          jnp.where(st.fired, cfg.hear_shot, 0.0)], 1)           # (N makers, 3 kinds) ranges
        made = made * st.alive[:, None]
        k = 2 * jnp.pi / cfg.sound_dirs

        def listen(w, o):
            me = st.pos[w]
            door = st.door_at[o]
            door = door + (me - door) / jnp.maximum(jnp.linalg.norm(me - door), 1e-6) * cfg.wall  # its face on my side
            src = jnp.stack([st.pos[o], door, st.pos[o]])                        # footsteps, door, shot
            d = src - me
            dist = jnp.linalg.norm(d, axis=1)
            walls = jax.vmap(lambda q: self.walls_between(st, me, q))(src)
            reach = made[o] * cfg.through_wall ** walls
            loud = jnp.where(reach > 0, jnp.clip(1 - dist / jnp.maximum(reach, 1e-6), 0, 1), 0.0)   # (3,)
            ang = jnp.round(jnp.arctan2(d[:, 1], d[:, 0]) / k) * k               # roughly where from
            return jnp.concatenate([jnp.stack([jnp.cos(ang), jnp.sin(ang)], 1), loud[:, None]], 1)

        new = jnp.stack([self._loudest([listen(w, o) for o in self.enemies[w]]) for w in range(self.N)])  # (N, 3, 3)
        age = jnp.minimum(st.sound_age + 1, self.sound_steps)
        old = jnp.where((age < self.sound_steps)[..., None], st.sound, 0.0)
        take = (new[..., 2] > 0) & (new[..., 2] >= old[..., 2])                  # louder than what's in mind
        st = st._replace(sound=jnp.where(take[..., None], new, old), sound_age=jnp.where(take, 0, age))
        if not self.bangs_on:
            return st

        def bang(w, o):                                                          # an enemy's flashbang
            me, q = st.pos[w], st.bang_pos[o]
            d = q - me
            reach = jnp.where(st.bang_now[o], cfg.hear_bang, 0.0) * cfg.through_wall ** self.walls_between(st, me, q)
            loud = jnp.where(reach > 0, jnp.clip(1 - jnp.linalg.norm(d) / jnp.maximum(reach, 1e-6), 0, 1), 0.0)
            ang = jnp.round(jnp.arctan2(d[1], d[0]) / k) * k
            return jnp.stack([jnp.cos(ang), jnp.sin(ang), loud])

        heard = jnp.stack([self._loudest([bang(w, o) for o in self.enemies[w]]) for w in range(self.N)])   # (N, 3)
        age = jnp.minimum(st.bang_age + 1, self.sound_steps)
        old = jnp.where((age < self.sound_steps)[:, None], st.bang_heard, 0.0)
        take = heard[:, 2] > 0
        return st._replace(bang_heard=jnp.where(take[:, None], heard, old), bang_age=jnp.where(take, 0, age))

    def _hear_reload(self, st: State) -> State:
        """A magazine change (an enemy's, starting this decision) is heard like footsteps: hear_reload metres,
        a third less per wall or shut door."""
        cfg = self.cfg
        k = 2 * jnp.pi / cfg.sound_dirs

        def listen(w, o):
            me, q = st.pos[w], st.pos[o]
            d = q - me
            reach = jnp.where(st.reload_now[o] & st.alive[o], cfg.hear_reload, 0.0) \
                * cfg.through_wall ** self.walls_between(st, me, q)
            loud = jnp.where(reach > 0, jnp.clip(1 - jnp.linalg.norm(d) / jnp.maximum(reach, 1e-6), 0, 1), 0.0)
            ang = jnp.round(jnp.arctan2(d[1], d[0]) / k) * k
            return jnp.stack([jnp.cos(ang), jnp.sin(ang), loud])

        heard = jnp.stack([self._loudest([listen(w, o) for o in self.enemies[w]]) for w in range(self.N)])
        age = jnp.minimum(st.reload_age + 1, self.sound_steps)
        old = jnp.where((age < self.sound_steps)[:, None], st.reload_heard, 0.0)
        take = heard[:, 2] > 0
        return st._replace(reload_heard=jnp.where(take[:, None], heard, old), reload_age=jnp.where(take, 0, age))

    def step(self, st: State, actions) -> tuple[State, jax.Array]:
        """actions: (N, 8) per operator (see FWD..RELOAD; 6 or 7 = the rest off). Returns the new state and Assault's
        reward (Defense's is its negative). A side is out when all of it is down. Once a round is over, nothing
        changes."""
        cfg = self.cfg
        if self.stun_on:   # stunned: no moving, turning, firing or anything else this decision
            off = jnp.concatenate([jnp.zeros(3), -jnp.ones(actions.shape[-1] - 3)])
            actions = jnp.where((st.stun_t > 0)[:, None], off, actions)
        pos, vel, face, impact = self._move(st, actions)
        alive = st.alive[:, None]
        pos, vel = jnp.where(alive, pos, st.pos), jnp.where(alive, vel, 0.0)
        hard = (impact > cfg.walk + 0.1) & st.alive               # a collision harder than walking into a wall
        bumped = hard if self.bump_on else st.bumped
        door_open, door_timer, door_event, door_at = self._doors(st, actions * alive, pos)
        new = st._replace(pos=pos, vel=vel, face=jnp.where(st.alive, face, st.face),
                          turn=jnp.clip(actions[:, TURN], -1, 1) * st.alive, door_open=door_open,
                          door_timer=door_timer, door_event=door_event,
                          door_at=jnp.where(door_event[:, None], door_at, st.door_at), t=st.t + 1, bumped=bumped)
        if self.stun_on:
            new = new._replace(stun_t=jnp.where(hard, self.stun_steps, jnp.maximum(st.stun_t - 1, 0)))
        if self.ready_on:   # running (the switch on and going faster than a walk) or a collision: the rifle drops
            ran = (actions[:, RUN] > 0) & (jnp.linalg.norm(vel, axis=1) > cfg.walk + 0.1)
            new = new._replace(ready_t=jnp.where(ran | hard, 0, jnp.minimum(st.ready_t + 1, self.ready_steps)))
        if self.bangs_on:
            new = self._flashbangs(new, actions)
        new = self._look(new)
        if self.guns_on:
            new = self._reload(new, actions)
        new = self._hear(self._shoot(new, actions))
        if self.guns_on:
            new = self._hear_reload(new)
        a_down, d_down = ~new.alive[:self.nA].any(), ~new.alive[self.nA:].any()
        timeout = (new.t >= self.T) & ~a_down & ~d_down
        outcome = jnp.where(d_down & ~a_down, 1, jnp.where(a_down & ~d_down, -1, jnp.where(timeout, -1, 0)))
        done = a_down | d_down | timeout
        reward = cfg.win_reward * outcome - cfg.time_cost * cfg.dt
        new = new._replace(done=done, outcome=outcome)
        new = jax.tree.map(lambda x, y: jnp.where(st.done, x, y), st, new)   # frozen once over
        return new, jnp.where(st.done, 0.0, reward)
