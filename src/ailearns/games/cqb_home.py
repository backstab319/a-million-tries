"""Realistic homes for CQB (episode 13, `--house home`): a single-storey house instead of a kill house.

The plan: a public zone (a living room at the front, a kitchen behind it) and a private wing: a hallway down its
middle with bedrooms and a bathroom on both sides. The front door opens into the living room, the back door into the
kitchen, and sometimes a door ends the hallway. Only real connections get doors (none between two bedrooms). Windows
sit in the outside walls (you see and shoot through them, you can't walk through them). The whole plan is mirrored at
random, so houses face different ways.

Furniture comes by room type and has two heights: **low** (sofa, bed, counter, table: blocks movement, not sight or
shots) and **tall** (wardrobe, fridge, shelving: blocks everything; it's the game's `cover`). Real furniture stands
against walls: a piece is flush with a wall or at least `cover_gap` from walls and other pieces (no slivers a body
can't pass); none stands in a doorway's way, no tall piece covers a window, and every room's doorways stay connected
for a body of the operators' size (checked on a 10 cm grid; a piece that breaks it is left out).

The result has make_building's keys, plus low_lo / low_hi, win_lo / win_hi, room_types and furniture (name, box,
tall, room: what the 3D draws, each piece exactly the box the game uses).
"""

import numpy as np

ROOM_TYPES = ("living", "kitchen", "hall", "bedroom", "bath")

# (name, along the wall (m), out from it (m), tall, placement) per room type, in placement order. "wall": its back
# against a wall; "float": in the room, cover_gap from everything.
FURNITURE = {
    "living": [("sofa", (2.0, 2.4), (0.85, 0.95), False, "wall"), ("tv_stand", (1.4, 1.8), (0.4, 0.5), False, "wall"),
               ("bookshelf", (0.8, 1.2), (0.3, 0.4), True, "wall"),
               ("coffee_table", (0.9, 1.2), (0.5, 0.6), False, "float")],
    "kitchen": [("counter", (2.0, 3.4), (0.6, 0.6), False, "wall"), ("fridge", (0.7, 0.8), (0.65, 0.75), True, "wall"),
                ("table", (1.2, 1.6), (0.8, 0.9), False, "float")],
    "bedroom": [("bed", (0.9, 1.6), (1.95, 2.05), False, "wall"), ("wardrobe", (1.0, 1.8), (0.55, 0.65), True, "wall"),
                ("dresser", (0.8, 1.2), (0.45, 0.5), False, "wall")],
    "bath": [("tub", (1.5, 1.7), (0.7, 0.8), False, "wall"), ("cabinet", (0.5, 0.7), (0.35, 0.45), True, "wall")],
    "hall": [],
}


def _snap(x, step=0.5):
    return float(np.round(x / step) * step)


def _gap(alo, ahi, blo, bhi) -> float:
    """Distance between two boxes (0 when they touch or overlap)."""
    d = np.maximum(0, np.maximum(blo - ahi, alo - bhi))
    return float(np.hypot(*d))


def _overlap(alo, ahi, blo, bhi, eps=1e-6) -> bool:
    return bool((alo < bhi - eps).all() and (blo < ahi - eps).all())


def _strip(rng, xa, xb, bath: bool):
    """Split [xa, xb] (a strip of the private wing) into rooms: (x0, x1, type). At most one bathroom (2.2-2.6 m);
    one to three bedrooms, each at least 3 m (3.3 m when there are three)."""
    sizes, kinds = [], []
    if bath:
        sizes.append(_snap(rng.uniform(2.2, 2.6), 0.1))
        kinds.append("bath")
    rest = (xb - xa) - sum(sizes)
    nb = 3 if rest >= 12.0 else 2 if rest >= 6.4 else 1
    cut = np.sort(rng.uniform(0.4, 0.6, nb - 1)) * rest if nb > 1 else np.array([])
    edges = [0.0] + [_snap(c, 0.1) for c in cut] + [rest]
    beds = [b - a for a, b in zip(edges[:-1], edges[1:])]
    if min(beds) < (3.3 if nb == 3 else 3.0):
        beds = [rest / nb] * nb
    sizes += beds
    kinds += ["bedroom"] * nb
    out, x = [], xa
    for i in rng.permutation(len(sizes)):
        out.append((x, x + sizes[i], kinds[i]))
        x += sizes[i]
    out[-1] = (out[-1][0], xb, out[-1][2])
    return out


def _furnish(rng, rl, rh, kind, keep, windows, placed, cfg):
    """Pieces for one room (inner box rl..rh: inside the wall faces). keep: boxes nothing may overlap (doorways'
    approaches); windows: boxes a tall piece mustn't stand in front of; placed: pieces so far (all rooms)."""
    g = cfg.cover_gap
    out = []
    for name, along, outw, tall, how in FURNITURE[kind]:
        for _ in range(40):
            a, o = _snap(rng.uniform(*along), 0.1), _snap(rng.uniform(*outw), 0.05)
            if how == "wall":
                side = int(rng.integers(4))          # 0 west, 1 east, 2 south, 3 north wall
                ax = 1 if side < 2 else 0            # the axis the piece runs along
                span = (rl[ax], rh[ax])
                if span[1] - span[0] < a:
                    continue
                if rng.random() < 0.5:                # in a corner (flush), or away from the corners
                    s = span[0] if rng.random() < 0.5 else span[1] - a
                else:
                    lo_, hi_ = span[0] + g, span[1] - g - a
                    if hi_ < lo_:
                        continue
                    s = _snap(rng.uniform(lo_, hi_), 0.1)
                blo, bhi = np.zeros(2), np.zeros(2)
                blo[ax], bhi[ax] = s, s + a
                if side in (0, 2):
                    blo[1 - ax], bhi[1 - ax] = rl[1 - ax], rl[1 - ax] + o
                else:
                    blo[1 - ax], bhi[1 - ax] = rh[1 - ax] - o, rh[1 - ax]
            else:
                free = (rh - rl) - 2 * g - np.array([a, o])
                if (free < 0).any():
                    continue
                blo = np.round((rl + g + rng.uniform(0, 1, 2) * free) * 10) / 10
                bhi = blo + np.array([a, o])
            # flush (0) or a body's width from each wall face, never past it
            gaps = np.concatenate([blo - rl, rh - bhi])
            if (gaps < -1e-6).any() or ((gaps > 1e-6) & (gaps < g - 1e-6)).any():
                continue
            if any(_overlap(blo, bhi, kl, kh) for kl, kh in keep):
                continue
            if tall and any(_gap(blo, bhi, wl, wh) < 0.05 for wl, wh in windows):
                continue
            others = placed + out
            if any(_overlap(blo, bhi, p["lo"], p["hi"]) or 1e-6 < _gap(blo, bhi, p["lo"], p["hi"]) < g - 1e-6
                   for p in others):
                continue
            out.append({"name": name, "lo": blo, "hi": bhi, "tall": tall, "kind": kind})
            break
    return out


def _connected(rl, rh, boxes, starts, r) -> bool:
    """Can a disc of radius r get from every start point to every other inside rl..rh, around `boxes`? (10 cm grid)"""
    step = 0.1
    nx, ny = int(np.ceil((rh[0] - rl[0]) / step)), int(np.ceil((rh[1] - rl[1]) / step))
    if nx < 1 or ny < 1:
        return True
    xs = rl[0] + (np.arange(nx) + 0.5) * step
    ys = rl[1] + (np.arange(ny) + 0.5) * step
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ok = (X >= rl[0] + r) & (X <= rh[0] - r) & (Y >= rl[1] + r) & (Y <= rh[1] - r)
    for bl, bh in boxes:
        dx = np.maximum(0, np.maximum(bl[0] - X, X - bh[0]))
        dy = np.maximum(0, np.maximum(bl[1] - Y, Y - bh[1]))
        ok &= np.hypot(dx, dy) >= r
    cells = [(int(np.clip((p[0] - rl[0]) / step, 0, nx - 1)), int(np.clip((p[1] - rl[1]) / step, 0, ny - 1)))
             for p in starts]
    if not cells or not all(ok[c] for c in cells):
        return False
    seen = np.zeros_like(ok)
    stack = [cells[0]]
    seen[cells[0]] = True
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and ok[a, b] and not seen[a, b]:
                seen[a, b] = True
                stack.append((a, b))
    return all(seen[c] for c in cells)


def _seen_from(p, eyes, lo, hi) -> bool:
    """(ep13 8.16) Can any of `eyes` see point p in a straight line, past every box (lo, hi: (M, 2))? Slab test."""
    for e in eyes:
        d = p - e
        with np.errstate(divide="ignore", invalid="ignore"):
            t0, t1 = (lo - e) / d, (hi - e) / d
        tn = np.nanmax(np.minimum(t0, t1), -1)
        tf = np.nanmin(np.maximum(t0, t1), -1)
        if not ((tf >= np.maximum(tn, 0)) & (tn <= 1)).any():
            return True
    return False


def _mirror(b: dict, ax: int) -> dict:
    """The house mirrored across its footprint's centre line on axis ax (0: east-west flip)."""
    S = b["lo"][ax] + b["hi"][ax]

    def pt(p):
        p = np.array(p, float).copy()
        p[..., ax] = S - p[..., ax]
        return p

    def boxes(lo, hi):
        lo, hi = np.array(lo, float).reshape(-1, 2).copy(), np.array(hi, float).reshape(-1, 2).copy()
        nlo, nhi = lo.copy(), hi.copy()
        nlo[:, ax], nhi[:, ax] = S - hi[:, ax], S - lo[:, ax]
        return nlo, nhi

    out = dict(b)
    out["rooms"] = [tuple(x[0] for x in boxes(l, h)) for l, h in b["rooms"]]
    lines = []
    for axis, c, a, e in b["lines"]:
        if axis == ax:          # a line across the flip: its position moves
            lines.append((axis, S - c, a, e))
        else:                   # a line along it: its ends swap
            lines.append((axis, c, S - e, S - a))
    out["lines"] = lines
    for k in ("wall", "cover", "low", "win"):
        out[f"{k}_lo"], out[f"{k}_hi"] = boxes(b[f"{k}_lo"], b[f"{k}_hi"])
    doors = []
    for d in b["doors"]:
        lo, hi = boxes(d["lo"], d["hi"])
        doors.append({**d, "lo": lo[0], "hi": hi[0], "center": pt(d["center"])})
    out["doors"] = doors
    out["entries"] = [(pt(p), float(np.pi - f) if ax == 0 else float(-f), r) for p, f, r in b["entries"]]
    out["entries"] = [(p, float(np.arctan2(np.sin(f), np.cos(f))), r) for p, f, r in out["entries"]]
    out["spawn"] = pt(b["spawn"])
    out["furniture"] = [{**f, **dict(zip(("lo", "hi"), (x[0] for x in boxes(f["lo"], f["hi"]))))}
                        for f in b["furniture"]]
    return out


def make_home(seed: int, cfg) -> dict:
    from ailearns.games.cqb import _box, _place_door, _room_at, _segments, _wall_boxes
    rng = np.random.default_rng(seed)
    t, door = cfg.wall, cfg.door
    size = np.array([_snap(rng.uniform(*cfg.width)), _snap(rng.uniform(*cfg.depth))])
    yard = np.array([cfg.yard_w, cfg.yard_h])
    slack = yard - size
    room_outside = cfg.entry_dist + 1.5
    lo = np.round((slack / 2 + rng.uniform(-1, 1, 2) * np.maximum(slack / 2 - room_outside, 0)) * 2) / 2
    hi = lo + size
    (x0, y0), (x1, y1) = lo, hi

    # the plan: public zone west of xs (living south of yk, kitchen north), the private wing east of it
    xs = _snap(x0 + size[0] * rng.uniform(0.38, 0.46))
    cw = _snap(rng.uniform(1.4, 1.6), 0.1)                                  # the hallway's width
    ya = _snap((y0 + y1) / 2 - cw / 2 + rng.uniform(-0.6, 0.6), 0.1)
    ya = float(np.clip(ya, y0 + 3.0, y1 - 3.0 - cw))
    yb = ya + cw
    yk = _snap(y0 + size[1] * rng.uniform(0.52, 0.62))
    if ya - 0.8 < yk < yb + 0.8:   # the living / kitchen wall keeps clear of the hallway's door into that side
        yk = _snap(yb + 1.0) if y1 - _snap(yb + 1.0) >= 3.0 else _snap(ya - 1.0)
    rooms = [(np.array([x0, y0]), np.array([xs, yk])), (np.array([x0, yk]), np.array([xs, y1])),
             (np.array([xs, ya]), np.array([x1, yb]))]
    types = ["living", "kitchen", "hall"]
    lines = [(0, xs, y0, y1), (1, ya, xs, x1), (1, yb, xs, x1), (1, yk, x0, xs)]
    bath_south = bool(rng.random() < 0.5)
    wing = {}
    for name, (ylo, yhi) in (("south", (y0, ya)), ("north", (yb, y1))):
        strip = _strip(rng, xs, x1, bath=(name == "south") == bath_south)
        wing[name] = []
        for k, (ra, rb, kind) in enumerate(strip):
            rooms.append((np.array([ra, ylo]), np.array([rb, yhi])))
            types.append(kind)
            wing[name].append((len(rooms) - 1, ra, rb))
            if k:
                lines.append((0, ra, ylo, yhi))
    perimeter = [(0, x0, y0, y1), (0, x1, y0, y1), (1, y0, x0, x1), (1, y1, x0, x1)]
    every = lines + perimeter
    line_of = {ln: i for i, ln in enumerate(every)}

    doors, gaps = [], {i: [] for i in range(len(every))}

    def add_door(line, s, e, outside, centred=False):
        axis, c = line[0], line[1]
        if centred or e - s < door + 2 * cfg.jamb:
            d0 = _snap((s + e - door) / 2, 0.1)
            d0, d1 = d0, d0 + door
        else:
            d0, d1 = _place_door(rng, [(s, e)], cfg)
        gaps[line_of[line]].append((d0, d1))
        doors.append((axis, c, d0, d1, outside))

    add_door((1, yk, x0, xs), x0, xs, False)                                # living <-> kitchen
    add_door((0, xs, y0, y1), ya, yb, False, centred=True)                  # living / kitchen side <-> the hallway
    for name, line in (("south", (1, ya, xs, x1)), ("north", (1, yb, xs, x1))):
        for _, ra, rb in wing[name]:                                         # every wing room opens onto the hallway
            add_door(line, ra, rb, False)
    add_door((1, y0, x0, x1), x0, xs, True)                                 # the front door, into the living room
    if rng.random() < 0.6:                                                  # the back door: kitchen's back or side
        add_door((1, y1, x0, x1), x0, xs, True)
    else:
        add_door((0, x0, y0, y1), yk, y1, True)
    end_door = rng.random() < 0.4                                           # sometimes a door ends the hallway
    if end_door:
        add_door((0, x1, y0, y1), ya, yb, True, centred=True)

    # windows: in outside walls, kept off corners and doorways
    win = []
    for r, (rl, rh) in enumerate(rooms):
        kind = types[r]
        for line in perimeter:
            axis, c = line[0], line[1]
            on = (axis == 0 and (abs(rl[0] - c) < 1e-6 or abs(rh[0] - c) < 1e-6)) or \
                 (axis == 1 and (abs(rl[1] - c) < 1e-6 or abs(rh[1] - c) < 1e-6))
            if not on:
                continue
            along = 1 if axis == 0 else 0
            s, e = rl[along], rh[along]
            if kind == "hall":
                if end_door or rng.random() < 0.5:
                    continue
                w = 0.8
            else:
                if rng.random() > (0.5 if kind == "bath" else 0.8):
                    continue
                w = 0.6 if kind == "bath" else _snap(rng.uniform(1.0, 1.6), 0.1)
            lo_, hi_ = s + 0.5, e - 0.5 - w
            if hi_ < lo_:
                continue
            for _ in range(20):
                w0 = _snap(rng.uniform(lo_, hi_), 0.1)
                w1 = w0 + w
                if all(w1 + 0.4 <= g0 or w0 >= g1 + 0.4 for g0, g1 in gaps[line_of[line]]):
                    gaps[line_of[line]].append((w0, w1))
                    win.append(_box(axis, c, w0, w1, t))
                    break

    wall_lo, wall_hi, soft = [], [], []
    for i, (axis, c, a, e) in enumerate(every):
        for bl, bh in _wall_boxes(axis, c, a, e, gaps[i], t):
            wall_lo.append(bl)
            wall_hi.append(bh)
            soft.append(i < len(lines))                                       # inside walls: drywall; outside: brick

    door_info = []
    for axis, c, d0, d1, outside in doors:
        bl, bh = _box(axis, c, d0, d1, t)
        mid = (bl + bh) / 2
        normal = np.eye(2)[axis]
        sides_ = []
        for sgn in (-1, 1):
            r = _room_at(rooms, mid + sgn * normal * 0.5)
            if r < 0:
                sides_.append((-1, False))
                continue
            rl, rh = rooms[r]
            al = 1 - axis
            gap = min(d0 - rl[al], rh[al] - d1) - t / 2
            sides_.append((r, bool(gap < cfg.corner_fed)))
        door_info.append({"lo": bl, "hi": bh, "center": mid, "axis": axis, "outside": outside,
                          "rooms": (sides_[0][0], sides_[1][0]), "corner_fed": (sides_[0][1], sides_[1][1])})

    # furniture: room by room, keeping doorways' approaches clear (1.2 m deep, 0.3 m wider each side)
    keep = []
    for d in door_info:
        n = np.eye(2)[d["axis"]]
        widen = (1 - n) * 0.3
        keep.append((d["lo"] - n * 1.2 - widen, d["hi"] + n * 1.2 + widen))
    win_lo = [w[0] for w in win]
    win_hi = [w[1] for w in win]
    furniture = []
    for r, (rl, rh) in enumerate(rooms):
        inner_lo, inner_hi = rl + t / 2, rh - t / 2
        pieces = _furnish(rng, inner_lo, inner_hi, types[r], keep, list(zip(win_lo, win_hi)), furniture, cfg)
        starts = []
        for d in door_info:
            if r in d["rooms"]:
                n = np.eye(2)[d["axis"]]
                side = 1 if _room_at([rooms[r]], d["center"] + n * 0.5) == 0 else -1
                starts.append(d["center"] + n * side * 0.5)
        while pieces and not _connected(inner_lo, inner_hi, [(p["lo"], p["hi"]) for p in pieces], starts,
                                        cfg.radius + 0.02):
            pieces.pop()                                                    # the last piece closed a way: leave it out
        for p in pieces:
            p["room"] = r
        furniture += pieces
    tall = [f for f in furniture if f["tall"]][:cfg.max_cover]
    low = [f for f in furniture if not f["tall"]][:cfg.max_low]
    furniture = tall + low

    entries = []
    for d in door_info:
        if d["outside"]:
            r_in = d["rooms"][0] if d["rooms"][0] >= 0 else d["rooms"][1]
            out_sgn = -1 if d["rooms"][0] < 0 else 1
            normal = np.eye(2)[d["axis"]] * out_sgn
            p = d["center"] + normal * cfg.entry_dist
            entries.append((p, float(np.arctan2(-normal[1], -normal[0])), r_in))

    # (ep13 8.16, the owner) no defender starts where an attacker's start sees him through a window: walls, tall
    # furniture and every door (shut, as outside doors start) block the view; a window doesn't
    eyes = [e for p_, f_, _ in entries for e in (p_, p_ - np.array([np.cos(f_), np.sin(f_)]) * cfg.stack_gap)]
    blk_lo = np.array(list(wall_lo) + [f["lo"] for f in furniture if f["tall"]] + [d["lo"] for d in door_info], float)
    blk_hi = np.array(list(wall_hi) + [f["hi"] for f in furniture if f["tall"]] + [d["hi"] for d in door_info], float)
    spawn, spawn_room = [], []
    clear = 0.6
    redraw = np.random.default_rng([seed, 816])   # its own draws: the layout's (mirroring after this) stay as they were
    order = rng.permutation(np.arange(cfg.spawn_points) % len(rooms))
    for r in order:
        rl, rh = rooms[r]
        c = np.minimum(clear, (rh - rl - t) / 2 - 0.05)                     # the hallway is narrow
        for _ in range(200):
            p = rng.uniform(rl + t / 2 + c, rh - t / 2 - c)
            if all(np.hypot(*(p - np.clip(p, f["lo"], f["hi"]))) > clear for f in furniture):
                break
        if _seen_from(p, eyes, blk_lo, blk_hi):   # seen through a window: another spot in the room (or keep it)
            for _ in range(200):
                q = redraw.uniform(rl + t / 2 + c, rh - t / 2 - c)
                if (all(np.hypot(*(q - np.clip(q, f["lo"], f["hi"]))) > clear for f in furniture)
                        and not _seen_from(q, eyes, blk_lo, blk_hi)):
                    p = q
                    break
        spawn.append(p)
        spawn_room.append(int(r))

    arr = lambda xs_: np.array(xs_, float).reshape(-1, 2)
    b = {"seed": seed, "lo": lo, "hi": hi, "rooms": rooms, "lines": lines,
         "wall_lo": np.array(wall_lo), "wall_hi": np.array(wall_hi), "wall_soft": np.array(soft), "doors": door_info,
         "cover_lo": arr([f["lo"] for f in tall]), "cover_hi": arr([f["hi"] for f in tall]),
         "low_lo": arr([f["lo"] for f in low]), "low_hi": arr([f["hi"] for f in low]),
         "win_lo": arr(win_lo), "win_hi": arr(win_hi), "entries": entries, "spawn": np.array(spawn),
         "spawn_room": np.array(spawn_room), "room_types": types, "furniture": furniture}
    for ax in (0, 1):
        if rng.random() < 0.5:
            b = _mirror(b, ax)
    return b
