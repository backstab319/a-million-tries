"""CQB: fair buildings (every room reachable, sealed when the doors are shut), and both kinds of doorway."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ailearns.games.cqb import (ASSAULT, DEFENSE, FIRE, FWD, INTERACT, RUN, SIDE, THROW, TURN, Buildings, CQBConfig,
                                CQBGame, _clear, cell_fracs, cell_points, make_building, make_buildings,
                                nav_graph, room_graph)

CFG = CQBConfig()
SEEDS = list(range(24)) + [100_000, 100_001]
R = 0.3  # an operator's radius


def _free_grid(b, doors_open: bool, step=0.1):
    """Where an operator's centre can be: a grid over the yard, free if no box is within R."""
    xs, ys = np.arange(R, CFG.yard_w - R, step), np.arange(R, CFG.yard_h - R, step)
    pts = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1)
    lo, hi = [b["wall_lo"], b["cover_lo"]], [b["wall_hi"], b["cover_hi"]]
    if not doors_open:
        lo.append(np.array([d["lo"] for d in b["doors"]]))
        hi.append(np.array([d["hi"] for d in b["doors"]]))
    lo, hi = np.vstack(lo), np.vstack(hi)
    free = np.ones(pts.shape[:2], bool)
    for bl, bh in zip(lo, hi):
        free &= np.linalg.norm(pts - np.clip(pts, bl, bh), axis=-1) > R
    return free, xs, ys


def _flood(free, start):
    seen = np.zeros_like(free)
    stack = [start]
    while stack:
        i, j = stack.pop()
        if seen[i, j] or not free[i, j]:
            continue
        seen[i, j] = True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, c = i + di, j + dj
            if 0 <= a < free.shape[0] and 0 <= c < free.shape[1] and not seen[a, c]:
                stack.append((a, c))
    return seen


def _cell(xs, ys, p):
    return int(np.argmin(abs(xs - p[0]))), int(np.argmin(abs(ys - p[1])))


@pytest.mark.parametrize("seed", SEEDS)
def test_rooms_are_big_enough_and_fit(seed):
    b = make_building(seed, CFG)
    assert CFG.rooms[0] <= len(b["rooms"]) <= CFG.rooms[1]
    for lo, hi in b["rooms"]:
        assert (hi - lo >= CFG.min_room - 1e-9).all()
        assert (hi - lo <= CFG.max_room + 1e-9).all() or len(b["rooms"]) == CFG.max_rooms
    assert len(b["wall_lo"]) <= CFG.max_walls and len(b["doors"]) <= CFG.max_doors
    assert len(b["cover_lo"]) <= CFG.max_cover
    assert CFG.outside_doors[0] <= sum(d["outside"] for d in b["doors"]) <= CFG.outside_doors[1]


@pytest.mark.parametrize("seed", SEEDS)
def test_every_room_reachable_through_doors(seed):
    """Doors open: from every Assault start you can walk to every Defense start (so into every room)."""
    b = make_building(seed, CFG)
    free, xs, ys = _free_grid(b, doors_open=True)
    for p, _, _ in b["entries"]:
        reach = _flood(free, _cell(xs, ys, p))
        for q in b["spawn"]:
            assert reach[_cell(xs, ys, q)], f"spawn {q} unreachable from entry {p}"


@pytest.mark.parametrize("seed", SEEDS[:8])
def test_building_sealed_when_doors_shut(seed):
    b = make_building(seed, CFG)
    free, xs, ys = _free_grid(b, doors_open=False)
    reach = _flood(free, _cell(xs, ys, b["entries"][0][0]))
    assert not any(reach[_cell(xs, ys, q)] for q in b["spawn"])


@pytest.mark.parametrize("seed", SEEDS)
def test_doors_join_the_right_places(seed):
    b = make_building(seed, CFG)
    for d in b["doors"]:
        r0, r1 = d["rooms"]
        if d["outside"]:
            assert (r0 < 0) != (r1 < 0)        # one room, one outside
        else:
            assert r0 >= 0 and r1 >= 0 and r0 != r1
        assert abs(max(d["hi"] - d["lo"]) - CFG.door) < 1e-6


@pytest.mark.parametrize("seed", SEEDS)
def test_starts_are_where_they_should_be(seed):
    b = make_building(seed, CFG)
    lo, hi = b["lo"], b["hi"]
    for p, face, _ in b["entries"]:
        assert ((p > 0) & (p < [CFG.yard_w, CFG.yard_h])).all()          # in the yard
        assert ((p < lo) | (p > hi)).any()                               # outside the building
    assert ((b["spawn"] > lo) & (b["spawn"] < hi)).all()                  # Defense inside
    for cl, ch in zip(b["cover_lo"], b["cover_hi"]):
        gap = np.linalg.norm(b["spawn"] - np.clip(b["spawn"], cl, ch), axis=1)
        assert (gap > R).all()
    assert set(b["spawn_room"]) == set(range(len(b["rooms"])))             # every room gets used


def test_both_kinds_of_doorway_are_common():
    """Center-fed and corner-fed entries are both common, so each has enough test cases."""
    fed = [f for s in range(200) for d in make_building(s, CFG)["doors"]
           for r, f in zip(d["rooms"], d["corner_fed"]) if r >= 0]
    share = np.mean(fed)
    assert 0.25 < share < 0.75, share


@pytest.mark.parametrize("seed", SEEDS)
def test_room_graph_finds_the_way(seed):
    """Following next_door from any room (or the outside) gets to any other in exactly hops doors."""
    b = make_building(seed, CFG)
    g = room_graph(b, CFG)
    R, n = CFG.max_rooms, len(b["rooms"])
    nodes = list(range(n)) + [R]
    side = lambda k: [R if r < 0 else r for r in b["doors"][k]["rooms"]]
    for i in nodes:
        for j in nodes:
            here, steps = i, 0
            while here != j:
                k = g["next_door"][here, j]
                a, c = side(k)
                assert here in (a, c)                      # the door is in this room's wall
                here = c if here == a else a
                steps += 1
                assert steps <= n + 1
            assert steps == g["hops"][i, j]
    assert set(range(n)) <= set(g["cell_room"])           # every room has at least one map cell


@pytest.mark.parametrize("seed", SEEDS)
def test_every_map_cell_can_be_seen(seed):
    """Every cell's point is somewhere you could stand (0.45 m clear of walls and cover), near its centre."""
    b = make_building(seed, CFG)
    pts = cell_points(b, CFG)
    lo, hi = np.vstack([b["wall_lo"], b["cover_lo"]]), np.vstack([b["wall_hi"], b["cover_hi"]])
    gap = np.linalg.norm(pts[:, None] - np.clip(pts[:, None], lo[None], hi[None]), axis=2).min(1)
    assert (gap >= 0.45 - 1e-9).all()
    size = (b["hi"] - b["lo"]) / np.array(CFG.seen_grid)
    centres = b["lo"] + cell_fracs(CFG) * (b["hi"] - b["lo"])
    assert (np.abs(pts - centres) <= size * 0.8 + 1e-9).all()      # in its cell (0.2% just outside it)


@pytest.mark.parametrize("seed", SEEDS)
def test_waypoints_reach_everywhere(seed):
    """Every map cell and every doorway has a usable waypoint, all joined up (checked on all 1,280 too)."""
    b = make_building(seed, CFG)
    g = nav_graph(b, CFG)
    cells = CFG.seen_grid[0] * CFG.seen_grid[1]
    used = np.r_[np.arange(cells), cells + np.arange(2 * len(b["doors"]))]
    assert g["real"][used].all()
    assert np.isfinite(g["dist"][np.ix_(used, used)]).all()


def test_padded_arrays():
    bs = make_buildings(range(8), CFG)
    assert bs.wall_lo.shape == (8, CFG.max_walls, 2) and bs.door_lo.shape == (8, CFG.max_doors, 2)
    assert bs.spawn.shape == (8, CFG.spawn_points, 2) and bs.entry.shape == (8, CFG.max_entries, 2)
    assert bs.door_valid.dtype == bool and bs.door_center_fed.dtype == bool
    assert (np.asarray(bs.n_entries) >= 2).all()


# ── Operators and doors ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def game():
    return CQBGame(make_buildings(SEEDS, CFG), CFG)


def _act(**kw):
    """One operator's action from named parts, e.g. _act(fwd=1, run=1)."""
    a = np.zeros(8, np.float32)
    for k, v in kw.items():
        a[{"fwd": FWD, "side": SIDE, "turn": TURN, "interact": INTERACT, "run": RUN, "throw": THROW, "reload": 7}[k]] = v
    return a


def _both(a0, a1=None):
    return jnp.asarray(np.stack([a0, np.zeros(8, np.float32) if a1 is None else a1]))


def _run(game, st, acts, n):
    step = jax.jit(game.step)
    for _ in range(n):
        st, _ = step(st, acts)
    return st


def test_start_positions(game):
    st = jax.vmap(lambda k: game.init(k))(jax.random.split(jax.random.PRNGKey(0), 256))
    b = game.b
    lo, hi = np.asarray(b.foot_lo)[np.asarray(st.building)], np.asarray(b.foot_hi)[np.asarray(st.building)]
    a, d = np.asarray(st.pos[:, ASSAULT]), np.asarray(st.pos[:, DEFENSE])
    assert (((a < lo) | (a > hi)).any(1)).all() and (((d > lo) & (d < hi)).all(1)).all()
    outside = np.asarray(b.door_outside)[np.asarray(st.building)]
    assert not np.asarray(st.door_open)[outside].any()                     # outside doors start shut
    assert 0.3 < np.asarray(st.door_open)[np.asarray(b.door_valid)[np.asarray(st.building)] & ~outside].mean() < 0.7


def test_nobody_goes_through_walls_or_shut_doors(game):
    """Random flailing for a long time, doors left as they are: never inside a wall, cover or a
    shut door, and Assault never gets into the building while the outside doors are shut."""
    def run(key):
        st = game.init(key)
        st = st._replace(door_open=jnp.zeros_like(st.door_open))

        def tick(st, k):
            a = jax.random.uniform(k, (2, 6), minval=-1, maxval=1).at[:, INTERACT].set(-1).at[:, FIRE].set(-1)
            st, _ = game.step(st, a)
            return st._replace(done=jnp.asarray(False), t=jnp.asarray(0)), st.pos
        return st.building, jax.lax.scan(tick, st, jax.random.split(key, 600))[1]

    bi, paths = jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(5), 64))
    paths, bi = np.asarray(paths), np.asarray(bi)
    b = game.b
    for n, k in enumerate(bi):
        lo = np.vstack([b.wall_lo[k], b.cover_lo[k], b.door_lo[k]])
        hi = np.vstack([b.wall_hi[k], b.cover_hi[k], b.door_hi[k]])
        p = paths[n].reshape(-1, 1, 2)
        gap = np.linalg.norm(p - np.clip(p, lo[None], hi[None]), axis=2)
        assert gap.min() > CFG.radius - 0.03
        a = paths[n][:, ASSAULT]
        inside = ((a > np.asarray(b.foot_lo[k])) & (a < np.asarray(b.foot_hi[k]))).all(1)
        assert not inside.any()
        d = paths[n][:, DEFENSE]
        assert ((d > np.asarray(b.foot_lo[k])) & (d < np.asarray(b.foot_hi[k]))).all()


def test_speeds_and_turning(game):
    st = game.init(jax.random.PRNGKey(1), 0)
    st = st._replace(pos=jnp.asarray([[2.0, 2.0], [38.0, 28.0]]), face=jnp.asarray([0.0, np.pi]))
    walk = _run(game, st, _both(_act(fwd=1)), 10)
    assert abs(float(jnp.linalg.norm(walk.vel[0])) - CFG.walk) < 1e-4
    run = _run(game, st, _both(_act(fwd=1, run=1)), 10)
    assert abs(float(jnp.linalg.norm(run.vel[0])) - CFG.run) < 1e-4
    back = _run(game, st, _both(_act(fwd=-1)), 10)
    assert abs(float(back.vel[0, 0]) + CFG.walk * CFG.sideways) < 1e-4    # backwards (west), slower
    left = _run(game, st, _both(_act(side=1)), 10)
    assert float(left.vel[0, 1]) > 0.9                                      # facing east, left is north
    turned = _run(game, st, _both(_act(turn=1)), 5)                         # half a second
    assert abs(np.rad2deg(float(turned.face[0])) - CFG.turn_rate / 2) < 1e-3


def _at_door(game, outside=True):
    """Assault standing 0.8 m outside the first outside door of building 0, facing it."""
    b = make_building(SEEDS[0], CFG)
    i = next(i for i, d in enumerate(b["doors"]) if d["outside"] == outside)
    d = b["doors"][i]
    sgn = -1 if d["rooms"][0] < 0 else 1
    normal = np.eye(2)[d["axis"]] * sgn                                      # pointing outwards
    p = d["center"] + normal * 0.8
    st = game.init(jax.random.PRNGKey(2), 0)
    st = st._replace(pos=st.pos.at[ASSAULT].set(jnp.asarray(p, jnp.float32)),
                     face=st.face.at[ASSAULT].set(float(np.arctan2(-normal[1], -normal[0]))),
                     door_open=jnp.zeros_like(st.door_open))
    return st, i, d


def test_shut_door_blocks_open_door_lets_you_in(game):
    st, i, d = _at_door(game)
    lo, hi = np.asarray(game.b.foot_lo[0]), np.asarray(game.b.foot_hi[0])
    shut = _run(game, st, _both(_act(fwd=1)), 30)
    assert ((np.asarray(shut.pos[ASSAULT]) < lo) | (np.asarray(shut.pos[ASSAULT]) > hi)).any()  # still outside
    st = _run(game, st, _both(_act(interact=1)), 1)
    assert bool(st.door_event[ASSAULT]) and not bool(st.door_open[i])     # started, not open yet
    st = _run(game, st, _both(_act()), game.door_steps - 2)
    assert not bool(st.door_open[i])
    st = _run(game, st, _both(_act()), 1)
    assert bool(st.door_open[i])                                           # open after door_time
    st = _run(game, st, _both(_act(fwd=1)), 30)
    assert ((np.asarray(st.pos[ASSAULT]) > lo) & (np.asarray(st.pos[ASSAULT]) < hi)).all()


def test_interact_needs_reach(game):
    st, i, d = _at_door(game)
    st = st._replace(pos=st.pos.at[ASSAULT].add(jnp.asarray(d["center"] - st.pos[ASSAULT]) * -1.0))  # 1.6 m away
    st = _run(game, st, _both(_act(interact=1)), game.door_steps + 1)
    assert not bool(st.door_open[i]) and not bool(st.door_event[ASSAULT])


def test_door_wont_close_on_someone(game):
    st, i, d = _at_door(game)
    st = st._replace(door_open=st.door_open.at[i].set(True),
                     pos=st.pos.at[ASSAULT].set(jnp.asarray(d["center"], jnp.float32)))
    st = _run(game, st, _both(_act(interact=1)), 1)
    st = _run(game, st, _both(_act()), game.door_steps)
    assert bool(st.door_open[i])


def test_operators_dont_overlap(game):
    st = game.init(jax.random.PRNGKey(3), 0)
    st = st._replace(pos=jnp.asarray([[3.0, 3.0], [5.0, 3.0]]), face=jnp.asarray([0.0, np.pi]))
    st = _run(game, st, _both(_act(fwd=1), _act(fwd=1)), 20)
    assert float(jnp.linalg.norm(st.pos[0] - st.pos[1])) > 2 * CFG.radius - 0.05


def test_round_ends_on_time(game):
    st = game.init(jax.random.PRNGKey(4), 0)
    st = _run(game, st, _both(_act()), game.T - 1)
    assert not bool(st.done)
    assert bool(_run(game, st, _both(_act()), 1).done)


# ── Seeing ───────────────────────────────────────────────────────────────────

def _step(g, st, a):
    return g.step(st, a)[0]


def _plain_game(walls=(), doors=(), cover=(), foot=((10.0, 5.0), (30.0, 25.0))):
    """One hand-made building: lists of (lo, hi) boxes. Doors start shut."""
    def pad(boxes, n):
        lo = np.full((n, 2), -1000.0)
        hi = np.full((n, 2), -1000.0)
        for k, (a, b) in enumerate(boxes):
            lo[k], hi[k] = a, b
        return jnp.asarray(lo[None], jnp.float32), jnp.asarray(hi[None], jnp.float32)
    wl, wh = pad(walls, CFG.max_walls)
    dl, dh = pad(doors, CFG.max_doors)
    cl, ch = pad(cover, CFG.max_cover)
    D = CFG.max_doors
    b = Buildings(wl, wh, dl, dh, jnp.asarray((np.arange(D) < len(doors))[None]), jnp.ones((1, D, 2), bool),
                  jnp.zeros((1, D), bool), cl, ch, jnp.asarray([[[5.0, 5.0]] * 3], jnp.float32), jnp.zeros((1, 3)),
                  jnp.asarray([1]), jnp.full((1, CFG.spawn_points, 2), 20.0), jnp.asarray([foot[0]], jnp.float32),
                  jnp.asarray([foot[1]], jnp.float32),
                  jnp.asarray((np.asarray(foot[0]) + cell_fracs(CFG) * (np.asarray(foot[1]) - np.asarray(foot[0])))[None],
                              jnp.float32))
    return CQBGame(b, CFG)


# a wall along x = 20 from y = 5 to 25, with a 1 m doorway at y 14.5-15.5, and a crate at (24-25, 19-20)
WALLS = [((19.9, 5.0), (20.1, 14.5)), ((19.9, 15.5), (20.1, 25.0))]
DOOR = [((19.9, 14.5), (20.1, 15.5))]
CRATE = [((24.0, 19.0), (25.0, 20.0))]


def _pose(g, a, a_face, d, d_face=0.0, open_door=False):
    st = g.init(jax.random.PRNGKey(0), 0)
    st = st._replace(pos=jnp.asarray([a, d], jnp.float32), face=jnp.asarray([a_face, d_face], jnp.float32),
                     door_open=st.door_open.at[0].set(open_door))
    return st


def test_light_cone():
    g = _plain_game(WALLS, DOOR, CRATE)
    east = 0.0
    assert bool(g.sees(_pose(g, (12, 10), east, (17, 10)), ASSAULT))              # straight ahead
    assert not bool(g.sees(_pose(g, (12, 10), np.pi, (17, 10)), ASSAULT))         # behind me: dark
    assert not bool(g.sees(_pose(g, (12, 10), np.pi / 2, (17, 10)), ASSAULT))     # 90 degrees off
    assert not bool(g.sees(_pose(g, (0.5, 4.0), east, (0.5 + 20.6, 4.0)), ASSAULT))  # beyond 20 m
    # 46 degrees off the centre: the near edge of its body is inside the 45 degree half-cone
    ang = np.deg2rad(46)
    st = _pose(g, (12, 10), 0.0, (12 + 2 * np.cos(ang), 10 + 2 * np.sin(ang)))
    assert bool(g.sees(st, ASSAULT))
    assert not bool(g.sees(st._replace(face=st.face.at[ASSAULT].set(-0.2)), ASSAULT))


def test_walls_doors_and_cover_block_sight():
    g = _plain_game(WALLS, DOOR, CRATE)
    assert not bool(g.sees(_pose(g, (15, 10), 0.0, (25, 10)), ASSAULT))           # wall between
    assert not bool(g.sees(_pose(g, (15, 15), 0.0, (25, 15)), ASSAULT))           # shut door between
    assert bool(g.sees(_pose(g, (15, 15), 0.0, (25, 15), open_door=True), ASSAULT))  # open: seen
    assert not bool(g.sees(_pose(g, (21, 19.5), 0.0, (28, 19.5)), ASSAULT))       # behind the crate
    # through the doorway at an angle: you see a slice of the room, not all of it
    assert bool(g.sees(_pose(g, (17, 14), 0.3, (24, 16.2), open_door=True), ASSAULT))
    assert not bool(g.sees(_pose(g, (17, 14), 0.9, (22, 20), open_door=True), ASSAULT))


def test_seeing_is_not_mutual():
    g = _plain_game()
    st = _step(g, _pose(g, (12, 10), 0.0, (17, 10), d_face=np.pi / 2), jnp.zeros((2, 6)))
    assert bool(st.seen[ASSAULT]) and not bool(st.seen[DEFENSE])


def test_memory_of_the_other():
    g = _plain_game(WALLS, DOOR, CRATE)
    st = g.init(jax.random.PRNGKey(0), 0)
    assert int(st.since[ASSAULT]) == g.T                                            # never seen yet
    st = _step(g, _pose(g, (15, 15), 0.0, (25, 15), open_door=True), jnp.zeros((2, 6)))
    assert bool(st.seen[ASSAULT]) and int(st.since[ASSAULT]) == 0
    st = st._replace(pos=st.pos.at[DEFENSE].set(jnp.asarray([25.0, 10.0])))          # it steps behind the wall
    st = _step(g, st, jnp.zeros((2, 6)))
    st = _step(g, st, jnp.zeros((2, 6)))
    assert not bool(st.seen[ASSAULT]) and int(st.since[ASSAULT]) == 2
    assert np.allclose(st.mem[ASSAULT], [25.0, 15.0], atol=1e-5)                    # where it was last seen


def test_seen_map():
    """Outside a shut door you've seen none of the building; open it and look in: some of it."""
    g = _plain_game(WALLS + [((10, 4.9), (30, 5.1)), ((10, 24.9), (30, 25.1)), ((9.9, 5), (10.1, 25)),
                             ((29.9, 5), (30.1, 25))], DOOR)
    st = _step(g, _pose(g, (19, 15), 0.0, (28, 7), d_face=np.pi), jnp.zeros((2, 6)))
    cells = np.asarray(g.cells(0))
    right = cells[:, 0] > 20
    assert not np.asarray(st.cleared[ASSAULT])[right].any()
    st = _step(g, st._replace(door_open=st.door_open.at[0].set(True)), jnp.zeros((2, 6)))
    seen = np.asarray(st.cleared[ASSAULT])
    assert seen[right].any() and not seen[right].all()                              # a slice through the door
    st = _step(g, st._replace(door_open=st.door_open.at[0].set(False)), jnp.zeros((2, 6)))
    assert (np.asarray(st.cleared[ASSAULT]) == seen).all()                          # it remembers


# ── Shooting ─────────────────────────────────────────────────────────────────

def _fire(**kw):
    a = _act(**kw)
    a[FIRE] = 1
    return a


def _hit_rate(g, st, acts, n=4000):
    """Fraction of n tries (different luck) in which this one decision takes Defense down."""
    if g not in _HIT_FNS:
        _HIT_FNS[g] = jax.jit(jax.vmap(lambda st, a, k: g.step(st._replace(key=k), a)[0].alive[DEFENSE],
                                       in_axes=(None, None, 0)))
    out = _HIT_FNS[g](st, acts, jax.random.split(jax.random.PRNGKey(7), n))
    return 1 - float(np.mean(np.asarray(out)))


_HIT_FNS = {}


def test_hit_chance_falls_with_distance_moving_and_turning():
    g = _plain_game()
    for dist in (2.0, 10.0):
        st = _pose(g, (12, 10), 0.0, (12 + dist, 10), d_face=np.pi / 2)
        want = CFG.hit_close - CFG.hit_falloff * dist
        assert abs(_hit_rate(g, st, _both(_fire())) - want) < 0.03
    st = _pose(g, (12, 10), 0.0, (17, 10), d_face=np.pi / 2)
    moving = st._replace(vel=st.vel.at[ASSAULT].set(jnp.asarray([CFG.walk, 0.0])))   # walking at it, full speed
    want = (CFG.hit_close - CFG.hit_falloff * (5 - CFG.walk * CFG.dt)) * (1 - CFG.move_penalty)
    assert abs(_hit_rate(g, moving, _both(_fire(fwd=1))) - want) < 0.03
    strafing = st._replace(vel=st.vel.at[ASSAULT].set(jnp.asarray([0.0, CFG.walk])))  # strafing tops out at 70%
    want = (CFG.hit_close - CFG.hit_falloff * 5) * (1 - CFG.move_penalty * CFG.sideways)
    assert abs(_hit_rate(g, strafing, _both(_fire(side=1))) - want) < 0.03
    turning = st._replace(face=st.face.at[ASSAULT].set(-np.deg2rad(CFG.turn_rate) * CFG.dt))  # turns onto it
    want = (CFG.hit_close - CFG.hit_falloff * 5) * (1 - CFG.turn_penalty)
    assert abs(_hit_rate(g, turning, _both(_fire(turn=1))) - want) < 0.03


def test_no_hit_off_aim_unlit_through_walls_or_running():
    g = _plain_game(WALLS, DOOR, CRATE)
    off = _pose(g, (12, 10), np.deg2rad(15), (17, 10), d_face=np.pi / 2)             # lit, but 15 degrees off
    assert bool(g.sees(off, ASSAULT)) and _hit_rate(g, off, _both(_fire()), 200) == 0
    assert _hit_rate(g, _pose(g, (15, 10), 0.0, (25, 10)), _both(_fire()), 200) == 0      # wall
    assert _hit_rate(g, _pose(g, (15, 15), 0.0, (25, 15)), _both(_fire()), 200) == 0      # shut door
    assert _hit_rate(g, _pose(g, (21, 19.5), 0.0, (28, 19.5)), _both(_fire()), 200) == 0  # crate
    st = _pose(g, (12, 10), 0.0, (14, 10), d_face=np.pi / 2)
    assert _hit_rate(g, st, _both(_fire(run=1)), 200) == 0                                # can't fire running
    assert not bool(g.step(st, _both(_fire(run=1)))[0].fired[ASSAULT])


def _sure_game(walls=()):
    from dataclasses import replace
    cfg = replace(CFG, hit_falloff=0.0, hit_close=1.0)
    return CQBGame(_plain_game(walls).b, cfg)


def test_one_hit_ends_the_round():
    g = _sure_game()
    st, r = g.step(_pose(g, (12, 10), 0.0, (15, 10), d_face=np.pi / 2), _both(_fire()))
    assert not bool(st.alive[DEFENSE]) and bool(st.done) and int(st.outcome) == 1 and float(r) == 1.0
    st, r = g.step(_pose(g, (12, 10), np.pi / 2, (15, 10), d_face=np.pi), _both(_act(), _fire()))
    assert not bool(st.alive[ASSAULT]) and int(st.outcome) == -1 and float(r) == -1.0
    # both fire at once: both go down, nobody wins
    st, r = g.step(_pose(g, (12, 10), 0.0, (15, 10), d_face=np.pi), _both(_fire(), _fire()))
    assert not np.asarray(st.alive).any() and int(st.outcome) == 0 and float(r) == 0.0
    # over means over: nothing moves and no more reward
    after, r2 = g.step(st, _both(_fire(fwd=1), _fire(fwd=1)))
    assert np.allclose(after.pos, st.pos) and float(r2) == 0.0 and int(after.t) == int(st.t)


def test_running_out_of_time_is_a_loss_for_assault():
    g = _plain_game()
    st = _pose(g, (12, 10), np.pi, (25, 20))
    total, step = 0.0, jax.jit(g.step)
    for _ in range(g.T):
        st, r = step(st, _both(_act()))
        total += float(r)
    assert bool(st.done) and int(st.outcome) == -1 and abs(total + 1.0) < 1e-6


def test_muzzle_flash_gives_the_shooter_away():
    g = _plain_game(WALLS, DOOR, CRATE)
    # Defense faces away from Assault (it can't see it); Assault fires into the dark
    st = _pose(g, (12, 10), np.pi / 2, (16, 10), d_face=0.0)
    st, _ = g.step(st, _both(_fire()))
    assert bool(st.fired[ASSAULT]) and bool(st.flash_seen[DEFENSE]) and not bool(st.seen[DEFENSE])
    assert np.allclose(st.mem[DEFENSE], st.pos[ASSAULT]) and int(st.since[DEFENSE]) == 0
    st, _ = g.step(st, _both(_act()))
    assert bool(st.flash_seen[DEFENSE])                                             # 0.2 s = 2 decisions
    st, _ = g.step(st, _both(_act()))
    assert not bool(st.flash_seen[DEFENSE])
    # behind a wall, the flash isn't seen (it will be heard: 1g)
    st, _ = g.step(_pose(g, (15, 10), np.pi / 2, (25, 10)), _both(_fire()))
    assert not bool(st.flash_seen[DEFENSE])


# ── Hearing ──────────────────────────────────────────────────────────────────

STEPS, DOORS, SHOTS = 0, 1, 2


def _heard(g, a_pos, d_pos, d_vel=(0.0, 0.0), acts=None, open_door=False):
    """Assault listens (facing away, so it can't see); Defense moves at d_vel. -> Assault's (3, 3) sound."""
    st = _pose(g, a_pos, np.pi, d_pos, open_door=open_door)
    st = st._replace(vel=st.vel.at[DEFENSE].set(jnp.asarray(d_vel, jnp.float32)))
    # keep Defense's speed this decision: ask for the same velocity in its own frame (it faces east)
    v = np.asarray(d_vel, np.float32)
    speed = np.linalg.norm(v)
    run = speed > CFG.walk + 0.1
    a1 = _act(fwd=v[0] / (CFG.run if run else CFG.walk), side=v[1] / (CFG.run if run else CFG.walk) / CFG.sideways,
              run=1 if run else 0) if acts is None else acts
    st, _ = g.step(st, _both(_act(), a1))
    return np.asarray(st.sound[ASSAULT]), st


def test_footsteps_carry_by_speed():
    g = _plain_game()
    run = (CFG.run, 0.0)
    assert _heard(g, (12, 10), (12 + 8, 10), run)[0][STEPS, 2] > 0              # running: 10 m
    assert _heard(g, (12, 10), (12 + 11, 10), run)[0][STEPS, 2] == 0
    walk = (CFG.walk, 0.0)
    assert _heard(g, (12, 10), (12 + 3, 10), walk)[0][STEPS, 2] > 0             # walking: 4 m
    assert _heard(g, (12, 10), (12 + 5, 10), walk)[0][STEPS, 2] == 0
    assert _heard(g, (12, 10), (12 + 1.5, 10), (0.0, 0.0))[0][STEPS, 2] == 0     # standing still: silent
    louder = _heard(g, (12, 10), (12 + 3, 10), run)[0][STEPS, 2]
    assert louder > _heard(g, (12, 10), (12 + 7, 10), run)[0][STEPS, 2]         # nearer is louder


def test_walls_muffle_sound():
    g = _plain_game(WALLS, DOOR)
    run = (CFG.run, 0.0)
    # one wall in the way: running carries 10 x 2/3 = 6.7 m (Defense runs east, away from the wall)
    assert _heard(g, (17, 10), (17 + 6, 10), run)[0][STEPS, 2] > 0
    assert _heard(g, (17, 10), (17 + 7, 10), run)[0][STEPS, 2] == 0
    assert _heard(g, (17, 15), (17 + 7, 15), run, open_door=True)[0][STEPS, 2] > 0   # through the open doorway
    assert int(_plain_game(WALLS, DOOR).walls_between(_pose(g, (17, 10), 0, (25, 10)), jnp.asarray([17.0, 10.0]),
                                                      jnp.asarray([25.0, 10.0]))) == 1


def test_doors_and_shots_are_heard():
    g = _plain_game(WALLS, DOOR)
    # Defense opens the door from 0.8 m on its side, 9 m from Assault, through the wall line
    snd, _ = _heard(g, (12, 15), (20.8, 15), (0.0, 0.0), acts=_act(interact=1))
    assert snd[DOORS, 2] > 0
    # a shot carries across the building, through walls
    snd, _ = _heard(g, (11, 6), (29, 24), (0.0, 0.0), acts=_fire())
    assert snd[SHOTS, 2] > 0 and snd[STEPS, 2] == 0


def test_rough_direction_and_short_memory():
    g = _plain_game()
    snd, st = _heard(g, (12, 10), (12 + 5, 10 + 5), (CFG.run, 0.0))
    ang = np.degrees(np.arctan2(snd[STEPS, 1], snd[STEPS, 0]))
    assert abs(ang - 45) <= 360 / CFG.sound_dirs / 2 + 1e-3                   # north-east, roughly
    step = jax.jit(g.step)
    st = st._replace(pos=st.pos.at[DEFENSE].set(jnp.asarray([35.0, 25.0])))    # far away now, standing still
    for k in range(1, int(CFG.sound_memory / CFG.dt) + 1):
        st, _ = step(st, _both(_act()))
        heard = float(st.sound[ASSAULT, STEPS, 2]) > 0
        assert heard == (k < CFG.sound_memory / CFG.dt), k                     # kept for 0.5 s, then gone


# ── What a brain gets ────────────────────────────────────────────────────────

def test_observation_size_and_range(game):
    assert CFG.obs_size == 107                                                   # docs/cqb-plan.md 2c

    def run(key):
        st = game.init(key)

        def tick(st, k):
            st, _ = game.step(st, jax.random.uniform(k, (2, 6), minval=-1, maxval=1))
            return st, jnp.stack([game.observe(st, ASSAULT), game.observe(st, DEFENSE)])
        return jax.lax.scan(tick, st, jax.random.split(key, 150))[1]

    obs = np.asarray(jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(9), 32)))
    assert obs.shape[-1] == CFG.obs_size and np.isfinite(obs).all()
    assert np.abs(obs).max() <= 2.0 + 1e-5


def test_the_fog_is_real():
    """Assault can't see or hear Defense: wherever Defense is, Assault gets exactly the same numbers."""
    g = _plain_game(WALLS, DOOR, CRATE)
    obs = []
    for d in [(25, 10), (25, 20), (28, 7)]:
        st = _step(g, _pose(g, (15, 10), np.pi, d, d_face=np.pi), jnp.zeros((2, 6)))
        assert not bool(st.seen[ASSAULT]) and float(st.sound[ASSAULT, :, 2].max()) == 0
        obs.append(np.asarray(g.observe(st, ASSAULT)))
    assert np.array_equal(obs[0], obs[1]) and np.array_equal(obs[0], obs[2])
    # once it's in the light, it shows
    st = _step(g, _pose(g, (15, 10), np.pi, (12, 10)), jnp.zeros((2, 6)))
    o = np.asarray(g.observe(st, ASSAULT))
    e = CFG.n_rays
    assert o[e] == 1.0 and abs(o[e + 1] * CFG.sight - 3.0) < 1e-4 and abs(o[e + 2]) < 1e-4  # 3 m straight ahead


def test_lasers_doors_and_map_in_my_frame():
    g = _plain_game(WALLS, DOOR, CRATE)
    st = _step(g, _pose(g, (15, 10), 0.0, (35, 25)), jnp.zeros((2, 6)))
    o = np.asarray(g.observe(st, ASSAULT))
    lasers = o[:CFG.n_rays] * CFG.sight
    assert abs(lasers[0] - 4.9) < 1e-4                                  # straight ahead: the wall face at x 19.9
    assert abs(lasers[CFG.n_rays // 4] - CFG.sight) < 1e-4              # to my left (north): open for 20 m
    assert abs(lasers[CFG.n_rays // 2] - 15.0) < 1e-4                   # behind me: the yard's edge
    k = CFG.n_rays + 10 + 9 + 7                                         # the nearest door
    assert np.allclose(o[k:k + 2] * 10, [5.0, 5.0], atol=1e-4)          # 5 m ahead, 5 m to my left
    assert o[k + 2] == 0.0                                              # shut
    cells = CFG.seen_grid[0] * CFG.seen_grid[1]
    assert np.array_equal(o[-1 - cells:-1], np.asarray(st.cleared[ASSAULT], np.float32))
    turned = _step(g, _pose(g, (15, 10), np.pi / 2, (35, 25)), jnp.zeros((2, 6)))
    o2 = np.asarray(g.observe(turned, ASSAULT))
    assert np.allclose(o2[k:k + 2] * 10, [5.0, -5.0], atol=1e-4)        # facing north: the door is ahead-right


def test_light_fan_matches_exact_sight(game):
    """The seen map uses a fan of 16 rays instead of exact sight lines (3x cheaper). Check they agree."""
    def one(key):
        k1, k2, k3 = jax.random.split(key, 3)
        st = game.init(k1)
        p = st.pos[DEFENSE]
        face = jax.random.uniform(k2, (), minval=-np.pi, maxval=np.pi)
        lo, hi = game.boxes(st)
        q = game.b.foot_lo[st.building] + jax.random.uniform(k3, (256, 2)) * (game.b.foot_hi[st.building]
                                                                               - game.b.foot_lo[st.building])
        exact = game.in_cone(p, face, q) & _clear(p, q, lo, hi)
        return exact, game.lit(p, face, q, lo, hi)
    exact, fan = map(np.asarray, jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(11), 128)))
    assert exact.sum() > 1000                                   # plenty of lit points to compare
    found = (fan & exact).sum() / exact.sum()                   # of the really lit points, the fan marks...
    right = (fan & exact).sum() / fan.sum()                     # ...and of what the fan marks, really lit
    print(f"lit points {exact.sum()}: fan finds {found:.1%}, fan is right {right:.1%}")
    assert found > 0.95 and right > 0.95, (found, right)


# ── Flashbangs (ep9 3t) ──────────────────────────────────────────────────────

from dataclasses import replace as _replace  # noqa: E402

BANG = _replace(CFG, flashbangs=(2, 0))


def _bang_game(walls=WALLS, doors=DOOR, cover=CRATE, cfg=BANG):
    return CQBGame(_plain_game(walls, doors, cover).b, cfg)


def _throw_from(g, p, face, open_door=False):
    """Assault at p facing `face` throws; Defense far away in a corner. -> where it lands."""
    st = _step(g, _pose(g, p, face, (11, 24), open_door=open_door), _both(_act(throw=1)))
    assert int(st.bang_timer[ASSAULT]) == g.fuse_steps and int(st.bangs[ASSAULT]) == 1
    return np.asarray(st.bang_pos[ASSAULT])


def test_no_flashbangs_unless_the_game_has_them():
    assert CFG.flashbangs == (0, 0) and CFG.obs_size == 107 and BANG.obs_size == 115
    g = _plain_game(WALLS, DOOR, CRATE)
    st = _run(g, _pose(g, (12, 10), 0.0, (25, 20)), _both(_act(throw=1), _act(throw=1)), 30)
    assert np.asarray(st.bangs).tolist() == [0, 0] and int(np.asarray(st.bang_timer).max()) == 0
    assert int(np.asarray(st.blind).max()) == 0
    # Defense has none in this game either
    g = _bang_game()
    st = _step(g, _pose(g, (12, 10), 0.0, (25, 20)), _both(_act(), _act(throw=1)))
    assert int(st.bang_timer[DEFENSE]) == 0 and int(st.bangs[DEFENSE]) == 0


def test_throw_lands_ahead_or_short_of_walls_and_shut_doors():
    g = _bang_game()
    assert np.allclose(_throw_from(g, (12, 10), 0.0), (18, 10), atol=1e-4)            # open floor: 6 m
    assert np.allclose(_throw_from(g, (15, 10), 0.0), (19.6, 10), atol=1e-3)          # wall at 19.9: 0.3 m short
    assert np.allclose(_throw_from(g, (17, 15), 0.0), (19.6, 15), atol=1e-3)          # shut door: the same
    assert np.allclose(_throw_from(g, (17, 15), 0.0, open_door=True), (23, 15), atol=1e-3)  # open: through it
    assert np.allclose(_throw_from(g, (20.5, 19.5), 0.0), (26.5, 19.5), atol=1e-3)    # over the crate
    assert np.allclose(_throw_from(g, (19.6, 10), 0.0), (19.6, 10), atol=1e-3)        # pressed to a wall: at your feet


def test_one_live_flashbang_at_a_time_and_only_two():
    g = _bang_game()
    st = _pose(g, (12, 10), 0.0, (11, 24))
    step = jax.jit(g.step)
    left, live = [], []
    for _ in range(40):
        st, _ = step(st, _both(_act(throw=1)))
        left.append(int(st.bangs[ASSAULT]))
        live.append(int(st.bang_timer[ASSAULT]))
    f = g.fuse_steps
    assert left[:f] == [1] * f and left[f:] == [0] * (40 - f)       # the second goes as soon as the first is off
    assert live[0] == f and live[f - 1] == 1 and live[f] == f and live[2 * f] == 0


def _bang_on(g, st, n=None):
    """Assault throws, then everyone stands still until it goes off (fuse_steps decisions after the throw).
    -> the state right after the bang (or after n decisions)."""
    st = _step(g, st, _both(_act(throw=1)))
    step = jax.jit(g.step)
    for _ in range(g.fuse_steps if n is None else n):
        st, _ = step(st, _both(_act()))
    return st


def test_bang_goes_off_after_the_fuse_and_blinds_who_should_be():
    g = _bang_game()
    # Assault (12, 10) throws east: it lands at (18, 10). Defense 4 m past it, facing it: 2 s blind
    st = _bang_on(g, _pose(g, (12, 10), 0.0, (18, 14), d_face=-np.pi / 2), n=g.fuse_steps - 1)
    assert int(st.blind[DEFENSE]) == 0 and not bool(st.bang_now[ASSAULT])         # 1.4 s: not yet
    st = _step(g, st, _both(_act()))
    assert bool(st.bang_now[ASSAULT]) and int(st.blind[DEFENSE]) == g.blind_steps  # 1.5 s
    assert int(st.blind[ASSAULT]) == g.blind_steps     # the thrower watched it from 6 m: blinded too


def _blind_after(g, d_pos, d_face, open_door=False, a_pos=(12, 10), a_face=0.0):
    st = _bang_on(g, _pose(g, a_pos, a_face, d_pos, d_face=d_face, open_door=open_door))
    assert bool(st.bang_now[ASSAULT])
    return int(st.blind[DEFENSE]), int(st.blind[ASSAULT])


def test_who_gets_blinded():
    g = _bang_game()
    full, half = g.blind_steps, g.blind_steps // 2
    # lands at (18, 10)
    assert _blind_after(g, (18, 14), -np.pi / 2)[0] == full                    # 4 m, facing it
    assert _blind_after(g, (18, 14), np.pi / 2)[0] == full                     # 4 m, facing away: the same
    away = _bang_game(cfg=_replace(BANG, blind_away=0.5))                       # (the option: facing away helps)
    assert _blind_after(away, (18, 14), np.pi / 2)[0] == half and _blind_after(away, (18, 14), -np.pi / 2)[0] == full
    assert _blind_after(g, (18, 17.5), -np.pi / 2)[0] == 0                     # 7.5 m: out of range
    assert _blind_after(g, (14, 10.5), 0.0, a_pos=(12, 10), a_face=0.0)[0] == full   # beside the thrower
    assert _blind_after(g, (23, 10), np.pi)[0] == 0                            # behind the wall
    # through the doorway (lands at (19.6, 15) when it's shut, (23, 15) when it's open)
    assert _blind_after(g, (17, 15), 0.0, a_pos=(17.5, 15.5), a_face=0.0)[0] == full  # shut: at the door, my side
    assert _blind_after(g, (23, 17), -np.pi / 2, a_pos=(17, 15), open_door=True)[0] == full
    assert _blind_after(g, (21, 20), -np.pi / 2, a_pos=(17, 15))[0] == 0           # shut: the other side is safe
    # cover doesn't shield you: lands at (26.5, 19.5)... thrown from (20.5, 19.5) over the crate at 24-25
    assert _blind_after(g, (24.5, 21.5), -np.pi / 2, a_pos=(20.5, 19.5))[0] == full
    # the thrower too, if it's close: turned away (120 degrees in 0.5 s) when it goes off, half (with the option)
    g = away
    st = _step(g, _pose(g, (15, 10), 0.0, (11, 24)), _both(_act(throw=1)))
    step = jax.jit(g.step)
    for i in range(g.fuse_steps):
        st, _ = step(st, _both(_act(turn=1 if i < 5 else 0)))
    assert bool(st.bang_now[ASSAULT]) and int(st.blind[ASSAULT]) == half


def test_blind_means_blind():
    g = _sure_game()
    g = CQBGame(g.b, _replace(g.cfg, flashbangs=(2, 0)))
    st = _pose(g, (12, 10), 0.0, (15, 10), d_face=np.pi)                          # face to face, 3 m
    st = _step(g, st, _both(_act()))
    assert bool(st.seen[ASSAULT]) and bool(st.seen[DEFENSE])
    st = st._replace(blind=jnp.asarray([0, 5], jnp.int32), cleared=jnp.zeros_like(st.cleared))
    st2 = _step(g, st, _both(_act()))
    assert bool(st2.seen[ASSAULT]) and not bool(st2.seen[DEFENSE])                 # can't see...
    assert not np.asarray(st2.cleared[DEFENSE]).any() and np.asarray(st2.cleared[ASSAULT]).any()  # ...or look...
    assert float(g.hit_chance(st2, DEFENSE)) == 0.0 and float(g.hit_chance(st2, ASSAULT)) > 0.9  # ...or hit
    st3, _ = g.step(st, _both(_act(), _fire()))
    assert bool(st3.alive[ASSAULT])                                                # it fires and can't hit
    st3, _ = g.step(st, _both(_fire()))
    assert not bool(st3.alive[DEFENSE])                                            # Assault can
    # Assault's muzzle flash isn't seen by a blinded Defense
    st4 = _step(g, st._replace(face=st.face.at[DEFENSE].set(0.0)), _both(_fire(turn=1)))
    assert not bool(st4.flash_seen[DEFENSE])
    # it wears off: 5 decisions
    step = jax.jit(g.step)
    st5 = st._replace(blind=jnp.asarray([0, 5], jnp.int32))
    seen = []
    for _ in range(6):
        st5, _ = step(st5, _both(_act()))
        seen.append(bool(st5.seen[DEFENSE]))
    assert seen == [False] * 4 + [True] * 2


def test_bang_is_heard_by_the_other_side_through_walls():
    g = _bang_game()
    # lands at (18, 10); Defense behind the wall at (25, 10), 7 m away: heard (100 m, a wall cuts it to 67)
    st = _bang_on(g, _pose(g, (12, 10), 0.0, (25, 10), d_face=np.pi / 2))
    hb = np.asarray(st.bang_heard[DEFENSE])
    assert hb[2] > 0.85 and np.allclose(hb[:2], (-1, 0), atol=1e-5)               # from the west
    assert int(st.blind[DEFENSE]) == 0
    assert float(st.bang_heard[ASSAULT, 2]) == 0                                    # its own isn't in "the other's"
    step = jax.jit(g.step)
    for _ in range(g.sound_steps - 1):
        st, _ = step(st, _both(_act()))
    assert float(st.bang_heard[DEFENSE, 2]) > 0
    st, _ = step(st, _both(_act()))
    assert float(st.bang_heard[DEFENSE, 2]) == 0                                    # 0.5 s later it's gone
    # nothing heard before the bang, wherever the flashbang lies (the throw is silent)
    st = _step(g, _pose(g, (12, 10), 0.0, (25, 10)), _both(_act(throw=1)))
    assert float(st.bang_heard[DEFENSE, 2]) == 0 and float(st.sound[DEFENSE, :, 2].max()) == 0


def test_observation_with_flashbangs():
    """8 more numbers at the end; the first 107 are exactly the game without flashbangs (brains from
    before read those); in range; and Assault's live flashbang doesn't show in Defense's numbers."""
    g0, g = _plain_game(WALLS, DOOR, CRATE), _bang_game()
    st = _pose(g, (12, 10), 0.3, (25, 20), d_face=np.pi)
    for _ in range(3):
        st = _step(g, st, _both(_act(fwd=1, turn=0.2, throw=1)))
        st0 = st._replace(bangs=jnp.zeros(2, jnp.int32))
        for w in (ASSAULT, DEFENSE):
            o, o0 = np.asarray(g.observe(st, w)), np.asarray(g0.observe(st0, w))
            assert o.shape == (115,) and np.array_equal(o[:107], o0)
    o = np.asarray(g.observe(st, ASSAULT))[107:]
    assert o[0] == 0.5 and o[1] == 0 and o[4] > 0.8                                   # 1 left, not blind, fuse
    d = np.asarray(st.bang_pos[ASSAULT] - st.pos[ASSAULT])
    c, s_ = np.cos(float(st.face[ASSAULT])), np.sin(float(st.face[ASSAULT]))
    assert np.allclose(o[2:4] * CFG.sight, (c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]), atol=1e-4)  # where, my frame
    # Defense: the same numbers wherever Assault's flashbang lies (it hasn't gone off)
    ds = [np.asarray(g.observe(st._replace(bang_pos=st.bang_pos.at[ASSAULT].set(jnp.asarray(q, jnp.float32))),
                               DEFENSE)) for q in ((14, 12), (22, 18), (27, 8))]
    assert np.array_equal(ds[0], ds[1]) and np.array_equal(ds[0], ds[2])

    def run(key):
        st = g.init(key)

        def tick(st, k):
            st, _ = g.step(st, jax.random.uniform(k, (2, 7), minval=-1, maxval=1))
            return st, jnp.stack([g.observe(st, ASSAULT), g.observe(st, DEFENSE)])
        return jax.lax.scan(tick, st, jax.random.split(key, 150))[1]
    big = CQBGame(make_buildings(SEEDS, BANG), BANG)
    g = big
    obs = np.asarray(jax.jit(jax.vmap(run))(jax.random.split(jax.random.PRNGKey(9), 16)))
    assert np.isfinite(obs).all() and np.abs(obs).max() <= 2.0 + 1e-5
    assert obs[..., 107 + 1].max() > 0                                             # somebody got blinded


def test_speed():
    """Decisions per second on this machine (scripts/cqb_speed.py has the full measurement). The floor
    is far below what the laptop does (~49k at 4096 rounds) and only catches big slowdowns."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cqb_speed", "scripts/cqb_speed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    g = CQBGame(make_buildings(range(64), CFG), CFG)
    r = mod.rate(g, (2, 6), lambda s: jnp.stack([g.observe(s, 0), g.observe(s, 1)]), 1024, steps=64, repeats=2)
    assert r > 15_000, f"{r:.0f} decisions/s"


# ── Scripted players ─────────────────────────────────────────────────────────

from ailearns.games import cqb_scripts as S  # noqa: E402

TEST = list(range(100_000, 100_032))


@pytest.fixture(scope="module")
def test_game():
    return CQBGame(make_buildings(TEST, CFG), CFG)


def test_walker_finds_someone_standing_still(test_game):
    r = S.play_rounds(test_game, "walker", "dummy", np.arange(len(TEST)), seed=1)
    assert (r["outcome"] == 1).mean() >= 0.95
    r = S.play_rounds(test_game, "random", "dummy", np.arange(len(TEST)), seed=1)
    assert (r["outcome"] == 1).mean() <= 0.1                    # flailing about rarely finds anyone


def test_camper_stays_put_and_holds_the_door(test_game):
    r = S.play_rounds(test_game, "walker", "camper", np.arange(len(TEST)), seed=1, record=True)
    pos = r["history"].pos[:, :, DEFENSE]
    assert np.abs(pos - pos[:, :1]).max() < 1e-4
    assert (r["outcome"] == -1).mean() > (r["outcome"] == 1).mean()  # camping beats walking straight in


@pytest.mark.parametrize("name,who", [("walker", ASSAULT), ("camper", DEFENSE), ("wanderer", DEFENSE)])
def test_scripts_dont_peek(test_game, name, who):
    """Like the fog test: where the other one stands (unseen, silent) doesn't change what a script
    does. Checked in many situations: the script somewhere inside, facing anywhere, with a random
    half of the building already looked at."""
    g = test_game
    other = 1 - who
    differ, checked = 0, 0
    for k in range(48):
        kb, ks, kf, kc, ko = jax.random.split(jax.random.PRNGKey(k), 5)
        bi = k % len(TEST)
        st = g.init(kb, bi)
        me = g.b.spawn[bi, jax.random.randint(ks, (), 0, CFG.spawn_points)]
        st = st._replace(pos=st.pos.at[who].set(me), face=st.face.at[who].set(jax.random.uniform(kf, (), minval=-3, maxval=3)),
                         cleared=st.cleared.at[who].set(jax.random.uniform(kc, (st.cleared.shape[1],)) < 0.5),
                         door_open=st.door_open | g.b.door_valid[bi])
        acts = []
        for q in g.b.spawn[bi, jax.random.randint(ko, (6,), 0, CFG.spawn_points)]:
            moved = st._replace(pos=st.pos.at[other].set(q))
            if not bool(g.sees(moved, who)):
                acts.append(np.asarray(S.SCRIPTS[name](g, moved, who, jax.random.PRNGKey(9))))
        if len(acts) >= 2:
            checked += 1
            differ += not all(np.array_equal(acts[0], a) for a in acts[1:])
    assert checked >= 20 and differ == 0, (checked, differ)


@pytest.fixture(scope="module")
def bang_test_game():
    return CQBGame(make_buildings(TEST, BANG), BANG)


def test_walker_flash_flashes_rooms_before_going_in(bang_test_game):
    g = bang_test_game
    r = S.play_rounds(g, "walker_flash", "dummy", np.arange(len(TEST)), seed=1, record=True)
    h = r["history"]
    assert (r["outcome"] == 1).mean() >= 0.9                                     # it still finds someone
    used = BANG.flashbangs[ASSAULT] - h.bangs[:, -1, ASSAULT]
    assert used.mean() >= 1.0
    assert (h.blind[:, :, ASSAULT] > 0).any(1).mean() <= 0.15                    # it seldom blinds itself
    # each bang is inside the building, in a room Assault hadn't looked into when it threw
    for i, s in enumerate(TEST):
        b = make_building(s, BANG)
        for t in np.where(h.bang_now[i, :, ASSAULT])[0]:
            q = h.bang_pos[i, t, ASSAULT]
            room = [k for k, (lo, hi) in enumerate(b["rooms"]) if (q > lo).all() and (q < hi).all()]
            assert room, (s, t, q)
            thrown = t - g.fuse_steps
            cells = np.asarray(g.b.cell_room[i]) == room[0]
            assert not h.cleared[i, thrown, ASSAULT][cells].all(), (s, t)
    # with no flashbangs it's the walker
    r0 = S.play_rounds(CQBGame(make_buildings(TEST, CFG), CFG), "walker_flash", "camper", np.arange(len(TEST)), seed=1)
    r1 = S.play_rounds(CQBGame(make_buildings(TEST, CFG), CFG), "walker", "camper", np.arange(len(TEST)), seed=1)
    assert np.array_equal(r0["outcome"], r1["outcome"]) and np.array_equal(r0["t"], r1["t"])


def test_walker_flash_doesnt_peek(bang_test_game):
    """As test_scripts_dont_peek, with flashbangs in hand and doors open (so it would be throwing)."""
    g = bang_test_game
    differ, checked = 0, 0
    for k in range(48):
        kb, ks, kf, kc, ko = jax.random.split(jax.random.PRNGKey(k), 5)
        bi = k % len(TEST)
        st = g.init(kb, bi)
        me = g.b.spawn[bi, jax.random.randint(ks, (), 0, CFG.spawn_points)]
        st = st._replace(pos=st.pos.at[ASSAULT].set(me),
                         face=st.face.at[ASSAULT].set(jax.random.uniform(kf, (), minval=-3, maxval=3)),
                         cleared=st.cleared.at[ASSAULT].set(jax.random.uniform(kc, (st.cleared.shape[1],)) < 0.5),
                         door_open=st.door_open | g.b.door_valid[bi])
        acts = []
        for q in g.b.spawn[bi, jax.random.randint(ko, (6,), 0, CFG.spawn_points)]:
            moved = st._replace(pos=st.pos.at[DEFENSE].set(q))
            if not bool(g.sees(moved, ASSAULT)):
                acts.append(np.asarray(S.walker_flash(g, moved, ASSAULT, jax.random.PRNGKey(9))))
        if len(acts) >= 2:
            checked += 1
            differ += not all(np.array_equal(acts[0], a) for a in acts[1:])
    assert checked >= 20 and differ == 0, (checked, differ)


def test_drill_start_is_beside_defenses_door(test_game):
    """init_stacked: Assault outside Defense's room, close beside the doorway Defense's room is left by, facing it,
    not inside anything, not in Defense's sight (it isn't facing that way, or the wall's in between), and the rest
    of the house counted as looked at."""
    g = test_game
    st = jax.vmap(lambda k: g.init_stacked(k))(jax.random.split(jax.random.PRNGKey(3), 256))
    hidden, opened = 0, 0
    for i in range(256):
        s = jax.tree.map(lambda x: x[i], st)
        bi = int(s.building)
        plan = make_building(TEST[bi], CFG)
        a, d = np.asarray(s.pos[ASSAULT]), np.asarray(s.pos[DEFENSE])
        room = lambda p: next((k for k, (lo, hi) in enumerate(plan["rooms"]) if (p > lo).all() and (p < hi).all()), -1)
        r = room(d)
        assert room(a) != r
        k = int(np.asarray(g.b.next_door[bi, r, CFG.max_rooms]))
        door = plan["doors"][k]
        gap = np.linalg.norm(a - np.clip(a, door["lo"], door["hi"]))
        assert 0.3 < gap < 1.2, (TEST[bi], gap)
        to = (door["lo"] + door["hi"]) / 2 - a
        assert abs(np.arctan2(np.sin(np.arctan2(to[1], to[0]) - float(s.face[ASSAULT])),
                              np.cos(np.arctan2(to[1], to[0]) - float(s.face[ASSAULT])))) < 1e-3
        lo = np.vstack([plan["wall_lo"], plan["cover_lo"]])
        hi = np.vstack([plan["wall_hi"], plan["cover_hi"]])
        assert (np.linalg.norm(a - np.clip(a, lo, hi), axis=1) >= CFG.radius - 1e-3).all()
        cells = np.asarray(g.b.cell_room[bi])
        seen = np.asarray(s.cleared[ASSAULT])
        assert seen[cells != r].all() and not seen[cells == r].all()     # (its light may already reach in a cell)
        hidden += not bool(s.seen[DEFENSE]) and not bool(s.seen[ASSAULT])
        opened += bool(s.door_open[k])
    assert hidden >= 0.85 * 256 and 0.35 * 256 < opened < 0.65 * 256


def test_buildings_cache_and_parallel(tmp_path, monkeypatch):
    """make_buildings: a cached set equals a fresh one; the key follows the seeds and the generator's settings, not
    the game's other settings; the parallel path (128+) gives the same buildings as one at a time."""
    from dataclasses import replace
    from ailearns.games import cqb
    monkeypatch.setenv("AILEARNS_BUILDINGS", str(tmp_path))
    seeds = list(range(64))
    fresh = cqb.make_buildings(seeds, CFG)
    assert len(list(tmp_path.glob("buildings-*.npz"))) == 1
    cached = cqb.make_buildings(seeds, CFG)
    for k in cqb.Buildings._fields:
        a, b = np.asarray(getattr(fresh, k)), np.asarray(getattr(cached, k))
        assert a.dtype == b.dtype and np.array_equal(a, b), k
    key = cqb.buildings_key(seeds, CFG)
    assert cqb.buildings_key(seeds, replace(CFG, flashbangs=(2, 0), time_cost=0.05, walk=1.0)) == key
    assert cqb.buildings_key(seeds, replace(CFG, cover=(0, 2))) != key
    assert cqb.buildings_key(seeds, replace(CFG, max_room=9.0)) != key
    assert cqb.buildings_key(seeds[:-1] + [999], CFG) != key
    par = cqb.make_buildings(range(128), CFG, cache=False)
    for i in (0, 37, 127):
        one = cqb._building_arrays(i, CFG)
        for k in cqb.Buildings._fields:
            want = np.asarray(one[k])
            got = np.asarray(getattr(par, k))[i]
            assert np.allclose(got, want.astype(got.dtype)) if got.dtype.kind == "f" else np.array_equal(got, want), k


# ── Weapons (ep9 3am) ────────────────────────────────────────────────────────

from ailearns.games.cqb import RELOAD  # noqa: E402

GUN3 = _replace(CFG, magazine=3, spare_mags=1)      # tiny magazines: quick to empty


def _gun_game(cfg=GUN3, walls=WALLS, doors=DOOR, cover=CRATE):
    return CQBGame(_plain_game(walls, doors, cover).b, cfg)


def _reload_act(**kw):
    return _act(reload=1, **kw)


def test_no_magazines_unless_the_game_has_them():
    assert CFG.magazine == 0 and CFG.obs_size == 107 and _replace(CFG, magazine=30).obs_size == 114
    assert _replace(CFG, magazine=30, flashbangs=(2, 0)).obs_size == 122
    g = _plain_game(WALLS, DOOR, CRATE)
    st, step, shots = _pose(g, (12, 10), np.pi, (25, 20)), jax.jit(g.step), 0
    for _ in range(60):                                              # no limit: it fires every decision
        st, _ = step(st, _both(_fire()))
        shots += int(st.fired[ASSAULT])
    assert shots == 60


def test_magazine_empties_and_reloads_by_itself():
    g = _gun_game()
    st, step, fired = _pose(g, (12, 10), np.pi, (25, 20)), jax.jit(g.step), []
    for t in range(1, 41):                                           # trigger held, facing a wall
        st, _ = step(st, _both(_fire()))
        if bool(st.fired[ASSAULT]):
            fired.append(t)
    r = g.reload_steps                                               # 2.5 s
    assert fired == [1, 2, 3, 4 + r, 5 + r, 6 + r]                    # 3, the reload starts at 4, 3 more, then nothing
    assert int(st.mag[ASSAULT]) == 0 and int(st.reserve[ASSAULT]) == 0 and int(st.reload_timer[ASSAULT]) == 0


def test_reload_switch_and_no_shooting_while_reloading():
    g = _gun_game()
    st = _step(g, _pose(g, (12, 10), np.pi, (25, 20)), _both(_fire()))
    assert int(st.mag[ASSAULT]) == 2
    st = _step(g, st, _both(_reload_act()))
    assert int(st.reload_timer[ASSAULT]) == g.reload_steps and bool(st.reload_now[ASSAULT])
    start = np.asarray(st.pos[ASSAULT])
    step = jax.jit(g.step)
    for _ in range(g.reload_steps - 1):                              # it can walk, not shoot; pressing again changes nothing
        st, _ = step(st, _both(np.maximum(_fire(fwd=1), _reload_act(fwd=1))))
        assert not bool(st.fired[ASSAULT])
    assert np.linalg.norm(np.asarray(st.pos[ASSAULT]) - start) > 1.0
    st, _ = step(st, _both(_act()))
    assert int(st.mag[ASSAULT]) == 3 and int(st.reserve[ASSAULT]) == 2 and int(st.reload_timer[ASSAULT]) == 0
    st, _ = step(st, _both(_reload_act()))                           # full: the switch does nothing
    assert int(st.reload_timer[ASSAULT]) == 0 and int(st.reserve[ASSAULT]) == 2


def test_recoil():
    """A shot right after a shot hits less (x(1 - recoil x what's left of the kick)); after 0.3 s, full again."""
    cfg = _replace(CFG, magazine=30, hit_falloff=0.0, hit_close=0.9)
    g = CQBGame(_plain_game().b, cfg)
    st = _pose(g, (12, 10), 0.0, (15, 10), d_face=np.pi / 2)
    fresh = _hit_rate(g, st, _both(_fire()))
    kicked = _hit_rate(g, st._replace(kick=st.kick.at[ASSAULT].set(1.0)), _both(_fire()))
    rested = _hit_rate(g, st._replace(kick=st.kick.at[ASSAULT].set(cfg.dt / cfg.recoil_recover)), _both(_fire()))
    assert abs(fresh - 0.9) < 0.03 and abs(rested - 0.9) < 0.03
    assert abs(kicked - 0.9 * (1 - 0.5 * (1 - cfg.dt / cfg.recoil_recover))) < 0.03     # 0.6
    shot = _step(g, st._replace(key=jax.random.PRNGKey(3)), _both(_fire()))
    assert float(shot.kick[ASSAULT]) == 1.0


def test_a_reload_is_heard_nearby():
    g = _gun_game()

    def heard(d_pos, a_pos=(15, 10)):
        st = _pose(g, a_pos, np.pi, d_pos, d_face=np.pi)
        st = st._replace(mag=st.mag.at[DEFENSE].set(1))
        st = _step(g, st, _both(_act(), _reload_act()))
        return np.asarray(st.reload_heard[ASSAULT]), st
    near, st = heard((17, 10))                                       # 2 m, same side of the wall
    assert near[2] > 0.4 and np.allclose(near[:2], (1, 0), atol=1e-5) and float(st.sound[ASSAULT, :, 2].max()) == 0
    far, _ = heard((18.5, 12))                                       # ~4 m: just out of reach
    assert far[2] == 0
    open3, _ = heard((15.5, 10), a_pos=(18.5, 10))                   # 3 m, nothing between: heard
    walled3, _ = heard((21.5, 10), a_pos=(18.5, 10))                 # 3 m through the wall (reach 4 x 2/3): not
    assert open3[2] > 0.2 and walled3[2] == 0


def test_observation_with_magazines():
    """7 more numbers; the first ones are exactly the game without magazines; and Assault's magazine doesn't show
    in Defense's numbers."""
    g0, g = _plain_game(WALLS, DOOR, CRATE), _gun_game(_replace(CFG, magazine=30))
    st = _pose(g, (12, 10), 0.3, (25, 20), d_face=np.pi)
    for _ in range(4):
        st = _step(g, st, _both(_fire(fwd=1, turn=0.2)))
        for w in (ASSAULT, DEFENSE):
            o, o0 = np.asarray(g.observe(st, w)), np.asarray(g0.observe(st, w))
            assert o.shape == (114,) and np.array_equal(o[:107], o0)
    o = np.asarray(g.observe(st, ASSAULT))[107:]
    assert abs(o[0] - 26 / 30) < 1e-6 and o[1] == 1.0 and o[2] == 0 and 0 < o[3] <= 1
    ds = [np.asarray(g.observe(st._replace(mag=st.mag.at[ASSAULT].set(m)), DEFENSE)) for m in (0, 7, 30)]
    assert np.array_equal(ds[0], ds[1]) and np.array_equal(ds[0], ds[2])


# ── Health (ep9 3ar) ─────────────────────────────────────────────────────────

HP = _replace(CFG, health=100.0)


def _hp_game(cfg=HP, walls=()):
    return CQBGame(_plain_game(walls).b, _replace(cfg, hit_falloff=0.0, hit_close=1.0))   # every aimed shot hits


def test_one_hit_isnt_always_down():
    """Where hits land (10% head, 55% torso, 35% limb over many tries), what each takes, and down only at 0."""
    g = _hp_game()
    st = _pose(g, (12, 10), 0.0, (15, 10), d_face=np.pi / 2)              # Defense facing away, 3 m ahead
    step = jax.jit(jax.vmap(lambda k: g.step(st._replace(key=k), _both(_fire()))[0]))
    after = step(jax.random.split(jax.random.PRNGKey(1), 4000))
    taken = np.round(100 - np.asarray(after.hp[:, DEFENSE]))
    assert set(np.unique(taken)) == {25, 40, 100}
    for dmg, share in ((100, 0.10), (40, 0.55), (25, 0.35)):
        assert abs((taken == dmg).mean() - share) < 0.03, dmg
    down = ~np.asarray(after.alive[:, DEFENSE])
    assert np.array_equal(down, taken == 100) and np.array_equal(np.asarray(after.done), down)
    assert np.asarray(after.landed[:, ASSAULT]).max() == 0                # it knows its shot landed


def test_down_after_enough_hits_and_wounds_slow_you():
    g = _hp_game()
    st = _pose(g, (12, 10), 0.0, (15, 10), d_face=np.pi / 2)
    st = st._replace(hp=st.hp.at[DEFENSE].set(20.0))
    after, _ = g.step(st._replace(key=jax.random.PRNGKey(5)), _both(_fire()))
    assert not bool(after.alive[DEFENSE]) and int(after.outcome) == 1     # any hit takes 25+: 20 -> down
    # wounded: slower and a worse shot
    fit, hurt = st, st._replace(hp=jnp.asarray([45.0, 100.0]))
    run = jax.jit(lambda s: _run(g, s, _both(_act(fwd=1)), 10).pos[ASSAULT])
    d_fit = float(np.linalg.norm(np.asarray(run(fit)) - (12, 10)))
    d_hurt = float(np.linalg.norm(np.asarray(run(hurt)) - (12, 10)))
    assert abs(d_hurt / d_fit - 0.75) < 0.05
    cfg = _replace(HP, hit_falloff=0.0, hit_close=0.8)
    g2 = CQBGame(_plain_game().b, cfg)
    s2 = _pose(g2, (12, 10), 0.0, (15, 10), d_face=np.pi / 2)
    assert abs(float(g2.hit_chance(s2, ASSAULT)) - 0.8) < 1e-6
    assert abs(float(g2.hit_chance(s2._replace(hp=jnp.asarray([45.0, 100.0])), ASSAULT)) - 0.6) < 1e-6


def test_both_hit_at_once_and_observation_with_health():
    g = _hp_game()
    st = _pose(g, (12, 10), 0.0, (15, 10), d_face=np.pi)                  # face to face
    after, _ = g.step(st._replace(key=jax.random.PRNGKey(2)), _both(_fire(), _fire()))
    assert float(after.hp[ASSAULT]) < 100 and float(after.hp[DEFENSE]) < 100     # both took a hit
    g0, gh = _plain_game(WALLS, DOOR, CRATE), CQBGame(_plain_game(WALLS, DOOR, CRATE).b, HP)
    s = _step(gh, _pose(gh, (12, 10), 0.3, (25, 20), d_face=np.pi), _both(_act(fwd=1)))
    s = s._replace(hp=jnp.asarray([70.0, 40.0]))
    for w in (ASSAULT, DEFENSE):
        o, o0 = np.asarray(gh.observe(s, w)), np.asarray(g0.observe(s, w))
        assert o.shape == (109,) and np.array_equal(o[:107], o0)
    assert np.allclose(np.asarray(gh.observe(s, ASSAULT))[107:], (0.7, 0.0))
    # the other's health doesn't show
    a = [np.asarray(gh.observe(s._replace(hp=s.hp.at[DEFENSE].set(v)), ASSAULT)) for v in (10.0, 60.0, 100.0)]
    assert np.array_equal(a[0], a[1]) and np.array_equal(a[0], a[2])
    # with everything on, health comes last (brains from before read the numbers in front)
    full = _replace(CFG, flashbangs=(2, 0), magazine=30, health=100.0)
    assert full.obs_size == 107 + 8 + 7 + 2


# ── Reaction time (ep9 3aw) ──────────────────────────────────────────────────

RT = _replace(CFG, react=0.3, hit_falloff=0.0, hit_close=0.9)


def test_aim_settles_over_the_reaction_time():
    """Spotting someone, a shot hits x1/3 (the first decision in sight), x2/3, then fully from the third on."""
    g = CQBGame(_plain_game().b, RT)
    assert g.react_steps == 3
    st = _pose(g, (12, 10), 0.0, (15, 10), d_face=np.pi / 2)              # Defense 3 m ahead, facing away
    for before, want in ((0, 0.3), (1, 0.6), (2, 0.9), (3, 0.9)):         # aim before this decision -> hit rate
        assert abs(_hit_rate(g, st._replace(aim=st.aim.at[ASSAULT].set(before)), _both(_fire())) - want) < 0.03
    assert float(g.hit_chance(st._replace(aim=st.aim.at[ASSAULT].set(0)), ASSAULT)) == 0.0
    s = st._replace(aim=jnp.zeros(2, jnp.int32))
    got = []
    for _ in range(5):                                                    # looking, not shooting: 1, 2, 3, 3, 3
        s = _step(g, s, _both(_act()))
        got.append(int(s.aim[ASSAULT]))
    assert got == [1, 2, 3, 3, 3] and int(s.aim[DEFENSE]) == 0           # Defense faces away: never in its sight


def test_aim_unsettles_out_of_sight_and_is_off_by_default():
    g = CQBGame(_plain_game().b, RT)
    st = _pose(g, (12, 10), np.pi, (15, 10), d_face=np.pi / 2)            # Defense behind Assault: out of sight
    s = st._replace(aim=jnp.asarray([3, 0], jnp.int32))
    got = []
    for _ in range(4):
        s = _step(g, s, _both(_act()))
        got.append(int(s.aim[ASSAULT]))
    assert got == [2, 1, 0, 0]
    # without a reaction time: no aim counter, and the first shot in sight hits fully
    g0 = CQBGame(_plain_game().b, _replace(RT, react=0.0))
    s0 = _pose(g0, (12, 10), 0.0, (15, 10), d_face=np.pi / 2)
    assert abs(_hit_rate(g0, s0._replace(aim=jnp.zeros(2, jnp.int32)), _both(_fire())) - 0.9) < 0.03
    assert int(_step(g0, s0, _both(_act())).aim.max()) == 0


def test_observation_with_reaction_time():
    """One more number, last: how settled my aim is; everything before is the game without it; the other's aim
    doesn't show."""
    g0 = _plain_game(WALLS, DOOR, CRATE)
    g = CQBGame(g0.b, _replace(CFG, react=0.3))
    s = _step(g, _pose(g, (12, 10), 0.3, (25, 20), d_face=np.pi), _both(_act(fwd=1)))
    s = s._replace(aim=jnp.asarray([2, 1], jnp.int32))
    for w in (ASSAULT, DEFENSE):
        o, o0 = np.asarray(g.observe(s, w)), np.asarray(g0.observe(s, w))
        assert o.shape == (108,) and np.array_equal(o[:107], o0)
    assert abs(float(np.asarray(g.observe(s, ASSAULT))[107]) - 2 / 3) < 1e-6
    a = [np.asarray(g.observe(s._replace(aim=s.aim.at[DEFENSE].set(v)), ASSAULT)) for v in (0, 2, 3)]
    assert np.array_equal(a[0], a[1]) and np.array_equal(a[0], a[2])
    full = _replace(CFG, flashbangs=(2, 0), magazine=30, health=100.0, react=0.3)
    assert full.obs_size == 107 + 8 + 7 + 2 + 1
    gf = CQBGame(g0.b, full)
    gh = CQBGame(g0.b, _replace(full, react=0.0))
    sf = _step(gf, _pose(gf, (12, 10), 0.3, (25, 20), d_face=np.pi), _both(_act(fwd=1)))
    of, oh = np.asarray(gf.observe(sf, ASSAULT)), np.asarray(gh.observe(sf, ASSAULT))
    assert of.shape == (125,) and np.array_equal(of[:124], oh)            # reaction time last
