"""Episode 13: realistic homes (games/cqb_home.py, `--house home`)."""

from dataclasses import replace

import numpy as np

from ailearns.games import cqb
from ailearns.games.cqb_home import _connected, _gap, _overlap, make_home

HOME = replace(cqb.CQBConfig(), **cqb.HOUSES["home"])
SEEDS = list(range(100_000, 100_032))


def test_home_rooms_and_doors():
    """A living room, a kitchen, a hallway, bedrooms and one bathroom; every bedroom and the bathroom has exactly one
    door, into the hallway; 2-3 ways in; every room can be reached from outside."""
    for s in SEEDS:
        b = make_home(s, HOME)
        types = b["room_types"]
        assert types.count("living") == 1 and types.count("kitchen") == 1 and types.count("hall") == 1, s
        assert types.count("bath") == 1 and types.count("bedroom") >= 2, (s, types)
        hall = types.index("hall")
        for r, kind in enumerate(types):
            ds = [d for d in b["doors"] if r in d["rooms"]]
            if kind in ("bedroom", "bath"):
                assert len(ds) == 1 and hall in ds[0]["rooms"], (s, r, kind)
        assert 2 <= sum(d["outside"] for d in b["doors"]) <= 3 and len(b["entries"]) >= 2, s
        g = cqb.room_graph(b, HOME)
        assert (g["hops"][HOME.max_rooms, :len(b["rooms"])] < 99).all(), s


def test_home_furniture_rules():
    """Furniture: never in a wall, window, doorway or a doorway's approach; flush against a wall face or at least
    cover_gap from it and from other pieces (no slivers); no tall piece in front of a window; every room's doorways
    stay connected for an operator's body."""
    t, g = HOME.wall, HOME.cover_gap
    for s in SEEDS:
        b = make_home(s, HOME)
        solid = list(zip(b["wall_lo"], b["wall_hi"])) + list(zip(b["win_lo"], b["win_hi"])) + \
            [(d["lo"], d["hi"]) for d in b["doors"]]
        fur = b["furniture"]
        assert len(b["cover_lo"]) <= HOME.max_cover and len(b["low_lo"]) <= HOME.max_low
        assert len(fur) == len(b["cover_lo"]) + len(b["low_lo"])
        for i, f in enumerate(fur):
            assert not any(_overlap(f["lo"], f["hi"], lo, hi) for lo, hi in solid), (s, f["name"])
            for d in b["doors"]:
                n = np.eye(2)[d["axis"]]
                assert not _overlap(f["lo"], f["hi"], d["lo"] - n * 1.2, d["hi"] + n * 1.2), (s, f["name"])
            rl, rh = b["rooms"][f["room"]]
            gaps = np.concatenate([f["lo"] - (rl + t / 2), (rh - t / 2) - f["hi"]])
            assert (gaps > -1e-6).all() and not ((gaps > 1e-6) & (gaps < g - 1e-6)).any(), (s, f["name"], gaps)
            for o in fur[i + 1:]:
                d = _gap(f["lo"], f["hi"], o["lo"], o["hi"])
                assert not _overlap(f["lo"], f["hi"], o["lo"], o["hi"]) and not 1e-6 < d < g - 1e-6, (s, f, o)
            if f["tall"]:
                assert all(_gap(f["lo"], f["hi"], wl, wh) >= 0.05 for wl, wh in zip(b["win_lo"], b["win_hi"])), s
        for r, (rl, rh) in enumerate(b["rooms"]):
            starts = []
            for d in b["doors"]:
                if r in d["rooms"]:
                    n = np.eye(2)[d["axis"]]
                    p = d["center"] + n * 0.5
                    starts.append(p if (p > rl).all() and (p < rh).all() else d["center"] - n * 0.5)
            boxes = [(f["lo"], f["hi"]) for f in fur if f["room"] == r]
            assert _connected(rl + t / 2, rh - t / 2, boxes, starts, HOME.radius + 0.02), (s, r)


def test_home_entries_face_their_door_after_mirroring():
    """Assault starts entry_dist outside each outside door, facing it, whichever way the house was mirrored."""
    for s in SEEDS:
        b = make_home(s, HOME)
        outside = [d for d in b["doors"] if d["outside"]]
        for (p, face, _), d in zip(b["entries"], outside):
            ahead = p + np.array([np.cos(face), np.sin(face)]) * HOME.entry_dist
            assert np.linalg.norm(ahead - d["center"]) < 1e-6, (s, p, face, d["center"])


def test_home_buildings_are_packed_and_kill_houses_unchanged():
    """Homes carry their low furniture and windows as arrays; kill houses carry none (and are what they were)."""
    B = cqb.make_buildings(SEEDS[:4], HOME, cache=False)
    assert B.low_lo.shape == (4, HOME.max_low, 2) and B.win_lo.shape == (4, HOME.max_windows, 2)
    assert B.cover_lo.shape == (4, HOME.max_cover, 2) and B.wall_lo.shape == (4, HOME.max_walls, 2)
    K = cqb.make_buildings(SEEDS[:4], cqb.CQBConfig(), cache=False)
    assert K.low_lo is None and K.win_lo is None
    a, b2 = make_home(SEEDS[0], HOME), make_home(SEEDS[0], HOME)          # the same seed, the same house
    assert np.array_equal(a["wall_lo"], b2["wall_lo"]) and [f["name"] for f in a["furniture"]] == \
        [f["name"] for f in b2["furniture"]]


def _home_game(seed, **rules):
    import jax
    cfg = replace(HOME, **rules)
    game = cqb.CQBGame(cqb.make_buildings([seed], cfg, cache=False), cfg)
    return game, game.init(jax.random.PRNGKey(0), 0), make_home(seed, cfg)


def _facing(game, st, a, b):
    """st with operator 0 at a facing b, operator 1 at b (both standing still, nobody blind)."""
    import jax.numpy as jnp
    pos = st.pos.at[0].set(jnp.asarray(a)).at[1].set(jnp.asarray(b))
    d = np.asarray(b) - np.asarray(a)
    return st._replace(pos=pos, face=st.face.at[0].set(float(np.arctan2(d[1], d[0]))), vel=jnp.zeros_like(st.vel),
                       blind=jnp.zeros_like(st.blind))


def test_low_furniture_blocks_walking_not_seeing():
    """Across a low piece (a bed, a sofa) you see each other, but you can't walk through it; the lasers feel it."""
    import jax.numpy as jnp
    game, st, b = _home_game(100_003, team_sizes=(1, 1))
    solid = list(zip(b["wall_lo"], b["wall_hi"])) + list(zip(b["low_lo"], b["low_hi"])) + \
        list(zip(b["cover_lo"], b["cover_hi"])) + list(zip(b["win_lo"], b["win_hi"]))
    free = lambda q: all(np.hypot(*(q - np.clip(q, lo, hi))) > 0.35 for lo, hi in solid)
    found = None
    for f in (f for f in b["furniture"] if not f["tall"]):   # a low piece with room to stand on two sides of it
        mid, half = (f["lo"] + f["hi"]) / 2, (f["hi"] - f["lo"]) / 2
        for ax in (0, 1):
            n = np.eye(2)[ax]
            a, c = mid - n * (half[ax] + 0.4), mid + n * (half[ax] + 0.4)
            if free(a) and free(c):
                found = (f, a, c)
                break
        if found:
            break
    f, a, c = found
    s = _facing(game, st, a, c)
    assert bool(game.sees(s, 0, target=1)), "a low piece mustn't block sight"
    las = float(game.lasers(s, 0)[0])                          # straight ahead: the piece, 0.4 m away
    assert abs(las - 0.4) < 0.05, las
    act = jnp.zeros((2, cqb.ACTION_SIZE)).at[0, cqb.FWD].set(1.0).at[:, cqb.FIRE:].set(-1.0)
    for _ in range(30):                                       # walk into it for 3 s
        s, _ = game.step(s, act)
    p = np.asarray(s.pos[0])
    assert np.hypot(*(p - np.clip(p, f["lo"], f["hi"]))) >= game.cfg.radius - 0.02, "walked into a low piece"


def test_windows_let_sight_and_bang_light_through_not_bodies_or_throws():
    """Through a window you see (and are seen); you can't walk through it; a flashbang's light gets through glass,
    a thrown one doesn't."""
    import jax.numpy as jnp
    game, st, b = _home_game(100_003, team_sizes=(1, 1), flashbangs=(2, 0))
    solid = list(zip(b["wall_lo"], b["wall_hi"])) + list(zip(b["low_lo"], b["low_hi"])) + \
        list(zip(b["cover_lo"], b["cover_hi"])) + list(zip(b["win_lo"], b["win_hi"]))
    free = lambda q: all(np.hypot(*(q - np.clip(q, lo, hi))) > 0.35 for lo, hi in solid)
    inside = lambda q: (q > b["lo"]).all() and (q < b["hi"]).all()
    for wl, wh in zip(b["win_lo"], b["win_hi"]):              # a window with room to stand on both sides
        mid = (wl + wh) / 2
        n = np.eye(2)[int(np.argmin(wh - wl))]                 # across the wall
        if inside(mid + n):
            n = -n                                             # from outside (a) in (c)
        n = -n
        a, c = mid - n * 1.5, mid + n * 1.5
        if free(a) and free(c) and not inside(a) and inside(c):
            break
    s = _facing(game, st, a, c)
    assert bool(game.sees(s, 0, target=1)) and bool(game.sees(s._replace(face=s.face.at[1].set(
        float(np.arctan2(*(a - c)[::-1])))), 1, target=0)), "a window mustn't block sight"
    las = float(game.lasers(s, 0)[0])
    assert abs(las - (1.5 - game.cfg.wall / 2)) < 0.05, las
    act = jnp.zeros((2, cqb.ACTION_SIZE)).at[0, cqb.FWD].set(1.0).at[:, cqb.FIRE:].set(-1.0)
    w = s
    for _ in range(30):
        w, _ = game.step(w, act)
    assert np.dot(np.asarray(w.pos[0]) - mid, n) < 0, "walked through a window"
    lo, hi = game.walls(s)                                     # a throw stops short of the glass
    land = float(cqb._cast(jnp.asarray(a), jnp.asarray(n)[None], lo, hi, game.cfg)[0])
    assert land < 1.5, land
    llo, lhi = game.walls(s, glass=False)                      # the bang's light passes it
    assert bool(cqb._clear(jnp.asarray(a), jnp.asarray(c)[None], llo, lhi)[0])


def test_bullets_go_through_drywall_not_brick():
    """With through_drywall, a shot can hit someone behind an inside wall it doesn't see (a reduced chance per layer,
    x unseen_hit), never through an outside (brick) wall; off, the game is as before (no hit without sight)."""
    for rate in (0.5, 0.0):
        game, st, b = _home_game(100_003, team_sizes=(1, 1), through_drywall=rate)
        solid = list(zip(b["wall_lo"], b["wall_hi"])) + list(zip(b["low_lo"], b["low_hi"])) + \
            list(zip(b["cover_lo"], b["cover_hi"])) + list(zip(b["win_lo"], b["win_hi"]))
        free = lambda q: all(np.hypot(*(q - np.clip(q, lo, hi))) > 0.35 for lo, hi in solid) and \
            (q > b["lo"] + 0.5).all() and (q < b["hi"] - 0.5).all()
        cases = {}
        for lo, hi, soft in zip(b["wall_lo"], b["wall_hi"], b["wall_soft"]):
            mid, n = (lo + hi) / 2, np.eye(2)[int(np.argmin(hi - lo))]
            if not soft:   # brick: an outside wall, someone 1.2 m outside it, someone 1.2 m inside
                a, c = mid - n * 1.2, mid + n * 1.2
                if free(c) or free(a):
                    out_ = c if not ((c > b["lo"]).all() and (c < b["hi"]).all()) else a
                    in_ = a if out_ is c else c
                    if free(in_) and "brick" not in cases:
                        cases["brick"] = (in_, out_)
            elif max(hi - lo) > 2.0 and "drywall" not in cases:
                a, c = mid - n * 1.2, mid + n * 1.2
                if free(a) and free(c):
                    cases["drywall"] = (a, c)
        for kind, (a, c) in cases.items():
            s = _facing(game, st, a, c)
            s = s._replace(face=s.face.at[1].set(float(np.arctan2(*(a - c)[::-1]))))
            s = game._look(s)
            assert not bool(s.seen[0, 1]), kind
            _, p = game.targets(s)
            want = 0.0 if (kind == "brick" or rate == 0) else \
                np.clip(game.cfg.hit_close - game.cfg.hit_falloff * np.linalg.norm(c - a), 0.05, 1) * \
                game.cfg.unseen_hit * rate
            assert abs(float(p[0]) - want) < 1e-5, (kind, rate, float(p[0]), want)
        assert set(cases) == {"brick", "drywall"}, cases


def test_the_hostage_can_be_hit_and_its_shooter_loses():
    """With hostage_hit, the hostage between a shooter and an enemy is the first body on the line: the shot goes to it
    (body N); a hit ends the round, lost for the side that fired. Off, the shot goes to the enemy as before."""
    import jax
    import jax.numpy as jnp
    for on in (True, False):
        game, st, b = _home_game(100_003, team_sizes=(1, 1), hostage=True, hostage_hit=on)
        solid = list(zip(b["wall_lo"], b["wall_hi"])) + list(zip(b["low_lo"], b["low_hi"])) + \
            list(zip(b["cover_lo"], b["cover_hi"]))
        free = lambda q: all(np.hypot(*(q - np.clip(q, lo, hi))) > 0.5 for lo, hi in solid)
        rl, rh = next((rl, rh) for (rl, rh), k in zip(b["rooms"], b["room_types"]) if k == "living")
        y = (rl[1] + rh[1]) / 2
        xs = [x for x in np.arange(rl[0] + 0.6, rh[0] - 0.6, 0.1)
              if all(free(np.array([x + d, y])) for d in (0.0, 1.0, 2.0))]
        a, h, c = np.array([xs[0], y]), np.array([xs[0] + 1.0, y]), np.array([xs[0] + 2.0, y])
        s = _facing(game, st, a, c)
        s = game._look(s._replace(hostage=jnp.asarray(h, jnp.float32), face=s.face.at[1].set(np.pi)))
        tgt, p = game.targets(s)
        if not on:
            assert int(tgt[0]) == 1
            continue
        assert int(tgt[0]) == game.N and float(p[0]) > 0.5, (int(tgt[0]), float(p[0]))
        act = jnp.zeros((2, cqb.ACTION_SIZE)).at[:, cqb.FIRE:].set(-1.0).at[0, cqb.FIRE].set(1.0)
        for k in range(20):                                   # fire until a shot lands (chance ~0.9)
            n, _ = game.step(s._replace(key=jax.random.PRNGKey(k)), act)
            if int(n.hostage_hit_by) >= 0:
                break
        assert int(n.hostage_hit_by) == cqb.ASSAULT and bool(n.done) and int(n.outcome) == -1
        assert bool(n.alive[1]), "the enemy behind the hostage wasn't hit"


def test_cached_homes_match_fresh_and_the_rules_dont_change_the_key(tmp_path, monkeypatch):
    """`cloud buildings` caches the home sets: a cached set equals a fresh one, and the ep13 rules, teams and hostage
    settings don't change the key (one cached set serves every home run); the home generator's settings do."""
    monkeypatch.setenv("AILEARNS_BUILDINGS", str(tmp_path))
    seeds = list(range(64))
    fresh = cqb.make_buildings(seeds, HOME)
    cached = cqb.make_buildings(seeds, HOME)
    assert len(list(tmp_path.glob("buildings-*.npz"))) == 1
    for k in cqb.Buildings._fields:
        a, b = np.asarray(getattr(fresh, k)), np.asarray(getattr(cached, k))
        assert a.dtype == b.dtype and np.array_equal(a, b), k
    key = cqb.buildings_key(seeds, HOME)
    rules = replace(HOME, through_drywall=0.5, unseen_hit=0.5, hostage_hit=True, hostage=True, hostage_deep=6.0,
                    extract=True, hostage_noise=8.0, team_sizes=(2, 2), split_entry=True)
    assert cqb.buildings_key(seeds, rules) == key
    assert cqb.buildings_key(seeds, cqb.CQBConfig()) != key
    assert cqb.buildings_key(seeds, replace(HOME, width=(16.0, 22.0))) != key


def test_no_defender_spawn_seen_through_a_window():
    """(ep13 8.16) No defender spawn spot can be seen from an attacker's start (its entry, or stacked behind it) with
    every door shut: windows were letting Assault shoot defenders before the round had begun."""
    import numpy as np
    from ailearns.games import cqb_home as H
    cfg = replace(cqb.CQBConfig(), layout="home")
    for seed in range(100000, 100032):
        b = H.make_home(seed, cfg)
        eyes = [e for p, f, _ in b["entries"] for e in (p, p - np.array([np.cos(f), np.sin(f)]) * cfg.stack_gap)]
        lo = np.concatenate([b["wall_lo"], b["cover_lo"], np.array([d["lo"] for d in b["doors"]])])
        hi = np.concatenate([b["wall_hi"], b["cover_hi"], np.array([d["hi"] for d in b["doors"]])])
        assert not any(H._seen_from(np.asarray(p), eyes, lo, hi) for p in b["spawn"]), seed


def test_house_follows_layout():
    """cqb.house (what the 2D footage draws) is the home under home rules, the kill house otherwise (ep13: the 2D
    drew kill houses under rounds played in homes)."""
    for s in SEEDS[:4]:
        assert np.array_equal(cqb.house(s, HOME)["wall_lo"], make_home(s, HOME)["wall_lo"]), s
        assert np.array_equal(cqb.house(s)["wall_lo"], cqb.make_building(s)["wall_lo"]), s
