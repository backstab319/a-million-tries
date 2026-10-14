"""CQB with teams (ep10): 2 Assault vs 1 Defense. Operators 0 and 1 are the Assault pair, 2 the defender."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.games import cqb, cqb_scripts as S
from ailearns.games.cqb import (ENEMY_OBS, FIRE, FWD, MATE_OBS, RUN, TEST_SEEDS, Buildings, CQBConfig, CQBGame,
                                cell_fracs, make_buildings)

CFG = CQBConfig()
TWO = replace(CFG, team_sizes=(2, 1))
A0, A1, D = 0, 1, 2


def _open_game(cfg=TWO, walls=()):
    """An open floor (footprint 10-30 x 5-25), optional walls; no doors, no cover."""
    def pad(boxes, n):
        lo, hi = np.full((n, 2), -1000.0), np.full((n, 2), -1000.0)
        for k, (a, b) in enumerate(boxes):
            lo[k], hi[k] = a, b
        return jnp.asarray(lo[None], jnp.float32), jnp.asarray(hi[None], jnp.float32)
    wl, wh = pad(walls, cfg.max_walls)
    dl, dh = pad((), cfg.max_doors)
    cl, ch = pad((), cfg.max_cover)
    Dn = cfg.max_doors
    foot = (np.array([10.0, 5.0]), np.array([30.0, 25.0]))
    b = Buildings(wl, wh, dl, dh, jnp.zeros((1, Dn), bool), jnp.ones((1, Dn, 2), bool), jnp.zeros((1, Dn), bool),
                  cl, ch, jnp.asarray([[[5.0, 5.0]] * 3], jnp.float32), jnp.zeros((1, 3)), jnp.asarray([1]),
                  jnp.full((1, cfg.spawn_points, 2), 20.0), jnp.asarray([foot[0]], jnp.float32),
                  jnp.asarray([foot[1]], jnp.float32),
                  jnp.asarray((foot[0] + cell_fracs(cfg) * (foot[1] - foot[0]))[None], jnp.float32))
    return CQBGame(b, cfg)


def _pose(g, pos, face):
    st = g.init(jax.random.PRNGKey(0), 0)
    return st._replace(pos=jnp.asarray(pos, jnp.float32), face=jnp.asarray(face, jnp.float32))


def _acts(**per):
    """Actions for the three, e.g. _acts(a1={"fire": 1}); everything else off (-1 for the switches)."""
    a = np.zeros((3, 8), np.float32)
    a[:, 3:] = -1
    for who, kw in per.items():
        for k, v in kw.items():
            a[{"a0": A0, "a1": A1, "d": D}[who], {"fwd": FWD, "fire": FIRE, "run": RUN}[k]] = v
    return jnp.asarray(a)


def _after_shots(g, st, acts, n=2000):
    """alive (n, 3) after one decision with n different lucks."""
    f = jax.jit(jax.vmap(lambda k: g.step(st._replace(key=k), acts)[0].alive))
    return np.asarray(f(jax.random.split(jax.random.PRNGKey(3), n)))


def test_default_is_one_v_one():
    g = _open_game(CFG)
    assert CFG.team_sizes == (1, 1) and g.N == 2 and not g.teams_on and CFG.obs_size == 107


def test_observation_size_and_slots():
    g = _open_game()
    assert TWO.obs_size == 107 + ENEMY_OBS + MATE_OBS + 1
    st = g.step(_pose(g, [(12, 10), (12, 12), (17, 10)], [0.0, 0.0, np.pi]), _acts())[0]
    obs = [np.asarray(g.observe(st, w)) for w in range(3)]
    assert all(len(o) == TWO.obs_size and np.isfinite(o).all() for o in obs)
    extra_enemy = slice(107, 107 + ENEMY_OBS)
    mate = slice(107 + ENEMY_OBS, 107 + ENEMY_OBS + MATE_OBS)
    assert not obs[A0][extra_enemy].any() and not obs[A1][extra_enemy].any()   # Assault has one enemy: empty slot
    assert obs[D][extra_enemy][0] == 1.0                                           # Defense sees the second one too
    assert obs[A0][mate].any() and not obs[D][mate].any()                          # the defender has no teammate
    assert obs[A0][-1] == 0.0 and obs[A1][-1] == 1.0                               # which one of my team I am


def test_pair_starts_stacked_at_one_door():
    cfg = TWO
    g = CQBGame(make_buildings(list(TEST_SEEDS)[:8], cfg), cfg)
    for k in range(16):
        st = g.init(jax.random.PRNGKey(k))
        p, f = np.asarray(st.pos), np.asarray(st.face)
        assert p.shape == (3, 2) and f[A0] == f[A1]
        assert np.allclose(p[A1] - p[A0], -0.8 * np.array([np.cos(f[A0]), np.sin(f[A0])]), atol=1e-5)  # right behind
        lo, hi = np.asarray(g.b.foot_lo[st.building]), np.asarray(g.b.foot_hi[st.building])
        assert ((p[D] > lo) & (p[D] < hi)).all() and not ((p[A0] > lo) & (p[A0] < hi)).all()
        assert all(np.isfinite(np.asarray(g.observe(st, w))).all() for w in range(3))


def test_split_entry_starts_the_pair_at_two_doors():
    """ep11: split_entry puts each of the pair outside its own outside door, facing it (default: stacked, above)."""
    cfg = replace(TWO, split_entry=True)
    g = CQBGame(make_buildings(list(TEST_SEEDS)[:8], cfg), cfg)
    stacked = CQBGame(g.b, TWO)
    apart = 0
    for k in range(16):
        st, st0 = g.init(jax.random.PRNGKey(k)), stacked.init(jax.random.PRNGKey(k))
        p, f = np.asarray(st.pos), np.asarray(st.face)
        bi = int(st.building)
        entries = np.asarray(g.b.entry[bi])[: int(g.b.n_entries[bi])]
        assert np.allclose(p[A0], np.asarray(st0.pos)[A0]) and np.allclose(p[D], np.asarray(st0.pos)[D])   # same draws
        if len(entries) > 1:
            assert np.linalg.norm(p[A0] - p[A1]) > 2.0 and np.isclose(np.abs(entries - p[A1]).sum(1), 0, atol=1e-5).any()
            k1 = int(np.argmin(np.abs(entries - p[A1]).sum(1)))
            assert np.isclose(f[A1], float(g.b.entry_face[bi, k1]))
            apart += 1
        assert all(np.isfinite(np.asarray(g.observe(st, w))).all() for w in range(3))
    assert apart >= 8


def test_team_shares_sightings():
    g = _open_game()
    st = g.step(_pose(g, [(12, 10), (12, 12), (17, 10)], [0.0, np.pi, 0.0]), _acts())[0]   # A1 looks away
    assert bool(st.seen[A0, D]) and not bool(st.seen[A1, D])
    assert int(st.since[A1, D]) == 0 and np.allclose(st.mem[A1, D], st.pos[D])          # A0 called it out
    assert not st.seen[D].any() and int(st.since[D, A0]) == g.T                         # the defender saw nobody
    o = np.asarray(g.observe(st, A1))
    assert o[24] == 0.0 and np.abs(o[29:31]).sum() > 0                                   # not seen, but where it was


def test_without_sharing_each_knows_only_its_own_sightings():
    """ep11: share_sight=False. A0 sees the defender, A1 looks away: A1 isn't told (that's the radio's job)."""
    g = _open_game(replace(TWO, share_sight=False))
    st = g.step(_pose(g, [(12, 10), (12, 12), (17, 10)], [0.0, np.pi, 0.0]), _acts())[0]
    assert bool(st.seen[A0, D]) and not bool(st.seen[A1, D])
    assert int(st.since[A0, D]) == 0 and np.allclose(st.mem[A0, D], st.pos[D])
    assert int(st.since[A1, D]) == g.T and not np.asarray(st.mem[A1, D]).any()           # A1 knows nothing
    o = np.asarray(g.observe(st, A1))
    assert o[24] == 0.0 and np.abs(o[29:31]).sum() == 0                                  # no "where it was"
    shared = np.asarray(_open_game().observe(st, A0))
    assert np.allclose(np.asarray(g.observe(st, A0)), shared)                            # the one who saw: as before
    # who's down: only the one who saw it fall (or shot it) knows
    f = jax.jit(jax.vmap(lambda k: g.step(st._replace(key=k), _acts(a0={"fire": 1}))[0]))
    out = f(jax.random.split(jax.random.PRNGKey(2), 64))
    hit = ~np.asarray(out.alive[:, D])
    assert hit.any()
    assert np.asarray(out.known_down[:, A0, D])[hit].all() and not np.asarray(out.known_down[:, A1, D]).any()


def test_relay_passes_on_fresh_sightings_only():
    """ep11 relay (no sharing): A1 hears where A0 sees the defender while that sighting is at most `relay` old; after
    that A1 is back to its own (empty) memory, while A0 keeps its own."""
    g = _open_game(replace(TWO, share_sight=False, relay=1.0))
    st = g.step(_pose(g, [(12, 10), (12, 12), (17, 10)], [0.0, np.pi, 0.0]), _acts())[0]
    o = np.asarray(g.observe(st, A1))
    assert int(st.since[A1, D]) == g.T and o[24] == 0.0 and np.abs(o[29:31]).sum() > 0      # told, not seen
    st = st._replace(face=st.face.at[A0].set(np.pi))                                       # A0 looks away too
    for k in range(12):
        st = g.step(st, _acts())[0]
        o1, o0 = np.asarray(g.observe(st, A1)), np.asarray(g.observe(st, A0))
        assert (np.abs(o1[29:31]).sum() > 0) == (k < 10)                                   # fresh for 1 s
        assert np.abs(o0[29:31]).sum() > 0                                                 # A0 remembers its own


def test_intel_tells_only_the_operators_on_the_feed():
    """ep11 intel (no sharing): A0 is on the drone feed and knows where the defender is right now, behind its back and
    as he moves; A1 isn't and knows nothing. The game itself doesn't change: only what A0's brain reads."""
    plain = _open_game(replace(TWO, share_sight=False))
    g = _open_game(replace(TWO, share_sight=False, intel=1))
    st = g.step(_pose(g, [(12, 10), (12, 12), (17, 10)], [np.pi, np.pi, 0.0]), _acts())[0]  # both look away
    assert not st.seen[A0, D] and not st.seen[A1, D]
    for k in range(5):
        st = st._replace(pos=st.pos.at[D].add(jnp.asarray([0.0, 0.5])))                    # he moves
        o0 = np.asarray(g.observe(st, A0))
        # where he is, in A0's frame (A0 faces west: x ahead, y to its left), as just seen
        rel = np.array([-(17 - 12), -(10 + 0.5 * (k + 1) - 10)]) / g.cfg.sight
        assert o0[24] == 0.0 and np.allclose(o0[29:31], rel, atol=1e-5) and o0[31] == 0.0
        assert np.allclose(np.asarray(g.observe(st, A1)), np.asarray(plain.observe(st, A1)))   # A1: as without
        assert np.allclose(np.asarray(g.observe(st, D)), np.asarray(plain.observe(st, D)))
    a, b = g.step(st, _acts())[0], plain.step(st, _acts())[0]                              # same game
    for f in st._fields:
        assert np.array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f))), f


def test_friendly_fire_hits_the_first_body_on_the_line():
    g = _open_game()
    # A1 aims east through A0 (2 m ahead) at the defender (5 m); the defender faces away
    st = g.step(_pose(g, [(14, 10), (12, 10), (17, 10)], [np.pi / 2, 0.0, 0.0]), _acts())[0]
    alive = _after_shots(g, st, _acts(a1={"fire": 1}))
    assert alive[:, D].all() and 0.75 < 1 - alive[:, A0].mean() < 0.95                 # A0 takes it, ~0.86
    # A0 steps out of the line: now the defender is hit, A0 never
    st = g.step(_pose(g, [(14, 11.5), (12, 10), (17, 10)], [np.pi / 2, 0.0, 0.0]), _acts())[0]
    alive = _after_shots(g, st, _acts(a1={"fire": 1}))
    assert alive[:, A0].all() and 1 - alive[:, D].mean() > 0.75


def test_a_side_is_out_only_when_all_of_it_is_down():
    g = _open_game()
    st = _pose(g, [(12, 10), (12, 20), (17, 10)], [0.0, 0.0, np.pi])
    one_down = g.step(st._replace(alive=jnp.asarray([False, True, True])), _acts())[0]
    assert not bool(one_down.done) and int(one_down.outcome) == 0
    both = g.step(st._replace(alive=jnp.asarray([False, False, True])), _acts())[0]
    assert bool(both.done) and int(both.outcome) == -1
    # the defender shoots A0 down for real: the round goes on, A1 still in it
    st = g.step(_pose(g, [(14, 10), (12, 20), (17, 10)], [0.0, 0.0, np.pi]), _acts())[0]
    f = jax.jit(jax.vmap(lambda k: g.step(st._replace(key=k), _acts(d={"fire": 1}))[0]))
    out = f(jax.random.split(jax.random.PRNGKey(1), 64))
    hit = ~np.asarray(out.alive[:, A0])
    assert hit.any() and not np.asarray(out.done)[hit].any()
    assert np.asarray(out.known_down[:, D, A0])[hit].all()                              # it saw it go down
    order = np.asarray(jax.vmap(lambda s: g.enemy_order(s, D))(out))
    assert (order[hit, -1] == A0).all()                                                 # down ones go last


def test_the_fallen_see_nothing():
    g = _open_game()
    st = _pose(g, [(12, 10), (12, 20), (17, 10)], [0.0, -np.pi / 2 - 1.0, 0.0])          # A0 looks at D, A1 away
    st = g.step(st._replace(alive=jnp.asarray([False, True, True])), _acts())[0]
    assert not st.seen[A0].any() and not st.seen[:, A0].any()
    assert int(st.since[A1, D]) == g.T                                                    # nobody called it out


def test_who_is_down_is_only_known_from_what_you_saw():
    g = _open_game()
    # A1 shoots A0 by accident, the defender faces away: Defense must not find out
    st = g.step(_pose(g, [(14, 10), (12, 10), (17, 10)], [np.pi / 2, 0.0, 0.0]), _acts())[0]
    f = jax.jit(jax.vmap(lambda k: g.step(st._replace(key=k), _acts(a1={"fire": 1}))[0]))
    out = f(jax.random.split(jax.random.PRNGKey(2), 64))
    hit = ~np.asarray(out.alive[:, A0])
    assert hit.any() and (~hit).any()
    assert (np.asarray(out.hit_by[:, A0])[hit] == A1).all() and (np.asarray(out.hit_by[:, A0])[~hit] == -1).all()
    assert not np.asarray(out.known_down[:, D, A0]).any()
    obs = np.asarray(jax.vmap(lambda s: g.observe(s, D))(out))
    assert np.allclose(obs[hit], obs[np.argmin(hit)])                                    # down or not: same numbers


def test_defense_doesnt_see_assault_it_cant_see_or_hear():
    g = _open_game()
    seen_by_d, seen_by_a0 = [], []
    for a1 in [(22, 10), (25, 14), (21, 6)]:                                             # all behind the defender
        st = g.step(_pose(g, [(12, 18), a1, (17, 10)], [np.pi, 0.0, np.pi]), _acts())[0]
        seen_by_d.append(np.asarray(g.observe(st, D)))
        seen_by_a0.append(np.asarray(g.observe(st, A0)))
    assert all(np.array_equal(seen_by_d[0], o) for o in seen_by_d[1:])
    assert not all(np.array_equal(seen_by_a0[0], o) for o in seen_by_a0[1:])            # (the placements differ)


def test_teammates_arent_heard_enemies_are():
    g = _open_game()
    st = _pose(g, [(12, 10), (12, 12), (20, 10)], [0.0, 0.0, np.pi / 2])
    for _ in range(4):
        st = g.step(st, _acts(a0={"fwd": 1, "run": 1}))[0]
    assert float(st.sound[A1, 0, 2]) == 0.0 and float(st.sound[D, 0, 2]) > 0


def test_three_never_overlap():
    g = _open_game()
    st = _pose(g, [(15, 10), (16, 10), (15.5, 10.4)], [0.0, np.pi, -np.pi / 2])
    for _ in range(15):
        st = g.step(st, _acts(a0={"fwd": 1}, a1={"fwd": 1}, d={"fwd": 1}))[0]
        p = np.asarray(st.pos)
        gaps = [np.linalg.norm(p[i] - p[j]) for i in range(3) for j in range(i + 1, 3)]
        assert min(gaps) > 2 * TWO.radius - 0.12                                         # pushed apart each decision


def test_scripts_hold_fire_with_a_teammate_in_the_way():
    g = _open_game()
    st = g.step(_pose(g, [(14, 10), (12, 10), (17, 10)], [np.pi / 2, 0.0, 0.0]), _acts())[0]
    assert bool(st.seen[A1, D])
    act = S._shoot_on_sight(g, st, A1)
    assert float(act[FIRE]) < 0 and float(act[1]) != 0                                    # A0 in the line: no; step aside
    st = g.step(_pose(g, [(14, 11.5), (12, 10), (17, 10)], [np.pi / 2, 0.0, 0.0]), _acts())[0]
    assert float(S._shoot_on_sight(g, st, A1)[FIRE]) > 0


def test_scripted_two_vs_one_rounds():
    for cfg in (TWO, replace(TWO, flashbangs=(2, 0), magazine=30, health=100.0, react=0.3)):
        g = CQBGame(make_buildings(list(TEST_SEEDS)[:6], cfg), cfg)
        r = S.play_rounds(g, "walker_flash" if cfg.magazine else "walker", "camper", np.arange(6), seed=1)
        assert set(np.unique(r["outcome"])) <= {-1, 0, 1} and r["first_seen"].shape == (6, 3)
        over = np.asarray(r["t"]) < g.T
        a_out = ~np.asarray(r["alive"])[:, :2].any(1)
        d_out = ~np.asarray(r["alive"])[:, 2]
        assert (a_out | d_out)[over].all()                                                # ended early: a side is out


def test_one_brain_trains_for_a_team(tmp_path):
    """Assault's pair shares one brain (two samples per round decision), widened from a 1v1 brain; Defense learns
    against two scripted walkers; both resume and evaluate."""
    from ailearns.agents.ppo_cqb import CQBPPOConfig, evaluate, load, train
    small = {"n_envs": 32, "n_steps": 16, "minibatch": 256, "hidden": 32, "buildings": 8}
    train(CQBPPOConfig(**small, total_steps=32 * 16), tmp_path / "solo")
    train(CQBPPOConfig(**small, total_steps=2 * 32 * 16, teams=(2, 1), init_from=f"{tmp_path / 'solo'}:1"),
          tmp_path / "pair")
    train(CQBPPOConfig(**small, total_steps=3 * 32 * 16, teams=(2, 1)), tmp_path / "pair", resume=True)
    params, ck = load(tmp_path / "pair" / "checkpoints" / "u00003.pt")
    assert params["body.0.weight"].shape[1] == TWO.obs_size and tuple(ck["cqb"]["team_sizes"]) == (2, 1)
    solo, _ = load(tmp_path / "solo" / "checkpoints" / "u00001.pt")
    start, _ = load(tmp_path / "pair" / "checkpoints" / "u00000.pt")
    assert np.allclose(start["body.0.weight"][:, :107], solo["body.0.weight"]) and not start["body.0.weight"][:, 107:].any()
    r = evaluate(str(tmp_path / "pair"), "3", n=4)
    assert r["result"]["alive"].shape == (4, 3) and 0 <= r["win_rate"] <= 1
    train(CQBPPOConfig(**small, total_steps=32 * 16, teams=(2, 1), side="defense", opponent="walker"),
          tmp_path / "defense")
    assert evaluate(str(tmp_path / "defense"), "1", n=4)["result"]["first_seen"].shape == (4, 3)


def test_gallery_of_styles(tmp_path):
    """MAP-Elites end to end, tiny: the start brain fills a cell, children fill or take cells, every cell's brain is
    saved where the footage tools find it, and the best is a normal checkpoint."""
    import json

    from ailearns.agents.ppo_cqb import CQBPPOConfig, load, train
    from ailearns.agents.qd_cqb import CQBQDConfig
    from ailearns.agents.qd_cqb import train as gallery
    small = {"n_envs": 32, "n_steps": 16, "minibatch": 256, "hidden": 32, "buildings": 8}
    train(CQBPPOConfig(**small, total_steps=32 * 16, teams=(2, 1), flashbangs=(2, 0)), tmp_path / "pair")
    gallery(CQBQDConfig(init_from=f"{tmp_path / 'pair'}:1", opponent="camper", teams=(2, 1), flashbangs=(2, 0),
                        children=6, rounds=2, generations=2, buildings=8, checkpoint_every=1), tmp_path / "g")
    g = json.loads((tmp_path / "g" / "gallery.json").read_text())
    assert g["generation"] == 2 and 1 <= len(g["cells"]) <= 16
    for c in g["cells"]:
        params, ck = load(tmp_path / "g" / "cells" / c["cell"] / "checkpoints" / "u00000.pt")
        assert params["body.0.weight"].shape[1] == TWO.obs_size + 8 and 0 <= c["won"] <= 1   # (+ flashbang inputs)
    assert (tmp_path / "g" / "checkpoints" / "u00002.pt").exists()
    lines = [json.loads(x) for x in open(tmp_path / "g" / "metrics.jsonl")]
    assert [x["update"] for x in lines] == [0, 1, 2] and lines[0]["cells"] >= 1
    # a second gallery seeded with the first one's cells: they're all in from generation 0
    gallery(CQBQDConfig(init_from=f"{tmp_path / 'pair'}:1", opponent="camper", teams=(2, 1), flashbangs=(2, 0),
                        children=6, rounds=2, generations=1, buildings=8, checkpoint_every=1,
                        seed_gallery=str(tmp_path / "g")), tmp_path / "g2")
    first = json.loads(open(tmp_path / "g2" / "metrics.jsonl").readline())
    assert first["cells"] >= len(g["cells"])


# ── habits (ep10 step 9): loud collisions, the rifle coming up after running ──

WALL = [((19.9, 5.0), (20.1, 25.0))]          # a wall along x = 20


def _run_into_wall(cfg, start=(16.0, 10.0), face=0.0, run=True, n=20):
    """A0 runs (or walks) the way it faces from `start`; the defender stands 13 m from where it meets the wall, too far
    to hear running steps (10 m) all the way. -> states after each decision."""
    g = _open_game(cfg, WALL)
    st = _pose(g, [start, (12, 20), (19.6, 23)], [face, 0.0, np.pi])
    out = []
    for _ in range(n):
        st = g.step(st, _acts(a0={"fwd": 1, "run": 1 if run else -1}))[0]
        out.append(st)
    return g, out


def test_habit_rules_are_off_by_default():
    assert CFG.bump == 0 and CFG.ready == 0 and replace(CFG, ready=0.3).obs_size == CFG.obs_size + 1
    g, sts = _run_into_wall(TWO)
    assert not any(bool(s.bumped.any()) for s in sts) and all(int(s.ready_t.min()) == g.ready_steps for s in sts)
    assert all(float(s.sound[D, 0, 2]) == 0 for s in sts)                     # running steps: too far to hear


def test_running_into_a_wall_is_loud():
    g, sts = _run_into_wall(replace(TWO, bump=15.0))
    hit = [i for i, s in enumerate(sts) if s.bumped[A0]]
    assert hit and float(sts[hit[0]].sound[D, 0, 2]) > 0                      # the thud carries 13 m
    assert all(float(s.sound[D, 0, 2]) == 0 for s in sts[:hit[0]])             # the running before it didn't
    # walking into it, or running along it, is quiet
    _, walked = _run_into_wall(replace(TWO, bump=15.0), run=False, n=30)
    assert not any(bool(s.bumped[A0]) for s in walked)
    _, slid = _run_into_wall(replace(TWO, bump=15.0), start=(19.6, 8.0), face=np.pi / 2 - 0.15)   # glancing
    assert not any(bool(s.bumped[A0]) for s in slid)


def test_the_rifle_comes_up_after_running_or_a_collision():
    cfg = replace(TWO, ready=0.3)
    g = _open_game(cfg)
    st = g.step(_pose(g, [(14, 11.5), (12, 10), (17, 10)], [np.pi / 2, 0.0, 0.0]), _acts())[0]
    full = float(g.hit_chance(st, A1, target=D))
    assert full > 0.5
    for k in range(4):
        assert abs(float(g.hit_chance(st._replace(ready_t=st.ready_t.at[A1].set(k)), A1, target=D)) - full * k / 3) < 1e-5
    s = st                                                                     # run, then stop: 0, then 1, 2, 3, 3
    for _ in range(6):
        s = g.step(s, _acts(a1={"fwd": 1, "run": 1}))[0]
    got = [int(s.ready_t[A1])]
    for _ in range(4):
        s = g.step(s, _acts())[0]
        got.append(int(s.ready_t[A1]))
    assert got == [0, 1, 2, 3, 3]
    g2, sts = _run_into_wall(replace(TWO, bump=15.0, ready=0.3))              # a collision knocks it down too
    hit = next(i for i, x in enumerate(sts) if x.bumped[A0])
    assert int(sts[hit].ready_t[A0]) == 0
    # it knows how far its rifle is up: one number, last; everything before as without the rule
    o, o0 = np.asarray(g.observe(s, A1)), np.asarray(_open_game(TWO).observe(s, A1))
    assert len(o) == len(o0) + 1 and np.array_equal(o[:-1], o0) and o[-1] == 1.0


def test_a_hard_collision_stuns():
    """stun 0.5 s: after running into a wall, 5 decisions of nothing (no moving, turning or firing), then free."""
    cfg = replace(TWO, stun=0.5)
    g, sts = _run_into_wall(cfg)
    hit = next(i for i, s in enumerate(sts) if int(s.stun_t[A0]) > 0)
    st = sts[hit]
    assert int(st.stun_t[A0]) == 5 and not bool(st.bumped.any())             # (no thud: bump is a separate rule)
    s, faces = st, []
    for _ in range(5):                                                          # trying to turn back and fire
        s = g.step(s, _acts(a0={"fwd": -1, "fire": 1}).at[A0, 2].set(1.0))[0]
        faces.append(float(s.face[A0]))
        assert not bool(s.fired[A0]) and np.allclose(s.pos[A0], st.pos[A0], atol=1e-6)
    assert faces[-1] == float(st.face[A0]) and int(s.stun_t[A0]) == 0
    s = g.step(s, _acts(a0={"fwd": -1}).at[A0, 2].set(1.0))[0]                # free again
    assert float(s.face[A0]) != float(st.face[A0]) and not np.allclose(s.pos[A0], st.pos[A0])
    _, walked = _run_into_wall(cfg, run=False, n=30)                            # walking into it: no stun
    assert not any(int(x.stun_t[A0]) for x in walked)
    o, o0 = np.asarray(g.observe(st, A0)), np.asarray(_open_game(TWO, WALL).observe(st, A0))
    assert len(o) == len(o0) + 1 and np.array_equal(o[:-1], o0) and o[-1] == 1.0
    assert CFG.stun == 0


def test_route_leads_to_the_enemy_it_knows_of():
    """ep11 10d route: an operator on the feed reads the way to the defender. Following that hint, a short step at a
    time, gets it to him in real houses without ever walking through a wall (a straight line to him would)."""
    cfg = replace(TWO, share_sight=False, intel=1, route=True)
    g = CQBGame(make_buildings(list(TEST_SEEDS)[:8], cfg), cfg)
    plain = CQBGame(g.b, replace(cfg, route=False))
    assert cfg.obs_size == plain.cfg.obs_size + 4
    obs = jax.jit(lambda s: g.observe(s, A0)[-4:])
    walls = lambda s: (np.asarray(jnp.concatenate([g.b.wall_lo[s.building], g.b.cover_lo[s.building]])),  # noqa: E731
                       np.asarray(jnp.concatenate([g.b.wall_hi[s.building], g.b.cover_hi[s.building]])))
    through_walls = 0
    for k in range(12):
        st = g.init(jax.random.PRNGKey(k))
        st = st._replace(door_open=jnp.ones_like(st.door_open))
        assert np.allclose(np.asarray(g.observe(st, A0))[:-4], np.asarray(plain.observe(st, A0)))   # the rest as before
        lo, hi = walls(st)
        him = np.asarray(st.pos[D])
        through_walls += not bool(cqb._clear(st.pos[A0], st.pos[D][None], jnp.asarray(lo) - 0.25,
                                             jnp.asarray(hi) + 0.25)[0])
        o1 = np.asarray(g.observe(st, A1))[-4:]
        assert not o1.any()                                                            # A1 knows of nobody
        for _ in range(400):
            o = np.asarray(obs(st))
            assert o[3] == 1.0 and 0 < o[2] <= 1
            if np.linalg.norm(np.asarray(st.pos[A0]) - him) < 0.6:
                break
            f = float(st.face[A0])
            d = np.array([np.cos(f) * o[0] - np.sin(f) * o[1], np.sin(f) * o[0] + np.cos(f) * o[1]])   # to the world
            new = np.asarray(st.pos[A0]) + 0.2 * d
            inside = ((new > lo + 0.05) & (new < hi - 0.05)).all(1)                    # never into a wall or crate
            assert not inside.any(), (k, new)
            st = st._replace(pos=st.pos.at[A0].set(jnp.asarray(new, jnp.float32)))
        assert np.linalg.norm(np.asarray(st.pos[A0]) - him) < 0.6, k                   # got there
    assert through_walls >= 6                                                          # most starts: he's behind walls


def test_big_houses():
    """ep11 10j: the "big" preset: 8-12 rooms, ~2.6x the floor, within the padding; every spawn point can be walked
    to from every entry (the waypoint graph joins them), and a 2v1 round runs with the same observation size."""
    cfg = replace(TWO, **cqb.HOUSES["big"])
    seeds = list(TEST_SEEDS)[:12]
    rooms = [len(cqb.make_building(s, cfg)["rooms"]) for s in seeds]
    assert min(rooms) >= 8 and max(rooms) <= 12
    g = CQBGame(make_buildings(seeds, cfg), cfg)
    assert cfg.obs_size == TWO.obs_size
    nodes, dist = np.asarray(g.b.nav_nodes), np.asarray(g.b.nav_dist)
    for bi in range(len(seeds)):
        def near(p):
            real = nodes[bi, :, 0] > -500
            return int(np.argmin(np.where(real, np.linalg.norm(nodes[bi] - p, axis=1), np.inf)))
        for e in np.asarray(g.b.entry[bi])[: int(g.b.n_entries[bi])]:
            for sp in np.asarray(g.b.spawn[bi])[::8]:
                assert dist[bi, near(e), near(sp)] < 1e3
    r = S.play_rounds(g, "walker", "camper", np.arange(len(seeds)), seed=0)
    assert np.isin(r["outcome"], [-1, 0, 1]).all()


def test_feed_is_its_own_numbers():
    """ep11 10m feed (1v1, real houses): the attacker reads, as 8 new numbers at the end, where the defender is (its
    frame), which way he faces, the way there and a 1; its memory of him is untouched (unlike `intel`); everything
    before the 8 is exactly as without; the defender reads zeros there."""
    one = replace(CFG, flashbangs=(2, 0), magazine=30, health=100.0, react=0.3, route=True)
    cfg = replace(one, feed=1)
    g = CQBGame(make_buildings(list(TEST_SEEDS)[:6], cfg), cfg)
    plain = CQBGame(g.b, one)
    assert cfg.obs_size == one.obs_size + 8
    for k in range(6):
        st = g.init(jax.random.PRNGKey(k))
        st = st._replace(face=st.face.at[1].set(0.7))
        o, o0 = np.asarray(g.observe(st, 0)), np.asarray(plain.observe(st, 0))
        assert np.allclose(o[:-8], o0)                                                   # the rest: unchanged
        f = o[-8:]
        p, me = np.asarray(st.pos), float(st.face[0])
        rel = p[1] - p[0]
        want = np.array([np.cos(-me) * rel[0] - np.sin(-me) * rel[1], np.sin(-me) * rel[0] + np.cos(-me) * rel[1]])
        assert np.allclose(f[:2], want / cfg.sight, atol=1e-5)
        assert np.allclose(f[2:4], [np.sin(0.7 - me), np.cos(0.7 - me)], atol=1e-5)
        assert f[6] > 0 and f[7] == 1.0 and abs(np.linalg.norm(f[4:6]) - 1) < 1e-4
        assert int(st.since[0, 1]) == g.T                                                # memory untouched
        assert not np.asarray(g.observe(st, 1))[-8:].any()                               # the defender: zeros


def test_the_pilot_only_talks_and_cant_be_hit():
    """ep11 10z pilot: Assault operator 1 (A0) is a drone pilot: its moves, turns and shots do nothing, its words go
    out; the defender, aiming straight at it in the open, neither sees nor hits it; nobody's shot stops at it; the
    round is lost when the other operator is down, though the pilot stands."""
    from ailearns.games.cqb import SAY
    cfg = replace(TWO, share_sight=False, radio=(4, 0), pilot=True)
    g, plain = _open_game(cfg), _open_game(replace(cfg, pilot=False))
    st = _pose(g, [(12, 10), (12, 14), (17, 10)], [0.0, 0.0, np.pi])        # the defender faces the pilot, 5 m
    a = np.asarray(_acts(a0={"fwd": 1, "fire": 1}, d={"fire": 1}))
    a = np.concatenate([a, -np.ones((3, 1), np.float32)], 1)
    a[0, SAY] = 2.0                                                          # the pilot says word 2
    a[0, 2] = 1.0                                                            # and tries to turn
    a = jnp.asarray(a)
    new = g.step(st, a)[0]
    assert np.allclose(np.asarray(new.pos[A0]), [12, 10]) and float(new.face[A0]) == 0.0      # it stayed put
    assert int(new.said[A0]) == 2                                                             # but it talked
    assert not bool(new.seen[D, A0]) and not bool(new.seen[A0, D])
    assert bool(plain.step(st, a)[0].seen[D, A0])                                             # without: seen
    alive = _after_shots(g, st, a)
    assert alive[:, A0].all()                                                                 # never hit
    assert not _after_shots(plain, st, a)[:, A0].all()                                        # without: hit
    down = st._replace(alive=st.alive.at[A1].set(False))
    end = g.step(down, _acts())[0]
    assert bool(end.done) and int(end.outcome) == -1                                         # lost, pilot standing
    assert not bool(plain.step(down, _acts())[0].done)


def test_callout_protocol_words():
    """ep11 10zc callout: the pilot says the protocol word for where the defender is as the entry operator (A1) faces,
    whatever word its own action asks for: 1 ahead, 2 left, 3 right, 4 behind; no new word once he's down."""
    from ailearns.games.cqb import SAY
    cfg = replace(TWO, share_sight=False, radio=(4, 0), pilot=True, callout=True)
    g = _open_game(cfg)
    a = np.concatenate([np.asarray(_acts()), -np.ones((3, 1), np.float32)], 1)
    a[0, SAY] = 3.0                                                          # the pilot "wants" to say 3
    a = jnp.asarray(a)
    for where, word in [((17, 12), 1), ((12, 17), 2), ((12, 7), 3), ((7, 12), 4), ((14, 16), 2), ((14, 8), 3), ((15, 13), 1)]:
        st = _pose(g, [(25, 20), (12, 12), where], [0.0, 0.0, 0.0])           # A1 faces east (+x); +y is its left
        assert int(g.step(st, a)[0].said[A0]) == word, (where, word)
    for where, word in [((17, 12), 3), ((7, 12), 2), ((12, 17), 1), ((12, 7), 4)]:   # A1 faces north now
        st = _pose(g, [(25, 20), (12, 12), where], [0.0, np.pi / 2, 0.0])
        assert int(g.step(st, a)[0].said[A0]) == word, (where, word)
    st = _pose(g, [(25, 20), (12, 12), (17, 12)], [0.0, 0.0, 0.0])
    st = st._replace(alive=st.alive.at[D].set(False))
    assert int(g.step(st, a)[0].said[A0]) == 0                               # nobody standing: silent
    assert int(_open_game(replace(cfg, callout=False)).step(_pose(g, [(25, 20), (12, 12), (17, 12)], [0.0] * 3),
                                                             a)[0].said[A0]) == 3   # without: its own word


def test_callout_way_calls_the_route_not_the_bearing():
    """ep11 10ze callout_way: in a real house, with the defender behind walls, the word is the direction of the next
    step of the walk to him (the route hint's own direction, sorted into ahead / left / right / behind as the entry
    operator faces), which differs from the bearing in some rounds."""
    cfg = replace(TWO, share_sight=False, radio=(4, 0), pilot=True, callout=True, callout_way=True, route=True)
    g = CQBGame(make_buildings(list(TEST_SEEDS)[:8], cfg), cfg)
    bearing = CQBGame(g.b, replace(cfg, callout_way=False))
    differ, n = 0, 0
    rng = np.random.default_rng(0)
    for k in range(32):                                          # the entry operator somewhere inside, facing anywhere
        st = g.init(jax.random.PRNGKey(k))
        cells = np.asarray(g.b.cell_point[st.building])
        st = st._replace(pos=st.pos.at[1].set(jnp.asarray(cells[rng.integers(len(cells))], jnp.float32)),
                         face=st.face.at[1].set(float(rng.uniform(-np.pi, np.pi))))
        n += 1
        u = np.asarray(g._way(st, 1, st.pos[D], jnp.asarray(True))[:2])
        a = np.degrees(np.arctan2(u[1], u[0]))
        want = 1 if abs(a) <= 45 else 2 if 45 < a <= 135 else 3 if -135 <= a < -45 else 4
        assert int(g.callout_word(st)) == want, k
        differ += int(g.callout_word(st)) != int(bearing.callout_word(st))
    assert differ >= 4, (differ, n)                                                  # not just the bearing again
