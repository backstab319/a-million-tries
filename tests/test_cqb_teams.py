"""CQB with teams (ep10): 2 Assault vs 1 Defense. Operators 0 and 1 are the Assault pair, 2 the defender."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.games import cqb_scripts as S
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


def test_team_shares_sightings():
    g = _open_game()
    st = g.step(_pose(g, [(12, 10), (12, 12), (17, 10)], [0.0, np.pi, 0.0]), _acts())[0]   # A1 looks away
    assert bool(st.seen[A0, D]) and not bool(st.seen[A1, D])
    assert int(st.since[A1, D]) == 0 and np.allclose(st.mem[A1, D], st.pos[D])          # A0 called it out
    assert not st.seen[D].any() and int(st.since[D, A0]) == g.T                         # the defender saw nobody
    o = np.asarray(g.observe(st, A1))
    assert o[24] == 0.0 and np.abs(o[29:31]).sum() > 0                                   # not seen, but where it was


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
