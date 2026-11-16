"""The JAX creatures against the numpy reference (plan section 4, port steps 2-4)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ailearns.games import ecosystem_jax as ej  # noqa: E402
from ailearns.games import ecosystem_jax_life as jl  # noqa: E402
from ailearns.games.ecosystem import World, WorldConfig  # noqa: E402
from ailearns.games.ecosystem_life import N_HID, N_IN, N_OUT, Life, LifeConfig  # noqa: E402
from ailearns.games.ecosystem_space import nearest_by_sector  # noqa: E402

from test_ecosystem_jax import numpy_storms  # noqa: E402

OB2 = N_IN * N_HID + N_HID + N_HID * N_OUT   # where the output biases start in a brain's weights


def make_life(seed=0, n=160, size=40, families=24, rain=True, thirst=True, gut=False, cold=False, cold_breath=False):
    """A small crowded world: random brains, `families` young placed on their mothers (milk, carried young)."""
    world = World.make(seed, WorldConfig(size=size, rain=rain))
    life = Life(world, n, seed, LifeConfig(thirst=thirst, gut=gut, cold=cold, cold_breath=cold_breath, max_creatures=n + 40))
    rng = np.random.default_rng(seed + 100)
    moms = rng.choice(n, families, replace=False)
    kids = rng.choice(np.setdiff1d(np.arange(n), moms), families, replace=False)
    life.mass[kids] = 0.2 * life.adult[kids]
    life.reserve[kids] = 0.5 * life.reserve_cap(life.mass[kids], life.fat[kids])   # (11a) not 2.5x full: they nurse
    life.parent[kids] = life.uid[moms]
    life.parent2[kids[::2]] = life.uid[kids[1::2]]
    life.pos[kids] = life.pos[moms] + rng.normal(0, 0.3, (families, 2))
    life.pos[kids] = np.clip(life.pos[kids], 0.1, size * world.cfg.cell_m - 0.1)
    life.calling[: n // 3] = True
    life.dry[:] = rng.random(len(life.dry)) * 0.05 * life.mass
    life.books.start = life.energy()   # the young were shrunk by hand: book what is there now
    return life


def no_sex_no_fights(life):
    """Port step 2 has no births or fights: brains that never want either."""
    life.w[:, OB2 + 3] = -100.0
    life.w[:, OB2 + 4] = -100.0


def test_brute_force_neighbours_match_the_compiled_numpy_search():
    rng = np.random.default_rng(3)
    pos = rng.random((300, 2)) * 40
    pos[10] = pos[11]                    # a tie
    pos[20] = pos[21] + [1e-3, 0]
    heading = rng.uniform(-np.pi, np.pi, 300)
    free = rng.random(300) < 0.9
    f = np.flatnonzero(free)
    near, near_d, sec_b, sec_d = nearest_by_sector(pos[f], heading[f], 6.0, 40.0, (-60.0, -20.0, 20.0, 60.0))
    with jax.enable_x64(True):
        jn, jnd, jb, jd = (np.asarray(v) for v in jl.nearest_brute(jnp.array(pos), jnp.array(heading),
                                                                    jnp.array(free), 6.0)[:4])
    np.testing.assert_array_equal(np.where(jn[f] >= 0, np.searchsorted(f, np.maximum(jn[f], 0)), -1), near)
    np.testing.assert_allclose(jnd[f], near_d)
    np.testing.assert_array_equal(np.where(jb[f] >= 0, np.searchsorted(f, np.maximum(jb[f], 0)), -1), sec_b)
    np.testing.assert_allclose(jd[f], sec_d)
    assert np.all(jn[~free] == -1)


@pytest.mark.parametrize("seed", range(4))
def test_the_grid_gives_the_brute_force_answer(seed):
    """Clumps, ties, creatures on bucket edges and the map's edge, some not free: same answer to the bit."""
    rng = np.random.default_rng(seed)
    n, size = 600, 96.0
    pos = rng.random((n, 2)) * size
    pos[:100] = pos[:100] * 0.1 + 40                     # a clump (~10 per bucket)
    pos[100:110] = pos[110:120]                           # exact ties
    pos[120:140, 0] = 6.0 * rng.integers(0, 16, 20)       # on bucket edges
    pos[140:150] = [[0.0, 0.0], [size, size]] * 5         # on the map's edge (and past it: clipped buckets)
    heading = rng.uniform(-np.pi, np.pi, n)
    free = rng.random(n) < 0.9
    args = (jnp.array(pos), jnp.array(heading), jnp.array(free), 6.0)
    with jax.enable_x64(True):
        brute = jl.nearest_brute(*args)
        grid = jl.nearest_grid(*args, size_m=size, per_bucket=64)
    for b, g in zip(brute[:4], grid[:4]):
        np.testing.assert_array_equal(np.asarray(g), np.asarray(b))
    assert int(grid[4]) == 0


def test_a_full_bucket_is_counted_not_hidden():
    pos = jnp.full((40, 2), 10.0) + jnp.arange(40)[:, None] * 0.01
    out = jl.nearest_grid(pos, jnp.zeros(40), jnp.ones(40, bool), 6.0, size_m=48.0, per_bucket=16)
    assert int(out[4]) == 24
    assert np.all(np.asarray(out[0]) >= 0)               # the listed 16 are still seen by everyone


@pytest.mark.parametrize("gut", [False, True])   # (11a, 11b) and with a gut that fits the body and the cold
def test_bodies_and_brains_match_numpy_tick_by_tick(gut):
    """Same world, same brains, no births or fights: 5 days of senses, moving, eating, milk, carrying, thirst,
    growth and deaths agree with numpy to float64 rounding, plants and carcasses included.

    Each tick starts JAX from numpy's state: left free, the two drift apart from rounding (1e-14) by ~1.5x a tick
    (brains react to senses that depend on where they are: chaos, measured in devlog 10b), so only one tick at a
    time can be compared exactly."""
    with jax.enable_x64(True):
        life = make_life(gut=gut, cold=gut, cold_breath=gut)
        no_sex_no_fights(life)
        wcfg, cfg = life.world.cfg, life.cfg
        tick = jax.jit(lambda st, wd: jl.tick(wcfg, cfg, st, wd, jax.random.key(0)))
        plant = jax.jit(lambda st, wd, s: jl.plant_step(wcfg, cfg, st, wd, s))
        per = jl.plant_every(wcfg, cfg)
        assert len(life._riders(np.flatnonzero(life.alive))[0]) > 10
        nursed = drank = died = 0
        for t in range(5 * 60):
            st, wd = jl.from_numpy(life)
            i = np.flatnonzero(life.alive)
            x_np = life.senses(i)
            storms = numpy_storms(life.world) if (life.ticks + 1) % per == 0 else None
            life.step()
            wd, info = tick(st, wd)
            if storms is not None:
                wd = plant(st, wd, storms)
            np.testing.assert_allclose(np.asarray(info["x"])[i], x_np, rtol=1e-9, atol=1e-9, err_msg=f"tick {t}")
            nursed += int((np.asarray(info["milk"]) > 0).sum())
            drank += int(np.asarray(info["drank"]).sum())
            died += len(i) - int(life.alive.sum())
            c = wd.c
            np.testing.assert_array_equal(np.asarray(c.alive), life.alive)
            a = life.alive
            for name in ("pos", "heading", "mass", "reserve", "dry", "age", "mem"):
                np.testing.assert_allclose(np.asarray(getattr(c, name))[a], getattr(life, name)[a],
                                           rtol=1e-9, atol=1e-9, err_msg=f"{name} at tick {t}")
            np.testing.assert_array_equal(np.asarray(c.calling)[a], life.calling[a])
            np.testing.assert_allclose(np.asarray(wd.meat).reshape(-1), life.meat, rtol=1e-9, atol=1e-6)
            for name in ("plants", "dead", "fruit", "wet"):
                np.testing.assert_allclose(np.asarray(getattr(wd.plants, name)), getattr(life.world, name),
                                           rtol=1e-9, atol=1e-6)
            for name in ("heat", "dung", "decay"):
                assert float(getattr(wd.books, name)) == pytest.approx(getattr(life.books, name), rel=1e-12)
            assert int(wd.books.deaths) == life.books.deaths
            assert abs(float(jl.error(wd, cfg))) < 1e-9
        assert nursed > 0 and drank > 0 and died > 0, (nursed, drank, died)


def test_many_worlds_run_compiled_in_float32():
    """Three worlds stacked, a month of ticks and plant steps in one call: books to float32 rounding."""
    lives = [make_life(seed=s, n=120) for s in range(3)]
    for life in lives:
        no_sex_no_fights(life)
    wcfg, cfg = lives[0].world.cfg, lives[0].cfg
    pairs = [jl.from_numpy(life) for life in lives]
    st = jax.tree.map(lambda *a: jnp.stack(a), *(p[0] for p in pairs))
    wd = jax.tree.map(lambda *a: jnp.stack(a), *(p[1] for p in pairs))
    run = jax.jit(jax.vmap(lambda s, w, k: jl.run(wcfg, cfg, s, w, k, 4 * 30)))
    out = run(st, wd, jax.random.split(jax.random.key(0), 3))
    err = np.asarray(jax.vmap(lambda w: jl.error(w, cfg))(out))
    assert np.all(np.abs(err) < 1e-4), err
    assert np.all(np.asarray(out.ticks) == 30 * 60)
    assert np.all(np.asarray(out.plants.day) == 30.0)


# --- port step 4: fights and births, with numpy's own random numbers ------------------------------------------------
class Recorder:
    """Wraps numpy's Generator and logs every draw (name, result), in order."""
    def __init__(self, rng):
        self.rng, self.log = rng, []

    def __getattr__(self, name):
        f = getattr(self.rng, name)

        def call(*a, **k):
            out = f(*a, **k)
            self.log.append((name, np.array(out)))
            return out
        return call


def numpy_draws(life, i, log, run, cap):
    """JAX `Draws` holding the numbers numpy drew this tick (`log`), each on the slot it was used for. `run(draws)`
    runs the JAX tick from the same state and returns its info; the masks it gives tell whose numbers came next.
    Rows nobody used get numbers that change nothing (never settle, never breed alone)."""
    cfg = life.cfg                       # i: the living before the tick (numpy's indices are into them)
    from ailearns.games.ecosystem_life import N_W
    B = jl.MAX_BIRTHS
    d = dict(settle=np.ones(cap), won=np.zeros(cap), prio=np.arange(cap) + cap, solo=np.ones(cap),
             pick=np.zeros((cap, 5)), gene=np.zeros((cap, 5)), heading=np.zeros(cap), w_pick=np.zeros((B, N_W)),
             w_noise=np.zeros((B, N_W)))
    n_kids = 0                           # brain draws are one row per child, in birth order
    mk = lambda: jl.Draws(*(jnp.asarray(d[k], jnp.int32 if k == "prio" else None) for k in jl.Draws._fields))
    log = list(log)
    nxt = lambda name: (lambda e: (e[1], e[0] == name or pytest.fail(f"expected {name}, got {e[0]}")))(log.pop(0))[0]
    info = run(mk())
    hit = np.flatnonzero(np.asarray(info["hit"]))
    if len(hit):
        d["settle"][hit], d["won"][hit] = nxt("random"), nxt("random")
        info = run(mk())
    cand = np.asarray(info["cand"])
    sig = (cfg.sigma_gene, cfg.sigma_gene, cfg.sigma_gene, cfg.sigma_diet, cfg.sigma_alone)
    if cand.any():
        perm = i[nxt("permutation")]
        d["prio"][perm] = np.arange(len(perm))
        info = run(mk())
        pairs = np.flatnonzero(np.asarray(info["taken"]))
        pairs = pairs[np.argsort(d["prio"][pairs])]          # numpy's order: the loop's
        for g in range(5):
            d["pick"][pairs, g] = nxt("random")
            d["gene"][pairs, g] = nxt("normal") / sig[g]
        info = run(mk())
        born = np.asarray(info["born"])[pairs]
        kids = pairs[born]
        d["w_pick"][:len(kids)] = nxt("random")[born]
        d["heading"][kids] = nxt("uniform")
        d["w_noise"][:len(kids)] = nxt("normal") / cfg.sigma_w
        n_kids = len(kids)
        info = run(mk())
    wait = np.asarray(info["wait"])
    if wait.any():
        d["solo"][i] = nxt("random")
        info = run(mk())
        solo = np.flatnonzero(np.asarray(info["solo"]))
        if len(solo):
            for g in range(5):
                d["gene"][solo, g] = nxt("normal") / sig[g]
            info = run(mk())
            kids = solo[np.asarray(info["born"])[solo]]
            d["heading"][kids] = nxt("uniform")
            d["w_noise"][n_kids:n_kids + len(kids)] = nxt("normal") / cfg.sigma_w
    assert not log, [e[0] for e in log]
    return mk()


def test_fights_and_births_match_numpy_with_its_own_dice():
    """Brains that attack and breed: each tick, JAX gets numpy's random numbers on the right slots, and kills,
    pairs, children (genes, brains, slots, uids, parents) and the books come out the same, over 4 days."""
    with jax.enable_x64(True):
        life = make_life(seed=1, n=220, size=24)
        ob = life.w[:, OB2 + 3:OB2 + 5]
        ob[:] = np.random.default_rng(0).normal(1.0, 1.0, ob.shape)   # most want to breed and to attack
        life.w[:, OB2 + 3:OB2 + 5] = ob
        life.alone[:] = 0.5
        life.reserve[:] = life.reserve_cap(life.mass, life.fat) * 0.9
        life.books.start = life.energy()
        wcfg, cfg = life.world.cfg, life.cfg
        cap = len(life.alive) + 300
        tick = jax.jit(lambda st, wd, d: jl.tick(wcfg, cfg, st, wd, None, draws=d))
        plant = jax.jit(lambda st, wd, s: jl.plant_step(wcfg, cfg, st, wd, s))
        per = jl.plant_every(wcfg, cfg)
        seen = dict(kills=0, pairs=0, alone=0, births=0)
        for t in range(4 * 60):
            if len(life.alive) < cap - 50 and life.alive.sum() > 0.8 * len(life.alive):
                life._grow(len(life.alive))                  # numpy doubles its arrays; JAX's cap stays
            st, wd = jl.from_numpy(life, cap)
            rec = Recorder(life.rng)
            life.rng = rec
            storms = numpy_storms(life.world) if (life.ticks + 1) % per == 0 else None
            before = life.books.births, life.books.births_alone, life.books.kills
            i = np.flatnonzero(life.alive)
            life.step()
            life.rng = rec.rng
            d = numpy_draws(life, i, rec.log, lambda dd: tick(st, wd, dd)[1], cap) if rec.log else None
            wd, info = tick(st, wd, d if d is not None else jl.draw(jax.random.key(t), cap))
            if storms is not None:
                wd = plant(st, wd, storms)
            m = len(life.alive)
            c = wd.c
            np.testing.assert_array_equal(np.asarray(c.alive)[:m], life.alive, err_msg=f"alive at tick {t}")
            assert not np.asarray(c.alive)[m:].any()
            a = life.alive
            for name in ("pos", "heading", "mass", "adult", "birth", "fat", "diet", "alone", "reserve", "dry", "age",
                         "w"):
                np.testing.assert_allclose(np.asarray(getattr(c, name))[:m][a], getattr(life, name)[a], rtol=1e-9,
                                           atol=1e-9, err_msg=f"{name} at tick {t}")
            for name in ("gen", "uid", "parent", "parent2"):
                np.testing.assert_array_equal(np.asarray(getattr(c, name))[:m][a], getattr(life, name)[a],
                                              err_msg=f"{name} at tick {t}")
            assert int(wd.books.capped) == 0
            for name in ("kills", "births", "births_alone", "deaths", "full"):
                assert int(getattr(wd.books, name)) == getattr(life.books, name), (name, t)
            for name in ("heat", "dung", "decay"):
                assert float(getattr(wd.books, name)) == pytest.approx(getattr(life.books, name), rel=1e-12)
            assert int(wd.next_uid) == life.next_uid
            assert abs(float(jl.error(wd, cfg))) < 1e-9
            seen["kills"] += life.books.kills - before[2]
            seen["alone"] += life.books.births_alone - before[1]
            seen["births"] += life.books.births - before[0]
            seen["pairs"] += int(np.asarray(info["taken"]).sum())
        assert all(v > 3 for v in seen.values()), seen


def test_parallel_matching_gives_the_loops_pairs():
    """`match` against numpy's loop on random proposal graphs (chains, cycles, mutual pairs)."""
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = 60
        near = rng.integers(0, n, n)
        near[near == np.arange(n)] = (near[near == np.arange(n)] + 1) % n
        propose = rng.random(n) < 0.7
        perm = rng.permutation(np.flatnonzero(propose))
        prio = np.full(n, 10 * n)
        prio[perm] = np.arange(len(perm))
        used, ma = set(), []
        for a in perm:
            b = int(near[a])
            if a not in used and b not in used:
                used |= {int(a), b}
                ma.append(a)
        taken, paired = jl.match(jnp.array(propose), jnp.array(near, jnp.int32), jnp.array(prio, jnp.int32))
        assert set(np.flatnonzero(np.asarray(taken))) == set(int(a) for a in ma)
        assert set(np.flatnonzero(np.asarray(paired))) == used


def test_probes_count_what_numpy_traces():
    """(port 5) Each tick from numpy's state: food kept by source, calls, attacks, steps and deaths as numpy's
    `trace` reports them (and its books)."""
    with jax.enable_x64(True):
        life = make_life(seed=2, n=200, size=24)
        life.w[:, OB2 + 3] = -100.0                          # no births (their dice would need mapping)
        life.w[:, OB2 + 4] = -100.0
        wcfg, cfg = life.world.cfg, life.cfg
        got = {}

        def trace(lf, i, attack, meat, plant, fruit, milk, call, **_):
            got.update(n=len(i), meat=meat.sum(), plant=plant.sum(), fruit=fruit.sum(), milk=milk.sum(),
                       calls=int(call.sum()), attacks=int(attack.sum()))
        life.trace = trace

        def one(st, wd):
            w2, info = jl.tick(wcfg, cfg, st, wd, jax.random.key(0))
            return jl.observe(wcfg, cfg, st, wd, info, jl.no_probe())
        one = jax.jit(one)
        total = jl.no_probe()
        deaths = 0
        for t in range(90):
            st, wd = jl.from_numpy(life)
            d0 = life.books.deaths
            p = one(st, wd)
            life.step()
            assert int(p.steps) == got["n"] and int(p.calls) == got["calls"] and int(p.attacks) == got["attacks"]
            food = np.asarray(p.food)
            np.testing.assert_allclose([food[0], food[1], food[2] + food[3], food[4]],
                                       [got["fruit"], got["meat"], got["plant"], got["milk"]], rtol=1e-9, atol=1e-9)
            assert int(np.asarray(p.dead).sum()) == life.books.deaths - d0
            assert int(p.drink) == int(life.drank.sum())
            deaths += life.books.deaths - d0
            total = jax.tree.map(lambda a, b: a + b, total, p)
        assert deaths > 0 and float(total.food[2]) > 0


def test_scramble_redraws_only_the_living_brains():
    life = make_life(seed=3, n=50)
    st, wd = jl.from_numpy(life, 80)
    s = jl.scramble(wd, jax.random.key(1))
    a = np.asarray(wd.c.alive)
    assert not np.allclose(np.asarray(s.c.w)[a], np.asarray(wd.c.w)[a])
    np.testing.assert_array_equal(np.asarray(s.c.w)[~a], np.asarray(wd.c.w)[~a])


def test_births_past_the_tick_limit_are_counted():
    """With room for one child's brain a tick, the other would-be parents are counted in `capped`, and pay nothing."""
    life = make_life(seed=1, n=220, size=24)
    life.w[:, OB2 + 3] = 50.0                                # everyone wants to breed
    life.alone[:] = 1.0
    life.reserve[:] = life.reserve_cap(life.mass, life.fat)
    wcfg, cfg = life.world.cfg, life.cfg
    st, wd = jl.from_numpy(life, 600)
    d = jl.draw(jax.random.key(0), 600, max_births=1)
    d = d._replace(solo=jnp.zeros(600))                      # every unpaired one tries alone
    w2, info = jl.tick(wcfg, cfg, st, wd, None, draws=d)
    assert int(w2.books.births) == 1 and int(w2.books.capped) > 5
    assert int(np.asarray(info["born"]).sum()) == 1
    paid = np.asarray(wd.c.reserve) - np.asarray(w2.c.reserve)
    alive = np.asarray(wd.c.alive) & np.asarray(w2.c.alive)
    assert (paid[alive] > 1000).sum() <= 2                   # only the one birth's parents paid for a child
