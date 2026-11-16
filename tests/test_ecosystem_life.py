"""Plant-eaters: real scaling laws, honest books through births and deaths, water that stops you."""

import numpy as np
import pytest

from ailearns.games.ecosystem import DESERT, WATER, World, WorldConfig
from ailearns.games.ecosystem_life import Life, LifeConfig, N_OUT

QUIET = [-5.0, 0.0, 0.0]   # the outputs after the first five: don't call (5a), memory 0 (5b)

WCFG = WorldConfig(size=48)


def _life(n=200, seed=0, cfg=LifeConfig()):
    return Life(World.make(seed, WCFG), n, seed, cfg)


def test_physiology_follows_the_scaling_laws():
    life = _life(1)
    m = np.array([1.0, 16.0])
    rest_day = life.bmr_s(m) * life.cfg.day_s
    assert rest_day == pytest.approx([293.0, 293.0 * 8])                     # Kleiber: 16^0.75 = 8
    assert life.walk_kj_m(m) * 1000 == pytest.approx(10.7 * m ** 0.684)      # Taylor, real cost per metre
    v = life.top_speed(m)
    assert 2.5 < v[0] < 4.5 and v[1] > v[0]                                    # a few m/s, a bit faster when bigger
    walking_power = life.walk_kj_m(m) * v
    real_rest = life.bmr_s(m) / life.real_s_per_s                             # resting kJ per real second
    assert walking_power == pytest.approx(10 * real_rest)                     # top speed costs the aerobic scope


def test_fat_weighs_what_it_weighs():
    life = _life(1)
    lean = np.array([2.0, 2.0])
    carried = lean * np.array([1.0, 1.4])                                     # lean, and carrying 40% fat
    assert life.top_speed(lean, carried)[1] < life.top_speed(lean, carried)[0]
    assert life.walk_kj_m(carried)[1] == pytest.approx(life.walk_kj_m(carried)[0] * 1.4 ** 0.684)
    assert life.reserve_cap(2.0, 0.4) == pytest.approx(0.4 * 2.0 * 37000)
    life.reserve[0] = life.reserve_cap(life.mass[0], life.fat[0])
    assert life.carried(0) == pytest.approx(life.mass[0] * (1 + life.fat[0]))


def test_books_balance_through_births_and_deaths():
    life = _life(300)
    for _ in range(3000):
        life.step()
        assert abs(life.error()) < 1e-9
    b = life.books
    assert b.deaths > 0 and b.heat > 0 and b.decay > 0


def test_books_balance_when_parents_split_every_chance():
    """Brains forced to eat and split: births happen and every one is paid for."""
    life = _life(100)
    life.w[:, -N_OUT:] = [0.0, -3.0, 5.0, 5.0, -5.0] + QUIET   # output biases: no turn, slow, eat, split, no attack
    life.w[:, :-N_OUT] = 0.0
    life.adult[:] = life.mass[:] = 1.0
    life.reserve[:] = life.reserve_cap(1.0, life.fat[:])
    life.books.start = 0.0
    life.books.start = life.energy()
    for _ in range(600):
        life.step()
        assert abs(life.error()) < 1e-9
    assert life.books.births > 0
    assert life.gen[life.alive].max() >= 1


def test_a_creature_that_never_eats_starves_and_its_body_becomes_a_carcass():
    life = _life(1)
    life.w[0, -N_OUT:] = [0.0, -5.0, -5.0, -5.0, -5.0] + QUIET  # never eats, barely moves
    life.w[0, :-N_OUT] = 0.0
    body = life.cfg.body_kj_kg * life.mass[0]
    ticks = 0
    while life.alive[0]:
        life.step()
        ticks += 1
    days = ticks / life.cfg.day_s
    stored = life.cfg.start_fill * life.reserve_cap(life.adult[0], life.fat[0])
    assert days == pytest.approx(stored / (life.bmr_s(life.adult[0]) * life.cfg.day_s), rel=0.1)
    cell, _ = life._cell(life.pos[0])
    assert life.meat[cell] + life.books.decay == pytest.approx(body, rel=0.01)
    assert abs(life.error()) < 1e-9


def test_nobody_walks_into_water_or_off_the_map():
    life = _life(400, seed=2)
    life.w[:, -N_OUT:] = [0.3, 5.0, -5.0, -5.0, -5.0] + QUIET  # run in circles, flat out
    life.w[:, :-N_OUT] = 0.0
    n, cm = WCFG.size, WCFG.cell_m
    for _ in range(400):
        life.step()
        a = life.alive
        rc = np.floor(life.pos[a] / cm).astype(int)
        assert ((rc >= 0) & (rc < n)).all()
        assert (life.world.biome[rc[:, 0], rc[:, 1]] != WATER).all()


def test_children_inherit_the_fat_gene_with_a_small_mutation():
    life = _life(100)
    life.w[:, -N_OUT:] = [0.0, -3.0, 5.0, 5.0, -5.0] + QUIET
    life.w[:, :-N_OUT] = 0.0
    life.adult[:] = life.mass[:] = 1.0
    life.fat[:] = 0.3
    life.reserve[:] = life.reserve_cap(1.0, 0.3)
    for _ in range(120):
        life.step()
    kids = life.alive & (life.gen >= 1)
    assert kids.any()
    assert np.abs(np.log(life.fat[kids] / 0.3)).max() < 0.3
    ids = life.uid[life.alive]
    assert len(np.unique(ids)) == len(ids)                      # every creature its own id
    assert (life.parent[kids] < life.uid[kids]).all() and (life.parent[kids] >= 0).all()


def test_adding_creatures_later_keeps_the_books():
    life = _life(50)
    for _ in range(200):
        life.step()
    n = life.alive.sum()
    life.add(30, diet=(0.9, 1.0))
    new = life.alive & (life.uid >= life.next_uid - 30)
    assert life.alive.sum() == n + 30 and new.sum() == 30
    assert (life.diet[new] >= 0.9).all() and (life.gen[new] == 0).all()
    for _ in range(200):
        life.step()
        assert abs(life.error()) < 1e-9


def test_school_hunters_copy_the_saved_genes_and_keep_the_books():
    life = _life(50)
    src = np.flatnonzero(life.alive)[:3]
    life.diet[src] = 0.95
    like = life.genes(src)
    idx = life.add(40, like=like)
    # each newcomer is one of the three, mutated a little: its brain is closest to its own source's
    d = ((life.w[idx][:, None] - like["w"][None]) ** 2).sum(-1)
    assert (d.min(1) < 0.1 * d.max(1)).all()
    assert (abs(life.diet[idx] - 0.95) < 0.2).all() and (life.mass[idx] == life.adult[idx]).all()
    for _ in range(200):
        life.step()
        assert abs(life.error()) < 1e-9


def test_young_drink_their_mothers_milk_near_her_and_the_books_balance():
    """(8f) A young one next to its mother gains, she loses ~1/0.7 of that; far from her, nothing; a stranger, nothing."""
    life = _pair(diet_a=1.0, mass_a=4.0, mass_b=0.4, attack=False)
    life.adult[1] = 4.0                                   # creature 1: a young one, a tenth grown
    life.reserve[0] = life.reserve_cap(life.mass[0], life.fat[0])
    life.reserve[1] = 0.0
    life.parent[1] = life.uid[0]
    i = np.array([0, 1])
    m, fat, no = life.mass[i].copy(), life.fat[i].copy(), np.zeros(2, bool)
    r0 = life.reserve[i].copy()
    got = life._nurse(i, m, fat, no)
    assert got[1] > 0 and got[0] == 0
    assert np.isclose(r0[0] - life.reserve[0], got[1] / life.cfg.milk_eff)
    life.pos[1] += [5.0, 0.0]                             # out of reach
    assert life._nurse(i, m, fat, no).sum() == 0
    life.pos[1] -= [5.0, 0.0]
    life.parent[1] = life.uid[0] + 99                     # not hers
    assert life._nurse(i, m, fat, no).sum() == 0
    life.parent[1] = life.uid[0]
    life.books.heat -= (1 - life.cfg.milk_eff) * got[1] / life.cfg.milk_eff   # the checks' milk, before the books
    life.books.start = 0.0
    life.books.start = life.energy()
    for _ in range(300):
        life.step()
        assert abs(life.error()) < 1e-9


def test_same_seed_same_story():
    a, b = _life(100, 5), _life(100, 5)
    for _ in range(500):
        a.step(), b.step()
    assert np.array_equal(a.alive, b.alive) and np.array_equal(a.pos, b.pos)


def _pair(diet_a=1.0, mass_a=2.0, mass_b=1.0, attack=True):
    """Creature 0 right behind creature 1, facing it; both brains fixed by their output biases."""
    life = _life(2)
    life.w[:, :-N_OUT] = 0.0
    life.w[0, -N_OUT:] = [0.0, -9.0, 5.0, -5.0, 5.0 if attack else -5.0] + QUIET   # still, eats, attacks
    life.w[1, -N_OUT:] = [0.0, -9.0, -5.0, -5.0, -5.0] + QUIET                      # still, doesn't eat
    land = np.argwhere(life.world.biome != WATER)[0] * WCFG.cell_m + WCFG.cell_m / 2
    life.pos[0], life.pos[1] = land, land + [0.3, 0.0]
    life.heading[0] = 0.0
    life.adult[:2] = life.mass[:2] = [mass_a, mass_b]
    life.diet[:2] = [diet_a, 0.0]
    life.reserve[:2] = 0.3 * life.reserve_cap(life.mass[:2], life.fat[:2])
    life.books.start = 0.0
    life.books.start = life.energy()
    return life


def test_an_attacker_kills_and_eats_what_it_killed():
    life = _pair(mass_a=4.0, mass_b=1.0)
    for _ in range(60):
        life.step()
        assert abs(life.error()) < 1e-9
        if not life.alive[1]:
            break
    if not life.alive[0]:
        pytest.skip("the attacker lost (6% chance at 4x the mass)")
    assert not life.alive[1] and life.books.kills == 1
    life.w[0, -1] = -5.0                     # stop attacking
    before = life.reserve[0]
    for _ in range(30):
        life.step()
    assert life.reserve[0] > before                                     # eating the carcass beats resting + attacking
    assert abs(life.error()) < 1e-9


def _fights(ma, mb, diet_a, n=40):
    """The share of n fights creature 0 (attacking) wins."""
    w = 0
    for seed in range(n):
        life = _pair(mass_a=ma, mass_b=mb, diet_a=diet_a)
        life.rng = np.random.default_rng(seed)
        for _ in range(60):
            life.step()
            if not (life.alive[0] and life.alive[1]):
                break
        assert life.alive[0] != life.alive[1]             # the fight was settled, one of them died
        w += life.alive[0]
        assert abs(life.error()) < 1e-9
    return w / n


def test_a_fight_goes_both_ways():
    """Attacking something 3x your mass mostly gets you killed; attacking something a third your mass mostly wins."""
    assert _fights(1.0, 3.0, 0.0) < 0.3 and _fights(3.0, 1.0, 0.0) > 0.7


def test_a_meat_eater_is_built_to_kill():
    """(8c) The same size: a grass-eater's attack is a coin flip, a pure meat-eater's wins ~4 in 5 (fights like 2x)."""
    assert 0.3 < _fights(2.0, 2.0, 0.0, 60) < 0.7 and _fights(2.0, 2.0, 1.0, 60) > 0.65


def test_no_attack_no_kill_and_the_eyes_see_who_is_ahead():
    life = _pair(attack=False)
    x = life.senses(np.array([0, 1]))
    assert x[0, 12] > 0.9                                               # creature 1 right ahead of 0, touching
    assert x[0, 15] == pytest.approx(np.log(life.carried(1) / life.carried(0)) / 3)
    assert x[0, 11] == 0 and x[0, 13] == 0                              # nothing left or right
    for _ in range(120):
        life.step()
    assert life.alive[1] and life.books.kills == 0


def test_any_gut_takes_meat_and_only_a_plant_eater_takes_plants():
    """(4c) A carcass feeds a plant-eater too (it eats it first); a meat-eater gets nothing from plants."""
    for diet in (0.0, 1.0):
        life = _pair(diet_a=diet, attack=False)
        life.alive[1] = False
        cell, _ = life._cell(life.pos[0])
        life.meat[cell] = 50_000.0
        flat = life.world.plants.reshape(-1)
        life.world.ledger.start += flat[cell] * -1 + life.world.capacity.reshape(-1)[cell]
        flat[cell] = life.world.capacity.reshape(-1)[cell]
        life.books.start = 0.0
        life.books.start = life.energy() + float(life.meat.sum())
        meat0, plants0 = life.meat[cell], life.world.plants.reshape(-1)[cell]
        for _ in range(10):
            life.step()
        assert life.meat[cell] < meat0
        if diet == 1.0:
            assert life.world.plants.reshape(-1)[cell] == pytest.approx(plants0)
        assert abs(life.error()) < 1e-9


def test_the_eyes_see_carcasses():
    life = _life(1)
    eye = life.pos[0] + life.cfg.sight_m * np.array([np.cos(life.heading[0]), np.sin(life.heading[0])])
    cell, inside = life._cell(eye[None])
    if not inside[0] or life.world.biome.reshape(-1)[cell[0]] == WATER:
        pytest.skip("eye off the map")
    life.meat[cell[0]] = 5 * life.cfg.body_kj_kg
    x = life.senses(np.array([0]))
    assert x[0, 18] == pytest.approx(1.0) and x[0, 17] == 0 and x[0, 19] == 0


def test_carcasses_rot_fast_in_summer_slow_in_winter_and_the_rot_is_booked():
    """(4e) 5 days at midsummer take 20%/day, at midwinter 20% / 2.5^2 = 3.2%/day."""
    for day, rate in ((180.0, 0.2), (0.0, 0.2 / 2.5 ** 2)):
        life = _life(1)
        life.world.day = day
        life.alive[0] = False
        life.meat[:] = 0.0
        life.meat[100] = 10_000.0
        life.books.start = 0.0
        life.books.start = float(life.meat.sum())
        for _ in range(int(life.cfg.day_s) * 5):
            life.step()
        assert life.meat[100] == pytest.approx(10_000 * np.exp(-rate * 5), rel=0.01)
        assert abs(life.error()) < 1e-9


def test_the_trace_only_watches():
    """(3g) A world with an observer runs exactly as one without, and the observer sees every kill the books count."""
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("eco_kills", Path(__file__).parents[1] / "scripts" / "eco_kills.py")
    eco_kills = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eco_kills)
    a, b = _life(300, seed=3), _life(300, seed=3)
    b.trace = watch = eco_kills.Watch(round(b.cfg.day_s / b.cfg.dt))
    for _ in range(3000):
        a.step()
        b.step()
    assert a.books.kills > 0
    assert np.array_equal(a.alive, b.alive) and np.array_equal(a.pos, b.pos) and np.array_equal(a.reserve, b.reserve)
    assert watch.kills.sum() == b.books.kills
    assert watch.starved + watch.old + watch.kills.sum() == b.books.deaths


def test_dead_grass_is_poorer_food():
    """(3h) A plant-eater gets dead_quality as much from a kJ of dead grass as from green; the rest is dung."""
    gain = {}
    for kind in ("none", "green", "dead"):
        life = _life(1, seed=0)
        w = life.world
        cell, _ = life._cell(life.pos[:1])
        w.plants.reshape(-1)[cell] = w.cfg.stubble * w.capacity.reshape(-1)[cell] + 500.0 * (kind == "green")
        w.dead.reshape(-1)[cell] = 500.0 * (kind == "dead")
        w.ledger.start = w.total_plants()
        life.diet[0] = 0.0
        r_was = life.reserve[0]
        life.reserve[0] = 0.2 * life.reserve_cap(life.mass[:1], life.fat[:1])[0]
        life.books.start += life.reserve[0] - r_was            # (booked, so the error check holds)
        life.w[0] = 0.0
        life.w[0, -N_OUT:] = [0, -5, 5, -5, -5] + QUIET   # output biases: stand still, eat, don't split or attack
        r0 = life.reserve[0]
        life.step()
        gain[kind] = life.reserve[0] - r0
        assert abs(life.error()) < 1e-9
    green, dead = gain["green"] - gain["none"], gain["dead"] - gain["none"]
    assert green > 0 and dead / green == pytest.approx(life.world.cfg.dead_quality, rel=1e-6)


def test_the_arrays_grow_instead_of_capping_births():
    """(3i) Past the arrays' first size, births are kept (the arrays double), up to the ceiling; books exact."""
    from dataclasses import replace
    life = _life(40, seed=1, cfg=replace(LifeConfig(), max_creatures=48, ceiling=200))
    uid = life.uid[life.alive].copy()
    life.add(30)                                       # 70 creatures: past 48
    assert life.alive.sum() == 70 and len(life.alive) == 96
    assert np.array_equal(life.uid[:40], uid)          # the first ones unchanged, in their slots
    life.add(500)                                      # past the ceiling: only 200 fit
    assert life.alive.sum() == 200 and len(life.alive) == 200 and len(life.w) == 200


def test_the_nose_smells_kind_and_family_ahead():
    """(4b) Creature 0 smells creature 1 ahead: how alike their diets are, and whether it's its child or sibling;
    creature 1 has nobody ahead, so it smells nothing."""
    life = _pair(diet_a=1.0, attack=False)          # 0 a meat-eater, 1 a plant-eater
    life.uid[:2], life.parent[:2] = [10, 11], [-1, -1]
    x = life.senses(np.array([0, 1]))
    assert x[0, 20] == pytest.approx(0.0) and x[0, 21] == 0             # another kind, no kin
    assert x[1, 20] == 0 and x[1, 21] == 0                              # nobody ahead of 1
    life.diet[1] = 0.9
    life.parent[1] = 10                                                 # 1 is 0's child
    x = life.senses(np.array([0, 1]))
    assert x[0, 20] == pytest.approx(0.9) and x[0, 21] == 1
    life.parent[:2] = [7, 7]                                            # siblings
    assert life.senses(np.array([0, 1]))[0, 21] == 1
    life.parent[:2] = [-1, -1]                                          # two first creatures aren't siblings
    assert life.senses(np.array([0, 1]))[0, 21] == 0


def test_a_grazer_keeps_60_percent_of_meat_a_meat_eater_85():
    """(4c) The same carcass meal is worth digest_meat_grazer to a plant-eater and digest_meat to a meat-eater."""
    gain = {}
    for diet in (0.0, 1.0):
        for food in (0.0, 50_000.0):
            life = _pair(diet_a=diet, attack=False)
            life.alive[1] = False
            cell, _ = life._cell(life.pos[0])
            flat = life.world.plants.reshape(-1)
            life.world.ledger.start -= flat[cell] - life.world.cfg.stubble * life.world.capacity.reshape(-1)[cell]
            flat[cell] = life.world.cfg.stubble * life.world.capacity.reshape(-1)[cell]   # no grass to eat
            life.meat[cell] = food
            life.books.start = 0.0
            life.books.start = life.energy() + float(life.meat.sum())
            r0 = life.reserve[0]
            life.step()
            assert abs(life.error()) < 1e-9
            gain[diet, food] = life.reserve[0] - r0
    grazer, hunter = gain[0.0, 50_000.0] - gain[0.0, 0.0], gain[1.0, 50_000.0] - gain[1.0, 0.0]
    cfg = LifeConfig()
    assert grazer > 0 and grazer / hunter == pytest.approx(cfg.digest_meat_grazer / cfg.digest_meat, rel=1e-6)


def test_an_omnivore_keeps_three_quarters_of_a_grazers_plant_energy():
    """(4d) Plant energy kept falls with diet^2: a grazer 0.5, an omnivore at 0.5 diet 0.375, a meat-eater 0."""
    gain = {}
    for diet in (0.0, 0.5, 1.0):
        for green in (0.0, 500.0):
            life = _pair(diet_a=diet, attack=False)
            life.alive[1] = False
            w = life.world
            cell, _ = life._cell(life.pos[0])
            flat = w.plants.reshape(-1)
            target = w.cfg.stubble * w.capacity.reshape(-1)[cell] + green
            w.ledger.start += target - flat[cell]
            flat[cell] = target
            life.books.start = 0.0
            life.books.start = life.energy()
            r0 = life.reserve[0]
            life.step()
            assert abs(life.error()) < 1e-9
            gain[diet, green] = life.reserve[0] - r0
    got = {d: gain[d, 500.0] - gain[d, 0.0] for d in (0.0, 0.5, 1.0)}
    assert got[0.0] > 0 and got[0.5] / got[0.0] == pytest.approx(0.75, rel=1e-6) and got[1.0] == pytest.approx(0.0, abs=1e-9)


def test_calling_costs_twice_the_resting_rate_on_top():
    """(5a) Two identical creatures standing still, one calling: the caller pays (1 + call_bmr) x resting."""
    life = _life(2)
    life.w[:, :-N_OUT] = 0.0
    life.w[0, -N_OUT:] = [0.0, -20.0, -5.0, -5.0, -5.0, 5.0, 0.0, 0.0]    # still, calls
    life.w[1, -N_OUT:] = [0.0, -20.0, -5.0, -5.0, -5.0, -5.0, 0.0, 0.0]   # still, quiet
    life.mass[1], life.adult[1], life.fat[1] = life.mass[0], life.adult[0], life.fat[0]
    life.reserve[1] = life.reserve[0]
    life.books.start = 0.0
    life.books.start = life.energy()
    r0 = life.reserve[:2].copy()
    life.step()
    spent = r0 - life.reserve[:2]
    walk = life.walk_kj_m(life.carried(np.array([0])))[0] * life.top_speed(life.mass[:1])[0] / (1 + np.exp(20)) * life.cfg.dt
    rest = life.bmr_s(life.mass[:1])[0] * life.cfg.dt
    assert spent[1] == pytest.approx(rest + walk, rel=1e-6)
    assert spent[0] - spent[1] == pytest.approx(life.cfg.call_bmr * rest, rel=1e-6)
    assert life.calling[0] and not life.calling[1]
    assert abs(life.error()) < 1e-9


def test_a_call_is_heard_on_the_right_side_and_fades_with_distance():
    """(5a) A creature facing +y with a caller 8 m to its right (+x) hears it louder on the right; 16 m off it's
    quieter; with nobody calling it hears nothing (the test map is 96 m across)."""
    life = _life(3)
    life.calling[:] = False
    assert np.all(life.hear(np.array([1, 2])) == 0)
    life.pos[0] = [70.0, 50.0]
    life.calling[0] = True
    life.pos[1], life.heading[1] = [62.0, 50.0], np.pi / 2           # facing +y; the caller is at +x: its right
    life.pos[2], life.heading[2] = [54.0, 50.0], np.pi / 2           # the same, further off
    h = life.hear(np.array([1, 2]))
    assert h[0, 1] > 0                                                 # louder on the right
    assert h[0, 0] > h[1, 0] > 0                                       # and quieter with distance
    life.heading[1] = -np.pi / 2                                      # turned round: now on its left
    assert life.hear(np.array([1]))[0, 1] < 0


def test_memory_is_what_the_brain_wrote_last_step():
    """(5b) A brain that writes (1, -2) to memory reads tanh of that back as senses 24-25 the next step; newborns and
    new creatures start blank."""
    life = _life(2)
    life.w[:, :-N_OUT] = 0.0
    life.w[:, -N_OUT:] = [0.0, -20.0, -5.0, -5.0, -5.0, -5.0, 1.0, -2.0]
    assert np.all(life.senses(np.array([0, 1]))[:, 24:26] == 0)
    life.step()
    x = life.senses(np.array([0, 1]))
    assert x[:, 24:26] == pytest.approx(np.tanh([[1.0, -2.0], [1.0, -2.0]]))
    life.add(1)
    assert np.all(life.mem[np.flatnonzero(life.alive)[-1]] == 0)


def _lovers(ready_b=True, sexual=True):
    """(6a) Creature 0 right behind 1, facing it; both grown, well fed, standing still; 0 ready to breed, 1 as asked.
    No mutation, so a child's genes are exactly its parents'."""
    from dataclasses import replace
    cfg = replace(LifeConfig(), sexual=sexual, sigma_w=0.0, sigma_gene=0.0, sigma_diet=0.0)
    life = _pair(mass_a=1.5, mass_b=1.5, attack=False)                  # the same size: the same upkeep
    life.cfg = cfg
    life.birth[:2] = 0.1                                                   # small newborns: one parent can pay too
    life.fat[:2] = life.fat[0]
    life.w[0, -N_OUT:] = [0.0, -20.0, -5.0, 5.0, -5.0] + QUIET            # still, ready
    life.w[1, -N_OUT:] = [0.0, -20.0, -5.0, 5.0 if ready_b else -5.0, -5.0] + QUIET
    life.w[1, :-N_OUT] = 0.5                                               # tell the parents' weights apart
    life.diet[:2] = [0.0, 0.0]
    life.mass[:2] = life.adult[:2]
    life.reserve[:2] = life.reserve_cap(life.mass[:2], life.fat[:2])
    life.uid[:2], life.parent[:2], life.parent2[:2] = [10, 11], [-1, -1], [-1, -1]
    life.alone[:2] = 0.0                                                   # (6b) never breed alone here
    life.books.start = 0.0
    life.books.start = life.energy()
    return life


def test_two_ready_creatures_have_one_child_and_share_its_cost():
    life = _lovers()
    r0 = life.reserve[:2].copy()
    life.step()
    kids = np.flatnonzero(life.alive)[2:]
    assert len(kids) == 1 and life.books.births == 1
    k = kids[0]
    assert {life.parent[k], life.parent2[k]} == {10, 11}
    paid = r0 - life.reserve[:2]
    assert paid[0] - paid[1] == pytest.approx(0.0, abs=1e-6)              # each paid half (and the same upkeep)
    from_a, from_b = life.w[k] == life.w[0], life.w[k] == life.w[1]
    assert np.all(from_a | from_b) and from_a.sum() > 0 and from_b[:-N_OUT].sum() > 0   # a mix of both
    assert abs(life.error()) < 1e-9


def test_unlike_kinds_have_no_child_together():
    """(8g) A grass-eater and a meat-eater, both ready and touching: no child (diet genes 1.0 apart); alike: one."""
    for diet_b, expect in ((1.0, 0), (0.2, 1)):
        life = _lovers()
        life.diet[1] = diet_b
        life.step()
        assert life.books.births == expect


def test_no_partner_ready_no_child_and_splitting_alone_still_works():
    life = _lovers(ready_b=False)
    life.step()
    assert life.books.births == 0
    life = _lovers(ready_b=False, sexual=False)
    life.step()
    assert life.books.births == 1 and life.parent2[np.flatnonzero(life.alive)[-1]] == -1


def test_with_no_mate_about_the_alone_gene_decides():
    """(6b) Ready, well fed, nobody in reach: alone = 1 splits alone within a few days (a copy, the full cost, counted
    as alone); alone = 0 never does."""
    for gene, expect in ((1.0, True), (0.0, False)):
        life = _lovers(ready_b=False)
        life.alive[1] = False
        life.alone[0] = gene
        life.rng = np.random.default_rng(0)
        for _ in range(int(4 * life.cfg.day_s / life.cfg.dt)):
            life.reserve[0] = life.reserve_cap(life.mass[:1], life.fat[:1])[0]   # kept fed (not booked: no error check)
            life.step()
            if life.books.births:
                break
        assert (life.books.births > 0) == expect and life.books.births_alone == life.books.births
        if expect:
            k = np.flatnonzero(life.alive)[-1]
            assert life.parent[k] == 10 and life.parent2[k] == -1


def test_a_grazer_on_fruit_eats_it_first_and_keeps_80_percent():
    """(7b) Fruit, a carcass and grass underfoot: the fruit goes first; with fruit and without, the extra kept is 0.8."""
    gain = {}
    for fruit in (0.0, 300.0):
        life = _pair(diet_a=0.0, attack=False)
        life.alive[1] = False
        w = life.world
        cell, _ = life._cell(life.pos[0])
        w.fruit.reshape(-1)[cell] = fruit
        flat = w.plants.reshape(-1)
        w.ledger.start += w.cfg.stubble * w.capacity.reshape(-1)[cell] - flat[cell]
        flat[cell] = w.cfg.stubble * w.capacity.reshape(-1)[cell]            # no grass to eat
        w.ledger.start += fruit
        life.books.start = 0.0
        life.books.start = life.energy()
        r0 = life.reserve[0]
        life.step()
        assert abs(life.error()) < 1e-9
        gain[fruit] = life.reserve[0] - r0
    eaten = 300.0 - life.world.fruit.reshape(-1)[cell]
    assert eaten > 0 and (gain[300.0] - gain[0.0]) == pytest.approx(life.cfg.digest_fruit * eaten, rel=1e-6)


def test_fruit_is_smelt_from_afar_on_the_right_side():
    """(7b) A fruit pile 12 m to a creature's right: its scent is stronger on the right, and weaker 24 m off."""
    life = _life(3)
    w = life.world
    w.fruit[:] = 0.0
    w.fruit[35, 25] = 5000.0                                     # cell (35, 25): around (71, 51) m
    life.pos[1], life.heading[1] = [59.0, 51.0], np.pi / 2       # facing +y; the pile is at +x: its right
    life.pos[2], life.heading[2] = [47.0, 51.0], np.pi / 2       # the same, 24 m off (scent carries ~3 cells)
    s = life.smell_fruit(np.array([1, 2]))
    assert s[0, 1] > 0 and s[0, 0] > s[1, 0] > 0
    life.heading[1] = -np.pi / 2
    assert life.smell_fruit(np.array([1]))[0, 1] < 0


def test_compiled_brains_match_numpy():
    """(ep17 speed) The compiled `think` gives the numpy einsum's outputs (to rounding)."""
    import ailearns.games.ecosystem_life as el
    life = _pair()
    rng = np.random.default_rng(1)
    life.w = rng.normal(0, 1, (50, el.N_W))
    i = rng.permutation(50)[:30]
    x = rng.normal(0, 1, (30, el.N_IN))
    got = life.think(i, x)
    keep, el._think_nb = el._think_nb, None
    try:
        want = life.think(i, x)
    finally:
        el._think_nb = keep
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)


def _mother_and_young():
    """(8h) Creature 0 a grown mother walking ahead; creature 1 her young (a tenth grown) a little behind her."""
    life = _pair(diet_a=0.0, mass_a=4.0, mass_b=0.4, attack=False)
    life.adult[1] = 4.0
    life.parent[1] = life.uid[0]
    life.w[0, -N_OUT:] = [0.0, 5.0, -5.0, -5.0, -5.0] + QUIET       # walks fast, no eating
    life.reserve[:2] = life.reserve_cap(life.mass[:2], life.fat[:2])
    life.books.start = 0.0
    life.books.start = life.energy()
    return life


def test_carried_young_ride_with_their_mother_and_she_pays_for_it():
    """(8h) The young stays at its mother's place as she walks; she covers less ground and pays more than alone."""
    with_young = _mother_and_young()
    alone = _mother_and_young()
    from dataclasses import replace
    alone.cfg = replace(alone.cfg, carry=False)
    p0 = with_young.pos[0].copy()
    r0 = with_young.reserve[0], alone.reserve[0]
    for _ in range(5):
        with_young.step()
        alone.step()
    np.testing.assert_allclose(with_young.pos[1], with_young.pos[0])
    went = np.hypot(*(with_young.pos[0] - p0)), np.hypot(*(alone.pos[0] - p0))
    assert 0 < went[0] < went[1]
    per_m = (r0[0] - with_young.reserve[0]) / went[0], (r0[1] - alone.reserve[0]) / went[1]
    assert per_m[0] > per_m[1]
    assert abs(with_young.error()) < 1e-9


def test_a_carried_young_cant_be_attacked_until_its_mother_dies():
    """(8h) A hunter facing a carried young sees nothing ahead; once the mother is gone, it sees (and can hit) it."""
    life = _mother_and_young()
    life.w[0, -N_OUT:] = [0.0, -9.0, -5.0, -5.0, -5.0] + QUIET      # the mother stands still
    life.pos[1] = life.pos[0]
    life.add(1, diet=(1.0, 1.0))
    hunter = np.flatnonzero(life.alive)[-1]
    life.w[hunter] = 0.0
    life.pos[hunter] = life.pos[0] - [0.5, 0.0]
    life.heading[hunter] = 0.0
    i = np.flatnonzero(life.alive)
    life.senses(i)
    h = int(np.flatnonzero(i == hunter)[0])
    assert life._ahead[h] == int(np.flatnonzero(i == 0)[0])         # the mother, not the young
    life.alive[0] = False
    i = np.flatnonzero(life.alive)
    life.senses(i)
    h = int(np.flatnonzero(i == hunter)[0])
    assert life._ahead[h] == int(np.flatnonzero(i == 1)[0])


def _thirsty(on_shore):
    """(9b) One creature standing still, eating, with thirst on; on a lake's shore or far from water."""
    from dataclasses import replace
    life = _pair(diet_a=0.0, mass_a=2.0, attack=False)
    life.cfg = replace(life.cfg, thirst=True)
    life.alive[1] = False
    shore = np.flatnonzero(life.drinkable())
    inland = np.flatnonzero(~life.drinkable() & (life.world.biome.reshape(-1) == DESERT))
    n, cm = life.world.cfg.size, life.world.cfg.cell_m
    c = shore[0] if on_shore else inland[0]
    life.pos[0] = (np.array([c // n, c % n]) + 0.5) * cm
    life.reserve[0] = life.reserve_cap(life.mass[:1], life.fat[:1])[0]
    life.dry[0] = 0.5 * life.cfg.dry_max_l_kg * life.mass[0]
    return life


def test_thirst_drink_at_the_shore_dry_out_far_from_water():
    """(9b) Half parched: eating at a lake's shore it drinks to full; in the desert (no green, no water) it dries
    out and dies of thirst, not hunger."""
    life = _thirsty(on_shore=True)
    life.step()
    assert life.dry[0] == 0.0 and life.drank.sum() == 1
    life = _thirsty(on_shore=False)
    life.world.plants[:] = 0.0
    for _ in range(int(10 * life.cfg.day_s / life.cfg.dt)):
        life.reserve[0] = life.reserve_cap(life.mass[:1], life.fat[:1])[0]   # never hungry
        life.step()
        if not life.alive[0]:
            break
    assert not life.alive[0]


def test_green_grass_brings_water_dead_grass_almost_none():
    """(9b) Per kJ: green grass ~25x the water of dead grass; spending outruns dead grass alone."""
    c = LifeConfig()
    assert c.water_green / c.water_dead > 20 and c.water_dead < c.water_per_kj < 1.5 * c.water_green


def test_a_small_gut_keeps_less_of_the_grass():
    """(11a) With `gut`, the share of grass energy kept follows plant_keep: ~0.38 at 40 g, ~0.49 at 2 kg, ~0.53 at 8 kg
    (a small gut passes fibre before it ferments), and the books still balance."""
    from ailearns.games.ecosystem_life import plant_keep
    cfg = LifeConfig(gut=True)
    assert plant_keep(cfg, 0.0, 0.04) == pytest.approx(0.383, abs=0.002)
    assert plant_keep(cfg, 0.0, 2.0) == pytest.approx(0.485, abs=0.002)
    assert plant_keep(cfg, 0.0, 8.0) == pytest.approx(0.531, abs=0.002)
    assert plant_keep(cfg, 0.5, 2.0) == pytest.approx(0.75 * plant_keep(cfg, 0.0, 2.0))
    kept = {}
    for mass in (0.04, 2.0, 8.0):
        life = _pair(diet_a=0.0, mass_a=mass, attack=False)
        life.cfg = cfg
        life.alive[1] = False
        w = life.world
        cell, _ = life._cell(life.pos[0])
        flat = w.plants.reshape(-1)
        target = w.cfg.stubble * w.capacity.reshape(-1)[cell] + 500.0
        w.ledger.start += target - flat[cell]
        flat[cell] = target
        w.dead.reshape(-1)[cell] = 0.0
        life.books.start = 0.0
        life.books.start = life.energy()
        r0, p0 = life.reserve[0], flat[cell]
        cost = life.bmr_s(life.mass[0]) * life.cfg.dt   # still: it pays only its resting cost this second
        life.step()
        assert abs(life.error()) < 1e-6
        kept[mass] = (life.reserve[0] - r0 + cost) / (p0 - w.plants.reshape(-1)[cell])
    for mass, k in kept.items():
        assert k == pytest.approx(plant_keep(cfg, 0.0, mass), rel=1e-3), (mass, kept)


def test_the_cold_costs_a_mouse_four_times_its_rest_and_a_big_body_nothing():
    """(11b) With `cold`, a still creature pays resting + keeping warm: a 40 g one ~4.3x its resting rate at 0 C and
    ~2x at 20 C; an 8 kg one nothing extra at 20 C. The books balance."""
    from ailearns.games.ecosystem_life import keep_warm_s
    cfg = LifeConfig(cold=True)
    rest = lambda m: 293.0 * m ** 0.75 / cfg.day_s
    for m, air, times in ((0.04, 0.0, 4.31), (0.04, 20.0, 1.98), (2.0, 0.0, 1.59), (8.0, 20.0, 1.0)):
        assert (rest(m) + keep_warm_s(cfg, m, air, rest(m))) / rest(m) == pytest.approx(times, abs=0.01)
    paid = {}
    for mass in (0.04, 8.0):
        life = _pair(diet_a=0.0, mass_a=mass, attack=False)
        life.cfg = cfg
        life.alive[1] = False
        life.w[0, -N_OUT:][2] = -5.0                     # doesn't eat either: only resting and the cold
        life.books.start = 0.0
        life.books.start = life.energy()
        r0, air = life.reserve[0], life.world.temperature()
        life.step()
        assert abs(life.error()) < 1e-6
        r = rest(life.mass[0]) * cfg.dt
        paid[mass] = (r0 - life.reserve[0]) / r
        assert paid[mass] == pytest.approx(1 + keep_warm_s(cfg, life.mass[0], air, r / cfg.dt) / (r / cfg.dt),
                                           rel=1e-3)
    assert paid[0.04] > 1.5 and paid[8.0] < paid[0.04]


def test_keeping_warm_costs_only_the_breaths_water():
    """(11d) With `cold_breath`, a still, unfed creature in the cold loses water_per_kj on its resting cost but only
    breath_water_per_kj on the heat it makes to keep warm."""
    from ailearns.games.ecosystem_life import keep_warm_s
    for breath in (False, True):
        cfg = LifeConfig(cold=True, thirst=True, cold_breath=breath)
        life = _pair(diet_a=0.0, mass_a=0.04, attack=False)
        life.cfg = cfg
        life.alive[1] = False
        life.w[0, -N_OUT:][2] = -5.0
        life.pos[0] = np.argwhere(life.world.biome == DESERT)[0] * WCFG.cell_m + WCFG.cell_m / 2   # no water here
        life.dry[0] = 0.0
        life.books.start = 0.0
        life.books.start = life.energy()
        air = life.world.temperature()
        life.step()
        rest = 293.0 * life.mass[0] ** 0.75 / cfg.day_s
        warm = keep_warm_s(cfg, life.mass[0], air, rest)
        assert warm > rest
        want = cfg.water_per_kj * rest + (cfg.breath_water_per_kj if breath else cfg.water_per_kj) * warm
        assert life.dry[0] == pytest.approx(want * cfg.dt, rel=1e-3), breath
