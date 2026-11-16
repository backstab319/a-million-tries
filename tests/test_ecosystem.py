"""The ecosystem world: honest energy books, plants that respect the soil and the seasons, fair grazing."""

import numpy as np
import pytest

from ailearns.games.ecosystem import DESERT, FOREST, GRASS, WATER, World, WorldConfig

CFG = WorldConfig(size=64)
STEPS_A_YEAR = int(CFG.year_days / CFG.dt_days)


@pytest.fixture
def world():
    return World.make(0, CFG)


def test_same_seed_same_world():
    a, b, c = World.make(3, CFG), World.make(3, CFG), World.make(4, CFG)
    assert np.array_equal(a.biome, b.biome) and np.array_equal(a.plants, b.plants)
    assert not np.array_equal(a.biome, c.biome)


def test_biome_shares_match_the_config(world):
    n = CFG.size ** 2
    water = (world.biome == WATER).sum() / n
    land = 1 - water
    assert water == pytest.approx(CFG.water_frac, abs=0.005)
    assert (world.biome == DESERT).sum() / n / land == pytest.approx(CFG.desert_frac, abs=0.01)
    assert (world.biome == FOREST).sum() / n / land == pytest.approx(CFG.forest_frac, abs=0.01)
    assert (world.biome == GRASS).sum() > 0


def test_books_balance_over_two_years_of_random_grazing(world):
    rng = np.random.default_rng(1)
    for _ in range(2 * STEPS_A_YEAR):
        world.step()
        cells = rng.integers(0, CFG.size ** 2, 300)
        world.graze(cells, rng.random(300) * 200)
        assert abs(world.error()) < 1e-9
    assert world.ledger.sun_in > 0 and world.ledger.eaten > 0 and world.ledger.dieback > 0


def test_plants_stay_between_zero_and_capacity_and_never_grow_in_water(world):
    for _ in range(STEPS_A_YEAR):
        world.step()
        assert (world.plants >= 0).all() and (world.plants <= world.capacity + 1e-9).all()
        assert (world.roots >= 0).all() and (world.roots <= world.root_capacity + 1e-9).all()
    assert (world.plants[world.biome == WATER] == 0).all() and (world.roots[world.biome == WATER] == 0).all()


def test_summer_grows_winter_dies_back(world):
    totals = []
    for _ in range(2 * STEPS_A_YEAR):
        world.step()
        totals.append(world.total_plants())
    year2 = np.asarray(totals[STEPS_A_YEAR:])
    peak, low = np.argmax(year2) / STEPS_A_YEAR, np.argmin(year2) / STEPS_A_YEAR
    assert 0.5 < peak < 1.0          # fullest after midsummer (day 180), before the next winter
    assert low < 0.25 or low > 0.95  # emptiest around midwinter
    assert year2.max() > 1.05 * year2.min()


def test_a_grazed_out_patch_regrows_from_its_neighbours():
    w = World.make(0, CFG)
    grass = np.argwhere(w.biome == GRASS)
    r, c = grass[len(grass) // 2]
    w.plants[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4] = 0.0
    w.roots[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4] = 0.0     # roots too: only seed can bring it back
    w.ledger.start = w.total_plants()           # a fresh book after clearing by hand
    for _ in range(2 * STEPS_A_YEAR):           # seedlings put roots down first: ~1.5 years to fill back in
        w.step()
    assert w.plants[r, c] > 0.5 * w.capacity[r, c]
    assert abs(w.error()) < 1e-9


def test_eaters_on_one_cell_share_what_is_there(world):
    cell = int(np.flatnonzero(world.biome.reshape(-1) == GRASS)[0])
    there = world.plants.reshape(-1)[cell]
    reach = there - CFG.stubble * world.capacity.reshape(-1)[cell]
    got = world.graze([cell, cell, cell], [there, there, 2 * there])
    assert got.sum() == pytest.approx(reach)
    assert got[2] == pytest.approx(2 * got[0])
    assert world.plants.reshape(-1)[cell] == pytest.approx(there - reach)
    assert abs(world.error()) < 1e-9


def test_eating_less_than_there_is_takes_exactly_that(world):
    cell = int(np.flatnonzero(world.biome.reshape(-1) == FOREST)[0])
    before = world.plants.reshape(-1)[cell]
    got = world.graze([cell], [10.0])
    assert got[0] == pytest.approx(10.0)
    assert world.plants.reshape(-1)[cell] == pytest.approx(before - 10.0)


def test_grazers_cant_reach_the_roots(world):
    roots = world.roots.copy()
    cells = np.flatnonzero(world.biome.reshape(-1) != WATER)
    world.graze(cells, np.full(len(cells), 1e9))
    stubble = CFG.stubble * world.capacity.reshape(-1)[cells]
    assert world.plants.reshape(-1)[cells] == pytest.approx(np.minimum(stubble, world.plants.reshape(-1)[cells]))
    assert (world.plants.reshape(-1)[cells] <= stubble + 1e-9).all()
    assert np.array_equal(world.roots, roots)
    assert abs(world.error()) < 1e-9


def test_a_world_grazed_flat_grows_back_from_its_roots():
    """Every reachable shoot eaten every quarter day for half a year: stubble and roots survive, and it comes back."""
    w = World.make(0, CFG)
    land = np.flatnonzero(w.biome.reshape(-1) != WATER)
    for _ in range(STEPS_A_YEAR // 2):
        w.step()
        w.graze(land, np.full(len(land), 1e9))
    assert (w.plants.sum() >= 0.099 * w.capacity.sum()) and w.roots.sum() > 0.03 * w.root_capacity.sum()
    for _ in range(STEPS_A_YEAR):               # grazing stops at midsummer: through winter to the next midsummer
        w.step()
    assert w.plants.sum() > 0.9 * w.capacity.sum()
    assert abs(w.error()) < 1e-9


def test_left_alone_both_pools_fill_up(world):
    for _ in range(2 * STEPS_A_YEAR + STEPS_A_YEAR // 2):     # to midsummer of year 3
        world.step()
    assert world.plants.sum() > 0.9 * world.capacity.sum()
    assert world.roots.sum() > 0.8 * world.root_capacity.sum()


def test_die_back_stands_as_dead_grass_that_weathers_away(world):
    """(3h) Over a winter the die-back piles up as standing dead grass, and some of it weathers away; books exact."""
    for _ in range(int(60 / CFG.dt_days)):     # the first two months of the year: midwinter
        world.step()
    assert world.dead.sum() > 0.05 * world.plants.sum()
    assert world.ledger.dieback > world.ledger.start * 0.002          # root turnover and weathering are booked
    stop = world.dead.copy()
    w2 = World.make(0, CFG)
    w2.plants[:] = 0.0                         # no shoots: nothing new dies back, the dead only weathers
    w2.roots[:] = 0.0
    w2.dead = stop.copy()
    w2.ledger.start = w2.total_plants()
    for _ in range(int(100 / CFG.dt_days)):
        w2.step()
    left = w2.dead.sum() / stop.sum()
    assert 0.6 < left < 0.75                   # roughly a third gone over a 100-day winter
    assert abs(world.error()) < 1e-9 and abs(w2.error()) < 1e-9


def test_grazers_take_green_first_then_the_dead_grass(world):
    cell = int(np.flatnonzero(world.biome.reshape(-1) == GRASS)[0])
    flat = world.plants.reshape(-1)
    reach = flat[cell] - CFG.stubble * world.capacity.reshape(-1)[cell]
    world.dead.reshape(-1)[cell] = 100.0
    world.ledger.start += 100.0
    green, dead = world.graze([cell], [reach + 30.0], split=True)
    assert green[0] == pytest.approx(reach) and dead[0] == pytest.approx(30.0)
    assert world.dead.reshape(-1)[cell] == pytest.approx(70.0)
    assert world.food().reshape(-1)[cell] == pytest.approx(flat[cell] + 70.0)
    assert abs(world.error()) < 1e-9


def test_a_bigger_map_has_more_patches_not_bigger_ones():
    """(4a) Patches are patch_m metres across whatever the map's size: two cells 32 m apart share a biome about as
    often on a 256 m map as on a 512 m one (they would far more often if the patches had grown with the map)."""
    def same_at(n, metres=32):
        k = int(metres / CFG.cell_m)
        same = [(World.make(s, WorldConfig(size=n)).biome[:, :-k] == World.make(s, WorldConfig(size=n)).biome[:, k:]).mean()
                for s in range(3)]
        return np.mean(same)
    small, big = same_at(128), same_at(256)
    assert abs(small - big) < 0.08          # (0.05 apart measured; doubled patches: 0.18)


def test_fruit_trees_drop_their_crop_in_season_and_it_rots():
    """(7a) Nothing falls before the season; over the season each tree drops its crop (booked as sun in) under its
    crown; on the ground it rots, and every kJ is booked."""
    w = World.make(0, CFG)
    assert len(w.trees) > 0
    a, b = CFG.fruit_days
    while w.day < a - 1:
        w.step()
    assert w.fruit.sum() == 0
    fell = 0.0
    while w.day < b:
        fell += w.fruit_fall(w.day, CFG.dt_days) * len(w.trees)
        w.step()
    assert fell == pytest.approx(len(w.trees) * CFG.fruit_kj, rel=0.01)
    cells, _ = w.crown()
    under = np.zeros(w.fruit.size, bool)
    under[cells] = True
    assert w.fruit.sum() > 0 and np.all(w.fruit.reshape(-1)[~under] == 0)   # only under the crowns
    assert w.fruit.sum() < fell                                              # some already rotted
    assert abs(w.error()) < 1e-9


def test_fallen_fruit_is_shared_and_booked():
    w = World.make(0, CFG)
    cells, _ = w.crown()
    c = int(cells[0])
    w.fruit.reshape(-1)[c] = 300.0
    w.ledger.start += 300.0
    got = w.eat_fruit([c, c], [200.0, 400.0])
    assert got.sum() == pytest.approx(300.0) and got[1] == pytest.approx(2 * got[0])
    assert w.fruit.reshape(-1)[c] == pytest.approx(0.0)
    assert abs(w.error()) < 1e-9


def test_rain_grass_grows_where_the_soil_is_wet_and_the_books_balance():
    """(9a) With rain on: a wet cell grows, the same cell bone dry doesn't (and browns); the books still balance."""
    cfg = WorldConfig(size=32, rain=True, storms_ha_day=0.0)          # no storms: we set the wetness by hand
    w = World.make(0, cfg)
    w.day = 180.0                                                      # midsummer
    land = np.argwhere(w.biome == GRASS)
    a, b = map(tuple, land[:2])
    w.plants[a] = w.plants[b] = 0.3 * w.capacity[a]
    w.capacity[b], w.roots[b], w.root_capacity[b] = w.capacity[a], w.roots[a], w.root_capacity[a]
    w.ledger.start = 0.0
    w.ledger.start = w.total_plants()
    w.wet[:] = 0.0
    w.wet[a] = 1.0
    p0 = w.plants[a]
    w.step()
    assert w.plants[a] > p0 > w.plants[b]
    assert w.dead[b] > w.dead[a]
    for _ in range(40):
        w.step()
    assert abs(w.error()) < 1e-9


def test_rain_storms_follow_the_front():
    """(9a) Over a month, the soil is wettest near the rainy front."""
    w = World.make(1, WorldConfig(size=64, rain=True))
    w.wet[:] = 0.0
    for _ in range(4 * 10):
        w.step()
    bx, by = w.belt()
    c = (np.arange(64) + 0.5) * w.cfg.cell_m
    near = np.hypot(c[:, None] - bx, c[None, :] - by) < w.cfg.belt_m
    assert w.wet[near].mean() > 2 * w.wet[~near].mean()
