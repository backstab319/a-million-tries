"""The neighbour search finds exactly the pairs a brute-force check finds."""

import numpy as np

from ailearns.games.ecosystem_space import pairs_within


def _brute(pos, r):
    d = np.linalg.norm(pos[None] - pos[:, None], axis=2)
    a, b = np.nonzero((d < r) & ~np.eye(len(pos), dtype=bool))
    return set(zip(a.tolist(), b.tolist()))


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    for n, r in [(0, 6.0), (1, 6.0), (2, 300.0), (500, 6.0), (800, 13.0)]:
        pos = rng.random((n, 2)) * 256
        a, b, d = pairs_within(pos, r, 256.0)
        assert set(zip(a.tolist(), b.tolist())) == _brute(pos, r)
        assert np.allclose(d, np.linalg.norm(pos[b] - pos[a], axis=1))


def test_points_on_the_map_edge_and_in_one_spot():
    pos = np.array([[0.0, 0.0], [0.0, 0.0], [255.99, 255.99], [256.0, 256.0], [250.0, 255.0]])
    a, b, _ = pairs_within(pos, 6.0, 256.0)
    assert set(zip(a.tolist(), b.tolist())) == _brute(pos, 6.0)


def test_compiled_nearest_by_sector_matches_numpy():
    """(ep17 speed) The compiled search gives exactly what the numpy one does, ties included (stacked newborns)."""
    from ailearns.games.ecosystem_space import _nearest_np, nearest_by_sector
    rng = np.random.default_rng(3)
    pos = rng.uniform(0, 200, (3000, 2))
    pos[:300] = pos[300:600]                     # 300 creatures stacked on others: distance 0, ties
    pos[600:650] = 0.0                           # and on the map's edge
    heading = rng.uniform(-np.pi, np.pi, len(pos))
    edges = (-60.0, -20.0, 20.0, 60.0)
    got, want = nearest_by_sector(pos, heading, 8.0, 200.0, edges), _nearest_np(pos, heading, 8.0, 200.0, edges)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)
