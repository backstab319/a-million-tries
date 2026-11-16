"""Who can see whom: all pairs of creatures closer than a radius, fast, in plain numpy.

Creatures are binned into square buckets as wide as the radius, so any pair closer than the radius sits in the
same or a neighbouring bucket. Each unordered pair is looked at once: a creature's own bucket (pairs with a later
creature in the sort) and 4 of its 8 neighbours (the other 4 see it from their side). For each we list every
candidate pair at once (sort by bucket, then expand the bucket ranges), keep the ones truly within reach (squared
distances, no square roots until the end), and return both directions.
"""

import numpy as np


def pairs_within(pos, radius, size):
    """All ordered pairs (a, b), a != b, with |pos[b] - pos[a]| < radius, on a size x size map.

    Returns (a, b, d): index arrays and the distances.
    """
    n = len(pos)
    if n < 2:
        e = np.zeros(0, int)
        return e, e, np.zeros(0)
    nb = max(int(np.ceil(size / radius)), 1)
    bxy = np.clip((pos // radius).astype(int), 0, nb - 1)
    bucket = bxy[:, 0] * nb + bxy[:, 1]
    order = np.argsort(bucket, kind="stable")
    sorted_b = bucket[order]
    a_all, b_all = [], []
    for dx, dy in ((0, 0), (0, 1), (1, -1), (1, 0), (1, 1)):
        ox, oy = bxy[:, 0] + dx, bxy[:, 1] + dy
        ok = (ox >= 0) & (ox < nb) & (oy >= 0) & (oy < nb)
        src = np.flatnonzero(ok)
        nbk = ox[src] * nb + oy[src]
        start = np.searchsorted(sorted_b, nbk, "left")
        end = np.searchsorted(sorted_b, nbk, "right")
        length = end - start
        total = int(length.sum())
        if not total:
            continue
        a = np.repeat(src, length)
        first = np.repeat(np.cumsum(length) - length, length)
        b = order[np.arange(total) - first + np.repeat(start, length)]
        if dx == 0 and dy == 0:
            keep = a < b                       # same bucket: each pair once
            a, b = a[keep], b[keep]
        a_all.append(a)
        b_all.append(b)
    if not a_all:
        e = np.zeros(0, int)
        return e, e, np.zeros(0)
    a, b = np.concatenate(a_all), np.concatenate(b_all)
    x, y = pos[:, 0], pos[:, 1]
    d2 = (x[b] - x[a]) ** 2 + (y[b] - y[a]) ** 2
    keep = d2 < radius * radius
    a, b, d = a[keep], b[keep], np.sqrt(d2[keep])
    return np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([d, d])


try:   # (ep17 speed) the same search, compiled: no pair lists, no sorts; falls back to numpy without numba
    import numba as _nb
except ImportError:  # pragma: no cover
    _nb = None


def _nearest_py(pos, heading, radius, size, edges):
    raise RuntimeError("numba missing")


if _nb is not None:
    @_nb.njit(cache=True)
    def _nearest_nb(pos, heading, radius, size, edges):
        n = pos.shape[0]
        nb = max(int(np.ceil(size / radius)), 1)
        bx = np.empty(n, np.int64)
        by = np.empty(n, np.int64)
        count = np.zeros(nb * nb + 1, np.int64)
        for k in range(n):
            x = min(max(int(pos[k, 0] // radius), 0), nb - 1)
            y = min(max(int(pos[k, 1] // radius), 0), nb - 1)
            bx[k], by[k] = x, y
            count[x * nb + y + 1] += 1
        for c in range(nb * nb):
            count[c + 1] += count[c]
        fill = count[:-1].copy()
        member = np.empty(n, np.int64)
        for k in range(n):   # in index order, so each bucket lists its members by index
            c = bx[k] * nb + by[k]
            member[fill[c]] = k
            fill[c] += 1
        near = np.full(n, -1, np.int64)
        near_d = np.full(n, np.inf)
        sec_b = np.full((n, 3), -1, np.int64)
        sec_d = np.full((n, 3), np.inf)
        r2 = radius * radius
        two_pi = 2 * np.pi
        for a in range(n):
            xa, ya = pos[a, 0], pos[a, 1]
            for ox in range(bx[a] - 1, bx[a] + 2):
                if ox < 0 or ox >= nb:
                    continue
                for oy in range(by[a] - 1, by[a] + 2):
                    if oy < 0 or oy >= nb:
                        continue
                    c = ox * nb + oy
                    for m in range(count[c], count[c + 1]):
                        b = member[m]
                        if b == a:
                            continue
                        dx, dy = pos[b, 0] - xa, pos[b, 1] - ya
                        d2 = dx * dx + dy * dy
                        if d2 >= r2:
                            continue
                        d = np.sqrt(d2)
                        if d < near_d[a] or (d == near_d[a] and b < near[a]):
                            near[a], near_d[a] = b, d
                        rel = np.degrees((np.arctan2(dy, dx) - heading[a] + np.pi) % two_pi - np.pi)
                        s = -1
                        for e in range(3):
                            if edges[e] <= rel < edges[e + 1]:
                                s = e
                        if s >= 0 and (d < sec_d[a, s] or (d == sec_d[a, s] and b < sec_b[a, s])):
                            sec_b[a, s], sec_d[a, s] = b, d
        return near, near_d, sec_b, sec_d
else:  # pragma: no cover
    _nearest_nb = _nearest_py


def nearest_by_sector(pos, heading, radius, size, edges):
    """(ep17 speed) For each creature: the nearest other within `radius` in any direction, and the nearest in each
    of 3 sectors between `edges` (4 angles in degrees, against its heading). Ties go to the lower index.
    -> near, near_d, sec_b (n x 3), sec_d (n x 3); -1 / inf where nobody. Without numba: from `pairs_within`."""
    pos = np.ascontiguousarray(pos, float)
    heading = np.ascontiguousarray(heading, float)
    edges = np.asarray(edges, float)
    if _nb is not None:
        return _nearest_nb(pos, heading, float(radius), float(size), edges)
    return _nearest_np(pos, heading, radius, size, edges)


def _nearest_np(pos, heading, radius, size, edges):
    """`nearest_by_sector` in plain numpy (the fallback, and the check on the compiled one)."""
    n = len(pos)
    near, near_d = np.full(n, -1), np.full(n, np.inf)
    sec_b, sec_d = np.full((n, 3), -1), np.full((n, 3), np.inf)
    a, b, d = pairs_within(pos, radius, size)
    if len(a):
        o = np.lexsort((b, d, a))
        f = np.ones(len(o), bool)
        f[1:] = a[o][1:] != a[o][:-1]
        near[a[o][f]], near_d[a[o][f]] = b[o][f], d[o][f]
        rel = np.arctan2(pos[b, 1] - pos[a, 1], pos[b, 0] - pos[a, 0])
        rel = np.degrees((rel - heading[a] + np.pi) % (2 * np.pi) - np.pi)
        s = np.searchsorted(edges, rel, "right") - 1
        ok = (s >= 0) & (s <= 2)
        a, b, d, s = a[ok], b[ok], d[ok], s[ok]
        o = np.lexsort((b, d, s, a))
        a, b, d, s = a[o], b[o], d[o], s[o]
        f = np.ones(len(a), bool)
        f[1:] = (a[1:] != a[:-1]) | (s[1:] != s[:-1])
        sec_b[a[f], s[f]], sec_d[a[f], s[f]] = b[f], d[f]
    return near, near_d, sec_b, sec_d
