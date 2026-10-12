"""CQB footage: the fog on screen hides exactly what the side whose eyes we're showing can't see."""

import numpy as np
import pygame

from ailearns import record_cqb as R
from ailearns.games import cqb
from ailearns.render import cqb_view as V
from ailearns.render import theme as T


def _frame(tr, i, eyes, view, cast):
    surf = pygame.Surface((960, 540))
    surf.fill(T.BG)
    R.draw_round(surf, view, tr, i, eyes=eyes, cast=cast)
    return pygame.surfarray.array3d(surf)


def _has(px, color):
    return bool((px == np.array(color)).all(-1).any())


def test_fog_on_screen_is_the_fog_it_plays_with():
    T.init()
    tr = R.rollout("walker", "camper", 100_064, seed=1)       # a round where Assault does find Defense
    b, h = tr["building"], tr["hist"]
    lo, size = b["lo"] - 3.5, (b["hi"] + 3.5) - (b["lo"] - 3.5)
    view = V.View(pygame.Rect(0, 0, 960, 540), tuple(size), tuple(lo))
    cast = R._light(tr)
    hidden = [i for i in range(tr["steps"] + 1) if not (h.seen[i][cqb.ASSAULT, cqb.DEFENSE] or h.flash_seen[i][cqb.ASSAULT, cqb.DEFENSE])]
    shown = [i for i in range(tr["steps"] + 1) if h.seen[i][cqb.ASSAULT, cqb.DEFENSE] and h.alive[i][cqb.DEFENSE]]
    assert hidden and shown
    for i in hidden[:: max(1, len(hidden) // 12)]:
        assert not _has(_frame(tr, i, "assault", view, cast), V.DEFENSE), i     # not a single red pixel
    for i in shown[:3]:
        assert _has(_frame(tr, i, "assault", view, cast), V.DEFENSE), i         # (and it is drawn when seen)
    assert _has(_frame(tr, hidden[0], "all", view, cast), V.DEFENSE)            # the reveal shows everything


def test_team_fog_shows_what_any_of_the_pair_sees():
    """2v1: the defender is drawn exactly while one of the pair sees it (or its muzzle flash); never otherwise."""
    from dataclasses import replace
    T.init()
    cfg = replace(cqb.CQBConfig(), team_sizes=(2, 1))
    tr = R.rollout("walker", "camper", 100_084, seed=1, cfg=cfg)   # number two sees it alone at times
    b, h = tr["building"], tr["hist"]
    lo, size = b["lo"] - 3.5, (b["hi"] + 3.5) - (b["lo"] - 3.5)
    view = V.View(pygame.Rect(0, 0, 960, 540), tuple(size), tuple(lo))
    cast = R._light(tr)
    sees = lambda i: any(h.seen[i][m, 2] or h.flash_seen[i][m, 2] for m in (0, 1))  # noqa: E731
    hidden = [i for i in range(tr["steps"] + 1) if not sees(i)]
    shown = [i for i in range(tr["steps"] + 1) if sees(i) and h.alive[i][2] and not h.flash[i][2]]  # (a muzzle flash
    assert hidden and shown                                                                          # covers the disc)
    for i in hidden[:: max(1, len(hidden) // 12)]:
        assert not _has(_frame(tr, i, "assault", view, cast), V.DEFENSE), i
    for i in shown[:3]:
        assert _has(_frame(tr, i, "assault", view, cast), V.DEFENSE), i
    only_two = [i for i in shown if not (h.seen[i][0, 2] or h.flash_seen[i][0, 2])]
    assert only_two
    for i in only_two:
        assert _has(_frame(tr, i, "assault", view, cast), V.DEFENSE), i
