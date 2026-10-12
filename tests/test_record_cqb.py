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
    hidden = [i for i in range(tr["steps"] + 1) if not (h.seen[i][cqb.ASSAULT] or h.flash_seen[i][cqb.ASSAULT])]
    shown = [i for i in range(tr["steps"] + 1) if h.seen[i][cqb.ASSAULT] and h.alive[i][cqb.DEFENSE]]
    assert hidden and shown
    for i in hidden[:: max(1, len(hidden) // 12)]:
        assert not _has(_frame(tr, i, "assault", view, cast), V.DEFENSE), i     # not a single red pixel
    for i in shown[:3]:
        assert _has(_frame(tr, i, "assault", view, cast), V.DEFENSE), i         # (and it is drawn when seen)
    assert _has(_frame(tr, hidden[0], "all", view, cast), V.DEFENSE)            # the reveal shows everything
