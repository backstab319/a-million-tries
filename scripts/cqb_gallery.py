"""A gallery of generated kill houses: `uv run --group jax python scripts/cqb_gallery.py [first_seed]`
-> out/cqb/gallery-<seed>.png (3 x 2 buildings, Assault's entries in blue, Defense's start spots in red,
doors amber, center-fed / corner-fed counted under each)."""

import sys
from pathlib import Path

import pygame

from ailearns.games.cqb import CQBConfig, make_building
from ailearns.render import cqb_view as V
from ailearns.render import theme as T


def main(first: int = 0) -> Path:
    T.init()
    cfg = CQBConfig()
    surf = pygame.Surface((T.WIDTH, T.HEIGHT))
    surf.fill(T.BG)
    font = T.font("mono", 18)
    cols, rows = 3, 2
    cw, ch = T.WIDTH // cols, T.HEIGHT // rows
    for k in range(cols * rows):
        seed = first + k
        b = make_building(seed, cfg)
        cell = pygame.Rect((k % cols) * cw + 10, (k // cols) * ch + 10, cw - 20, ch - 50)
        V.draw_building(surf, b, V.View(cell, (cfg.yard_w, cfg.yard_h)), starts=True)
        fed = [f for d in b["doors"] for r, f in zip(d["rooms"], d["corner_fed"]) if r >= 0]
        label = f"seed {seed} · {len(b['rooms'])} rooms · {len(fed) - sum(fed)} center / {sum(fed)} corner-fed"
        surf.blit(font.render(label, True, T.MUTED), (cell.x, cell.bottom + 8))
    out = Path("out/cqb") / f"gallery-{first}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, out)
    return out


if __name__ == "__main__":
    print(main(int(sys.argv[1]) if len(sys.argv) > 1 else 0))
