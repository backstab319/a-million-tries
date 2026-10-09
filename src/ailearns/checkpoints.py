"""Finding a run's saved checkpoints (runs/<name>/checkpoints/uNNNNN.pt)."""

import re
from pathlib import Path


def checkpoints(run: str | Path) -> list[Path]:
    return sorted((Path(run) / "checkpoints").glob("u*.pt"))


def pick_checkpoint(run: str | Path, which: str) -> Path:
    """'first', 'last', an update number, or a path."""
    all_ = checkpoints(run)
    if which == "first":
        return all_[0]
    if which == "last":
        return all_[-1]
    if re.fullmatch(r"\d+", which):
        return min(all_, key=lambda p: abs(int(p.stem[1:]) - int(which)))
    return Path(which)
