"""Live training metrics on Weights & Biases, so a cloud run can be watched while it trains.

Optional: does nothing unless WANDB_API_KEY is set (environment, or the repo's
.env) and the `wandb` package is installed (`uv sync --group track`). Every run
directory gets a fixed W&B run id (wandb_id.txt), so a resumed run continues
the same W&B run. Nothing here may stop training: any W&B failure just means
no live metrics.
"""

import os
import tempfile
import uuid
from pathlib import Path

PROJECT = "ai-learns"


def _key() -> str | None:
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    env = Path(__file__).resolve().parents[3] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("WANDB_API_KEY=") and line.split("=", 1)[1].strip():
                os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip()
                return os.environ["WANDB_API_KEY"]
    return None


def start(run_dir: Path, config: dict):
    """A W&B run for this run directory, or None (no key, no wandb, or W&B unreachable)."""
    if not _key():
        return None
    try:
        import wandb
    except ImportError:
        print("W&B: key found but `wandb` isn't installed; no live metrics", flush=True)
        return None
    try:
        id_file = run_dir / "wandb_id.txt"
        if not id_file.exists():
            id_file.write_text(uuid.uuid4().hex[:12])
        run = wandb.init(project=PROJECT, name=run_dir.name, id=id_file.read_text().strip(), resume="allow",
                         config=config, dir=tempfile.gettempdir(), settings=wandb.Settings(init_timeout=60))
        print(f"W&B: live metrics at {run.url}", flush=True)
        return run
    except Exception as e:  # never let tracking stop training
        print(f"W&B: couldn't start ({type(e).__name__}); training without live metrics", flush=True)
        return None


def log(run, stats: dict) -> None:
    if run is None:
        return
    try:
        run.log({k: v for k, v in stats.items() if v is not None}, step=stats["steps"])
    except Exception:
        pass


def finish(run) -> None:
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass
