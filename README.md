# A Million Tries

The games and the AIs that learn to play them, from the YouTube channel
[A Million Tries](https://www.youtube.com/@amilliontries). The games and the
reinforcement learning (PPO) are built from scratch.

**Play against the AIs in your browser:**
[race car](https://huggingface.co/spaces/amilliontries/beat-my-ai) ·
[Snake](https://huggingface.co/spaces/amilliontries/beat-my-ai-snake).
**Trained models:** [huggingface.co/amilliontries](https://huggingface.co/amilliontries).

## What's in here

| | |
|---|---|
| `src/ailearns/games/` | Snake (NumPy, PyTorch and JAX versions that agree move for move) and the race car (JAX: random tracks, grip-limited physics, laser sensors) |
| `src/ailearns/agents/` | PPO from scratch: PyTorch, JAX across several GPUs, and the car's continuous steering/pedal version |
| `src/ailearns/play.py`, `drive.py` | play Snake or drive the race car yourself |
| `web/` | the browser games: JavaScript ports of the games and the trained AIs, checked against Python (`node web/*/check.mjs`) |
| `tests/` | the games and trainers, checked against each other |

## Run it

```
uv sync --group jax
uv run games play                                        # Snake, you play
uv run games drive                                       # the race car, you drive
uv run games train runs/snake-1 --arch cnn-deep --view ego2 --steps 20e6
uv run games eval runs/snake-1                           # how often it fills the board
uv run games train-car runs/car-1 --steps 300e6          # a GPU helps: ~4 min on 2x T4
uv run games eval-car runs/car-1                         # 48 tracks it has never seen
uv run pytest
```

The episode-3 Snake AI (92% of games filled) used `--arch cnn-deep --view ego2 --backend jax --steps 100e6`;
the episode-5 driver used `train-car --messy-starts --ray-range 120`. Both are on Hugging Face.

## Licence

MIT (code). Fonts: Inter and JetBrains Mono, under their own licences in `assets/fonts/`.
