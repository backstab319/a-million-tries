# A Million Tries

The games and the AIs that learn to play them, from the YouTube channel
[A Million Tries](https://www.youtube.com/@amilliontries). The games and the
reinforcement learning (PPO) are built from scratch.

**Play against the AIs in your browser:**
[race car](https://huggingface.co/spaces/amilliontries/beat-my-ai) ·
[Snake](https://huggingface.co/spaces/amilliontries/beat-my-ai-snake) ·
[tag](https://huggingface.co/spaces/amilliontries/beat-my-ai-tag) ·
[the kill house](https://huggingface.co/spaces/amilliontries/beat-my-ai-cqb).
**Trained models:** [huggingface.co/amilliontries](https://huggingface.co/amilliontries).

## What's in here

| | |
|---|---|
| `src/ailearns/games/` | Snake (NumPy, PyTorch and JAX versions that agree move for move), the race car (JAX: random tracks, grip-limited physics, laser sensors), tag (JAX: random arenas, line of sight, a chaser and a runner) and CQB, "the kill house" (JAX: generated houses, a weapon light in the dark, hearing through walls; optional flashbangs, magazines, health, reaction time) with its scripted players |
| `src/ailearns/agents/` | PPO from scratch: PyTorch, JAX across several GPUs, the car's continuous steering/pedal version, tag's (either side learns; the opponent can be a script, a frozen brain or a pool of them: self-play) and CQB's, plus CQB's evolution strategies (`es_cqb.py`) and imitation (`bc_cqb.py`) |
| `src/ailearns/record_cqb.py` | CQB rounds drawn top-down as in the episodes (`games watch-cqb`; needs ffmpeg) |
| `src/ailearns/play.py`, `drive.py`, `play_tag.py` | play Snake, drive the race car, or play tag against a trained AI yourself |
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
uv run games train-tag runs/tag-1 --closer-reward 0.05 --remember-heading   # a robot that chases a script
uv run games train-tag runs/blob-1 --side runner --opponent runs/tag-1:last --closer-reward 0.05 --remember-heading
uv run games tag-table --chasers runs/tag-1 --runners flee runs/blob-1      # who catches whom, unseen arenas
uv run games tag --vs runs/blob-1                        # you chase it (--as blob: you run)
uv run games train-cqb runs/cqb-1 --opponent dummy --time-cost 0.05 --ent-coef 0.003   # CQB: find the target first
uv run games train-cqb runs/cqb-2 --init-from runs/cqb-1:last --opponent camper        # then one that shoots back
uv run games eval-cqb runs/cqb-2 --first 64 --n 192      # test houses it never trained in
uv run games watch-cqb runs/cqb-2:last camper 100064     # one round as a video
uv run pytest
```

The episode-3 Snake AI (92% of games filled) used `--arch cnn-deep --view ego2 --backend jax --steps 100e6`;
the episode-5 driver used `train-car --messy-starts --ray-range 120`. The tag robots and blobs (episodes 6-8)
took turns, each new generation training against a pool of the other side's past ones
(`--opponent runs/a:175,runs/b:190`). All of them are on Hugging Face.

## Snapshots: the code as each video used it

Snake, the car and tag were finished before they were shared. CQB keeps changing between episodes, so each
episode that uses it gets a tag here: check it out to get the game, the rules and the trainers exactly as that
video used them (the model pages on Hugging Face name the tag their brains run with).

| episode | tag | the game then |
|---|---|---|
| 9, "The Kill House" (1 vs 1) | [`episode-9`](https://github.com/backstab319/a-million-tries/tree/episode-9) | one attacker vs one defender; flashbangs, magazines, health and reaction time as options |

```
git checkout episode-9
uv run games watch-cqb runs/<brain>:<update> camper 100064 --batch 64,64   # a round exactly as in the video
```

## Licence

MIT (code). Fonts: Inter and JetBrains Mono, under their own licences in `assets/fonts/`.
