# Snake, and an AI that learns it, from zero

One file, `snake.py`, about 300 lines of plain Python. You build the game, then an AI that learns to play it
with a table of scores (Q-learning). Nothing to install except pygame.

## Run it

1. Install Python 3 (python.org), then in a terminal: `pip install pygame`
2. `python snake.py`

**Nothing to install:** open `snake.ipynb` in Google Colab (link in the video description) and run the cells
top to bottom. Same code, cut into pieces; the games show as animations, and `play()` is left out (no keyboard there).

At the bottom of the file you pick what runs: put a `#` in front of the lines you don't want.

- `play()`: play it yourself with the arrow keys.
- `test_bot(...)`: plays 1,000 games with no window and prints the average score.
- `train(...)` and `watch(...)`: the AI learns, and you watch it before and after.

## What's in the file, in order

| Part | What it does |
|---|---|
| `new_game`, `print_board` | the board: the snake is a list of squares, head first |
| `step` | the whole game in one move: walls, tail, apples, hunger. Returns the reward (+1 apple, -1 dying) |
| `draw`, `play` | the window, kept apart from the game so the AI can play without one |
| `random_bot`, `apple_chaser`, `test_bot` | two bots written by hand, and a fair way to measure any bot |
| `observe`, `turn` | what the AI sees (six yes/no answers) and what it can do (three turns) |
| `table`, `pick_turn`, `train` | the learning: a score per turn for everything it has seen, nudged after every move |
| `watch` | the AI playing in the window |

## What to expect

On our laptop: random bot ~0.15, apple chaser ~6, the AI after 1,000 games (about 3 seconds of training) ~16-25.
The averages printed while it trains are lower, because there it still tries random turns 1 in 10 moves.

## Common errors

- `ModuleNotFoundError: No module named 'pygame'`: run `pip install pygame` (on some computers `pip3` or
  `python -m pip install pygame`).
- The window opens and closes at once: everything after `play()` has run; check which lines at the bottom have a `#`.

## Learn more

Checked 2026-10-23.

- **New to Python:** the official Python tutorial, start here if a line in the video lost you:
  https://docs.python.org/3/tutorial/
- **Q-learning, step by step:** Hugging Face's free Deep RL Course, unit 2 (Q-learning, with code you run):
  https://huggingface.co/learn/deep-rl-course/unit2/introduction
- **The textbook:** Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd edition, free PDF); section 6.5
  is Q-learning: http://incompleteideas.net/book/the-book-2nd.html
- **The original paper (optional, maths):** Watkins & Dayan, "Q-learning" (1992): https://doi.org/10.1007/BF00992698
- **What comes after a table (optional):** Mnih et al., "Human-level control through deep reinforcement learning"
  (*Nature*, 2015), a neural network instead of the table, playing Atari: https://www.nature.com/articles/nature14236
