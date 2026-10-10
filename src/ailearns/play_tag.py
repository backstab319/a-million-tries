"""Play tag yourself (arrow keys / WASD) against the AI: as the robot (you chase) or the blob
(you run). Episode 8, "me vs the tag AI".

    uv run --group jax ailearns tag                  # practice: you're the robot, vs blob gen 4
    uv run --group jax ailearns tag --as blob        # practice: you're the blob, vs robot gen 4
    uv run --group jax ailearns tag --vs runs/tag-b2:190 --arena 100010
    uv run --group jax ailearns tag --match          # the scored ladder (as the robot; --as blob too)

Same game code and settings the AI plays with (games/tag.py), same 10 decisions per
second, and you see what your character sees: the fog hides everything behind obstacles,
and the other player only shows while it's in sight (a dashed ring marks where you last
saw it). The window fits your screen and can be resized.

Practice uses test arenas 0-63 (100000-100063), which the AIs never trained in. The scored
ladder (--match) plays every generation of the other side, oldest first, 10 rounds each,
in the arenas and from the starting spots episode 7's table was measured on (test arenas
64-255, starts from PRNGKey(5000)): every scored round has an AI twin. Scored rounds can't
be retried, and quitting in the middle of one counts as a loss. It carries on where you
stopped last time.

Every round is saved to episodes/08-me-vs-the-tag-ai/data/human-tag/: rounds.jsonl (result,
time, your try number) and the whole round (.npz), for replays in the episode. It's in git,
so commit after playing.
"""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from ailearns.games.tag import CHASER, RUNNER, TEST_SEEDS, TagGame, make_arena, make_arenas
from ailearns.agents.ppo_tag import game_config, load_player
from ailearns.render import theme as T
from ailearns.render.tag_view import CHASER_COLOR, RUNNER_COLOR, ArenaView, draw_clock

ROOT = Path(__file__).resolve().parents[2]
EPISODE = ROOT / "episodes" / "08-me-vs-the-tag-ai"  # where episode 8's rounds live; elsewhere: out/human-tag
OUT = EPISODE / "data" / "human-tag" if EPISODE.exists() else ROOT / "out" / "human-tag"
W, H = 1600, 900
OPPONENT = {"robot": "runs/tag-b4:190", "blob": "runs/tag-r4:190"}  # practice: the last generation of the other side
# the ladder: every generation of the other side, oldest first (the checkpoints in episode 7's table)
LADDER = {"robot": [("blob gen 1", "runs/tag-b1-vs-v4:150"), ("blob gen 2", "runs/tag-b2:190"),
                    ("blob gen 3", "runs/tag-b3:175"), ("blob gen 4", "runs/tag-b4:190")],
          "blob": [("episode 6 robot", "runs/tag-v4-memory:175"), ("robot gen 2", "runs/tag-r2:190"),
                   ("robot gen 3", "runs/tag-r3:190"), ("robot gen 4", "runs/tag-r4:190")]}
PER_RUNG = 10
PRACTICE, SCORED = 64, 192  # test arenas 0-63 to practise on; 64-255 are the table's (and the ladder's)
FIRST_INDEX = {"robot": 0, "blob": PER_RUNG * 4}  # robot rungs use scored rounds 0-39, blob rungs 40-79
TABLE_KEYS = jax.random.split(jax.random.PRNGKey(5000), SCORED)  # the table's starts (agents.ppo_tag.match)


def _input(keys) -> np.ndarray:
    x = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])
    y = (keys[pygame.K_UP] or keys[pygame.K_w]) - (keys[pygame.K_DOWN] or keys[pygame.K_s])
    return np.array([x, y], np.float32)  # world y is up; the game caps the length at 1 (diagonals too)


def ladder(side: str, log: list[dict]) -> list[dict]:
    """Every scored round of `side`'s ladder in order, each with its result if it's been played."""
    done = {(r["vs"], r["index"]): r for r in log if r.get("scored") and r["as"] == side}
    out = []
    for k, (label, spec) in enumerate(LADDER[side]):
        for j in range(PER_RUNG):
            i = FIRST_INDEX[side] + k * PER_RUNG + j
            out.append({"rung": k, "label": label, "vs": spec, "index": i, "arena": TEST_SEEDS[PRACTICE + i],
                        "result": done.get((spec, i))})
    return out


def run(side: str = "robot", vs: str | None = None, arena: int = 100_000, fog: bool = True,
        match: bool = False) -> None:
    me = CHASER if side == "robot" else RUNNER
    other = 1 - me
    win_word = "caught" if side == "robot" else "escaped"
    log_file = OUT / "rounds.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    log = [json.loads(l) for l in log_file.read_text().splitlines()] if log_file.exists() else []
    if not match and not TEST_SEEDS[0] <= arena < TEST_SEEDS[PRACTICE]:
        raise SystemExit(f"practice arenas are {TEST_SEEDS[0]}-{TEST_SEEDS[PRACTICE - 1]} "
                         "(the rest are for the scored ladder)")

    T.init(headless=False)
    window = T.Window((W, H), "ai-learns: tag")
    screen = window.canvas
    clock = pygame.time.Clock()
    side_px = H - 40
    rect = pygame.Rect(W - side_px - 20, 20, side_px, side_px)
    colors = {CHASER: CHASER_COLOR, RUNNER: RUNNER_COLOR}
    cfg = game_config(load_player(OPPONENT[side], other)[1])  # every generation was trained with the same settings
    game = TagGame(make_arenas(list(TEST_SEEDS), cfg), cfg)
    steps = {}

    def stepper(spec: str):
        if spec not in steps:
            ai = load_player(spec, other)[0]

            @jax.jit
            def step(st, mine, key):
                acts = [None, None]
                acts[me], acts[other] = mine, ai(game, st, key)
                return game.step(st, jnp.stack(acts))[0]
            steps[spec] = step
        return steps[spec]

    def next_round():
        """-> the round to play (dict), or None when the ladder is done."""
        if match:
            todo = [r for r in ladder(side, log) if r["result"] is None]
            if not todo:
                return None
            rnd = dict(todo[0])
            key = TABLE_KEYS[rnd["index"]]
        else:
            seed = TEST_SEEDS[0] + (arena - TEST_SEEDS[0] + practice_count) % PRACTICE
            rnd = {"vs": vs or OPPONENT[side], "arena": seed, "label": vs or OPPONENT[side]}
            key = jax.random.fold_in(jax.random.PRNGKey(0), seed)
        st = game.init(key, rnd["arena"] - TEST_SEEDS[0])
        stepper(rnd["vs"])(st, jnp.zeros(2), key)  # compile before the countdown, not during it
        rnd["view"] = ArenaView(make_arena(rnd["arena"], cfg), cfg.size, rect)
        return rnd, st

    def save(rnd: dict, hist: list, quit: bool = False) -> None:
        st = hist[-1]
        caught = bool(st.caught)
        rec = {"as": side, "vs": rnd["vs"], "arena": rnd["arena"], "caught": caught,
               "won": caught if side == "robot" else not caught and not quit,
               "seconds": round(int(st.t) * cfg.dt, 1), "when": time.strftime("%Y-%m-%d %H:%M"),
               "try": len(log) + 1, "round": f"{side}-{rnd['arena']}-{int(time.time())}.npz"}
        if match:
            rec |= {"scored": True, "rung": rnd["label"], "index": rnd["index"]}
        if quit:
            rec["quit"] = True  # left in the middle of a scored round: counts as lost
        h = jax.device_get(jax.tree.map(lambda *xs: jnp.stack(xs), *hist))
        np.savez(OUT / rec["round"], pos=h.pos, vel=h.vel, mem=h.mem, mem_vel=h.mem_vel, seen=h.seen,
                 caught=caught, arena=rnd["arena"])
        log.append(rec)
        with log_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    practice_count = 0
    nxt = next_round()
    rnd, st = nxt if nxt else (None, None)
    prev, hist, acc, countdown, ended, confirm = None, [st], 0.0, 3.0, None, False
    while True:
        dt = clock.tick(60) / 1000
        playing = rnd is not None and countdown <= 0 and ended is None
        for e in pygame.event.get():
            esc = e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE
            if e.type == pygame.QUIT or (esc and (confirm or not (match and playing))):
                if match and playing:
                    save(rnd, hist, quit=True)
                pygame.quit()
                return
            if esc:
                confirm = True  # a scored round is running: quitting it counts as a loss, so ask first
            elif e.type == pygame.KEYDOWN:
                confirm = False
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_SPACE, pygame.K_r) and ended is not None:
                practice_count += 1
                nxt = next_round()
                if nxt:
                    rnd, st = nxt
                    prev, hist, acc, countdown, ended = None, [st], 0.0, 3.0, None
                else:
                    rnd = None
        if rnd is not None and countdown > 0:
            countdown -= dt
        elif playing:
            acc += dt
            step = stepper(rnd["vs"])
            while acc >= cfg.dt and ended is None:  # 10 decisions per second, like the AI's
                acc -= cfg.dt
                prev = st
                st = step(st, jnp.asarray(_input(pygame.key.get_pressed())), jax.random.PRNGKey(int(st.t)))
                hist.append(st)
                if bool(st.done):
                    ended, confirm = time.time(), False
                    save(rnd, hist)

        screen.fill(T.BG)
        x = 40
        T.text(screen, f"You: the {side}", (x, 30), "display", 56, colors[me])
        if rnd is None:
            _ladder_done(screen, x, side, log, win_word)
            window.show()
            continue
        view = rnd["view"]
        # draw, interpolating between the last two decisions for smooth motion
        a = 1.0 if prev is None or ended is not None else min(1.0, acc / cfg.dt)
        p0 = np.asarray(prev.pos if prev is not None else st.pos)
        pos = p0 + (np.asarray(st.pos) - p0) * a
        vel = np.asarray(st.vel)
        seen, over = bool(st.seen), ended is not None
        t = int(st.t) * cfg.dt
        view.draw_arena(screen)
        if fog and not over:
            view.draw_shadows(screen, pos[me])
        if fog and not seen and not over:
            view.draw_last_seen(screen, np.asarray(st.mem[me]), cfg.radius, colors[other],
                                np.asarray(st.mem_vel[me]) if cfg.remember_heading else None)
        show_other = seen or over or not fog
        if show_other or me == RUNNER:
            view.draw_runner(screen, pos[RUNNER], None if over else vel[RUNNER],
                             threat=pos[CHASER] if seen else None, radius=cfg.radius, t=t,
                             scared=seen and float(np.linalg.norm(pos[CHASER] - pos[RUNNER])) < 6)
        if show_other or me == CHASER:
            look = (pos[RUNNER] - pos[CHASER]) if seen else (np.asarray(st.mem[CHASER]) - pos[CHASER])
            view.draw_chaser(screen, pos[CHASER], None if over else vel[CHASER], look=look,
                             mood="happy" if over and bool(st.caught) else ("hunt" if seen else "lost"),
                             radius=cfg.radius, t=t)
        if over and bool(st.caught):
            view.draw_tag(screen, np.asarray(st.pos[RUNNER]), time.time() - ended)

        n_try = len(log) + (0 if over else 1)
        T.text(screen, f"vs {rnd['label']}", (x, 105), "semibold", 32, T.TEXT)
        if match:
            done = [r for r in ladder(side, log) if r["rung"] == rnd["rung"] and r["result"]]
            k = len(done) + (0 if over else 1)
            T.text(screen, f"SCORED  rung {rnd['rung'] + 1} of 4  ·  round {k} of {PER_RUNG}", (x, 150),
                   "semibold", 26, T.AMBER)
            score = f"{win_word} {sum(r['result']['won'] for r in done)} of {len(done)} so far"
        else:
            T.text(screen, "practice (not scored)", (x, 150), "semibold", 26, T.MUTED)
            mine = [r for r in log if r["as"] == side and r["vs"] == rnd["vs"] and not r.get("scored")]
            score = f"{win_word} {sum(r['won'] for r in mine)} of {len(mine)}"
        draw_clock(screen, pygame.Rect(x, 270, rect.x - 2 * x, 28), t, cfg.time_limit, bool(st.caught), over)
        T.text(screen, score, (x, 340), "semibold", 36, T.GREEN)
        T.text(screen, f"MY TRY #{n_try:,}", (x, 395), "bold", 30, colors[me])
        T.text(screen, f"arena {rnd['arena']}", (x, 440), "regular", 24, T.MUTED)
        T.text(screen, "arrows/WASD   Space next round   Esc quit", (x, H - 50), "regular", 22, T.MUTED)
        if countdown > 0:
            T.text(screen, str(int(countdown) + 1), rect.center, "display", 160, T.TEXT, anchor="center")
        if over:
            won = log[-1]["won"]
            T.text(screen, "YOU WIN" if won else "AI WINS", (x, 520), "display", 80, T.GREEN if won else T.RED)
            T.text(screen, "press Space for the next round", (x, 620), "semibold", 30, T.TEXT)
        if confirm:
            T.text(screen, "Esc again to quit: this round counts as lost", (x, 700), "semibold", 28, T.RED)
        window.show()


def _ladder_done(screen, x: int, side: str, log: list[dict], win_word: str) -> None:
    T.text(screen, "Ladder done!", (x, 140), "display", 64, T.GREEN)
    for k, (label, spec) in enumerate(LADDER[side]):
        rs = [r["result"] for r in ladder(side, log) if r["rung"] == k]
        T.text(screen, f"vs {label}: {win_word} {sum(r['won'] for r in rs)} of {len(rs)}", (x, 250 + 60 * k),
               "semibold", 36, T.TEXT)
    T.text(screen, "Esc to quit", (x, H - 50), "regular", 22, T.MUTED)
