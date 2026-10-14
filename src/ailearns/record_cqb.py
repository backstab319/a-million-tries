"""Footage of CQB: rounds played by the real game (games/cqb.py) and real players (scripts now,
brains later); nothing is animated by hand.

`eyes`: whose fog the picture shows. "assault" / "defense": only what that side sees right now
(its light), a faint memory of what it has already looked at (its "seen" map), the other one only
while it's in sight (or lit by its own muzzle flash), else a ghost where it was last seen, and
what it hears as an arc on the side it came from. That's exactly the knowledge it plays with.
"all": everything (the reveal).

Between decisions (10 a second) positions and facings are smoothed for the 60 fps picture; the
light is cast again at each smoothed pose. What each side knows changes only at decisions.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from ailearns.games import cqb
from ailearns.games import cqb_scripts as S
from ailearns.media.video import VideoWriter
from ailearns.render import cqb_view as V
from ailearns.render import theme as T

LABELS = {"walker": "walker (a script)", "camper": "camper (a script)", "wanderer": "wanderer (a script)",
          "dummy": "dummy: stands still", "random": "random moves", "walker_flash": "flash and enter (a script)"}
SIDES = {"assault": cqb.ASSAULT, "defense": cqb.DEFENSE}
NAMES = {cqb.ASSAULT: "Assault", cqb.DEFENSE: "Defense"}
COLORS = {cqb.ASSAULT: V.ASSAULT, cqb.DEFENSE: V.DEFENSE}


def op_name(game: cqb.CQBGame, w: int) -> str:
    """An operator's name on screen: its side ("Assault"), numbered when the side has more than one ("Assault 2")."""
    side = int(game.team[w])
    ops = game.side_ops(side)
    return NAMES[side] + (f" {ops.index(w) + 1}" if len(ops) > 1 else "")


def player(spec: str, who: int, sample: bool = False):
    """A script name, or a brain "runs/x:ckpt" (its best guess; `sample`: it tries moves around it,
    like in training: an untrained brain's best guess just stands still)."""
    if spec in S.SCRIPTS:
        return S.SCRIPTS[spec]
    from ailearns.agents.ppo_cqb import brain, load
    from ailearns.checkpoints import pick_checkpoint
    run, _, ck = spec.partition(":")
    return brain(load(pick_checkpoint(run, ck or "last"))[0], who, sample_moves=sample)


def label(spec: str) -> str:
    if spec in LABELS:
        return LABELS[spec]
    run, _, ck = spec.partition(":")
    return f"AI ({run.rstrip('/').split('/')[-1]}, update {ck or 'last'})"


def game_config(*specs: str) -> cqb.CQBConfig:
    """The rules a round is played by: a brain's own (from its checkpoint; the one with the most rules wins: a brain
    trained with fewer reads the numbers in front), else the default (scripts, and every brain from before them)."""
    from ailearns.agents.ppo_cqb import load
    from ailearns.checkpoints import pick_checkpoint
    cfg = cqb.CQBConfig()
    for spec in specs:
        if spec in S.SCRIPTS:
            continue
        run, _, ck = spec.partition(":")
        saved = load(pick_checkpoint(run, ck or "last"))[1]["cqb"]
        c = cqb.CQBConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()})
        if c.obs_size > cfg.obs_size:   # the most rules win (brains from before read the numbers in front)
            cfg = c
    return cfg


def rollout(assault: str, defense: str, building_seed: int, seed: int = 0, cfg: cqb.CQBConfig | None = None,
            sample: str = "", batch: tuple | None = None) -> dict:
    """One round, every decision recorded (state after each decision; index 0 = the start).
    sample: "assault" / "defense" / "both": those brains try moves around their best guess.
    cfg: the rules (default: game_config of the players).
    batch: (first, n): the round exactly as it was played in a batch over test buildings first..first+n-1 (a round's
    luck comes from its place in the batch: footage picked from a batch, ep9 3ay)."""
    cfg = cfg or game_config(assault, defense)
    if batch:
        seeds = list(cqb.TEST_SEEDS)[batch[0]:batch[0] + batch[1]]
        game, slot = cqb.CQBGame(cqb.make_buildings(seeds, cfg), cfg), seeds.index(building_seed)
    else:
        game, slot = cqb.CQBGame(cqb.make_buildings([building_seed], cfg), cfg), 0
    a = player(assault, cqb.ASSAULT, sample in ("assault", "both"))
    d = player(defense, cqb.DEFENSE, sample in ("defense", "both"))
    r = S.play_rounds(game, a, d, np.array([slot]), seed=seed, record=True)
    h = jax.tree.map(lambda x: np.asarray(x[0]), r["history"])
    done = h.done
    last = int(np.argmax(done)) if done.any() else game.T
    h = jax.tree.map(lambda x: x[:last + 1], h)
    return {"cfg": cfg, "game": game, "building": cqb.make_building(building_seed, cfg), "seed": building_seed,
            "hist": h, "steps": last, "outcome": int(h.outcome[-1]), "assault": assault, "defense": defense}


def _interp(tr: dict, i: float):
    """Positions and facings at (fractional) decision i."""
    h = tr["hist"]
    a = int(np.clip(np.floor(i), 0, tr["steps"]))
    b = min(a + 1, tr["steps"])
    f = float(np.clip(i - a, 0, 1))
    pos = h.pos[a] * (1 - f) + h.pos[b] * f
    d = np.arctan2(np.sin(h.face[b] - h.face[a]), np.cos(h.face[b] - h.face[a]))
    return pos, h.face[a] + d * f, a, b


def _light(tr: dict):
    """A compiled light caster for this round: (pos, face, door_open) -> each fan ray's reach."""
    game = tr["game"]
    st0 = jax.tree.map(lambda x: jnp.asarray(x[0]), tr["hist"])

    @jax.jit
    def cast(p, face, door_open):
        lo, hi = game.boxes(st0._replace(door_open=door_open))
        return game.light(p, face, lo, hi)
    return cast


def _bang_light(tr: dict, rays: int = 72):
    """A compiled caster for a flashbang's light: (where, door_open) -> the lit area's edge points (rays, 2):
    every way round, stopped by walls and shut doors (not cover), out to bang_range."""
    game = tr["game"]
    st0 = jax.tree.map(lambda x: jnp.asarray(x[0]), tr["hist"])
    ang = jnp.arange(rays) * 2 * jnp.pi / rays
    dirs = jnp.stack([jnp.cos(ang), jnp.sin(ang)], -1)

    @jax.jit
    def area(q, door_open):
        lo, hi = game.walls(st0._replace(door_open=door_open))
        return q + dirs * jnp.minimum(cqb._cast(q, dirs, lo, hi, game.cfg), game.cfg.bang_range)[:, None]
    return area


BANG_SHOW = 4   # decisions a bang's light stays on screen (fading)


def _team(tr: dict, eyes: str) -> list:
    """The operators whose knowledge the picture shows (eyes "assault" / "defense"), or [] for "all"."""
    return tr["game"].side_ops(SIDES[eyes]) if eyes in SIDES else []


def _shown(tr: dict, st_i: int, eyes: str, w: int) -> bool:
    """Is operator w on screen for these eyes: everyone for "all", the shown side's own, and enemies it sees now."""
    h = tr["hist"]
    mine = _team(tr, eyes)
    return eyes == "all" or w in mine or any(bool(h.seen[st_i][m, w]) for m in mine)


def draw_round(surf: pygame.Surface, view: V.View, tr: dict, i: float, eyes: str = "assault", cast=None,
               bang=None) -> None:
    """bang: _bang_light(tr) (made here if the game has flashbangs and it isn't given). With teams, a side's picture
    is the union of its operators' lights and "seen" maps, and an enemy shows while any of them sees it."""
    cfg, b, h, game = tr["cfg"], tr["building"], tr["hist"], tr["game"]
    pos, face, a, k = _interp(tr, i)
    st_i = k if i - a > 0.999 else a                       # what's known: the state at the last decision reached
    doors = h.door_open[st_i]
    V.draw_building(surf, b, view, doors_open=doors[:len(b["doors"])])
    angles = np.asarray(game.cone_angles)
    everyone = range(game.N)
    mine = _team(tr, eyes)
    color = lambda w: COLORS[int(game.team[w])]           # noqa: E731
    polys = {}
    pilot = 0 if game.pilot_on else None                   # (ep11) the drone pilot: no weapon light, its own marker
    for w in everyone:
        if h.alive[st_i][w] and w != pilot:
            reach = np.asarray(cast(jnp.asarray(pos[w]), jnp.asarray(face[w]), jnp.asarray(doors)))
            polys[w] = V.light_polygon(view, pos[w], face[w], angles, reach)
    if mine:
        size = (b["hi"] - b["lo"]) / np.array(cfg.seen_grid)
        centres = b["lo"] + cqb.cell_fracs(cfg) * (b["hi"] - b["lo"])
        looked = np.any([h.cleared[st_i][m] for m in mine], axis=0)
        seen_cells = [(c - size / 2, c + size / 2) for c, s in zip(centres, looked) if s]
        V.draw_fog(surf, view, [polys[m] for m in mine if m in polys], remembered=seen_cells)
    for w, poly in polys.items():
        if not mine or w in mine:
            V.draw_light_tint(surf, poly)
    if any(cfg.flashbangs):                                   # bangs under the operators (they stay visible)
        bang = bang or _bang_light(tr)
        for w in everyone:
            for j in range(max(0, int(i) - BANG_SHOW + 1), int(i) + 1):     # went off a moment ago
                if h.bang_now[j][w]:
                    area = np.asarray(bang(jnp.asarray(h.bang_pos[j][w]), jnp.asarray(h.door_open[j])))
                    V.draw_bang(surf, [view.px(q) for q in area], 1 - (i - j) / BANG_SHOW)
    for w in everyone:
        visible = not mine or w in mine
        if mine and w not in mine:
            visible = any(bool(h.seen[st_i][m, w] or h.flash_seen[st_i][m, w]) for m in mine)
            since = min(int(h.since[st_i][m, w]) for m in mine)
            if not visible and since < game.T and not h.known_down[st_i][mine[0], w]:   # a ghost: last seen there
                V.draw_operator(surf, view, h.mem[st_i][mine[0], w], 0.0, color(w), alpha=70)
        if visible and w == pilot:
            V.draw_pilot(surf, view, pos[w], color(w))
        elif visible:
            V.draw_operator(surf, view, pos[w], face[w], color(w), down=not h.alive[st_i][w])
            ops = game.side_ops(int(game.team[w]))
            if len(ops) > 1:                                  # which one of its side: "1", "2"
                c = view.px(pos[w])
                T.text(surf, str(ops.index(w) + 1), (c[0] + max(6, 0.45 * view.k), c[1] - max(6, 0.45 * view.k)),
                       kind="semibold", size=max(14, int(0.5 * view.k)), color=color(w), anchor="midleft")
        if h.flash[st_i][w] > 0 and (visible or eyes == "all"):
            V.draw_flash(surf, view, pos[w], face[w], strength=h.flash[st_i][w] / 2)
    if pilot is not None or any(cfg.radio) or game.feed_ops or game.intel_ops:
        _draw_radio(view, surf, tr, i, st_i, pos, face)
    for m in mine:                                            # what each of the shown side hears
        if not h.alive[st_i][m] or h.seen[st_i][m].any():
            continue
        for kind in range(3):
            loud = h.sound[st_i][m, kind, 2]
            if loud > 0 and h.sound_age[st_i][m, kind] == 0:
                V.draw_heard(surf, view, pos[m], h.sound[st_i][m, kind, :2], float(loud))
    if any(cfg.flashbangs):
        _draw_flashbangs(surf, view, tr, i, st_i, pos, eyes)
    if cfg.health:                                            # health bars (for whoever's shown)
        for w in everyone:
            if h.alive[st_i][w] and _shown(tr, st_i, eyes, w) and w != pilot:   # (the pilot doesn't fight)
                V.draw_health(surf, view, pos[w], float(h.hp[st_i][w]) / cfg.health)
    if cfg.magazine:                                          # who's reloading (if they're shown)
        steps = cfg.reload_time / cfg.dt
        for w in everyone:
            left = h.reload_timer[st_i][w]
            if left > 0 and h.alive[st_i][w] and _shown(tr, st_i, eyes, w):
                V.draw_reloading(surf, view, pos[w], 1 - left / steps)


CALL_TURN = {1: 0.0, 2: np.pi / 2, 3: -np.pi / 2, 4: np.pi}   # protocol word -> turn from the entry operator's facing
CALL_SHOW = 1.0                                                  # seconds a call's bubble stays up


def _draw_radio(view, surf, tr, i, st_i, pos, face) -> None:
    """(ep11 10zo) The pilot game: the drone feed's lock on each standing defender (when the pilot's on the feed), and
    the last call the entry operator heard: its word in a bubble, and (with a protocol) an arrow the way it points,
    as the operator faced when it heard it. Muted: nothing arrives, so no bubble."""
    cfg, h, game = tr["cfg"], tr["hist"], tr["game"]
    fed = [w for w in sorted(set(game.feed_ops) | set(game.intel_ops)) if h.alive[st_i][w]]
    if fed:                                                   # anyone on the (perfect) feed: show the lock
        for j in game.side_ops(cqb.DEFENSE):
            if h.alive[st_i][j]:
                V.draw_feed_ring(surf, view, pos[j], (i * cfg.dt / 1.2) % 1.0)
    if not any(cfg.radio):
        return
    if not game.pilot_on:                                     # a pair's own words: a bubble at whoever said it
        for w in game.side_ops(cqb.ASSAULT):
            d = st_i
            while d > 0 and not (h.said_age[d][w] == 0 and h.sent[d][w] > 0):
                d -= 1
            age = (i - d) * cfg.dt
            if h.alive[st_i][w] and h.said_age[d][w] == 0 and h.sent[d][w] > 0 and age <= CALL_SHOW:
                V.draw_call(surf, view, pos[w], WORDS[int(h.sent[d][w])], None, 1.0 - age / CALL_SHOW)
        return
    entry = 1
    if not h.alive[st_i][entry]:
        return
    d = st_i
    while d > 0 and not (h.said_age[d][0] == 0 and h.sent[d][0] > 0):   # the last call sent, at or before now
        d -= 1
    w = int(h.sent[d][0])
    age = (i - d) * cfg.dt
    if not (h.said_age[d][0] == 0 and w > 0) or age > CALL_SHOW:
        return
    turn = CALL_TURN.get(w) if cfg.callout else None
    direction = None if turn is None else float(h.face[d][entry]) + turn
    V.draw_call(surf, view, pos[entry], WORDS[w], direction, 1.0 - age / CALL_SHOW)


def _draw_flashbangs(surf, view, tr, i, st_i, pos, eyes) -> None:
    """Live flashbangs (only the thrower's side knows where one lies, so other eyes don't show it) and who's
    blinded. (A bang's light is drawn under the operators, in draw_round: everyone sees that.)"""
    cfg, h, game = tr["cfg"], tr["hist"], tr["game"]
    mine = _team(tr, eyes)
    for w in range(game.N):
        if h.bang_timer[st_i][w] > 0 and (not mine or w in mine):
            V.draw_flashbang(surf, view, h.bang_pos[st_i][w], blink=int(i * 3) % 2 == 0)
    blind_steps = cfg.blind_time / cfg.dt
    for w in range(game.N):
        left = h.blind[st_i][w]
        if left > 0 and h.alive[st_i][w] and _shown(tr, st_i, eyes, w):
            V.draw_blind(surf, view, pos[w], min(left / blind_steps, 1.0))


def frame(tr: dict, margin: float = 3.5):
    """What the camera shows: the building and everywhere anyone went, with a margin. -> (lo, size)"""
    b, pos = tr["building"], tr["hist"].pos.reshape(-1, 2)
    lo = np.minimum(b["lo"], pos.min(0)) - margin
    hi = np.maximum(b["hi"], pos.max(0)) + margin
    return lo, hi - lo


def _result(tr: dict) -> tuple[str, tuple]:
    o, alive = tr["outcome"], tr["hist"].alive[-1]
    a_out, _ = tr["game"].sides_out(np.asarray(alive))
    if o == 1:
        return "ASSAULT WINS", V.ASSAULT
    if o == -1:
        return ("DEFENSE WINS" if a_out else "TIME UP: DEFENSE WINS"), V.DEFENSE
    return ("ALL DOWN" if tr["game"].teams_on else "BOTH DOWN"), T.TEXT


def render_round(assault: str, defense: str, building: int, out: str, *, seed: int = 0, eyes: str = "assault",
                 title: str = "", slow: float = 1.0, hold: float = 2.0, preset: str = "medium", sample: str = "",
                 cfg: cqb.CQBConfig | None = None) -> dict:
    """One round, landscape: the building on the right, title, who's who, clock and result on the left."""
    tr = rollout(assault, defense, building, seed, cfg=cfg, sample=sample)
    render_played(tr, out, eyes=eyes, title=title, slow=slow, hold=hold, preset=preset)
    return tr


def played(game: cqb.CQBGame, cfg: cqb.CQBConfig, result: dict, i: int, seeds, assault: str, defense: str) -> dict:
    """Round i of a recorded play_rounds batch (buildings `seeds`, game `game`) in rollout's form, for render_played:
    so rounds can be picked from a batch and drawn exactly as they were played."""
    h = jax.tree.map(lambda x: np.asarray(x[i]), result["history"])
    last = int(np.argmax(h.done)) if h.done.any() else game.T
    h = jax.tree.map(lambda x: x[:last + 1], h)
    return {"cfg": cfg, "game": game, "building": cqb.make_building(int(seeds[i]), cfg), "seed": int(seeds[i]),
            "hist": h, "steps": last, "outcome": int(h.outcome[-1]), "assault": assault, "defense": defense}


def _timeline(steps: int, dt: float, slow: float, hold: float, min_seconds: float) -> list[tuple[float, bool]]:
    """(round step, replaying?) for every frame. The round plays at `slow`, then holds on the result. When the shot
    must last longer (a long narration line) it doesn't freeze: after a short hold the end of the round replays in
    slow motion (at least its last 3 s, more if that would crawl), then holds again."""
    play = [(min(f / T.FPS * slow / dt, steps), False) for f in range(int(steps * dt / slow * T.FPS))]
    pause, end = 1.2, 1.0
    spare = min_seconds - len(play) / T.FPS - pause - end
    if spare < 2.5:   # not worth a replay: just hold longer
        return play + [(steps, False)] * int(max(hold, min_seconds - len(play) / T.FPS) * T.FPS)
    window = min(steps * dt, max(3.0, 0.2 * spare))
    n = int(spare * T.FPS)
    replay = [(steps - window / dt + window / dt * k / (n - 1), True) for k in range(n)]
    return play + [(steps, False)] * int(pause * T.FPS) + replay + [(steps, False)] * int(end * T.FPS)


def _sounds(tr: dict, timeline: list, events: list) -> None:
    """The round's sounds at the moments they happen on screen (replays included): shots, doors, reloads,
    flashbangs, hits. Shots are thinned to one per shooter every 0.12 s so blind fire stays listenable."""
    h, last_shot = tr["hist"], [-1.0] * tr["game"].N
    radio = tr["game"].pilot_on and any(tr["cfg"].radio)
    voiced = (0, -10.0)                                       # (word, time) last spoken aloud
    prev = None
    for f, (i, _) in enumerate(timeline):
        k, t = int(i), f / T.FPS
        if prev is None or k < prev:                          # the start, or a replay jumping back
            prev = k
        for d in range(prev + 1, k + 1):
            for w in range(tr["game"].N):
                if h.fired[d][w] and t - last_shot[w] >= 0.12:
                    events.append((t, "gunshot", 0.55))
                    last_shot[w] = t
                if h.door_event[d][w]:
                    events.append((t, "door", 0.5))
                if h.reload_now[d][w]:
                    events.append((t, "reload", 0.6))
                if h.bang_now[d][w]:
                    events.append((t, "flashbang", 0.9))
                if (h.alive[d - 1][w] and not h.alive[d][w]) or h.hp[d][w] < h.hp[d - 1][w]:
                    events.append((t, "hit", 0.7))
            if radio and h.said_age[d][0] == 0 and h.sent[d][0] > 0:   # a call reached the entry operator
                word = int(h.sent[d][0])
                if word != voiced[0] or t - voiced[1] >= 2.0:    # say it aloud when it changes, else every 2 s
                    events.append((t, f"radio_{WORDS[word].lower()}", 0.9))
                    voiced = (word, t)
        prev = k


WORDS = ["", "ALPHA", "BRAVO", "CHARLIE", "DELTA"]
MEANING_WAY = {1: "your way: ahead", 2: "your way: left", 3: "your way: right", 4: "your way: behind"}
MEANING_BEARING = {1: "he's ahead of you", 2: "he's to your left", 3: "he's to your right", 4: "he's behind you"}


def _radio_line(tr: dict, now: int):
    """(ep11 10zm) With a drone pilot on the radio: the call the entry operator last heard, what it means under the
    protocol (callout), how fresh it is (dimmer when old); "RADIO MUTED" when muted. -> (text, colour) or None."""
    cfg, game, h = tr["cfg"], tr["game"], tr["hist"]
    if not any(cfg.radio):
        return None
    if cfg.radio_test == "mute":
        return "radio: MUTED", T.RED
    if not game.pilot_on:                                     # a pair: each one's last word, while fresh
        said = [f"{k + 1}: {WORDS[int(h.sent[now][w])]}" for k, w in enumerate(game.side_ops(cqb.ASSAULT))
                if h.sent[now][w] > 0 and h.said_age[now][w] * cfg.dt <= cfg.radio_memory]
        return ("radio  " + "   ".join(said), T.AMBER) if said else ("radio: (silent)", T.MUTED)
    w, age = int(h.sent[now][0]), int(h.said_age[now][0])
    if w <= 0 or age * cfg.dt > cfg.radio_memory:
        return "pilot: (silent)", T.MUTED
    meaning = (MEANING_WAY if cfg.callout_way else MEANING_BEARING).get(w, "") if cfg.callout else ""
    return f"pilot: {WORDS[w]}" + (f"  ({meaning})" if meaning else ""), T.AMBER if age <= 5 else T.MUTED


def render_played(tr: dict, out: str, *, eyes: str = "assault", title: str = "", slow: float = 1.0, hold: float = 2.0,
                  preset: str = "medium", vertical: bool = False, labels: tuple | None = None,
                  min_seconds: float = 0.0, events: list | None = None, panel: bool = True) -> dict:
    """Draw a round already played (rollout or played) as a landscape video (vertical: 9:16 for Shorts, the house
    on top and the words under it). labels: what to call (Assault, Defense) on screen (default: run names).
    min_seconds: the shot's length at least; a short round fills it with a slow-motion replay of its end.
    panel: False (ep11, for side_by_side): only the house, and the replay label inside it (no text panel to crop)."""
    assault, defense = tr["assault"], tr["defense"]
    cfg, b = tr["cfg"], tr["building"]
    W, H = (1080, 1920) if vertical else (T.WIDTH, T.HEIGHT)
    surf = pygame.Surface((W, H))
    lo, size = frame(tr, margin=1.5 if vertical else 3.5)   # Shorts: a bigger house on the phone (ep10 review)
    if vertical:   # tag's Shorts lessons (ep8): the bottom ~500 px belong to YouTube's overlay, and the right edge to
        # its buttons; clock and result go in the header (ep9 4e), so the band under the house is free for captions
        rect, x, y = pygame.Rect(20, 330, W - 40, 980), 60, dict(title=110, names=205, step=48, showing=None,
                                                                 clock=275, result=275, lines=None)
    else:
        rect, x, y = pygame.Rect(560, 40, W - 600, H - 80), 70, dict(title=90, names=250, step=60, showing=390,
                                                                       clock=470, result=560, lines=660)
    view = V.View(rect, tuple(size), tuple(lo))
    cast = _light(tr)
    bang = _bang_light(tr) if any(cfg.flashbangs) else None
    timeline = _timeline(tr["steps"], cfg.dt, slow, hold, min_seconds)
    if events is not None:
        _sounds(tr, timeline, events)
    with VideoWriter(out, (W, H), preset=preset) as vw:
        for i, replaying in timeline:
            surf.fill(T.BG)
            draw_round(surf, view, tr, i, eyes=eyes, cast=cast, bang=bang)
            if not panel:
                if replaying:
                    T.text(surf, "REPLAY · SLOW MOTION", (rect.x + 30, rect.y + 20), kind="display", size=46,
                           color=T.AMBER)
                elif slow >= 2 and i < tr["steps"]:
                    T.text(surf, f"{slow:g}x SPEED", (rect.x + 30, rect.y + 20), kind="display", size=40,
                           color=T.MUTED)
                vw.write(surf)
                continue
            if title:
                T.text(surf, title, (x, y["title"]), kind="display", size=56 if vertical else 50)
            for n, (w, spec) in enumerate(((cqb.ASSAULT, assault), (cqb.DEFENSE, defense))):
                yy = y["names"] + y["step"] * n
                pygame.draw.circle(surf, COLORS[w], (x + 14, yy), 12)
                name = labels[w] if labels else label(spec)
                T.text(surf, f"{NAMES[w]}: {name}", (x + 40, yy), kind="semibold", size=30, anchor="midleft")
            view_of = {"assault": "what Assault sees", "defense": "what Defense sees", "all": "everything (the reveal)"}[eyes]
            if y["showing"]:
                T.text(surf, f"showing: {view_of}", (x, y["showing"]), kind="regular", size=26, color=T.MUTED)
            t = i * cfg.dt
            T.text(surf, f"{max(cfg.time_limit - t, 0):4.1f} s left", (x, y["clock"]), kind="mono-bold",
                   size=38 if vertical else 44)
            if slow >= 2 and not replaying and i < tr["steps"]:   # fast-forwarded: say so
                T.text(surf, f"{slow:g}x SPEED", (x + 300, y["clock"] + 4) if vertical else (x, y["clock"] + 56),
                       kind="display", size=30, color=T.MUTED)
            if replaying:
                T.text(surf, "REPLAY · SLOW MOTION", (W - x, y["result"]) if vertical else (x, y["result"]),
                       kind="display", size=36 if vertical else 46, color=T.AMBER,
                       anchor="topright" if vertical else "topleft")
            if vertical and i >= tr["steps"]:   # the result beside the clock; no counters (they'd sit under the overlay)
                result, color = _result(tr)
                T.text(surf, result, (W - x, y["result"]), kind="display", size=40, color=color, anchor="topright")
                vw.write(surf)
                continue
            if vertical:
                vw.write(surf)
                continue
            now = int(min(i, tr["steps"]))
            game, h_ = tr["game"], tr["hist"]
            lines = [f"{NAMES[s]} flashbangs: {sum(int(h_.bangs[now][w]) for w in game.side_ops(s))} of "
                     f"{cfg.flashbangs[s] * len(game.side_ops(s))}" for s in (cqb.ASSAULT, cqb.DEFENSE) if cfg.flashbangs[s]]
            if cfg.health:
                lines += [f"{op_name(game, w)} health: {int(round(float(h_.hp[now][w])))}" for w in range(game.N)]
            if cfg.magazine:
                for w in range(game.N):
                    lines.append(f"{op_name(game, w)} ammo: {int(h_.mag[now][w])} | {int(h_.reserve[now][w])}"
                                 + ("  reloading" if h_.reload_timer[now][w] > 0 else ""))
            radio = _radio_line(tr, now)
            if radio:   # (ep11 10zm) what the pilot is calling, and what the call means
                T.text(surf, radio[0], (x, y["lines"]), kind="semibold", size=30, color=radio[1])
                lines = [""] + lines
            for n, line in enumerate(lines):
                T.text(surf, line, (x, y["lines"] + 40 * n), kind="regular", size=26, color=T.MUTED)
            if i >= tr["steps"]:
                result, color = _result(tr)
                T.text(surf, result, (x, y["result"]), kind="display", size=46, color=color)
            vw.write(surf)
    return tr


def side_by_side(halves: list, out: str, heading: str = "", preset: str = "medium", labels=None, results=None) -> None:
    """(ep11) Two landscape round clips side by side in one 16:9 frame. Each is cropped to its house (the text panel
    off: on a phone two full frames side by side leave each house tiny), with its label above (labels: two strings)
    and its result under it from the moment its round ends (results: two (text, colour, seconds) or None); a heading
    on top; the shorter clip holds its last frame. Video only (compose adds the sound)."""
    import subprocess
    import tempfile
    durs = [float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(h)],
                                 capture_output=True, text=True).stdout) for h in halves]
    m, W, H = max(durs), T.WIDTH, T.HEIGHT
    x0, cw = 560, 1320                        # render_played's landscape house area: x 560-1880
    hw = (W - 60) // 2                        # each half's width on screen
    hh = int(H * hw / cw) // 2 * 2
    top = 190 if heading else 120
    with tempfile.TemporaryDirectory() as td:
        bg = Path(td) / "bg.png"
        surf = pygame.Surface((W, H))
        surf.fill(T.BG)
        if heading:
            T.text(surf, heading, (W // 2, 40), kind="display", size=50, anchor="midtop")
        for k, lab in enumerate(labels or []):
            T.text(surf, lab, (20 + k * (hw + 20) + hw // 2, top - 12), kind="display", size=40, anchor="midbottom")
        pygame.image.save(surf, str(bg))
        inputs, chain, last = ["-loop", "1", "-framerate", str(T.FPS), "-i", str(bg)], [], "0:v"
        for k, h in enumerate(halves):
            inputs += ["-i", str(h)]
            chain.append(f"[{k + 1}:v]tpad=stop_mode=clone:stop_duration={m},crop={cw}:{H}:{x0}:0,scale={hw}:{hh}[h{k}]")
            chain.append(f"[{last}][h{k}]overlay={20 + k * (hw + 20)}:{top}[o{k}]")
            last = f"o{k}"
        for k, res in enumerate(results or []):
            if not res:
                continue
            text, color, at = res
            img = pygame.Surface((hw, 70), pygame.SRCALPHA)
            T.text(img, text, (hw // 2, 35), kind="display", size=44, color=color, anchor="center")
            png = Path(td) / f"res{k}.png"
            pygame.image.save(img, str(png))
            idx = sum(1 for a in inputs if a == "-i")
            inputs += ["-loop", "1", "-framerate", str(T.FPS), "-i", str(png)]
            chain.append(f"[{last}][{idx}:v]overlay={20 + k * (hw + 20)}:{top + hh + 10}:enable='gte(t,{at:.2f})'[r{k}]")
            last = f"r{k}"
        chain.append(f"[{last}]trim=duration={m},setpts=PTS-STARTPTS[v]")
        subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", ";".join(chain), "-map", "[v]",
                        "-an", "-c:v", "libx264", "-preset", preset, "-crf", "18", "-pix_fmt", "yuv420p",
                        "-r", str(T.FPS), str(out)], check=True)


def render_houses(out: str, *, first: int = 0, cols: int = 3, rows: int = 2, seconds: float = 6.0, per: float = 0.8,
                  heading: str = "", footnote: str = "", preset: str = "medium", house: str = "") -> None:
    """A gallery of generated kill houses, cycling: a new set every `per` seconds (seeds from `first`; the
    practice houses are 0-999, the test houses 100000 on). house: a preset (cqb.HOUSES; ep11: "big")."""
    from dataclasses import replace
    cfg = replace(cqb.CQBConfig(), **cqb.HOUSES[house])
    W, H = T.WIDTH, T.HEIGHT
    top, bottom = (130 if heading else 30), (60 if footnote else 20)
    cw, ch = (W - 40) // cols, (H - top - bottom) // rows
    rects = [pygame.Rect(20 + (k % cols) * cw + 10, top + (k // cols) * ch + 10, cw - 20, ch - 20)
             for k in range(cols * rows)]
    surf = pygame.Surface((W, H))
    houses: dict = {}
    with VideoWriter(out, (W, H), preset=preset) as vw:
        for f in range(int(seconds * T.FPS)):
            n = int(f / T.FPS / per)
            surf.fill(T.BG)
            if heading:
                T.text(surf, heading, (W // 2, 40), kind="display", size=54, anchor="midtop")
            for j, r in enumerate(rects):
                seed = first + n * len(rects) + j
                if seed not in houses:
                    houses[seed] = cqb.make_building(seed, cfg)
                b = houses[seed]
                lo, hi = b["lo"] - 3.0, b["hi"] + 3.0   # room for the entry arrows
                V.draw_building(surf, b, V.View(r, tuple(hi - lo), tuple(lo)), starts=True)
            if footnote:
                T.text(surf, footnote, (W // 2, H - 30), kind="regular", size=26, color=T.MUTED, anchor="midbottom")
            vw.write(surf)


def _input_groups(cfg: cqb.CQBConfig, n_in: int) -> list[tuple[str, int]]:
    """observe()'s numbers in order, as (what they are, how many), cut to the n_in a brain reads."""
    g = [("walls around it", cfg.n_rays), ("the other one", 10), ("what it hears", 9), ("itself", 7),
         ("nearest doors", 8), ("rooms it has seen", cfg.seen_grid[0] * cfg.seen_grid[1]), ("clock", 1)]
    if any(cfg.flashbangs):
        g.append(("flashbangs", 8))
    if cfg.magazine:
        g.append(("its rifle", 7))
    if cfg.health:
        g.append(("health", 2))
    if cfg.react:
        g.append(("aim", 1))
    out, n = [], 0
    for name, k in g:
        if n >= n_in:
            break
        out.append((name, min(k, n_in - n)))
        n += k
    return out


def _fire_color(a: float):
    """tanh firing -1..1: blue = negative, amber = positive, fading to the background at 0 (as record_car's)."""
    return T.mix(T.PANEL, T.AMBER if a > 0 else T.BLUE, min(1.0, abs(a)) ** 0.8)


def brain_activity(tr: dict, params: dict, who: int) -> dict:
    """Every number a (memoryless) brain computed in this round, recomputed from the recorded states: what it read
    (obs), both layers (h1, h2), its moves and its on/off switches as chances (0..1). Decision k uses state k."""
    from ailearns.agents.ppo_cqb import N_MOVE, has_memory
    assert not has_memory(params), "brains with memory: not drawn yet"
    p = {k: np.asarray(v) for k, v in params.items()}
    n_in = p["body.0.weight"].shape[1]
    hist = jax.tree.map(jnp.asarray, tr["hist"])
    obs = np.asarray(jax.vmap(lambda st: tr["game"].observe(st, who))(hist))[:, :n_in]
    h1 = np.tanh(obs @ p["body.0.weight"].T + p["body.0.bias"])
    h2 = np.tanh(h1 @ p["body.2.weight"].T + p["body.2.bias"])
    out = h2 @ p["policy.weight"].T + p["policy.bias"]
    return {"obs": obs, "h1": h1, "h2": h2, "moves": np.clip(out[:, :N_MOVE], -1, 1),
            "switch": 1 / (1 + np.exp(-out[:, N_MOVE:])), "p": p}


SWITCH_NAMES = ["FIRE", "OPEN DOOR", "RUN", "THROW", "RELOAD"]


class CQBBrainPanel:
    """One operator's brain as it played this round: what it reads (grouped), 2 layers of neurons, what it does.
    Neurons sit in a fixed order, sorted by how they move with its trigger over the round, so the "fire" neurons
    gather at one end of each grid. Lines: the strongest pushes at that moment (weight x firing)."""

    def __init__(self, act: dict, cfg: cqb.CQBConfig, rect: pygame.Rect, who: int, heading: str = ""):
        self.a, self.rect, self.who = act, rect, who
        self.heading = heading or f"{NAMES[who].upper()}'S BRAIN"
        self.groups = _input_groups(cfg, act["obs"].shape[1])
        fire = act["switch"][:, 0]
        self.order = []
        for h in (act["h1"], act["h2"]):
            corr = np.nan_to_num([np.corrcoef(h[:, j], fire)[0, 1] if h[:, j].std() > 1e-6 else 0.0
                                  for j in range(h.shape[1])])
            self.order.append(np.argsort(-corr))
        self.slot = [np.argsort(o) for o in self.order]   # neuron -> its place in the grid
        n = act["h1"].shape[1]
        self.cols = 16
        self.cell = min(14, (rect.h - 230) // (n // self.cols))
        x, y = rect.x, rect.y + 160
        self.in_x, self.dot = x + 170, 12
        self.in_rows, yy = [], y
        for name, k in self.groups:                       # one or more rows of dots per group, 24 to a row
            rows = (k + 23) // 24
            self.in_rows.append((name, k, yy))
            yy += 20 * rows + 14
        self.grids = [(x + 500, y), (x + 500 + self.cols * self.cell + 50, y)]
        self.out_x = self.grids[1][0] + self.cols * self.cell + 50

    def _in_xy(self, i: int):
        n = 0
        for name, k, yy in self.in_rows:
            if i < n + k:
                j = i - n
                return self.in_x + (j % 24) * self.dot + 6, yy + (j // 24) * 20 + 10
            n += k

    def _neuron_xy(self, layer: int, j: int):
        slot = int(self.slot[layer][j])
        gx, gy = self.grids[layer]
        return gx + (slot % self.cols) * self.cell + self.cell // 2, gy + (slot // self.cols) * self.cell + self.cell // 2

    def _lines(self, surf, w, x, src, dst, top: int) -> None:
        c = w * x[None, :]
        peak = np.abs(c).max() + 1e-9
        for f in np.argsort(-np.abs(c), axis=None)[:top]:
            o, i = np.unravel_index(f, c.shape)
            s = abs(c[o, i]) / peak
            pygame.draw.aaline(surf, (*(T.AMBER if c[o, i] > 0 else T.BLUE), int(60 + 150 * s)), src(i), dst(o))

    def draw(self, surf: pygame.Surface, k: int) -> None:
        a, p = self.a, self.a["p"]
        obs, h1, h2 = a["obs"][k], a["h1"][k], a["h2"][k]
        x0, y0 = self.rect.x, self.rect.y
        T.text(surf, self.heading, (x0, y0 + 10), "display", 44, T.TEXT)
        T.text(surf, "recomputed from this round, 10 decisions a second", (x0, y0 + 64), "semibold", 22, T.MUTED)
        n_sw = a["switch"].shape[1]
        outs = [(self.out_x, self.grids[1][1] + 30 + 62 * n) for n in range(n_sw)]
        lines = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        self._lines(lines, p["body.0.weight"], obs, self._in_xy, lambda j: self._neuron_xy(0, j), 60)
        self._lines(lines, p["body.2.weight"], h1, lambda j: self._neuron_xy(0, j), lambda j: self._neuron_xy(1, j), 60)
        self._lines(lines, p["policy.weight"][3:], h2, lambda j: self._neuron_xy(1, j), lambda o: outs[o], 30)
        surf.blit(lines, (0, 0))
        n = 0
        for name, cnt, yy in self.in_rows:
            T.text(surf, name, (self.in_x - 12, yy + 10), "semibold", 21, T.TEXT, anchor="midright")
            for j in range(cnt):
                pygame.draw.circle(surf, _fire_color(float(obs[n + j])), self._in_xy(n + j), 5)
            n += cnt
        for layer, h in enumerate((h1, h2)):
            gx, gy = self.grids[layer]
            for j in range(len(h)):
                pygame.draw.circle(surf, _fire_color(float(h[j])), self._neuron_xy(layer, j), max(2, self.cell // 2 - 2))
            T.text(surf, f"layer {layer + 1}", (gx + self.cols * self.cell // 2, gy - 30), "bold", 24, T.TEXT,
                   anchor="center")
            T.text(surf, f"{len(h)} neurons", (gx + self.cols * self.cell // 2, gy + len(h) // self.cols * self.cell + 22),
                   "semibold", 20, T.MUTED, anchor="center")
        for n, (ox, oy) in enumerate(outs):
            on = float(a["switch"][k, n])
            col = T.RED if n == 0 else T.AMBER
            if on > 0.5:
                T.glow(surf, (ox, oy), 22, col, strength=90)
            pygame.draw.circle(surf, col if on > 0.5 else T.PANEL_EDGE, (ox, oy), 12)
            T.text(surf, SWITCH_NAMES[n], (ox + 24, oy - 22), "bold", 22, col if on > 0.5 else T.MUTED)
            r = pygame.Rect(ox + 24, oy + 6, 120, 10)
            pygame.draw.rect(surf, T.PANEL_EDGE, r, border_radius=5)
            pygame.draw.rect(surf, col, pygame.Rect(r.x, r.y, max(3, on * r.w), r.h), border_radius=5)
        ly = self.rect.bottom - 30
        for x, col, txt in ((x0, T.AMBER, "fires positive"), (x0 + 210, T.BLUE, "fires negative")):
            pygame.draw.circle(surf, col, (x + 10, ly + 12), 9)
            T.text(surf, txt, (x + 28, ly), "semibold", 21, T.TEXT)
        T.text(surf, "lines = strongest signals right now  ·  bars = how sure it is", (x0 + 420, ly), "semibold", 21,
               T.MUTED)


def render_brain(tr: dict, params: dict, who: int, out: str, *, eyes: str = "all", title: str = "", heading: str = "",
                 slow: float = 1.0, hold: float = 2.0, preset: str = "medium", labels: tuple | None = None,
                 min_seconds: float = 0.0, events: list | None = None) -> dict:
    """The round on the left (small), one side's brain firing on the right (CQBBrainPanel). Under the house:
    does that side see the other right now? (The blind-fire beat: the trigger on, nobody in its light.)"""
    cfg = tr["cfg"]
    W, H = T.WIDTH, T.HEIGHT
    surf = pygame.Surface((W, H))
    lo, size = frame(tr)
    view = V.View(pygame.Rect(30, 170, 620, 620), tuple(size), tuple(lo))
    cast = _light(tr)
    bang = _bang_light(tr) if any(cfg.flashbangs) else None
    panel = CQBBrainPanel(brain_activity(tr, params, who), cfg, pygame.Rect(690, 30, W - 720, H - 60), who,
                           heading)
    shots = np.cumsum(np.asarray(tr["hist"].fired)[:, who])
    timeline = _timeline(tr["steps"], cfg.dt, slow, hold, min_seconds)
    if events is not None:
        _sounds(tr, timeline, events)
    with VideoWriter(out, (W, H), preset=preset) as vw:
        for i, replaying in timeline:
            surf.fill(T.BG)
            draw_round(surf, view, tr, i, eyes=eyes, cast=cast, bang=bang)
            if title:
                T.text(surf, title, (40, 40), kind="display", size=44)
            for n, (w, spec) in enumerate(((cqb.ASSAULT, tr["assault"]), (cqb.DEFENSE, tr["defense"]))):
                yy = 115 + 36 * n
                pygame.draw.circle(surf, COLORS[w], (52, yy), 10)
                T.text(surf, f"{NAMES[w]}: {labels[w] if labels else label(spec)}",
                       (72, yy), kind="semibold", size=24, anchor="midleft")
            k = int(min(i, tr["steps"]))
            sees = bool(tr["hist"].seen[k][who].any())
            T.text(surf, f"{max(cfg.time_limit - i * cfg.dt, 0):4.1f} s left", (40, 820), kind="mono-bold", size=36)
            T.text(surf, f"can it see the {('attacker', 'defender')[1 - who]}? " + ("YES" if sees else "no"), (40, 875),
                   kind="semibold", size=30, color=T.RED if sees else T.MUTED)
            T.text(surf, f"shots fired: {int(shots[k])}", (40, 920), kind="semibold", size=30, color=T.TEXT)
            if replaying:
                T.text(surf, "REPLAY · SLOW MOTION", (40, 975), kind="display", size=36, color=T.AMBER)
            elif i >= tr["steps"]:
                result, color = _result(tr)
                T.text(surf, result, (40, 975), kind="display", size=36, color=color)
            panel.draw(surf, k)
            vw.write(surf)
    return tr


BAR_COLORS = {"assault": V.ASSAULT, "defense": V.DEFENSE, "muted": T.MUTED, "amber": T.AMBER}


def render_bars(bars: list[dict], out: str, *, heading: str = "", footnote: str = "", per: float = 0.9,
                hold: float = 2.5, preset: str = "medium", top: float = 1.0, number: bool = False) -> None:
    """Numbers as horizontal bars that grow one after another (e.g. Assault's win rate as each rule is added).
    bars: [{label, value 0..1, color: assault|defense|muted|amber, note}]; `top`: the value a full bar stands for;
    `number`: print values as plain numbers (crashes a round: 7.3), not percentages."""
    W, H = T.WIDTH, T.HEIGHT
    n = len(bars)
    y0 = 300 if heading else 200
    row = min(150, (H - y0 - 140) // max(n, 1))
    label_w = max(T.font("semibold", 40).size(b["label"])[0] for b in bars) + 60
    x0, full = 120 + label_w, W - 120 - label_w - 340
    start = [0.5 + k * per for k in range(n)]
    total = start[-1] + 0.6 + hold
    surf = pygame.Surface((W, H))
    with VideoWriter(out, (W, H), preset=preset) as vw:
        for f in range(int(total * T.FPS)):
            t = f / T.FPS
            surf.fill(T.BG)
            if heading:
                T.text(surf, heading, (120, 110), kind="display", size=60)
            for k, b in enumerate(bars):
                y = y0 + k * row
                grow = T.ease_out(min(max((t - start[k]) / 0.6, 0.0), 1.0))
                T.text(surf, b["label"], (120, y + row // 2 - 10), kind="semibold", size=40, anchor="midleft",
                       color=T.TEXT if grow > 0 else T.MUTED)
                pygame.draw.rect(surf, T.PANEL, (x0, y + 18, full, row - 56), border_radius=12)
                w = int(full * min(b["value"] / top, 1.0) * grow)
                if w > 0:
                    pygame.draw.rect(surf, BAR_COLORS.get(b.get("color", "assault"), V.ASSAULT),
                                     (x0, y + 18, w, row - 56), border_radius=12)
                if grow > 0:
                    T.text(surf, f"{b['value'] * grow:.2g}" if number else f"{b['value'] * grow:.0%}", (x0 + w + 24, y + row // 2 - 10), kind="bold", size=48,
                           anchor="midleft", alpha=int(255 * grow))
                    if b.get("note"):
                        T.text(surf, b["note"], (x0 + w + 150, y + row // 2 - 10), kind="regular", size=30,
                               color=T.MUTED, anchor="midleft", alpha=int(255 * grow))
            if footnote:
                T.text(surf, footnote, (120, H - 90), kind="regular", size=28, color=T.MUTED)
            vw.write(surf)
