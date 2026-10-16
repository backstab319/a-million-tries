"""The games and the AIs that learn them: play, train, evaluate.

    uv run games play                         # Snake, you play
    uv run --group jax games drive            # the race car, you drive
    uv run --group jax games tag              # tag vs the AI, you're the robot (--as blob: you run)
    uv run games train runs/snake-1 --arch cnn-deep --steps 20e6
    uv run --group jax games train-car runs/car-1 --steps 300e6
    uv run games eval runs/snake-1            # best-move games: win rate, mean score
    uv run --group jax games eval-car runs/car-1
    uv run --group jax games train-tag runs/tag-1 --steps 50e6
    uv run --group jax games eval-tag runs/tag-1
    uv run --group jax games train-cqb runs/cqb-1 --steps 200e6      # CQB: Assault vs the camper
    uv run --group jax games eval-cqb runs/cqb-1
    uv run --group jax games watch-cqb runs/cqb-1 camper 100064      # one round as a video (needs ffmpeg)
"""

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="games", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("play", help="play Snake yourself")
    sp.add_argument("--size", type=int, default=10)
    sp.add_argument("--speed", type=float, default=8.0, help="moves per second")

    sp = sub.add_parser("drive", help="drive the race car yourself (needs --group jax)")
    sp.add_argument("--track", type=int, default=100_000, help="track seed (100000+: tracks the AI never trains on)")

    sp = sub.add_parser("tag", help="play tag yourself against the AI (needs --group jax)")
    sp.add_argument("--as", dest="side", choices=["robot", "blob"], default="robot",
                    help="robot: you chase; blob: you run")
    sp.add_argument("--vs", help="the AI you play (default: the last generation of the other side)")
    sp.add_argument("--arena", type=int, default=100_000, help="first arena (100000+: arenas the AI never trains in)")
    sp.add_argument("--no-fog", action="store_true", help="see the whole arena (the AI doesn't)")
    sp.add_argument("--match", action="store_true", help="the scored ladder: every generation, 10 rounds each")

    sp = sub.add_parser("train", help="train a Snake AI with PPO")
    sp.add_argument("run", help="run directory, e.g. runs/snake-1")
    sp.add_argument("--size", type=int, default=10)
    sp.add_argument("--steps", type=float, default=20e6)
    sp.add_argument("--envs", type=int, default=128)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--arch", choices=["mlp", "cnn", "cnn-deep"], default="mlp")
    sp.add_argument("--view", choices=["ego2", "ego", "board"], default="ego2")
    sp.add_argument("--gamma", type=float, default=0.99, help="how much it values the future (per move)")
    sp.add_argument("--resume", action="store_true", help="continue from <run>/latest.pt")
    sp.add_argument("--max-hours", type=float, help="stop cleanly after this long (resumable)")
    sp.add_argument("--device", default="cpu", help="cpu | cuda (torch backend)")
    sp.add_argument("--backend", choices=["torch", "jax"], default="torch",
                    help="jax: compiled training loop on every device (needs `uv sync --group jax`)")

    sp = sub.add_parser("train-car", help="train a race car AI with PPO (needs --group jax)")
    sp.add_argument("run", help="run directory, e.g. runs/car-1")
    sp.add_argument("--steps", type=float, default=300e6)
    sp.add_argument("--envs", type=int, default=4096)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--hidden", type=int, default=256, help="neurons per layer")
    sp.add_argument("--ray-range", type=float, default=60.0, help="how far the lasers reach (m)")
    sp.add_argument("--messy-starts", action="store_true", help="practice from off-centre, angled, moving starts")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--max-hours", type=float)

    sp = sub.add_parser("train-tag", help="train a tag AI (the chaser, by default) with PPO (needs --group jax)")
    sp.add_argument("run", help="run directory, e.g. runs/tag-1")
    sp.add_argument("--steps", type=float, default=50e6)
    sp.add_argument("--envs", type=int, default=2048)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--hidden", type=int, default=256, help="neurons per layer")
    sp.add_argument("--side", choices=["chaser", "runner"], default="chaser", help="who learns")
    sp.add_argument("--opponent", default="flee",
                    help="who it plays against: flee | smart | still | naive, or frozen brains runs/a:175,runs/b:200")
    sp.add_argument("--closer-reward", type=float, default=0.0, help="reward per metre closer (0: catches only)")
    sp.add_argument("--init-from", default="", help="start from another run's brain, e.g. runs/tag-v2-closer:190")
    sp.add_argument("--remember-heading", action="store_true", help="out of sight: remember which way it was going")
    sp.add_argument("--reset-std", action="store_true", help="with --init-from: experiment again (fresh move noise)")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--max-hours", type=float)

    sp = sub.add_parser("train-cqb", help="train a CQB AI (Assault, by default) with PPO, or evolution with --es "
                                          "(needs --group jax)")
    sp.add_argument("run", help="run directory, e.g. runs/cqb-1")
    sp.add_argument("--steps", type=float, default=200e6)
    sp.add_argument("--envs", type=int, default=2048)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--hidden", type=int, default=256, help="neurons per layer")
    sp.add_argument("--side", choices=["assault", "defense"], default="assault", help="who learns")
    sp.add_argument("--opponent", default="camper",
                    help="who it plays against: camper | wanderer | walker | dummy | random, a brain runs/a:175, "
                         "or a pool of them: camper,runs/a:175 (round i plays member i %% K)")
    sp.add_argument("--explore-reward", type=float, default=0.0, help="per new map cell seen (0: the round result only)")
    sp.add_argument("--time-cost", type=float, default=0.0, help="Assault pays this per second (0: the round result only)")
    sp.add_argument("--init-from", default="", help="start from another run's brain, e.g. runs/cqb-v3-dummy:38")
    sp.add_argument("--ent-coef", type=float, default=0.0, help="bonus for keeping its options open (0: none)")
    sp.add_argument("--reset-std", action="store_true", help="with --init-from: experiment again (fresh move noise)")
    sp.add_argument("--buildings", type=int, default=1024, help="how many training buildings")
    sp.add_argument("--flashbangs", default="0,0", help="per round, Assault,Defense (e.g. 2,0)")
    sp.add_argument("--memory", type=int, default=0, help="a memory (GRU) of this size (0: none)")
    sp.add_argument("--magazine", type=int, default=0, help="rounds per magazine (0: no magazines or reloading)")
    sp.add_argument("--health", type=float, default=0.0, help="health per operator (0: one hit and you're down)")
    sp.add_argument("--react", type=float, default=0.0, help="seconds in sight before your aim settles (0: none)")
    sp.add_argument("--teams", default="1,1", help="operators per side, Assault,Defense (e.g. 2,1)")
    sp.add_argument("--bump", type=float, default=0.0, help="metres a collision is heard (0: silent)")
    sp.add_argument("--ready", type=float, default=0.0, help="seconds to bring the rifle up after running (0: none)")
    sp.add_argument("--stun", type=float, default=0.0, help="seconds a hard collision stuns you (0: none)")
    sp.add_argument("--no-share", action="store_true", help="teams don't share sightings (each knows its own)")
    sp.add_argument("--radio", default="0,0", help="radio words per side, Assault,Defense (e.g. 4,0; ep11)")
    sp.add_argument("--split", action="store_true", help="the Assault team starts at different doors (ep11)")
    sp.add_argument("--listen", type=float, default=0.0, help="radio: nudge to act on words heard (ep11)")
    sp.add_argument("--relay", type=float, default=0.0, help="with --no-share: fresh callouts, at most this old (s)")
    sp.add_argument("--intel", type=int, default=0, help="Assault operators who always know where the defender is (ep11)")
    sp.add_argument("--feed", type=int, default=0, help="Assault operators on a drone feed of their own (ep11)")
    sp.add_argument("--route", action="store_true", help="each operator reads the way to the enemy it knows of (ep11)")
    sp.add_argument("--pilot", action="store_true", help="Assault operator 1 is a drone pilot who only talks (ep11)")
    sp.add_argument("--callout", action="store_true", help="with --pilot: the pilot speaks a fixed protocol (ep11)")
    sp.add_argument("--callout-way", action="store_true", help="with --callout: call the way to him, not the bearing")
    sp.add_argument("--vf-warmup", type=int, default=0, help="first N updates train only the value head (ep11)")
    sp.add_argument("--gamma", type=float, default=0.995, help="how much it values the future (per decision; ep11: 0.999)")
    sp.add_argument("--critic-priv", action="store_true", help="asymmetric critic: the value also reads where he really is (training only; ep11)")
    sp.add_argument("--house", default="", help="a building preset (cqb.HOUSES: big = 8-12 rooms; ep11)")
    sp.add_argument("--signal", type=float, default=0.0, help="radio: nudge to say different words in different cases")
    sp.add_argument("--plateau", type=int, default=0, help="stop after this many updates without a new best (0: never)")
    sp.add_argument("--stacked", type=float, default=0.0, help="share of rounds that are entry drills (start beside "
                                                                 "Defense's door)")
    # episode 12: less ammo, the flashbang fix, the hostage
    sp.add_argument("--spare-mags", type=int, default=3, help="magazines carried besides the loaded one (ep12: 1)")
    sp.add_argument("--time-limit", type=float, default=45.0, help="seconds a round lasts")
    sp.add_argument("--blind-away", type=float, default=1.0, help="flashbang blindness when it wasn't in view, x the full time (ep12: 0.25)")
    sp.add_argument("--hostage", action="store_true", help="a hostage in the house; securing it wins for Assault (ep12)")
    sp.add_argument("--hostage-deep", type=float, default=0.0, help="with --hostage: it starts at least this far (m) from every way in (ep12: 6)")
    sp.add_argument("--extract", action="store_true", help="with --hostage: secured, it must be walked out of the house to win (ep12)")
    sp.add_argument("--hostage-noise", type=float, default=0.0, help="with --extract: Defense hears it being moved this far, m (ep12: 8)")
    # evolution strategies (ep9 on; ep12's coached evolution)
    sp.add_argument("--es", action="store_true", help="evolution strategies instead of PPO")
    sp.add_argument("--pop", type=int, default=256, help="es: members a generation")
    sp.add_argument("--rounds", type=int, default=48, help="es: rounds each member plays a generation")
    sp.add_argument("--sigma", type=float, default=0.005, help="es: size of a member's change to the brain")
    sp.add_argument("--sigma-throw", type=float, default=0.5, help="es: ... and to its throw output")
    sp.add_argument("--sigma-hostage", type=float, default=0.0, help="es: ... and to the hostage inputs' weights (0: as --sigma; ep12: 0.05)")
    sp.add_argument("--lr-rel", type=float, default=0.1, help="es: step per generation, x each weight's sigma")
    sp.add_argument("--generations", type=int, default=200, help="es: generations")
    sp.add_argument("--guard", type=int, default=2, help="es: undo a step losing more than this many rounds (-1: off)")
    sp.add_argument("--aim-bonus", type=float, default=0.0, help="es coaching (ranking only): up to this a round for aim on him at first sight (ep11)")
    sp.add_argument("--hostage-bonus", type=float, default=0.0, help="es coaching (ranking only): for progress with the hostage (defenders: for denying it; ep12: 0.5)")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--max-hours", type=float)

    sp = sub.add_parser("eval-cqb", help="CQB: one best-guess round in each unseen test building")
    sp.add_argument("run")
    sp.add_argument("--checkpoint", default="last")
    sp.add_argument("--first", type=int, default=0, help="first test building (choose on 0-63, report 64-255)")
    sp.add_argument("--n", type=int, default=64)
    sp.add_argument("--seed", type=int, default=5000)
    sp.add_argument("--opponent", help="play someone else than the one it trained against")

    sp = sub.add_parser("watch-cqb", help="CQB: one round as a top-down video, drawn as in the videos (needs ffmpeg)")
    sp.add_argument("assault", help="a brain runs/x:update, or a script: walker | walker_flash | wanderer | dummy ...")
    sp.add_argument("defense", help="a brain runs/x:update, or a script: camper | dummy ...")
    sp.add_argument("building", type=int, help="a building seed: 0-1023 to practise in, 100000-100255 for tests")
    sp.add_argument("--seed", type=int, default=5000)
    sp.add_argument("--batch", default="", help="first,n: the round exactly as played in a batch over test buildings "
                                                 "(the videos use 64,64 with seed 5000)")
    sp.add_argument("--eyes", choices=["assault", "defense", "all"], default="all", help="whose view (all: no fog)")
    sp.add_argument("--slow", type=float, default=1.0, help="playback speed (below 1: slow motion)")
    sp.add_argument("--out", default="", help="default: out/cqb-<building>.mp4")

    sp = sub.add_parser("eval", help="Snake: best-move games from a checkpoint")
    sp.add_argument("run")
    sp.add_argument("--checkpoint", default="last", help="first | last | <update> | <path>")
    sp.add_argument("--games", type=int, default=256)
    sp.add_argument("--seed", type=int, default=5000)

    sp = sub.add_parser("eval-car", help="race car: one standing-start run on each unseen test track")
    sp.add_argument("run")
    sp.add_argument("--checkpoint", default="last")
    sp.add_argument("--tracks", type=int, default=48)
    sp.add_argument("--first", type=int, default=16)

    sp = sub.add_parser("eval-tag", help="tag: one best-guess round in each unseen test arena")
    sp.add_argument("run")
    sp.add_argument("--checkpoint", default="last")
    sp.add_argument("--rounds", type=int, default=192)
    sp.add_argument("--first", type=int, default=64)
    sp.add_argument("--opponent", help="a script or runs/x:ckpt (default: the one it trained against)")

    sp = sub.add_parser("tag-table", help="tag: tournament table, every chaser vs every runner (catch %%)")
    sp.add_argument("--chasers", nargs="+", required=True)
    sp.add_argument("--runners", nargs="+", required=True)
    sp.add_argument("--rounds", type=int, default=192)
    sp.add_argument("--first", type=int, default=64)
    sp.add_argument("--out", help="also save the table as json (for footage)")

    sp = sub.add_parser("tag-match", help="tag: any chaser vs any runner (scripts or runs/x:ckpt), unseen arenas")
    sp.add_argument("chaser")
    sp.add_argument("runner")
    sp.add_argument("--rounds", type=int, default=192)
    sp.add_argument("--first", type=int, default=64)

    args = p.parse_args()
    if args.cmd == "play":
        from ailearns import play
        play.run(args.size, args.speed)
    elif args.cmd == "drive":
        from ailearns import drive
        drive.run(args.track)
    elif args.cmd == "tag":
        from ailearns import play_tag
        play_tag.run(args.side, args.vs, args.arena, fog=not args.no_fog, match=args.match)
    elif args.cmd == "train":
        from ailearns.agents.ppo import PPOConfig, train
        cfg = PPOConfig(size=args.size, view=args.view, arch=args.arch, total_steps=int(args.steps), n_envs=args.envs,
                        seed=args.seed, gamma=args.gamma)
        if args.backend == "jax":
            from ailearns.agents.ppo_jax import train as jax_train
            jax_train(cfg, args.run, resume=args.resume, max_hours=args.max_hours)
        else:
            train(cfg, args.run, resume=args.resume, max_hours=args.max_hours, device=args.device)
    elif args.cmd == "train-car":
        from ailearns.agents.ppo_car import CarPPOConfig, train
        train(CarPPOConfig(total_steps=int(args.steps), n_envs=args.envs, seed=args.seed, hidden=args.hidden,
                           ray_range=args.ray_range, messy_starts=args.messy_starts),
              args.run, resume=args.resume, max_hours=args.max_hours)
    elif args.cmd == "train-tag":
        from ailearns.agents.ppo_tag import TagPPOConfig, train
        train(TagPPOConfig(total_steps=int(args.steps), n_envs=args.envs, seed=args.seed, hidden=args.hidden,
                           side=args.side, opponent=args.opponent, closer_reward=args.closer_reward,
                           init_from=args.init_from, remember_heading=args.remember_heading,
                           reset_std=args.reset_std),
              args.run, resume=args.resume, max_hours=args.max_hours)
    elif args.cmd == "train-cqb" and args.es:   # evolution (episode 12: coached, for the hostage)
        from ailearns.agents.es_cqb import CQBESConfig
        from ailearns.agents.es_cqb import train as evolve
        evolve(CQBESConfig(side=args.side, init_from=args.init_from, opponent=args.opponent, time_cost=args.time_cost,
                           flashbangs=tuple(int(x) for x in args.flashbangs.split(",")), magazine=args.magazine,
                           health=args.health, react=args.react, pop=args.pop, rounds=args.rounds, sigma=args.sigma,
                           sigma_throw=args.sigma_throw, generations=args.generations, seed=args.seed,
                           lr_rel=args.lr_rel, stacked=args.stacked, guard=args.guard, aim_bonus=args.aim_bonus,
                           house=args.house, route=args.route, feed=args.feed,
                           teams=tuple(int(x) for x in args.teams.split(",")), bump=args.bump, ready=args.ready,
                           stun=args.stun, share_sight=not args.no_share, split_entry=args.split,
                           radio=tuple(int(x) for x in args.radio.split(",")), pilot=args.pilot,
                           spare_mags=args.spare_mags, time_limit=args.time_limit, blind_away=args.blind_away,
                           hostage=args.hostage, hostage_deep=args.hostage_deep, extract=args.extract,
                           hostage_noise=args.hostage_noise, sigma_hostage=args.sigma_hostage,
                           hostage_bonus=args.hostage_bonus),
               args.run, max_hours=args.max_hours, resume=args.resume)
    elif args.cmd == "train-cqb":
        from ailearns.agents.ppo_cqb import CQBPPOConfig, train
        train(CQBPPOConfig(total_steps=int(args.steps), n_envs=args.envs, seed=args.seed, hidden=args.hidden,
                           side=args.side, opponent=args.opponent, explore_reward=args.explore_reward,
                           time_cost=args.time_cost, init_from=args.init_from, reset_std=args.reset_std,
                           ent_coef=args.ent_coef, buildings=args.buildings,
                           flashbangs=tuple(int(x) for x in args.flashbangs.split(",")), stacked=args.stacked,
                           memory=args.memory, plateau=args.plateau, magazine=args.magazine, health=args.health,
                           react=args.react, teams=tuple(int(x) for x in args.teams.split(",")), bump=args.bump,
                           ready=args.ready, stun=args.stun, share_sight=not args.no_share,
                           radio=tuple(int(x) for x in args.radio.split(",")), split_entry=args.split,
                           listen_coef=args.listen, signal_coef=args.signal, relay=args.relay, intel=args.intel, feed=args.feed, route=args.route, house=args.house, pilot=args.pilot, callout=args.callout, callout_way=args.callout_way, vf_warmup=args.vf_warmup, critic_priv=args.critic_priv,
                           gamma=args.gamma, spare_mags=args.spare_mags, time_limit=args.time_limit,
                           blind_away=args.blind_away, hostage=args.hostage, hostage_deep=args.hostage_deep,
                           extract=args.extract, hostage_noise=args.hostage_noise),
              args.run, resume=args.resume, max_hours=args.max_hours)
    elif args.cmd == "eval-cqb":
        from ailearns.agents.ppo_cqb import evaluate
        r = evaluate(args.run, args.checkpoint, first=args.first, n=args.n, seed=args.seed, opponent=args.opponent)
        print(f"{r['run']} update {r['update']} ({r['moves']:,} moves): won {r['win_rate']:.0%}, trades "
              f"{r['trade_rate']:.0%}, time-outs {r['timeout_rate']:.0%}, saw the enemy in {r['met_rate']:.0%} "
              f"of {r['rounds']} rounds (test buildings {args.first}-{args.first + args.n - 1})")
    elif args.cmd == "watch-cqb":
        from ailearns import record_cqb as R
        from ailearns.render import theme as T
        T.init()
        out = args.out or f"out/cqb-{args.building}.mp4"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        tr = R.rollout(args.assault, args.defense, args.building, seed=args.seed,
                       batch=tuple(int(x) for x in args.batch.split(",")) if args.batch else None)
        R.render_played(tr, out, eyes=args.eyes, slow=args.slow)
        print(f"wrote {out} ({tr['steps'] * tr['cfg'].dt:.1f} s round)")
    elif args.cmd == "eval-tag":
        from ailearns.agents.ppo_tag import evaluate
        r = evaluate(args.run, args.checkpoint, args.rounds, args.first, opponent=args.opponent)
        t = r["catch_time"]
        print(f"{args.run} u{r['update']} ({r['moves'] / 1e6:.0f}M moves): caught {r['catch_rate']:.0%} "
              f"of {r['rounds']} rounds" + (f", in {t:.1f} s on average" if t else ""))
    elif args.cmd == "tag-table":
        import json
        from ailearns.agents.ppo_tag import match
        cells = [[match(c, r, args.rounds, args.first)["catch_rate"] for r in args.runners] for c in args.chasers]
        w = max(map(len, args.chasers))
        print(" " * w + "".join(f"{r[-24:]:>26}" for r in args.runners))
        for c, row in zip(args.chasers, cells):
            print(f"{c:<{w}}" + "".join(f"{x:>26.0%}" for x in row))
        if args.out:
            Path(args.out).write_text(json.dumps({"chasers": args.chasers, "runners": args.runners, "rounds": args.rounds,
                                                  "first": args.first, "catch_rate": cells}, indent=2))
    elif args.cmd == "tag-match":
        from ailearns.agents.ppo_tag import match
        r = match(args.chaser, args.runner, args.rounds, args.first)
        t = r["catch_time"]
        print(f"{args.chaser} vs {args.runner}: caught {r['catch_rate']:.0%} of {args.rounds} rounds"
              + (f", in {t:.1f} s on average" if t else ""))
    elif args.cmd == "eval":
        r = eval_snake(args.run, args.checkpoint, args.games, args.seed)
        print(f"{args.run} u{r['update']}: filled the board in {r['wins']}/{r['games']} games "
              f"({r['wins'] / r['games']:.0%}), mean {r['mean_score']:.1f} apples")
    elif args.cmd == "eval-car":
        import numpy as np
        from ailearns.agents.ppo_car import evaluate
        r = evaluate(args.run, args.checkpoint, args.tracks, args.first)
        laps = list(r["flying_laps"].values())
        print(f"{args.run} u{r['update']}: {r['tracks']} tracks, {r['crashes']} crashes, "
              f"{r['completed_lap']} lapped, mean flying lap {np.mean(laps):.2f} s")


def eval_snake(run: str, checkpoint: str = "last", games: int = 256, seed: int = 5000) -> dict:
    """Best-move games (the highest-scoring move every turn) until each one ends."""
    import numpy as np

    from ailearns.agents.ppo import act, load_policy
    from ailearns.checkpoints import pick_checkpoint
    from ailearns.games.snake import SnakeConfig, SnakeGames
    net, ck = load_policy(pick_checkpoint(run, checkpoint))
    cfg = ck["config"]
    g = SnakeGames(games, SnakeConfig(size=cfg["size"]), seed=seed)
    done, won, score = np.zeros(games, bool), np.zeros(games, bool), np.zeros(games, int)
    while not done.all():
        _, d, w = g.step(act(net, g.observe(cfg["view"])))
        new = d & ~done
        won |= new & w
        score[new] = g.score[new]
        done |= d
    return {"update": ck["update"], "games": games, "wins": int(won.sum()), "mean_score": float(score.mean())}


if __name__ == "__main__":
    main()
