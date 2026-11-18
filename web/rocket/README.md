---
title: Land It Yourself
emoji: 🚀
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: Land a rocket booster from 70 km next to the AI from ep 18
---

# Land it yourself

From episode 18 of **A Million Tries** ("My AI Landed a Rocket From 70 km. Then I Caught It Cheating."). Fly a
Falcon 9 first stage from 70 km to a drone ship at sea, next to the AI (the video's Try 29) flying the same start.

- **Take over at 3 km:** the AI flies the entry burn and the long fall, you fly the landing burn. Start here.
- **From 70 km:** the whole flight, both burns, as in the video.

The engine can't hover (its lowest throttle still pushes harder than the booster weighs). Two lights: the entry burn
(three engines) and the landing burn (one). Land under 3 m/s, under 1 m/s sideways, leaning less than 8 degrees,
with the legs on the 52 m deck. Too much heat, air load (150 kPa) or bending (swinging across the airflow) and it
breaks up.

Controls: ← → engine angle (it leans the booster; the small thrusters move with it), ↑ ↓ throttle, Space engine
on/off (held through a half-second start), 1 2 3 for 1×/4×/10× speed. On a phone: the buttons. **The grid fins** are
flown by an AI copilot (the same network, reading your booster) unless you untick it, then A / D. Why: we measured
that a landing needs the engine angle and the fins moved separately (the AI itself, given one shared steer for both:
0 landings in 100; engine angle + fins: 96), and that's too much at once for two thumbs in 15 seconds.

**The physics is the episode's simulator** (`games/rocket.py`), ported line for line to `engine.js`. `check.mjs`
replays the Python simulator's flights in the browser engine: they agree to 1e-9 for the first ~70 s and end the
same way (`node check.mjs`). The AI is the network from `runs/rocket-ppo29` update 1400 (`data/pilot.json`), flying
live in your browser; on 200 new starts in JS it lands 98.0% (the simulator, 4,000 flights: 98.0%).

- Code: https://github.com/backstab319/a-million-tries/tree/episode-18
- The AI: https://huggingface.co/amilliontries/rocket-booster-ep18
- The channel: https://www.youtube.com/@amilliontries
