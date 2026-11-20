---
title: Dogfight My AI
emoji: ✈️
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: mit
short_description: Fly an F-16 against the AI that taught itself to dogfight
---

# Dogfight my AI

From episode 21 of **A Million Tries** ("I Taught an AI to Dogfight. Fighting Itself Made It Worse."). Two F-16s,
guns only. You fly blue; red is the video's AI (self-play run 3), which learned to fight by fighting itself. In the
episode it beat its own previous version 57% to 14%.

Controls (keyboard): S or ↓ pull (up to 9 g), W or ↑ push, A D or ← → roll, Shift / Ctrl throttle (the top quarter
is afterburner), Space fire (511 rounds), P pause. Put the yellow circle on the red jet and fire: it marks where your
shells will be when they reach its range. A fight ends when a jet is shot down or flies into the ground, or after
two minutes.

- **The jet** is a 3-D point-mass flight model built from public F-16 figures (mass, wing area, thrust, drag, the 9 g
  and 25° angle-of-attack limits, the M61 cannon), checked against published turn rates, corner speed and top
  speed. The stick is the real jet's fly-by-wire one: it asks for g and a roll rate, and the computer won't pull more
  than the wings can make. Not modelled: the jet's rotation about its own axes, sideslip, fuel, blackouts, missiles.
- **The AI** is the episode's network (38 senses -> 256 -> 256 -> 4), flown as in training: its stick with a little
  trained randomness, its trigger from a learned firing probability. It sees what the sim gives it, nothing else.
- **Same physics as the video:** `sim.js` is ported line by line from the Python sim, and `check.mjs` replays Python
  fights in it (worst position error 5 mm over 838 steps; the network's outputs within 5e-6).
- Jets are drawn 3x real size so you can see them, and the smoke is added so you can follow the fight.

Code: https://github.com/backstab319/a-million-tries (tag `episode-21`). Sounds: ElevenLabs.
