---
title: The Kill House vs My AI
emoji: 🔦
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: mit
short_description: Clear a dark house against the AI from episode 9, or hold it
---

# The Kill House vs my AI

One attacker, one defender, a dark house you've never seen. Play the attacker against the trained AI defender from
episode 9 of **A Million Tries**, or hold a room against its trained attacker.

- **The same game, as it was in episode 9.** The rules (the light cone, hearing through walls, doors, rifles with
  magazines and recoil, flashbangs, and in defender mode health and reaction time) are the real game the AIs trained
  in, ported to JavaScript and checked against the original (`node check.mjs`): every reference round under 30
  seconds ends the same way at the same tenth of a second, flashbangs included (17 of 20 overall; the long ones drift
  apart through rounding, as they also do between the Python game on a GPU and on a CPU).
- **You see what your light shows.** The house's walls show faintly (the AI feels them with its distance lasers);
  the other operator only when it's in your light. Sounds come as marks around you (grey footsteps, amber doors,
  red shots) and through your speakers.
- **The AIs are real.** Their trained brains (122-125 numbers in, two layers of 512 neurons, eight decisions out)
  run in your browser, ten decisions a second, same as yours.
- These 64 houses are test houses from the video: no AI ever trained in them.
- **This is a snapshot.** The game changes in later episodes; this page stays the episode 9 version
  ([its code](https://github.com/backstab319/a-million-tries/tree/episode-9)).

**Controls:** WASD move · mouse aim · click or Space fire · F door · Shift run · G flashbang · R reload · Esc menu.
Needs a keyboard and a mouse.

From the YouTube channel **A Million Tries**: https://www.youtube.com/@amilliontries
Models: https://huggingface.co/amilliontries/cqb-defender-ep9 · https://huggingface.co/amilliontries/cqb-attacker-ep9
