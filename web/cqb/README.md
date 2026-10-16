---
title: The Kill House vs My AI
emoji: 🔦
colorFrom: blue
colorTo: red
sdk: static
pinned: false
license: mit
short_description: A dark house vs my AI, 1v1, 2v1 or 2v2 with a hostage
---

# The Kill House vs my AI

A dark house you've never seen. Play the attacker against the trained AI defender from episode 9 of
**A Million Tries**, or hold a room against its trained attacker; from episode 10, hold the room against the AI pair
or attack with the AI as your partner; and from episode 12, two against two with a hostage: rescue it with an AI
partner and walk it out of the house, or hold it with one.

- **The same game the AIs trained in,** ported to JavaScript and checked against the original (`node check.mjs`):
  every reference round under 20 seconds ends the same way (38 of 44 overall; longer ones can drift apart through
  rounding, as they also do between the Python game on a GPU and on a CPU).
- **The hostage (episode 12):** stand beside it for 3 seconds with a clear line to secure it, then walk it out of the
  house; you're slower while you lead it, and if you go down it stays where it is. The defenders know where it is
  until it's moved, after that only while they see it; they hear it being moved (8 m, less through walls). It's
  simplified: it can't be hit.
- **Two vs one and two vs two:** a sighting by anyone on a team is shared, friendly fire is on, and crashing into a
  wall is loud and stuns you for half a second.
- **You see what your light shows.** The house's walls show faintly (the AI feels them with its distance lasers);
  the others only when they're in your side's light. Sounds come as marks around you (grey footsteps, amber doors,
  red shots, cyan: the hostage being moved) and through your speakers.
- **The AIs are real.** Their trained brains (two layers of 512 neurons, eight decisions out) run in your browser,
  ten decisions a second, same as yours. The episode 12 AIs also get the way to the hostage worked out for them (the
  game's route helper, as in the video); you see the house instead.
- These 64 houses are test houses from the videos: no AI ever trained in them.
- **Snapshots.** The game changes between episodes; the modes here are as each episode had them
  ([episode 9's code](https://github.com/backstab319/a-million-tries/tree/episode-9),
  [episode 10's](https://github.com/backstab319/a-million-tries/tree/episode-10),
  [episode 12's](https://github.com/backstab319/a-million-tries/tree/episode-12)).

**Controls:** WASD move · mouse aim · click or Space fire · F door · Shift run · G flashbang · R reload · Esc menu.
Needs a keyboard and a mouse.

From the YouTube channel **A Million Tries**: https://www.youtube.com/@amilliontries

Voices and sounds: ElevenLabs; night ambience: Freesound (CC0).
