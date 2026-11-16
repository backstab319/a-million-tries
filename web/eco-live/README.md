---
title: A World With No Rewards
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: mit
short_description: Grow a world of animals whose brains evolve with no reward
---

# A world with no rewards

Animals with bodies that eat, drink, keep warm, breed and die, and brains that start random. Nothing ever tells a
brain what's good: a brain only gets passed on if its animal lives long enough to breed. From episode 17 of
[A Million Tries](https://www.youtube.com/@amilliontries), where 15 of 16 such worlds died of thirst.

- **Start from scratch:** 1,000 grass-eaters with random brains, at midwinter, as in the episode.
- **Start with the survivors:** the 134 animals of the one world still alive after 25 years (9,000 sim days, run on
  a GPU), dropped into new land. Their brains and genes are the
  [ecosystem-survivors-ep17](https://huggingface.co/amilliontries/ecosystem-survivors-ep17) model.
- **Give everyone random brains:** the episode's control experiment. Same bodies, same place, random brains: watch
  what the evolved brains were worth.
- Click an animal to follow it: its thirst, its energy, and what its brain is telling it to do.

**It's the real sim.** The land, the grass and the rain, the bodies (real scaling laws for metabolism, walking,
growth, keeping warm; thirst; milk; carried young; fights) and the brains (33 senses -> 8 -> 8 outputs) are the
episode's Python code ported to JavaScript line by line, and checked against it on Python's own dice: two reference
worlds (a winter with grass-eaters; an autumn with every diet, fights, falling fruit and copies of the survivors)
run a whole sim day with every birth, death, kill and random draw the same. Positions stay within a micrometre
(the check re-syncs every minute of sim time, because random brains are chaotic: a 1e-15 difference between maths
libraries grows tenfold every 10 seconds). The energy books balance to ~1e-16.

Animals are drawn bigger than life so you can see them. A big world runs slower: every animal senses, thinks and
pays for every second.

Code: [github.com/backstab319/a-million-tries](https://github.com/backstab319/a-million-tries) (`games/ecosystem*.py`).
