---
title: Beat My AI
emoji: 🏎️
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: mit
short_description: Race an AI driver, or build a car to beat the AI engineer
---

# Beat my AI

Two games from the YouTube channel **A Million Tries**: https://www.youtube.com/@amilliontries

## Race my AI (episodes 4-5)

One lap, standing start, against an AI that taught itself to drive. Its time: **30.3 seconds**.
Its creator, on a keyboard: 34.7.

- **Same game for both of you.** The track, the physics and the ten decisions a second are the
  real game the AI trained in, ported to JavaScript and checked against the original: a recorded
  lap replays to within half a millimetre, and the AI's lap comes out at the same 30.3 s.
- **The green car is the real AI**, running its trained brain (14 inputs: 11 distance lasers,
  its speed and its last move; two layers of 256 neurons; out come steering and a pedal) right
  here in your browser. Press **L** to see its lasers.
- It learned in about half a million tries, on other tracks. It had never seen this one.

**Controls:** arrows or WASD (on a phone: the buttons). R to restart.

## Beat the engineer (episode 14)

In episode 14 an AI engineer designed a racing car under a **$1M price list**. Set the same 13 dials
yourself (engine power and position, front and rear wings, tyre widths, wheelbase, drive layout, brake and
roll balance, gearing, ride height, springs), then your car and the engineer's race for 60 seconds from a
standing start on a circuit neither has seen.

- **The car alone decides.** The same AI driver drives both cars: it learned on random cars of every
  design (300 million steps), never on yours or the engineer's.
- **The real physics.** Four tyres with their own grip, temperature and wear, downforce from the wings and
  the floor (which stalls if it runs too low), weight from every part, fuel, hills and banked corners: the
  episode's simulator, ported to JavaScript and checked against the original (recorded runs replay to
  within half a millimetre over 5 s; the driver gets within 1% of the same distance).
- 12 test circuits, 2-4 km each. The engineer's car isn't unbeatable: on these circuits it's about 1%
  ahead of a plain mid-range car on average, and the winner changes from circuit to circuit.

## Watch the final race (episode 14)

The episode's finalists race each other on any of the 12 test circuits, **each with its own AI driver** (trained on
cars like it), as in the video's final race: the engineer's car ($972k), the sweep's best ($979k) and an F1-like car
that's over the budget ($1,365k). Add the mid-range car or your own from the garage (those get the general driver).
Follow the leader or any car, or watch the whole circuit. In the video, over 56 circuits: the engineer's car 3,554 m
a minute on average, the F1-like car 3,545 m, the sweep's best 3,390 m; here on 12 circuits the engineer's car beats
the sweep's best on all 12 and ties the F1-like car.
