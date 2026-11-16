---
title: Beat My AI
emoji: 🏎️
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: mit
short_description: Race an AI, beat the AI engineer, AI race day + Safety Car
---

# Beat my AI

Games from the YouTube channel **A Million Tries**: https://www.youtube.com/@amilliontries

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

## Watch race day (episode 15)

The episode's final race, replayed: **8 cars, 25 laps** (about 45 minutes) on test circuit 100003, with contact,
damage, tyre wear, a fuel limit and a pit lane. **One AI driver in every car**
([race-day-driver-ep15](https://huggingface.co/amilliontries/race-day-driver-ep15)). The pit calls are the contest:
cars 1-4 have **the AI pit wall** ([race-pit-wall-ep15](https://huggingface.co/amilliontries/race-pit-wall-ep15)),
which learned when to stop and which tyre to fit; cars 5-8 have **the rule** (box when the worst tyre is 60% worn).

- **A replay, not a simulation:** every car every 0.2 s from the Python race, drawn like the video's TV view: the
  map, a battle camera on the closest fight (or the leader), and the timing tower (tyres, fuel, stops, "BOX, BOX").
- Play at 1x to 60x, scrub anywhere, tap a car in the tower to follow it.
- One race is noisy. Over 128 races on circuits it never trained on, the AI pit wall's cars averaged **P4.08** and
  the rule's **P4.92**.

## Be the pit wall (episode 15)

Run car 1's strategy in a live race against **the AI pit wall** (cars 2-4) and **the rule** (cars 5-8). The same AI
driver drives all 8 cars; only the pit calls differ. Pick a circuit and car 1's starting tyre, then once a lap, as
your car nears the pit entry, the race pauses and asks: **box or stay out, and which tyres** (or set a standing order
and let it run).

- **Live, not a replay:** the whole race simulation (contact, damage, dirty air, tyre wear, fuel, the pit lane) runs
  in your browser, ported from the episode's Python and checked against it (the same senses from Python snapshots;
  whole races land on the same positions on average).
- **New random cars every race**, as in the 128-race comparison (or pick the video race's 8 cars). The number to
  beat: the AI pit wall's cars averaged **P4.08** over 128 races, the rule's **P4.92**.
- The race ends 3 minutes after the winner takes the flag; everyone else is classified where they are.
- It's heavy: about 7x real time on a laptop (a 25-lap race in ~6-7 minutes), faster on a desktop. "Just watch"
  runs the AI pit wall v the rule with nobody at the wall.

## Race control (episode 16)

Episode 16's race, live: the same contest (you, or nobody, on car 1's pit wall; **the AI pit wall** on cars 2-4,
**the rule** on 5-8), now with **race control**: yellow and double yellow flags by marshalling sector, the Virtual
Safety Car, the Safety Car (the white car), red flags, the closed pit entry and **the stewards** (5 s for a pass under
yellow, the VSC or the Safety Car not given back, a sector faster than the neutralised minimum time, speeding under a
double yellow or a collision caused in the pit lane; 10 s for entering a closed pit lane; disqualification for one tyre compound all
race or five offences). The final order is the stewards'.

- **Episode 16's brains:** the driver `race-control-driver-ep16` (D1-8:8: 130 senses, including the flags, the
  Safety Car and "a stopped car ahead"; it learned to drive under the flags from demonstrations) and the pit wall
  `race-pit-wall-ep16` (strat-ep16-1:91), on episode 16's two-lane pit lane.
- **Breakdowns:** none, 1 or 2 a race (a random car stops dead on the track, or its engine dies), at a random moment
  between minutes 2 and 40. Without them a Safety Car comes out about one race in five; the pit wall trained and was
  tested with 2.
- **When the VSC or the Safety Car comes out**, the AI pit wall gets a second call, and so do you.
- **Honest about the driver:** it still crashes (in testing about 4 cars in 10 don't finish), the Safety Car waits for
  its call in the pit lane's fast lane, and a car pinned in a contact can keep its speed: the gaps the video shows.
- The number to beat: over 256 races on 64 circuits it never trained on (2 breakdowns each), the AI pit wall's cars
  averaged **P4.41**, the rule's **P4.59**; cars 1-4 start with a handicap (the rule itself in those seats: P4.57).
