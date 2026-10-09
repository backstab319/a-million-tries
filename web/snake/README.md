---
title: Beat My AI at Snake
emoji: 🐍
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: mit
short_description: Play Snake next to an AI that fills the board 92% of games
---

# Beat my AI: Snake

Your board on the left, the AI's on the right. Same 10x10 board, same rules, same speed.
The AI fills the **whole board in 92% of its games** (236 of 256 games it had never seen).

- **The AI is the real trained brain**, running in your browser: it sees a 19x19 window around
  its head (turned to face where it's going), passes it through 5 convolution layers, and
  picks straight, left or right. It learned in 100 million moves.
- **Same game for both of you**: the rules are a JavaScript port of the game it trained in,
  checked against the original by replaying one of its real games: all 831 moves match, and it
  fills the board the same way.
- Starving counts too: go too long without an apple and the game ends (for both of you).

**Controls:** arrows, WASD or swipe. 1-3 change the speed. After you're out, F fast-forwards the AI.

From the YouTube channel **A Million Tries**: https://www.youtube.com/@amilliontries
