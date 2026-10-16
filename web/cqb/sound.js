// The kill house's sound (the episode's own effects and radio), for the game (app.js) and the AI-vs-AI page
// (app3d.js). You hear what one operator (`me`) would: your side's noise, and the enemy's only as loud as it reached
// you; the radio is your side's callouts, each said only when its event really happened.
import { ASSAULT } from "./engine.js";

// radio callouts (scripts/cqb_callouts.py)
const LINES = ["contact", "reloading", "hit", "mandown", "tangodown", "bangout", "movement"];
const SFX = ["gunshot", "door", "reload", "flashbang", "hit", "step", "amb_out", "amb_in", "far_dog", "far_siren",
             ...["assault", "defense"].flatMap((s) => LINES.map((l) => `call_${s}_${l}`)),
             "call_assault_onhostage", "call_assault_secured", "call_defense_onhostage"];   // (ep12) the hostage

// ambience gains (files at -30 LUFS; a near footstep plays at about -41): outside -51, inside -54 plus the muffled outside
const AMB = { out: 0.09, in: 0.065, muffled: 0.12, far: 0.08 };

export class Sound {
  constructor() {
    this.audio = null; this.sfx = {};
    this.lastCall = 0; this.quietSince = 0; this.callAt = {}; this.stepDist = {};
    this.clock = () => performance.now();   // ms; footage (SoundLog) runs on game time instead
  }

  // browsers keep sound off until a click or key press: call init() from one (or call it early and resume() later)
  async init() {
    if (this.audio) return this.resume();
    this.audio = new AudioContext();
    await Promise.all(SFX.map(async (n) => {
      this.sfx[n] = await this.audio.decodeAudioData(await (await fetch(`sfx/${n}.mp3`)).arrayBuffer());
    }));
    this.ambience();
  }

  // the night under everything, well below the faintest footstep you can hear (the files are all -30 LUFS): outside,
  // crickets and a far road; inside, a vent's hum with the outside muffled through the walls. place() crossfades.
  // Now and then something far off (a dog, a siren): nothing a player could take for a step, a door or a voice.
  ambience() {
    const a = this.audio, loop = (name) => {
      const s = a.createBufferSource(); s.buffer = this.sfx[name]; s.loop = true;
      s.start(0, Math.random() * s.buffer.duration); return s;
    };
    const gain = (v) => { const g = a.createGain(); g.gain.value = v; g.connect(a.destination); return g; };
    const out = loop("amb_out"), muffle = a.createBiquadFilter();
    muffle.type = "lowpass"; muffle.frequency.value = 600;
    this.amb = { out: gain(AMB.out), muffled: gain(0), in: gain(0) };
    out.connect(this.amb.out); out.connect(muffle).connect(this.amb.muffled);
    loop("amb_in").connect(this.amb.in);
    this.inside = false;
    const far = () => {
      setTimeout(far, 25000 + Math.random() * 35000);
      if (this.on) this.play(Math.random() < 0.6 ? "far_dog" : "far_siren", (this.inside ? 0.5 : 1) * AMB.far / 0.6);
    };
    setTimeout(far, 10000 + Math.random() * 20000);
  }
  place(inside) {   // where you're listening from: inside the house or out in the yard (eased over half a second)
    if (!this.amb || inside === this.inside) return;
    this.inside = inside;
    const t = this.audio.currentTime, set = (g, v) => g.gain.setTargetAtTime(v, t, 0.15);
    set(this.amb.out, inside ? 0 : AMB.out); set(this.amb.muffled, inside ? AMB.muffled : 0); set(this.amb.in, inside ? AMB.in : 0);
  }
  at(house, p) {   // listening from game point p: inside the building's footprint or not
    this.place(p[0] > house.footLo[0] && p[0] < house.footHi[0] && p[1] > house.footLo[1] && p[1] < house.footHi[1]);
  }
  resume() { if (this.audio?.state === "suspended") this.audio.resume(); }
  get on() { return this.audio?.state === "running"; }

  play(name, vol, rate = 1) {
    const { audio, sfx } = this;
    if (!audio || !sfx[name] || vol <= 0.02) return;
    const s = audio.createBufferSource(), v = audio.createGain();
    s.buffer = sfx[name]; s.playbackRate.value = rate;
    v.gain.value = Math.min(1, vol) * 0.6; s.connect(v).connect(audio.destination); s.start();
  }

  say(game, t, w, line) {   // operator w says it on the radio (one at a time, each line not too often)
    const now = this.clock();
    if (now - this.lastCall < 900 || now - (this.callAt[line + w] || -1e9) < 4000) return;
    this.lastCall = now; this.callAt[line + w] = now;
    (globalThis.cqbCalls ??= []).push([t, w, line]);   // for automated checks
    this.play(`call_${game.team[w] === ASSAULT ? "assault" : "defense"}_${line}`, 1.2);
  }

  // what `me` hears from decision a to b. `all`: a spectator watching everyone hears every operator as if close
  // (the radio stays `team`'s).
  hear(game, a, b, me, team, foes, all = false) {
    const heard = (kind) => b.sound[me][kind][2], say = (w, line) => this.say(game, b.t, w, line);
    for (const w of game.ops) {
      const ours = all || team.includes(w), near = w === me ? 1 : 0.6;
      if (b.fired[w]) this.play("gunshot", ours ? near : 0.25 + 0.75 * heard(2));
      if (b.doorEvent[w] && (ours || heard(1) > 0)) this.play("door", ours ? 0.7 * near : 0.3 + 0.7 * heard(1));
      if (b.reloadNow[w] && (ours || b.reloadHeard[me][2] > 0)) this.play("reload", ours ? 0.8 * near : 0.3 + 0.7 * b.reloadHeard[me][2]);
      if (b.bangNow[w]) this.play("flashbang", ours ? near : 0.3 + 0.7 * b.bangHeard[me][2]);
      if ((a.alive[w] && !b.alive[w]) || b.hp[w] < a.hp[w]) this.play("hit", w === me ? 1 : 0.7);
      if (b.bumped && b.bumped[w] && (ours || heard(0) > 0)) this.play("door", ours ? 0.5 * near : 0.2 + 0.5 * heard(0));
      // footsteps every ~0.7 m: yours and your partner's always, the enemy's only as loud as you hear them
      if (b.alive[w]) {
        this.stepDist[w] = (this.stepDist[w] || 0) + Math.hypot(b.pos[w][0] - a.pos[w][0], b.pos[w][1] - a.pos[w][1]);
        if (this.stepDist[w] > 0.7) {
          this.stepDist[w] = 0;
          this.play("step", ours ? (w === me ? 0.35 : 0.22) : 0.5 * heard(0), 0.9 + Math.random() * 0.2);
        }
      }
    }
    // the radio: your side's callouts, each only when its event really happened this decision
    const up = team.filter((k) => b.alive[k]), anyUp = (pref) => (b.alive[pref] ? pref : up[0]);
    for (const k of team) {
      if (b.alive[k] && b.hp[k] < a.hp[k]) say(k, "hit");
      if (b.alive[k] && b.reloadNow[k]) say(k, "reloading");
      if (b.bangs[k] < a.bangs[k]) say(k, "bangout");
      if (a.alive[k] && !b.alive[k] && up.length) say(up.find((x) => x !== k) ?? up[0], "mandown");
    }
    for (const j of foes) {
      if (a.alive[j] && !b.alive[j] && up.length) say(anyUp(me), "tangodown");
      const spotter = up.find((k) => b.seen[k][j]);
      if (spotter !== undefined && b.since[me][j] === 0 && a.since[me][j] >= 20) say(spotter, "contact");
    }
    if (game.cfg.hostage && b.hostage && up.length) {   // (ep12) the hostage
      const by = (w) => Math.hypot(b.pos[w][0] - b.hostage[0], b.pos[w][1] - b.hostage[1]) <= game.cfg.secure_radius;
      if (game.team[team[0]] === ASSAULT) {   // the count started: whoever is beside it says so; it won: it's secure
        if (b.secure > 0 && !(a.secure > 0)) say(team.find((k) => b.alive[k] && by(k)) ?? up[0], "onhostage");
        if (b.done && b.outcome === 1 && foes.some((j) => b.alive[j])) say(anyUp(me), "secured");
      } else {   // Defense knows only what it sees: one of them sees an attacker at the hostage
        const spotter = up.find((k) => foes.some((j) => b.alive[j] && b.seen[k][j] && by(j)));
        if (spotter !== undefined) say(spotter, "onhostage");
      }
    }
    const seeing = foes.some((j) => b.alive[j] && b.since[me][j] === 0);
    if (seeing) this.quietSince = b.t;
    if (up.length && !seeing && heard(0) > 0 && a.sound[me][0][2] <= 0 && b.t - this.quietSince > 50) {
      say(anyUp(me), "movement"); this.quietSince = b.t;
    }
  }
}

// (ep12) Footage: the same rules as Sound, but every sound is written down with its game time instead of played
// (capture runs slower than real time). scripts/cqb3d_audio.py mixes the log into the video's soundtrack with the
// same files and gains. `t`: the game time (s) of the decision being heard; set before each hear().
export class SoundLog extends Sound {
  constructor() { super(); this.log = []; this.t = 0; this.clock = () => this.t * 1000; this.inside = null; }
  get on() { return true; }
  play(name, vol, rate = 1) { if (vol > 0.02) this.log.push([+this.t.toFixed(3), name, +Math.min(1, vol).toFixed(3), +rate.toFixed(3)]); }
  place(inside) { if (inside !== this.inside) { this.inside = inside; this.log.push([+this.t.toFixed(3), "@inside", inside ? 1 : 0, 1]); } }
}
