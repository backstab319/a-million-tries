"""Two jet fighters, guns only, in JAX (episode 21; plan: docs/dogfight-plan.md).

A 3-D point-mass flight model of a clean F-16C: lift, drag with the transonic rise, thrust that falls with air
density, gravity, the standard atmosphere. The controls are the real jet's fly-by-wire ones: the stick asks for a
load factor (+9 / -3 g) and a roll rate, and the computer limits angle of attack to 25°, so at low speed the wings
can't make the g asked for (the stall). Bullets fly as packets with gravity and drag; they hit if they pass close
to the other jet, so lead has to be pulled.

Pure functions of a `State` holding both planes (axis 0 = plane 0, plane 1); `jax.vmap` makes thousands of
fights, `jax.jit` compiles them. No rendering here: training runs headless.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.games.rocket_jax import us76

G = 9.80665
ALIVE, SHOT, CRASHED = 0, 1, 2
N_ACTIONS = 4   # g command, roll-rate command, throttle, trigger; each in [-1, 1]
Z = np.array([0.0, 0.0, 1.0], np.float32)


@dataclass(frozen=True)
class FighterConfig:
    dt: float = 0.05                 # s, 20 steps a second
    time_limit: float = 120.0        # s, then a draw
    # the jet (devlog step 1 has the sources)
    mass: float = 10_500.0           # kg: empty 8,570 + pilot + half internal fuel + gun ammo
    wing_area: float = 27.87         # m^2
    cl_alpha: float = 3.5            # lift slope, per rad
    alpha_max: float = float(np.radians(25.0))   # the flight computer's angle-of-attack limit
    alpha_min: float = float(np.radians(-8.0))
    cd0: float = 0.0175              # zero-lift drag, subsonic
    k_induced: float = 0.125         # 1 / (pi * aspect 3.2 * e 0.8)
    thrust_idle: float = 4_000.0     # N, sea level static
    thrust_mil: float = 76_300.0     # N, full power without afterburner (F110-GE-129)
    thrust_max: float = 129_000.0    # N, full afterburner
    spool: float = 1.5               # s, engine time constant
    g_max: float = 9.0
    g_min: float = -3.0
    g_lag: float = 0.2               # s, g onset time constant
    roll_rate_max: float = float(np.radians(240.0))
    roll_lag: float = 0.15           # s
    roll_q: float = 10_000.0         # Pa: below this dynamic pressure roll authority fades
    # the gun: M61A1 20 mm
    rounds: float = 511.0
    rate: float = 100.0              # rounds/s (6,000 a minute)
    muzzle_speed: float = 1_036.0    # m/s
    dispersion: float = 0.003        # rad, the spread of the stream (1 sigma)
    bullet_drag: float = 4.7e-4      # 1/m at sea level: rho * Cd * A / (2 m) for a 100 g, 20 mm round, Cd 0.25
    bullet_life: float = 2.0         # s
    target_radius: float = 4.0       # m, an F-16 seen from any side, roughly
    hits_to_kill: float = 10.0
    # where fights start
    alt_lo: float = 3_000.0
    alt_hi: float = 7_000.0

    @property
    def weight(self):
        return self.mass * G

    @property
    def n_packets(self):
        return int(round(self.bullet_life / self.dt))


class State(NamedTuple):
    pos: jax.Array      # (2, 3) m, z up
    vel: jax.Array      # (2, 3) m/s
    up: jax.Array       # (2, 3) unit lift direction, perpendicular to vel
    load: jax.Array     # (2,) g the wings pull now
    roll: jax.Array     # (2,) rad/s now
    power: jax.Array    # (2,) engine: 0 idle, 0.75 full military, 1 full afterburner
    health: jax.Array   # (2,)
    ammo: jax.Array     # (2,) rounds left
    cause: jax.Array    # (2,) int32 ALIVE, SHOT or CRASHED
    ppos: jax.Array     # (2, K, 3) bullet packets in flight
    pvel: jax.Array     # (2, K, 3)
    prounds: jax.Array  # (2, K) rounds in each packet (0 = empty)
    page: jax.Array     # (2, K) s since fired
    prho: jax.Array     # (2, K) air density where it was fired (for its drag)
    time: jax.Array     # ()
    n: jax.Array        # () int32 steps
    # the scripted target's manoeuvre (used when plane 1 is flown by `scripted_action`)
    man_kind: jax.Array  # () int32: 0 straight, 1 level turn, 2 weave, 3 break turn when threatened
    man_g: jax.Array     # () g of its turns
    man_dir: jax.Array   # () +1 / -1
    man_alt: jax.Array   # () m it holds
    man_period: jax.Array  # () s between weave reversals


# ── small vector helpers (last axis = xyz) ──────────────────────────────────

def _dot(a, b):
    return (a * b).sum(-1)


def _norm(a):
    return jnp.sqrt(_dot(a, a))


def _unit(a, eps=1e-6):
    return a / jnp.maximum(_norm(a), eps)[..., None]


def _smooth(x, a, b):
    s = jnp.clip((x - a) / (b - a), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _rotate(u, k, angle):
    """Rotate u (perpendicular to unit k) about k by angle (right hand)."""
    return u * jnp.cos(angle)[..., None] + jnp.cross(k, u) * jnp.sin(angle)[..., None]


class Fighters:
    """Functions for one fight (two jets); vmap them for many."""

    def __init__(self, config: FighterConfig | None = None):
        self.cfg = config or FighterConfig()

    # ── the airframe and engine ─────────────────────────────────────────────

    def cd0(self, mach):
        c = self.cfg
        return c.cd0 * (1.0 + 2.2 * _smooth(mach, 0.85, 1.1) - 0.8 * _smooth(mach, 1.2, 2.0))  # wave drag (devlog step 1)

    def k(self, mach):
        return self.cfg.k_induced * (1.0 + 1.0 * _smooth(mach, 0.9, 1.4))

    def thrust(self, power, rho, mach):
        c = self.cfg
        sigma = rho / 1.225
        dry = c.thrust_idle + (c.thrust_mil - c.thrust_idle) * jnp.clip(power / 0.75, 0.0, 1.0)
        wet = (c.thrust_max - c.thrust_mil) * jnp.clip((power - 0.75) / 0.25, 0.0, 1.0)
        return dry * sigma ** 0.8 * (1.0 + 0.15 * mach) + wet * sigma ** 0.7 * (1.0 + 0.3 * mach)

    def load_limits(self, q):
        """The g the wings can make at this dynamic pressure, inside the flight computer's limits."""
        c = self.cfg
        qs = q * c.wing_area / c.weight
        return jnp.maximum(c.g_min, qs * c.cl_alpha * c.alpha_min), jnp.minimum(c.g_max, qs * c.cl_alpha * c.alpha_max)

    def excess_power(self, alt, speed, load, power=1.0):
        """Specific excess power (m/s): how fast the jet gains energy height at this speed, height and g. 0 = the
        sustained turn. The same forces as `step`."""
        c = self.cfg
        rho, _, a = us76(alt)
        q = 0.5 * rho * speed ** 2
        mach = speed / a
        cl = load * c.weight / (q * c.wing_area)
        drag = q * c.wing_area * (self.cd0(mach) + self.k(mach) * cl ** 2)
        return (self.thrust(power, rho, mach) - drag) * speed / c.weight

    # ── a fight ─────────────────────────────────────────────────────────────

    def init(self, key, level=1.0) -> State:
        """Plane 1 (the target) flies level; plane 0 starts behind it at level 0, anywhere within ~4 km at level 1."""
        c = self.cfg
        k = jax.random.split(key, 13)
        u = lambda i, lo=0.0, hi=1.0: jax.random.uniform(k[i], (), minval=lo, maxval=hi)  # noqa: E731
        alt = u(0, c.alt_lo, c.alt_hi)
        psi = u(1, -np.pi, np.pi)
        fwd1 = jnp.stack([jnp.cos(psi), jnp.sin(psi), 0.0])
        v1 = u(2, 200.0, 260.0)
        # where plane 0 is: range, angle off the target's tail, height difference
        rng = 600.0 + level * u(3, 0.0, 3400.0)
        off = level * u(4, -np.pi, np.pi)
        tail = jnp.stack([-jnp.cos(psi + off), -jnp.sin(psi + off), 0.0])
        p0 = jnp.stack([0.0, 0.0, alt]) + rng * tail + jnp.stack([0.0, 0.0, level * u(5, -1000.0, 1000.0)])
        head0 = psi + level * u(6, -np.pi, np.pi) + (1 - level) * 0.05 * u(7, -1.0, 1.0)
        fwd0 = jnp.stack([jnp.cos(head0), jnp.sin(head0), 0.0])
        v0 = u(8, 200.0, 280.0)
        K = c.n_packets
        return State(
            pos=jnp.stack([p0, jnp.stack([0.0, 0.0, alt])]),
            vel=jnp.stack([fwd0 * v0, fwd1 * v1]),
            up=jnp.tile(jnp.asarray(Z), (2, 1)),
            load=jnp.ones(2), roll=jnp.zeros(2), power=jnp.full(2, 0.75),
            health=jnp.ones(2), ammo=jnp.full(2, c.rounds), cause=jnp.zeros(2, jnp.int32),
            ppos=jnp.zeros((2, K, 3)), pvel=jnp.zeros((2, K, 3)), prounds=jnp.zeros((2, K)),
            page=jnp.zeros((2, K)), prho=jnp.ones((2, K)),
            time=jnp.float32(0.0), n=jnp.int32(0),
            # (2d) the ladder phases the target's manoeuvres in: straight at level 0, any of the four at level 1
            man_kind=jnp.where(u(11) < level, jax.random.randint(k[9], (), 0, 4), 0), man_g=u(10, 2.0, 6.0),
            man_dir=jnp.where(u(12) < 0.5, -1.0, 1.0), man_alt=alt, man_period=8.0 + 4.0 * u(7))

    def nose(self, st: State):
        """Where each jet points: the velocity tilted up by the angle of attack."""
        c = self.cfg
        rho, _, _ = us76(st.pos[:, 2])
        q = 0.5 * rho * _dot(st.vel, st.vel)
        alpha = st.load * c.weight / (jnp.maximum(q, 1.0) * c.wing_area * c.cl_alpha)
        alpha = jnp.clip(alpha, c.alpha_min, c.alpha_max)
        vhat = _unit(st.vel)
        return vhat * jnp.cos(alpha)[:, None] + st.up * jnp.sin(alpha)[:, None], alpha

    def step(self, st: State, action):
        """action (2, 4) in [-1, 1] -> (state, reward (2,), done). Reward: damage dealt - damage taken."""
        c = self.cfg
        dt = c.dt
        a = jnp.clip(action, -1.0, 1.0)
        alive = st.cause == ALIVE

        # 1. Air, speed, what the wings can do.
        alt = st.pos[:, 2]
        rho, _, sound = us76(alt)
        speed = _norm(st.vel)
        vhat = _unit(st.vel)
        q = 0.5 * rho * speed ** 2
        mach = speed / sound
        g_lo, g_hi = self.load_limits(q)

        # 2. The flight computer follows the stick, the engine spools.
        g_cmd = jnp.where(a[:, 0] >= 0, a[:, 0] * c.g_max, -a[:, 0] * c.g_min)
        load = st.load + (jnp.clip(g_cmd, g_lo, g_hi) - st.load) * (1.0 - np.exp(-dt / c.g_lag))
        load = jnp.clip(load, g_lo, g_hi)
        roll_cmd = a[:, 1] * c.roll_rate_max * jnp.clip(q / c.roll_q, 0.0, 1.0)
        roll = st.roll + (roll_cmd - st.roll) * (1.0 - np.exp(-dt / c.roll_lag))
        power = st.power + ((a[:, 2] + 1.0) / 2.0 - st.power) * (1.0 - np.exp(-dt / c.spool))

        # 3. Forces: lift along `up`, drag against the airflow, thrust along the nose, gravity.
        lift = load * c.weight
        cl = lift / (jnp.maximum(q, 1.0) * c.wing_area)
        alpha = jnp.clip(cl / c.cl_alpha, c.alpha_min, c.alpha_max)
        drag = q * c.wing_area * (self.cd0(mach) + self.k(mach) * cl ** 2)
        thrust = self.thrust(power, rho, mach)
        nose = vhat * jnp.cos(alpha)[:, None] + st.up * jnp.sin(alpha)[:, None]
        force = lift[:, None] * st.up - drag[:, None] * vhat + thrust[:, None] * nose
        acc = force / c.mass - G * jnp.asarray(Z)
        vel = st.vel + acc * dt
        pos = st.pos + vel * dt

        # 4. Roll the lift direction about the flight path, keep it perpendicular to the new velocity.
        up = _rotate(st.up, vhat, roll * dt)
        vhat_new = _unit(vel)
        up_new = _unit(up - _dot(up, vhat_new)[:, None] * vhat_new)
        up = jnp.where((_norm(vel) > 1.0)[:, None], up_new, st.up)

        # 5. The gun: fire a packet into this step's slot of the ring; move the packets already flying.
        K = c.n_packets
        slot = st.n % K
        fire = alive & (a[:, 3] > 0) & (st.ammo > 0)
        shot = jnp.where(fire, jnp.minimum(st.ammo, c.rate * dt), 0.0)
        muzzle = st.pos + nose * 8.0
        bvel = st.vel + nose * c.muzzle_speed
        ppos = st.ppos.at[:, slot].set(jnp.where(fire[:, None], muzzle, st.ppos[:, slot]))
        pvel = st.pvel.at[:, slot].set(jnp.where(fire[:, None], bvel, st.pvel[:, slot]))
        prounds = st.prounds.at[:, slot].set(shot)
        page = st.page.at[:, slot].set(0.0)
        prho = st.prho.at[:, slot].set(rho)
        # hits: the closest approach of each packet to the OTHER jet during this step, both moving straight
        tgt_pos, tgt_vel = st.pos[::-1], st.vel[::-1]
        r0 = ppos - tgt_pos[:, None]                    # (2, K, 3)
        w = pvel - tgt_vel[:, None]
        tc = -_dot(r0, w) / jnp.maximum(_dot(w, w), 1e-6)
        passing = (tc >= 0.0) & (tc < dt) & (prounds > 0)
        miss = _norm(r0 + w * jnp.clip(tc, 0.0, dt)[..., None])
        spread = c.dispersion * c.muzzle_speed * (page + tc)
        s2 = c.target_radius ** 2 + spread ** 2
        hits = jnp.where(passing, prounds * c.target_radius ** 2 / s2 * jnp.exp(-miss ** 2 / (2 * s2)), 0.0)
        prounds = jnp.where(passing, 0.0, prounds)
        damage_to = hits.sum(-1)[::-1] / c.hits_to_kill   # damage each jet takes
        # bullets fly on: drag (slows them through the air) and gravity
        bspeed = _norm(pvel)
        pvel = pvel - (c.bullet_drag * prho / 1.225 * bspeed)[..., None] * pvel * dt - G * jnp.asarray(Z) * dt
        ppos = ppos + pvel * dt
        page = page + dt
        prounds = jnp.where(page > c.bullet_life, 0.0, prounds)

        # 6. Damage, the ground, the end.
        health = jnp.where(alive, jnp.maximum(st.health - damage_to, 0.0), st.health)
        crashed = alive & (pos[:, 2] <= 0.0)
        health = jnp.where(crashed, 0.0, health)
        cause = jnp.where(~alive, st.cause, jnp.where(crashed, CRASHED, jnp.where(health <= 0.0, SHOT, ALIVE)))
        lost = st.health - health
        reward = lost[::-1] - lost
        time = st.time + dt
        done = (cause != ALIVE).any() | (time >= c.time_limit - 1e-6)
        new = st._replace(pos=pos, vel=vel, up=up, load=load, roll=roll, power=power, health=health,
                          ammo=st.ammo - shot, cause=cause, ppos=ppos, pvel=pvel, prounds=prounds, page=page,
                          prho=prho, time=time, n=st.n + 1)
        return new, reward, done

    # ── what a pilot sees ───────────────────────────────────────────────────

    def bank(self, st: State):
        """Bank angle of each jet: the lift direction's tilt from vertical, about the flight path (rad)."""
        vhat = _unit(st.vel)
        zp = _unit(jnp.asarray(Z) - _dot(jnp.asarray(Z), vhat)[:, None] * vhat)
        return jnp.arctan2(_dot(jnp.cross(zp, st.up), vhat), _dot(zp, st.up))

    def observe(self, st: State, i: int):
        """Plane i's senses (n_obs,), in its own body frame (nose f, right wing r, canopy u)."""
        c = self.cfg
        j = 1 - i
        nose, alpha = self.nose(st)
        f = nose[i]
        u = _unit(st.up[i] - _dot(st.up[i], f) * f)
        r = jnp.cross(f, u)
        frame = lambda v: jnp.stack([_dot(v, f), _dot(v, r), _dot(v, u)])  # noqa: E731
        p, v = st.pos[i], st.vel[i]
        speed = _norm(v)
        _, _, sound = us76(p[2])
        d = st.pos[j] - p
        dist = _norm(d)
        dhat = d / jnp.maximum(dist, 1.0)
        w = st.vel[j] - v
        # the gunsight: where a burst fired now would pass the target, if both flew straight
        bv = v + f * c.muzzle_speed - st.vel[j]
        tof = jnp.clip(_dot(d, bv) / jnp.maximum(_dot(bv, bv), 1.0), 0.0, c.bullet_life)
        pip = frame(bv * tof - d)
        own = jnp.stack([speed / 300.0, speed / sound, p[2] / 5000.0, v[2] / 100.0, f[2], u[2], r[2],
                         st.load[i] / 9.0, alpha[i] / c.alpha_max, st.power[i], st.ammo[i] / c.rounds,
                         st.health[i], (p[2] + speed ** 2 / (2 * G)) / 8000.0])
        other = jnp.concatenate([
            frame(d) / 1000.0, jnp.stack([dist / 1000.0, jnp.log1p(dist / 100.0)]), frame(dhat), frame(w) / 300.0,
            frame(nose[j]), jnp.stack([_norm(st.vel[j]) / 300.0, d[2] / 1000.0, -_dot(nose[j], dhat),
                                       -_dot(w, dhat) / 300.0, st.health[j], st.ammo[j] / c.rounds]),
            pip / 50.0, jnp.stack([tof, 1.0 - st.time / c.time_limit])])
        return jnp.concatenate([own, other])

    # ── a scripted target ───────────────────────────────────────────────────

    def scripted_action(self, st: State, i: int = 1):
        """Plane i flies its manoeuvre: straight, a level turn, a weave, or a break turn when the other jet is
        behind it within 1.5 km. Holds its height and ~240 m/s. Never fires."""
        c = self.cfg
        j = 1 - i
        v = st.vel[i]
        speed = _norm(v)
        phi = self.bank(st)[i]
        d = st.pos[j] - st.pos[i]
        behind = (_dot(d, v) < 0) & (_norm(d) < 1500.0)
        weave_dir = jnp.where(jnp.floor(st.time / st.man_period) % 2 == 0, 1.0, -1.0) * st.man_dir
        kind = st.man_kind
        turn_g = jnp.where(kind == 3, jnp.where(behind, c.g_max, 1.0), jnp.where(kind == 0, 1.0, st.man_g))
        side = jnp.where(kind == 2, weave_dir, st.man_dir)
        phi_want = side * jnp.arccos(jnp.clip(1.0 / turn_g, -1.0, 1.0))
        err = jnp.arctan2(jnp.sin(phi_want - phi), jnp.cos(phi_want - phi))
        hold = 1.0 + 0.002 * (st.man_alt - st.pos[i, 2]) - 0.03 * v[2]
        g_want = jnp.clip(hold / jnp.maximum(jnp.cos(phi), 0.1), 0.0, c.g_max)
        thr = jnp.clip(0.75 + 0.02 * (240.0 - speed), 0.0, 1.0)
        return jnp.stack([g_want / c.g_max, jnp.clip(2.5 * err / c.roll_rate_max, -1.0, 1.0), 2.0 * thr - 1.0, -1.0])


def n_obs() -> int:
    return 13 + 25
