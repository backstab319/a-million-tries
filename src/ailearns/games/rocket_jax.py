"""The booster's landing burn once more, in JAX, so the whole training loop can be compiled.

Same physics, judge, reward and observations as `rocket.py` (the reference; the look sheets and the classic
controller use it, and `tests/test_rocket_jax.py` checks the two agree step for step). Written for one booster as
pure functions of a `State`; `jax.vmap` turns them into thousands, `jax.jit` compiles them. The physics is
deterministic, so only `init` and `reset_if` take a random key.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.games.rocket import (CRASHED, FLYING, G0, LANDED, N_ACTIONS, NEVER, SUTTON_GRAVES, US76, entry, REWARD_LANDED, REWARD_TIMEOUT, TIMEOUT,
                                   RocketConfig)


class State(NamedTuple):
    x: jax.Array          # () position of the base, m
    y: jax.Array
    vx: jax.Array
    vy: jax.Array
    tilt: jax.Array       # rad, positive = top leaning towards +x
    spin: jax.Array       # rad/s
    fuel: jax.Array       # kg
    thrust: jax.Array     # N, what the engine pushes now
    gimbal: jax.Array     # rad
    time: jax.Array       # s
    lit: jax.Array        # () bool
    ignitions: jax.Array  # () int32, lights left
    outcome: jax.Array    # () int32: FLYING, LANDED, CRASHED or TIMEOUT
    # this burn's weather (ep18 9b; Boosters.wind_at, deck_y)
    wind0: jax.Array      # m/s at 10 m, signed
    gust: jax.Array       # m/s
    gust_phase: jax.Array
    heave: jax.Array      # m
    heave_phase: jax.Array
    fin: jax.Array        # rad, the grid fins' angle (ep18 10a; stays 0 without fins)
    fluid: jax.Array      # rad of fin travel left in the open-loop hydraulics (10c; 0 and unused when closed loop)
    fin_fail_t: jax.Array     # s: when the fins stick (10e; NEVER = they don't)
    gimbal_fail_t: jax.Array  # s: when the gimbal jams
    engine_eff: jax.Array     # the share of asked-for thrust the engine gives (1 = healthy)
    heat: jax.Array       # J/m^2 taken on the engine end so far (11a)
    peak_q: jax.Array     # Pa, the highest dynamic pressure so far
    hold: jax.Array       # s the engine command has asked for a light / shutdown (11e)


def us76(y):
    """rocket.us76 in JAX: density, pressure / sea level, speed of sound."""
    t = jnp.asarray(US76)
    y = jnp.clip(y, 0.0, US76[-1, 0])
    return (jnp.exp(jnp.interp(y, t[:, 0], jnp.log(t[:, 1]))),
            jnp.exp(jnp.interp(y, t[:, 0], jnp.log(t[:, 2]))) / US76[0, 2],
            jnp.sqrt(1.4 * 287.05 * jnp.interp(y, t[:, 0], t[:, 3])))


class JaxBoosters:
    """Functions for one booster; vmap them for many."""

    def __init__(self, config: RocketConfig | None = None):
        self.cfg = c = config or RocketConfig()
        self.exhaust_speed = c.isp * G0
        self.area_ax = np.pi * (c.diameter / 2) ** 2
        self.arm = (np.arange(c.drag_slices) + 0.5) * c.length / c.drag_slices - c.cg_height  # slice centres
        self.lag = 1.0 if c.thrust_lag <= 0 else 1.0 - np.exp(-c.dt / c.thrust_lag)

    def mass(self, st: State):
        return self.cfg.dry_mass + st.fuel

    def inertia(self, st: State):
        c = self.cfg
        return self.mass(st) * (c.length**2 / 12 + (c.length / 2 - c.cg_height) ** 2)

    def lowest_foot(self, st: State):
        return st.y - self.cfg.leg_span / 2 * jnp.abs(jnp.sin(st.tilt)) - self.deck(st, st.time)[0]

    def init(self, key) -> State:
        c = self.cfg
        k = jax.random.split(key, 7)
        u = lambda k, r: jax.random.uniform(k, minval=r[0], maxval=r[1])  # noqa: E731
        zero = jnp.zeros(())
        return State(x=u(k[0], c.start_x), y=u(k[1], c.start_height), vx=u(k[2], c.start_vx),
                     vy=-u(k[3], c.start_fall), tilt=u(k[4], c.start_tilt), spin=u(k[5], c.start_spin),
                     fuel=u(k[6], c.start_fuel), thrust=zero, gimbal=zero, time=zero, lit=jnp.bool_(False),
                     ignitions=jnp.int32(c.ignitions), outcome=jnp.int32(FLYING), **self.weather(key), fin=zero,
                     fluid=jax.random.uniform(jax.random.fold_in(key, 10), minval=c.fin_fluid[0], maxval=c.fin_fluid[1]),
                     **self.faults(key), heat=zero, peak_q=zero, hold=zero)._replace(**self.slant(key, u(k[3], c.start_fall),
                                                                                            u(k[4], c.start_tilt)))

    def slant(self, key, speed, tilt) -> dict:
        """(11b) Coming in along a slanted path, engine first (Boosters.reset); nothing without start_fpa."""
        c = self.cfg
        if not c.start_fpa:
            return {}
        path = jax.random.uniform(jax.random.fold_in(key, 12), minval=c.start_fpa[0], maxval=c.start_fpa[1])
        vx, vy = speed * jnp.cos(path), -speed * jnp.sin(path)
        return {"vx": vx, "vy": vy, "tilt": tilt - jnp.arctan2(vx, -vy)}

    def faults(self, key) -> dict:
        """(10e) This burn's hardware faults (Boosters.reset)."""
        c = self.cfg
        k = jax.random.split(jax.random.fold_in(key, 11), 6)
        hit = lambda k: jax.random.uniform(k) < c.fault_p  # noqa: E731
        when = lambda k: jax.random.uniform(k, minval=c.fault_window[0], maxval=c.fault_window[1])  # noqa: E731
        return {"fin_fail_t": jnp.where(hit(k[0]), when(k[1]), NEVER),
                "gimbal_fail_t": jnp.where(hit(k[2]), when(k[3]), NEVER),
                "engine_eff": jnp.where(hit(k[4]), jax.random.uniform(k[5], minval=c.engine_weak[0],
                                                                         maxval=c.engine_weak[1]), 1.0)}

    def weather(self, key) -> dict:
        c = self.cfg
        k = jax.random.split(jax.random.fold_in(key, 9), 6)
        u = lambda k, r: jax.random.uniform(k, minval=r[0], maxval=r[1])  # noqa: E731
        sign = jnp.where(jax.random.bernoulli(k[0]), 1.0, -1.0)
        return {"wind0": u(k[1], c.wind_mean) * sign, "gust": u(k[2], c.wind_gust),
                "gust_phase": u(k[3], (0.0, 2 * np.pi)), "heave": u(k[4], c.heave),
                "heave_phase": u(k[5], (0.0, 2 * np.pi))}

    def wind_at(self, st: State, y, t):
        c = self.cfg
        steady = st.wind0 * (jnp.maximum(y, 1.0) / 10.0) ** c.wind_shear
        gusts = st.gust * (0.6 * jnp.sin(2 * np.pi * t / 7.0 + st.gust_phase)
                           + 0.4 * jnp.sin(2 * np.pi * t / 2.3 + 2.7 * st.gust_phase))
        return c.wind + steady + gusts

    def deck(self, st: State, t):
        """-> (the deck's height, its upward speed) at time t."""
        w = 2 * np.pi / self.cfg.heave_period
        return st.heave * jnp.sin(w * t + st.heave_phase), st.heave * w * jnp.cos(w * t + st.heave_phase)

    def reset_if(self, st: State, done, key) -> State:
        fresh = self.init(key)
        return jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, st)

    def observe(self, st: State):
        c = self.cfg
        deck_y, deck_vy = self.deck(st, st.time)
        return jnp.stack([
            st.x / 500.0, (st.y - deck_y) / 2500.0, st.vx / 50.0, st.vy / 300.0,
            jnp.sin(st.tilt), jnp.cos(st.tilt), st.spin,
            st.fuel / c.start_fuel[1], st.thrust / c.thrust_max, st.gimbal / c.gimbal_max,
            st.lit.astype(st.x.dtype), st.ignitions.astype(st.x.dtype),
            jnp.sign(st.x) * jnp.sqrt(jnp.abs(st.x) / 500.0), jnp.sqrt(jnp.maximum(st.y - deck_y, 0.0) / 2500.0),
            jnp.sign(st.vx) * jnp.sqrt(jnp.abs(st.vx) / 50.0), jnp.sign(st.vy) * jnp.sqrt(jnp.abs(st.vy) / 300.0),
            self.wind_at(st, st.y, st.time) / 20.0, deck_vy / 3.0,
            *([st.fin / c.fin_max] if c.fin_area > 0 else []),
            *([st.fluid / c.fin_fluid[1]] if c.fin_fluid[1] > 0 else []),
            *([(st.time >= st.fin_fail_t).astype(st.x.dtype), (st.time >= st.gimbal_fail_t).astype(st.x.dtype)]
              if c.fault_p > 0 else []),
            *([st.y / 70e3, st.vy / 2000.0, st.vx / 1000.0, st.x / 30e3,
               jnp.hypot(st.vx - self.wind_at(st, st.y, st.time), st.vy) / us76(st.y)[2] / 6.0,
               st.peak_q / c.q_limit, st.heat / c.heat_limit, st.hold / max(c.start_seq, 1e-9)] if entry(c) else []),
        ])

    def step(self, st: State, action):
        """Advance one dt if still flying. `action` is (3,) in [-1, 1], or (4,) with the grid fins last. Returns (state, reward, done);
        a finished burn is not reset. Line for line the same as `Boosters.step`."""
        c, dt = self.cfg, self.cfg.dt
        a = jnp.asarray(action).reshape(-1)
        a = jnp.clip(jnp.pad(a, (0, N_ACTIONS + 1 - a.shape[0])), -1.0, 1.0)
        if c.fin_area <= 0:
            a = a.at[3].set(0.0)
        fly = st.outcome == FLYING

        # Engine: once lit it burns until touchdown or an empty tank; 0 or below = lowest throttle.
        shut = fly & st.lit & (st.ignitions > 0) & (a[0] < -c.switch_band)  # (11a) a light in hand: shuts it down
        light = fly & (a[0] > c.switch_band) & ~st.lit & (st.ignitions > 0) & (st.fuel > 0)
        hold = st.hold
        if c.start_seq > 0:  # (11e) held through the start sequence
            hold = jnp.where(fly, jnp.clip(hold + jnp.where(shut | light, dt, -dt), 0.0, c.start_seq), hold)
            done_seq = hold >= c.start_seq - 1e-9
            shut, light = shut & done_seq, light & done_seq
            hold = jnp.where(shut | light, 0.0, hold)
        lit = st.lit & ~shut
        ignitions = st.ignitions - light.astype(jnp.int32)
        lit = (lit | light) & (st.fuel > 0)
        engines = jnp.asarray(c.light_engines)[jnp.clip(c.ignitions - ignitions - 1, 0, len(c.light_engines) - 1)]
        target = jnp.where(lit, c.thrust_min + (c.thrust_max - c.thrust_min) * jnp.maximum(a[0], 0.0), 0.0)
        target = target * engines
        target = target * st.engine_eff  # (10e)
        thrust = jnp.where(fly, st.thrust + (target - st.thrust) * self.lag, st.thrust)
        step_g = c.gimbal_rate * dt
        gimbal = jnp.where(fly & (st.time < st.gimbal_fail_t), st.gimbal + jnp.clip(a[1] * c.gimbal_max - st.gimbal, -step_g, step_g), st.gimbal)
        step_f = c.fin_rate * dt
        move = jnp.where(fly & (st.time < st.fin_fail_t), jnp.clip(a[3] * c.fin_max - st.fin, -step_f, step_f), 0.0)
        fluid = st.fluid
        if c.fin_fluid[1] > 0:  # (10c) open loop: a move uses its travel in fluid
            move = jnp.clip(move, -fluid, fluid)
            fluid = fluid - jnp.abs(move)
        fin = st.fin + move
        limp = (c.fin_fluid[1] > 0) & (fluid <= 0)

        # Fuel burnt this step; the speed it gives is exact (rocket equation over the step).
        m0 = self.mass(st)
        burn = jnp.where(fly, jnp.minimum(thrust / self.exhaust_speed * dt, st.fuel), 0.0)
        fuel = st.fuel - burn
        m1 = c.dry_mass + fuel
        boost = 1.0
        if c.thrust_vac > 0:  # (11a) thin air: the same flow pushes harder
            boost = c.thrust_vac / c.thrust_max - (c.thrust_vac / c.thrust_max - 1.0) * us76(st.y)[1]
        dv_thrust = self.exhaust_speed * boost * jnp.log(m0 / m1)
        thrust_used = jnp.where(burn > 0, thrust * boost, 0.0)
        out = fuel <= 0
        lit = lit & ~out
        thrust = jnp.where(out, 0.0, thrust)

        st_, ct = jnp.sin(st.tilt), jnp.cos(st.tilt)
        dirx, diry = jnp.sin(st.tilt - gimbal), jnp.cos(st.tilt - gimbal)

        # Drag along the body, and across it slice by slice (the turning adds to each slice's speed).
        rho = us76(st.y)[0] if c.atmosphere == "us76" else c.rho0 * jnp.exp(-jnp.maximum(st.y, 0.0) / c.scale_height)
        rvx = st.vx - self.wind_at(st, st.y, st.time)
        v_ax = rvx * st_ + st.vy * ct
        v_side = rvx * ct - st.vy * st_
        speed = jnp.hypot(rvx, st.vy)
        cd_ax = c.cd_axial
        if c.cd_mach:
            table = np.asarray(c.cd_mach)
            cd_ax = c.cd_axial * jnp.interp(speed / us76(st.y)[2], table[:, 0], table[:, 1])
        f_ax = -0.5 * rho * cd_ax * self.area_ax * jnp.abs(v_ax) * v_ax
        heat = st.heat + jnp.where(fly, SUTTON_GRAVES * jnp.sqrt(rho / c.heat_radius) * speed**3 * dt, 0.0)
        peak_q = jnp.where(fly, jnp.maximum(st.peak_q, 0.5 * rho * speed**2), st.peak_q)
        bend = 0.5 * rho * speed**2 * jnp.degrees(jnp.arctan2(jnp.abs(v_side), jnp.abs(v_ax)))  # (12m) q-alpha
        v_slice = v_side + st.spin * self.arm
        f_slice = -0.5 * rho * c.cd_side * (c.length / c.drag_slices * c.diameter) * jnp.abs(v_slice) * v_slice
        f_side = f_slice.sum()
        aero_torque = (f_slice * self.arm).sum()
        if c.body_lift:  # (11a) slender-body lift at the leading end
            lift = -rho * self.area_ax * v_side * jnp.abs(v_ax)
            f_side = f_side + lift
            aero_torque = aero_torque + lift * jnp.where(v_ax < 0, -c.cg_height, c.length - c.cg_height)
        # Grid fins (10a), as in Boosters.step: a plate at angle `fin`, high up, pushed normal to itself.
        arm_f = c.fin_height - c.cg_height
        v_fin = v_side + st.spin * arm_f
        sf, cf = jnp.sin(fin), jnp.cos(fin)
        w_n, w_t = v_ax * sf - v_fin * cf, -v_ax * cf - v_fin * sf
        aoa = jnp.arctan2(w_n, jnp.abs(w_t))
        f_fin = jnp.where(limp, 0.0,
                          0.5 * rho * (v_ax**2 + v_fin**2) * c.fin_area * c.fin_cn * jnp.clip(aoa, -c.fin_stall, c.fin_stall))
        f_ax = f_ax - f_fin * sf
        f_side = f_side + f_fin * cf
        aero_torque = aero_torque + f_fin * cf * arm_f
        ax = (f_ax * st_ + f_side * ct) / m0
        ay = (f_ax * ct - f_side * st_) / m0 - c.gravity

        torque = (c.cg_height * thrust_used * jnp.sin(gimbal) + aero_torque
                  + a[2] * c.rcs_force * (c.length - c.cg_height))
        inertia = m1 * (c.length**2 / 12 + (c.length / 2 - c.cg_height) ** 2)
        spin_new = st.spin + torque / inertia * dt

        vx_new = st.vx + ax * dt + dv_thrust * dirx
        vy_new = st.vy + ay * dt + dv_thrust * diry
        f = fly.astype(st.x.dtype)
        x = st.x + f * 0.5 * (st.vx + vx_new) * dt
        y = st.y + f * 0.5 * (st.vy + vy_new) * dt
        vx = jnp.where(fly, vx_new, st.vx)
        vy = jnp.where(fly, vy_new, st.vy)
        tilt = st.tilt + f * 0.5 * (st.spin + spin_new) * dt
        spin = jnp.where(fly, spin_new, st.spin)
        time = st.time + f * dt

        # Touchdown judge.
        deck_y, deck_vy = self.deck(st, time)
        down = fly & (y - c.leg_span / 2 * jnp.abs(jnp.sin(tilt)) - deck_y <= 0.0)
        soft = ((jnp.abs(vy - deck_vy) <= c.land_vy) & (jnp.abs(vx) <= c.land_vx) & (jnp.abs(tilt) <= c.land_tilt)
                & (jnp.abs(spin) <= c.land_spin) & (jnp.abs(x) <= c.pad_radius))
        outcome = jnp.where(down & soft, LANDED, jnp.where(down, CRASHED, st.outcome))
        broke = fly & ~down & (((c.heat_limit > 0) & (heat > c.heat_limit)) | ((c.q_limit > 0) & (peak_q > c.q_limit))
                               | ((c.bend_limit > 0) & (bend > c.bend_limit)))  # (12m) buckled sideways
        outcome = jnp.where(broke, CRASHED, outcome)
        outcome = jnp.where(fly & ~down & (time >= c.max_time), TIMEOUT, outcome).astype(jnp.int32)
        ended = fly & (outcome != FLYING)
        lit = lit & ~ended
        thrust = jnp.where(ended, 0.0, thrust)

        over = (jnp.maximum(jnp.abs(vy - deck_vy) - c.land_vy, 0) / c.land_vy + jnp.maximum(jnp.abs(vx) - c.land_vx, 0) / c.land_vx
                + jnp.maximum(jnp.abs(tilt) - c.land_tilt, 0) / c.land_tilt
                + jnp.maximum(jnp.abs(spin) - c.land_spin, 0) / c.land_spin
                + jnp.maximum(jnp.abs(x) - c.pad_radius, 0) / c.pad_radius)
        wear = 0.5 * (((vy - deck_vy) / c.land_vy) ** 2 + (vx / c.land_vx) ** 2)  # the legs' crush cores (Boosters.leg_wear)
        reward = jnp.where(outcome == LANDED, REWARD_LANDED * (1.0 - wear),
                           jnp.where(outcome == TIMEOUT, REWARD_TIMEOUT, -1.0 - jnp.log1p(over)))
        reward = jnp.where(ended, reward, 0.0)
        st = State(x, y, vx, vy, tilt, spin, fuel, thrust, gimbal, time, lit, ignitions, outcome,
                   st.wind0, st.gust, st.gust_phase, st.heave, st.heave_phase, fin, fluid,
                   st.fin_fail_t, st.gimbal_fail_t, st.engine_eff, heat, peak_q, hold)
        return st, reward, outcome != FLYING
