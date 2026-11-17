import numpy as np

from ailearns.games.rocket import (CRASHED, FLYING, LANDED, TIMEOUT, Boosters, RocketConfig, n_actions, n_obs, us76,
                                   world)


def still(b, **state):
    """Put every booster in a hand-picked state: upright, engine off, no spin."""
    for name in ("x", "vx", "vy", "tilt", "spin", "thrust", "gimbal", "fin", "fluid", "time"):
        getattr(b, name)[:] = 0.0
    b.lit[:] = False
    for name, v in state.items():
        getattr(b, name)[:] = v


def coast(n=1):
    return np.tile([-1.0, 0.0, 0.0], (n, 1))


def test_free_fall_matches_the_formula():
    b = Boosters(1, RocketConfig(rho0=0.0), seed=0)
    still(b, y=2000.0, vy=-100.0)
    for _ in range(100):  # 5 s
        b.step(coast())
    t, g = 5.0, b.cfg.gravity
    assert np.isclose(b.vy[0], -100.0 - g * t)
    assert np.isclose(b.y[0], 2000.0 - 100.0 * t - 0.5 * g * t * t)


def test_full_burn_matches_the_rocket_equation():
    cfg = RocketConfig(rho0=0.0, gravity=0.0, thrust_lag=0.0, max_time=1e9)
    b = Boosters(1, cfg, seed=0)
    still(b, y=1e6, fuel=6000.0)
    for _ in range(1000):
        b.step(np.array([[1.0, 0.0, 0.0]]))
    assert b.fuel[0] == 0.0 and not b.lit[0]
    ideal = cfg.isp * 9.80665 * np.log((cfg.dry_mass + 6000.0) / cfg.dry_mass)
    assert np.isclose(b.vy[0], ideal)
    assert np.isclose(b.vx[0], 0.0)


def test_it_cannot_hover_even_at_lowest_throttle():
    cfg = RocketConfig()
    weight_full = (cfg.dry_mass + cfg.start_fuel[1]) * cfg.gravity
    assert cfg.thrust_min > 1.5 * weight_full
    b = Boosters(1, RocketConfig(rho0=0.0, thrust_lag=0.0), seed=0)
    still(b, y=1000.0, fuel=6000.0)
    b.step(np.array([[1e-9, 0.0, 0.0]]))  # just lit = lowest throttle
    assert b.vy[0] > 0.0


def test_once_lit_it_burns_to_touchdown():
    """No shutdown in flight: a coast command while lit means the lowest throttle (a real landing burn)."""
    b = Boosters(1, RocketConfig(thrust_lag=0.0), seed=0)
    still(b, y=2000.0, fuel=6000.0)
    b.step(np.array([[0.5, 0.0, 0.0]]))
    assert b.lit[0] and b.ignitions[0] == 0
    b.step(coast())
    assert b.lit[0] and np.isclose(b.thrust[0], b.cfg.thrust_min)
    b.y[0], b.vy[0] = 0.05, -50.0
    b.step(coast())
    assert b.outcome[0] == CRASHED and not b.lit[0] and b.thrust[0] == 0.0


def test_gimbal_turns_the_booster_and_pushes_it_the_other_way():
    b = Boosters(1, RocketConfig(rho0=0.0, gravity=0.0, thrust_lag=0.0, gimbal_rate=10.0), seed=0)
    still(b, y=1000.0, fuel=6000.0)
    for _ in range(4):
        b.step(np.array([[0.5, 1.0, 0.0]]))
    assert b.spin[0] > 0 and b.tilt[0] > 0  # top leans to +x
    assert b.vx[0] < 0  # the engine's push is to -x at first


def test_falling_engine_first_is_stable_in_air():
    b = Boosters(1, seed=0)
    still(b, y=2400.0, vy=-280.0, tilt=np.radians(3.0), fuel=6000.0)
    peak = 0.0
    for _ in range(60):
        b.step(coast())
        peak = max(peak, abs(b.tilt[0]))
    assert peak <= np.radians(3.0) + 1e-9
    assert abs(b.tilt[0]) < np.radians(3.0)


def test_air_damps_a_tumble():
    """A booster flung into a 60°/s spin swings back to engine-first and settles, like a dart."""
    b = Boosters(1, RocketConfig(max_time=1e9), seed=0)
    still(b, y=2000.0, vy=-50.0, spin=np.radians(60.0), fuel=0.0)
    for _ in range(400):  # 20 s
        b.step(coast())
        assert abs(b.tilt[0]) < np.pi  # never goes over the top
    assert abs(b.tilt[0]) < np.radians(10.0) and abs(b.spin[0]) < np.radians(10.0)


def test_drag_gives_a_terminal_speed_near_the_telemetry():
    """At 2.5 km, falling ~260 m/s engine-first, drag about cancels the weight."""
    b = Boosters(1, seed=0)
    still(b, y=2500.0, vy=-260.0, fuel=6000.0)
    b.step(coast())
    assert abs(b.vy[0] + 260.0) / b.cfg.dt < 0.5  # net acceleration under 0.5 m/s²


def test_touchdown_judge():
    b = Boosters(5, seed=0)
    still(b, y=0.05, vy=-2.0, fuel=1000.0)
    b.vy[1] = -6.0  # too fast
    b.vx[2] = 2.0  # sliding
    b.tilt[3] = np.radians(12.0)  # leaning
    b.x[4] = 100.0  # off the pad
    b.step(coast(5))
    assert list(b.outcome) == [LANDED, CRASHED, CRASHED, CRASHED, CRASHED]


def test_finished_boosters_stay_put_and_timeout():
    b = Boosters(2, RocketConfig(max_time=0.1), seed=0)
    still(b, y=1000.0, fuel=1000.0)
    b.y[0] = 0.01
    b.vy[0] = -50.0
    _, r, done = b.step(coast(2))
    assert b.outcome[0] == CRASHED and r[0] < -1 and done[0]
    y0 = b.y[0]
    b.step(coast(2))
    assert b.y[0] == y0
    assert b.outcome[1] == TIMEOUT


def test_same_seed_same_flights():
    def run(seed):
        b = Boosters(8, seed=seed)
        rng = np.random.default_rng(1)
        for _ in range(200):
            b.step(rng.uniform(-1, 1, (8, 3)))
        return np.stack([b.x, b.y, b.vy, b.tilt, b.fuel])
    assert np.array_equal(run(3), run(3))
    assert not np.array_equal(run(3), run(4))


def test_reset_only_the_masked_ones():
    b = Boosters(4, seed=0)
    b.outcome[:] = CRASHED
    b.reset(np.array([True, False, True, False]))
    assert list(b.outcome) == [FLYING, CRASHED, FLYING, CRASHED]


def test_the_classic_controller_lands_most_of_them():
    from ailearns.agents.rocket_classic import act
    b = Boosters(200, seed=11)
    while (b.outcome == FLYING).any():
        b.step(act(b))
    assert np.mean(b.outcome == LANDED) > 0.75


def test_a_softer_crash_scores_better():
    b = Boosters(3, seed=0)
    still(b, y=0.05, x=0.0, fuel=5000.0)
    b.vy[:] = [-2.0, -6.0, -250.0]
    _, r, done = b.step(coast(3))
    assert done.all() and 0.5 < r[0] < 0.8 and -2.0 < r[1] < -1.0 and r[2] < -5.0  # landed at ~2 m/s: some leg wear


def test_a_softer_landing_scores_better():
    """The legs' crush cores absorb the touchdown: a soft landing scores near +1, one at the limit near 0."""
    b = Boosters(3, seed=0)
    still(b, y=0.05, x=0.0, fuel=5000.0)
    b.vy[:] = [-0.3, -1.2, -2.3]  # gravity adds ~1 m/s before they touch
    r = np.zeros(3)
    while (b.outcome == FLYING).any():
        r += b.step(coast(3))[1]
    assert (b.outcome == LANDED).all() and r[0] > r[1] > r[2] > 0.0 and r[0] > 0.8


# -- grid fins (ep18 10a) ------------------------------------------------------

def fins_fall(fin_cmd, seconds=3.0, vy=-280.0, **cfg):
    """A booster falling unlit through still air, fins held at `fin_cmd`; -> the booster after `seconds`."""
    b = Boosters(1, world("pad+fins") if not cfg else RocketConfig(fin_area=3.6, **cfg), seed=0)
    still(b, y=2400.0, vy=vy, fuel=6000.0)
    for _ in range(int(seconds / b.cfg.dt)):
        b.step(np.array([[-1.0, 0.0, 0.0, fin_cmd]]))
    return b


def test_fins_swing_at_their_rate():
    b = fins_fall(1.0, seconds=0.25)
    assert np.isclose(b.fin[0], np.radians(10.0))  # 40°/s for 0.25 s, short of the 20° stop
    assert np.isclose(fins_fall(1.0, seconds=1.0).fin[0], np.radians(20.0))


def test_fins_steer_the_fall_both_ways():
    left, straight, right = fins_fall(-1.0), fins_fall(0.0), fins_fall(1.0)
    assert abs(straight.x[0]) < 1e-9 and abs(straight.tilt[0]) < 1e-9
    assert left.x[0] * right.x[0] < 0 and abs(right.x[0]) > 1.0  # opposite ways, by metres in 3 s
    assert np.isclose(left.x[0], -right.x[0]) and np.isclose(left.tilt[0], -right.tilt[0])


def test_fins_need_air_speed():
    """At 280 m/s the fins turn the booster hard; at 10 m/s (the last moments) they hardly do anything."""
    fast, slow = fins_fall(1.0, seconds=1.0), fins_fall(1.0, seconds=1.0, vy=-10.0)
    assert abs(fast.spin[0]) > 100 * abs(slow.spin[0])


def test_fins_steady_a_swinging_top():
    """Held straight, the fins resist the top swinging sideways, like the feathers on a dart."""
    def swing(cfg):
        b = Boosters(1, cfg, seed=0)
        still(b, y=2400.0, vy=-280.0, spin=np.radians(10.0), fuel=6000.0)
        for _ in range(10):
            b.step(np.array([[-1.0, 0.0, 0.0, 0.0]]))
        return abs(b.spin[0])
    assert swing(world("pad+fins")) < 0.8 * swing(world("pad"))


def test_fins_add_a_control_and_a_sense_only_when_fitted():
    assert (n_actions(world("storm")), n_obs(world("storm"))) == (3, 18)
    assert (n_actions(world("storm+fins")), n_obs(world("storm+fins"))) == (4, 19)
    assert Boosters(5, world("ship+fins"), seed=0).obs().shape == (5, 19)
    # a fourth column on a booster without fins changes nothing (the old controllers send three)
    a, b = Boosters(4, seed=2), Boosters(4, seed=2)
    for _ in range(100):
        a.step(np.tile([0.5, 0.2, 0.1], (4, 1)))
        b.step(np.tile([0.5, 0.2, 0.1, 1.0], (4, 1)))
    assert np.array_equal(a.x, b.x) and not b.fin.any()


# -- open-loop hydraulics (ep18 10c) --------------------------------------------

def test_fluid_pays_for_every_move_and_runs_out():
    b = Boosters(1, world("pad+fins+fluid"), seed=0)
    still(b, y=2400.0, vy=-280.0, fuel=6000.0, fluid=np.radians(30.0))
    for cmd in (1.0, 1.0, 1.0, 1.0, 1.0):  # 5 steps at 2°/step: 10° of travel
        b.step(np.array([[-1.0, 0.0, 0.0, cmd]]))
    assert np.isclose(b.fluid[0], np.radians(20.0)) and np.isclose(b.fin[0], np.radians(10.0))
    for _ in range(20):  # holding still costs nothing
        b.step(np.array([[-1.0, 0.0, 0.0, 0.5]]))
    assert np.isclose(b.fluid[0], np.radians(20.0))
    for _ in range(40):  # swinging back and forth drains it, and the last move stops where the fluid does
        b.step(np.array([[-1.0, 0.0, 0.0, -1.0 if (b.time[0] // 0.5) % 2 else 1.0]]))
    assert b.fluid[0] == 0.0 and abs(b.fin[0]) <= np.radians(20.0)


def test_dry_fins_go_limp():
    """Out of fluid, deflected fins push nothing: the booster falls as if it had none."""
    def fall(where, fluid):
        b = Boosters(1, world(where), seed=0)
        still(b, y=2400.0, vy=-280.0, fuel=6000.0, fin=np.radians(15.0), fluid=fluid)
        for _ in range(40):
            b.step(np.array([[-1.0, 0.0, 0.0, 0.75]]))
        return b
    dry, wet, finless = fall("pad+fins+fluid", 0.0), fall("pad+fins+fluid", 1.0), fall("pad", 0.0)
    assert abs(wet.x[0]) > 0.5 and dry.x[0] == finless.x[0] and dry.tilt[0] == finless.tilt[0]


def test_fluid_is_a_sense_and_drawn_per_burn():
    cfg = world("storm+fins+fluid")
    b = Boosters(500, cfg, seed=0)
    assert n_obs(cfg) == 20 and b.obs().shape == (500, 20) and n_actions(cfg) == 4
    assert cfg.fin_fluid[0] <= b.fluid.min() and b.fluid.max() <= cfg.fin_fluid[1] and b.fluid.std() > 0
    # the other worlds draw exactly as before
    assert np.array_equal(Boosters(50, world("storm+fins"), seed=3).x, Boosters(50, cfg, seed=3).x)


# -- hardware faults (ep18 10e) ------------------------------------------------

def test_stuck_fins_and_a_jammed_gimbal_stop_answering_and_say_so():
    b = Boosters(1, world("pad+fins+faults"), seed=0)
    still(b, y=2400.0, vy=-280.0, fuel=6000.0)
    b.fin_fail_t[:], b.gimbal_fail_t[:], b.engine_eff[:] = 0.24, 0.24, 1.0  # after 5 steps' moves
    assert not b.obs()[0, -2:].any()
    for _ in range(5):  # 0.25 s at full command: 10° of fin, 5° (the stop) of gimbal
        b.step(np.array([[0.5, 1.0, 0.0, 1.0]]))
    fin, gimbal = b.fin[0], b.gimbal[0]
    for _ in range(20):
        b.step(np.array([[0.5, -1.0, 0.0, -1.0]]))
    assert b.fin[0] == fin and b.gimbal[0] == gimbal and (b.obs()[0, -2:] == 1).all()
    assert np.isclose(fin, np.radians(10.0))


def test_a_weak_engine_gives_less_than_asked():
    def burn(eff):
        b = Boosters(1, world("pad+faults"), seed=0)
        still(b, y=2400.0, vy=-100.0, fuel=6000.0)
        b.fin_fail_t[:], b.gimbal_fail_t[:], b.engine_eff[:] = 1e9, 1e9, eff
        for _ in range(40):
            b.step(np.array([[1.0, 0.0, 0.0]]))
        return b
    weak, healthy = burn(0.85), burn(1.0)
    assert np.isclose(weak.thrust[0], 0.85 * healthy.thrust[0]) and weak.vy[0] < healthy.vy[0]


def test_faults_are_drawn_per_burn():
    cfg = world("storm+fins+faults")
    b = Boosters(4000, cfg, seed=0)
    assert n_obs(cfg) == 21 and b.obs().shape == (4000, 21)
    for t in (b.fin_fail_t, b.gimbal_fail_t):
        hit = t < 1e9
        assert 0.08 < hit.mean() < 0.12 and t[hit].min() >= 0 and t[hit].max() <= 15
    weak = b.engine_eff < 1
    assert 0.08 < weak.mean() < 0.12 and b.engine_eff[weak].min() >= 0.8 and b.engine_eff.max() == 1.0
    assert np.array_equal(Boosters(50, world("storm+fins"), seed=3).x, Boosters(50, cfg, seed=3).x)


# -- from 70 km (ep18 step 11) -------------------------------------------------

def test_the_standard_atmosphere():
    rho, p, a = us76(np.array([0.0, 11e3, 40e3, 70e3]))
    assert np.allclose(rho, [1.225, 0.3639, 0.003996, 8.283e-5], rtol=1e-6)
    assert np.isclose(p[0], 1.0) and p[-1] < 1e-4
    assert np.allclose(a, [340.3, 295.1, 317.6, 297.1], atol=0.5)


def entry_start(b, burn=False, fuel=20000.0, speed=1800.0):
    """Every booster at 70 km, 1.8 km/s, 60° below the horizon, engine first."""
    still(b, y=70e3, fuel=fuel)
    b.vx[:], b.vy[:] = speed * np.cos(np.radians(60)), -speed * np.sin(np.radians(60))
    b.tilt[:] = -np.arctan2(b.vx, -b.vy)
    b.ignitions[:] = b.cfg.ignitions


def test_the_entry_burn_lights_three_engines_shuts_down_and_the_landing_burn_lights_one():
    b = Boosters(1, world("entry"), seed=0)
    entry_start(b)
    for _ in range(50):
        b.step(np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert b.ignitions[0] == 1 and np.isclose(b.thrust[0], 3 * b.cfg.thrust_max, rtol=5e-3)
    for _ in range(9):  # a light still in hand: held below 0 through the 0.5 s start sequence shuts it down
        b.step(np.array([[-1.0, 0.0, 0.0, 0.0]]))
    assert b.lit[0]
    b.step(np.array([[-1.0, 0.0, 0.0, 0.0]]))
    assert not b.lit[0] and b.ignitions[0] == 1
    for _ in range(50):
        b.step(np.array([[1.0, 0.0, 0.0, 0.0]]))
    assert b.ignitions[0] == 0 and np.isclose(b.thrust[0], b.cfg.thrust_max, rtol=5e-3)
    b.step(np.array([[-1.0, 0.0, 0.0, 0.0]]))  # the last light can't be shut down: lowest throttle
    assert b.lit[0]


def test_thin_air_lets_the_engine_push_harder():
    def dv(y):
        b = Boosters(1, world("entry"), seed=0)
        still(b, y=y, fuel=20000.0)
        b.cfg = RocketConfig(**{**b.cfg.__dict__, "rho0": 0.0, "atmosphere": "us76", "gravity": 0.0,
                                     "start_seq": 0.0})
        v0 = b.vy[0]
        for _ in range(20):
            b.step(np.array([[1.0, 0.0, 0.0, 0.0]]))
        return b.vy[0] - v0
    assert np.isclose(dv(70e3) / dv(0.0), 914.0 / 845.2, rtol=2e-3)


def test_without_an_entry_burn_it_breaks_up():
    """The reason for the entry burn: falling from 70 km at 1.8 km/s, the heat and the air load are too much."""
    from ailearns.agents.rocket_entry import EntryGains, act

    def fall(gains):  # the hand-written entry holds the attitude; only the burn differs
        b = Boosters(1, world("entry"), seed=0)
        entry_start(b)
        while b.outcome[0] == FLYING and b.y[0] > 3500:
            b.step(act(b, gains))
        return b
    assert fall(EntryGains(light_h=-1.0)).outcome[0] == CRASHED  # never lights
    ok = fall(EntryGains())
    assert ok.outcome[0] == FLYING and ok.peak_q[0] < ok.cfg.q_limit and ok.heat[0] < ok.cfg.heat_limit


def test_body_lift_pushes_across_in_proportion_to_the_angle():
    def side(angle, lift=True):
        b = Boosters(1, RocketConfig(body_lift=lift, gravity=0.0), seed=0)
        still(b, y=1000.0, vy=-200.0, tilt=np.radians(angle), fuel=6000.0)
        b.step(np.array([[-1.0, 0.0, 0.0]]))
        return b.vx[0]
    lift1, lift2 = side(1.0) - side(1.0, False), side(2.0) - side(2.0, False)
    assert lift1 != 0 and np.isclose(lift2 / lift1, 2.0, rtol=0.01)


def test_the_hand_written_entry_lands_some():
    from ailearns.agents.rocket_entry import act
    b = Boosters(100, world("entry"), seed=7)
    while (b.outcome == FLYING).any():
        b.step(act(b))
    assert (b.outcome == LANDED).mean() > 0.2 and n_obs(b.cfg) == b.obs().shape[1] == 27


def test_a_flicker_cannot_light_the_engine():
    """(11e) The start sequence: only a command held 0.5 s lights it; noise that flickers above 0 doesn't."""
    b = Boosters(2, world("entry"), seed=0)
    entry_start(b)
    rng = np.random.default_rng(0)
    for _ in range(400):  # 20 s of a command centred below 0 with a spread of 1 (training's exploration noise)
        a = np.zeros((2, 4))
        a[:, 0] = -1.0 + rng.normal(0, 1, 2)
        b.step(a)
    assert (b.ignitions == 2).all() and not b.lit.any()
    for _ in range(10):
        b.step(np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)))
    assert b.lit.all() and (b.ignitions == 1).all()


def test_a_command_leaning_on_still_lights_through_the_noise():
    """(11e) The spin-up climbs while asked and falls back while not: noise centred above the band lights it."""
    b = Boosters(200, world("entry"), seed=0)
    entry_start(b)
    rng = np.random.default_rng(1)
    for _ in range(60):  # 3 s
        a = np.zeros((200, 4))
        a[:, 0] = 1.0 + rng.normal(0, 1, 200)
        b.step(a)
    assert (b.ignitions == 1).mean() > 0.95


def test_lowest_throttle_is_not_the_off_switch():
    """(11g) A lit entry burn with its command hovering around 0 stays lit at the lowest throttle."""
    b = Boosters(50, world("entry"), seed=0)
    entry_start(b)
    for _ in range(20):
        b.step(np.tile([1.0, 0.0, 0.0, 0.0], (50, 1)))
    assert b.lit.all()
    rng = np.random.default_rng(2)
    for _ in range(100):
        a = np.zeros((50, 4))
        a[:, 0] = rng.uniform(-0.45, 0.45, 50)
        b.step(a)
    assert b.lit.all() and (b.thrust < 3 * b.cfg.thrust_min * 1.2).all()  # low throttle: positive commands nudge it
