"""CQB's radio (ep11): words with no given meaning, heard by teammates only. 2v1: operators 0 and 1 are the pair."""

from dataclasses import replace

import jax
import numpy as np

from ailearns.games.cqb import ACTION_SIZE, SAY, CQBConfig, radio_obs

from test_cqb_teams import A0, A1, D, TWO, _acts, _open_game, _pose

RADIO = replace(TWO, radio=(4, 0), share_sight=False)
K = 4


def _say(g, st, **words):
    """One decision in which the named operators say a word (a0=2: Bravo); everyone stands still."""
    a = np.asarray(_acts())
    a = np.concatenate([a, -np.ones((3, ACTION_SIZE - a.shape[1]), np.float32)], 1)
    for who, w in words.items():
        a[{"a0": A0, "a1": A1, "d": D}[who], SAY] = w
    return g.step(st, a)[0]


def _start(cfg=RADIO):
    g = _open_game(cfg)
    return g, _pose(g, [(12, 10), (12, 12), (17, 10)], [np.pi, np.pi, np.pi])   # nobody sees anybody


def _radio(g, st, who):
    return np.asarray(g.observe(st, who))[-radio_obs(K):]


def test_no_radio_by_default():
    assert CQBConfig().radio == (0, 0) and CQBConfig().obs_size == 107 and TWO.obs_size == 124
    assert RADIO.obs_size == TWO.obs_size + 2 * (K + 1)


def test_a_word_reaches_the_partner_not_the_enemy():
    g, st = _start()
    st = _say(g, st, a0=2)
    heard, mine = _radio(g, st, A1)[:K + 1], _radio(g, st, A0)[K + 1:]
    assert np.allclose(heard, [0, 1, 0, 0, 1]) and np.allclose(mine, [0, 1, 0, 0, 1])   # Bravo, just now
    assert not _radio(g, st, A1)[K + 1:].any() and not _radio(g, st, A0)[:K + 1].any()  # A1 said nothing, heard by A0: no
    assert not _radio(g, st, D).any()                                                    # the defender hears nothing
    for _ in range(10):                                                                  # it fades over 3 s
        st = _say(g, st)
    assert 0 < _radio(g, st, A1)[K] < 1 and _radio(g, st, A1)[1] == 1


def test_a_word_takes_half_a_second():
    g, st = _start()
    st = _say(g, st, a0=1)
    for _ in range(4):                                         # still saying "Alpha": a new word is ignored
        st = _say(g, st, a0=3)
        assert int(st.said[A0]) == 1
    st = _say(g, st, a0=3)
    assert int(st.said[A0]) == 3 and int(st.said_age[A0]) == 0


def test_silence_and_out_of_range_words():
    g, st = _start()
    st = _say(g, st, a0=0, a1=-1)
    assert not np.asarray(st.said).any()
    st = _say(g, st, a1=9)                                     # clipped to the last word
    assert int(st.said[A1]) == K
    st2 = _say(g, _start()[1], d=2)                            # the defender has no radio
    assert int(st2.said[D]) == 0


def test_the_downed_dont_talk():
    g, st = _start()
    st = st._replace(alive=st.alive.at[A0].set(False))
    st = _say(g, st, a0=2)
    assert int(st.said[A0]) == 0 and not _radio(g, st, A1).any()


def test_latest_word_wins():
    g, st = _start()
    st = _say(g, st, a0=1)
    st = _say(g, st, a1=4)
    assert np.allclose(_radio(g, st, A0)[:K], [0, 0, 0, 1])   # A0 hears Delta
    for _ in range(4):
        st = _say(g, st)
    st = _say(g, st, a0=2)
    assert np.allclose(_radio(g, st, A1)[:K], [0, 1, 0, 0])


def test_mute_and_scramble_for_measuring():
    g, st = _start(replace(RADIO, radio_test="mute"))
    st = _say(g, st, a0=2)
    assert not _radio(g, st, A1)[:K + 1].any() and _radio(g, st, A0)[K + 1:].any()   # said, but nobody heard it
    g, st0 = _start(replace(RADIO, radio_test="scramble"))
    got = set()
    for k in range(32):
        st = _say(g, st0._replace(key=jax.random.PRNGKey(k)), a0=2)
        got.add(int(np.argmax(_radio(g, st, A1)[:K])) + 1)
        assert int(st.said[A0]) == 2
    assert got == {1, 2, 3, 4}


def test_radio_off_ignores_the_word():
    g, st = _start(TWO)
    st = _say(g, st, a0=2)
    assert int(st.said[A0]) == 0 and len(g.observe(st, A1)) == TWO.obs_size


def test_word_head_sampling_and_log_prob():
    """The word is a choice among words + 1 (0 = silence) in the last column; log_prob and entropy count it only
    where the operator can talk."""
    import jax.numpy as jnp
    from ailearns.agents.ppo_cqb import _split, add_radio, best_guess, entropy, forward, init_params, log_prob, sample, \
        to_action
    params = add_radio(init_params(jax.random.PRNGKey(0), RADIO.obs_size, 32, -0.5, 3), K)
    obs = jax.random.normal(jax.random.PRNGKey(1), (64, RADIO.obs_size))
    mean, logits, log_std, _ = forward(params, obs)
    assert logits.shape == (64, 3 + K + 1) and not np.asarray(_split(logits, K)[1]).any()    # uniform at first
    moves, sw = sample(jax.random.PRNGKey(2), mean, logits, log_std, K)
    w = np.asarray(sw[:, -1])
    assert sw.shape == (64, 4) and set(w.astype(int)) <= set(range(K + 1)) and len(set(w)) > 2
    live = jnp.ones((64, 4))
    mute = live.at[:, -1].set(0.0)
    gap = np.asarray(log_prob(mean, logits, log_std, moves, sw, live, K) - log_prob(mean, logits, log_std, moves, sw,
                                                                                     mute, K))
    assert np.allclose(gap, -np.log(K + 1), atol=1e-5)                                    # uniform over 5
    assert np.isclose(float(entropy(logits, log_std, live, K) - entropy(logits, log_std, mute, K)), np.log(K + 1),
                      atol=1e-5)
    act = np.asarray(to_action(moves, sw, K))
    assert act.shape == (64, ACTION_SIZE) and (act[:, SAY] == w).all() and (act[:, 6:8] == -1).all()
    assert best_guess(params, obs).shape == (64, ACTION_SIZE)


def test_a_pair_trains_with_a_radio(tmp_path):
    """A pair widened from a radio-less pair gets a word head (zero weights), trains, talks, and resumes."""
    import json
    from ailearns.agents.ppo_cqb import CQBPPOConfig, evaluate, load, n_words, train
    small = {"n_envs": 32, "n_steps": 16, "minibatch": 256, "hidden": 32, "buildings": 8, "teams": (2, 1)}
    train(CQBPPOConfig(**small, total_steps=32 * 16), tmp_path / "pair")
    radio = {**small, "share_sight": False, "radio": (K, 0), "init_from": f"{tmp_path / 'pair'}:1"}
    train(CQBPPOConfig(**radio, total_steps=2 * 32 * 16), tmp_path / "radio")
    train(CQBPPOConfig(**radio, total_steps=3 * 32 * 16), tmp_path / "radio", resume=True)
    params, ck = load(tmp_path / "radio" / "checkpoints" / "u00003.pt")
    assert n_words(params) == K and params["body.0.weight"].shape[1] == RADIO.obs_size
    assert tuple(ck["cqb"]["radio"]) == (K, 0) and ck["cqb"]["share_sight"] is False
    start, _ = load(tmp_path / "radio" / "checkpoints" / "u00000.pt")
    assert not np.asarray(start["say.weight"]).any() and not np.asarray(start["body.0.weight"][:, TWO.obs_size:]).any()
    m = [json.loads(x) for x in open(tmp_path / "radio" / "metrics.jsonl")]
    assert all(x["words_per_s"] > 0 and abs(sum(x[f"p_say{k}"] for k in range(K + 1)) - 1) < 1e-4 for x in m)
    assert 0 <= evaluate(str(tmp_path / "radio"), "3", n=4)["win_rate"] <= 1


def test_listening_and_signalling_nudges(tmp_path):
    """listen_kl is zero when the word heard changes nothing and grows when it does; a tiny run with both nudges
    trains and logs them."""
    import json
    import jax.numpy as jnp
    from ailearns.agents.ppo_cqb import CQBPPOConfig, add_radio, forward, init_params, listen_kl, train
    params = add_radio(init_params(jax.random.PRNGKey(0), RADIO.obs_size, 32, -0.5, 3), K)
    obs = jax.random.normal(jax.random.PRNGKey(1), (16, RADIO.obs_size))
    live = jnp.ones((16, 4))
    m, lg, ls, _ = forward(params, obs)
    assert np.allclose(np.asarray(listen_kl(m, lg, ls, m, lg, live, K)), 0.0, atol=1e-6)
    m2, lg2, _, _ = forward(params, obs.at[:, 0].add(3.0))
    assert (np.asarray(listen_kl(m, lg, ls, m2, lg2, live, K)) > 0).all()
    small = {"n_envs": 32, "n_steps": 16, "minibatch": 256, "hidden": 32, "buildings": 8, "teams": (2, 1),
             "share_sight": False, "radio": (K, 0), "listen_coef": 0.1, "signal_coef": 0.1}
    train(CQBPPOConfig(**small, total_steps=2 * 32 * 16), tmp_path / "nudged")
    x = [json.loads(line) for line in open(tmp_path / "nudged" / "metrics.jsonl")]
    assert all(r["listen_kl"] >= 0 and "word_info" in r for r in x)
