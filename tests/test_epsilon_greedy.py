import numpy as np
import pytest

from explorekit.policies.epsilon_greedy import EpsilonGreedyPolicy


N_ACTIONS = 10


@pytest.fixture
def policy():
    return EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=0.2, seed=42)


def test_distribution_sums_to_one(policy):
    cand = np.arange(N_ACTIONS)
    scores = np.random.default_rng(0).normal(size=N_ACTIONS)
    dist = policy.action_distribution(context=None, candidate_items=cand, base_scores=scores)
    assert np.isclose(dist.sum(), 1.0, atol=1e-9)
    assert (dist >= 0).all()


def test_non_candidates_get_zero(policy):
    cand = np.array([1, 3, 5])
    scores = np.random.default_rng(0).normal(size=N_ACTIONS)
    dist = policy.action_distribution(None, cand, scores)
    non_cand = np.setdiff1d(np.arange(N_ACTIONS), cand)
    assert (dist[non_cand] == 0).all()


def test_propensity_matches_probabilities(policy):
    cand = np.arange(N_ACTIONS)
    scores = np.random.default_rng(0).normal(size=N_ACTIONS)
    d = policy.select_action(None, cand, scores)
    assert np.isclose(d.propensity, d.probabilities[d.chosen_item])
    assert d.propensity > 0


def test_blacklist_excludes_items(policy):
    policy.blacklist = np.array([0, 1])
    cand = np.arange(N_ACTIONS)
    scores = np.random.default_rng(0).normal(size=N_ACTIONS)
    dist = policy.action_distribution(None, cand, scores)
    assert dist[0] == 0 and dist[1] == 0


def test_epsilon_zero_is_deterministic():
    p = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=0.0, seed=0)
    cand = np.arange(N_ACTIONS)
    scores = np.array([0.1] * 5 + [0.9] + [0.1] * 4)  # best = index 5
    for _ in range(20):
        d = p.select_action(None, cand, scores)
        assert d.chosen_item == 5
        assert d.is_exploration is False


def test_epsilon_one_is_uniform():
    p = EpsilonGreedyPolicy(n_actions=N_ACTIONS, epsilon=1.0, seed=0)
    cand = np.arange(N_ACTIONS)
    scores = np.zeros(N_ACTIONS)
    dist = p.action_distribution(None, cand, scores)
    assert np.allclose(dist, 1.0 / N_ACTIONS)


def test_chosen_item_rejects_string():
    from explorekit.policies.base import Decision
    with pytest.raises(TypeError):
        Decision(
            chosen_item="item_42",
            propensity=0.5,
            probabilities=np.array([0.5, 0.5]),
            is_exploration=False,
        )