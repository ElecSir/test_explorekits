import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from explorekit.policies.config import load_policies_config
from explorekit.policies.epsilon_greedy import EpsilonGreedyPolicy
from explorekit.ope.base import build_ope_input
from explorekit.ope.ips import IPSEstimator


# --- Конфиг ---
cfg = load_policies_config()
spec = cfg["policies"]["smoke_epsilon_greedy"]

n_actions = spec["n_actions"]
epsilon = spec["epsilon"]
seed = spec.get("seed", cfg["seed"])

print(f"Config: n_actions={n_actions}, epsilon={epsilon}, seed={seed}")

# --- Синтетические логи (behavior = uniform) ---
n_rounds = 5000
rng = np.random.default_rng(0)
true_ctr = np.array([0.4, 0.2, 0.3, 0.1, 0.35])
behavior_pscore = np.full(n_rounds, 1.0 / n_actions)
logged_actions = rng.integers(0, n_actions, size=n_rounds)
reward = rng.binomial(1, true_ctr[logged_actions]).astype(float)

print(f"Mean reward in logs: {reward.mean():.4f}")

# --- Политика ИЗ КОНФИГА ---
policy = EpsilonGreedyPolicy(
    n_actions=n_actions,
    epsilon=epsilon,
    seed=seed,
)

context = np.zeros((n_rounds, 1))
base_scores = np.zeros((n_rounds, n_actions))
eval_dist = policy.action_distribution(context, base_scores)

# --- OPE ---
logs = pd.DataFrame({
    "chosen_item": logged_actions,
    "propensity": behavior_pscore,
    "reward": reward,
})
ope_input = build_ope_input(
    logs=logs,
    evaluation_action_dist=eval_dist,
    policy_name="uniform",
)

ips = IPSEstimator(max_weight=15.0)
result = ips.estimate_with_ci(ope_input, n_bootstrap=500, seed=0)
print(f"IPS(ε=1.0):          {result.estimate:.4f}")

expected = reward.mean()
assert abs(result.estimate - expected) < 1e-6, \
    f"IPS должен равняться mean(reward): {result.estimate} vs {expected}"

# --- Проверка, что политика собрана ИЗ КОНФИГА ---
assert policy.n_actions == n_actions, "n_actions не из конфига!"
assert policy.epsilon == epsilon, "epsilon не из конфига!"
print(f"OK: policy собрана из configs/policies.yaml "
      f"(n_actions={policy.n_actions}, epsilon={policy.epsilon})")
print("OK: IPS == mean(reward) — интеграция работает корректно")


# ── Проверка, что все параметры из YAML дошли до политики ──
expected_min = spec.get("min_epsilon", 0.0)
expected_max = spec.get("max_epsilon", 1.0)
expected_subset = spec.get("exploration_subset", None)

assert policy.min_epsilon == expected_min, \
    f"min_epsilon не из конфига: {policy.min_epsilon} vs {expected_min}"
assert policy.max_epsilon == expected_max, \
    f"max_epsilon не из конфига: {policy.max_epsilon} vs {expected_max}"

if expected_subset is not None:
    assert np.array_equal(policy.exploration_subset, np.array(expected_subset)), \
        "exploration_subset не из конфига"

print(f"OK: все параметры политики из configs/policies.yaml")
print(f"    epsilon={policy.epsilon} (clipped в [{policy.min_epsilon}, {policy.max_epsilon}])")