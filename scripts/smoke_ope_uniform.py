import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from explorekit.policies.epsilon_greedy import EpsilonGreedyPolicy
from explorekit.ope.base import build_ope_input
from explorekit.ope.ips import IPSEstimator


# --- Свои логи, где behavior = uniform random ---
n_actions = 5
n_rounds = 5000
rng = np.random.default_rng(0)

# True CTR по действиям (это наша "скрытая истина")
true_ctr = np.array([0.4, 0.2, 0.3, 0.1, 0.35])

# Behavior = uniform
behavior_pscore = np.full(n_rounds, 1.0 / n_actions)
logged_actions = rng.integers(0, n_actions, size=n_rounds)
reward = rng.binomial(1, true_ctr[logged_actions]).astype(float)

print(f"Mean reward in logs: {reward.mean():.4f}")

# Твоя политика с ε=1.0 → тоже uniform
policy = EpsilonGreedyPolicy(n_actions=n_actions, epsilon=1.0, seed=0)
context = np.zeros((n_rounds, 1))
base_scores = np.zeros((n_rounds, n_actions))

eval_dist = policy.action_distribution(context, base_scores)
# eval_dist[i, a] = 1/n_actions для всех i, a

# Собираем OPEInput
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

# Проверка: IPS должен ТОЧНО совпасть с mean(reward), т.к. w=1
expected = reward.mean()
assert abs(result.estimate - expected) < 1e-6, \
    f"IPS должен равняться mean(reward): {result.estimate} vs {expected}"
print("OK: IPS == mean(reward) — интеграция работает корректно")