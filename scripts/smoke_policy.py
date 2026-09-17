import numpy as np
import pandas as pd
from explorekit.policies.epsilon_greedy import EpsilonGreedyPolicy
from explorekit.ope.base import build_ope_input
from explorekit.ope.ips import IPSEstimator
from experiments._stub_environment import make_replication

# 1. Готовим "логи" через стаб Полины
rep = make_replication(seed=0, n_rounds=5000, epsilon=0.1)
ope_input = rep.ope_input
# ope_input.behavior_pscore, ope_input.reward, ope_input.evaluation_pscore уже есть

# 2. Симулируем "сырые логи" (как будто их написал участник 4)
#    chosen_item — int-индекс, reward, propensity = behavior_pscore
logs = pd.DataFrame({
    "chosen_item": np.argmax(ope_input.evaluation_pscore[:, None] == ope_input.evaluation_pscore, axis=1),  # заглушка
    "propensity": ope_input.behavior_pscore,
    "reward": ope_input.reward,
})

# 3. Твоя политика считает распределение для ВСЕХ раундов батчем
policy = EpsilonGreedyPolicy(n_actions=10, epsilon=1.0, seed=0)
context = np.zeros((len(logs), 5))          # (n_rounds, n_features)
base_scores = np.zeros((len(logs), 10))     # (n_rounds, n_actions)

eval_dist = policy.action_distribution(context, base_scores)
# eval_dist.shape == (5000, 10)

# 4. Собираем OPEInput через её адаптер
ope_input_new = build_ope_input(
    logs=logs,
    evaluation_action_dist=eval_dist,
    policy_name="my_epsilon_greedy",
)

# 5. Запускаем IPS
estimator = IPSEstimator(max_weight=15.0)
result = estimator.estimate_with_ci(ope_input_new, n_bootstrap=500, seed=42)
print(f"IPS: {result.format_ctr()}")

# 6. Сравниваем с истиной
print(f"True CTR: {rep.true_value:.4f}")

print(f"Mean reward in logs: {logs['reward'].mean():.4f}")
print(f"IPS(ε=1.0):          {result.estimate:.4f}")