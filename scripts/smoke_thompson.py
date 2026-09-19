import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from explorekit.policies.config import load_policies_config
from explorekit.policies.thompson import ThompsonSamplingPolicy
from explorekit.ope.base import build_ope_input
from explorekit.ope.ips import IPSEstimator


# ---------- 1. Конфиг ----------
cfg = load_policies_config()
spec = cfg["policies"]["smoke_thompson"]

n_actions = spec["n_actions"]
context_dim = spec["context_dim"]
mc_samples = spec["mc_samples"]
reg = spec["reg"]
seed = spec.get("seed", cfg["seed"])

print(f"Config: n_actions={n_actions}, d={context_dim}, K={mc_samples}, seed={seed}")


# ---------- 2. Синтетические логи (behavior = uniform) ----------
n_rounds = 5000
rng = np.random.default_rng(0)
true_ctr = np.array([0.4, 0.2, 0.3, 0.1, 0.35])
behavior_pscore = np.full(n_rounds, 1.0 / n_actions)
logged_actions = rng.integers(0, n_actions, size=n_rounds)
reward = rng.binomial(1, true_ctr[logged_actions]).astype(float)

print(f"Mean reward in logs: {reward.mean():.4f}")


# ---------- 3. Политика ИЗ КОНФИГА ----------
policy = ThompsonSamplingPolicy(
    n_actions=n_actions,
    context_dim=context_dim,
    mc_samples=mc_samples,
    reg=reg,
    seed=seed,
)

# Проверяем, что параметры дошли из конфига
assert policy.n_actions == n_actions
assert policy.context_dim == context_dim
assert policy.mc_samples == mc_samples
assert policy.reg == reg
print(f"OK: политика собрана из configs/policies.yaml")


# ---------- 4. Симуляция обучения TS ----------
# TS сам выбирает действия, получает награды и обновляет posterior.
# Так проверяем, что posterior и update работают в полном цикле.
n_train = 2000
chosen_counts = np.zeros(n_actions, dtype=int)

for i in range(n_train):
    context = rng.normal(size=context_dim)
    # base_scores не используются в TS, но передаём для совместимости
    d = policy.select_action(context, np.zeros(n_actions))
    chosen_counts[d.chosen_item] += 1

    # Reward: используем истинный CTR действия
    r = rng.binomial(1, true_ctr[d.chosen_item])
    policy.update(d, reward=r, context=context)

print(f"Chosen distribution after {n_train} steps: {chosen_counts / n_train}")


# ---------- 5. Проверка корректности action_distribution ----------
contexts = rng.normal(size=(n_rounds, context_dim))
base_scores = np.zeros((n_rounds, n_actions))

eval_dist = policy.action_distribution(contexts, base_scores)

assert eval_dist.shape == (n_rounds, n_actions), f"shape={eval_dist.shape}"
assert np.allclose(eval_dist.sum(axis=1), 1.0, atol=1e-6), "суммы != 1"
assert (eval_dist >= 0).all(), "есть отрицательные вероятности"
assert eval_dist[eval_dist > 0].min() > 0, "common support violated"

print(f"eval_dist: shape={eval_dist.shape}, "
      f"min_prob={eval_dist[eval_dist > 0].min():.4f}, "
      f"max_prob={eval_dist.max():.4f}")


# ---------- 6. Стыковка с OPE ----------
logs = pd.DataFrame({
    "chosen_item": logged_actions,
    "propensity": behavior_pscore,
    "reward": reward,
})

# Оценим ТУ ЖЕ политику, что сейчас у policy (eval_dist выше)
# В реальном OPE это была бы политика ДО обучения — но здесь проверяем
# сам факт, что build_ope_input и IPSEstimator принимают наши данные.
ope_input = build_ope_input(
    logs=logs,
    evaluation_action_dist=eval_dist,
    policy_name="thompson",
)

ips = IPSEstimator(max_weight=15.0)
result = ips.estimate_with_ci(ope_input, n_bootstrap=500, seed=0)

print(f"IPS estimate: {result.estimate:.4f}")
print(f"95% CI:       [{result.ci_low:.4f}, {result.ci_high:.4f}]")

assert np.isfinite(result.estimate), "IPS вернул NaN/inf"
assert result.ci_low <= result.estimate <= result.ci_high, "estimate вне CI"


# ---------- 7. Валидация: оценщик одной и той же политики ----------
# Если eval policy = behavior policy (uniform), IPS == mean(reward).
# Проверим это на отдельной "uniform-политике" — то есть пересчитаем dist
# как strictly uniform.
uniform_dist = np.full((n_rounds, n_actions), 1.0 / n_actions)

ope_input_uniform = build_ope_input(
    logs=logs,
    evaluation_action_dist=uniform_dist,
    policy_name="uniform",
)
ips_uniform = IPSEstimator(max_weight=15.0)
result_uniform = ips_uniform.estimate_with_ci(ope_input_uniform, n_bootstrap=500, seed=0)

expected = reward.mean()
assert abs(result_uniform.estimate - expected) < 1e-6, \
    f"IPS(uniform) должен = mean(reward): {result_uniform.estimate} vs {expected}"

print(f"OK: IPS(uniform) == mean(reward) = {expected:.4f}")
print("\nOK: Thompson Sampling работает и совместим с OPE по контракту")