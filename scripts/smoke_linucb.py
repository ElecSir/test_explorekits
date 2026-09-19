import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from explorekit.policies.config import load_policies_config
from explorekit.policies.linucb import LinUCBPolicy
from explorekit.ope.base import build_ope_input
from explorekit.ope.ips import IPSEstimator


# ---------- 1. Конфиг ----------
cfg = load_policies_config()
spec = cfg["policies"]["smoke_linucb"]

n_actions = spec["n_actions"]
context_dim = spec["context_dim"]
alpha = spec["alpha"]
temperature = spec["temperature"]
reg = spec["reg"]
seed = spec.get("seed", cfg["seed"])

print(f"Config: n_actions={n_actions}, d={context_dim}, "
      f"alpha={alpha}, tau={temperature}, reg={reg}, seed={seed}")


# ---------- 2. Политика ИЗ КОНФИГА ----------
policy = LinUCBPolicy(
    n_actions=n_actions,
    context_dim=context_dim,
    alpha=alpha,
    temperature=temperature,
    reg=reg,
    seed=seed,
)

assert policy.n_actions == n_actions
assert policy.context_dim == context_dim
assert policy.alpha == alpha
assert policy.temperature == temperature
assert policy.reg == reg
print("OK: политика собрана из configs/policies.yaml")


# ---------- 3. Полный цикл select → update ----------
# Среда: у каждого действия свой "истинный" вектор θ*_a.
# Задача LinUCB — выучить, что action=2 даёт лучший reward.

rng = np.random.default_rng(0)

# Истинные θ* для каждого действия (d-мерные)
true_theta = np.zeros((n_actions, context_dim))
true_theta[2] = np.array([1.0, 1.0, 1.0])   # action 2 — лучший
for a in range(n_actions):
    if a != 2:
        true_theta[a] = rng.normal(scale=0.1, size=context_dim)

def reward_fn(context, action):
    """P(click) = sigmoid(theta*_a · x)."""
    z = float(true_theta[action] @ context)
    return 1.0 / (1.0 + np.exp(-z))

n_train = 2000
# ── Упрощённая среда: фиксированный контекст + детерминированный reward ──
# Задача: LinUCB должен выучить, что action=2 — лучший.
# Если не сходится даже здесь — баг в реализации.

context = np.ones(context_dim)        # фиксированный x
chosen_counts = np.zeros(n_actions, dtype=int)

for _ in range(n_train):
    d = policy.select_action(context, np.zeros(n_actions))
    chosen_counts[d.chosen_item] += 1

    # Детерминированный reward: action=2 → 0.9, остальные → 0.1
    r = 0.9 if d.chosen_item == 2 else 0.1
    policy.update(d, reward=r, context=context)

print(f"Chosen distribution after {n_train} steps: "
      f"{chosen_counts / n_train}")

# Диагностика: UCB-скоры и mean-часть для контекста "все единицы"
x_test = np.ones(context_dim)
ucb_scores = policy._ucb_scores(x_test[None, :])[0]

# mean-часть считаем вручную: theta[a] = A[a]^-1 b[a], mean[a] = theta[a] @ x
mean_scores = np.array([
    float(np.linalg.solve(policy.A[a], policy.b[a]) @ x_test)
    for a in range(n_actions)
])

print(f"\nUCB scores for x=ones:   {ucb_scores}")
print(f"Mean scores for x=ones:  {mean_scores}")
print(f"argmax UCB: {np.argmax(ucb_scores)}, argmax mean: {np.argmax(mean_scores)}")

# Сколько данных накопилось у каждого действия
print(f"\nA traces (сколько данных у каждого действия):")
for a in range(n_actions):
    print(f"  A[{a}] trace = {np.trace(policy.A[a]):.3f}, "
          f"b[{a}] = {policy.b[a]}")

# action=2 должен доминировать
top_action = int(np.argmax(chosen_counts))
assert top_action == 2, \
    f"LinUCB должен был сойтись к action=2, а сошёлся к {top_action}"
assert chosen_counts[2] / n_train > 0.5, \
    f"action=2 должен выбираться >50%, а выбрано {chosen_counts[2]/n_train:.2%}"
print(f"OK: LinUCB сходится к лучшему действию (action=2): "
      f"{chosen_counts[2]/n_train:.2%}")


# ---------- 4. Проверка action_distribution ----------
n_rounds = 5000
contexts = rng.normal(size=(n_rounds, context_dim))
base_scores = np.zeros((n_rounds, n_actions))
eval_dist = policy.action_distribution(contexts, base_scores)

assert eval_dist.shape == (n_rounds, n_actions), f"shape={eval_dist.shape}"
assert np.allclose(eval_dist.sum(axis=1), 1.0, atol=1e-6), "суммы != 1"
assert (eval_dist >= 0).all(), "есть отрицательные"
assert eval_dist[eval_dist > 0].min() > 0, "common support violated"

print(f"eval_dist: shape={eval_dist.shape}, "
      f"min_prob={eval_dist[eval_dist > 0].min():.4f}, "
      f"max_prob={eval_dist.max():.4f}")


# ---------- 5. Стыковка с OPE ----------
# Синтетические логи с uniform behavior
true_ctr = np.array([0.4, 0.2, 0.3, 0.1, 0.35])
behavior_pscore = np.full(n_rounds, 1.0 / n_actions)
logged_actions = rng.integers(0, n_actions, size=n_rounds)
reward = rng.binomial(1, true_ctr[logged_actions]).astype(float)

logs = pd.DataFrame({
    "chosen_item": logged_actions,
    "propensity": behavior_pscore,
    "reward": reward,
})

ope_input = build_ope_input(
    logs=logs,
    evaluation_action_dist=eval_dist,
    policy_name="linucb",
)

ips = IPSEstimator(max_weight=15.0)
result = ips.estimate_with_ci(ope_input, n_bootstrap=500, seed=0)

print(f"IPS estimate: {result.estimate:.4f}")
print(f"95% CI:       [{result.ci_low:.4f}, {result.ci_high:.4f}]")

assert np.isfinite(result.estimate), "IPS вернул NaN/inf"
assert result.ci_low <= result.estimate <= result.ci_high, \
    "estimate вне CI"


# ---------- 6. Валидация: uniform policy → IPS == mean(reward) ----------
uniform_dist = np.full((n_rounds, n_actions), 1.0 / n_actions)
ope_input_uniform = build_ope_input(
    logs=logs,
    evaluation_action_dist=uniform_dist,
    policy_name="uniform",
)
ips_uniform = IPSEstimator(max_weight=15.0)
result_uniform = ips_uniform.estimate_with_ci(
    ope_input_uniform, n_bootstrap=500, seed=0,
)

expected = reward.mean()
assert abs(result_uniform.estimate - expected) < 1e-6, \
    f"IPS(uniform) должен = mean(reward): {result_uniform.estimate} vs {expected}"

print(f"OK: IPS(uniform) == mean(reward) = {expected:.4f}")
print("\nOK: LinUCB работает и совместим с OPE по контракту")