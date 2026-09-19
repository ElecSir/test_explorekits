# Bandit Policies (участник 1)

Раздел для включения в основной README. Описывает bandit-слой ExploreKit:
политики exploration, контракт `BasePolicy` / `Decision`, конфигурируемость
через YAML, стыковку с OPE-модулем.

## Быстрый старт

```bash
pip install -r requirements.txt

# юнит-тесты политик
pytest tests/test_epsilon_greedy.py tests/test_linucb.py tests/test_thompson.py -v

# smoke-тесты (стыковка с OPE + сходимость)
python -m scripts.smoke_ope_uniform
python -m scripts.smoke_linucb
python -m scripts.smoke_thompson

# ноутбук с демонстрацией (доработанный от OPE)
jupyter notebook notebooks/ope_policies.ipynb
```

Все параметры — в configs/policies.yaml. Меняешь YAML — меняется поведение, код можно не изменять.

## Публичный API

```python
from explorekit.policies import (
    BasePolicy, Decision,
    EpsilonGreedyPolicy, LinUCBPolicy, ThompsonSamplingPolicy,
    load_policies_config, build_policy, build_active_policy,
)
```

Типовое использование:

```python
from explorekit.policies import load_policies_config, build_active_policy

cfg = load_policies_config()                       # читает configs/policies.yaml
policy = build_active_policy(cfg)                  # собирает политику по active_policy

# батчевый режим (для OPE)
dist = policy.action_distribution(context, base_scores)
# dist.shape == (n_rounds, n_actions), сумма по оси 1 = 1

# пошаговый режим (для симулятора)
decision = policy.select_action(context, base_scores, candidate_items=None)
print(decision.chosen_item, decision.propensity, decision.is_exploration)

# обновление после reward
policy.update(decision, reward=1.0, context=context)
```

## Контракт с остальными модулями

Единственная точка стыковки — метод `action_distribution`. От него требуется:

| Что | Кому | Формат |
|---|---|---|
| `context` | от участника 3 (симулятор) | `(n_rounds, d)` или `(d,)` для одного раунда |
| `base_scores` | от базового ранкера | `(n_rounds, n_actions)` или `(n_actions,)` |
| `candidate_items` | опционально | int-массив индексов `[0, n_actions)` |
| `action_distribution` → `(n_rounds, n_actions)` | участнику 2 (OPE) | строки суммируются в 1, все > 0 |
| `Decision` | участнику 4 (логгер) | `chosen_item: int`, `propensity: float`, `probabilities: np.ndarray`, `is_exploration: bool` |

Важно: `chosen_item` — **int-индекс** в диапазоне `[0, n_actions)`, не
строковый `item_id`. Маппинг `item_id → index` делается до вызова политики
(на стороне симулятора или логгера).

### Совместимость с `ActionDistributionFn` (участник 2)

Протокол `explorekit.ope.base.ActionDistributionFn` описывает метод
`__call__(context, base_scores) -> np.ndarray`. Метод
`BasePolicy.action_distribution(context, base_scores, candidate_items=None)`
структурно совместим: третий аргумент опционален, поэтому вызов
`fn(ctx, scores)` работает без изменений. Адаптеры не нужны.

## Что реализовано

**`Decision`** (`base.py`) — dataclass с инвариантами:
- `chosen_item` — int-индекс, не строка;
- `propensity == probabilities[chosen_item]` и > 0;
- `probabilities.sum() == 1`.

Нарушение → `ValueError` **до** того, как некорректные данные попадут в
логи. Это защищает OPE от мусора.

**`BasePolicy`** (`base.py`) — ABC с тремя `@abstractmethod`:
`action_distribution`, `select_action`, `update`. Наследник обязан
реализовать все три, иначе `TypeError` при создании объекта.

**`EpsilonGreedyPolicy`** (`epsilon_greedy.py`) — P(best) = 1 − ε + ε/|E|,
P(other in E) = ε/|E|. Поддерживает:
- `min_epsilon` / `max_epsilon` — клиппинг (защита от слишком агрессивного exploration);
- `whitelist` / `blacklist` — int-индексы разрешённых/запрещённых айтемов;
- `exploration_subset` — ε размазывается только по подмножеству.

**`LinUCBPolicy`** (`linucb.py`) — Linear UCB.
Состояние `A[a]`, `b[a]`, `θ[a] = A[a]⁻¹ b[a]`. Score:
`θ[a]ᵀx + α · sqrt(xᵀ A[a]⁻¹ x)`. Обновление: `A[a] += xxᵀ`, `b[a] += r · x`.
Использует `np.linalg.solve` вместо явного `inv` — численно устойчивее.

**Stochastic wrapper для OPE.** Чистый argmax даёт `propensity = 0` для
не-argmax действий → OPE математически невозможен. Решение — softmax поверх
UCB-скоров. Параметры **разделены**:
- `temperature` — как политика **действует** (`select_action`);
- `ope_temperature` — как её **оцениваем офлайн** (`action_distribution`).

Разделение введено, потому что резкий softmax (τ=0.1) даёт пропенсити
~1e-44 для не-argmax действий, что убивает ESS и делает OPE ненадёжным.

**`ThompsonSamplingPolicy`** (`thompson.py`) — Linear Thompson Sampling. Апостериор: `θ[a] ~ N(θ̂[a], A[a]⁻¹)`,
сэмплирование через Cholesky. Propensity — Monte Carlo: `K` сэмплов
параметров, доля побед действия. Это **то же распределение**, из которого
делается выбор — иначе `propensity = 0` при малых `K`.

**Конфигурируемость** (`config.py`) — все параметры в `configs/policies.yaml`:
тип политики, `epsilon`, `alpha`, `temperature`, `ope_temperature`, `reg`,
`mc_samples`, `min/max_epsilon`, `exploration_subset`, seed. Один глобальный
seed, политики наследуют; отдельный seed только у smoke-политик.

## Результаты проверок

### Юнит-тесты

| Модуль | Тестов | Что проверяется |
|---|---|---|
| `test_epsilon_greedy.py` | 16 | сумма = 1, неотрицательность, propensity = probs[chosen], clip, subset, fallback |
| `test_linucb.py` | 12 | UCB-формула, update A/b, softmax, ope_temperature изолирован |
| `test_thompson.py` | 10 | Cholesky-сэмплирование, MC-propensity, сходимость в полном цикле |

Итого: **38 тестов**, `pytest tests/test_epsilon_greedy.py tests/test_linucb.py tests/test_thompson.py -v` — зелёный.

### Smoke-тесты (интеграция с OPE)

`scripts/smoke_ope_uniform.py`, `scripts/smoke_linucb.py`,
`scripts/smoke_thompson.py`:

| Проверка | Результат |
|---|---|
| YAML читается, параметры доходят до политик | ✅ |
| `action_distribution` возвращает `(n, A)`, сумма = 1, все > 0 | ✅ |
| LinUCB сходится к лучшему действию на тривиальной среде | ✅ >99% за 2000 шагов |
| `IPS(uniform) == mean(reward)` | ✅ до машинной точности |
| Воспроизводимость: тот же seed → тот же результат | ✅ |

Smoke-тесты **dimension-agnostic** — работают при `n_actions=5..80`,
`context_dim=3..10`. Проверено на обеих конфигурациях.

## Известные ограничения

**Скорость при больших `n_actions`.** LinUCB делает `n_actions` инверсий
матриц `d×d` на каждый батч. При `n_actions=80, d=3` это ~10 секунд на
2000 шагов. В продакшене Recsflow с `n_actions=500, d=100` потребуется
оптимизация (Sherman-Morrison для обновления `A⁻¹` за O(d²), батчинг,
разреженные матрицы).

**`context_dim` на реальных данных.** В OBD контекст — one-hot
пользовательских признаков (~60). В проде Recsflow будет `d ≈ 30–150` в
зависимости от схемы фичей. Проверено на `d=10`, для `d=100` нужно
отдельно замерить скорость.

**Hyperparameter tuning.** `alpha`, `reg`, `temperature`, `ope_temperature`
сейчас — разумные дефолты, не оптимизированные. В проде их нужно тюнить
офлайн по сетке.

**Cold-Item Booster.** Вынесен в отдельный модуль (участник 3), не входит
в bandit-слой. `BasePolicy` поддерживает `set_item_metadata()` для
совместимости, но логика буста живёт в другом модуле.

**Whitelist/blacklist по категориям.** Сейчас работают по int-индексам
айтемов, не по категориям. Маппинг `item_id → category` — на стороне
симулятора.

## Параметры конфига

Все алгоритмические параметры — в `configs/policies.yaml`. В коде не
захардкожено ничего.

`active_policy` позволяет переключать политику без правки кода — одна
строка в YAML.

## Definition of done

- [x] код запускается (`pytest tests/test_epsilon_greedy.py tests/test_linucb.py tests/test_thompson.py -v`, `python -m scripts.smoke_*.py`)
- [x] есть тесты — 38 юнит-тестов + 3 smoke, всё зелёное
- [x] параметры вынесены в `configs/policies.yaml`
- [x] результат воспроизводится с seed
- [x] модуль стыкуется с OPE-модулем через `action_distribution`
- [x] `Decision` защищает от некорректных логов (инварианты)
