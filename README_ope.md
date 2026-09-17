# OPE-модуль (участник 2 — Полина)

Раздел для включения в основной README. Описывает офлайн-оценку политик:
оценщики, доверительные интервалы, диагностику надёжности и валидацию.

## Быстрый старт

```bash
pip install -r requirements.txt

# данные в Git не хранятся — скачиваем (открытый доступ, без авторизации)
bash scripts/download_obd.sh

# демонстрационный notebook со всем сценарием
jupyter notebook notebooks/ope_demo.ipynb

# валидация оценщиков на синтетике с известной истиной
python experiments/ope_synthetic.py --config configs/ope.yaml

# оценка политик на реальном Open Bandit Dataset
python experiments/open_bandit.py --config configs/ope.yaml

# тесты модуля
pytest tests/
```

## Публичный API

```python
from explorekit.ope import (
    OPEInput, OPEResult, build_ope_input,
    IPSEstimator, SNIPSEstimator, DREstimator,
    RewardModel, split_for_reward_model,
    bootstrap_ci, compute_reliability,
)
```

Типовое использование:

```python
ope_input = build_ope_input(
    logs=decision_log_df,                    # DecisionLog от участника 4
    evaluation_action_dist=action_dist,      # (n, n_actions) от участника 1
    policy_name="moderate(eps=0.1)",
    q_hat_all_actions=q_hat,                 # опционально, нужно для DR
)

result = DREstimator(max_weight=15.0).estimate_with_ci(
    ope_input, n_bootstrap=1000, confidence_level=0.95, seed=42
)
print(result.format_ctr())          # '0.36% [0.34%, 0.39%]'

reliability = compute_reliability(ope_input, max_weight=15.0)
print(reliability.status.value)     # 'Reliable' / 'Use with caution' / 'Unreliable'
print(reliability.explanation)      # человекочитаемая причина
```

## Контракт с остальными модулями

Единственная точка стыковки — функция `build_ope_input`. От неё требуется:

| Что | Откуда | Формат |
|---|---|---|
| `logs` | участник 4 | DataFrame с колонками `chosen_item` (int-индекс действия), `propensity`, `reward` |
| `evaluation_action_dist` | участник 1 | `(n_rounds, n_actions)`, строки суммируются в 1 |
| `q_hat_all_actions` | участник 2 (`RewardModel`) | `(n_rounds, n_actions)`, опционально |
| истинный CTR политики | участник 3 | `float`, только для валидации |

Всё внутри `explorekit.ope` работает исключительно с `OPEInput` и ничего не
знает про симулятор, Parquet и классы политик — поэтому модуль тестируется
независимо от готовности остальных.

Важно: `chosen_item` должен быть индексом в диапазоне `[0, n_actions)`.
Если в DecisionLog хранятся строковые item_id, маппинг делается до вызова.

## Что реализовано

**Оценщики** (`ips.py`, `snips.py`, `dr.py`) — общий базовый класс
`OPEEstimator`, у всех одинаковый интерфейс `estimate` / `estimate_with_ci`.
Клиппинг весов настраивается через `max_weight` из конфига.

**Bootstrap CI** (`bootstrap.py`) — percentile bootstrap, один helper на все
оценщики. Seed только из конфига, `np.random.default_rng`.

**Диагностика** (`diagnostics.py`) — ESS, ESS/N, max/mean/p99 вес, доля
обрезанных весов, нарушения common support. Диагностика дисперсии:
дисперсия весов, коэффициент вариации CV = std(w)/mean(w) и аналитическая
стандартная ошибка IPS (дополняет bootstrap — заметное расхождение между
ними означает, что распределение слишком тяжелохвостое и ЦПТ на таком n
ещё не работает). Выдаёт статус `Reliable` / `Use with caution` /
`Unreliable` плюс текстовое объяснение причины.

**Валидация** (`validation.py`) — Bias, Std, RMSE и CI coverage по 50–100
прогонам симулятора с известной истиной.

**Загрузчик OBD** (`datasets/open_bandit.py`) — приводит Open Bandit Dataset
к общему формату проекта.

## Результаты проверок

### Сверка с Open Bandit Pipeline

Наши формулы сверены с независимой реализацией авторов датасета
(`tests/test_obp_crosscheck.py`) — совпадение до машинной точности:

| Оценщик | наш | obp | расхождение |
|---|---|---|---|
| IPS | 0.0003000000 | 0.0003000000 | 0.00e+00 |
| SNIPS | 0.0003595398 | 0.0003595398 | 0.00e+00 |
| DR | 0.0007826407 | 0.0007826407 | 1.41e-18 |

### Synthetic validation (50 прогонов, истинный CTR 5.28%)

| Оценщик | Среднее | Bias | Std | RMSE | CI coverage |
|---|---|---|---|---|---|
| IPS(clip=15) | 0.0510 | −0.0004 | 0.0067 | 0.0064 | 96% |
| SNIPS(clip=15) | 0.0506 | −0.0007 | 0.0064 | 0.0062 | 98% |
| DR(clip=15) | 0.0510 | −0.0004 | 0.0064 | 0.0062 | 94% |

Смещение у всех трёх около нуля, CI coverage близок к заявленным 95%,
у DR наименьший RMSE и самый узкий интервал — ожидаемое поведение.

Отдельно проверено само свойство двойной робастности: DR попадает в истину
и когда reward model заведомо кривая, и когда испорчены propensity
(`tests/test_dr.py::TestDoubleRobustness`).

## Известные ограничения

**На текущем сэмпле OBD все оценки помечаются как Unreliable.** Это не баг:
в примере датасета всего 38 кликов на 10 000 показов, ESS падает до 58–100
из 5000. IPS даёт 0.03%, DR — 0.36%; расхождение в 10 раз возникает потому,
что при таком числе кликов DR почти целиком определяется reward-моделью, а
IPS-поправка шумит. Для отчёта нужен полный датасет (26 млн показов,
https://github.com/st-tech/zr-obp) — код при этом не меняется, только путь
в конфиге.

**Стаб симулятора.** `experiments/_stub_environment.py` — временная замена
модуля участника 3, реализует контракт `make_replication(seed) -> Replication`.
Когда настоящий симулятор будет готов, меняется одна строка импорта в
`experiments/ope_synthetic.py`; код валидации остаётся прежним.

**Заглушка torch.** Библиотека `obp` импортирует `torch` на верхнем уровне
ради аннотаций типов, хотя numpy-путь его не использует. Ставить ~8 ГБ
CUDA-зависимостей ради этого нецелесообразно, поэтому в
`experiments/_obp_crosscheck.py` подставляется минимальная заглушка. Если в
окружении есть настоящий torch, используется он. Тесты сверки
пропускаются (skip), если `obp` недоступен, и не блокируют CI команды.

## Данные

Датасет в репозитории не хранится (`data/obd/` в `.gitignore`). Загрузка:

```bash
bash scripts/download_obd.sh          # по умолчанию в data/obd
```

Скрипт качает сэмпл (10 000 записей на пару campaign/policy) напрямую из
открытого репозитория авторов, без авторизации, и проверяет наличие
обязательных колонок после загрузки.

| Что | Ссылка |
|---|---|
| Сэмпл (используется по умолчанию) | https://github.com/st-tech/zr-obp/tree/master/obd |
| Полная версия, 26 млн записей | https://research.zozo.com/data.html |
| Статья | https://arxiv.org/abs/2008.07146 |
| Лицензия данных | Apache 2.0 (см. репозиторий zr-obp) |

Синтетические данные генерируются воспроизводимо с seed из конфига, скрипт —
`experiments/_stub_environment.py`.

## Параметры конфига

Все алгоритмические параметры — в `configs/ope.yaml`, в коде не захардкожено
ничего: `max_weight`, `n_bootstrap`, `confidence_level`, параметры reward
model, пороги статусов диагностики, число прогонов валидации, epsilon
оцениваемых политик.

Порог `max_weight` подбирается по диагностике: если `fraction_clipped`
уходит выше ~5%, смещение от обрезки становится существенным и его надо
указывать в отчёте.

## Definition of done

- [x] код запускается (`experiments/ope_synthetic.py`, `experiments/open_bandit.py`)
- [x] notebook `notebooks/ope_demo.ipynb` исполняется целиком без ошибок
- [x] есть тесты — 57 тестов, `pytest` зелёный
- [x] `ruff check .` без замечаний, `black` применён
- [x] параметры вынесены в `configs/ope.yaml`
- [x] результат воспроизводится с seed
- [x] модуль стыкуется с остальными через `build_ope_input`
