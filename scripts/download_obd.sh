#!/usr/bin/env bash
#
# Загрузка Open Bandit Dataset (сэмпл 10 000 записей на пару campaign/policy).
# Сам датасет в Git не хранится — требование формата сдачи.
#
# Источник (открытый доступ, без авторизации):
#   https://github.com/st-tech/zr-obp/tree/master/obd
# Полная версия (26 млн записей):
#   https://research.zozo.com/data.html
#
# Лицензия данных: Apache 2.0 (см. репозиторий zr-obp).
# Цитирование: Saito, Aihara, Matsutani, Narita.
#   "Open Bandit Dataset and Pipeline", https://arxiv.org/abs/2008.07146

set -euo pipefail

TARGET="${1:-data/obd}"
BASE="https://raw.githubusercontent.com/st-tech/zr-obp/master/obd"

echo "Загрузка Open Bandit Dataset в ${TARGET}"

for policy in random bts; do
  for campaign in all men women; do
    dir="${TARGET}/${policy}/${campaign}"
    mkdir -p "${dir}"
    for file in "${campaign}.csv" "item_context.csv"; do
      url="${BASE}/${policy}/${campaign}/${file}"
      echo "  ${policy}/${campaign}/${file}"
      curl -fsSL -o "${dir}/${file}" "${url}"
    done
  done
done

echo
echo "Готово. Проверка:"
python3 - "$TARGET" << 'PY'
import sys, pathlib, csv
target = pathlib.Path(sys.argv[1])
path = target / "random" / "all" / "all.csv"
with path.open(encoding="utf-8") as handle:
    header = next(csv.reader(handle))
    rows = sum(1 for _ in handle)
required = {"item_id", "position", "click", "propensity_score"}
missing = required - set(header)
if missing:
    raise SystemExit(f"ОШИБКА: в {path} нет колонок {sorted(missing)}")
print(f"  {path}: {rows} строк, обязательные колонки на месте")
PY
