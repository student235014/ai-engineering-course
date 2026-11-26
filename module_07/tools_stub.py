"""
Заглушка инструментов Модуля 6 (MCP-сервер). Менять не нужно.

Практика Модуля 7 работает на этой заглушке, а не на MCP-сервере Модуля 6:
module_07 автономен (пакет mcp не нужен), а инструменты имеют
ЗАФИКСИРОВАННЫЙ контракт:

    get_equipment_status(equipment_id: str) -> dict
        {"equipment_id", "name", "location", "status", "last_maintenance"}
        неизвестный код -> {"error": "..."}  (словарь, а НЕ исключение)

    create_incident(equipment_id: str, summary: str, severity: str) -> dict
        {"incident_id": "INC-2026-0NN", "status": "created", ...}
        + запись в локальное хранилище data/incidents_store.jsonl

Данные не выдуманы: реестр — module_07/data/equipment_registry.json (копия
реестра Модуля 2, 9 единиц), статус/расположение/дата ТО — из карточек
оборудования корпуса Модуля 3, нумерация инцидентов продолжает журнал
Модуля 3 (в нём записи INC-2026-001 … INC-2026-012).

Настоящий MCP-сервер Модуля 6 (module_06/mcp_server_basic.py) устроен
иначе: собственный реестр из 5 единиц (PUMP-07, SEP-04, ...), сигнатура
create_incident(equipment_id, description, priority, role), ответы вида
{"ok": ...} и приоритеты low/medium/high/critical. Прямой замены импорта
нет — эта заглушка остаётся рабочим набором инструментов агента.
"""

import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
REGISTRY_FILE = DATA_DIR / "equipment_registry.json"
INCIDENTS_STORE = DATA_DIR / "incidents_store.jsonl"

# Журнал Модуля 3 заканчивается на INC-2026-012 — наша нумерация продолжает его
FIRST_INCIDENT_NO = 13

# ----------------------------------------------------------------------
# Статус, расположение и дата ТО — из карточек оборудования корпуса
# Модуля 3 (generated/equipment_cards.json). Разнобой форматов дат и
# регистра — из исходных карточек, он намеренный: так выглядят реальные
# данные, собранные из разных источников.
# ----------------------------------------------------------------------

_CARD_FIELDS = {
    "КМ-101":     {"location": "Цех розлива",             "status": "в работе", "last_maintenance": "01-10-2022"},
    "НМ-205":     {"location": "Участок водоподготовки",  "status": "на ТО",    "last_maintenance": "15-03-2023"},
    "П-7":        {"location": "Участок сборки",          "status": "в работе", "last_maintenance": "05-07-2021"},
    "ЭЛОУ-АВТ-6": {"location": "Нефтеперерабатывающий завод", "status": "в работе", "last_maintenance": "10-12-2022"},
    "Линия-3":    {"location": "Цех розлива",             "status": "в работе", "last_maintenance": "20-09-2023"},
    "ЧПУ-12":     {"location": "участок мехобработки",    "status": "в работе", "last_maintenance": "16.07.2023"},
    "Р-12":       {"location": "цех сборки",              "status": "на ТО",    "last_maintenance": "15.11.2022"},
    "КТ-500.2":   {"location": "производственный цех №3", "status": "в работе", "last_maintenance": "08.09.2023"},
    "ВЕНТ-А1":    {"location": "корпус А",                "status": "в работе", "last_maintenance": "22.06.2022"},
}

# Кириллица и латиница неотличимы на глаз (К/K, М/M…) — модели иногда пишут
# код латиницей. Приём из валидатора Модуля 2: транслитерация двойников.
_LAT_TO_CYR = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")


def _norm(equipment_id: str) -> str:
    """Код к каноническому виду: без краёв, двойники — в кириллицу, верхний регистр."""
    return str(equipment_id).strip().translate(_LAT_TO_CYR).upper()


def _load_registry() -> dict:
    """Реестр оборудования: {нормализованный код: {"id", "name"}}."""
    data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {_norm(rec["id"]): rec for rec in data["equipment"]}


_REGISTRY = _load_registry()


def get_equipment_status(equipment_id: str) -> dict:
    """Статус оборудования по коду из реестра.

    Контракт Модуля 6: словарь с полями equipment_id, name, location,
    status, last_maintenance; для неизвестного кода — {"error": "..."}.
    """
    record = _REGISTRY.get(_norm(equipment_id))
    if record is None:
        return {"error": f"Оборудование с кодом «{equipment_id}» в реестре не найдено"}
    card = _CARD_FIELDS[record["id"]]
    return {
        "equipment_id": record["id"],
        "name": record["name"],
        "location": card["location"],
        "status": card["status"],
        "last_maintenance": card["last_maintenance"],
    }


def _next_incident_id() -> str:
    """Следующий номер: продолжает журнал Модуля 3 и записи этого хранилища."""
    count = 0
    if INCIDENTS_STORE.exists():
        with open(INCIDENTS_STORE, encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
    return f"INC-2026-{FIRST_INCIDENT_NO + count:03d}"


def create_incident(equipment_id: str, summary: str, severity: str) -> dict:
    """Регистрирует инцидент и сохраняет его в локальное хранилище.

    Контракт Модуля 6: возвращает запись с incident_id вида INC-2026-0NN
    и status="created"; запись дописывается в data/incidents_store.jsonl.
    """
    incident = {
        "incident_id": _next_incident_id(),
        "equipment_id": equipment_id,
        "summary": summary,
        "severity": severity,
        "status": "created",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    INCIDENTS_STORE.parent.mkdir(parents=True, exist_ok=True)
    with open(INCIDENTS_STORE, "a", encoding="utf-8") as f:
        f.write(json.dumps(incident, ensure_ascii=False) + "\n")
    return incident


if __name__ == "__main__":
    print("Известное оборудование:")
    print(json.dumps(get_equipment_status("КМ-101"), ensure_ascii=False, indent=2))
    print("\nКод латиницей (KM-101 — K и M латинские):")
    print(json.dumps(get_equipment_status("KM-101"), ensure_ascii=False, indent=2))
    print("\nНеизвестный код:")
    print(json.dumps(get_equipment_status("ГМ-9"), ensure_ascii=False, indent=2))
    print("\nРегистрация инцидента:")
    print(json.dumps(
        create_incident("НМ-205", "Повторный сбой реле давления", "средняя"),
        ensure_ascii=False, indent=2,
    ))
