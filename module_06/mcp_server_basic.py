"""
Модуль 6, Занятие 11 — базовый MCP-сервер: интерфейс к системам и инструменты.

ТВОЯ РАБОТА: TODO 1 (NEW_EQUIPMENT — заполнить учебную запись оборудования),
TODO 2 (get_equipment_status — инструмент чтения статуса оборудования),
TODO 3 (create_incident — инструмент создания заявки на инцидент).

Что такое MCP в двух словах: это протокол, по которому LLM-хост (например,
чат-ассистент) обращается не напрямую к вашей базе/API, а к отдельному
MCP-серверу. Сервер публикует:
  - tools     — действия с чёткой схемой аргументов (что вызвать, с какими
                параметрами, что вернётся);
  - resources — данные для чтения по URI, без вызова инструмента;
  - prompts   — заготовки промптов (в этом занятии не используем).
Хост видит только эти три интерфейса и ничего не знает о внутреннем
устройстве систем ПромТеха — в этом и есть смысл MCP как безопасного
адаптера: LLM не получает прямой доступ к боевой системе, а работает
через контролируемый и журналируемый набор инструментов.

ВАЖНО: этот файл — сам MCP-сервер, а не обычный скрипт. Если запустить его
напрямую (python mcp_server_basic.py), он не выведет ничего в консоль и
«зависнет» — это нормально, сервер ждёт сообщений по протоколу MCP на
stdin. Чтобы увидеть сервер в работе, запусти:

    python mcp_client_demo.py

Скрипт поднимет этот файл как подпроцесс (stdio) и станет MCP-хостом:
ты пишешь обычную фразу («статус PUMP-07»), GigaChat сам выбирает tool
из каталога сервера и вызывает его по протоколу MCP.

Разделение read/write: get_equipment_status ничего не меняет в системе —
его можно вызывать сколько угодно раз (идемпотентная операция чтения).
create_incident каждый раз создаёт новую заявку — операция с побочным
эффектом, поэтому у неё есть дополнительная проверка прав (параметр role).
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import MCPServer

BASE_DIR = Path(__file__).parent
EQUIPMENT_PATH = BASE_DIR / "data" / "equipment_registry.json"
INCIDENTS_PATH = BASE_DIR / "data" / "incidents.json"
AUDIT_LOG_PATH = BASE_DIR / "data" / "audit_log.jsonl"

# Роли, которым разрешено создавать инциденты (write-инструмент).
# "viewer" — только чтение, дефолт нарочно "безопасный": если вызывающий
# не передал роль явно, он получит отказ на create_incident, а не заявку.
ALLOWED_TO_CREATE_INCIDENT = {"engineer", "admin"}
ALLOWED_PRIORITIES = {"low", "medium", "high", "critical"}


# ====================================================================
#  TODO 1 — учебная запись оборудования
# ====================================================================
#
# В боевой системе ПромТеха это оборудование уже стоит на учёте, но в
# нашем учебном реестре (data/equipment_registry.json) его пока нет —
# заполни карточку по описанию ниже и запись подключится к общему
# реестру при старте сервера.
#
# Описание для карточки:
#   Транспортёрная лента ТЛ-9, установлена в цехе дробления,
#   сейчас в работе, дата последнего технического обслуживания — 10
#   февраля 2026 года (формат даты — как в остальных записях реестра,
#   "YYYY-MM-DD").
#
# Сейчас три поля ниже — заглушки None: замени их на значения из описания.
NEW_EQUIPMENT = {
    "id": "CONV-09",
    "name": "Транспортёрная лента ТЛ-9",
    "type": "транспортёр",
    "location": "Цех дробления",
    "status": "в работе",
    "last_service_date": "2026-02-10",
}


# ====================================================================
#  ИНФРАСТРУКТУРА (дано, менять не нужно)
# ====================================================================

class CorporateSystemUnavailable(Exception):
    """Имитация отказа боевой системы ПромТеха (сценарий outage, занятие 12)."""


# Переключатель сбоя «боевой» системы. Сам по себе действует только внутри
# того процесса, где его выставили. mcp_client_demo.py поднимает сервер
# отдельным процессом — поэтому shift_review_m12.py включает сбой через
# файл data/simulate_outage.flag, а сервер читает его при каждом вызове.
SIMULATE_OUTAGE = False
OUTAGE_FLAG_PATH = BASE_DIR / "data" / "simulate_outage.flag"


def outage_is_on() -> bool:
    return SIMULATE_OUTAGE or OUTAGE_FLAG_PATH.exists()


def set_simulate_outage(on: bool) -> None:
    """Включает/выключает сбой и для текущего процесса, и для MCP-сервера в хосте."""
    global SIMULATE_OUTAGE
    SIMULATE_OUTAGE = bool(on)
    if on:
        OUTAGE_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTAGE_FLAG_PATH.write_text("1", encoding="utf-8")
    elif OUTAGE_FLAG_PATH.exists():
        OUTAGE_FLAG_PATH.unlink()


def _load_equipment() -> dict:
    """Читает реестр оборудования и подмешивает учебную запись (TODO 1)."""
    registry = json.loads(EQUIPMENT_PATH.read_text(encoding="utf-8-sig"))
    by_id = {item["id"]: item for item in registry["equipment"]}
    by_id[NEW_EQUIPMENT["id"]] = NEW_EQUIPMENT
    return by_id


EQUIPMENT = _load_equipment()


def _load_incidents() -> list:
    if not INCIDENTS_PATH.exists():
        return []
    return json.loads(INCIDENTS_PATH.read_text(encoding="utf-8-sig"))


def _save_incident(incident: dict) -> None:
    incidents = _load_incidents()
    incidents.append(incident)
    INCIDENTS_PATH.write_text(
        json.dumps(incidents, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def log_action(action: str, params: dict, result: dict) -> None:
    """Журналирует каждый вызов инструмента: кто, что и с каким итогом.

    В промышленном контуре это не «для галочки» — по такому логу потом
    разбирают инциденты и проверяют, что ИИ-ассистент не вышел за рамки
    разрешённых действий. Формат — JSON Lines: одна запись — одна строка,
    удобно дописывать и удобно потом построчно прочитать при аудите.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "params": params,
        "ok": bool(result.get("ok")),
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _simulate_corporate_lookup(equipment_id: str) -> None:
    """Заглушка вызова реальной корпоративной системы ПромТеха.

    В учебном проекте статус и так лежит в локальном реестре, но в реальном
    MCP-сервере get_equipment_status обычно дергает внутренний API/БД —
    и этот вызов может упасть (сеть, сервис не отвечает, права не те).
    Этой функцией моделируем такой отказ: переменная SIMULATE_OUTAGE
    или файл data/simulate_outage.flag (его ставит shift_review_m12.py).
    """
    if outage_is_on():
        raise CorporateSystemUnavailable(
            "Корпоративная система ПромТеха недоступна (имитация сбоя)"
        )


app = MCPServer(
    name="promtech-equipment",
    instructions=(
        "Учебный MCP-сервер ПромТеха: узнать статус оборудования и "
        "зарегистрировать инцидент без прямого доступа к боевой системе."
    ),
)


@app.resource("equipment://list")
def list_equipment() -> str:
    """Ресурс: полный реестр оборудования для чтения хостом без вызова tool.

    Отличие ресурса от инструмента: инструмент — это действие с аргументами
    («покажи статус вот этого id»), ресурс — просто данные по фиксированному
    адресу (URI), которые хост может прочитать заранее, не дожидаясь вопроса
    пользователя.
    """
    return json.dumps(list(EQUIPMENT.values()), ensure_ascii=False, indent=2)


# ====================================================================
#  TODO 2 — инструмент чтения: get_equipment_status
# ====================================================================

@app.tool()
def get_equipment_status(equipment_id: str) -> dict:
    """Возвращает текущий статус оборудования по его id.

    TODO 2: инструмент чтения — идемпотентный, без побочных эффектов.
    Каждая ветка возвращает словарь с ключом "ok" и логируется через log_action.
    """
    # 1. Проверка существования
    if equipment_id not in EQUIPMENT:
        result = {"ok": False, "error": f"Оборудование {equipment_id} не найдено"}
        log_action("get_equipment_status", {"equipment_id": equipment_id}, result)
        return result

    # 2. Обращение к «боевой системе» с перехватом сбоя
    try:
        _simulate_corporate_lookup(equipment_id)
    except CorporateSystemUnavailable as exc:
        result = {"ok": False, "error": str(exc)}
        log_action("get_equipment_status", {"equipment_id": equipment_id}, result)
        return result

    # 3. Успех — отдаём всю карточку
    item = EQUIPMENT[equipment_id]
    result = {
        "ok": True,
        "equipment_id": item["id"],
        "name": item["name"],
        "type": item["type"],
        "location": item["location"],
        "status": item["status"],
        "last_service_date": item["last_service_date"],
    }
    log_action("get_equipment_status", {"equipment_id": equipment_id}, result)
    return result


# ====================================================================
#  TODO 3 — инструмент записи: create_incident
# ====================================================================

@app.tool()
def create_incident(
    equipment_id: str,
    description: str,
    priority: str,
    role: str = "viewer",
) -> dict:
    """Создаёт заявку на инцидент по оборудованию и сохраняет её локально.

    TODO 3: порядок проверок важен — права, затем оборудование, затем priority.
    Каждая ветка логируется через log_action перед return.
    """
    audit_params = {
        "equipment_id": equipment_id,
        "priority": priority,
        "role": role,
    }

    # 1. Права доступа — ПЕРВОЙ
    if role not in ALLOWED_TO_CREATE_INCIDENT:
        result = {
            "ok": False,
            "error": f"Роль {role!r} не имеет прав на создание инцидентов",
        }
        log_action("create_incident", audit_params, result)
        return result

    # 2. Существование оборудования
    if equipment_id not in EQUIPMENT:
        result = {
            "ok": False,
            "error": f"Оборудование {equipment_id} не найдено",
        }
        log_action("create_incident", audit_params, result)
        return result

    # 3. Корректность приоритета
    if priority not in ALLOWED_PRIORITIES:
        result = {
            "ok": False,
            "error": (
                f"Недопустимый приоритет {priority!r}, "
                f"ожидается один из {sorted(ALLOWED_PRIORITIES)}"
            ),
        }
        log_action("create_incident", audit_params, result)
        return result

    # 4. Все проверки пройдены — создаём
    incident = {
        "id": uuid.uuid4().hex[:8],
        "equipment_id": equipment_id,
        "description": description,
        "priority": priority,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_incident(incident)
    result = {"ok": True, "incident": incident}
    log_action("create_incident", audit_params, result)
    return result


if __name__ == "__main__":
    app.run()
