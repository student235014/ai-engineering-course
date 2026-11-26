
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_gigachat.chat_models import GigaChat

from rag_stub import answer as rag_answer
from tools_stub import get_equipment_status as equipment_status_impl

load_dotenv(Path(__file__).parent.parent / ".env")

TICKETS_FILE = Path(__file__).parent / "data" / "tickets.jsonl"

# ------------------------------------------------------------------
# TODO 3. Лимит шагов цикла. Один шаг = один вызов модели.
#
# Штатному сценарию хватает трёх-четырёх шагов. Поставь лимит с запасом,
# но КОНЕЧНЫЙ: зациклившийся агент без лимита будет жечь токены, пока их
# не отберёт провайдер. Это требование к любому агентному циклу.
# ------------------------------------------------------------------
MAX_STEPS = 6

# Инструменты, которые МЕНЯЮТ состояние. Их агент не вызывает сам —
# только после явного подтверждения человеком (human-in-the-loop).
WRITE_TOOLS = {"create_ticket"}

SYSTEM_PROMPT = (
    "Ты — ассистент технической поддержки промышленного предприятия ПромТех. "
    "Решай задачу пользователя по шагам, вызывая инструменты.\n"
    "Правила:\n"
    "1. Факты бери только из инструментов. Не выдумывай ни статусы "
    "оборудования, ни содержание документов.\n"
    "2. Прежде чем оформлять заявку на обслуживание, проверь статус "
    "оборудования.\n"
    "3. Если в задаче нет кода оборудования из реестра — попроси уточнить "
    "код, а не угадывай его.\n"
    "4. Если данные инструментов противоречат друг другу, назови противоречие "
    "и оставь решение за человеком.\n"
    "5. Критические действия (запуск, останов оборудования) не выполняй и не "
    "советуй выполнять без специалиста.\n"
    "6. Отвечай обычным текстом, без LaTeX и математической разметки."
)


# ====================================================================
#  ИНСТРУМЕНТЫ
# ====================================================================
# Описание инструмента (докстринг) и схема аргументов (сигнатура + Args) —
# это ВСЁ, что агент знает об инструменте. Он не видит код функции: решение
# «вызывать или нет» принимается только по этому тексту. Два готовых
# инструмента ниже — образец того, как такое описание выглядит.

@tool(parse_docstring=True)
def get_equipment_status(equipment_id: str) -> dict:
    """Возвращает статус оборудования из реестра предприятия.

    Вызывай, когда нужно узнать текущее состояние конкретной единицы
    оборудования: статус (в работе / на ТО), расположение, дату последнего
    обслуживания. Для неизвестного кода вернёт поле error.

    Args:
        equipment_id: Код оборудования из реестра, например КМ-101 или НМ-205.
    """
    return equipment_status_impl(equipment_id)


@tool(parse_docstring=True)
def search_docs(question: str) -> dict:
    """Ищет ответ в базе документации: регламенты, паспорта, журнал инцидентов.

    Вызывай для вопросов о правилах, сроках, характеристиках и истории
    инцидентов. Возвращает ответ и список источников; если источники пусты —
    ответа в базе нет.

    Args:
        question: Вопрос обычным языком, например «Срок устранения критичных
            неисправностей насосов».
    """
    return rag_answer(question)



@tool(parse_docstring=True)
def create_ticket(equipment_id: str, problem: str, priority: str, reason: str) -> dict:
    """Создаёт заявку на обслуживание оборудования и сохраняет её в журнал заявок.

    Вызывай только после того, как проверил статус оборудования через
    get_equipment_status и убедился, что заявка действительно нужна.
    Создание заявки — write-действие: перед выполнением его подтверждает
    человек, поэтому не вызывай инструмент без явной просьбы оформить
    заявку. Не вызывай для справочных вопросов и для оборудования, которого
    нет в реестре, — сначала получи корректный код.

    Args:
        equipment_id: Код оборудования из реестра, например КМ-101 или НМ-205.
        problem: Краткое описание неисправности или проблемы обычным языком.
        priority: Приоритет заявки: low / medium / high / critical.
        reason: Основание для заявки — почему её нужно создать (симптом,
            результат проверки статуса, регламент).
    """
    count = 0
    if TICKETS_FILE.exists():
        with TICKETS_FILE.open("r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())

    ticket = {
        "ticket_id": f"TKT-2026-{count + 1:03d}",
        "equipment_id": equipment_id,
        "problem": problem,
        "priority": priority,
        "reason": reason,
        "status": "created",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TICKETS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, ensure_ascii=False) + "\n")

    return ticket


TOOLS = [get_equipment_status, search_docs, create_ticket]
TOOL_BY_NAME = {t.name: t for t in TOOLS}


# ====================================================================
#  ЦИКЛ АГЕНТА
# ====================================================================

def make_llm() -> GigaChat:
    """Модель агента. Параметры — как в Модуле 5."""
    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
        model="GigaChat-2",  # модель ВСЕГДА явно: алиасов «по умолчанию» на новом API нет
        temperature=0.0,     # решения агента должны быть повторяемыми
        max_tokens=500,
    )


def execute_tool(name: str, args: dict):
    """Выполняет инструмент. Ошибка инструмента — это наблюдение, а не крах."""
    if name not in TOOL_BY_NAME:
        return {"error": f"Неизвестный инструмент: {name}"}
    try:
        return TOOL_BY_NAME[name].invoke(args)
    except Exception as exc:
        return {"error": str(exc)}


def ask_user(name: str, args: dict) -> bool:
    """Подтверждение write-действия человеком (консоль)."""
    print(f"\n  Агент запрашивает действие {name}:")
    for key, value in args.items():
        print(f"      {key}: {value}")
    reply = input("  Подтвердить? [да/нет] ").strip().lower()
    return reply in ("да", "д", "y", "yes")


def run_agent(task: str, llm=None, confirm=ask_user, verbose=True):
    """Прогоняет одну задачу через цикл агента.

    Возвращает (final_answer, stats), где stats:
      steps         — сколько раз вызвана модель;
      tools         — имена вызванных инструментов по порядку;
      confirmations — сколько раз запрашивалось подтверждение человека;
      tokens        — суммарные токены всех шагов (prompt + completion);
      stop          — причина остановки: "ответ" или "лимит шагов".
    """
    llm = llm or make_llm()
    agent = llm.bind_tools(TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=task)]
    stats = {"steps": 0, "tools": [], "confirmations": 0, "tokens": 0, "stop": ""}

    def say(text):
        if verbose:
            print(text)

    say("=" * 70)
    say(f"Задача агенту: {task}")

    for step in range(1, MAX_STEPS + 1):
        msg = agent.invoke(messages)
        stats["steps"] = step
        usage = msg.response_metadata.get("token_usage")
        if usage:
            stats["tokens"] += usage.get("total_tokens", 0)

        say(f"\n--- Шаг {step} ---")
        if msg.content:
            say(f"  Мысль      : {msg.content}")

        # --------------------------------------------------------------
        # TODO 4. Условие остановки.
        # Модель не запросила ни одного инструмента — её текст и есть
        # финальный ответ. Выходим из цикла, не дожидаясь лимита шагов.
        # --------------------------------------------------------------
        if not msg.tool_calls:
            stats["stop"] = "ответ"
            say(f"\nФинальный ответ агента:\n{msg.content}")
            return msg.content, stats

        messages.append(msg)
        for call in msg.tool_calls:
            name, args = call["name"], call["args"]
            say(f"  Действие   : {name}({json.dumps(args, ensure_ascii=False)})")

            if name in WRITE_TOOLS:
                stats["confirmations"] += 1
                if confirm(name, args):
                    result = execute_tool(name, args)
                else:
                    result = {"cancelled": "Человек отклонил действие: заявка НЕ создана. "
                                           "Сообщи об этом честно и не повторяй попытку."}
            else:
                result = execute_tool(name, args)

            stats["tools"].append(name)
            say(f"  Наблюдение : {json.dumps(result, ensure_ascii=False)[:200]}")
            messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                        tool_call_id=call["id"]))

    stats["stop"] = "лимит шагов"
    answer = "Остановлено по лимиту шагов: задача не решена до конца."
    say(f"\n{answer}")
    return answer, stats


# ====================================================================
#  ДЕМО-ЗАДАЧИ ЗАНЯТИЯ 13
# ====================================================================

DEMO_TASKS = [
    # штатный сценарий: статус -> заявка (с подтверждением человека)
    "Конвейер КМ-101 сильно шумит и вибрирует. Проверь его статус и оформи "
    "заявку на обслуживание, если это необходимо.",
    # поиск информации: агент должен обойтись одним чтением документации
    "В какой срок по регламенту устраняются критичные неисправности насосов?",
]


def main():
    if "TODO" in (create_ticket.description or ""):
        print("Сначала опиши инструмент create_ticket (TODO 1): пока в его "
              "описании стоит TODO, агент не поймёт, когда его вызывать.")
        return
    if MAX_STEPS < 1:
        print("Поставь лимит шагов цикла (TODO 3): с MAX_STEPS = 0 агент "
              "не сделает ни одного шага.")
        return
    for task in DEMO_TASKS:
        _, stats = run_agent(task)
        print(f"\nИтог: шагов {stats['steps']}, инструментов {len(stats['tools'])} "
              f"{stats['tools']}, подтверждений {stats['confirmations']}, "
              f"токенов {stats['tokens']}, остановка: {stats['stop']}")
        print()


if __name__ == "__main__":
    main()
