"""
Модуль 5, Занятие 10, практика 1 — диалог с памятью поверх RAG.

Цепочка Занятия 9 отвечает на один вопрос и всё забывает. Инженер же
спрашивает диалогом: «Что случилось с насосом НМ-205?» — и следом
«А какая у него производительность?». Второй вопрос без первого не имеет
смысла: «у него» — это у кого?

Схема диалогового RAG:

    вопрос + история -> [переформулировка] -> самостоятельный вопрос
                     -> [поиск по базе]    -> ответ + источники
    история пополняется парой (вопрос, ответ) и обрезается скользящим окном.

Память здесь — просто список сообщений, который мы передаём сами:
у LLM API нет никакой «памяти между вызовами», это иллюзия, которую
собирает клиентский код. Окно ограничивает список последними сообщениями,
иначе история растёт с каждой репликой — вместе с токенами каждого запроса
и с риском утащить в промпт лишние данные.

ТВОЯ РАБОТА — TODO 1-4: скользящее окно, промпт переформулировки, сборка
condense-цепочки и шаг диалога. Цикл диалога и замер истории готовы.

Запуск (из папки module_05):
    python rag_lcel.py            # диалог с живой моделью
    python rag_lcel.py --stats    # локальный замер роста истории, без ключей
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_gigachat.chat_models import GigaChat

from rag_stub import answer as rag_answer

load_dotenv(Path(__file__).parent.parent / ".env")

# сколько последних сообщений истории попадает в запрос (3 пары вопрос-ответ)
WINDOW = 6

DIALOG = [
    "Что случилось с насосом НМ-205?",
    "А какая у него производительность?",
    "В какой срок должны устраняться критичные неисправности таких насосов?",
]


def make_llm() -> GigaChat:
    return GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
        model="GigaChat-2",   # модель ВСЕГДА явно: алиасов «по умолчанию» на новом API нет
        temperature=0.0,
        max_tokens=200,
    )


# ====================================================================
#  ПАМЯТЬ: скользящее окно
# ====================================================================

def trim_history(history: list, max_messages: int = WINDOW) -> list:
    """Скользящее окно: оставляет последние max_messages сообщений.

    TODO 1. Требования:
      * вернуть НОВЫЙ список из последних max_messages сообщений
        (исходный список не менять);
      * диалог после обрезки должен начинаться с вопроса человека: пока
        первым элементом стоит AIMessage (ответ без своего вопроса) —
        отбрасывай его;
      * короткая история (не длиннее окна) возвращается целиком.
    """
    raise NotImplementedError("TODO 1: скользящее окно истории")


# ====================================================================
#  ПЕРЕФОРМУЛИРОВКА: уточняющий вопрос -> самостоятельный
# ====================================================================

# ------------------------------------------------------------------
# TODO 2. Напиши системный промпт переформулировки.
#
# Модель получит историю диалога и последний вопрос инженера. Она должна
# вернуть ТОЛЬКО переформулированный вопрос — понятный без диалога:
# вместо местоимений и отсылок («у него», «а для этого же оборудования»)
# должны стоять конкретные названия и коды из истории.
# ------------------------------------------------------------------
CONDENSE_SYSTEM = ""

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CONDENSE_SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def build_condense_chain(model):
    """Цепочка переформулировки: промпт с историей -> модель -> строка.

    TODO 3. Собери цепочку из CONDENSE_PROMPT, модели и StrOutputParser().
    """
    raise NotImplementedError("TODO 3: цепочка переформулировки")


# ====================================================================
#  ДИАЛОГОВЫЙ ШАГ: переформулировка (если есть история) + поиск
# ====================================================================

def dialog_step(question: str, history: list, model) -> dict:
    """Один шаг диалога. Возвращает {"answer", "sources", "standalone"}.

    TODO 4. Правила:
      * истории нет -> вопрос уже самостоятельный, модель НЕ вызывается;
      * история есть -> вызови condense-цепочку с окном истории:
        {"history": trim_history(history), "question": question},
        результату сделай .strip() — это самостоятельный вопрос;
      * отправь самостоятельный вопрос в rag_answer(...) и верни
        {"answer": ..., "sources": ..., "standalone": самостоятельный вопрос}.
    """
    raise NotImplementedError("TODO 4: шаг диалога")


def run_dialog():
    llm = make_llm()
    history = []

    for question in DIALOG:
        print("=" * 70)
        print(f"Инженер : {question}")
        step = dialog_step(question, history, llm)
        if step["standalone"] != question:
            print(f"          (переформулировано: «{step['standalone']}»)")
        print(f"Ассистент: {step['answer']}")
        print(f"Источники: {', '.join(step['sources']) or '— нет'}")

        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=step["answer"]))
        history = trim_history(history, max_messages=WINDOW)


# ====================================================================
#  ЗАМЕР: как растёт история без окна и с окном (локально, без модели)
# ====================================================================

def run_stats():
    """Считает объём истории в символах на длинном диалоге.

    Ответы берутся из rag_stub, модель не вызывается — замер можно
    запускать без ключей и сети (нужен выполненный TODO 1). Токенов
    у GigaChat примерно вдвое-втрое меньше, чем символов русского
    текста, порядок роста тот же.
    """
    questions = [
        "Что случилось с насосом НМ-205?",
        "Какая производительность у насоса НМ-205?",
        "В какой срок устраняются критичные неисправности насосов?",
        "В каком журнале ведётся учёт текущего контроля насосов?",
        "Какая производительность у конвейера КМ-101?",
        "Какие типовые неисправности у пресса П-7?",
        "Как часто проводится проверка знаний у персонала?",
        "В какой срок нужно сообщить об инциденте?",
        "За какое время комиссия оценивает степень опасности инцидента?",
        "Что случилось с реле давления насоса НМ-205?",
    ]

    def size(messages: list) -> int:
        return sum(len(m.content) for m in messages)

    full, windowed = [], []
    print(f"{'шаг':>3} | {'история целиком':>16} | {'окно ' + str(WINDOW):>10}")
    print("-" * 40)
    for i, q in enumerate(questions, 1):
        result = rag_answer(q)
        for h in (full, windowed):
            h.append(HumanMessage(content=q))
            h.append(AIMessage(content=result["answer"]))
        windowed[:] = trim_history(windowed, max_messages=WINDOW)
        print(f"{i:>3} | {size(full):>13} с. | {size(windowed):>7} с.")

    print("-" * 40)
    print(f"История целиком копится бесконечно, окно {WINDOW} выходит на полку:")
    print("в каждый запрос уезжает не больше трёх последних пар реплик.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Диалог с памятью поверх RAG")
    parser.add_argument("--stats", action="store_true",
                        help="локальный замер роста истории (без ключей и сети)")
    args = parser.parse_args()
    run_stats() if args.stats else run_dialog()
