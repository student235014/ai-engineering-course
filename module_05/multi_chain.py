"""
Модуль 5, Занятие 10, практика 2 — маршрутизация обращений.

Не каждый вопрос стоит отдавать модели. Маршрутизатор решает ДО вызова LLM,
куда пойдёт обращение:

    safety    -> немедленно человеку: диспетчер и ответственный за ОТ.
                 Модель не вызывается вообще — цена ошибки слишком высока.
    technical -> «инженерная» ветка: технический промпт.
    general   -> «общая» ветка: дружелюбный помощник.
    пустой или односложный вопрос -> просьба переформулировать, без вызова.

Классификатор здесь — ПРАВИЛА, а не модель: списки основ слов и реестр
оборудования из Модуля 2. Правила дешёвые (ноль токенов), мгновенные
и предсказуемые: одно и то же обращение всегда уходит одним маршрутом,
и контрольный прогон --dry показывает это без сети и без токенов.

Слова ищутся по границе слова (\\bоснова...), а не подстрокой: наивное
`"стано" in text` считает техническим вопрос про «переустановку приложения».

ТВОЯ РАБОТА — TODO 1-4: дополнить TECH_WORDS, найти оборудование по реестру,
собрать классификатор и вилку RunnableBranch. Ветки, промпты и прогоны готовы.

Запуск (из папки module_05):
    python multi_chain.py --dry   # только маршруты, без модели и ключей
    python multi_chain.py         # маршруты + ответы живой модели
"""

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_gigachat.chat_models import GigaChat

load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path(__file__).parent / "data"

REGISTRY = json.loads(
    (DATA_DIR / "equipment_registry.json").read_text(encoding="utf-8")
)["equipment"]

CASES = json.loads(
    (DATA_DIR / "routing_cases.json").read_text(encoding="utf-8")
)["cases"]


# ====================================================================
#  СЛОВА-ПРИЗНАКИ
# ====================================================================

# ------------------------------------------------------------------
# TODO 1. Дополни список основ слов технического вопроса минимум до 12.
#
# Именно ОСНОВЫ: «ошибк» покрывает «ошибка / ошибку / ошибки». Подбирай
# по контрольным случаям: python multi_chain.py --dry покажет, какие
# технические вопросы ещё уходят не туда.
# ------------------------------------------------------------------
TECH_WORDS = [
    "ошибк", "оборудован", "стано", "датчик", "mes",
]

# Основы слов, при которых обращение НЕМЕДЛЕННО уходит человеку.
# Здесь ложное срабатывание дешевле пропуска: пусть лучше диспетчер
# прочитает лишнее сообщение, чем модель «поможет» при пожаре.
SAFETY_WORDS = [
    "пожар", "дым", "задымлен", "возгоран", "газ",
    "травм", "пострадав", "искрит", "горит",
]


# ====================================================================
#  РЕЕСТР ОБОРУДОВАНИЯ (нормализация — из Модуля 2)
# ====================================================================

# Латинские двойники кириллицы: «KM-101» латиницей на глаз не отличить.
_LAT_TO_CYR = str.maketrans("ABCEHKMOPTXYabcehkmoptxy", "АВСЕНКМОРТХУавсенкмортху")


def _norm(value: str) -> str:
    """«км 101», «KM-101» (латиница) и «КМ-101» — один ключ."""
    return value.strip().replace(" ", "-").translate(_LAT_TO_CYR).casefold()


REGISTRY_LOOKUP = {_norm(item["id"]): item["id"] for item in REGISTRY}


def _tokens(text: str) -> list:
    """Слова текста без обрамляющей пунктуации («КТ-500.2» — одно слово)."""
    return [w.strip(".,;:!?()«»\"'") for w in re.findall(r"[\w.\-]+", text)]


def mentions_equipment(question: str) -> bool:
    """Есть ли в вопросе код оборудования из реестра.

    TODO 2. Код — самый надёжный технический признак: «KM-101 странно шумит»
    не содержит ни одного слова из TECH_WORDS, но код выдаёт маршрут.
    Пройди по _tokens(question), нормализуй каждое слово через _norm(...)
    и проверь, есть ли оно в REGISTRY_LOOKUP.
    """
    raise NotImplementedError("TODO 2: поиск кода оборудования в вопросе")


# ====================================================================
#  КЛАССИФИКАТОР ПРАВИЛ
# ====================================================================

def _has_word(text: str, stems: list) -> bool:
    """Ищет основы слов по границе слова: «газом» — да, «магазин» — нет."""
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(stem)}\w*", lowered) for stem in stems)


def classify(question: str) -> str:
    """Маршрут вопроса: safety / technical / general.

    TODO 3. Порядок проверок — это приоритет: сообщение «искрит датчик
    у насоса» содержит и safety-, и технические признаки, но люди важнее
    железа. Правила:
      * нашлась основа из SAFETY_WORDS               -> "safety";
      * основа из TECH_WORDS ИЛИ код из реестра      -> "technical";
      * иначе                                        -> "general".
    """
    raise NotImplementedError("TODO 3: классификатор правил")


# ====================================================================
#  ВЕТКИ И ВИЛКА
# ====================================================================

TECH_PROMPT = ChatPromptTemplate.from_template(
    "Ты — технический специалист промышленного предприятия. Отвечай кратко, "
    "по пунктам, без выдуманных фактов. Если данных мало — скажи, каких "
    "не хватает.\nВопрос: {question}"
)

GENERAL_PROMPT = ChatPromptTemplate.from_template(
    "Ты — дружелюбный помощник сотрудников предприятия. Отвечай просто "
    "и коротко.\nВопрос: {question}"
)

SAFETY_ANSWER = (
    "Похоже на угрозу безопасности. Сообщение передано диспетчеру и "
    "ответственному за охрану труда, с вами свяжутся немедленно. "
    "Если есть угроза людям — действуйте по инструкции эвакуации."
)

CLARIFY_ANSWER = ("Сформулируйте вопрос подробнее: что за оборудование "
                  "или система, что именно происходит.")

# короче этого (в символах) вопрос не маршрутизируем, а просим уточнить
MIN_QUESTION_LEN = 4


def build_router_chain(model):
    """Вилка маршрутов. Вход {"question": ...}, выход {"route", "answer"}.

    TODO 4. RunnableBranch перебирает пары (условие, ветка) и запускает
    первую ветку, чьё условие сработало; последний аргумент — ветка
    по умолчанию. Собери вилку из четырёх маршрутов:

      1) len(x["question"].strip()) < MIN_QUESTION_LEN
             -> wrap("уточнить", CLARIFY_ANSWER)
      2) classify(...) == "safety"
             -> wrap("safety -> человеку", SAFETY_ANSWER)   # БЕЗ модели!
      3) classify(...) == "technical"
             -> wrap("technical", tech_chain.invoke(x))
      4) по умолчанию
             -> wrap("general", general_chain.invoke(x))

    Каждая ветка — RunnableLambda(lambda x: ...). Условие — обычная
    lambda x: ... (первым элементом пары).
    """
    tech_chain = TECH_PROMPT | model | StrOutputParser()
    general_chain = GENERAL_PROMPT | model | StrOutputParser()

    def wrap(route, answer):
        return {"route": route, "answer": answer}

    raise NotImplementedError("TODO 4: собери RunnableBranch из четырёх маршрутов")


# ====================================================================

def run_dry():
    """Прогон маршрутизатора по контрольным случаям — без модели и сети."""
    correct = 0
    print(f"{'ожидали':<12} {'получили':<12} вопрос")
    print("-" * 74)
    for case in CASES:
        got = ("уточнить" if len(case["question"].strip()) < MIN_QUESTION_LEN
               else classify(case["question"]))
        ok = got == case["route"]
        correct += ok
        mark = "  " if ok else "X "
        print(f"{mark}{case['route']:<12} {got:<12} {case['question'][:99]}")
    print("-" * 74)
    print(f"ИТОГО: {correct}/{len(CASES)}")


def run_live():
    llm = GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
        model="GigaChat-2",   # модель ВСЕГДА явно: алиасов «по умолчанию» на новом API нет
        temperature=0.0,
        max_tokens=300,
    )
    chain = build_router_chain(llm)

    demo = [
        "Станок ЧПУ-12 выдаёт ошибку шпинделя, что делать?",
        "KM-101 странно шумит после запуска.",
        "Пахнет газом на участке водоподготовки!",
        "Как настроить рабочую почту на телефоне?",
        "?",
    ]
    for question in demo:
        result = chain.invoke({"question": question})
        print("=" * 70)
        print(f"Вопрос : {question}")
        print(f"Маршрут: {result['route']}")
        print(f"Ответ  : {result['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Маршрутизация обращений")
    parser.add_argument("--dry", action="store_true",
                        help="проверить только маршруты, без вызова модели")
    args = parser.parse_args()
    run_dry() if args.dry else run_live()
