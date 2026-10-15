"""
Модуль 2, практика 1 — промпт-дизайн для классификации обращений.

Задача: разложить обращения сотрудников ПромТеха по четырём категориям —
регламент / доступ / инцидент / документация. Это первый шаг маршрутизации:
от категории зависит, кому обращение уедет.

ТВОЯ РАБОТА — писать ТЕКСТ промптов (TODO 1-3). Код, который вызывает модель
и считает точность, уже готов.

Запуск (из папки module_02):
    python prompt_lab.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat

import token_tracker

load_dotenv(Path(__file__).parent.parent / ".env")

CATEGORIES = ["регламент", "доступ", "инцидент", "документация"]

DATA = json.loads((Path(__file__).parent / "data" / "tickets.json").read_text(encoding="utf-8"))
TICKETS = DATA["tickets"]


# ====================================================================
#  НАСТРОЙКИ СТУДЕНТА — пиши ТЕКСТ промптов
# ====================================================================

# TODO 1: zero-shot — инструкция без примеров.
# Базовый вариант дан. Попробуй его улучшить: объясни, чем категории отличаются
# друг от друга, и запрети модели отвечать чем-либо кроме одного слова.
ZERO_SHOT = (
    "Определи категорию обращения сотрудника промышленного предприятия.\n"
    "Ответь ОДНИМ словом из списка: регламент, доступ, инцидент, документация."
)

# TODO 2: few-shot — та же инструкция, но с примерами.
# Замени каждый ___ на своё обращение и его категорию. Нужно минимум 4 примера.
# Бери РАЗНЫЕ по смыслу обращения и обязательно возьми спорный случай: например,
# вопрос «в какой срок оформить карточку инцидента» — это регламент, а не
# инцидент, хотя слово «инцидент» в тексте есть.
FEW_SHOT = """Определи категорию обращения сотрудника.
Ответь ОДНИМ словом из списка: регламент, доступ, инцидент, документация.

Примеры:
Обращение: "В дашборде метрика конверсии воронки отображается некорректно" -> инцидент
Обращение: "Выдайте аналитику доступ к панели метрик продукта" -> доступ
Обращение: "В какой срок нужно оформить карточку инцидента после остановки?" -> регламент
Обращение: "Где найти гайдлайн по дизайн-системе и токенам?" -> документация
Обращение: "Не могу зайти в систему, учётка заблокирована" -> доступ
Обращение: "Где посмотреть актуальную версию инструкции?" -> документация
"""

# TODO 3 (бонус «+», по желанию): chain-of-thought.
# Попроси модель СНАЧАЛА одним предложением рассудить, чего хочет человек,
# и только потом вывести строку вида:  Категория: <одно слово>
# Если оставить пустым — стратегия просто не будет прогоняться.
COT = ""


# ====================================================================
#  КОД НИЖЕ УЖЕ РАБОТАЕТ — менять не нужно, но прочитай
# ====================================================================

def build_message(strategy_prompt: str, text: str) -> str:
    """Склеивает промпт-стратегию с конкретным обращением."""
    return f'{strategy_prompt}\n\nОбращение: "{text}"'


def pick_category(response: str) -> str:
    """Достаёт категорию из ответа модели.

    Модель почти никогда не отвечает ровно одним словом: добавляет пояснения,
    точки, кавычки. В CoT-режиме ответ вообще идёт после слова «Категория:».
    Поэтому ответ приходится разбирать, а не сравнивать напрямую.
    """
    text = response.lower()
    if "категория" in text:
        text = text.split("категория", 1)[1]

    found = [(text.find(c), c) for c in CATEGORIES if c in text]
    if not found:
        return "не определено"
    found.sort()
    return found[0][1]


def classify(strategy_prompt: str, text: str) -> str:
    """Спрашивает GigaChat и возвращает распознанную категорию."""
    with GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
    ) as client:
        response = client.chat({
            "messages": [{"role": "user", "content": build_message(strategy_prompt, text)}],
            "model": "GigaChat",
            "temperature": 0.0,   # классификация должна быть повторяемой
            "max_tokens": 200,
        })

    usage = response.usage
    token_tracker.record("GigaChat", usage.prompt_tokens, usage.completion_tokens, quiet=True)
    return pick_category(response.choices[0].message.content)


def evaluate(name: str, strategy_prompt: str) -> tuple:
    """Прогоняет стратегию по всем обращениям. Возвращает (верно, всего, ошибки)."""
    print(f"\n=== {name} ===")
    correct = 0
    errors = []

    for t in TICKETS:
        prediction = classify(strategy_prompt, t["text"])
        ok = prediction == t["category"]
        if ok:
            correct += 1
        else:
            errors.append((t["text"], t["category"], prediction))
        mark = "OK" if ok else " X"
        print(f"  {mark}  {t['text'][:46]:<46} -> {prediction:<13} (верно: {t['category']})")

    print(f"  ИТОГО {name}: {correct}/{len(TICKETS)}")
    return correct, len(TICKETS), errors


def main():
    if "___" in FEW_SHOT:
        print("В FEW_SHOT остались ___ — заполни примеры (TODO 2) и сохрани файл.")
        return

    results = {"zero-shot": evaluate("zero-shot", ZERO_SHOT),
               "few-shot": evaluate("few-shot", FEW_SHOT)}
    if COT.strip():
        results["CoT"] = evaluate("CoT", COT)

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ СТРАТЕГИЙ")
    print("=" * 60)
    for name, (correct, total, _) in results.items():
        print(f"  {name:<12} {correct}/{total}   ({correct / total * 100:.0f}%)")

    print("\nНа чём ошибались:")
    any_error = False
    for name, (_, _, errors) in results.items():
        for text, truth, pred in errors:
            any_error = True
            print(f"  [{name}] «{text[:42]}...»")
            print(f"           ждали: {truth}, получили: {pred}")
    if not any_error:
        print("  ошибок нет")

    token_tracker.print_report()


if __name__ == "__main__":
    main()
