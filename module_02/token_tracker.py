"""
Модуль 1 — учёт токенов.

Зачем: у GigaChat есть бесплатный лимит (900 000 токенов). Каждый вызов модели
его расходует. В проде расход токенов — это деньги, поэтому его считают всегда.

Этот файл трогать не нужно. llm_client.py вызывает record(...) после каждого
запроса, а статистика копится в файле .token_usage.json.

Посмотреть отчёт в любой момент:
    python token_tracker.py
"""

import json
from pathlib import Path

# Бесплатный лимит GigaChat Lite (вход + выход)
FREE_LIMIT_TOKENS = 900_000

WARN_AT = 0.75   # 75% лимита — предупреждение
ALERT_AT = 0.90  # 90% лимита — алерт

# Статистика лежит в КОРНЕ проекта, а не внутри модуля: расход копится сквозь
# все модули курса. Если положить файл в папку модуля, счётчик обнулялся бы
# на каждом новом модуле, и до лимита в 900 000 токенов можно было бы доехать
# незаметно.
STATS_FILE = Path(__file__).parent.parent / ".token_usage.json"


def _fmt(n: int) -> str:
    """1842 -> '1 842' (разряды разделяем пробелом, как принято в русском тексте)."""
    return f"{n:,}".replace(",", " ")


def _calls(n: int) -> str:
    """1 вызов / 2 вызова / 5 вызовов."""
    last, last_two = n % 10, n % 100
    if last == 1 and last_two != 11:
        word = "вызов"
    elif last in (2, 3, 4) and last_two not in (12, 13, 14):
        word = "вызова"
    else:
        word = "вызовов"
    return f"{n} {word}"


def _load() -> dict:
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"total_tokens": 0, "calls": 0}


_stats = _load()
_session_tokens = 0
_session_calls = 0


def record(provider: str, prompt_tokens: int, completion_tokens: int, quiet: bool = False) -> None:
    """Записать расход токенов одного вызова модели.

    quiet=True — не печатать строку по каждому вызову. Нужно там, где вызовов
    десятки (например, прогон классификатора в модуле 2): иначе лог не читается.
    Итог всё равно попадёт в print_report().
    """
    global _session_tokens, _session_calls

    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    total = prompt_tokens + completion_tokens

    _stats["total_tokens"] += total
    _stats["calls"] += 1
    _session_tokens += total
    _session_calls += 1

    STATS_FILE.write_text(
        json.dumps(_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not quiet:
        print(
            f"   [токены] {provider}: +{total} "
            f"(вход: {prompt_tokens}, выход: {completion_tokens})"
        )

    ratio = _stats["total_tokens"] / FREE_LIMIT_TOKENS
    remaining = FREE_LIMIT_TOKENS - _stats["total_tokens"]
    if ratio >= ALERT_AT:
        print(f"   [!] Осталось {_fmt(remaining)} токенов из {_fmt(FREE_LIMIT_TOKENS)}. Экономь запросы.")
    elif ratio >= WARN_AT:
        print(f"   [!] Использовано {ratio * 100:.0f}% бесплатного лимита.")


def print_report() -> None:
    """Итоговый отчёт по расходу токенов."""
    used = _stats["total_tokens"]
    remaining = max(0, FREE_LIMIT_TOKENS - used)
    ratio = used / FREE_LIMIT_TOKENS * 100

    print()
    print("=" * 60)
    print("РАСХОД ТОКЕНОВ")
    print("=" * 60)
    print(f"  Использовано : {_fmt(used)} токенов ({ratio:.1f}% лимита)")
    print(f"  Осталось     : {_fmt(remaining)} токенов")
    print(f"  Вызовов всего: {_stats['calls']}")
    if _session_calls:
        print(f"  В этом запуске: {_fmt(_session_tokens)} токенов ({_calls(_session_calls)})")
    print()
    print("  Учёт ведётся с Модуля 1. Вызовы из Модуля 0 в него не попали.")
    print("=" * 60)


if __name__ == "__main__":
    print_report()
