"""
Модуль 4, Занятие 8 — оценка качества и защита RAG-сценария.

ТВОЯ РАБОТА: TODO 1 (classify — оценить ответ по разметке контрольного
набора: отличить ok / retrieval_error / generation_error / hallucination /
over_refusal), TODO 2 (llm_judge — попросить GigaChat оценить ответ
ассистента по трём критериям: релевантность ответа, релевантность контекста
и достоверность ответа), TODO 3 (sweep_chunking — прогнать несколько
CHUNK_SIZE/OVERLAP и посмотреть, как меняется качество по обоим способам
оценки), TODO 4 (resolve_conflicts — при нескольких версиях одного
регламента в выдаче использовать актуальную, а не первую по score).

Занятие 7 научило rag_assistant.py отвечать с источником и отказываться
при низкой уверенности. Здесь мы сравниваем два способа проверки:

  1. По разметке eval-набора — быстро, дёшево, воспроизводимо.
  2. Через LLM-as-Judge — гибко и наглядно: GigaChat видит вопрос,
     контекст поиска и ответ ассистента и объясняет, что именно не так.

Запуск (из папки module_04, индекс должен быть собран):
    python evaluate_rag.py                  # сравнение двух способов оценки
    python evaluate_rag.py --sweep          # перебор CHUNK_SIZE/OVERLAP
    python evaluate_rag.py --conflict       # демо конфликта версий регламента
    python evaluate_rag.py --embedded       # без Docker
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat
from sentence_transformers import SentenceTransformer

import ingestion
from qdrant_connect import make_client
from vector_store import MODEL_NAME, load_manifest, search
from rag_assistant import answer, format_context, REFUSAL_TEXT

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "qdrant_chunks"
CORPUS_DIR = BASE_DIR / "data" / "corpus"
CONFLICT_DIR = BASE_DIR / "data" / "corpus_conflict"

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

QUESTIONS = json.loads(
    (BASE_DIR / "data" / "questions.json").read_text(encoding="utf-8")
)["questions"]

# Комбинации для перебора — те же, что в Занятии 6, чтобы результаты
# было с чем сравнивать. Пересборка индекса на каждую комбинацию занимает
# несколько минут: --sweep запускай отдельно, не в общем прогоне.
CHUNK_COMBOS = [(1000, 200), (600, 0), (600, 150), (400, 0), (400, 100), (300, 150)]

_SEP  = "─" * 88
_SEP2 = "═" * 88

# Системный промпт судьи. Менять не нужно — он уже настроен на корпус курса.
JUDGE_SYSTEM = (
    "Ты — независимый оценщик качества RAG-ассистента. "
    "Ассистент отвечает на вопросы по корпусу производственных документов "
    "промышленного предприятия.\n\n"
    "Тебе даются три части:\n"
    "  ВОПРОС — то, что спросил пользователь.\n"
    "  КОНТЕКСТ — фрагменты документов, которые нашёл поиск и передал ассистенту.\n"
    "  ОТВЕТ — что ответил ассистент.\n\n"
    "Оцени ответ по трём критериям:\n"
    "  1. answer_relevance — отвечает ли ответ именно на вопрос пользователя.\n"
    "  2. context_relevance — действительно ли найденный контекст подходит для этого вопроса.\n"
    "  3. answer_faithfulness — не выдумывает ли ответ факты сверх переданного контекста.\n\n"
    "Для каждого критерия используй одно из значений: high, medium, low.\n\n"
    "Верни ТОЛЬКО JSON-объект без какого-либо текста до или после него:\n"
    '{"answer_relevance": "<high|medium|low>", '
    '"context_relevance": "<high|medium|low>", '
    '"answer_faithfulness": "<high|medium|low>", '
    '"overall_verdict": "<ok|retrieval_problem|generation_problem|hallucination|refusal_ok|mixed>", '
    '"reason": "<обоснование на русском, 1–2 предложения>"}\n\n'
    "Смысл overall_verdict:\n"
    '  "ok"                — ответ релевантен, контекст релевантен, факты не выдуманы.\n'
    '  "retrieval_problem" — найденный контекст плохо подходит к вопросу.\n'
    '  "generation_problem" — контекст в целом подходит, но ответ слабый, неполный или не по существу.\n'
    '  "hallucination"     — ответ содержит факты, которых нет в контексте.\n'
    '  "refusal_ok"        — отказ корректен: контекст не даёт ответа.\n'
    '  "mixed"             — смешанный случай, который не сводится к одной простой категории.'
)


# ====================================================================
#  ОЦЕНКА ПО РАЗМЕТКЕ
# ====================================================================

def classify(question: dict, hits, result: dict) -> str:
    """TODO 1: оценка ответа по разметке eval-набора.

    Верни одну из меток:
      - ok
      - hallucination
      - over_refusal
      - retrieval_error
      - generation_error

    Логика:
      1) Если question["in_corpus"] == False:
         - ответ == REFUSAL_TEXT -> ok
         - иначе -> hallucination
      2) Если вопрос из корпуса, но ассистент отказал (REFUSAL_TEXT) -> over_refusal
      3) Проверить источники: хотя бы один doc_id из question["expected"]
         должен быть в result["sources"], иначе -> retrieval_error
      4) Проверить фактоид: question["must_contain"] должен встречаться в answer
         (без учёта регистра), иначе -> generation_error
      5) Иначе -> ok
    """
    # TODO 1: замени заглушку на реализацию по шагам выше
    return "ok"


# ====================================================================
#  LLM-КАК-СУДЬЯ
# ====================================================================

def llm_judge(question: str, context: str, answer_text: str) -> dict:
    """TODO 2: попросить GigaChat оценить ответ ассистента как судья.

    Что сделать:
      1) Собрать user_msg из question/context/answer_text.
      2) Вызвать GigaChat с system=JUDGE_SYSTEM и user=user_msg.
      3) Распарсить JSON из ответа модели и вернуть dict.
      4) В except вернуть fallback:
         {"overall_verdict": "error", "reason": str(e)}

    Ожидаемые поля в словаре судьи:
      answer_relevance, context_relevance, answer_faithfulness,
      overall_verdict, reason
    """
    # TODO 2: замени заглушку на реальный вызов судьи
    return {
        "answer_relevance": "low",
        "context_relevance": "low",
        "answer_faithfulness": "low",
        "overall_verdict": "error",
        "reason": "TODO 2: llm_judge не реализован",
    }


# ====================================================================
#  КОНТРОЛЬНЫЙ ПРОГОН
# ====================================================================

def run_eval(client, model, alias: str, verbose=True) -> dict:
    """Прогоняет контрольный набор и выводит два отдельных отчёта."""
    rows = []

    for q in QUESTIONS:
        hits = search(client, model, q["question"], alias)
        result = answer(client, model, q["question"], alias)
        ctx = format_context(hits) if hits else "(ничего не найдено)"

        rows.append({
            "id":        q.get("id", ""),
            "question":  q["question"],
            "in_corpus": q["in_corpus"],
            "markup":    classify(q, hits, result),
            "judgment":  llm_judge(q["question"], ctx, result["answer"]),
        })

    if verbose:
        _print_markup_report(rows)
        _print_judge_report(rows)

    # агрегаты для sweep_chunking
    counts = {"markup": {}, "judge": {}}
    for r in rows:
        mk = r["markup"]
        jv = r["judgment"].get("overall_verdict", "error")
        counts["markup"][mk] = counts["markup"].get(mk, 0) + 1
        counts["judge"][jv]  = counts["judge"].get(jv, 0) + 1
    counts["_rows"] = rows
    return counts


# ---------------------------------------------------------------------------
#  Метки для человекочитаемого вывода
# ---------------------------------------------------------------------------

_MARKUP_LABEL = {
    "ok":               "✓ верно",
    "hallucination":    "✗ галлюцинация",
    "retrieval_error":  "✗ документ не найден",
    "generation_error": "⚠ ответ неполный",
    "over_refusal":     "⚠ отказал зря",
}

_JUDGE_VERDICT_LABEL = {
    "ok":                   "всё хорошо",
    "retrieval_problem":    "контекст не подходит",
    "generation_problem":   "ответ слабый",
    "hallucination":        "галлюцинация",
    "refusal_ok":           "отказ обоснован",
    "mixed":                "смешанный случай",
    "error":                "ошибка вызова судьи",
}

_LEVEL_LABEL = {
    "high":   "✓ высокая",
    "medium": "⚠ средняя",
    "low":    "✗ низкая",
}


def _short(text: str, width: int) -> str:
    return text[:width - 1] + "…" if len(text) > width else text


def _print_markup_report(rows: list) -> None:
    print(_SEP2)
    print("ОТЧЁТ 1 — ОЦЕНКА ПО РАЗМЕТКЕ EVAL-НАБОРА")
    print(_SEP2)
    print()

    counts = {}
    for r in rows:
        mk = r["markup"]
        label = _MARKUP_LABEL.get(mk, mk)
        tag = "(вне базы)" if not r["in_corpus"] else ""
        qid = r["id"] or "—"
        print(f"  {qid:<4}  {label:<25}  {_short(r['question'], 55)}  {tag}")
        counts[mk] = counts.get(mk, 0) + 1

    print()
    print("  Итого:")
    order = ["ok", "retrieval_error", "generation_error", "hallucination", "over_refusal"]
    for mk in order:
        n = counts.get(mk, 0)
        if n or mk == "ok":
            print(f"    {_MARKUP_LABEL.get(mk, mk):<25}  {n}")
    print()


def _print_judge_report(rows: list) -> None:
    print(_SEP2)
    print("ОТЧЁТ 2 — ОЦЕНКА GigaChat-СУДЬЁЙ (LLM-as-Judge)")
    print(_SEP2)

    counts_verdict = {}
    for r in rows:
        j = r["judgment"]
        verdict = j.get("overall_verdict", "error")
        counts_verdict[verdict] = counts_verdict.get(verdict, 0) + 1

        def _level(val: str) -> str:
            # модель иногда возвращает нестандартное значение (напр. "refusal_ok")
            # нормализуем: берём только high/medium/low, остальное → "—"
            v = (val or "").lower().strip()
            if v in _LEVEL_LABEL:
                return _LEVEL_LABEL[v]
            for key in _LEVEL_LABEL:
                if v.startswith(key):
                    return _LEVEL_LABEL[key]
            return f"({val})"

        ar = _level(j.get("answer_relevance",    ""))
        cr = _level(j.get("context_relevance",   ""))
        af = _level(j.get("answer_faithfulness", ""))
        vl = _JUDGE_VERDICT_LABEL.get(verdict, verdict)
        reason = j.get("reason", "")
        tag = "(вне базы)" if not r["in_corpus"] else ""
        qid = r["id"] or "—"

        print()
        print(f"  {qid}  {_short(r['question'], 65)}  {tag}")
        print(f"      Ответ отвечает на вопрос:    {ar}")
        print(f"      Контекст подходит к вопросу: {cr}")
        print(f"      Ответ не выдумывает факты:   {af}")
        print(f"      Итог судьи: {vl}")
        print(f"      Пояснение: «{reason}»")

    print()
    print(_SEP)
    print("  Итого по судье:")
    order = ["ok", "refusal_ok", "generation_problem", "retrieval_problem",
             "hallucination", "mixed", "error"]
    for v in order:
        n = counts_verdict.get(v, 0)
        if n:
            print(f"    {_JUDGE_VERDICT_LABEL.get(v, v):<25}  {n}")
    print()


# ====================================================================
#  ПЕРЕБОР CHUNK_SIZE / OVERLAP
# ====================================================================

def sweep_chunking(client, model, alias: str):
    """Автоматически перебирает параметры нарезки и сравнивает качество двумя способами.

    Эту функцию запускают командой:  python evaluate_rag.py --sweep
    Вручную запускать ingestion.py несколько раз не нужно — функция сама
    пересобирает индекс Qdrant на каждой комбинации параметров.

    TODO 3: сейчас функция только печатает шапку таблицы — заглушка.
    Напиши цикл по CHUNK_COMBOS. На каждой итерации:
      1. ingestion.CHUNK_SIZE = size; ingestion.OVERLAP = overlap
      2. ingestion.build_index(client, model)  — пересобирает индекс, возвращает manifest
      3. counts = run_eval(client, model, alias, verbose=False)
         Посчитай для вопросов из базы (in_corpus=True):
           - верных по разметке:  counts["markup"].get("ok", 0)
           - верных по судье:     counts["judge"].get("ok", 0)
      4. Выведи строку таблицы (см. формат ниже).

    После цикла верни параметры к 600/150 и пересобери индекс ещё раз —
    иначе следующий обычный запуск будет работать на последних параметрах.

    Ожидаемый вид таблицы (8 вопросов из базы из 11 всего):

        Размер чанка / перекрытие  | чанков |  ✓ по разметке | ✓ по судье
        ──────────────────────────────────────────────────────────────────────
        1000 / 200                 |    563 |          6 / 8 |      5 / 8
         600 / 0                   |    715 |          7 / 8 |      7 / 8
         600 / 150                 |    921 |          7 / 8 |      7 / 8
         400 / 0                   |   1028 |          5 / 8 |      5 / 8
         400 / 100                 |   1337 |          7 / 8 |      6 / 8
         300 / 150                 |   2600 |          4 / 8 |      3 / 8

    Разметка и судья могут расходиться: разметка проверяет, есть ли нужная
    строка в ответе; судья оценивает, насколько ответ полезен и честен.
    """
    in_corpus_total = sum(1 for q in QUESTIONS if q["in_corpus"])
    col = f"{'Размер чанка / перекрытие':<26}"
    print(f"\n  {col} | {'чанков':>7} | {'✓ по разметке':>14} | {'✓ по судье':>10}")
    print("  " + _SEP[:72])
    original = (ingestion.CHUNK_SIZE, ingestion.OVERLAP)
    # TODO 3: перебор здесь
    ingestion.CHUNK_SIZE, ingestion.OVERLAP = original


# ====================================================================
#  КОНФЛИКТ ВЕРСИЙ ОДНОГО РЕГЛАМЕНТА
# ====================================================================

def resolve_conflicts(hits):
    """Если в выдаче несколько версий одного регламента — оставляет свежую.

    TODO 4: сейчас возвращает hits без изменений — заглушка. С ней
    demo_conflict() покажет саму проблему, но не решение.

    В data/corpus_conflict/REGL-GEN-01-OLD.json лежит устаревшая (version
    2.0, дата 2022-05-14) редакция регламента, который в основном корпусе
    существует как REGL-GEN-01 (version 3.1, дата 2024-01-10) — тот же
    регламент, тот же номер РГЛ-НО-014, другие сроки. Оба документа
    отвечают на один вопрос по смыслу — Qdrant не знает, что один из них
    отменён, и вернёт оба, если оба лежат в индексе.

    Что сделать: сгруппировать hits по базовому doc_id (REGL-GEN-01-OLD
    и REGL-GEN-01 — одна группа: суффикс -OLD не признак другой группы);
    в каждой группе оставить запись с максимальным payload["date"]
    (строки ГГГГ-ММ-ДД сравниваются как даты), остальные выбросить.
    Порядок и score оставшихся точек не менять.
    """
    return hits


def demo_conflict(client, model, alias: str):
    """Подмешивает устаревшую редакцию регламента и показывает конфликт."""
    print(_SEP2)
    print("КОНФЛИКТ ВЕРСИЙ: две редакции одного регламента в индексе")
    print(_SEP2)

    fixture = CONFLICT_DIR / "REGL-GEN-01-OLD.json"
    target  = CORPUS_DIR / fixture.name
    shutil.copy(fixture, target)
    print(f"Добавлен в корпус: {fixture.name} (version 2.0, 2022-05-14) — "
          f"рядом с действующим REGL-GEN-01 (version 3.1, 2024-01-10).")
    print("Пересобираю индекс...")
    ingestion.build_index(client, model)

    question = "В какой срок должны устраняться критичные неисправности насосного оборудования?"
    hits = search(client, model, question, alias, doc_type="регламент", limit=5)

    print(f"\nВопрос: {question}")
    print("\nБез resolve_conflicts — что реально лежит в топ-5:")
    for h in hits:
        p = h.payload
        print(f"  {h.score:.3f}  {p['doc_id']:<18} версия {p.get('version') or '—':<5} "
              f"дата {p.get('date') or '—':<11} {p.get('section') or ''}")

    filtered = resolve_conflicts(hits)
    print("\nПосле resolve_conflicts:")
    for h in filtered:
        p = h.payload
        print(f"  {h.score:.3f}  {p['doc_id']:<18} версия {p.get('version') or '—':<5} "
              f"дата {p.get('date') or '—':<11} {p.get('section') or ''}")

    print(
        "\nЕсли в выдаче до сих пор обе редакции — TODO 4 ещё не сделан, и "
        "ассистент рискует процитировать «до 3 рабочих дней» из отменённого "
        "документа вместо действующих «до 1 рабочего дня». Разница на "
        "бумаге небольшая, а на производстве — это прямое нарушение "
        "действующего регламента, выданное как совет системы."
    )

    print("\nУбираю фикстуру и возвращаю исходный индекс...")
    target.unlink()
    ingestion.build_index(client, model)


# ====================================================================
#  ОСНОВНОЙ ПРОГОН
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Оценка качества и защита RAG")
    parser.add_argument("--embedded", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="перебор CHUNK_SIZE/OVERLAP (пересобирает индекс)")
    parser.add_argument("--conflict", action="store_true",
                        help="демо конфликта версий регламента")
    args = parser.parse_args()

    manifest = load_manifest()
    client = make_client(args.embedded, DB_PATH)
    print("Загружаю модель...")
    model = SentenceTransformer(MODEL_NAME)
    alias = manifest["alias"]

    try:
        if args.sweep:
            sweep_chunking(client, model, alias)
        elif args.conflict:
            demo_conflict(client, model, alias)
        else:
            print(_SEP2)
            print("КОНТРОЛЬНЫЙ ПРОГОН: разметка + LLM-as-Judge (GigaChat)")
            print(_SEP2)
            run_eval(client, model, alias)
    finally:
        client.close()


if __name__ == "__main__":
    main()
