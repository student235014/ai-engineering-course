"""
Модуль 3, Занятие 6, практика 2 — поиск по индексу и оценка качества.

ТВОЯ РАБОТА: TODO 1 (свои запросы), TODO 2 (фильтр по типу документа),
TODO 3 (порог уверенности). И TODO 2 в ingestion.py — подбор CHUNK_SIZE
и OVERLAP по цифрам, которые напечатает этот скрипт.

Индекс построен в практике 1 (ingestion.py). Здесь мы по нему ищем и меряем,
насколько хорошо он ищет.

Мерить обязательно. «Вроде находит» — не результат: на одном запросе повезло,
на другом нет, а решение о том, класть ли ассистента в производственный контур,
принимают по цифрам. Поэтому в конце этого скрипта считаются две метрики
на одном и том же наборе контрольных вопросов:

  нужный документ в тройке — попал ли в выдачу тот документ, где ответ есть;
  нужный факт в тексте    — оказалась ли в найденных кусках сама цифра.

Вторая метрика существует потому, что первая обманывает. Документ может попасть
в тройку «по общему смыслу», а нужного абзаца в найденных кусках не будет —
и генератор в Модуле 4 честно ответит неправильно, имея на руках правильный
источник. Расхождение между двумя колонками показывает, чья это будет ошибка:
поиска или генерации.

Qdrant работает в Docker (веб-интерфейс http://localhost:6333/dashboard).
Резервный режим без Docker: python vector_store.py --embedded

Запуск (из папки module_03):
    python ingestion.py        # сначала собрать индекс
    python vector_store.py     # потом искать и мерить
"""

import argparse
import json
import sys
from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

from corpus_loader import corpus_fingerprint, document_text, load_documents
from qdrant_connect import DASHBOARD_URL, make_client

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "qdrant_chunks"
MANIFEST_PATH = BASE_DIR / "data" / "index_manifest.json"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

QUESTIONS = json.loads(
    (BASE_DIR / "data" / "questions.json").read_text(encoding="utf-8")
)["questions"]

TOP_K = 3


# ====================================================================
#  НАСТРОЙКИ СТУДЕНТА
# ====================================================================

# TODO 1: запросы, которые прогоняем по базе.
# Первые четыре не трогай — именно с ними твой вывод совпадёт с примерами
# в задании. На место ___ впиши два своих запроса: один настоящий рабочий
# вопрос и один каверзный, где легко промахнуться.
QUERIES = [
    "что делать если лента конвейера буксует",
    "как часто проводить углублённую проверку насоса",
    "кто согласует ремонт насосного оборудования",
    "что за инцидент был с гидравлическим прессом",
    "___",
    "___",
]

# TODO 3: порог уверенности. Если лучший score ниже этого числа — считаем,
# что в базе ответа нет, и честно об этом говорим.
# Сейчас 0.0 — порога нет вовсе, в выводе видно всё подряд, включая мусор.
# Подбери его по последнему блоку вывода: там напечатаны худший score
# вопроса из базы и лучший score вопроса, которого в базе нет.
MIN_SCORE = 0.0


# ====================================================================
#  ПОИСК
# ====================================================================

def load_manifest() -> dict:
    """Читает манифест индекса: чем и с какими параметрами он собран.

    Манифест — это ответ на вопрос «а что сейчас в базе». Без него легко
    полдня мерить старый индекс, собранный другими параметрами.
    """
    if not MANIFEST_PATH.exists():
        print("Индекса нет. Сначала собери его: python ingestion.py")
        sys.exit(1)
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def search(client, model, query: str, alias: str,
           doc_type: str = None, limit: int = TOP_K):
    """Ищет в базе куски, близкие к запросу.

    TODO 2: сейчас функция игнорирует doc_type и всегда ищет по всей базе.
    Инженеру нужен ответ из регламента, а не похожий по словам кусок
    из журнала инцидентов — поэтому поиск должен уметь ограничиваться типом.

    Что сделать (Filter, FieldCondition и MatchValue уже импортированы):
      1. Если doc_type — None, фильтра нет: query_filter остаётся None.
      2. Иначе собрать фильтр:
         Filter(must=[FieldCondition(key="type",
                                     match=MatchValue(value=doc_type))])
      3. Передать его в query_points параметром query_filter.

    Поле «type» проиндексировано в ingestion.py — фильтрация по нему
    не заставляет Qdrant перебирать всю коллекцию.
    """
    vector = model.encode(query).tolist()

    query_filter = None

    return client.query_points(
        collection_name=alias,
        query=vector,
        query_filter=query_filter,
        limit=limit,
    ).points


def where(payload: dict) -> str:
    """Ссылка на источник: страница спарсенного PDF или раздел документа."""
    if payload.get("page") is not None:
        return f"стр. {payload['page']}"
    return payload.get("section") or "—"


def show(hits, query: str):
    print(f"\n  Запрос: «{query}»")
    if not hits:
        print("    ничего не найдено")
        return
    for h in hits:
        p = h.payload
        print(f"    {h.score:5.3f}  [{p['type']:<21}] {p['doc_id']:<15} "
              f"{where(p)}")
    if hits[0].score < MIN_SCORE:
        print(f"    -> лучший score {hits[0].score:.3f} ниже порога {MIN_SCORE}: "
              f"считаем, что ответа в базе НЕТ")


# ====================================================================
#  ОЦЕНКА
# ====================================================================

def check_questions(client, model, alias: str) -> tuple:
    """Прогоняет контрольные вопросы и считает две метрики."""
    print("\n" + "=" * 92)
    print(f"КОНТРОЛЬНЫЕ ВОПРОСЫ (топ-{TOP_K})")
    print("=" * 92)

    doc_hits, fact_hits, total = 0, 0, 0
    width = max(len(q["question"]) for q in QUESTIONS)

    for q in QUESTIONS:
        found = search(client, model, q["question"], alias)
        top_docs = [p.payload["doc_id"] for p in found]
        best = found[0].score if found else 0.0

        if not q["in_corpus"]:
            print(f"  ??  {q['question']:<{width}} score={best:.3f}  <- вопроса нет в базе")
            print(f"      нашли: {', '.join(top_docs)}   (ответа быть НЕ должно)")
            continue

        total += 1
        ok_doc = any(d in q["expected"] for d in top_docs)
        doc_hits += ok_doc

        fact = q.get("must_contain", "")
        ok_fact = bool(fact) and any(fact.lower() in p.payload["text"].lower()
                                     for p in found)
        fact_hits += ok_fact

        mark = "OK" if ok_doc else " X"
        print(f"  {mark}  {q['question']:<{width}} score={best:.3f}   "
              f"документ: {'да' if ok_doc else 'нет':<3} факт: {'да' if ok_fact else 'нет'}")
        print(f"      нашли: {', '.join(top_docs)}   ждали: {', '.join(q['expected'])}")

    print(f"\n  Нужный документ в тройке: {doc_hits}/{total}")
    print(f"  Нужный факт в найденных кусках: {fact_hits}/{total}")
    return doc_hits, fact_hits, total


def baseline_whole_documents(model) -> tuple:
    """Сколько находится, если чанки не делать вовсе.

    Считаем прямо здесь, ничего не индексируя: 28 документов — это 28 векторов,
    их можно сравнить с вопросом в памяти. Коллекции из целых документов в Qdrant
    не создаём: так не делают, и заводить её даже ради демонстрации не стоит.
    """
    print("\n" + "=" * 92)
    print("БАЗОВАЯ ЛИНИЯ: тот же поиск, но по документам ЦЕЛИКОМ (без чанков)")
    print("=" * 92)

    docs = load_documents()
    doc_vectors = model.encode([document_text(d) for d in docs],
                               batch_size=16, normalize_embeddings=True)

    hits, total = 0, 0
    real, traps = [], []
    for q in QUESTIONS:
        qv = model.encode(q["question"], normalize_embeddings=True)
        scores = doc_vectors @ qv                      # косинус: векторы нормированы
        best = sorted(range(len(docs)), key=lambda i: -scores[i])[:TOP_K]
        top_docs = [docs[i]["doc_id"] for i in best]
        top_score = float(scores[best[0]])

        if not q["in_corpus"]:
            traps.append((top_score, q["question"]))
            continue

        total += 1
        real.append((top_score, q["question"]))
        ok = any(d in q["expected"] for d in top_docs)
        hits += ok
        print(f"  {'OK' if ok else ' X'}  {q['question'][:60]:<60} "
              f"score={top_score:.3f}  нашли: {', '.join(top_docs)}")

    print()
    print(f"  Нужный документ в тройке: {hits}/{total}")

    # Порог проверяем и здесь, ДО всякого чанкинга. Иначе легко решить, что
    # его сломал чанкинг, — а он не работал с самого начала.
    worst_real, best_trap = min(real), max(traps)
    print(f"  Худший вопрос из базы:  {worst_real[0]:.3f}  «{worst_real[1]}»")
    print(f"  Лучший вопрос вне базы: {best_trap[0]:.3f}  «{best_trap[1]}»")
    return hits, total


def threshold_report(client, model, alias: str):
    """Проверяет, существует ли порог, отделяющий вопросы из базы от вопросов вне её."""
    print("\n" + "=" * 92)
    print("ПОРОГ УВЕРЕННОСТИ")
    print("=" * 92)

    real, traps = [], []
    for q in QUESTIONS:
        found = search(client, model, q["question"], alias, limit=1)
        best = found[0].score if found else 0.0
        (real if q["in_corpus"] else traps).append((best, q["question"]))

    worst_real = min(real)
    best_trap = max(traps)

    print(f"  Худший вопрос ИЗ базы:  {worst_real[0]:.3f}  «{worst_real[1]}»")
    print(f"  Лучший вопрос ВНЕ базы: {best_trap[0]:.3f}  «{best_trap[1]}»")

    if best_trap[0] >= worst_real[0]:
        print("\n  Порога, который отсечёт второй и оставит первый, не существует:")
        print("  вопрос не из базы набрал больше настоящего. Одним числом эта")
        print("  задача не решается — к ней вернёмся в Модуле 4.")
    else:
        print(f"\n  Порог можно поставить между {best_trap[0]:.3f} и {worst_real[0]:.3f}.")
        print("  На этом корпусе повезло — на большем так не будет.")


def main():
    args = argparse.ArgumentParser(description="Поиск по индексу и оценка качества")
    args.add_argument("--embedded", action="store_true",
                      help="резервный режим без Docker (база в локальной папке)")
    args = args.parse_args()

    if any("___" in q for q in QUERIES):
        print("Сначала впиши свои запросы (TODO 1) — в QUERIES остались ___.")
        sys.exit(1)

    manifest = load_manifest()
    client = make_client(args.embedded, DB_PATH)
    print("Загружаю модель...")
    model = SentenceTransformer(MODEL_NAME)

    if manifest["model"] != MODEL_NAME:
        print(f"\nВНИМАНИЕ: индекс собран моделью {manifest['model']}, "
              f"а ищем моделью {MODEL_NAME}.")
        print("Векторы разных моделей несравнимы — пересобери индекс.\n")

    fingerprint = corpus_fingerprint()
    if manifest.get("corpus_fingerprint") != fingerprint:
        print()
        print("Индекс собран по ДРУГОМУ корпусу.")
        print(f"  в манифесте: {manifest.get('corpus_fingerprint')}")
        print(f"  в data/corpus сейчас: {fingerprint}")
        print("Искать по нему бессмысленно: ответы придут из документов,")
        print("которых в корпусе уже нет. Пересобери индекс: python ingestion.py")
        sys.exit(1)

    alias = manifest["alias"]
    print(f"Индекс: коллекция «{manifest['collection']}» (алиас «{alias}»), "
          f"собран {manifest['built_at']}")
    print(f"        CHUNK_SIZE={manifest['chunk_size']}, OVERLAP={manifest['overlap']}, "
          f"чанков {manifest['chunks']}")

    try:
        print("\n" + "=" * 92)
        print("ПОИСК ПО ВСЕЙ БАЗЕ")
        print("=" * 92)
        for q in QUERIES:
            show(search(client, model, q, alias), q)

        print("\n" + "=" * 92)
        print("ПОИСК С ФИЛЬТРОМ: только регламенты")
        print("=" * 92)
        for q in QUERIES:
            show(search(client, model, q, alias, doc_type="регламент"), q)

        check_questions(client, model, alias)
        baseline_whole_documents(model)
        threshold_report(client, model, alias)

        if not args.embedded:
            print(f"\nПосмотреть эти же точки глазами: {DASHBOARD_URL}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
