"""
Мост между Модулем 4 и цепочками Модуля 5. Менять не нужно.

Цепочки Занятий 9 и 10 зависят от ОДНОЙ функции с простым контрактом:

    answer(question: str) -> {"answer": str, "sources": list[str]}

Ровно этот контракт даёт заглушка rag_stub.py. Настоящий ассистент Модуля 4
отвечает тем же по смыслу словарём, но просит на вход больше: клиента Qdrant,
модель эмбеддингов и алиас коллекции —

    rag_assistant.answer(client, model, question, alias, doc_type=None)

Такое расхождение — обычное дело на стыке двух модулей, и переписывать из-за
него цепочки не нужно. Адаптер один раз поднимает тяжёлые объекты (клиент базы
и модель эмбеддингов живут до конца работы программы) и закрывает разницу
сигнатур. Цепочки продолжают знать только про answer(question).

Как включить настоящий RAG вместо заглушки — заменить один импорт:

    from rag_stub import answer as rag_answer        # заглушка
    from rag_adapter import answer as rag_answer     # Модуль 4

Требуется до запуска:
  * выполненный Модуль 4 в папке module_04 рядом с module_05;
  * собранный индекс: python ingestion.py в module_04 (с Docker или --embedded);
  * ключ GigaChat в .env — ответ по-прежнему генерирует модель.
"""

import sys
from pathlib import Path

M4_DIR = Path(__file__).parent.parent / "module_04"

if not M4_DIR.exists():
    raise ImportError(
        f"Не найдена папка Модуля 4: {M4_DIR}. Адаптер работает только вместе "
        f"с выполненным Модулем 4; пока его нет — используй rag_stub."
    )

sys.path.insert(0, str(M4_DIR))

from sentence_transformers import SentenceTransformer  # noqa: E402
from qdrant_connect import make_client                 # noqa: E402
from vector_store import MODEL_NAME, load_manifest     # noqa: E402

import rag_assistant                                   # noqa: E402

# режим базы: True — embedded (без Docker), False — сервер Qdrant в Docker
EMBEDDED = True

_client = None
_model = None
_alias = None


def _ready():
    """Поднимает клиента базы и модель эмбеддингов один раз за запуск."""
    global _client, _model, _alias
    if _client is None:
        _client = make_client(EMBEDDED, rag_assistant.DB_PATH)
        _model = SentenceTransformer(MODEL_NAME)
        manifest = load_manifest()
        _alias = manifest.get("alias", "promtech") if manifest else "promtech"
    return _client, _model, _alias


def answer(question: str) -> dict:
    """Контракт rag_stub поверх RAG-ассистента Модуля 4.

    Лишние поля ответа Модуля 4 (best_score, refused_by) наружу не отдаём:
    цепочкам они не нужны, а контракт должен остаться узким.
    """
    client, model, alias = _ready()
    result = rag_assistant.answer(client, model, question, alias)
    return {"answer": result["answer"], "sources": result["sources"]}


if __name__ == "__main__":
    for q in ["В какой срок устраняются критичные неисправности насосов?",
              "Какая столица Австралии?"]:
        r = answer(q)
        print(f"Вопрос   : {q}")
        print(f"Ответ    : {r['answer'][:200]}")
        print(f"Источники: {r['sources'] or '— нет'}")
        print()
