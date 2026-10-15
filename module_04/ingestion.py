"""
Модуль 3, Занятие 6, практика 1 (индекс чанков) — ЭТАЛОННОЕ РЕШЕНИЕ
(не выдавать студентам до сдачи). Заполнены TODO 1 (chunk_text) и TODO 2 (параметры).

Этот скрипт строит поисковый индекс. Ищет по нему следующий — vector_store.py.

Что кладём в базу. Не документы, а чанки — куски по несколько абзацев.
Полный документ в векторное хранилище не кладут никогда: регламент на 20 страниц
превращается в один вектор, «средний смысл всего регламента» не близок ни
к одному конкретному вопросу, и найти по нему ничего нельзя. Насколько именно
нельзя — измерим в vector_store.py, там это отдельный блок.

Два параметра решают всё:
  CHUNK_SIZE — размер куска. Слишком большой — смысл размывается.
               Слишком маленький — ответ разрывается пополам.
  OVERLAP    — нахлёст между кусками. Нужен, чтобы предложение, попавшее
               на границу, не потерялось: оно войдёт и в конец одного чанка,
               и в начало следующего.

Три вещи, которые здесь сделаны не «по-учебному», а как в рабочем проекте:

  1. Идентификатор чанка считается из doc_id, страницы/раздела и номера куска
     (UUID5). Он не зависит от порядка обработки, поэтому повторная загрузка
     обновляет те же самые точки, а не плодит дубликаты.
  2. Индекс собирается в НОВУЮ коллекцию, и только потом на неё переключается
     алиас promtech. Пока идёт сборка, поиск продолжает работать по старой
     коллекции. Так выкатывают индексы в проде — не «удалить и залить заново».
  3. Рядом с индексом пишется манифест: какой моделью, с какими параметрами
     и по какому корпусу он собран. Без манифеста через месяц невозможно
     ответить, что вообще лежит в базе.

Qdrant работает в Docker (веб-интерфейс http://localhost:6333/dashboard).
Если Docker поставить не удалось — резервный режим: python ingestion.py --embedded

Запуск (из папки module_03):
    python ingestion.py
"""

import argparse
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation,
    Distance, PayloadSchemaType, PointStruct, VectorParams,
)
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from corpus_loader import CORPUS_DIR, load_records
from qdrant_connect import DASHBOARD_URL, make_client

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = Path(__file__).parent / "qdrant_chunks"
MANIFEST_PATH = Path(__file__).parent / "data" / "index_manifest.json"

# Под этим именем индекс ищет vector_store.py. Это алиас, а не коллекция:
# коллекция у каждой сборки своя, алиас всегда показывает на актуальную.
ALIAS = "promtech"
COLLECTION_PREFIX = "promtech_chunks"

# Сколько прошлых коллекций оставляем: одну — чтобы было куда откатить алиас,
# если новая сборка окажется хуже.
KEEP_COLLECTIONS = 2

# Кодируем пачками, а не по одному вектору: на 1000 чанков разница
# в несколько раз, и видно прогресс.
ENCODE_BATCH = 64

# Фиксированное пространство имён для UUID5. Менять нельзя: от него зависят
# идентификаторы всех чанков.
NAMESPACE = uuid.UUID("6f8a9c2e-3d51-4b7a-9f10-2c4d6e8b0a13")


# ====================================================================
#  НАСТРОЙКИ СТУДЕНТА
# ====================================================================

# TODO 2 (подбирается в практике 2, после первого замера): размер куска
# и нахлёст. Начни с этих значений, поменяешь их позже — когда увидишь,
# сколько вопросов находится.
CHUNK_SIZE = 600
OVERLAP = 150


# ====================================================================
#  НАРЕЗКА НА ЧАНКИ
# ====================================================================

def chunk_text(text: str, size: int, overlap: int) -> list:
    """Режет текст на куски по `size` символов с нахлёстом `overlap`.

    Решение TODO 1.
    """
    if overlap >= size:
        raise ValueError(
            f"overlap ({overlap}) должен быть меньше size ({size}): "
            f"иначе шаг окна нулевой и нарезка зациклится"
        )

    chunks = []
    step = size - overlap
    start = 0

    while start < len(text):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(piece)
        start += step

    return chunks


# ====================================================================
#  КОД НИЖЕ УЖЕ РАБОТАЕТ — менять не нужно
# ====================================================================

def chunk_id(record: dict, chunk_no: int) -> str:
    """Идентификатор чанка, который не меняется от прогона к прогону.

    Считается из документа, номера записи в нём, места (страница или раздел)
    и номера куска — а не из того, каким по счёту его обработали во всём
    корпусе. Иначе после правки одного документа сдвинутся id у всех
    следующих, и обновить в базе один документ станет невозможно.

    Номер записи в ключе обязателен, и вот почему. Названия разделов в
    сгенерированных документах повторяются: модель дважды пишет
    «Проверяемые факты», а к тому же собственная шапка документа может
    совпасть по названию с разделом. Без номера записи два разных куска
    получили бы один идентификатор, и один молча затёр бы другой при
    загрузке — в базе стало бы меньше точек, чем чанков.
    """
    key = (f"{record['doc_id']}|{record.get('record_no')}|{record.get('page')}"
           f"|{record.get('section')}|{chunk_no}")
    return str(uuid.uuid5(NAMESPACE, key))


def build_chunks(records: list) -> list:
    """Режет каждую запись корпуса отдельно и переносит на чанк её метаданные.

    Режем именно запись, а не документ целиком: тогда чанк не может
    склеиться из двух разных страниц, и ссылка на источник остаётся точной.
    """
    chunks = []
    for record in records:
        pieces = chunk_text(record["text"], CHUNK_SIZE, OVERLAP)
        for chunk_no, piece in enumerate(pieces):
            chunks.append({
                "id": chunk_id(record, chunk_no),
                "payload": {
                    "doc_id": record["doc_id"],
                    "type": record["type"],
                    "title": record["title"],
                    "equipment": record["equipment"],
                    "version": record["version"],
                    "date": record["date"],
                    "page": record["page"],
                    "section": record["section"],
                    "chunk_no": chunk_no,
                    "text": piece,
                },
            })
    return chunks


def corpus_fingerprint() -> str:
    """Отпечаток корпуса: по нему видно, что индекс собран по этим самым файлам."""
    h = hashlib.sha256()
    for path in sorted(CORPUS_DIR.glob("*.json")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def switch_alias(client: QdrantClient, collection: str):
    """Переводит алиас на новую коллекцию одной атомарной операцией.

    Удаление несуществующего алиаса — не ошибка, поэтому первый запуск
    отрабатывает так же, как все последующие.
    """
    client.update_collection_aliases(change_aliases_operations=[
        DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=ALIAS)),
        CreateAliasOperation(create_alias=CreateAlias(
            collection_name=collection, alias_name=ALIAS)),
    ])


def drop_old_collections(client: QdrantClient, keep: int):
    """Убирает старые сборки индекса, оставляя последние `keep`."""
    ours = sorted(c.name for c in client.get_collections().collections
                  if c.name.startswith(COLLECTION_PREFIX))
    for name in ours[:-keep]:
        client.delete_collection(name)
        print(f"    удалена старая коллекция: {name}")


def build_index(client: QdrantClient, model: SentenceTransformer) -> tuple:
    """Собирает новую коллекцию из чанков и переключает на неё алиас."""
    records = load_records()
    chunks = build_chunks(records)
    if not chunks:
        raise RuntimeError("нарезка не дала ни одного чанка — проверь chunk_text")

    print(f"Записей корпуса: {len(records)}  ->  чанков: {len(chunks)}")
    print("Считаю эмбеддинги...")
    vectors = model.encode([c["payload"]["text"] for c in chunks],
                           batch_size=ENCODE_BATCH, show_progress_bar=True)

    collection = f"{COLLECTION_PREFIX}_{time.strftime('%Y%m%d_%H%M%S')}"
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=model.get_embedding_dimension(),
            distance=Distance.COSINE,
        ),
    )

    # Индекс по полям, по которым будем фильтровать. Без него Qdrant при
    # фильтрации перебирает всю коллекцию — на учебном корпусе незаметно,
    # на реальном заметно сразу.
    for field in ("type", "doc_id", "equipment"):
        client.create_payload_index(collection_name=collection, field_name=field,
                                    field_schema=PayloadSchemaType.KEYWORD)

    client.upsert(
        collection_name=collection,
        points=[PointStruct(id=c["id"], vector=v.tolist(), payload=c["payload"])
                for c, v in zip(chunks, vectors)],
        wait=True,
    )

    switch_alias(client, collection)
    drop_old_collections(client, KEEP_COLLECTIONS)

    manifest = {
        "alias": ALIAS,
        "collection": collection,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "vector_size": model.get_embedding_dimension(),
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "documents": len({r["doc_id"] for r in records}),
        "records": len(records),
        "chunks": len(chunks),
        "corpus_fingerprint": corpus_fingerprint(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    return manifest, chunks


def show_sample(chunks: list, count: int = 3):
    """Показывает несколько чанков так, как их увидит поиск."""
    print("\n" + "=" * 88)
    print("ЧТО ЛЕЖИТ В ИНДЕКСЕ (первые чанки)")
    print("=" * 88)
    for c in chunks[:count]:
        p = c["payload"]
        where = f"стр. {p['page']}" if p["page"] is not None else p["section"]
        preview = " ".join(p["text"].split())[:110]
        print(f"\n  {p['doc_id']} · {where} · кусок {p['chunk_no']}")
        print(f"    {preview}…")


def main():
    args = argparse.ArgumentParser(description="Сборка индекса чанков в Qdrant")
    args.add_argument("--embedded", action="store_true",
                      help="резервный режим без Docker (база в локальной папке)")
    args = args.parse_args()

    client = make_client(args.embedded, DB_PATH)
    print("Загружаю модель (первый раз — несколько минут)...")
    model = SentenceTransformer(MODEL_NAME)

    try:
        started = time.time()
        manifest, chunks = build_index(client, model)
        show_sample(chunks)

        print("\n" + "=" * 88)
        print(f"ИНДЕКС СОБРАН за {time.time() - started:.1f} с")
        print("=" * 88)
        print(f"  CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={OVERLAP}")
        print(f"  документов {manifest['documents']}, записей {manifest['records']}, "
              f"чанков {manifest['chunks']}")
        print(f"  коллекция «{manifest['collection']}», алиас «{ALIAS}»")
        print(f"  манифест: {MANIFEST_PATH.name}")
        if not args.embedded:
            print(f"  веб-интерфейс: {DASHBOARD_URL}")
        print("\n  Дальше — поиск и оценка: python vector_store.py")
    finally:
        client.close()


if __name__ == "__main__":
    main()
