"""
Загрузчик учебного корпуса ПромТеха. Менять не нужно.

Один документ корпуса — один файл data/corpus/<doc_id>.json:

    {
      "doc_id": "REGL-GEN-01",
      "type": "регламент",
      "title": "Регламент технического обслуживания насосного оборудования",
      "equipment": [],
      "version": "3.1",
      "date": "10.03.2024",
      "source_file": "generated/reglament_1.md",
      "source_url": "",
      "retrieved_at": "2026-07-31",
      "sha256": "9f2c...",
      "records": [
        {"page": null, "section": "5. Устранение неисправностей", "text": "..."}
      ]
    }

Документ хранится не сплошным текстом, а списком записей. Запись — это то,
на что можно сослаться в ответе: страница спарсенного PDF или раздел
сгенерированного документа. Из записи ссылка наследуется чанком, из чанка
попадает в payload Qdrant, из payload — в ответ ассистента. Разорвать эту
цепочку можно только в одном месте — в самом начале, здесь.

Метаданные тоже не украшение: по type и equipment фильтруется поиск,
по version и date видно, не отвечает ли система по отменённому документу,
по sha256 и source_url — откуда текст вообще взялся.

Две функции, потому что нужны оба взгляда на корпус:
    load_documents() — документы целиком (например, чтобы замерить, что
                       будет, если чанки не делать вовсе);
    load_records()   — плоский список записей, из которого строится индекс.
"""

import json
from pathlib import Path

CORPUS_DIR = Path(__file__).parent / "data" / "corpus"

# Поля, без которых документ бесполезен: по ним фильтруют и ссылаются.
REQUIRED_FIELDS = ("doc_id", "type", "title", "records")


def load_document(path: Path) -> dict:
    """Читает и проверяет один файл корпуса."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path.name}: файл не разбирается как JSON ({e})") from e

    for field in REQUIRED_FIELDS:
        if not doc.get(field):
            raise ValueError(f"{path.name}: не заполнено поле «{field}»")

    for i, rec in enumerate(doc["records"]):
        if not rec.get("text", "").strip():
            raise ValueError(f"{path.name}: запись {i} пустая")
        if rec.get("page") is None and not rec.get("section"):
            raise ValueError(
                f"{path.name}: у записи {i} нет ни page, ни section — "
                f"на такую запись нельзя будет сослаться")

    doc.setdefault("equipment", [])
    doc.setdefault("version", "")
    doc.setdefault("date", "")
    return doc


def load_documents() -> list:
    """Все документы корпуса, каждый со своим списком записей."""
    files = sorted(CORPUS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"Корпус пуст: в {CORPUS_DIR} нет файлов .json.\n"
            f"Сначала собери его: python parse_documents.py и python generate_corpus.py")
    return [load_document(f) for f in files]


def load_records() -> list:
    """Плоский список записей корпуса — из него строится индекс.

    Каждая запись несёт метаданные своего документа: иначе после нарезки
    на чанки станет непонятно, откуда кусок текста взялся.
    """
    records = []
    for doc in load_documents():
        for i, rec in enumerate(doc["records"]):
            records.append({
                "doc_id": doc["doc_id"],
                "record_no": i,
                "type": doc["type"],
                "title": doc["title"],
                "equipment": doc["equipment"],
                "version": doc["version"],
                "date": doc["date"],
                "page": rec.get("page"),
                "section": rec.get("section"),
                "text": rec["text"],
            })
    return records


def document_text(doc: dict) -> str:
    """Документ одной строкой — все записи подряд."""
    return "\n\n".join(rec["text"] for rec in doc["records"])


def location(rec: dict) -> str:
    """Ссылка на место в документе: «стр. 12» или название раздела."""
    if rec.get("page") is not None:
        return f"стр. {rec['page']}"
    return rec.get("section") or "—"


if __name__ == "__main__":
    docs = load_documents()
    records = load_records()

    print(f"Документов: {len(docs)}   записей: {len(records)}\n")
    print(f"  {'doc_id':<18} {'тип':<22} {'записей':>7} {'символов':>9}  версия/дата")
    print("  " + "-" * 76)
    for d in docs:
        chars = sum(len(r["text"]) for r in d["records"])
        stamp = " / ".join(x for x in (d["version"], d["date"]) if x) or "—"
        print(f"  {d['doc_id']:<18} {d['type']:<22} {len(d['records']):>7} "
              f"{chars:>9}  {stamp}")

    types = sorted({d["type"] for d in docs})
    print(f"\n  Типов документов: {len(types)} — {', '.join(types)}")
    no_version = [d["doc_id"] for d in docs if not d["version"]]
    print(f"  Без версии: {len(no_version)} из {len(docs)}")
