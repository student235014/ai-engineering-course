"""
Модуль 3, Занятие 5 (часть А) — парсинг публичных нормативных документов в корпус.

ТВОЯ РАБОТА: TODO 1 (SOURCES), TODO 2 (clean_text), TODO 3 (has_text_layer).

RAG отвечает по документам, а не по памяти модели. Значит, первый шаг —
собрать корпус: взять реальные нормативные документы (ФНП, ГОСТ, регламенты)
и вытащить из них текст.

Текст вытаскиваем ПОСТРАНИЧНО, и каждая страница становится отдельной записью
документа. Так в корпусе сохраняется номер страницы, и ответ ассистента сможет
выглядеть как «ГОСТ 32601-2022, стр. 14», а не «где-то в ГОСТе». Ссылка на
источник — требование к любому инженерному ассистенту: ответ, который нельзя
проверить, инженеру не нужен.

PDF бывают двух видов:
  - с текстовым слоем — текст извлекается программно;
  - скан без текстового слоя — там только картинки страниц. Такой файл
    ОТБРАКОВЫВАЕМ (распознавание сканов — OCR — отдельная задача, здесь её нет).

Два режима запуска (из папки module_03):
    python parse_documents.py                  # взять PDF из data/sources,
                                               # недостающие скачать по URL
    python parse_documents.py --local <папка>  # разобрать PDF из своей папки
                                               # (если сайты недоступны — PDF
                                               # раздаёт преподаватель)

Результат: файлы data/corpus/<doc_id>.json.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from pypdf import PdfReader

# Из длинного документа берём не больше 50 страниц: для учебного корпуса
# этого достаточно, а индексация не растягивается на часы.
MAX_PAGES = 50

# Если на страницу приходится меньше этого числа символов, текстового слоя
# в PDF скорее всего нет (это скан) — файл отбраковываем.
MIN_CHARS_PER_PAGE = 200

# Страницы короче этого в корпус не кладём: обложка, разделитель, пустой
# оборот. Записи из двух слов только мешают поиску.
MIN_CHARS_PER_RECORD = 80

BASE_DIR = Path(__file__).parent
SOURCES_DIR = BASE_DIR / "data" / "sources"
CORPUS_DIR = BASE_DIR / "data" / "corpus"

# Журнал скачиваний: что, откуда и когда взяли. Три маленьких PDF лежат
# в репозитории вместе с этим журналом, большой ГОСТ каждый качает себе сам —
# и в обоих случаях видно происхождение файла.
FETCH_LOG = SOURCES_DIR / "fetch_log.json"

# ====================================================================
#  ИСТОЧНИКИ КОРПУСА — TODO 1
# ====================================================================
# TODO 1: заполнена только первая запись — это образец. Заполни три
# остальные. Все значения есть в задании, в таблице «Источники корпуса»,
# — переноси оттуда, выдумывать ничего не надо.
#
# Из чего состоит запись и почему поля важны:
#   url    — ПРЯМАЯ ссылка на файл PDF. Если ссылка ведёт на страницу
#            просмотрщика правовой базы, скачается HTML, и парсер его не
#            возьмёт: PdfReader упадёт на первом же байте;
#   file   — под каким именем PDF ляжет в data/sources;
#   doc_id — идентификатор документа в корпусе и имя файла корпуса
#            (EXT-FNP-461 -> data/corpus/EXT-FNP-461.json). Обязан быть
#            уникальным: именно им ассистент ссылается на источник ответа;
#   type   — тип документа. По нему в Занятии 6 фильтруется поиск, поэтому
#            у всех публичных норм он одинаковый: «нормативный документ»;
#   title  — человекочитаемое название: попадёт в метаданные и в выдачу поиска;
#   date   — дата документа в виде ГГГГ-ММ-ДД или просто год. Такие даты
#            сортируются как обычные строки, «26.11.2020» — нет. Если в
#            документе даты нет, оставь пустую строку: пустое поле честнее
#            выдуманной даты.
#
# Всё это вместе — паспорт документа. Мы кладём в базу не голый текст,
# а текст с паспортом: без него нельзя ни отфильтровать поиск, ни сослаться
# на источник.
# --------------------------------------------------------------------
SOURCES = [
    {   # ОБРАЗЕЦ — эта запись уже заполнена, менять её не нужно
        "url": ("http://mos.gosnadzor.ru/about/documents/"
                "Приказ РТН № 461 от 26.11.2020.pdf"),
        "file": "ext_fnp461.pdf",
        "doc_id": "EXT-FNP-461",
        "type": "нормативный документ",
        "title": "ФНП: Правила безопасности при работе с подъёмными сооружениями "
                 "(приказ Ростехнадзора № 461)",
        "date": "2020-11-26",
    },
    {   # ГОСТ 32601-2022, насосы центробежные — заполни по образцу
        "url": ("https://files.stroyinf.ru/Data/784/78426.pdf"),
        "file": "ext_gost32601.pdf",
        "doc_id": "EXT-GOST-32601",
        "type": "нормативный документ",
        "title": "ГОСТ 32601-2022: Насосы центробежные для нефтяной, "
                 "нефтехимической и газовой промышленности",
        "date": "2022",
    },
    {   # Регламент ТО и ремонта технологических трубопроводов
        "url": ("https://gostbank.metaltorg.ru/data/norms_/jr/12.pdf"),
        "file": "ext_reglament_truboprovody.pdf",
        "doc_id": "EXT-REGL-TRUB",
        "type": "нормативный документ",
        "title": "Регламент технического обслуживания и ремонта "
                 "технологических трубопроводов",
        "date": "-",
    },
    {   # Регламент технического обслуживания оборудования (PERCo)
        "url": ("https://www.perco.ru/download/documentation/rus/Reglament-TO.pdf"),
        "file": "ext_reglament_to.pdf",
        "doc_id": "EXT-REGL-TO",
        "type": "нормативный документ",
        "title": "Регламент технического обслуживания оборудования (PERCo)",
        "date": "-",
    },
]

# Поля записи источника — по ним проверяется, что TODO 1 сделан.
SOURCE_FIELDS = ("url", "file", "doc_id", "type", "title")


def unfilled_sources() -> list:
    """Записи SOURCES, в которых остались незаполненные поля."""
    bad = []
    for i, s in enumerate(SOURCES, 1):
        missing = [f for f in SOURCE_FIELDS if not str(s.get(f, "")).strip()]
        if missing:
            bad.append((i, missing))
    return bad


# ====================================================================
#  СКАЧИВАНИЕ И ПРОИСХОЖДЕНИЕ ФАЙЛА
# ====================================================================

def sha256(path: Path) -> str:
    """Контрольная сумма файла.

    Через полгода придётся отвечать на вопрос «это тот же документ, по которому
    мы отвечали в июле?». Ответ даёт только хэш — имя файла и размер совпадут
    и у новой редакции.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_log() -> dict:
    if FETCH_LOG.exists():
        return json.loads(FETCH_LOG.read_text(encoding="utf-8"))
    return {}


def write_log(log: dict):
    FETCH_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2),
                         encoding="utf-8")


def remember_fetch(name: str, url: str, path: Path):
    log = read_log()
    log[name] = {
        "url": url,
        "retrieved_at": time.strftime("%Y-%m-%d"),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    write_log(log)


def download(url: str, dest: Path) -> bool:
    """Скачивает файл по URL. Возвращает True при успехе.

    Тонкости, ради которых функция длиннее одной строки:
      - в URL бывают русские буквы и пробелы — их надо закодировать;
      - у части госсайтов сертификаты не из стандартных корневых центров,
        поэтому при ошибке SSL пробуем ещё раз без проверки сертификата
        (для публичных документов это допустимо);
      - сайт может быть недоступен — тогда честно сообщаем и идём дальше.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    уже на диске: {dest.name} ({dest.stat().st_size // 1024} КБ)")
        return True

    safe_url = urllib.parse.quote(url, safe=":/%?&=")
    headers = {"User-Agent": "Mozilla/5.0 (learning-project; corpus-builder)"}

    for verify in (True, False):
        try:
            resp = requests.get(safe_url, headers=headers, timeout=60, verify=verify)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            note = "" if verify else "  (сертификат сайта не проверялся)"
            print(f"    скачан: {dest.name} ({len(resp.content) // 1024} КБ){note}")
            remember_fetch(dest.name, url, dest)
            return True
        except requests.exceptions.SSLError:
            if verify:
                continue  # вторая попытка — без проверки сертификата
            print(f"    ОШИБКА SSL: {url}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"    НЕ СКАЧАЛСЯ: {url}\n    причина: {e}")
            return False
    return False


# ====================================================================
#  ИЗВЛЕЧЕНИЕ ТЕКСТА
# ====================================================================

def clean_text(text: str) -> str:
    """Приводит извлечённый из PDF текст в порядок."""
    # 1. Разбить на строки
    lines = text.splitlines()
    # 2. В каждой строке схлопнуть пробелы/табы и обрезать края
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    # 3. Склеить обратно
    text = "\n".join(lines)
    # 4. Три и больше переводов строки → два (граница абзаца)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 5. Обрезать края всего текста
    return text.strip()


def extract_pages(pdf_path: Path) -> tuple:
    """Извлекает текст постранично (не больше MAX_PAGES страниц).

    Возвращает (список записей, число обработанных страниц, всего страниц).
    Запись — это {"page": номер, "section": None, "text": текст страницы}.
    Решение об отбраковке принимает вызывающий код.
    """
    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    pages = reader.pages[:MAX_PAGES]

    records = []
    for number, page in enumerate(pages, 1):
        # extraction_mode="layout" сохраняет пробелы между словами:
        # обычный режим на многих русских PDF рвёт каждое слово на свою строку.
        text = clean_text(page.extract_text(extraction_mode="layout") or "")
        if len(text) >= MIN_CHARS_PER_RECORD:
            records.append({"page": number, "section": None, "text": text})

    return records, len(pages), total


def has_text_layer(text: str, pages: int) -> bool:
    """Скан или настоящий текст?"""
    if pages == 0:
        return False
    chars_per_page = len(text) / pages
    return chars_per_page >= MIN_CHARS_PER_PAGE


# ====================================================================
#  СБОРКА ФАЙЛА КОРПУСА
# ====================================================================

def write_corpus_file(meta: dict, records: list, pdf_path: Path) -> Path:
    """Пишет документ корпуса: метаданные, происхождение файла и записи."""
    log = read_log().get(pdf_path.name, {})
    doc = {
        "doc_id": meta["doc_id"],
        "type": meta["type"],
        "title": meta["title"],
        "equipment": [],
        "version": "",
        "date": meta.get("date", ""),
        "source_file": f"sources/{pdf_path.name}",
        "source_url": log.get("url", meta.get("url", "")),
        "retrieved_at": log.get("retrieved_at", ""),
        "sha256": log.get("sha256") or sha256(pdf_path),
        "records": records,
    }

    out = CORPUS_DIR / f"{meta['doc_id']}.json"
    # ensure_ascii=False — иначе кириллица уедет в \uXXXX-последовательности
    # и корпус станет нечитаемым; indent=2 — чтобы в git было видно построчный дифф.
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    return out


def process_pdf(pdf_path: Path, meta: dict) -> bool:
    """Полный цикл для одного PDF: страницы -> проверка слоя -> файл корпуса."""
    try:
        records, pages, total = extract_pages(pdf_path)
    except Exception as e:
        print(f"    ОТБРАКОВАН: {pdf_path.name} — файл не читается как PDF ({e})")
        return False

    text = "\n".join(r["text"] for r in records)

    if not has_text_layer(text, pages):
        print(f"    ОТБРАКОВАН: {pdf_path.name} — в PDF нет текстового слоя "
              f"(это скан: {len(text)} симв. на {pages} стр.).")
        print("    Такой файл нужно сначала распознать (OCR) — "
              "в этом задании сканы не обрабатываем.")
        return False

    out = write_corpus_file(meta, records, pdf_path)
    pages_note = f"{pages} из {total}" if total > pages else str(total)
    print(f"    OK: {out.name} — страниц {pages_note}, записей {len(records)}, "
          f"{len(text)} символов")
    return True


# ====================================================================
#  РЕЖИМЫ ЗАПУСКА
# ====================================================================

def run_from_sources() -> tuple:
    """Режим по умолчанию: взять PDF из data/sources, недостающие — скачать.

    Маленькие документы лежат в репозитории, и для них шаг скачивания
    пропускается. Большой ГОСТ (75 МБ) в репозиторий не кладут — он
    качается сюда же при первом запуске.
    """
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    ok, rejected = 0, 0
    for meta in SOURCES:
        print(f"\n[{meta['doc_id']}] {meta['title']}")
        pdf_path = SOURCES_DIR / meta["file"]
        if not download(meta["url"], pdf_path):
            rejected += 1
            continue
        if process_pdf(pdf_path, meta):
            ok += 1
        else:
            rejected += 1
    return ok, rejected


def run_from_dir(folder: Path) -> tuple:
    """Режим --local: обработать все PDF из указанной папки."""
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"В папке {folder} нет PDF-файлов.")
        return 0, 0

    ok, rejected = 0, 0
    for i, pdf_path in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] {pdf_path.name}")
        # Известным файлам (совпало имя с SOURCES) оставляем их метаданные,
        # остальным — собираем метаданные из имени файла.
        meta = next(
            (s for s in SOURCES if s["file"] == pdf_path.name),
            {
                "file": pdf_path.name,
                "doc_id": f"EXT-LOCAL-{i:02d}",
                "type": "нормативный документ",
                "title": pdf_path.stem.replace("_", " ").replace("-", " "),
                "date": "",
            },
        )
        if process_pdf(pdf_path, meta):
            ok += 1
        else:
            rejected += 1
    return ok, rejected


def main():
    parser = argparse.ArgumentParser(description="Парсинг PDF в учебный корпус")
    parser.add_argument("--local", metavar="ПАПКА",
                        help="взять PDF из своей папки вместо data/sources")
    args = parser.parse_args()

    unfilled = unfilled_sources()
    if unfilled and not args.local:
        print("TODO 1 не сделан: в списке SOURCES остались пустые поля.")
        for i, missing in unfilled:
            print(f"  запись {i}: не заполнено — {', '.join(missing)}")
        sys.exit(1)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"ПАРСИНГ ДОКУМЕНТОВ В КОРПУС (не больше {MAX_PAGES} страниц с документа)")
    print("=" * 78)

    if args.local:
        folder = Path(args.local)
        if not folder.is_dir():
            print(f"Папка не найдена: {folder}")
            sys.exit(1)
        ok, rejected = run_from_dir(folder)
    else:
        ok, rejected = run_from_sources()

    print("\n" + "=" * 78)
    print(f"ИТОГ: в корпус попало {ok}, отбраковано/не скачалось {rejected}")
    print(f"Файлы корпуса: {CORPUS_DIR}")
    print("=" * 78)


if __name__ == "__main__":
    main()
