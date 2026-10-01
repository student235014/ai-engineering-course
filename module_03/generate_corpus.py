"""
Модуль 3, Занятие 5 (часть Б) — генерация внутренних документов ПромТеха
через GigaChat.

ТВОЯ РАБОТА: TODO 4 (validate_equipment_id) и TODO 5 (extract_json_array).
Сделай их ДО запуска: без TODO 5 скрипт спотыкается на ответах в ```json```,
без TODO 4 в корпус пролезают карточки выдуманного оборудования.

Публичные документы (ФНП, ГОСТ) мы спарсили — но у настоящего предприятия
основа базы знаний ВНУТРЕННЯЯ: карточки оборудования, журнал инцидентов,
свои регламенты. Таких документов в интернете нет, поэтому для учебного
предприятия мы их генерируем.

Почему проверять RAG будем именно по сгенерированным документам: публичные
ФНП и ГОСТы могли попасть в обучающую выборку модели — тогда непонятно,
отвечает она ПО БАЗЕ или ПО ПАМЯТИ. Сгенерированный вчера регламент
в обучении быть не мог: если модель называет срок из него — значит, RAG
действительно работает.

Три типа документов (три промпта):
  1. Карточки оборудования — по одной на каждую единицу из реестра
     (equipment_registry.json — источник истины, id не выдумываем).
  2. Журнал инцидентов — записи ссылаются на equipment_id из карточек.
  3. Внутренние регламенты — с конкретными проверяемыми фактами
     (сроки, пороги, роли) и разделом исключений.

Генерируем батчами по 3-5 объектов: короткий ответ проще проверить,
битый JSON в маленьком батче дешевле перегенерировать.

Сырые ответы модели складываются в data/generated и остаются на диске.
Из них собираются файлы корпуса — и пересобрать корпус можно, ничего
не генерируя заново:

    python generate_corpus.py                 # всё подряд (обращается к API)
    python generate_corpus.py --part cards    # только карточки
    python generate_corpus.py --part incidents
    python generate_corpus.py --part reglaments
    python generate_corpus.py --rebuild       # пересобрать корпус из data/generated,
                                              # без обращения к API и без расхода токенов

Результат:
    data/generated/       — сырые ответы модели
    data/corpus/*.json    — документы корпуса
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated"
CORPUS_DIR = DATA_DIR / "corpus"

# ключ ищем сначала рядом с модулем, потом в корне проекта
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

# Реестр оборудования — источник истины для equipment_id (тот же, что в М2).
REGISTRY = json.loads(
    (DATA_DIR / "equipment_registry.json").read_text(encoding="utf-8")
)["equipment"]

# Латиница-двойники: модель может написать «KM-101» латиницей (см. М2).
_LAT_TO_CYR = str.maketrans("ABCEHKMOPTXYabcehkmoptxy", "АВСЕНКМОРТХУавсенкмортху")

def _norm(value: str) -> str:
    return value.strip().replace(" ", "-").translate(_LAT_TO_CYR).casefold()

REGISTRY_LOOKUP = {_norm(item["id"]): item["id"] for item in REGISTRY}


def validate_equipment_id(value) -> str | None:
    """Сверяет идентификатор оборудования с реестром.

    TODO 4: сейчас функция возвращает значение как есть — это заглушка.
    Она пропустит в корпус и «км 101», и латинское «KM-101», и полностью
    выдуманный моделью код — а в М2 мы уже выяснили, чем это кончается.

    Что сделать (всё уже готово выше — как в Занятии 4, модуль М2):
      1. Нормализовать написание: key = _norm(str(value)).
      2. Поискать key в REGISTRY_LOOKUP.
      3. Нашли — вернуть КАНОНИЧНУЮ запись из реестра (значение словаря):
         «км 101» -> «КМ-101», латиница -> кириллица.
      4. Не нашли — вернуть None: модель выдумала оборудование,
         объект будет отбракован.
    """
    return str(value).strip()


BATCH_SIZE = 5        # объектов за один вызов (батчи по 3-5)
N_INCIDENTS = 12      # записей журнала инцидентов

# Раздел короче этого приклеиваем к предыдущему: запись из одного заголовка
# и двух строк — плохая единица поиска, она ни на что не отвечает.
MIN_SECTION_CHARS = 200

# учёт токенов за прогон
_usage = {"prompt": 0, "completion": 0}


# ====================================================================
#  ПРОМПТЫ (утверждённые)
# ====================================================================

PROMPT_CARDS = """Ты — технический писатель промышленного предприятия ПромТех.
Сгенерируй {N} карточек оборудования в формате JSON по схеме:
{
"equipment_id": "строка — идентификатор из списка ниже",
"type": "тип оборудования (насос/теплообменник/реактор/трансформатор и т.д.)",
"location": "цех/участок (вымышленный)",
"install_date": "дата",
"last_maintenance": "дата",
"status": "в работе / на ТО / выведено из эксплуатации",
"spec_summary": "2-3 предложения технических характеристик",
"known_issues": ["список типовых неисправностей"]
}
Стилистика — как в реальных карточках оборудования нефтегазовой/энергетической отрасли,
но все данные полностью вымышленные, без привязки к реальным предприятиям.
Не используй существующие названия компаний или реальные номера объектов.

Используй СТРОГО следующие equipment_id из реестра предприятия —
ровно эти, по одной карточке на каждый (тип и характеристики должны
соответствовать названию из реестра):
{REGISTRY_LIST}

Верни только JSON-массив из {N} объектов, без пояснений и без markdown."""

PROMPT_INCIDENTS = """Сгенерируй {N} записей журнала инцидентов в формате JSON по схеме:
{
"incident_id": "INC-2026-001",
"date": "дата",
"equipment_id": "ссылка на equipment_id из карточек оборудования",
"description": "описание инцидента, 2-4 предложения, инженерный стиль",
"severity": "низкая/средняя/высокая",
"resolution": "как был устранён инцидент",
"resolved_by": "роль сотрудника (оператор/инженер/специалист по ТО)"
}
Данные вымышленные, но реалистичные для промышленного предприятия
(насосы, трансформаторы, котлы, ПЛК, трубопроводы).

Поле equipment_id выбирай СТРОГО из этого списка (id существующих карточек
оборудования, разные записи — про разное оборудование):
{CARD_IDS}

incident_id нумеруй с INC-2026-{START:03d} подряд без пропусков.
Верни только JSON-массив из {N} объектов, без пояснений и без markdown."""

PROMPT_REGLAMENT = """Ты — специалист по нормативной документации промышленного предприятия «ПромТех».

Сгенерируй внутренний регламент предприятия на тему: «{ТЕМА_РЕГЛАМЕНТА}»

Документ должен быть оформлен в Markdown и содержать:

1. Шапку документа:
- Наименование предприятия: ПромТех (вымышленное)
- Номер документа (напр. РГЛ-НО-014)
- Версия (напр. 3.1)
- Дата утверждения и дата вступления в силу
- Кем утверждён (вымышленная должность, без реальных ФИО)
- Область действия (какие подразделения/объекты)
- Периодичность пересмотра (напр. раз в 3 года)

2. 5-8 пронумерованных разделов по теме, КАЖДЫЙ раздел должен содержать
минимум 2-3 КОНКРЕТНЫХ проверяемых факта: точные сроки, пороговые значения,
интервалы, единицы измерения, названия ролей/должностей, номера форм отчётности.
Не используй общие фразы без цифр и конкретики — весь смысл в том, чтобы
по этим фактам потом можно было точно проверить, что RAG отвечает по документу,
а не придумывает.

Пример нужной конкретики: "Плановый осмотр проводится не реже 1 раза в 45 дней",
"Ответственный — начальник участка по эксплуатации", "Порог давления для
аварийной остановки — 1,6 МПа", "Заявка оформляется по форме РГЛ-Ф-07".

3. Раздел "Исключения и особые случаи" — 2-3 пункта с пограничными ситуациями,
которые регламент явно НЕ покрывает (например, "требования к работам на
объектах, не относящихся к [тип оборудования], регулируются отдельным
регламентом РГЛ-ХХ-НН" или "при отсутствии данных о типе оборудования
решение принимается инженером по эксплуатации в индивидуальном порядке").
Это нужно, чтобы протестировать, отказывается ли RAG отвечать там, где
в документе реально нет ответа.

4. Раздел "Ссылки на связанные документы" — 2-3 вымышленных, но правдоподобных
ссылки на другие внутренние регламенты/ГОСТы (в свободной форме, без выдачи
реальных номеров ГОСТов, если не уверен, что они существуют).

Все данные о предприятии, ролях, номерах документов — полностью вымышленные,
не привязывай к реальным компаниям."""

REGLAMENT_TOPICS = [
    "Регламент технического обслуживания насосного оборудования",
    "Регламент допуска персонала к работам повышенной опасности",
    "Регламент реагирования на инциденты на технологических установках",
]

# Номер РГЛ-НО-014 стоит в промпте как ПРИМЕР — модель любит копировать его
# во все документы подряд. Три регламента с одним номером — конфликт для RAG,
# поэтому после генерации присваиваем каждому документу свой номер.
REGLAMENT_NUMBERS = ["РГЛ-НО-014", "РГЛ-ДП-021", "РГЛ-РИ-007"]


def polish_reglament(text: str, number: str) -> str:
    """Чистит типовые артефакты генерации (промпт не меняем — правим результат)."""
    text = text.replace("РГЛ-НО-014", number)
    # пометки вида «(вымышленный человек)» и примечания про вымышленность:
    # промпт просил вымышленные данные, а модель иногда прямо это подписывает
    text = re.sub(r"\s*\((?:Примечание:)?[^()]*вымышленн[^()]*\)", "", text)
    # заглушки номеров ГОСТ из сплошных «Х» приводим к виду ХХХХ-ХХХХ
    text = re.sub(r"Х{5,}(?:-Х+)?", "ХХХХ-ХХХХ", text)
    return text


# ====================================================================
#  МЕТАДАННЫЕ: ДАТЫ, ВЕРСИЯ, РАЗДЕЛЫ
# ====================================================================

MONTHS = {m: i for i, m in enumerate(
    ["январ", "феврал", "март", "апрел", "ма", "июн", "июл",
     "август", "сентябр", "октябр", "ноябр", "декабр"], 1)}


def normalize_date(value) -> str:
    """Приводит дату к виду ГГГГ-ММ-ДД.

    Модель возвращает даты как придётся: в нашем прогоне встретились
    «01-10-2022», «16.07.2023», «2026-09-15» и «2026-08-15T14:20:00» —
    всё это в одном корпусе. Сравнивать и сортировать такое невозможно,
    а Модулю 4 придётся отвечать на вопрос «какой документ свежее».
    Поэтому дату нормализуем на входе, а не при каждом использовании.

    Что не разобралось — возвращаем пустой строкой: пустое поле честнее
    выдуманной даты.
    """
    text = str(value).strip()
    if not text:
        return ""

    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)          # 2026-09-15, с временем или без
    if m:
        return f"{m[1]}-{m[2]}-{m[3]}"

    m = re.match(r"^(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})", text)   # 16.07.2023, 01-10-2022
    if m:
        return f"{m[3]}-{int(m[2]):02d}-{int(m[1]):02d}"

    m = re.match(r"^(\d{1,2})\s+([А-Яа-я]+)\s+(\d{4})", text)       # 10 марта 2024 г.
    if m:
        for name, number in MONTHS.items():
            if m[2].lower().startswith(name):
                return f"{m[3]}-{number:02d}-{int(m[1]):02d}"

    m = re.match(r"^(\d{4})$", text)                                 # просто год
    if m:
        return m[1]

    return ""


def _plain(text: str) -> str:
    """Снимает markdown-разметку, чтобы шапку можно было читать регулярками."""
    return text.replace("*", " ").replace("|", " ").replace("#", " ")


def extract_version(text: str) -> str:
    """Достаёт номер версии из шапки регламента.

    Модель оформляет шапку по-разному: жирным «**Версия:** **3.1**»,
    выравненным текстом «**Версия:**     3.1», строкой markdown-таблицы
    «| Версия документа | 3.1 |». Версия в документе есть всегда, просто
    лежит каждый раз иначе — поэтому сначала снимаем разметку.

    Без этого поля Модуль 4 не сможет отличить действующую редакцию
    от отменённой, а это одна из главных ошибок RAG в проде.
    """
    head = _plain(text[:1500])
    m = re.search(r"Верси\w*(?:\s+документа)?\s*:?\s+(\d+(?:\.\d+)?)", head)
    return m[1] if m else ""


def extract_approval_date(text: str) -> str:
    """Дата утверждения из шапки регламента."""
    head = _plain(text[:1500])
    m = re.search(r"Дата\s+утверждения\s*:?\s+([^\n]+)", head)
    return normalize_date(m[1].strip()) if m else ""


def split_sections(text: str, fallback: str) -> list:
    """Режет документ на записи по заголовкам — раздел становится записью.

    На раздел, а не на страницу: у сгенерированного документа страниц нет,
    зато есть структура. Название раздела попадёт в payload и в ответ
    ассистента — «см. раздел 5.1 регламента РГЛ-НО-014».

    Заголовки первого уровня не трогаем: «#» — это название всего документа.
    Резать строго по «##» нельзя: в одном нашем регламенте таких заголовков
    девять, а в другом всего два, остальная структура на «###».
    """
    def clean_heading(raw: str) -> str:
        """Название раздела без markdown-разметки.

        Модель ставит заголовки как придётся: «## **Шапка документа**»,
        «### 🔧 Оглавление». Разметка и значки попадут в payload и в ответ
        ассистента, поэтому снимаем их здесь, а не при выводе.
        """
        name = re.sub(r"[*_`#]+", " ", raw)
        name = re.sub(r"[^\w\s.,:;()«»/-]", " ", name, flags=re.UNICODE)
        return re.sub(r"\s{2,}", " ", name).strip(" .:-") or raw.strip()

    headings = list(re.finditer(r"(?m)^#{2,6}\s+(.+?)\s*$", text))
    if not headings:
        return [{"page": None, "section": fallback, "text": text.strip()}]

    records = []
    # всё до первого заголовка — шапка документа: номер, версия, кем утверждён
    preamble = text[:headings[0].start()].strip()
    if preamble:
        records.append({"page": None, "section": "Шапка документа", "text": preamble})

    for i, h in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[h.start():end].strip()
        if not body:
            continue
        if records and len(body) < MIN_SECTION_CHARS:
            records[-1]["text"] += "\n\n" + body
        else:
            records.append({"page": None, "section": clean_heading(h[1]),
                            "text": body})

    return records


# ====================================================================
#  ВЫЗОВ МОДЕЛИ
# ====================================================================

def ask_model(prompt: str, max_tokens: int = 2500) -> str:
    """Один вызов GigaChat. Копим usage, чтобы в конце показать цену прогона."""
    with GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
    ) as client:
        response = client.chat({
            "messages": [{"role": "user", "content": prompt}],
            "model": "GigaChat",
            "temperature": 0.7,
            "max_tokens": max_tokens,
        })

    _usage["prompt"] += response.usage.prompt_tokens
    _usage["completion"] += response.usage.completion_tokens
    return response.choices[0].message.content


def extract_json_array(raw: str):
    """Достаёт JSON-массив из ответа модели.

    TODO 5: сейчас функция пробует разобрать ответ целиком — это заглушка.
    Она работает, только пока модель вернула ЧИСТЫЙ JSON. Но модель любит
    обернуть ответ в ```json ... ``` или добавить «Вот результат:» — и тогда
    json.loads падает, хотя массив в ответе есть.

    Что сделать (как extract_json_block в М2, только для массива [...]):
      1. Поискать блок в ограде: re.search(r"```(?:json)?\\s*(\\[.*?\\])\\s*```",
         raw, re.DOTALL) — если нашёлся, взять .group(1).
      2. Иначе взять кусок от первой [ до последней ]:
         re.search(r"\\[.*\\]", raw, re.DOTALL) — .group(0).
      3. Ничего не нашлось — вернуть None.
      4. json.loads(блок); если JSONDecodeError — вернуть None
         (вызывающий код сам перегенерирует батч).
    """
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def ask_for_json(prompt: str, expected: int, what: str, max_tokens: int = 2500):
    """Вызов с повтором: битый JSON в батче — перегенерируем один раз."""
    for attempt in (1, 2):
        data = extract_json_array(ask_model(prompt, max_tokens))
        if isinstance(data, list) and data:
            if len(data) != expected:
                print(f"    внимание: просили {expected} {what}, получили {len(data)}")
            return data
        print(f"    попытка {attempt}: модель вернула не JSON-массив, повторяю...")
    raise RuntimeError(f"дважды не удалось получить JSON ({what}) — прогон остановлен")


def batches(items: list, size: int) -> list:
    return [items[i:i + size] for i in range(0, len(items), size)]


# ====================================================================
#  ЗАПИСЬ В КОРПУС
# ====================================================================

def sha256(path: Path) -> str:
    """Контрольная сумма сырого ответа модели — чтобы было видно,
    из чего собран документ корпуса."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_corpus_file(doc_id: str, doc_type: str, title: str, equipment: list,
                      version: str, date: str, source: Path, records: list):
    """Пишет один документ корпуса в data/corpus/<doc_id>.json."""
    doc = {
        "doc_id": doc_id,
        "type": doc_type,
        "title": title,
        "equipment": equipment,
        "version": version,
        "date": date,
        "source_file": f"generated/{source.name}",
        "source_url": "",
        "retrieved_at": time.strftime("%Y-%m-%d",
                                      time.localtime(source.stat().st_mtime)),
        "sha256": sha256(source),
        "records": records,
    }
    (CORPUS_DIR / f"{doc_id}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_old(prefix: str):
    for f in CORPUS_DIR.glob(f"{prefix}*.json"):
        f.unlink()


# ====================================================================
#  СБОРКА КОРПУСА ИЗ СЫРЫХ ОТВЕТОВ МОДЕЛИ
# ====================================================================

def build_cards(cards: list):
    """Карточка оборудования — короткий документ, режем его на одну запись."""
    source = GENERATED_DIR / "equipment_cards.json"
    clear_old("CARD-")
    names = {it["id"]: it["name"] for it in REGISTRY}

    for card in cards:
        eq_id = card["equipment_id"]
        title = f"Карточка оборудования {eq_id} — {names[eq_id]}"
        issues = "".join(f"- {x}\n" for x in card.get("known_issues", []))
        text = (
            f"# {title}\n\n"
            f"Тип оборудования: {card.get('type', '—')}\n"
            f"Расположение: {card.get('location', '—')}\n"
            f"Дата ввода в эксплуатацию: {card.get('install_date', '—')}\n"
            f"Последнее техническое обслуживание: {card.get('last_maintenance', '—')}\n"
            f"Статус: {card.get('status', '—')}\n\n"
            f"Характеристики: {card.get('spec_summary', '—')}\n\n"
            f"Типовые неисправности:\n{issues}"
        )
        write_corpus_file(
            doc_id=f"CARD-{eq_id}",
            doc_type="карточка оборудования",
            title=title,
            equipment=[eq_id],
            version="",
            date=normalize_date(card.get("last_maintenance", "")),
            source=source,
            records=[{"page": None, "section": title, "text": text.strip()}],
        )


def build_incidents(incidents: list):
    source = GENERATED_DIR / "incidents.json"
    clear_old("INC-")

    for inc in incidents:
        title = f"Инцидент {inc['incident_id']} — {inc['equipment_id']}"
        text = (
            f"# Запись журнала инцидентов {inc['incident_id']}\n\n"
            f"Дата: {inc.get('date', '—')}\n"
            f"Оборудование: {inc['equipment_id']}\n"
            f"Серьёзность: {inc.get('severity', '—')}\n\n"
            f"Описание: {inc.get('description', '—')}\n\n"
            f"Устранение: {inc.get('resolution', '—')}\n"
            f"Устранил: {inc.get('resolved_by', '—')}\n"
        )
        write_corpus_file(
            doc_id=inc["incident_id"],
            doc_type="журнал инцидентов",
            title=title,
            equipment=[inc["equipment_id"]],
            version="",
            date=normalize_date(inc.get("date", "")),
            source=source,
            records=[{"page": None, "section": title, "text": text.strip()}],
        )


def build_reglaments():
    """Регламенты лежат в data/generated как markdown — режем их на разделы."""
    clear_old("REGL-GEN-")
    built = 0
    for i, topic in enumerate(REGLAMENT_TOPICS, 1):
        source = GENERATED_DIR / f"reglament_{i}.md"
        if not source.exists():
            print(f"    нет файла {source.name} — регламент {i} пропущен")
            continue
        text = source.read_text(encoding="utf-8").strip()
        records = split_sections(text, fallback=topic)
        version = extract_version(text)
        date = extract_approval_date(text)

        write_corpus_file(
            doc_id=f"REGL-GEN-{i:02d}",
            doc_type="регламент",
            title=topic,
            equipment=[],
            version=version,
            date=date,
            source=source,
            records=records,
        )
        print(f"    REGL-GEN-{i:02d}: разделов {len(records)}, "
              f"версия {version or '—'}, утверждён {date or '—'}")
        built += 1
    return built


def rebuild():
    """Пересобрать корпус из data/generated, не обращаясь к модели.

    Схема файла корпуса меняется чаще, чем сами документы. Держать ради
    этого повторную генерацию — значит каждый раз платить токенами
    и получать ДРУГОЙ корпус, а вместе с ним другие контрольные вопросы
    и другие цифры в отчётах.
    """
    print("\n" + "=" * 78)
    print("ПЕРЕСБОРКА КОРПУСА ИЗ data/generated (модель не вызывается)")
    print("=" * 78)

    cards_file = GENERATED_DIR / "equipment_cards.json"
    if cards_file.exists():
        cards = json.loads(cards_file.read_text(encoding="utf-8"))["cards"]
        build_cards(cards)
        print(f"    карточек: {len(cards)}")
    else:
        print(f"    нет {cards_file.name} — карточки пропущены")

    inc_file = GENERATED_DIR / "incidents.json"
    if inc_file.exists():
        incidents = json.loads(inc_file.read_text(encoding="utf-8"))["incidents"]
        build_incidents(incidents)
        print(f"    инцидентов: {len(incidents)}")
    else:
        print(f"    нет {inc_file.name} — журнал инцидентов пропущен")

    build_reglaments()


# ====================================================================
#  ЧАСТЬ 1 — КАРТОЧКИ ОБОРУДОВАНИЯ
# ====================================================================

def generate_cards() -> list:
    print("\n" + "=" * 78)
    print("КАРТОЧКИ ОБОРУДОВАНИЯ — по одной на каждую единицу из реестра")
    print("=" * 78)

    cards = []
    for batch in batches(REGISTRY, BATCH_SIZE):
        registry_list = "\n".join(f"- {it['id']} — {it['name']}" for it in batch)
        prompt = (PROMPT_CARDS
                  .replace("{N}", str(len(batch)))
                  .replace("{REGISTRY_LIST}", registry_list))
        print(f"\n  батч: {', '.join(it['id'] for it in batch)}")
        raw_cards = ask_for_json(prompt, len(batch), "карточек")

        for card in raw_cards:
            canon = validate_equipment_id(card.get("equipment_id", ""))
            if canon is None:
                print(f"    ОТБРАКОВАНА карточка «{card.get('equipment_id')}» — "
                      f"такого id нет в реестре (модель выдумала)")
                continue
            card["equipment_id"] = canon  # каноничное написание
            cards.append(card)
            print(f"    OK: {card['equipment_id']} ({card.get('type', '?')})")

    (GENERATED_DIR / "equipment_cards.json").write_text(
        json.dumps({"cards": cards}, ensure_ascii=False, indent=2), encoding="utf-8")

    build_cards(cards)
    print(f"\n  Итого карточек: {len(cards)} из {len(REGISTRY)}")
    return cards


# ====================================================================
#  ЧАСТЬ 2 — ЖУРНАЛ ИНЦИДЕНТОВ
# ====================================================================

def generate_incidents():
    cards_file = GENERATED_DIR / "equipment_cards.json"
    if not cards_file.exists():
        print("Сначала сгенерируй карточки: python generate_corpus.py --part cards")
        sys.exit(1)
    cards = json.loads(cards_file.read_text(encoding="utf-8"))["cards"]
    card_ids = [c["equipment_id"] for c in cards]

    print("\n" + "=" * 78)
    print(f"ЖУРНАЛ ИНЦИДЕНТОВ — {N_INCIDENTS} записей, id из сгенерированных карточек")
    print("=" * 78)

    incidents = []
    start = 1
    for size in [len(b) for b in batches(list(range(N_INCIDENTS)), 4)]:
        prompt = (PROMPT_INCIDENTS
                  .replace("{N}", str(size))
                  .replace("{CARD_IDS}", ", ".join(card_ids))
                  .replace("{START:03d}", f"{start:03d}"))
        print(f"\n  батч: {size} записей, начиная с INC-2026-{start:03d}")
        raw = ask_for_json(prompt, size, "инцидентов")

        for inc in raw:
            canon = validate_equipment_id(inc.get("equipment_id", ""))
            if canon is None:
                print(f"    ОТБРАКОВАНА запись: оборудование "
                      f"«{inc.get('equipment_id')}» не из карточек")
                continue
            inc["equipment_id"] = canon
            inc["incident_id"] = f"INC-2026-{len(incidents) + 1:03d}"  # сквозная нумерация
            incidents.append(inc)
            print(f"    OK: {inc['incident_id']} [{inc['equipment_id']}] "
                  f"{str(inc.get('severity', '?'))}")
        start += size

    (GENERATED_DIR / "incidents.json").write_text(
        json.dumps({"incidents": incidents}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    build_incidents(incidents)
    print(f"\n  Итого записей: {len(incidents)}")


# ====================================================================
#  ЧАСТЬ 3 — ВНУТРЕННИЕ РЕГЛАМЕНТЫ
# ====================================================================

def generate_reglaments():
    print("\n" + "=" * 78)
    print("ВНУТРЕННИЕ РЕГЛАМЕНТЫ — проверяемые факты + раздел исключений")
    print("=" * 78)

    for i, topic in enumerate(REGLAMENT_TOPICS, 1):
        print(f"\n  [{i}/{len(REGLAMENT_TOPICS)}] {topic}")
        prompt = PROMPT_REGLAMENT.replace("{ТЕМА_РЕГЛАМЕНТА}", topic)
        text = polish_reglament(ask_model(prompt, max_tokens=4000).strip(),
                                REGLAMENT_NUMBERS[i - 1])

        if len(text) < 1500:
            print(f"    внимание: регламент подозрительно короткий ({len(text)} симв.)")

        (GENERATED_DIR / f"reglament_{i}.md").write_text(text + "\n", encoding="utf-8")

    build_reglaments()


# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Генерация документов ПромТеха")
    parser.add_argument("--part", choices=["cards", "incidents", "reglaments", "all"],
                        default="all")
    parser.add_argument("--rebuild", action="store_true",
                        help="пересобрать корпус из data/generated без вызова модели")
    args = parser.parse_args()

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild:
        rebuild()
        return

    if not os.getenv("GIGACHAT_CREDENTIALS"):
        print("Нет GIGACHAT_CREDENTIALS в .env — генерировать нечем.")
        sys.exit(1)

    if args.part in ("cards", "all"):
        generate_cards()
    if args.part in ("incidents", "all"):
        generate_incidents()
    if args.part in ("reglaments", "all"):
        generate_reglaments()

    total = _usage["prompt"] + _usage["completion"]
    print("\n" + "=" * 78)
    print(f"Токены за прогон: {_usage['prompt']} промпт + "
          f"{_usage['completion']} ответ = {total}")
    print("=" * 78)


if __name__ == "__main__":
    main()
