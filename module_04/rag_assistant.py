"""
Модуль 4, Занятие 7 — RAG-ассистент: ответ по источникам.

ТВОЯ РАБОТА: TODO 1 (format_context — оформить найденные куски с указанием
источника), TODO 2 (build_prompt — промпт с инструкцией отвечать только
по контексту и обязательно цитировать источник), TODO 3 (confidence_gate —
отказ «в базе нет данных», если лучший score ниже порога).

Вопросы вне базы в контрольном наборе специально отвлечённые
(кулинария, кино, география): если спросить «давление в шинах» или
«зарплату мастера», retriever находит похожие технические куски, и модель
начинает отвечать из общих знаний. Отвлечённый вопрос даёт низкий score —
это первая линия обороны (confidence_gate). Вторая — промпт: даже если
кусок как-то попал в контекст, модель должна ответить только по нему
или сказать «не знаю».

Retrieval здесь не пишем заново — импортируем search() из vector_store.py,
уже написанного и настроенного в Занятии 6. Если он у тебя не готов —
сначала закончи Занятие 6.

Генерация — GigaChat, тот же способ вызова, что в generate_corpus.py
(Занятие 5): ключ в .env, client.chat({"messages": [...]}).

Запуск (из папки module_04, индекс должен быть собран — python ingestion.py):
    python rag_assistant.py            # выбрать один вопрос и увидеть весь путь ответа
    python rag_assistant.py --case     # разбор кейса инженерной документации
    python rag_assistant.py --embedded # без Docker
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat
from sentence_transformers import SentenceTransformer

from qdrant_connect import DASHBOARD_URL, make_client
from vector_store import MODEL_NAME, load_manifest, search, where

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "qdrant_chunks"

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

QUESTIONS = json.loads(
    (BASE_DIR / "data" / "questions.json").read_text(encoding="utf-8")
)["questions"]

# ====================================================================
#  НАСТРОЙКИ СТУДЕНТА
# ====================================================================

TOP_K = 3

# Порог берём как первую (не единственную) линию обороны — то самое
# значение, которое ты подобрал в Занятии 6.
# 0.5 — стартовое значение, если своего ещё нет.
MIN_SCORE = 0.5


# ====================================================================
#  ФОРМИРОВАНИЕ ОТВЕТА
# ====================================================================

def format_context(chunks: list) -> str:
    """Оформляет найденные чанки с указанием источника.

    Каждый чанк идёт с номером, doc_id, местом в документе
    (стр. N или раздел) и полным текстом.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        payload = chunk.payload
        place = where(payload)               # из vector_store.py
        header = f"[{i}] Источник: {payload['doc_id']}, {place}"
        parts.append(f"{header}\n{payload['text']}")
    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> list:
    """Формирует сообщения для GigaChat: system + user."""
    system = (
        "Ты — ассистент инженера промышленного предприятия. "
        "Отвечай ТОЛЬКО на основании текста в разделе КОНТЕКСТ. "
        "Не используй собственные знания о промышленном оборудовании.\n\n"
        "Для каждого факта, который приводишь в ответе, указывай источник "
        "в квадратных скобках в формате [doc_id, место] — например "
        "[REGL-GEN-01, раздел 5.1] или [EXT-GOST-32601, стр. 24].\n\n"
        "Если в предоставленном контексте нет ответа на вопрос — ответь "
        "ровно этой фразой, без каких-либо дополнений:\n"
        "«В базе нет данных по этому вопросу.»"
    )
    user = f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def confidence_gate(chunks: list) -> bool:
    """True, если есть смысл идти в GigaChat."""
    if not chunks:
        return False
    best = chunks[0].score
    return best >= MIN_SCORE


REFUSAL_TEXT = "В базе нет данных по этому вопросу."


def ask_model(messages: list, max_tokens: int = 700) -> str:
    """Один вызов GigaChat с готовыми сообщениями. Менять не нужно."""
    creds = (os.getenv("GIGACHAT_CREDENTIALS") or "").strip().strip("\"'")
    with GigaChat(
        credentials=creds,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
    ) as client:
        response = client.chat({
            "messages": messages,
            "model": "GigaChat",
            "temperature": 0.2,
            "max_tokens": max_tokens,
        })
    return response.choices[0].message.content.strip()


def answer(client, model, question: str, alias: str, doc_type: str = None) -> dict:
    """Полный цикл: поиск -> проверка уверенности -> промпт -> ответ модели."""
    hits = search(client, model, question, alias, doc_type=doc_type, limit=TOP_K)

    if not confidence_gate(hits):
        return {
            "answer": REFUSAL_TEXT,
            "sources": [],
            "best_score": hits[0].score if hits else 0.0,
            "refused_by": "score",
        }

    context = format_context(hits)
    messages = build_prompt(question, context)
    raw = ask_model(messages)

    return {
        "answer": raw,
        "sources": [h.payload["doc_id"] for h in hits],
        "best_score": hits[0].score,
        "refused_by": None,
    }


# ====================================================================
#  КЕЙС: ассистент по инженерной документации
# ====================================================================

def demo_case(client, model, alias: str):
    """Один вопрос — два ответа: модель без контекста и RAG-ассистент.

    Вопрос про срок устранения критичных неисправностей насосного
    оборудования: ответ есть только во внутреннем регламенте REGL-GEN-01
    (раздел 5.1) — документе, которого не существовало до Занятия 5
    и который модель никак не могла видеть при обучении. Если RAG работает
    правильно, «догадка из памяти» и «ответ по регламенту» разойдутся —
    и только второй будет с точной ссылкой на источник.
    """
    question = next(q["question"] for q in QUESTIONS if q["id"] == "q1")

    print("=" * 92)
    print("КЕЙС: ассистент по инженерной документации")
    print("=" * 92)
    print(f"\nВопрос: {question}\n")

    print("--- Ответ БЕЗ RAG (модель отвечает по памяти, без контекста) ---")
    naive = ask_model([{"role": "user", "content": question}])
    print(naive)

    print("\n--- Ответ С RAG (по действующему регламенту, с источником) ---")
    result = answer(client, model, question, alias)
    print(result["answer"])
    print(f"\nИсточники: {', '.join(result['sources']) or '—'}")
    print(
        "\nСравни два ответа: наивный, скорее всего, звучит уверенно и "
        "правдоподобно, но срок в нём — вероятностная догадка модели, а не "
        "факт из регламента. RAG-ответ обязан назвать тот же срок, что "
        "написан в REGL-GEN-01, раздел 5.1, и явно на него сослаться."
    )


# ====================================================================
#  РАЗБОР ОДНОГО ВОПРОСА
# ====================================================================

_SEP  = "─" * 88
_SEP2 = "═" * 88
_SEARCH_METHOD = (
    "dense retrieval · модель: paraphrase-multilingual-MiniLM-L12-v2 · "
    "метрика: косинусное сходство эмбеддингов"
)


def _fmt_chunk(n: int, h) -> str:
    """Печатает один найденный чанк с оценкой и текстом."""
    p = h.payload
    loc = f"раздел «{p['section']}»" if p.get("section") else f"стр. {p.get('page', '?')}"
    lines = [
        f"  ┌─ чанк #{n}  score={h.score:.4f}",
        f"  │  документ : {p['doc_id']}  ({p['type']})",
        f"  │  место    : {loc}",
        f"  │  chunk_no : {p.get('chunk_no', '?')}",
        f"  └─ текст (первые 300 симв.):",
    ]
    for line in p.get("text", "")[:300].splitlines():
        lines.append(f"     {line}")
    return "\n".join(lines)


def _fmt_context(context: str) -> str:
    """Выводит собранный контекст для промпта с рамкой."""
    if not context:
        return "  (пустая строка — TODO 1 ещё не реализован)"
    lines = ["  ┌─ контекст для промпта ─────────────────────────────────────"]
    for line in context.splitlines():
        lines.append(f"  │ {line}")
    lines.append("  └────────────────────────────────────────────────────────────")
    return "\n".join(lines)


def _fmt_messages(messages: list) -> str:
    """Выводит сообщения, которые уйдут в GigaChat."""
    lines = []
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")
        lines.append(f"  ┌─ [{role}] ──────────────────────────────────────────────────")
        for line in content.splitlines():
            lines.append(f"  │ {line}")
        lines.append("  └──────────────────────────────────────────────────────────────")
    return "\n".join(lines)


def print_question_menu():
    print(_SEP2)
    print("RAG-АССИСТЕНТ. Выберите ОДИН вопрос — будет показан весь путь ответа")
    print("(поиск → порог → контекст → промпт → GigaChat).")
    print(_SEP2)
    print(f"{'id':<6} {'где':<10} вопрос")
    print(_SEP)
    for q in QUESTIONS:
        tag = "из базы" if q["in_corpus"] else "вне базы"
        print(f"{q['id']:<6} {tag:<10} {q['question']}")
    print(_SEP)


def resolve_question_id(raw: str):
    """Принимает '3', 'q3', 'Q3'. Возвращает карточку вопроса или None."""
    raw = (raw or "").strip().lower()
    if not raw:
        return None
    if raw.isdigit():
        raw = f"q{raw}"
    if not raw.startswith("q"):
        raw = "q" + raw
    by_id = {q["id"].lower(): q for q in QUESTIONS}
    return by_id.get(raw)


def ask_question_id():
    print_question_menu()
    try:
        raw = input("Номер вопроса (например 3 или q3): ").strip()
    except EOFError:
        print("Номер не введён.")
        sys.exit(1)
    q = resolve_question_id(raw)
    if q is None:
        print(f"Нет вопроса «{raw}». Запустите снова и введите id из списка.")
        sys.exit(1)
    return q


def trace_question(client, model, alias: str, q: dict):
    """Полный путь одного вопроса: чанки, gate, контекст, промпт, ответ."""
    tag = "ИЗ БАЗЫ " if q["in_corpus"] else "ВНЕ БАЗЫ"

    print()
    print(_SEP2)
    print(f"[{tag}]  {q['id']}: {q['question']}")
    print(f"Метод поиска: {_SEARCH_METHOD}")
    print(_SEP2)

    print(f"\n{'─'*4} 1. RETRIEVAL {'─'*64}")
    print("Запрос → эмбеддинг → ближайшие векторы в Qdrant (cosine).")
    hits = search(client, model, q["question"], alias, limit=TOP_K)
    print(f"Найдено чанков: {len(hits)}  (TOP_K={TOP_K})")
    for i, h in enumerate(hits, 1):
        print()
        print(_fmt_chunk(i, h))

    print(f"\n{'─'*4} 2. CONFIDENCE GATE {'─'*57}")
    best_score = hits[0].score if hits else 0.0
    gate_pass = confidence_gate(hits)
    if gate_pass:
        verdict = "ПРОЙДЕН → вызываем GigaChat"
    else:
        verdict = (f"ОТКАЗ → лучший score {best_score:.4f} < MIN_SCORE {MIN_SCORE}")
    print(f"best_score={best_score:.4f}  MIN_SCORE={MIN_SCORE}  → {verdict}")

    if not gate_pass:
        print(f"\nОтвет: «{REFUSAL_TEXT}»")
        print(_SEP)
        return

    print(f"\n{'─'*4} 3. FORMAT CONTEXT {'─'*58}")
    context = format_context(hits)
    print(_fmt_context(context))

    print(f"\n{'─'*4} 4. BUILD PROMPT (сообщения → GigaChat) {'─'*36}")
    messages = build_prompt(q["question"], context)
    print(_fmt_messages(messages))

    print(f"\n{'─'*4} 5. ОТВЕТ GigaChat {'─'*58}")
    raw = ask_model(messages)
    print(f"\n  {raw}\n")

    print(_SEP)
    sources = [h.payload["doc_id"] for h in hits]
    print(f"Источники: {', '.join(sources)}")
    mc = q.get("must_contain", "")
    if mc:
        hit_mc = mc.lower() in raw.lower()
        mark = "ДА" if hit_mc else "НЕТ  ← generation_error"
        print(f"must_contain={mc!r}  в ответе: {mark}")
    elif not q["in_corpus"]:
        refused = raw.strip() == REFUSAL_TEXT
        print("Вопрос вне базы: ожидается отказ, а не ответ из общих знаний.")
        print("  отказ: " + ("да" if refused else "нет — модель ответила поверх корпуса"))
    print(_SEP)


def main():
    parser = argparse.ArgumentParser(description="RAG-ассистент с ответом по источникам")
    parser.add_argument("--embedded", action="store_true",
                        help="резервный режим без Docker")
    parser.add_argument("--case", action="store_true",
                        help="только кейс инженерной документации")
    args = parser.parse_args()

    manifest = load_manifest()
    client = make_client(args.embedded, DB_PATH)
    print("Загружаю модель...")
    model = SentenceTransformer(MODEL_NAME)
    alias = manifest["alias"]

    try:
        if args.case:
            demo_case(client, model, alias)
        else:
            q = ask_question_id()
            trace_question(client, model, alias, q)
    finally:
        client.close()


if __name__ == "__main__":
    main()


def where(payload: dict) -> str:
    """Ссылка на источник: страница спарсенного PDF или раздел документа."""
    if payload.get("page") is not None:
        return f"стр. {payload['page']}"
    return payload.get("section") or "—"