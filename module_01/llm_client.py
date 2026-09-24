"""
Модуль 1 — параметры LLM-запроса, подсчёт токенов и резервный провайдер.

Что делает этот файл:
  call_gigachat(...)   спрашивает GigaChat  — основной провайдер
  call_huggingface(...) спрашивает HuggingFace — резервный провайдер
  ask_llm(...)         решает, кого спросить, и меряет время ответа

Занятие 2: меняешь только блок "НАСТРОЙКИ СТУДЕНТА" (TODO 1-4) и смотришь,
           как параметры влияют на ответ, на токены и на повторяемость.
Занятие 3: дописываешь тело функции ask_llm (TODO 5) — резервный провайдер.

Запуск:
    python llm_client.py
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat
from openai import OpenAI

import token_tracker

# Ключи берутся из .env в корне проекта (создан в Занятии 1)
load_dotenv(Path(__file__).parent.parent / ".env")


# ====================================================================
#  НАСТРОЙКИ СТУДЕНТА — меняй только этот блок
# ====================================================================

# TODO 1: впиши свой вопрос на русском языке.
# Возьми рабочий вопрос, а не "привет". Например:
#   "Перечисли 3 причины остановки конвейера и что проверить в каждом случае."
MY_QUESTION = "Назови 2 редфлага в портфолио продуктового дизайнера."

# TODO 2: температура — насколько свободно отвечает модель.
#   0.0 = строго и предсказуемо (один и тот же ответ на один и тот же вопрос)
#   1.0 = разнообразно и "творчески" (ответы будут отличаться)
# В практике Занятия 2 ты запустишь файл дважды с 0.0 и дважды с 1.0 и сравнишь.
TEMPERATURE = 0.0

# TODO 3: max_tokens — потолок длины ответа.
#   Если ответ обрывается на полуслове, в выводе будет finish_reason = length.
#   Попробуй 50 (ответ обрежется) и 500 (ответ поместится целиком).
MAX_TOKENS = 500

# TODO 4: system prompt — "роль" модели. Задаёт тон и рамки ответа.
#   Попробуй минимум два варианта, например:
#     "Ты - строгий технический специалист, отвечаешь кратко и по делу."
#     "Ты - терпеливый наставник, объясняешь как новичку, с примерами."
SYSTEM_PROMPT = "Ты - терпеливый наставник, объясняешь как новичку, с примерами."


# ====================================================================
#  КОД НИЖЕ УЖЕ РАБОТАЕТ — менять не нужно, но прочитай его
# ====================================================================

def call_gigachat(question: str) -> dict:
    """Спрашивает GigaChat. Возвращает ответ, токены и причину остановки."""
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise ValueError("GIGACHAT_CREDENTIALS не найден в .env")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    with GigaChat(
        credentials=credentials,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
    ) as client:
        response = client.chat({
            "messages": messages,
            "model": "GigaChat",
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        })

    usage = response.usage
    token_tracker.record("GigaChat", usage.prompt_tokens, usage.completion_tokens)

    return {
        "answer": response.choices[0].message.content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "finish_reason": response.choices[0].finish_reason,
    }


def call_huggingface(question: str) -> dict:
    """Спрашивает HuggingFace — резервный провайдер. Тот же формат ответа."""
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise ValueError("HF_API_KEY не найден в .env")

    client = OpenAI(api_key=api_key, base_url="https://router.huggingface.co/v1")
    model = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

    response = client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    usage = response.usage
    token_tracker.record("HuggingFace", usage.prompt_tokens, usage.completion_tokens)

    return {
        "answer": response.choices[0].message.content,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "finish_reason": response.choices[0].finish_reason,
    }


def ask_llm(question: str) -> tuple:
    """Задаёт вопрос модели. Возвращает (результат, имя_провайдера, latency).

    TODO 5 (Занятие 3): сейчас резерва нет — если GigaChat недоступен,
    скрипт просто падает. Твоя задача — сделать так, чтобы при ЛЮБОЙ ошибке
    GigaChat вопрос уходил в HuggingFace, а пользователь всё равно получал ответ.

    Что нужно сделать:
      1. Оберни вызов call_gigachat в try / except Exception as e.
      2. В блоке except: напечатай предупреждение с текстом ошибки,
         вызови call_huggingface(question) и поставь provider = "HuggingFace".
      3. Строку с latency и return не трогай — она уже написана.
    """
    start = time.time()

    # --- заменить этот блок на try/except (см. пункты 1-3 выше) ---
    result = call_gigachat(question)
    provider = "GigaChat"
    # --- конец блока ---

    latency = time.time() - start
    return result, provider, latency


def main():
    if not MY_QUESTION:
        print("Сначала впиши свой вопрос в MY_QUESTION (TODO 1) и сохрани файл.")
        return

    print("=" * 60)
    print("ВЫЗОВ LLM")
    print("=" * 60)
    print(f"Вопрос      : {MY_QUESTION}")
    print(f"temperature : {TEMPERATURE}")
    print(f"max_tokens  : {MAX_TOKENS}")
    print(f"роль        : {SYSTEM_PROMPT}")
    print()

    try:
        result, provider, latency = ask_llm(MY_QUESTION)
    except Exception as e:
        print(f"Оба провайдера недоступны: {e}")
        return

    print(f"[{provider}] ответил за {latency:.2f} сек.")
    print()
    print(f"Ответ: {result['answer']}")
    print()
    print(f"Токены: {result['total_tokens']} "
          f"(вход: {result['prompt_tokens']}, выход: {result['completion_tokens']})")
    print(f"finish_reason: {result['finish_reason']}")

    if result["finish_reason"] == "length":
        print("  ^ ответ обрезан по лимиту max_tokens — увеличь MAX_TOKENS")


if __name__ == "__main__":
    main()
