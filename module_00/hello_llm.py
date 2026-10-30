"""
Модуль 0: Первый вызов LLM
============================
Задача: сравнить два провайдера — GigaChat и HuggingFace.

Твоя работа — раскомментировать нужные строки (убрать # в начале)
и заполнить переменные там где указано TODO.
"""

import os
import time
from pathlib import Path

# ШАГ 1: Раскомментируй три строки ниже (убери # в начале каждой)
from dotenv import load_dotenv
from gigachat import GigaChat
from openai import OpenAI


# ШАГ 2: Раскомментируй строку ниже — она загружает ключи из .env
load_dotenv(Path(__file__).parent.parent / ".env")


# ШАГ 3: Укажи вопрос, который отправим в оба провайдера
QUESTION = "Что такое продуктовые метрики? Ответь в 2-3 предложениях."


# ---------------------------------------------------------------
# Дальше код уже написан — просто запусти и посмотри результат
# ---------------------------------------------------------------

def call_gigachat(question: str) -> dict:
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise ValueError("GIGACHAT_CREDENTIALS не найден в .env")

    start = time.time()
    with GigaChat(credentials=credentials, verify_ssl_certs=False) as client:
        response = client.chat(question)
    latency = time.time() - start

    return {
        "answer": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "latency": latency,
    }


def call_huggingface(question: str) -> dict:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise ValueError("HF_API_KEY не найден в .env")

    model = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")

    start = time.time()
    client = OpenAI(api_key=api_key, base_url="https://router.huggingface.co/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}]
    )
    latency = time.time() - start

    return {
        "answer": response.choices[0].message.content,
        "tokens": response.usage.total_tokens,
        "latency": latency,
    }


def print_result(provider_name: str, result: dict):
    print(f"=== {provider_name} ===")
    print(f"Ответ: {result['answer']}")
    print(f"Токены: {result['tokens']}")
    print(f"Latency: {result['latency']:.2f} сек.")
    print()


def main():
    print("=" * 60)
    print("СРАВНЕНИЕ ПРОВАЙДЕРОВ LLM")
    print("=" * 60)
    print()

    gigachat_result = None
    hf_result = None

    try:
        gigachat_result = call_gigachat(QUESTION)
        print_result("GigaChat", gigachat_result)
    except Exception as e:
        print(f"GigaChat ошибка: {e}\n")

    try:
        hf_result = call_huggingface(QUESTION)
        model_name = os.getenv("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        print_result(f"HuggingFace ({model_name})", hf_result)
    except Exception as e:
        print(f"HuggingFace ошибка: {e}\n")

    if gigachat_result and hf_result:
        print("=" * 60)
        print(
            f"Итог: GigaChat ответил за {gigachat_result['latency']:.2f} сек., "
            f"HuggingFace за {hf_result['latency']:.2f} сек."
        )
        print("=" * 60)


if __name__ == "__main__":
    main()
