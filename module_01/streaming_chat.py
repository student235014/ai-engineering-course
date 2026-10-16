"""
Модуль 1 — ДОПОЛНИТЕЛЬНО (по желанию): ответ «по словам» (streaming).

Это готовый демонстрационный скрипт. Менять ничего не нужно — просто запусти
и посмотри, как ответ модели печатается постепенно, а не появляется целиком.
Так работают чат-интерфейсы вроде ChatGPT.

    python streaming_chat.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat

load_dotenv(Path(__file__).parent.parent / ".env")

QUESTION = "Назови 3 преимущества MES-системы на производстве."


def main():
    payload = {
        "messages": [{"role": "user", "content": QUESTION}],
        "model": "GigaChat",
    }
    print(f"Вопрос: {QUESTION}\n")
    print("Ответ (печатается по мере поступления):\n")

    with GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
    ) as client:
        for chunk in client.stream(payload):
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
