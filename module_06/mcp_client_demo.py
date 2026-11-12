"""
Модуль 6, Занятие 11 — MCP-хост: запрос на естественном языке → выбор tool.

Менять в этом файле ничего не нужно. Это и есть «настоящий» смысл MCP:

  ты пишешь обычную фразу («какой статус у PUMP-07?», «создай инцидент…»),
  хост (GigaChat) смотрит на каталог tools с MCP-сервера и САМ выбирает,
  какой инструмент вызвать и с какими аргументами,
  затем хост вызывает выбранный tool по протоколу MCP (stdio),
  а ты видишь весь путь: фраза → выбор модели → вызов → ответ.

Без MCP модели пришлось бы знать эндпоинты и форматы корпоративных систем.
С MCP она видит только tools/resources — безопасный адаптер.

Запуск (из папки module_06, после TODO 1–3 в mcp_server_basic.py):

    python mcp_client_demo.py

Нужен ключ GigaChat в .env (как в модуле 4):
    GIGACHAT_CREDENTIALS=...
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import (
    Chat,
    Function,
    FunctionCall,
    FunctionParameters,
    Messages,
    MessagesRole,
)
from gigachat.models.chat import FunctionParametersProperty
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

BASE_DIR = Path(__file__).parent
SERVER_SCRIPT = BASE_DIR / "mcp_server_basic.py"

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR.parent / "module_04" / ".env")

_SEP = "─" * 88
_SEP2 = "═" * 88

HOST_SYSTEM = """\
Ты — ассистент инженера ПромТеха. У тебя НЕТ прямого доступа к корпоративным
системам. Есть только инструменты (functions), которые приходят от MCP-сервера.

Правила:
1. Если пользователь спрашивает статус / состояние оборудования — вызови
   get_equipment_status с нужным equipment_id.
2. Если просит создать / зарегистрировать инцидент или заявку — вызови
   create_incident и ОБЯЗАТЕЛЬНО передай role.
   По умолчанию role=engineer (пользователь — инженер). Ставь role=viewer
   только если пользователь явно сказал «viewer» / «от имени viewer».
   priority: low/medium/high из запроса; «высокий» → high; если не указан — medium.
3. Не выдумывай статусы и id инцидентов сам: сначала вызови инструмент.
4. Если запрос не про оборудование и не про инциденты — ответь текстом
   без вызова функции.
"""


def _print_json(title: str, obj) -> None:
    print(f"\n  {title}:")
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    for line in text.splitlines():
        print(f"    {line}")


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        print("\nВвод недоступен — завершаю.")
        sys.exit(0)


def _require_credentials() -> str:
    creds = (os.getenv("GIGACHAT_CREDENTIALS") or "").strip().strip("\"'")
    if not creds:
        print(_SEP2)
        print("Нет ключа GigaChat.")
        print("Положите в module_06/.env строку:")
        print("  GIGACHAT_CREDENTIALS=ваш_ключ")
        print("(тот же ключ, что в модуле 4). Без него хост не сможет выбрать tool.")
        print(_SEP2)
        sys.exit(1)
    return creds


def mcp_tools_to_gigachat_functions(mcp_tools) -> list[Function]:
    """Превращает MCP list_tools → формат functions для GigaChat.

    Схема аргументов уже есть у MCP-сервера (input_schema).
    Хост не описывает её руками — только передаёт модели.
    """
    functions: list[Function] = []
    for t in mcp_tools:
        schema = t.input_schema or {"type": "object", "properties": {}}
        props_in = schema.get("properties") or {}
        props_out: dict[str, FunctionParametersProperty] = {}
        for name, prop in props_in.items():
            props_out[name] = FunctionParametersProperty(
                type=prop.get("type", "string"),
                description=prop.get("description") or prop.get("title") or name,
                enum=prop.get("enum"),
            )
        desc = (t.description or "").strip()
        first = desc.splitlines()[0] if desc else t.name
        functions.append(
            Function(
                name=t.name,
                description=first,
                parameters=FunctionParameters(
                    type="object",
                    properties=props_out,
                    required=schema.get("required") or [],
                ),
            )
        )
    return functions


def ask_model_which_tool(user_text: str, functions: list[Function], creds: str) -> Messages:
    """Один шаг хоста: модель либо вызывает function, либо отвечает текстом."""
    with GigaChat(
        credentials=creds,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
    ) as client:
        response = client.chat(
            Chat(
                model=os.getenv("GIGACHAT_MODEL", "GigaChat-2"),
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=HOST_SYSTEM),
                    Messages(role=MessagesRole.USER, content=user_text),
                ],
                functions=functions,
                function_call="auto",
                temperature=0.1,
                max_tokens=400,
            )
        )
    return response.choices[0].message


def ask_model_finalize(
    user_text: str,
    fn_name: str,
    fn_args: dict,
    tool_result: dict,
    creds: str,
) -> str:
    """Второй шаг: модель формулирует ответ человеку по результату tool."""
    with GigaChat(
        credentials=creds,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        verify_ssl_certs=False,
    ) as client:
        response = client.chat(
            Chat(
                model=os.getenv("GIGACHAT_MODEL", "GigaChat-2"),
                messages=[
                    Messages(role=MessagesRole.SYSTEM, content=HOST_SYSTEM),
                    Messages(role=MessagesRole.USER, content=user_text),
                    Messages(
                        role=MessagesRole.ASSISTANT,
                        content="",
                        function_call=FunctionCall(name=fn_name, arguments=fn_args),
                    ),
                    Messages(
                        role=MessagesRole.FUNCTION,
                        name=fn_name,
                        content=json.dumps(tool_result, ensure_ascii=False),
                    ),
                ],
                temperature=0.2,
                max_tokens=500,
            )
        )
    return (response.choices[0].message.content or "").strip()


async def call_mcp_tool(session: ClientSession, name: str, args: dict) -> dict:
    result = await session.call_tool(name, args)
    return json.loads(result.content[0].text)


async def show_catalog(session: ClientSession):
    print(f"\n{'─'*4} 1. КАТАЛОГ MCP (то, что видит хост) {'─'*42}")
    tools = await session.list_tools()
    print(f"\n  Tools ({len(tools.tools)}) — действия, которые модель может выбрать:")
    for t in tools.tools:
        desc = (t.description or "").strip()
        first = desc.splitlines()[0] if desc else "(без описания)"
        print(f"  • {t.name} — {first}")
        print(f"    schema: {json.dumps(t.input_schema, ensure_ascii=False)}")

    resources = await session.list_resources()
    print(f"\n  Resources ({len(resources.resources)}) — данные без вызова tool:")
    for r in resources.resources:
        print(f"  • {r.uri} — {r.name}")

    print(f"\n{_SEP}")
    print("Дальше пишите обычной фразой. Хост (GigaChat) сам выберет tool.")
    print("Примеры:")
    print("  • какой статус у насоса PUMP-07?")
    print("  • создай инцидент по SEP-04: растёт вибрация, приоритет high")
    print("  • узнай статус CONV-09")
    print("  • зарегистрируй аварию на NO-SUCH-ID (увидишь отказ сервера)")
    print("Команды: «помощь», «каталог», «выход».")
    return tools.tools


def print_help() -> None:
    print(f"\n{_SEP}")
    print("ПОДСКАЗКА")
    print(_SEP)
    print("Суть упражнения: вы НЕ выбираете tool руками.")
    print("Вы говорите как инженер → модель выбирает tool → MCP его вызывает.")
    print()
    print("Что попробовать после TODO 1–3:")
    print("  1) статус PUMP-07          → модель вызовет get_equipment_status")
    print("  2) статус CONV-09          → то же, если TODO 1 заполнен")
    print("  3) статус NO-SUCH-ID       → tool вернёт ok:false «не найдено»")
    print("  4) «создай инцидент …»     → create_incident с role=engineer")
    print("  5) инцидент с priority=urgent → сервер отвергнет валидацией")
    print("Смотрите блок «Модель выбрала» — там видно решение хоста.")


async def handle_user_request(
    session: ClientSession,
    user_text: str,
    functions: list[Function],
    creds: str,
) -> None:
    print(f"\n{_SEP2}")
    print(f"ЗАПРОС: {user_text}")
    print(_SEP2)

    print(f"\n{'─'*4} 2. ХОСТ СПРАШИВАЕТ МОДЕЛЬ (какой tool?) {'─'*36}")
    print("GigaChat видит ваш текст + список functions из MCP…")
    message = ask_model_which_tool(user_text, functions, creds)

    fn = message.function_call
    if fn is None or not fn.name:
        print("\n  Модель НЕ вызвала tool — ответила текстом:")
        print(f"  {(message.content or '').strip() or '(пусто)'}")
        print("\n  Если ждали вызов инструмента — переформулируйте запрос")
        print("  явнее: «статус PUMP-07» или «создай инцидент по SEP-04…».")
        return

    args = dict(fn.arguments or {})
    print("\n  Модель выбрала tool (это и есть суть MCP-хоста):")
    print(f"    name : {fn.name}")
    print(f"    args : {json.dumps(args, ensure_ascii=False)}")

    # Политика хоста: инженер за консолью. На сервере у role дефолт viewer
    # (deny-by-default) — если модель забыла role, подставляем engineer сами.
    if fn.name == "create_incident" and not str(args.get("role") or "").strip():
        args["role"] = "engineer"
        print("  Хост дополнил args: role=engineer")
        print("  (на сервере дефолт — viewer; без явной роли запись запрещена)")

    print(f"\n{'─'*4} 3. ВЫЗОВ TOOL ПО ПРОТОКОЛУ MCP {'─'*44}")
    print(f"  call_tool({fn.name}, {json.dumps(args, ensure_ascii=False)})")
    try:
        tool_result = await call_mcp_tool(session, fn.name, args)
    except Exception as exc:
        print(f"\n  Ошибка вызова MCP: {exc}")
        return

    _print_json("Ответ MCP-сервера", tool_result)

    print(f"\n{'─'*4} 4. МОДЕЛЬ ФОРМУЛИРУЕТ ОТВЕТ ИНЖЕНЕРУ {'─'*38}")
    try:
        final = ask_model_finalize(user_text, fn.name, args, tool_result, creds)
        print(f"\n  {final or '(модель вернула пустой текст)'}")
    except Exception as exc:
        print(f"\n  Не удалось получить финальный текст: {exc}")
        print("  Смотрите сырой ответ MCP выше — для проверки TODO его достаточно.")


async def main() -> None:
    creds = _require_credentials()

    print(_SEP2)
    print("MCP-ХОСТ ПромТеха: фраза → выбор tool (GigaChat) → вызов MCP")
    print(_SEP2)
    print("Сервер: mcp_server_basic.py (подпроцесс, транспорт stdio)")
    print("Модель: GigaChat выбирает tool из каталога MCP, не из «своей памяти».")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("\nСессия MCP открыта.")

            mcp_tools = await show_catalog(session)
            functions = mcp_tools_to_gigachat_functions(mcp_tools)
            if not functions:
                print("Сервер не отдал ни одного tool — проверьте mcp_server_basic.py")
                return

            while True:
                print(f"\n{_SEP}")
                user_text = _ask("Ваш запрос (или «помощь» / «выход»): ")
                if not user_text:
                    continue
                low = user_text.lower()
                if low in {"выход", "exit", "quit", "q", "0"}:
                    print("\nСессия закрывается. На Занятии 12 — shift_review_m12.py: смена и audit.")
                    break
                if low in {"помощь", "help", "?"}:
                    print_help()
                    continue
                if low in {"каталог", "tools", "список"}:
                    await show_catalog(session)
                    continue

                await handle_user_request(session, user_text, functions, creds)


if __name__ == "__main__":
    asyncio.run(main())
