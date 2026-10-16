"""
Подключение к Qdrant. Менять не нужно.

Основной режим — Qdrant в Docker: полноценный сервер с веб-интерфейсом,
в котором видно коллекции, точки и метаданные (так это выглядит в проде).

    docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.18.3
    Веб-интерфейс: http://localhost:6333/dashboard

Резервный режим — embedded: если Docker поставить не удалось, Qdrant умеет
работать прямо из python-процесса, складывая базу в локальную папку.
Всё то же самое, только без сервера и веб-интерфейса:

    python vector_store.py --embedded
"""


import sys
from pathlib import Path

from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
DASHBOARD_URL = QDRANT_URL + "/dashboard"

DOCKER_HINT = f"""
Qdrant не отвечает на {QDRANT_URL}.

Запусти его в Docker (один раз, дальше он останется в фоне):

    docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.18.3

Если контейнер уже создавался, но остановлен:

    docker start qdrant

Проверка: открой веб-интерфейс {DASHBOARD_URL}

Если Docker поставить не удалось — запусти скрипт в резервном режиме:

    python {{script}} --embedded
"""


def make_client(embedded: bool, db_path: Path) -> QdrantClient:
    """Возвращает клиент Qdrant: Docker-сервер или embedded-резерв."""
    if embedded:
        # Папку НЕ чистим: индекс пересобирается в новую коллекцию, и старая
        # остаётся на случай отката — так же, как на сервере.
        db_path.mkdir(parents=True, exist_ok=True)
        print(f"Режим embedded: база в папке {db_path.name}/ (без Docker и UI)")
        return QdrantClient(path=str(db_path))

    client = QdrantClient(url=QDRANT_URL)
    try:
        client.get_collections()
    except Exception:
        print(DOCKER_HINT.format(script=Path(sys.argv[0]).name))
        sys.exit(1)
    print(f"Qdrant в Docker: {QDRANT_URL}  (веб-интерфейс: {DASHBOARD_URL})")
    return client
