import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

SERVICE_DIR = Path(__file__).resolve().parent
MODEL_DIR = SERVICE_DIR / "models" / "bge-m3"


def main():
    if MODEL_DIR.exists():
        raise RuntimeError(
            f"Папка уже существует: {MODEL_DIR}. "
            "Перед повторной загрузкой проверь её содержимое."
        )

    revision = MODEL_REVISION
    print(f"Версия: {revision}")
    print("Скачиваем модель...")

    model = SentenceTransformer(
        MODEL_ID,
        revision=revision,
        device="cpu",
        trust_remote_code=False,
    )

    if model.get_sentence_embedding_dimension() != 1024:
        raise RuntimeError("Получена неожиданная размерность модели")

    model.save_pretrained(str(MODEL_DIR))

    metadata = {
        "model_id": MODEL_ID,
        "revision": revision,
        "dimensions": 1024,
    }

    metadata_path = MODEL_DIR / "filmets-model.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Модель сохранена: {MODEL_DIR}")


if __name__ == "__main__":
    main()