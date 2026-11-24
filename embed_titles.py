"""Calculate and persist title embeddings. Run from the project virtual environment."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import psycopg
import torch
from dotenv import load_dotenv
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer


SERVICE_DIR = Path(__file__).resolve().parent
TEXT_VERSION = 1
SOURCE_SQL = (SERVICE_DIR / "sql" / "embedding-source.sql").read_text(encoding="utf-8")

SAVE_SQL = """
INSERT INTO title_embeddings (
  title_id, model_name, model_revision, text_version,
  source_text, source_hash, embedding
) VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (title_id) DO UPDATE SET
  model_name = EXCLUDED.model_name,
  model_revision = EXCLUDED.model_revision,
  text_version = EXCLUDED.text_version,
  source_text = EXCLUDED.source_text,
  source_hash = EXCLUDED.source_hash,
  embedding = EXCLUDED.embedding,
  updated_at = now()
"""


def source_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def needs_embedding(row, metadata):
    return (
        row["source_hash"] != source_hash(row["source_text"])
        or row["model_name"] != metadata["model_id"]
        or row["model_revision"] != metadata["revision"]
        or row["text_version"] != TEXT_VERSION
    )


def main():
    parser = argparse.ArgumentParser(description="Векторизация каталога PostgreSQL")
    parser.add_argument("--env-file", type=Path, default=SERVICE_DIR.parent / "backend" / ".env")
    parser.add_argument("--model-dir", type=Path, default=SERVICE_DIR / "models" / "bge-m3")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 64:
        parser.error("batch-size должен быть от 1 до 64")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Проверьте PyTorch или укажите --device cpu")

    load_dotenv(args.env_file, override=False)
    required = ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise RuntimeError("Не заданы параметры: " + ", ".join(missing))

    metadata = json.loads((args.model_dir / "filmets-model.json").read_text(encoding="utf-8"))
    if metadata.get("model_id") != "BAAI/bge-m3" or metadata.get("dimensions") != 1024 or not metadata.get("revision"):
        raise RuntimeError("Некорректные метаданные BGE-M3")

    model = None
    scanned = saved = unchanged = skipped = changed_during_encoding = 0
    with psycopg.connect(
        host=os.environ["POSTGRES_HOST"], port=int(os.environ["POSTGRES_PORT"]),
        dbname=os.environ["POSTGRES_DB"], user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        connect_timeout=10, autocommit=True, row_factory=dict_row,
        application_name="filmets-embed-titles",
    ) as connection:
        # Один процесс обновления эмбеддингов за раз. Блокировка снимается при закрытии соединения.
        lock = connection.execute("SELECT pg_try_advisory_lock(74125, 1) AS acquired").fetchone()
        if not lock["acquired"]:
            raise RuntimeError("Другой процесс уже обновляет эмбеддинги каталога")

        last_id = 0
        while True:
            rows = connection.execute(SOURCE_SQL, (last_id, args.batch_size)).fetchall()
            if not rows:
                break
            last_id = rows[-1]["id"]
            scanned += len(rows)
            pending = []
            for row in rows:
                if not row["has_description"]:
                    skipped += 1
                    print(f"Пропущен ID {row['id']}: отсутствует описание", flush=True)
                elif not needs_embedding(row, metadata):
                    unchanged += 1
                else:
                    pending.append(row)
            if not pending:
                continue

            if model is None:
                print(f"Загружаем BGE-M3, устройство {args.device}...", flush=True)
                model = SentenceTransformer(
                    str(args.model_dir), device=args.device, local_files_only=True,
                    trust_remote_code=False,
                    tokenizer_kwargs={"fix_mistral_regex": False},
                )

            texts = [row["source_text"] for row in pending]
            lengths = model.tokenizer(texts, truncation=False, padding=False, return_length=True)["length"]
            if any(length > model.max_seq_length for length in lengths):
                raise RuntimeError("Текст превышает лимит модели. Нужна явная стратегия сокращения или разбиения текста.")

            vectors = model.encode(
                texts, batch_size=args.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
            if vectors.shape != (len(pending), 1024) or not np.isfinite(vectors).all():
                raise RuntimeError("Модель вернула некорректные векторы")
            if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=0.0001):
                raise RuntimeError("Векторы не нормализованы")

            for row, vector in zip(pending, vectors, strict=True):
                with connection.transaction():
                    # Не держим транзакцию открытой во время вычислений на GPU.
                    exists = connection.execute("SELECT id FROM titles WHERE id = %s FOR UPDATE", (row["id"],)).fetchone()
                    current = connection.execute(SOURCE_SQL, (row["id"] - 1, 1)).fetchone() if exists else None
                    if not current or current["id"] != row["id"] or current["source_text"] != row["source_text"]:
                        changed_during_encoding += 1
                        continue
                    connection.execute(SAVE_SQL, (
                        row["id"], metadata["model_id"], metadata["revision"], TEXT_VERSION,
                        row["source_text"], source_hash(row["source_text"]),
                        json.dumps(vector.tolist(), allow_nan=False),
                    ))
                saved += 1
                print(f"Сохранён ID {row['id']}: {row['title']} (1024 чисел)", flush=True)

    print(f"Проверено: {scanned}; сохранено: {saved}; актуальных: {unchanged}; без описания: {skipped}; изменились во время расчёта: {changed_during_encoding}")


if __name__ == "__main__":
    main()
