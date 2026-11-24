from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


SERVICE_DIR = Path(__file__).resolve().parent
MODEL_DIR = SERVICE_DIR / "models" / "bge-m3"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Проверь установку PyTorch.")

    print("Видеокарта:", torch.cuda.get_device_name(0))
    print("Загружаем модель с диска...")

    model = SentenceTransformer(
        str(MODEL_DIR),
        device="cuda",
        local_files_only=True,
        trust_remote_code=False,
        tokenizer_kwargs={"fix_mistral_regex": False},
    )

    text = (
        "Название: Игра Престолов\n"
        "Описание: Знатные дома борются за власть над королевствами, "
        "пока на севере пробуждается древняя угроза.\n"
        "Жанры: драма, НФ и Фэнтези, Боевик и Приключения\n"
        "Ключевые слова: dragon, intrigue, king, kingdom"
    )

    vectors = model.encode(
        [text],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    vector = vectors[0]

    if vector.shape != (1024,):
        raise RuntimeError(f"Неожиданная форма вектора: {vector.shape}")

    if not np.isfinite(vector).all():
        raise RuntimeError("Вектор содержит некорректные числовые значения")

    norm = float(np.linalg.norm(vector))

    if not np.isclose(norm, 1.0, atol=0.0001):
        raise RuntimeError(f"Неожиданная длина вектора: {norm}")

    print("Устройство модели:", model.device)
    print("Форма результата:", vectors.shape)
    print("Размерность вектора:", vector.size)
    print("Тип чисел:", vector.dtype)
    print("Длина вектора:", round(norm, 6))
    print("Первые 8 значений:", vector[:8].tolist())
    print("Проверка завершена успешно.")


if __name__ == "__main__":
    main()