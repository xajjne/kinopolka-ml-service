"""Internal embedding API. Start with one Uvicorn worker."""

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
import numpy as np
from pydantic import BaseModel, ConfigDict, StringConstraints
from sentence_transformers import SentenceTransformer
import torch


SERVICE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", str(SERVICE_DIR / "models" / "bge-m3")))
    metadata = json.loads((model_dir / "filmets-model.json").read_text(encoding="utf-8"))
    if metadata.get("model_id") != "BAAI/bge-m3" or metadata.get("dimensions") != 1024 or not metadata.get("revision"):
        raise RuntimeError("Некорректные метаданные BGE-M3")
    device = os.environ.get("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(
        str(model_dir), device=device, local_files_only=True, trust_remote_code=False,
        tokenizer_kwargs={"fix_mistral_regex": False},
    )
    if model.get_sentence_embedding_dimension() != 1024:
        raise RuntimeError("Неожиданная размерность модели")
    model.encode(["Проверка готовности"], normalize_embeddings=True, show_progress_bar=False)
    app.state.model = model
    app.state.metadata = metadata
    app.state.inference_lock = Lock()
    yield
    del app.state.model


app = FastAPI(title="Filmets Embeddings", lifespan=lifespan)


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=1000)]


@app.get("/health")
def health(request: Request):
    return {"status": "ready", **request.app.state.metadata, "device": str(request.app.state.model.device)}


@app.post("/embed")
def embed(body: EmbedRequest, request: Request):
    state = request.app.state
    # Ограничиваем параллельное использование GPU, не накапливая очередь запросов.
    if not state.inference_lock.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Модель занята, повторите запрос позже")
    try:
        length = len(state.model.tokenizer.encode(body.text, truncation=False))
        if length > state.model.max_seq_length:
            raise HTTPException(status_code=422, detail="Запрос превышает лимит токенов модели")
        vector = state.model.encode(
            [body.text], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        if vector.shape != (1024,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1.0, atol=0.0001):
            raise RuntimeError("Модель вернула некорректный вектор")
        return {**state.metadata, "embedding": vector.tolist()}
    finally:
        state.inference_lock.release()
