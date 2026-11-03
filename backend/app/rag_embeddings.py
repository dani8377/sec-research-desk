"""Local LangChain/Ollama embedding boundary with explicit vector validation."""

import math

from langchain_ollama import OllamaEmbeddings

from app.config import Settings


class RAGError(Exception):
    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code


class LocalEmbeddings:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.batch_size = settings.embedding_batch_size
        self.client = OllamaEmbeddings(
            model=self.model,
            base_url=settings.ollama_base_url,
            client_kwargs={"timeout": settings.embedding_timeout_seconds},
        )

    def documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = [
                self._prefix("search_document", text)
                for text in texts[start : start + self.batch_size]
            ]
            vectors.extend(self._embed(batch))
        return vectors

    def query(self, text: str) -> list[float]:
        return self._embed([self._prefix("search_query", text)])[0]

    def _prefix(self, task: str, text: str) -> str:
        # Nomic requires asymmetric retrieval prefixes; other models vary.
        if self.model.split(":")[0] == "nomic-embed-text":
            return f"{task}: {text}"
        return text

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self.client.embed_documents(texts)
        except Exception as exc:
            raise RAGError(
                "Local embedding request failed; check Ollama and the configured model"
            ) from exc
        if len(vectors) != len(texts) or any(
            len(vector) != self.dimensions
            or not all(math.isfinite(value) for value in vector)
            or not any(value != 0 for value in vector)
            for vector in vectors
        ):
            raise RAGError(
                "Ollama returned invalid embeddings or unexpected dimensions", 502
            )
        return vectors
