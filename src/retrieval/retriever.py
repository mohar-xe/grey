from typing import List, Dict, Any, Optional

from core.config import settings
from core.exceptions import RetrievalError
from processing.embedder import Embedder
from retrieval.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        batch_size: int = None,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.batch_size = batch_size or settings.embedding.batch_size

    def ingest(self, chunks: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        total = 0

        for i in range(0, len(texts), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_texts = texts[i : i + self.batch_size]

            try:
                embeddings = self.embedder.embed_documents(batch_texts)
            except Exception as e:
                raise RetrievalError(
                    f"Embedding failed for batch starting at index {i}: {e}"
                ) from e

            total += self.vector_store.upsert(batch_chunks, embeddings)

        return total

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:

        try:
            query_vector = self.embedder.embed_query(query)
        except Exception as e:
            raise RetrievalError(f"Failed to embed query: {e}") from e

        return self.vector_store.query(
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )
