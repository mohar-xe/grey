import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)

from core.config import settings
from core.exceptions import RetrievalError


class VectorStore:
    def __init__(
        self,
        collection_name: str = None,
        url: str = None,
        api_key: str = None,
        embedding_dim: int = None,
    ):
        self.collection_name = collection_name or settings.qdrant.collection_name
        self.embedding_dim = embedding_dim or settings.embedding.dimension

        url = url or settings.qdrant.url
        api_key = api_key or settings.qdrant.api_key or None

        self.client = QdrantClient(url=url, api_key=api_key)

    def create_collection(self, recreate: bool = False) -> None:
        """Create the Qdrant collection. No-ops if it already exists unless recreate=True."""
        existing = {c.name for c in self.client.get_collections().collections}

        if self.collection_name in existing:
            if not recreate:
                return
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    def collection_info(self) -> Dict[str, Any]:
        """Return basic stats about the collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }

    def upsert(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:

        if len(chunks) != len(embeddings):
            raise RetrievalError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": chunk["text"], **chunk.get("metadata", {})},
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def delete(self, ids: List[str]) -> None:
        """Delete points by their UUIDs."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:

        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                ]
            )

        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=score_threshold,
            )
        except Exception as e:
            raise RetrievalError(f"Qdrant search failed: {e}") from e

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            }
            for hit in hits
        ]
