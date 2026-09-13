import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("qdrant_store")


class QdrantVectorStore:
    """
    In-memory stub implementation of Qdrant vector store to validate structural subtyping
    and protocol compatibility across multiple backends without external service dependencies.
    """

    def __init__(self, collection_name: str = "resumes"):
        self.collection_name = collection_name
        self._storage: Dict[str, Dict[str, Any]] = {}

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        for item_id, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            self._storage[item_id] = {
                "document": doc,
                "embedding": emb,
                "metadata": meta,
            }

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        all_ids = list(self._storage.keys())[:n_results]
        all_docs = [self._storage[i]["document"] for i in all_ids]
        all_metas = [self._storage[i]["metadata"] for i in all_ids]
        all_distances = [0.1] * len(all_ids)
        return {
            "ids": [all_ids],
            "documents": [all_docs],
            "metadatas": [all_metas],
            "distances": [all_distances],
        }

    def get_all(self) -> Dict[str, Any]:
        all_ids = list(self._storage.keys())
        all_docs = [self._storage[i]["document"] for i in all_ids]
        all_metas = [self._storage[i]["metadata"] for i in all_ids]
        return {
            "ids": all_ids,
            "documents": all_docs,
            "metadatas": all_metas,
        }

    def count(self) -> int:
        return len(self._storage)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        if ids:
            for item_id in ids:
                self._storage.pop(item_id, None)
        if where:
            to_remove = []
            for item_id, item in self._storage.items():
                meta = item.get("metadata", {})
                matches = all(meta.get(k) == v for k, v in where.items())
                if matches:
                    to_remove.append(item_id)
            for item_id in to_remove:
                self._storage.pop(item_id, None)
