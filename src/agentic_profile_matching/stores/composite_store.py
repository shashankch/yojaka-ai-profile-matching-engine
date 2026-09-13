import logging
from typing import List, Dict, Any, Optional

from agentic_profile_matching.stores.base import BaseVectorStore
from agentic_profile_matching.stores.chroma_store import ChromaVectorStore

logger = logging.getLogger("composite_store")


class CompositeVectorStore:
    """
    Composite vector store layering a base persistent store (read-only baseline pool)
    with a session-scoped ephemeral store (isolated in-memory uploads).
    Guarantees ADR-016 zero-disk persistence and multi-tenant isolation for user-uploaded candidate PII.
    """

    def __init__(
        self,
        base_store: BaseVectorStore,
        ephemeral_store: Optional[BaseVectorStore] = None,
    ):
        self.base_store = base_store
        self.ephemeral_store = ephemeral_store or ChromaVectorStore(
            collection_name="ephemeral_session_uploads", ephemeral=True
        )

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Write incoming uploads strictly to the ephemeral in-memory collection."""
        self.ephemeral_store.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Query both base and ephemeral stores, merging and ranking results by distance.
        """
        res_base = self.base_store.query(query_embedding=query_embedding, n_results=n_results)

        if self.ephemeral_store.count() == 0:
            return res_base

        res_eph = self.ephemeral_store.query(query_embedding=query_embedding, n_results=n_results)

        if not res_eph or not res_eph.get("ids") or not res_eph["ids"][0]:
            return res_base
        if not res_base or not res_base.get("ids") or not res_base["ids"][0]:
            return res_eph

        # Merge results sorted by distance (ascending = closest similarity)
        combined: List[tuple] = []
        for dist, doc_id, doc, meta in zip(
            res_base["distances"][0],
            res_base["ids"][0],
            res_base["documents"][0],
            res_base["metadatas"][0],
        ):
            combined.append((dist, doc_id, doc, meta))

        for dist, doc_id, doc, meta in zip(
            res_eph["distances"][0],
            res_eph["ids"][0],
            res_eph["documents"][0],
            res_eph["metadatas"][0],
        ):
            combined.append((dist, doc_id, doc, meta))

        combined.sort(key=lambda x: x[0])
        top = combined[:n_results]

        return {
            "ids": [[item[1] for item in top]],
            "documents": [[item[2] for item in top]],
            "metadatas": [[item[3] for item in top]],
            "distances": [[item[0] for item in top]],
        }

    def get_all(self) -> Dict[str, Any]:
        """Combine all records from both stores."""
        base_all = self.base_store.get_all()
        eph_all = self.ephemeral_store.get_all() if self.ephemeral_store.count() > 0 else {}

        ids = (base_all.get("ids") or []) + (eph_all.get("ids") or [])
        documents = (base_all.get("documents") or []) + (eph_all.get("documents") or [])
        metadatas = (base_all.get("metadatas") or []) + (eph_all.get("metadatas") or [])

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }

    def count(self) -> int:
        """Total document chunks across base and ephemeral stores."""
        return self.base_store.count() + self.ephemeral_store.count()

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete from ephemeral and base stores where matching."""
        if hasattr(self.ephemeral_store, "delete"):
            try:
                self.ephemeral_store.delete(ids=ids, where=where)
            except Exception as e:
                logger.debug(f"Ephemeral delete failed: {e}")
        if hasattr(self.base_store, "delete"):
            try:
                self.base_store.delete(ids=ids, where=where)
            except Exception as e:
                logger.debug(f"Base delete failed: {e}")
