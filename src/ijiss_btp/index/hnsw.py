"""Protected-Domain Vector Indexing via Faiss HNSW and IVF-PQ."""
from __future__ import annotations

import time
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class ProtectedVectorIndex:
    """Wrapper for high-throughput ANN search over cancelable templates."""

    def __init__(
        self,
        dim: int,
        index_type: str = "hnsw",
        m: int = 32,
        ef_construction: int = 128,
        ef_search: int = 64,
        is_binary: bool = False,
    ) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu or faiss-gpu is required for ProtectedVectorIndex")

        self.dim = dim
        self.is_binary = is_binary
        self.index_type = index_type

        if is_binary:
            # Binary HNSW for BioHashing and PS-IoM bitsets
            self.index = faiss.IndexBinaryHNSW(dim, m)
            self.index.hnsw.efSearch = ef_search
            self.index.hnsw.efConstruction = ef_construction
        else:
            # Float HNSW with Inner Product (Cosine on L2-normalized templates)
            self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = ef_search
            self.index.hnsw.efConstruction = ef_construction

    def add(self, templates: np.ndarray) -> float:
        """Adds protected templates to the index; returns build time in seconds."""
        t0 = time.perf_counter()
        if self.is_binary:
            self.index.add(templates)
        else:
            self.index.add(np.ascontiguousarray(templates, dtype=np.float32))
        return time.perf_counter() - t0

    def search(
        self, query_templates: np.ndarray, top_k: int = 10
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Performs 1:N ANN search; returns (distances/similarities, indices, latency_ms)."""
        t0 = time.perf_counter()
        if self.is_binary:
            distances, indices = self.index.search(query_templates, top_k)
        else:
            distances, indices = self.index.search(
                np.ascontiguousarray(query_templates, dtype=np.float32), top_k
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return distances, indices, elapsed_ms

