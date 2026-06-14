import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional


class FaissIndexManager:
    def __init__(self, dim: int, index_path: str = "data/index.faiss", meta_path: str = "data/meta.json"):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.index: Optional[faiss.Index] = None
        self.metadatas: List[Dict[str, Any]] = []
        self._load()

    def _create_index(self):
        # inner product index (embeddings expected normalized)
        self.index = faiss.IndexFlatIP(self.dim)

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.metadatas = json.load(f)
            except Exception:
                self.index = None
                self.metadatas = []

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f)

    def add(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]):
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = embeddings.astype("float32")
        if self.index is None:
            self._create_index()
        # add into index
        self.index.add(embeddings)
        # append metadatas preserving id order
        self.metadatas.extend(metadatas)
        self.save()

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> List[List[Dict[str, Any]]]:
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        query_embeddings = query_embeddings.astype("float32")
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in range(query_embeddings.shape[0])]
        D, I = self.index.search(query_embeddings, k)
        results = []
        for row_ids, row_dists in zip(I, D):
            row = []
            for iid, dist in zip(row_ids, row_dists):
                if iid < 0 or iid >= len(self.metadatas):
                    continue
                md = dict(self.metadatas[int(iid)])
                md["score"] = float(dist)
                md["_id"] = int(iid)
                row.append(md)
            results.append(row)
        return results
