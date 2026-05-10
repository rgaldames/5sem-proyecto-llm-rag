import os
import json
import uuid
import time
import numpy as np

class MemoryStore:
    def __init__(self, embeddings, path="./memories.json"):
        self.path = path
        self.embeddings = embeddings
        self.memories = []
        self._vectors = None
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
            except Exception:
                self.memories = []
        else:
            self.memories = []
        if self.memories:
            self._vectors = np.array([m["embedding"] for m in self.memories], dtype=float)
        else:
            self._vectors = np.zeros((0,))

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def add_memory(self, text, metadata=None):
        if metadata is None:
            metadata = {}
        # if the caller provided original_text inside metadata, keep it
        emb = self.embeddings.embed_documents([text])[0]
        mem = {
            "id": str(uuid.uuid4()),
            "text": text,
            "metadata": metadata,
            "timestamp": int(time.time()),
            "embedding": emb,
        }
        self.memories.append(mem)
        # update numpy vectors
        if self._vectors.size == 0:
            self._vectors = np.array([emb], dtype=float)
        else:
            self._vectors = np.vstack([self._vectors, np.array(emb, dtype=float)])
        self._save()

    def _cosine_sim(self, a, b):
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
        b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
        return (a_n @ b_n.T).squeeze()

    def retrieve(self, query, top_k=5):
        if len(self.memories) == 0:
            return []
        qv = np.array(self.embeddings.embed_query(query), dtype=float)
        sims = self._cosine_sim(self._vectors, qv)
        # get top_k indices
        idx = np.argsort(-sims)[:top_k]
        results = [self.memories[i] for i in idx]
        return results
