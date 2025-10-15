# utils/rag.py
from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
IDX_DIR = ROOT / "data" / "qa_index"

class RagIndex:
    def __init__(self):
        meta_path = IDX_DIR / "meta.json"
        idx_path = IDX_DIR / "index.faiss"
        if not meta_path.exists() or not idx_path.exists():
            raise FileNotFoundError("RAG index not found. Run scripts/build_qa_index.py first.")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.docs = self.meta["docs"]
        self.model = SentenceTransformer(self.meta["model"])
        self.index = faiss.read_index(str(idx_path))

    def retrieve(self, query: str, k: int = 5):
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.index.search(q, k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            d = self.docs[idx]
            hits.append({"score": float(score), "source": d["source"], "text": d["text"]})
        return hits

def answer_from_docs(query: str, hits):
    # a tiny heuristic: stitch top chunks
    ctx = "\n\n".join([f"[{h['source']}] {h['text']}" for h in hits])
    # Simple templating for answer
    return f"**Answer (based on project docs):**\n\n{ctx}\n\n— If you need a calculation on current data (stations/forecast), ask things like: `How many stations in Sfax?` or `Chargers needed in 2030?`"
