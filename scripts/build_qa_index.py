# scripts/build_qa_index.py
from pathlib import Path
import re, json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAGES = ROOT / "pages"
OUT_DIR = ROOT / "data" / "qa_index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Which files to index (add more if you want)
CANDIDATES = [
    ROOT / "README.md",
    ROOT / "app.py",
    PAGES / "0_Smoke_Test.py",
    PAGES / "2_Analogs_and_ML.py",
    PAGES / "4_ROI_Sensitivity.py",
    # add any docs you have:
    DATA / "COLUMN_SCHEMAS.md",            # optional
]

# Helper: extract helpful text from CSVs (headers + sample)
def csv_summary(path: Path, max_rows=5) -> str:
    try:
        df = pd.read_csv(path)
    except Exception:
        return ""
    sample = df.head(max_rows).to_dict(orient="records")
    return f"CSV {path.name} columns={list(df.columns)} sample={json.dumps(sample, ensure_ascii=False)}"

# Add CSV descriptions if present
CSV_FILES = [
    DATA / "tunisia_agil_stations.csv",
    DATA / "tunisia_charging_stations.csv",
    DATA / "processed_sites.csv",
    ROOT / "tn_ev_forecast.csv",           # if you export it here
]

def read_text_file(p: Path) -> str:
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        t = ""
    return t

def chunk_text(text: str, max_len=800):
    # naive splitter by paragraphs
    paras = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n")]
    buf, cur = [], ""
    for p in paras:
        if len(cur) + len(p) < max_len:
            cur = (cur + " " + p).strip()
        else:
            if cur: buf.append(cur)
            cur = p
    if cur: buf.append(cur)
    return [x for x in buf if x]

def main():
    docs = []
    for p in CANDIDATES:
        if p.exists():
            txt = read_text_file(p)
            for ch in chunk_text(txt):
                docs.append({"source": str(p.relative_to(ROOT)), "text": ch})

    for p in CSV_FILES:
        if p.exists():
            docs.append({"source": str(p.relative_to(ROOT)), "text": csv_summary(p)})

    if not docs:
        print("No docs found.")
        return

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode([d["text"] for d in docs], normalize_embeddings=True)
    emb = np.array(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine via dot since normalized
    index.add(emb)
    faiss.write_index(index, str(OUT_DIR / "index.faiss"))

    meta = {"model": MODEL_NAME, "docs": docs}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(docs)} chunks. Saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
