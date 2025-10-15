# pages/6_Chatbot.py — robust Assistant (RAG + live data + upload + auto-fallback)
# Paste this whole file into agil-ev-mvp/pages/6_Chatbot.py

import sys, re, json, unicodedata, difflib
from pathlib import Path
import pandas as pd
import streamlit as st

# ---------------- Project paths ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAGES = ROOT / "pages"
IDX_DIR = DATA / "qa_index"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------- Optional RAG helper import ----------------
try:
    from utils.rag import RagIndex, answer_from_docs
except Exception:
    RagIndex = None
    def answer_from_docs(q, hits):  # fallback text if RAG missing
        return "Knowledge index not available. Click **Build / Refresh** above."

from utils.ui import set_page, sidebar_nav, breadcrumbs, status_pills
set_page("Assistant", icon="💬")
sidebar_nav(__file__)
breadcrumbs([
    ("Home", "pages/0_Home.py"),
    ("Assistant", "pages/6_Chatbot.py"),
])
status_pills()


# ---------------- Streamlit page setup ----------------
st.set_page_config(page_title="🤖 Assistant", layout="wide")
st.title("🤖 Project Assistant")

# ============================================================
# 1) Normalization helpers (governorates) + fuzzy matching
# ============================================================
GOV_CANON = [
    "ariana","beja","ben arous","bizerte","gabes","gafsa","jendouba","kairouan",
    "kasserine","kebili","kef","mahdia","manouba","medenine","monastir","nabeul",
    "sfax","sidi bouzid","siliana","sousse","tataouine","tozeur","tunis","zaghouan"
]
GOV_ALIASES = {
    "ben arous": {"benarous","ben  arous","ben-arous"},
    "kef": {"le kef"},
    "kebili": {"kébili","kbili"},
    "beja": {"béja","bejA"},
    "mahdia": {"el mahdia"},
    "medenine": {"médenine"},
    "nabeul": {"nabul","nabEul"},
    "sidi bouzid": {"sidi-bouzid","sidi  bouzid","sidibouzid"},
    "tunis": {"grand tunis","tunis centre"},
    "tozeur": {"tuzer"},
    "zaghouan": {"zaghwan"},
    "bizerte": {"bizert"},
    "sousse": {"soussa"},
    "sfax": {"safax","sfax ville"}
}

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def normalize_gov(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _strip_accents(s).lower().strip()
    s = " ".join(s.replace("-", " ").split())
    for canon, alset in GOV_ALIASES.items():
        if s in alset:
            return canon
    return s

def match_governorate(query: str, universe: list[str]) -> str | None:
    q = normalize_gov(query)
    if q in universe:
        return q
    best = difflib.get_close_matches(q, universe, n=1, cutoff=0.6)
    return best[0] if best else None

# ============================================================
# 2) CSV coercers (FORECAST + STATIONS)
# ============================================================
def coerce_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns to: year, chargers_needed, ev_stock (optional)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]

    # rename common alternates
    colmap = {}
    if "year" not in out.columns:
        for c in out.columns:
            if c in {"annee","yr"}: colmap[c] = "year"
    if "chargers_needed" not in out.columns:
        for c in out.columns:
            if c in {"chargers","public_chargers","needed_chargers","nb_chargers"}:
                colmap[c] = "chargers_needed"
    if "ev_stock" not in out.columns:
        for c in out.columns:
            if c in {"evs","stock","ev_stock_tn"}:
                colmap[c] = "ev_stock"
    if colmap:
        out = out.rename(columns=colmap)

    # types
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    for c in ["chargers_needed","ev_stock"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # drop rows without year
    if "year" in out.columns:
        out = out.dropna(subset=["year"])
    return out

def coerce_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure lat/lon numeric, standardize governorate as 'gov' + add _gov_norm."""
    if df is None or df.empty:
        return df
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    for c in ["lat","lon"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    gov_col = None
    for c in out.columns:
        if c.lower() in {"gov","governorate","gouvernorat","gouvernorate"}:
            gov_col = c
            break
    if gov_col and gov_col != "gov":
        out = out.rename(columns={gov_col: "gov"})
    if "gov" in out.columns:
        out["_gov_norm"] = out["gov"].astype(str).map(normalize_gov)
    else:
        out["_gov_norm"] = ""
    return out

# ============================================================
# 3) Data sources: session, uploaders, fallback auto-load
# ============================================================
stations_df = st.session_state.get("stations_df")
forecast_df = st.session_state.get("forecast_df")

def status_badge(ok: bool) -> str:
    return "✅" if ok else "❌"

st.subheader("Data status & quick loaders")
col_a, col_b, col_c = st.columns([1.2,1.2,1])

with col_a:
    st.markdown("**Stations (Agil)**")
    up_st = st.file_uploader("Upload stations CSV (id,name,lat,lon[,gov])",
                             type=["csv"], key="chat_stations")
    if up_st is not None:
        tmp = pd.read_csv(up_st)
        tmp = coerce_stations(tmp)
        stations_df = tmp
        st.session_state["stations_df"] = stations_df.copy()
        st.success(f"Loaded stations from upload: {len(stations_df)} rows.")

    if stations_df is None:
        # try fallback files
        for cand in [DATA/"tunisia_agil_stations.csv", DATA/"processed_sites.csv"]:
            if cand.exists():
                try:
                    tmp = pd.read_csv(cand)
                    tmp = coerce_stations(tmp)
                    if not tmp.empty:
                        stations_df = tmp
                        st.session_state["stations_df"] = stations_df.copy()
                        st.info(f"Auto-loaded stations from `{cand}`.")
                        break
                except Exception:
                    pass

    st.caption(f"Status: {status_badge(stations_df is not None and not getattr(stations_df,'empty',True))}")

with col_b:
    st.markdown("**Forecast (Tunisia)**")
    up_fc = st.file_uploader("Upload forecast CSV (year, chargers_needed[, ev_stock])",
                             type=["csv"], key="chat_forecast")
    if up_fc is not None:
        tmp = pd.read_csv(up_fc)
        tmp = coerce_forecast(tmp)
        forecast_df = tmp
        st.session_state["forecast_df"] = forecast_df.copy()
        st.success(f"Loaded forecast from upload: {len(forecast_df)} rows.")

    if forecast_df is None:
        # try fallback files
        for cand in [ROOT/"tn_ev_forecast.csv", DATA/"tn_ev_forecast.csv", DATA/"forecast.csv"]:
            if cand.exists():
                try:
                    tmp = pd.read_csv(cand)
                    tmp = coerce_forecast(tmp)
                    if not tmp.empty and {"year","chargers_needed"}.issubset(tmp.columns):
                        forecast_df = tmp
                        st.session_state["forecast_df"] = forecast_df.copy()
                        st.info(f"Auto-loaded forecast from `{cand}`.")
                        break
                except Exception:
                    pass

    st.caption(f"Status: {status_badge(forecast_df is not None and not getattr(forecast_df,'empty',True))}")

with col_c:
    if st.button("Show small previews"):
        if isinstance(stations_df, pd.DataFrame):
            st.write("**Stations preview**")
            st.dataframe(stations_df.head(10), use_container_width=True)
        if isinstance(forecast_df, pd.DataFrame):
            st.write("**Forecast preview**")
            st.dataframe(forecast_df.head(10), use_container_width=True)

st.divider()

# ============================================================
# 4) Knowledge index build/refresh (inline)
# ============================================================
def chunk_text(text: str, max_len=800):
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

def csv_summary(path: Path, max_rows=5) -> str:
    try:
        df = pd.read_csv(path)
    except Exception:
        return ""
    sample = df.head(max_rows).to_dict(orient="records")
    return f"CSV {path.name} columns={list(df.columns)} sample={json.dumps(sample, ensure_ascii=False)}"

def build_index_inline() -> int:
    from sentence_transformers import SentenceTransformer
    import faiss, numpy as np

    IDX_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    candidates = [
        ROOT / "README.md",
        ROOT / "app.py",
        PAGES / "0_Smoke_Test.py",
        PAGES / "2_Analogs_and_ML.py",
        PAGES / "4_ROI_Sensitivity.py",
    ]
    csvs = [
        DATA / "tunisia_agil_stations.csv",
        DATA / "tunisia_charging_stations.csv",
        DATA / "processed_sites.csv",
        ROOT / "tn_ev_forecast.csv",
        DATA / "tn_ev_forecast.csv",
        DATA / "forecast.csv",
    ]

    docs = []
    for p in candidates:
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for ch in chunk_text(txt):
                docs.append({"source": str(p.relative_to(ROOT)), "text": ch})
    for p in csvs:
        if p.exists():
            docs.append({"source": str(p.relative_to(ROOT)), "text": csv_summary(p)})

    if not docs:
        raise RuntimeError("No docs found to index; make sure README/app/pages exist and CSVs are in /data.")

    model = SentenceTransformer(MODEL_NAME := "sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode([d["text"] for d in docs], normalize_embeddings=True)

    import numpy as _np
    emb = _np.array(emb, dtype="float32")
    # emb = pd.np.array(emb, dtype="float32")  # compat trick for some numpy versions

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine via dot on normalized
    index.add(emb)
    faiss.write_index(index, str(IDX_DIR / "index.faiss"))

    meta = {"model": MODEL_NAME, "docs": docs}
    (IDX_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(docs)

col_build1, col_build2 = st.columns([1,1])
with col_build1:
    if st.button("🛠️ Build / Refresh knowledge index"):
        try:
            n = build_index_inline()
            st.success(f"Indexed {n} chunks. You can ask doc questions now.")
        except Exception as e:
            st.error(f"Index build failed: {e}")
with col_build2:
    if (IDX_DIR / "index.faiss").exists():
        st.success("Knowledge index found.")
    else:
        st.warning("No knowledge index yet. Click the button to build.")

st.divider()

# ============================================================
# 5) Load RAG (if ready)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_rag():
    return RagIndex() if RagIndex is not None else None

try:
    rag = load_rag()
except Exception:
    rag = None

# ============================================================
# 6) Intent router — live numeric answers first
# ============================================================
def try_live_answer(q: str):
    ql = q.lower().strip()

    # chargers needed in YEAR
    m = re.search(r"chargers?\s+needed\s+in\s+(\d{4})", ql)
    if m and isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        year = int(m.group(1))
        df = forecast_df.copy()
        if "year" not in df.columns:
            return None
        row = df[df["year"] == year]
        if row.empty:
            # nearest year fallback
            try:
                yr = df["year"].dropna().astype(int)
                closest = int(yr.iloc[(yr - year).abs().argsort().iloc[0]])
                row = df[df["year"] == closest]
                if not row.empty and "chargers_needed" in row.columns:
                    val = int(row["chargers_needed"].iloc[0])
                    return f"Chargers needed in **{year}**: not found; nearest is **{closest} → {val}**."
            except Exception:
                pass
            return f"I couldn’t find `chargers_needed` for **{year}**."
        if "chargers_needed" in row.columns:
            val = int(row["chargers_needed"].iloc[0])
            return f"Chargers needed in **{year}**: **{val}**"

    # EV stock in YEAR
    m = re.search(r"(?:ev\s*stock|number of evs|evs)\s+in\s+(\d{4})", ql)
    if m and isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        year = int(m.group(1))
        df = forecast_df.copy()
        if "year" not in df.columns or "ev_stock" not in df.columns:
            return None
        row = df[df["year"] == year]
        if row.empty:
            return f"I couldn’t find `ev_stock` for **{year}**."
        val = int(row["ev_stock"].iloc[0])
        return f"EV stock in **{year}**: **{val}**"

    # stations in governorate  (robust matching)
    m = re.search(r"(?:agil|stations?)\s+(?:in|at)\s+([a-zA-Z\u00C0-\u017F\s\-']+)\??$", ql)
    if m and isinstance(stations_df, pd.DataFrame) and not stations_df.empty:
        query_gov = m.group(1).strip()
        tmp = stations_df.copy()
        # get/prepare normalized column
        if "_gov_norm" not in tmp.columns:
            col = None
            for c in tmp.columns:
                if c.lower() in {"gov","governorate","gouvernorat","gouvernorate"}:
                    col = c; break
            if not col:
                return "I don't see a governorate column in the stations data."
            tmp["_gov_norm"] = tmp[col].astype(str).map(normalize_gov)

        matched = match_governorate(query_gov, GOV_CANON)
        if not matched:
            return f"I couldn't match **{query_gov}** to a known governorate."
        cnt = int((tmp["_gov_norm"] == matched).sum())
        pretty = " ".join(w.capitalize() for w in matched.split())
        return f"Agil stations in **{pretty}**: **{cnt}**"

    # total stations
    if "how many stations" in ql and isinstance(stations_df, pd.DataFrame) and not stations_df.empty:
        return f"Total Agil stations loaded: **{len(stations_df)}**"

    return None

# ============================================================
# 7) Chat UI
# ============================================================
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

prompt = st.chat_input("Ask about the project, data, or quick metrics…")
if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    reply = try_live_answer(prompt)
    if reply is None:
        if rag:
            hits = rag.retrieve(prompt, k=5)
            reply = answer_from_docs(prompt, hits)
        else:
            reply = "I don’t have the knowledge index yet. Click **Build / Refresh knowledge index** above, or upload data for live answers."

    st.session_state.chat.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)

# ============================================================
# 8) Debug helper (optional)
# ============================================================
with st.expander("🔎 Debug governorates (optional)"):
    if isinstance(stations_df, pd.DataFrame) and not stations_df.empty:
        if "_gov_norm" in stations_df.columns:
            vals = stations_df["_gov_norm"].dropna().unique().tolist()
            st.write("Unique `_gov_norm` values:", sorted(vals))
        else:
            st.write("No `_gov_norm` column; it will be created on the fly during queries.")
