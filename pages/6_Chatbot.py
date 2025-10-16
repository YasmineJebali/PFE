# pages/6_Chatbot.py — Repo-only ChatGPT: RAG + keyword + code-intel + live data — COMPLETE

from __future__ import annotations
import sys, re, json, unicodedata, difflib, textwrap, traceback, glob, ast, inspect
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PAGES = ROOT / "pages"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------
# Optional shared UI (falls back if utils.ui missing)
# ------------------------------------------------------------
try:
    from utils.ui import set_page, sidebar_nav, breadcrumbs, status_pills
    set_page("Assistant", icon="💬")
    sidebar_nav(__file__)
    breadcrumbs([("Home", "pages/0_Home.py"), ("Assistant", "pages/6_Chatbot.py")])
    status_pills()
except Exception:
    st.set_page_config(page_title="🤖 Assistant", layout="wide", page_icon="💬")

st.title("🤖 Project Assistant (repo-only)")

# ============================================================
# 0) Helpers
# ============================================================
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

# Governorate utilities for live answers
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
def normalize_gov(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _strip_accents(s).lower().strip()
    s = " ".join(s.replace("-", " ").split())
    for canon, alset in GOV_ALIASES.items():
        if s in alset: return canon
    return s
def match_governorate(query: str, universe: List[str]) -> Optional[str]:
    q = normalize_gov(query)
    if q in universe: return q
    best = difflib.get_close_matches(q, universe, n=1, cutoff=0.6)
    return best[0] if best else None

def status(ok: bool) -> str:
    return "✅" if ok else "❌"

# ============================================================
# 1) Data sources: session, uploaders, fallback auto-load
# ============================================================
stations_df = st.session_state.get("stations_df")
forecast_df = st.session_state.get("forecast_df")

st.subheader("Data status & quick loaders")
cA, cB, cC = st.columns([1.2,1.2,1])

with cA:
    st.markdown("**Stations (Agil)**")
    up_st = st.file_uploader("Upload stations CSV (id,name,lat,lon[,gov])", type=["csv"], key="chat_stations")
    if up_st is not None:
        try:
            tmp = pd.read_csv(up_st)
            tmp.columns = [c.strip() for c in tmp.columns]
            for c in ["lat","lon"]:
                if c in tmp.columns: tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            gov_col = None
            for c in tmp.columns:
                if c.lower() in {"gov","governorate","gouvernorat","gouvernorate"}:
                    gov_col = c; break
            if gov_col and gov_col != "gov": tmp = tmp.rename(columns={gov_col: "gov"})
            tmp["_gov_norm"] = tmp["gov"].astype(str).map(normalize_gov) if "gov" in tmp.columns else ""
            stations_df = tmp
            st.session_state["stations_df"] = stations_df.copy()
            st.success(f"Loaded stations: {len(stations_df)} rows.")
        except Exception as e:
            st.error(f"Stations CSV error: {e}")

    if stations_df is None:
        for cand in [DATA/"processed_sites.csv", DATA/"tunisia_agil_stations.csv"]:
            if cand.exists():
                try:
                    tmp = pd.read_csv(cand)
                    tmp.columns = [c.strip() for c in tmp.columns]
                    for c in ["lat","lon"]:
                        if c in tmp.columns: tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                    tmp["_gov_norm"] = tmp["gov"].astype(str).map(normalize_gov) if "gov" in tmp.columns else ""
                    if not tmp.empty:
                        stations_df = tmp
                        st.session_state["stations_df"] = stations_df.copy()
                        st.info(f"Auto-loaded stations from `{cand.name}`.")
                        break
                except Exception:
                    pass

    st.caption(f"Status: {status(stations_df is not None and not getattr(stations_df,'empty',True))}")

with cB:
    st.markdown("**Forecast (Tunisia)**")
    up_fc = st.file_uploader("Upload forecast CSV (year, chargers_needed[, ev_stock])", type=["csv"], key="chat_forecast")
    if up_fc is not None:
        try:
            tmp = pd.read_csv(up_fc)
            tmp.columns = [c.strip().lower() for c in tmp.columns]
            if "ev_stock_tn" in tmp.columns and "ev_stock" not in tmp.columns:
                tmp = tmp.rename(columns={"ev_stock_tn":"ev_stock"})
            if "year" in tmp.columns:
                tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce").astype("Int64")
            for c in ["chargers_needed","ev_stock"]:
                if c in tmp.columns: tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            tmp = tmp.dropna(subset=["year"])
            forecast_df = tmp
            st.session_state["forecast_df"] = forecast_df.copy()
            st.success(f"Loaded forecast: {len(forecast_df)} rows.")
        except Exception as e:
            st.error(f"Forecast CSV error: {e}")

    if forecast_df is None:
        for cand in [DATA/"tn_ev_forecast.csv", ROOT/"tn_ev_forecast.csv", DATA/"forecast.csv"]:
            if cand.exists():
                try:
                    tmp = pd.read_csv(cand)
                    tmp.columns = [c.strip().lower() for c in tmp.columns]
                    if "ev_stock_tn" in tmp.columns and "ev_stock" not in tmp.columns:
                        tmp = tmp.rename(columns={"ev_stock_tn":"ev_stock"})
                    if "year" in tmp.columns:
                        tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce").astype("Int64")
                    for c in ["chargers_needed","ev_stock"]:
                        if c in tmp.columns: tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                    tmp = tmp.dropna(subset=["year"])
                    if not tmp.empty and {"year","chargers_needed"}.issubset(tmp.columns):
                        forecast_df = tmp
                        st.session_state["forecast_df"] = forecast_df.copy()
                        st.info(f"Auto-loaded forecast from `{cand.name}`.")
                        break
                except Exception:
                    pass

    st.caption(f"Status: {status(forecast_df is not None and not getattr(forecast_df,'empty',True))}")

with cC:
    if st.button("Show previews"):
        if isinstance(stations_df, pd.DataFrame): st.dataframe(stations_df.head(10), use_container_width=True)
        if isinstance(forecast_df, pd.DataFrame): st.dataframe(forecast_df.head(10), use_container_width=True)

st.divider()

# ============================================================
# 2) RAG (semantic) + keyword fallback + code discovery
# ============================================================
def discover_files() -> List[Path]:
    files: List[Path] = []
    files += [Path(p) for p in glob.glob(str(PAGES / "*.py"))]
    files += [Path(p) for p in glob.glob(str(ROOT / "models" / "*.py"))]
    files += [Path(p) for p in glob.glob(str(ROOT / "utils" / "*.py"))]
    files += [ROOT / "app.py", ROOT / "README.md", ROOT / "requirements.txt",
              ROOT / "process_sites.py", ROOT / "fetch_osm_tunisia_ev_agil.py",
              ROOT / "schemas.py", ROOT / "config.py"]
    # Dedup
    seen, out = set(), []
    for p in files:
        if not p.exists(): continue
        k = str(p.resolve())
        if k in seen: continue
        seen.add(k); out.append(p)
    return out

def chunk_text(text: str, max_len=1000) -> List[str]:
    paras = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n")]
    buf, cur = [], ""
    for p in paras:
        if not p: continue
        if len(cur) + len(p) + 1 <= max_len: cur = (cur + " " + p).strip()
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

class SimpleRAG:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    def __init__(self):
        self.model = None
        self.index = None
        self.meta: List[Dict[str,str]] = []
        self.enabled = True
    def build(self, docs: List[Dict[str,str]]):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except Exception:
            self.enabled = False
            raise RuntimeError("Optional deps missing: install `faiss-cpu sentence-transformers` to enable semantic search.")
        self.model = SentenceTransformer(self.MODEL_NAME)
        texts = [d["text"] for d in docs]
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        emb = np.array(emb, dtype="float32")
        import faiss
        idx = faiss.IndexFlatIP(emb.shape[1]); idx.add(emb)
        self.index, self.meta = idx, docs
        st.session_state["_rag_index"] = idx
        st.session_state["_rag_meta"] = docs
        st.session_state["_rag_model_name"] = self.MODEL_NAME
    def load_from_session(self) -> bool:
        ok = "_rag_index" in st.session_state and "_rag_meta" in st.session_state
        if ok:
            self.index = st.session_state["_rag_index"]
            self.meta  = st.session_state["_rag_meta"]
            self.enabled = True
        return ok
    def is_ready(self) -> bool:
        return self.enabled and (self.index is not None) and (self.meta is not None)
    def retrieve(self, q: str, k: int = 6) -> List[Dict[str,Any]]:
        if not self.is_ready(): return []
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(st.session_state.get("_rag_model_name", self.MODEL_NAME))
            except Exception:
                return []
        v = self.model.encode([q], normalize_embeddings=True, show_progress_bar=False).astype("float32")
        D, I = self.index.search(v, k)
        hits = []
        for r, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.meta): continue
            hits.append({"rank": r+1, "score": float(D[0][r]), **self.meta[idx]})
        return hits

def build_index_inline() -> int:
    docs: List[Dict[str,str]] = []
    for p in discover_files():
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for ch in chunk_text(txt, max_len=1000):
                docs.append({"source": str(p.relative_to(ROOT)), "text": ch})
        except Exception:
            continue
    if DATA.exists():
        for p in DATA.glob("*.csv"):
            s = csv_summary(p)
            if s:
                docs.append({"source": str(p.relative_to(ROOT)), "text": s})
    if not docs: raise RuntimeError("No docs found to index; make sure repo files exist and /data has CSVs.")
    rag = SimpleRAG(); rag.build(docs)
    st.session_state["_rag_ready"] = True
    return len(docs)

def keyword_search(question: str, topk: int = 6) -> Optional[str]:
    q = question.lower()
    scored = []
    for p in discover_files():
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        score = sum(txt.lower().count(tok) for tok in re.findall(r"\w+", q))
        if score > 0:
            snippet = textwrap.shorten(txt, width=900, placeholder=" …")
            scored.append((score, str(p.relative_to(ROOT)), snippet))
    if not scored: return None
    scored.sort(reverse=True)
    out = "\n".join([f"- **{name}**: {snip}" for _, name, snip in scored[:topk]])
    return "I searched your repo and found:\n\n" + out

def answer_from_hits(hits: List[Dict[str,str]]) -> str:
    if not hits:
        return "Nothing obvious in the index. Try **Build / Refresh knowledge index** or use a command like `search \"pattern\"`."
    bullets = []
    for h in hits[:5]:
        bullets.append(f"- **{h['source']}**: {textwrap.shorten(h['text'], width=900, placeholder=' …')}")
    return "Here’s what I found:\n\n" + "\n".join(bullets)

# ============================================================
# 3) Code intelligence (show/search/explain/where defined)
# ============================================================
def list_files(pattern: str = "**/*.py") -> List[str]:
    files = [str(p.relative_to(ROOT)) for p in ROOT.glob(pattern) if p.is_file()]
    return sorted(files)

def load_text(rel_path: str) -> Optional[str]:
    p = (ROOT / rel_path).resolve()
    if not p.exists(): return None
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

def grep_regex(pattern: str, in_glob: str = "**/*.py", max_hits=200) -> List[Tuple[str,int,str]]:
    rx = re.compile(pattern, re.IGNORECASE)
    matches = []
    for p in ROOT.glob(in_glob):
        if not p.is_file(): continue
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh, start=1):
                    if rx.search(line):
                        matches.append((str(p.relative_to(ROOT)), i, line.rstrip("\n")))
                        if len(matches) >= max_hits:
                            return matches
        except Exception:
            continue
    return matches

def parse_functions(rel_path: str) -> Dict[str, Dict[str,Any]]:
    txt = load_text(rel_path)
    if txt is None: return {}
    try:
        tree = ast.parse(txt)
    except Exception:
        return {}
    out: Dict[str, Dict[str,Any]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            doc = ast.get_docstring(node) or ""
            # find start/end lines
            try:
                # Python 3.8+: node.lineno; to get end, find next sibling by lineno
                lineno = node.lineno
                # heuristic: scan source lines; stop before next top-level def/class
                lines = txt.splitlines()
                end = len(lines)
                for j in range(node.lineno, len(lines)):
                    if re.match(r'^\s*(def|class)\s+\w+', lines[j]) and j+1>node.lineno:
                        end = j; break
                code = "\n".join(lines[lineno-1:end])
            except Exception:
                code = ""
            out[name] = {"doc": doc, "code": code, "lineno": node.lineno}
    return out

def where_defined(name: str) -> List[Tuple[str,int]]:
    hits = []
    for p in discover_files():
        if p.suffix != ".py": continue
        funcs = parse_functions(str(p.relative_to(ROOT)))
        if name in funcs:
            hits.append((str(p.relative_to(ROOT)), funcs[name]["lineno"]))
    return hits

# ============================================================
# 4) Build / Refresh index
# ============================================================
c1, c2 = st.columns([1,1])
with c1:
    if st.button("🛠️ Build / Refresh knowledge index"):
        try:
            n = build_index_inline()
            st.success(f"Indexed {n} chunks.")
        except Exception as e:
            st.error(f"Index build failed: {e}")
            st.caption("Tip: optional deps → `pip install faiss-cpu sentence-transformers`")
with c2:
    st.caption("Commands: `help`, `list files`, `show file <path|glob>`, `search \"regex\" [in <glob>]`, "
               "`find function <name>`, `explain function <name> [in <path>]`, `where defined <name>`, "
               "`summarize file <path>`, `npv capex=... ...`")

st.divider()

# ============================================================
# 5) Live intents (data-aware) + finance
# ============================================================
def live_answer(q: str) -> Optional[str]:
    ql = q.lower().strip()

    if re.search(r'^(help|what can you do|\?)$', ql):
        return (
            "I’m specialized for **this repo**. I can:\n"
            "• Answer questions about your files (code/logic) using semantic search or keywords\n"
            "• Do code intelligence: `list files`, `show file`, `search \"regex\"`, `find function`, `explain function`, `where defined`\n"
            "• Answer live from your loaded CSVs (chargers/year, EV stock, stations by governorate)\n"
            "• Quick finance: `npv capex=36000 opex=2500 margin=0.55 kwh=22 sessions=3 rate=0.11 years=8`\n"
            "• Build/refresh the knowledge index with the button above"
        )

    # npv calculator
    if ql.startswith("npv "):
        kv = dict(re.findall(r"(\w+)\s*=\s*([-+]?\d*\.?\d+)", ql))
        try:
            capex = float(kv.get("capex", "36000"))
            opex = float(kv.get("opex", "2500"))
            margin = float(kv.get("margin", "0.55"))
            kwh = float(kv.get("kwh", "22"))
            sessions = float(kv.get("sessions", "3"))
            rate = float(kv.get("rate", "0.11"))
            years = int(float(kv.get("years", "8")))
            rev_y = margin * kwh * sessions * 365.0
            net_y = rev_y - opex
            cf = np.zeros(years + 1, dtype=float); cf[0] = -capex
            if years >= 1: cf[1:] = net_y
            disc = np.array([(1.0 + rate) ** t for t in range(len(cf))], dtype=float)
            npv = float(np.sum(cf / disc))
            cum = np.cumsum(cf); idx = np.where(cum >= 0)[0]; pb = int(idx[0]) if len(idx) else None
            return (f"Per-charger NPV = **{npv:,.0f} TND**  "
                    f"(capex={capex:,.0f}, opex={opex:,.0f}, margin={margin}, kWh={kwh}, sessions={sessions}, "
                    f"rate={rate}, years={years}). Payback: **{pb if pb is not None else 'None'}**")
        except Exception as e:
            return f"Couldn’t compute NPV: {e}"

    # Data-aware Q&A (requires uploaded/auto-loaded data)
    if "chargers needed in" in ql:
        m = re.search(r"chargers needed in (\d{4})", ql)
        if m and isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
            year = int(m.group(1))
            df = forecast_df.copy()
            if "year" in df.columns:
                row = df[df["year"] == year]
                if not row.empty and "chargers_needed" in row.columns:
                    return f"Chargers needed in **{year}**: **{int(row['chargers_needed'].iloc[0])}**"
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
    if re.search(r"(?:ev\s*stock|number of evs|evs)\s+in\s+(\d{4})", ql):
        m = re.search(r"(?:ev\s*stock|number of evs|evs)\s+in\s+(\d{4})", ql)
        if m and isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty and "ev_stock" in forecast_df.columns:
            year = int(m.group(1))
            row = forecast_df[forecast_df["year"] == year]
            if not row.empty:
                return f"EV stock in **{year}**: **{int(row['ev_stock'].iloc[0])}**"

    m = re.search(r"(?:agil|stations?)\s+(?:in|at)\s+([a-zA-Z\u00C0-\u017F\s\-']+)\??$", ql)
    if m and isinstance(stations_df, pd.DataFrame) and not stations_df.empty:
        query_gov = m.group(1).strip()
        tmp = stations_df.copy()
        if "_gov_norm" not in tmp.columns:
            if "gov" in tmp.columns: tmp["_gov_norm"] = tmp["gov"].astype(str).map(normalize_gov)
            else: return "I don't see a governorate column in the stations data."
        matched = match_governorate(query_gov, GOV_CANON)
        if not matched:
            return f"I couldn't match **{query_gov}** to a known governorate."
        cnt = int((tmp["_gov_norm"] == matched).sum())
        pretty = " ".join(w.capitalize() for w in matched.split())
        return f"Agil stations in **{pretty}**: **{cnt}**"

    if "how many stations" in ql and isinstance(stations_df, pd.DataFrame) and not stations_df.empty:
        return f"Total Agil stations loaded: **{len(stations_df)}**"
    if "how many years" in ql and isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        return f"Forecast covers **{forecast_df['year'].nunique()}** years (from {int(forecast_df['year'].min())} to {int(forecast_df['year'].max())})."

    # Quick docs help
    if re.search(r"\bexplain\b.*(logistic|k[, ]?r[, ]?t0|k r t0)", ql):
        return ("Logistic EV adoption:  y(t) = K / (1 + exp(-r*(t - t0))).\n"
                "• K = carrying capacity (max)\n• r = growth rate\n• t0 = inflection (50% K)\n"
                "Fitted with non-linear least squares; RMSE/MAPE for quality; `rough_fit` for scarce points.")

    if re.search(r"\bexplain\b.*(npv|payback|irr)", ql):
        return ("Finance: NPV = Σ CF_t/(1+r)^t (capex at install, annual net profit thereafter). "
                "Payback = first t with cumulative ≥ 0 (undiscounted). IRR solves NPV=0.")

    return None

# ============================================================
# 6) Command router (code-intel & file QA)
# ============================================================
def run_command(q: str) -> Optional[str]:
    q = q.strip()

    # list files [glob]
    m = re.match(r"^list files(?:\s+(.*))?$", q, flags=re.I)
    if m:
        patt = m.group(1) or "**/*.py"
        files = list_files(patt)
        if not files: return f"No files match `{patt}`."
        return "Files:\n" + "\n".join(f"- {f}" for f in files[:400])

    # show file <path|glob>
    m = re.match(r"^show file\s+(.+)$", q, flags=re.I)
    if m:
        arg = m.group(1).strip()
        paths = list(ROOT.glob(arg)) if any(ch in arg for ch in "*?[]") else [ROOT / arg]
        shown = 0
        out = []
        for p in paths:
            if not p.exists() or not p.is_file(): continue
            rel = str(p.relative_to(ROOT))
            txt = load_text(rel) or ""
            short = txt if len(txt) <= 6000 else (txt[:6000] + "\n... [truncated]")
            out.append(f"### {rel}\n```python\n{short}\n```")
            shown += 1
            if shown >= 3: break
        return "\n\n".join(out) if out else f"No readable file for `{arg}`."

    # search "regex" [in <glob>]
    m = re.match(r'^search\s+"(.+?)"(?:\s+in\s+(.+))?$', q, flags=re.I)
    if m:
        pattern = m.group(1)
        in_glob = m.group(2) or "**/*.py"
        hits = grep_regex(pattern, in_glob=in_glob, max_hits=120)
        if not hits: return f"No matches for /{pattern}/ in `{in_glob}`."
        lines = [f"- {path}:{ln} — {textwrap.shorten(line.strip(), 160)}" for path, ln, line in hits]
        return f"Matches for /{pattern}/ in `{in_glob}`:\n\n" + "\n".join(lines)

    # find function <name>
    m = re.match(r"^find function\s+([A-Za-z_]\w*)$", q, flags=re.I)
    if m:
        name = m.group(1)
        places = where_defined(name)
        if not places: return f"Function/class `{name}` not found."
        return "Definitions:\n" + "\n".join(f"- {p}:{ln}" for p, ln in places)

    # where defined <name>
    m = re.match(r"^where defined\s+([A-Za-z_]\w*)$", q, flags=re.I)
    if m:
        name = m.group(1)
        places = where_defined(name)
        if not places: return f"`{name}` not defined in indexed files."
        return "Found at:\n" + "\n".join(f"- {p}:{ln}" for p, ln in places)

    # explain function <name> [in <path>]
    m = re.match(r"^explain function\s+([A-Za-z_]\w*)(?:\s+in\s+(.+))?$", q, flags=re.I)
    if m:
        name = m.group(1); path_hint = m.group(2)
        candidates = []
        if path_hint:
            for p in ROOT.glob(path_hint):
                if p.suffix == ".py":
                    candidates.append(str(p.relative_to(ROOT)))
        else:
            candidates = [str(p.relative_to(ROOT)) for p in discover_files() if p.suffix==".py"]

        for rel in candidates:
            funcs = parse_functions(rel)
            if name in funcs:
                info = funcs[name]
                doc = info["doc"] or "(no docstring)"
                code = info["code"].strip()
                short = "\n".join(code.splitlines()[:120])
                out = f"**{name}** in `{rel}` (line {info['lineno']})\n\n**Docstring:**\n{doc}\n\n**Code:**\n```python\n{short}\n```"
                if len(code.splitlines()) > 120:
                    out += "\n\n*(truncated; use `show file {rel}` to see full)*"
                return out
        return f"Couldn’t find `{name}`. Try `find function {name}` or provide `in <path>`."

    # summarize file <path>
    m = re.match(r"^summarize file\s+(.+)$", q, flags=re.I)
    if m:
        rel = m.group(1).strip()
        txt = load_text(rel)
        if txt is None: return f"File not found: `{rel}`."
        # naive summary: grab imports, defs, classes
        lines = txt.splitlines()
        imports = [ln for ln in lines if re.match(r'^\s*(from\s+\S+\s+import|import\s+\S+)', ln)]
        defs = [ln for ln in lines if re.match(r'^\s*def\s+\w+', ln)]
        clss = [ln for ln in lines if re.match(r'^\s*class\s+\w+', ln)]
        return (f"**Summary of `{rel}`**\n\n"
                f"- Imports ({len(imports)}):\n" + "\n".join("  " + textwrap.shorten(x, 100) for x in imports[:40]) + "\n\n"
                f"- Classes ({len(clss)}):\n" + "\n".join("  " + x for x in clss[:40]) + "\n\n"
                f"- Functions ({len(defs)}):\n" + "\n".join("  " + x for x in defs[:80]))

    return None

# ============================================================
# 7) Chat UI
# ============================================================
if "chat" not in st.session_state:
    st.session_state.chat = []

# RAG instance
rag = SimpleRAG()
_ = rag.load_from_session()

# history
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

prompt = st.chat_input("Ask anything about THIS project. Try: help | list files | search \"logistic\" | explain function fit_logistic")
if prompt:
    st.session_state.chat.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # 1) Commands (code-intel)
        reply = run_command(prompt)
        # 2) Live intents / data-aware
        if reply is None:
            reply = live_answer(prompt)
        # 3) Semantic RAG
        if reply is None and rag.is_ready():
            hits = rag.retrieve(prompt, k=6)
            reply = answer_from_hits(hits)
        # 4) Keyword fallback
        if reply is None:
            reply = keyword_search(prompt)
        # 5) Last resort
        if reply is None:
            reply = ("I couldn’t answer yet. Click **Build / Refresh knowledge index** above, "
                     "or use commands like `search \"pattern\"`, `show file <path>`, `find function <name>`.")
    except Exception as e:
        reply = f"An error occurred:\n\n```\n{traceback.format_exc()}\n```"

    st.session_state.chat.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)

# ============================================================
# 8) Debug (optional)
# ============================================================
with st.expander("🔎 Debug (optional)"):
    st.write("RAG ready:", rag.is_ready())
    if isinstance(stations_df, pd.DataFrame) and not stations_df.empty:
        st.write("Stations rows:", len(stations_df))
        if "_gov_norm" in stations_df.columns:
            st.write("Governorate examples:", stations_df["_gov_norm"].dropna().unique()[:10].tolist())
    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        st.write("Forecast years:", f"{int(forecast_df['year'].min())}–{int(forecast_df['year'].max())}")
