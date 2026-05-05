"""
research_pipeline.py
====================
Pipeline deterministico per espandere e formalizzare un pensiero utente
attraverso ricerca su fonti scientifiche accreditate.

Flusso:
  1. Decomposizione del pensiero → sotto-query strutturate (JSON)
  2. Retrieval parallelo: Arxiv, PubMed, Semantic Scholar, OpenAlex
  3. Deduplicazione + scoring (citation count, anno, rilevanza semantica)
  4. Sintesi finale con citazioni verificabili
  5. Salvataggio run completa (JSONL tracciato)

Dipendenze:
  pip install anthropic arxiv biopython requests pyalex sentence-transformers
"""

import math
import os
import json
import re
import time
import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, TypedDict

from openai import OpenAI
import arxiv
import pyalex
import requests
from Bio import Entrez
from pyalex import Works
from sentence_transformers import SentenceTransformer, util
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

# ─────────────────────────────────────────────
# CONFIGURAZIONE GLOBALE — modifica qui
# ─────────────────────────────────────────────

CONFIG = {
    # GitHub Copilot (OpenAI-compatible)
    # Modelli disponibili: gpt-4o, gpt-4o-mini, o1, o3-mini,
    #                      claude-3.5-sonnet, claude-3-haiku, ...
    "model": "gpt-4o",
    "temperature": 0,           # determinismo massimo
    "max_tokens_decompose": 1500,
    "max_tokens_synthesize": 4000,

    # Retrieval
    "max_results_per_source": 8,    # paper per sorgente
    "min_total_papers": 10,         # soglia minima prima di sintetizzare
    "max_retry_cycles": 2,          # ricicli se risultati insufficienti

    # Scoring
    "weight_citations": 0.4,
    "weight_recency": 0.2,
    "weight_semantic": 0.4,
    "min_year": 2015,               # filtra paper troppo vecchi (0 = disabilitato)

    # PubMed — inserisci la tua email (obbligatorio per le API NCBI)
    "pubmed_email": "itg.ekin@gmail.com",

    # Embedding model (fisso = deterministico)
    "embedding_model": "all-MiniLM-L6-v2",

    # Output
    "output_dir": "./runs",
    "cache_db": "./cache.db",

    # API keys opzionali (aumentano i rate limit delle API pubbliche)
    # Semantic Scholar: https://www.semanticscholar.org/product/api
    "semantic_scholar_api_key": os.environ.get("SEMANTIC_SCHOLAR_API_KEY", ""),
    # OpenAlex: una email attiva il "polite pool" (nessuna registrazione)
    "openalex_email": os.environ.get("OPENALEX_EMAIL", ""),

    # Modalità sintesi
    "hierarchical_synthesis": True,   # True = outline + sezioni (+ optional reflection)
    "enable_reflection": False,       # True = self-reflection extra per ogni sezione (raddoppia token!)
    "max_tokens_outline": 1500,        # token per la generazione dell'indice
    "max_tokens_section": 2500,        # token per la scrittura di ogni sezione

    # Ottimizzazione token: limiti dei contenuti inviati all'LLM
    "abstract_chars_synthesis": 300,   # caratteri abstract per la sintesi gerarchica (era 600)
    "abstract_chars_outline": 0,       # 0 = non includere abstract nell'outline
    "abstract_chars_simple": 300,      # caratteri abstract per synthesize() flat (era 400)

    # ── Features avanzate ────────────────────────────────────────────────────
    # Fallback model se il primario restituisce 429
    "fallback_model": "gpt-4o-mini",
    # Cache TTL per query keyword (giorni; 0 = no scadenza)
    "cache_ttl_days": 30,
    # Citation graph: recupera le referenze dei top-paper via SS batch API
    "enable_citation_graph": True,
    "citation_graph_max_papers": 8,    # top-N paper da cui espandere il grafo
    # MMR: selezione diversificata (Maximal Marginal Relevance)
    "enable_mmr": True,
    "mmr_lambda": 0.6,   # 0=max diversità, 1=max rilevanza
    "mmr_top_k": 25,     # paper passati a outline/synthesis dopo MMR
    # Query rewriting: rigenera query per subtopic con troppo pochi risultati
    "enable_query_rewrite": True,
    "min_papers_per_subtopic": 3,
    # Validatore citazioni post-synthesis: controlla che ogni URL sia raggiungibile
    "enable_citation_validation": True,
}

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
Entrez.email = CONFIG["pubmed_email"]

# Caricato una sola volta a livello di modulo (singleton) — evita reload ad ogni run
log.info(f"Caricamento embedding model '{CONFIG['embedding_model']}'...")
_EMBEDDING_MODEL = SentenceTransformer(CONFIG["embedding_model"])


# ─────────────────────────────────────────────
# LLM STATS — tracciamento di chiamate, token e tempo
# ─────────────────────────────────────────────

class LLMStats:
    """Tracker globale delle chiamate LLM (calls, token in/out, tempo cumulativo)."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_seconds = 0.0
        self.start_ts: Optional[float] = None

    def start_run(self):
        self.start_ts = time.time()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_seconds = 0.0

    def track(self, response, elapsed: float):
        self.calls += 1
        self.total_seconds += elapsed
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def wall_seconds(self) -> float:
        return time.time() - self.start_ts if self.start_ts else 0.0

    def summary_md(self, model: str) -> str:
        """Restituisce un blocco Markdown con i metadati della run."""
        return (
            f"- **Modello LLM**: `{model}`\n"
            f"- **Chiamate LLM**: {self.calls}\n"
            f"- **Token totali**: {self.total_tokens:,} "
            f"(prompt: {self.prompt_tokens:,}, completion: {self.completion_tokens:,})\n"
            f"- **Tempo LLM**: {self.total_seconds:.1f}s\n"
            f"- **Tempo totale run**: {self.wall_seconds:.1f}s"
        )

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_seconds": round(self.total_seconds, 2),
            "wall_seconds": round(self.wall_seconds, 2),
        }


LLM_STATS = LLMStats()


def llm_chat(client: OpenAI, *, system: str, user: str, max_tokens: int) -> str:
    """Wrapper unificato: esegue una chat completion, traccia stats, gestisce fallback 429."""
    models_to_try = [CONFIG["model"]]
    fallback = CONFIG.get("fallback_model", "")
    if fallback and fallback != CONFIG["model"]:
        models_to_try.append(fallback)

    last_exc: Optional[Exception] = None
    for attempt, model in enumerate(models_to_try):
        t0 = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=CONFIG["temperature"],
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            LLM_STATS.track(response, time.time() - t0)
            if attempt > 0:
                log.info(f"  [↪ fallback→{model}] chiamata completata")
            return response.choices[0].message.content
        except Exception as e:
            last_exc = e
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower() or "too many" in str(e).lower()
            if is_rate_limit and attempt < len(models_to_try) - 1:
                log.warning(f"  Rate limit su {model}, fallback a {models_to_try[attempt + 1]}...")
                continue
            raise
    raise last_exc  # type: ignore



# ─────────────────────────────────────────────
# CACHE SQLite — evita chiamate API ridondanti
# ─────────────────────────────────────────────

def _init_cache() -> sqlite3.Connection:
    # timeout=30: evita "database is locked" con connessioni concorrenti
    conn = sqlite3.connect(CONFIG["cache_db"], timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_cache (
            key TEXT PRIMARY KEY,
            value TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


def _cache_get(conn: sqlite3.Connection, key: str, ttl_days: Optional[int] = None) -> Optional[str]:
    """Legge dalla cache SQLite. Se ttl_days≠0, ritorna None per entry scadute."""
    effective_ttl = ttl_days if ttl_days is not None else CONFIG.get("cache_ttl_days", 0)
    if effective_ttl and effective_ttl > 0:
        cutoff = (datetime.utcnow() - timedelta(days=effective_ttl)).isoformat()
        row = conn.execute(
            "SELECT value FROM api_cache WHERE key=? AND created_at > ?", (key, cutoff)
        ).fetchone()
    else:
        row = conn.execute("SELECT value FROM api_cache WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _cache_set(conn: sqlite3.Connection, key: str, value: str):
    conn.execute(
        "INSERT OR REPLACE INTO api_cache (key, value, created_at) VALUES (?,?,?)",
        (key, value, datetime.utcnow().isoformat())
    )
    conn.commit()


def _cache_key(*args) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()


# ─────────────────────────────────────────────
# MODELLO DATI
# ─────────────────────────────────────────────

class Paper(TypedDict):
    source: str
    title: str
    authors: list[str]
    year: Optional[int]
    abstract: str
    url: str
    doi: str
    citations: int
    subtopic_id: str
    concepts: list[str]       # keyword accademiche (da OpenAlex)
    affiliations: list[str]   # istituzioni degli autori (da OpenAlex)


# ─────────────────────────────────────────────
# HTTP CON RETRY (tenacity)
# ─────────────────────────────────────────────

def _is_retryable_http(exc: BaseException) -> bool:
    """Restituisce True per errori HTTP transitori (429, 5xx)."""
    if isinstance(exc, requests.HTTPError):
        return exc.response is not None and exc.response.status_code in (
            429, 500, 502, 503, 504
        )
    return False


@retry(
    retry=retry_if_exception(_is_retryable_http),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _http_get(url: str, **kwargs) -> requests.Response:
    """requests.get con retry automatico su errori transitori (429, 5xx)."""
    resp = requests.get(url, **kwargs)
    resp.raise_for_status()
    return resp


# ─────────────────────────────────────────────
# SEMANTIC SCHOLAR — lookup, risoluzione, grafo
# ─────────────────────────────────────────────

_SS_FIELDS = "title,authors,year,abstract,externalIds,citationCount,url"


def _extract_ss_id_from_url(url: str) -> Optional[str]:
    """Estrae l'ID Semantic Scholar (40-char hex) da un URL semanticscholar.org."""
    m = re.search(r'semanticscholar\.org/paper/[^/]+/([a-f0-9]{40})', url)
    if m:
        return m.group(1)
    m = re.search(r'semanticscholar\.org/paper/([a-f0-9]{40})', url)
    if m:
        return m.group(1)
    m = re.search(r'semanticscholar\.org/paper/[^/]+/([A-Za-z0-9]{6,})', url)
    if m:
        return m.group(1)
    return None


def _build_ss_paper_lookup_id(url: str) -> Optional[str]:
    """Costruisce l'ID per l'API SS Graph da un URL: DOI:{...}, ARXIV:{...}, o SS hash."""
    url_lower = url.lower()
    if "doi.org/" in url_lower:
        doi = re.sub(r'^https?://doi\.org/', '', url, flags=re.IGNORECASE).strip()
        if doi:
            return f"DOI:{doi}"
    arxiv_m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9v]+)', url, re.IGNORECASE)
    if arxiv_m:
        return f"ARXIV:{arxiv_m.group(1)}"
    ss_id = _extract_ss_id_from_url(url)
    if ss_id:
        return ss_id
    return None


def _ss_paper_to_normalized(ss_data: dict, subtopic_id: str = "expansion") -> Paper:
    """Converte un paper SS Graph response al formato Paper normalizzato."""
    doi = (ss_data.get("externalIds") or {}).get("DOI", "")
    authors = [a.get("name", "") for a in (ss_data.get("authors") or [])]
    url = ss_data.get("url", "") or (f"https://doi.org/{doi}" if doi else "")
    return _normalize_paper("semantic_scholar", {
        "title": ss_data.get("title", ""),
        "authors": authors,
        "year": ss_data.get("year"),
        "abstract": ss_data.get("abstract", "") or "",
        "url": url,
        "doi": doi,
        "citations": ss_data.get("citationCount", 0) or 0,
        "subtopic_id": subtopic_id,
    })


def resolve_paper_ss(url: str) -> Optional[dict]:
    """Risolve un URL → metadati paper via SS Graph API. Usa cache TTL-free (dati stabili)."""
    lookup_id = _build_ss_paper_lookup_id(url)
    if not lookup_id:
        return None

    conn = _init_cache()
    cache_k = _cache_key("ss_resolve", lookup_id)
    cached = _cache_get(conn, cache_k, ttl_days=0)  # paper metadata non scade
    if cached:
        data = json.loads(cached)
        return data if data else None

    ss_url = f"https://api.semanticscholar.org/graph/v1/paper/{lookup_id}"
    headers = {}
    if CONFIG.get("semantic_scholar_api_key"):
        headers["x-api-key"] = CONFIG["semantic_scholar_api_key"]
    try:
        with _SS_LOCK:
            resp = _http_get(ss_url, params={"fields": _SS_FIELDS}, headers=headers, timeout=15)
            time.sleep(1.0)
        data = resp.json()
        _cache_set(conn, cache_k, json.dumps(data))
        return data
    except Exception as e:
        log.debug(f"  SS resolve fallito per '{lookup_id}': {e}")
        _cache_set(conn, cache_k, json.dumps({}))
        return None


def get_ss_recommendations(ss_paper_id: str) -> list[Paper]:
    """Recupera paper correlati tramite SS Recommendations API."""
    conn = _init_cache()
    cache_k = _cache_key("ss_recommendations", ss_paper_id)
    cached = _cache_get(conn, cache_k, ttl_days=60)
    if cached:
        return json.loads(cached)

    rec_url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{ss_paper_id}"
    headers = {}
    if CONFIG.get("semantic_scholar_api_key"):
        headers["x-api-key"] = CONFIG["semantic_scholar_api_key"]
    try:
        with _SS_LOCK:
            resp = _http_get(
                rec_url,
                params={"fields": _SS_FIELDS, "limit": 10},
                headers=headers,
                timeout=15,
            )
            time.sleep(1.0)
        papers = [
            _ss_paper_to_normalized(p)
            for p in resp.json().get("recommendedPapers", [])
            if p.get("title")
        ]
        _cache_set(conn, cache_k, json.dumps(papers))
        return papers
    except Exception as e:
        log.debug(f"  SS Recommendations fallita per {ss_paper_id}: {e}")
        _cache_set(conn, cache_k, json.dumps([]))
        return []


def fetch_ss_citation_graph(papers: list[Paper], subtopic_id: str = "citation_graph") -> list[Paper]:
    """
    Dato un elenco di paper, recupera le loro REFERENZE via SS batch API.
    Cattura i paper seminali che le keyword-search mancano perché non compaiono
    nelle query ma sono citati dai lavori già trovati.
    """
    max_p = CONFIG.get("citation_graph_max_papers", 8)

    # Costruisce gli SS lookup IDs per i paper con DOI o arXiv
    ids_to_query: list[str] = []
    for p in papers[:max_p]:
        doi = (p.get("doi") or "").strip()
        if doi:
            ids_to_query.append(f"DOI:{doi}")
            continue
        arxiv_m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9v]+)', p.get("url", ""), re.IGNORECASE)
        if arxiv_m:
            ids_to_query.append(f"ARXIV:{arxiv_m.group(1)}")

    if not ids_to_query:
        log.info("  Citation graph: nessun paper con DOI/arXiv ID, skip.")
        return []

    conn = _init_cache()
    cache_k = _cache_key("citation_graph_batch", sorted(ids_to_query))
    cached = _cache_get(conn, cache_k, ttl_days=60)
    if cached:
        return json.loads(cached)

    batch_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    ref_fields = "references.title,references.authors,references.year,references.abstract,references.externalIds,references.citationCount,references.url"
    headers = {"Content-Type": "application/json"}
    if CONFIG.get("semantic_scholar_api_key"):
        headers["x-api-key"] = CONFIG["semantic_scholar_api_key"]

    try:
        with _SS_LOCK:
            resp = requests.post(
                batch_url,
                json={"ids": ids_to_query},
                params={"fields": ref_fields},
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            time.sleep(1.0)
        new_papers: list[Paper] = []
        for entry in resp.json():
            if not entry:
                continue
            for ref in entry.get("references", []):
                if ref.get("title"):
                    new_papers.append(_ss_paper_to_normalized(ref, subtopic_id))
        _cache_set(conn, cache_k, json.dumps(new_papers))
        log.info(f"  Citation graph: {len(new_papers)} referenze da {len(ids_to_query)} paper top")
        return new_papers
    except Exception as e:
        log.warning(f"  Citation graph batch fallito: {e}")
        return []


# ─────────────────────────────────────────────
# STEP 1 — DECOMPOSIZIONE DEL PENSIERO
# ─────────────────────────────────────────────

DECOMPOSE_SYSTEM = """Sei un assistente di ricerca scientifica. Scomponi un macro-argomento in sotto-domande MECE (Mutually Exclusive, Collectively Exhaustive) ottimizzate per database accademici (Arxiv, PubMed, Semantic Scholar, OpenAlex).

Rispondi SOLO con l'oggetto JSON (nessun testo aggiuntivo, nessun ragionamento, nessun blocco di codice):
{
  "macro_topic": "stringa",
  "description": "stringa (2-3 frasi)",
  "subtopics": [
    {"id": "S1", "question": "stringa", "queries": ["query1", "query2", "query3"]}
  ]
}
Genera 4-6 subtopics, 3 query per subtopic. Le query devono essere brevi (3-7 parole)."""


def decompose_thought(client: OpenAI, thought: str) -> dict:
    log.info("STEP 1 — Decomposizione del pensiero...")
    raw = llm_chat(
        client,
        system=DECOMPOSE_SYSTEM,
        user=thought,
        max_tokens=CONFIG["max_tokens_decompose"],
    ).strip()
    # Estrae il JSON in modo robusto: ignora eventuali backtick o testo discorsivo
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"Nessun oggetto JSON trovato nella risposta LLM: {raw[:200]}")
    result = json.loads(match.group())
    log.info(f"  Macro-argomento: {result['macro_topic']}")
    log.info(f"  Sotto-argomenti: {len(result['subtopics'])}")
    return result


# ─────────────────────────────────────────────
# STEP 1b — QUERY REWRITING (subtopic con pochi risultati)
# ─────────────────────────────────────────────

QUERY_REWRITE_SYSTEM = """Sei un esperto di ricerca bibliografica. Una sotto-domanda ha prodotto pochi risultati in database accademici (Arxiv, PubMed, Semantic Scholar, OpenAlex).
Dato il titolo della sotto-domanda e le query usate finora, genera 3 nuove query alternative più specifiche o con sinonimi diversi.
Rispondi SOLO con JSON (nessun testo aggiuntivo): {"queries": ["q1", "q2", "q3"]}
Le query devono essere brevi (3-7 parole), in inglese se possibile."""


def rewrite_queries(client: OpenAI, subtopic: dict, existing_papers: list[Paper]) -> list[str]:
    """Rigenera le query per un subtopic che ha prodotto troppo pochi risultati."""
    sample_titles = [p["title"] for p in existing_papers[:3]]
    prompt = (
        f"SOTTO-DOMANDA: {subtopic['question']}\n"
        f"QUERY GIÀ USATE: {json.dumps(subtopic['queries'])}\n"
        f"RISULTATI TROVATI: {len(existing_papers)} (sotto la soglia)\n"
        f"TITOLI DI ESEMPIO: {json.dumps(sample_titles)}\n\n"
        "Genera 3 nuove query alternative per trovare più paper pertinenti."
    )
    try:
        raw = llm_chat(client, system=QUERY_REWRITE_SYSTEM, user=prompt, max_tokens=200).strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return []
        return json.loads(match.group()).get("queries", [])
    except Exception as e:
        log.debug(f"  Query rewriting fallito: {e}")
        return []


# ─────────────────────────────────────────────
# STEP 2 — RETRIEVAL MULTI-SORGENTE
# ─────────────────────────────────────────────

def _normalize_paper(source: str, data: dict) -> Paper:
    """Schema comune per tutti i paper indipendentemente dalla sorgente."""
    return {
        "source": source,
        "title": (data.get("title") or "").strip(),
        "authors": data.get("authors") or [],
        "year": data.get("year"),
        "abstract": (data.get("abstract") or "").strip(),
        "url": data.get("url") or "",
        "doi": data.get("doi") or "",
        "citations": data.get("citations") or 0,
        "subtopic_id": data.get("subtopic_id") or "",
        "concepts": data.get("concepts") or [],
        "affiliations": data.get("affiliations") or [],
    }


def fetch_arxiv(queries: list[str], subtopic_id: str) -> list[Paper]:
    conn = _init_cache()
    results = []
    for query in queries:
        key = _cache_key("arxiv", query)
        cached = _cache_get(conn, key)
        if cached:
            results.extend(json.loads(cached))
            continue

        search = arxiv.Search(
            query=query,
            max_results=CONFIG["max_results_per_source"],
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for p in search.results():
            papers.append(_normalize_paper("arxiv", {
                "title": p.title,
                "authors": [str(a) for a in p.authors],
                "year": p.published.year,
                "abstract": p.summary,
                "url": p.entry_id,
                "doi": p.doi or "",
                "citations": 0,
                "subtopic_id": subtopic_id,
            }))
        _cache_set(conn, key, json.dumps(papers))
        results.extend(papers)
        time.sleep(0.3)  # rate limit
    return results


def fetch_pubmed(queries: list[str], subtopic_id: str) -> list[Paper]:
    conn = _init_cache()
    results = []
    for query in queries:
        key = _cache_key("pubmed", query)
        cached = _cache_get(conn, key)
        if cached:
            results.extend(json.loads(cached))
            continue

        try:
            handle = Entrez.esearch(
                db="pubmed", term=query,
                retmax=CONFIG["max_results_per_source"], sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            ids = record["IdList"]
            if not ids:
                continue

            fetch_handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml", retmode="xml")
            records = Entrez.read(fetch_handle)
            fetch_handle.close()

            papers = []
            for art in records.get("PubmedArticle", []):
                med = art["MedlineCitation"]
                article = med["Article"]
                title = str(article.get("ArticleTitle", ""))
                abstract = ""
                if "Abstract" in article:
                    texts = article["Abstract"].get("AbstractText", [])
                    abstract = " ".join(str(t) for t in texts)
                year = None
                if "Journal" in article:
                    ji = article["Journal"].get("JournalIssue", {})
                    pd = ji.get("PubDate", {})
                    y = pd.get("Year", pd.get("MedlineDate", ""))
                    try:
                        year = int(str(y)[:4])
                    except (ValueError, TypeError):
                        year = None
                authors = []
                for a in article.get("AuthorList", []):
                    ln = str(a.get("LastName", ""))
                    fn = str(a.get("ForeName", ""))
                    if ln:
                        authors.append(f"{ln} {fn}".strip())
                pmid = str(med["PMID"])
                papers.append(_normalize_paper("pubmed", {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "abstract": abstract,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "doi": "",
                    "citations": 0,
                    "subtopic_id": subtopic_id,
                }))
            _cache_set(conn, key, json.dumps(papers))
            results.extend(papers)
        except Exception as e:
            log.warning(f"  PubMed error per '{query}': {e}")
        time.sleep(0.5)
    return results


# Serializza le richieste a Semantic Scholar per evitare burst di 429
_SS_LOCK = threading.Semaphore(1)


def fetch_semantic_scholar(queries: list[str], subtopic_id: str) -> list[Paper]:
    conn = _init_cache()
    results = []
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,authors,year,abstract,externalIds,citationCount,url"
    headers = {}
    if CONFIG["semantic_scholar_api_key"]:
        headers["x-api-key"] = CONFIG["semantic_scholar_api_key"]

    for query in queries:
        key = _cache_key("semantic", query)
        cached = _cache_get(conn, key)
        if cached:
            results.extend(json.loads(cached))
            continue

        try:
            with _SS_LOCK:
                resp = _http_get(base_url, params={
                    "query": query,
                    "limit": CONFIG["max_results_per_source"],
                    "fields": fields,
                }, headers=headers, timeout=15)
                time.sleep(1.0)  # rispetta il rate limit (100 req/min senza API key)
            data = resp.json()
            papers = []
            for p in data.get("data", []):
                doi = (p.get("externalIds") or {}).get("DOI", "")
                papers.append(_normalize_paper("semantic_scholar", {
                    "title": p.get("title"),
                    "authors": [a.get("name", "") for a in (p.get("authors") or [])],
                    "year": p.get("year"),
                    "abstract": p.get("abstract"),
                    "url": p.get("url"),
                    "doi": doi,
                    "citations": p.get("citationCount"),
                    "subtopic_id": subtopic_id,
                }))
            _cache_set(conn, key, json.dumps(papers))
            results.extend(papers)
        except Exception as e:
            log.warning(f"  SemanticScholar error per '{query}': {e}")
    return results


def fetch_openalex(queries: list[str], subtopic_id: str) -> list[Paper]:
    conn = _init_cache()
    if CONFIG["openalex_email"]:
        pyalex.config.email = CONFIG["openalex_email"]
    results = []
    for query in queries:
        key = _cache_key("openalex", query)
        cached = _cache_get(conn, key)
        if cached:
            results.extend(json.loads(cached))
            continue

        try:
            works = (
                Works()
                .search(query)
                .select(["title", "authorships", "publication_year",
                         "abstract_inverted_index", "doi",
                         "cited_by_count", "id", "concepts"])
                .get(per_page=CONFIG["max_results_per_source"])
            )
            papers = []
            for w in works:
                # OpenAlex usa abstract_inverted_index → ricostruisce testo
                inv = w.get("abstract_inverted_index") or {}
                if inv:
                    words = sorted(
                        [(pos, word) for word, positions in inv.items() for pos in positions]
                    )
                    abstract = " ".join(w for _, w in words)
                else:
                    abstract = ""
                authors = [
                    a["author"].get("display_name", "")
                    for a in w.get("authorships", [])
                ]
                doi = w.get("doi", "") or ""
                url = doi if doi.startswith("http") else (f"https://doi.org/{doi}" if doi else w.get("id", ""))
                concepts = [
                    c.get("display_name", "")
                    for c in (w.get("concepts") or [])[:5]
                    if c.get("display_name")
                ]
                inst_set: list[str] = []
                for a in w.get("authorships", []):
                    for inst in a.get("institutions", []):
                        name = inst.get("display_name", "")
                        if name and name not in inst_set:
                            inst_set.append(name)
                papers.append(_normalize_paper("openalex", {
                    "title": w.get("title", ""),
                    "authors": authors,
                    "year": w.get("publication_year"),
                    "abstract": abstract,
                    "url": url,
                    "doi": doi,
                    "citations": w.get("cited_by_count", 0),
                    "subtopic_id": subtopic_id,
                    "concepts": concepts,
                    "affiliations": inst_set,
                }))
            _cache_set(conn, key, json.dumps(papers))
            results.extend(papers)
        except Exception as e:
            log.warning(f"  OpenAlex error per '{query}': {e}")
        time.sleep(0.3)
    return results


def retrieve_all_sources(decomposed: dict, client: Optional[OpenAI] = None) -> list[Paper]:
    """
    Retrieval parallelo da tutte le sorgenti.
    Se `client` è fornito e `enable_query_rewrite` è True, riscrive le query per
    i subtopic che producono meno di `min_papers_per_subtopic` risultati.
    """
    log.info("STEP 2 — Retrieval parallelo da tutte le sorgenti...")
    all_papers: list[Paper] = []
    papers_by_subtopic: dict[str, list[Paper]] = {}

    def _fetch_subtopic(subtopic: dict) -> list[Paper]:
        sid = subtopic["id"]
        queries = subtopic["queries"]
        sub_papers: list[Paper] = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [
                ex.submit(fetch_arxiv,            queries, sid),
                ex.submit(fetch_pubmed,           queries, sid),
                ex.submit(fetch_semantic_scholar, queries, sid),
                ex.submit(fetch_openalex,         queries, sid),
            ]
            for f in as_completed(futs):
                try:
                    sub_papers.extend(f.result())
                except Exception as e:
                    log.warning(f"  Task retrieval fallito ({sid}): {e}")
        return sub_papers

    # Prima passata
    with ThreadPoolExecutor(max_workers=4) as executor:
        fut_map = {executor.submit(_fetch_subtopic, st): st for st in decomposed["subtopics"]}
        for future in as_completed(fut_map):
            st = fut_map[future]
            try:
                results = future.result()
                papers_by_subtopic[st["id"]] = results
                all_papers.extend(results)
            except Exception as e:
                log.warning(f"  Subtopic {st['id']} fallito: {e}")

    # Query rewriting per subtopic deficitari
    if client and CONFIG.get("enable_query_rewrite", False):
        threshold = CONFIG.get("min_papers_per_subtopic", 3)
        for st in decomposed["subtopics"]:
            sid = st["id"]
            current = papers_by_subtopic.get(sid, [])
            unique_current = list({p["title"]: p for p in current}.values())
            if len(unique_current) < threshold:
                log.info(f"  Query rewriting per '{st['id']}' ({len(unique_current)} paper trovati)...")
                new_queries = rewrite_queries(client, st, unique_current)
                if new_queries:
                    rewritten = _fetch_subtopic({"id": sid, "queries": new_queries})
                    all_papers.extend(rewritten)
                    log.info(f"    +{len(rewritten)} paper con query riscritte")

    log.info(f"  Paper grezzi recuperati: {len(all_papers)}")
    return all_papers


# ─────────────────────────────────────────────
# STEP 3 — DEDUPLICAZIONE + SCORING
# ─────────────────────────────────────────────

def deduplicate(papers: list[Paper]) -> list[Paper]:
    """
    Deduplicazione a tre livelli, in ordine di priorità:
      1. DOI (massima affidabilità)
      2. arXiv ID (estratto dall'URL)
      3. Titolo normalizzato (fallback)
    Quando ci sono duplicati, mantiene quello con più citazioni.
    """
    seen: dict[str, Paper] = {}
    for p in papers:
        # Canonical ID: DOI > arXiv > titolo
        doi = (p.get("doi") or "").strip().lower()
        if doi:
            cid = f"doi:{doi}"
        else:
            arxiv_m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9v]+)', p.get("url", ""), re.IGNORECASE)
            if arxiv_m:
                cid = f"arxiv:{arxiv_m.group(1)}"
            else:
                key = "".join(c.lower() for c in p["title"] if c.isalnum())
                if not key:
                    continue
                cid = f"title:{key}"

        if cid not in seen or p["citations"] > seen[cid]["citations"]:
            seen[cid] = p

    result = list(seen.values())
    log.info(f"  Dopo deduplicazione: {len(result)} paper unici")
    return result


def filter_by_year(papers: list[Paper]) -> list[Paper]:
    if CONFIG["min_year"] == 0:
        return papers
    filtered = [p for p in papers if p.get("year") and p["year"] >= CONFIG["min_year"]]
    log.info(f"  Dopo filtro anno (>={CONFIG['min_year']}): {len(filtered)} paper")
    return filtered


def score_papers(papers: list[Paper], macro_topic: str) -> list[Paper]:
    """
    Score composito:
      - citation_score  (log-normalizzato 0-1, robusto agli outlier)
      - recency_score   (anno più recente = score maggiore)
      - semantic_score  (similarità embedding con macro_topic)
    """
    log.info("STEP 3 — Scoring e ranking...")
    if not papers:
        return []

    # Embedding semantico
    topic_emb = _EMBEDDING_MODEL.encode(macro_topic, convert_to_tensor=True)
    texts = [f"{p['title']} {p['abstract'][:300]}" for p in papers]
    paper_embs = _EMBEDDING_MODEL.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    semantic_scores = util.cos_sim(topic_emb, paper_embs)[0].tolist()

    # Normalizzazione citazioni logaritmica (robusta agli outlier)
    max_cit = max((p["citations"] for p in papers), default=1) or 1
    current_year = datetime.now().year
    max_age = max((current_year - (p["year"] or current_year) for p in papers), default=1) or 1

    for i, p in enumerate(papers):
        cit_score = math.log1p(p["citations"]) / math.log1p(max_cit)
        age = current_year - (p["year"] or current_year)
        rec_score = 1.0 - (age / (max_age + 1))
        sem_score = float(semantic_scores[i])

        p["score"] = (
            CONFIG["weight_citations"] * cit_score +
            CONFIG["weight_recency"]   * rec_score +
            CONFIG["weight_semantic"]  * sem_score
        )
        p["score_detail"] = {
            "citation": round(cit_score, 3),
            "recency":  round(rec_score, 3),
            "semantic": round(sem_score, 3),
        }

    papers.sort(key=lambda x: x["score"], reverse=True)
    log.info(f"  Top score: {papers[0]['score']:.3f} — '{papers[0]['title'][:60]}...'")
    return papers


def mmr_select(papers: list[Paper], k: int, lambda_mmr: float = 0.6) -> list[Paper]:
    """
    Maximal Marginal Relevance: seleziona k paper bilanciando rilevanza e diversità.
    Evita che i top-K siano tutti paper quasi identici sullo stesso sotto-aspetto.
    lambda_mmr=1.0 → solo rilevanza (identico a sort by score)
    lambda_mmr=0.0 → solo diversità
    """
    if len(papers) <= k:
        return papers

    texts = [f"{p['title']} {p['abstract'][:200]}" for p in papers]
    embs = _EMBEDDING_MODEL.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    scores = [p.get("score", 0.0) for p in papers]

    selected: list[int] = []
    remaining = list(range(len(papers)))

    # Primo elemento: il paper con score più alto
    best_first = max(remaining, key=lambda i: scores[i])
    selected.append(best_first)
    remaining.remove(best_first)

    while len(selected) < k and remaining:
        best_mmr, best_i = -1.0, -1
        for i in remaining:
            rel = scores[i]
            max_sim = max(
                float(util.cos_sim(embs[i].unsqueeze(0), embs[j].unsqueeze(0))[0][0])
                for j in selected
            )
            mmr = lambda_mmr * rel - (1 - lambda_mmr) * max_sim
            if mmr > best_mmr:
                best_mmr, best_i = mmr, i
        if best_i < 0:
            break
        selected.append(best_i)
        remaining.remove(best_i)

    return [papers[i] for i in selected]


# ─────────────────────────────────────────────
# STEP 4 — SINTESI FINALE CON CITAZIONI
# ─────────────────────────────────────────────

SYNTHESIZE_SYSTEM = """Sei un ricercatore accademico senior. 
Ricevi un macro-argomento, la sua scomposizione in sotto-domande, 
e una lista di paper scientifici recuperati da database accademici.

Il tuo compito è produrre un documento strutturato in Markdown che:
1. Introduce e formalizza il macro-argomento
2. Per ogni sotto-domanda, sviluppa un paragrafo argomentativo basato sui paper forniti
3. Ogni claim deve essere supportato da almeno una citazione nel formato [Autore et al., Anno](URL)
4. Conclude con una sintesi delle evidenze e possibili direzioni future
5. Termina con una bibliografia completa

IMPORTANTE:
- Cita SOLO i paper forniti, non inventare riferimenti
- Se un sotto-argomento ha poche fonti, segnalalo esplicitamente
- Usa un tono accademico formale ma chiaro
- Non aggiungere informazioni non supportate dai paper"""


def build_synthesis_prompt(decomposed: dict, top_papers: list[Paper]) -> str:
    abs_chars = CONFIG["abstract_chars_simple"]
    papers_block = json.dumps([
        {
            "id": i + 1,
            "title": p["title"],
            "authors": p["authors"][:3],
            "year": p["year"],
            "abstract": p["abstract"][:abs_chars],
            "url": p["url"],
            "citations": p["citations"],
            "subtopic_id": p["subtopic_id"],
        }
        for i, p in enumerate(top_papers)
    ], ensure_ascii=False)

    subtopics_min = [
        {"id": s["id"], "question": s["question"]}
        for s in decomposed["subtopics"]
    ]

    return (
        f"MACRO-ARGOMENTO: {decomposed['macro_topic']}\n"
        f"DESCRIZIONE: {decomposed['description']}\n\n"
        f"SOTTO-DOMANDE:\n{json.dumps(subtopics_min, ensure_ascii=False)}\n\n"
        f"PAPER:\n{papers_block}\n\n"
        "Produci il documento strutturato richiesto."
    )


def synthesize(client: OpenAI, decomposed: dict, papers: list[Paper]) -> str:
    log.info("STEP 4 — Sintesi finale...")

    # Usa i top-N paper (bilanciati per sotto-argomento)
    top_per_subtopic = {}
    for p in papers:
        sid = p.get("subtopic_id", "unknown")
        top_per_subtopic.setdefault(sid, [])
        if len(top_per_subtopic[sid]) < 5:
            top_per_subtopic[sid].append(p)

    top_papers = [p for group in top_per_subtopic.values() for p in group]
    log.info(f"  Paper usati per sintesi: {len(top_papers)}")

    return llm_chat(
        client,
        system=SYNTHESIZE_SYSTEM,
        user=build_synthesis_prompt(decomposed, top_papers),
        max_tokens=CONFIG["max_tokens_synthesize"],
    )


# ─────────────────────────────────────────────
# STEP 4b — SINTESI GERARCHICA
# ─────────────────────────────────────────────

OUTLINE_SYSTEM = """Sei un ricercatore accademico senior specializzato nell'organizzazione di survey scientifiche.
Dato un macro-argomento, le sue sotto-domande e una lista di paper, progetta la struttura ottimale
per un documento accademico con 5-7 sezioni tematiche.

Assegna a ogni sezione i paper più rilevanti usando il campo "id" dei paper.
I paper possono apparire in più sezioni se rilevanti per entrambe.

Rispondi SOLO con l'oggetto JSON (nessun testo aggiuntivo):
{
  "sections": [
    {
      "id": "SEC1",
      "title": "Titolo della sezione",
      "focus": "Breve descrizione del focus (1 frase)",
      "paper_ids": [1, 3, 7]
    }
  ]
}"""

SECTION_SYSTEM = """Sei un ricercatore accademico senior. Scrivi una sezione completa di una survey scientifica.
La sezione deve:
- Essere di 800-1000 parole
- Ogni claim deve essere supportato da una citazione nel formato [Autore et al., Anno](URL)
- Confrontare e mettere in dialogo le fonti, non limitarsi a elencarle
- Usare un tono accademico formale e preciso
- Citare SOLO i paper forniti, senza inventare riferimenti
- Concludere con una sintesi critica della sezione"""

REFLECT_SYSTEM = """Sei un revisore accademico critico. Analizza la bozza di una sezione scientifica e migliora:
1. Il rigore nel confronto tra le fonti (sono effettivamente messe in dialogo?)
2. La correttezza delle citazioni (ogni citazione esiste nella lista paper fornita?)
3. Il tono accademico (evita generalizzazioni, mantieni precisione scientifica)
4. La chiarezza della struttura argomentativa

Restituisci la sezione migliorata in Markdown, mantenendo approssimativamente la stessa lunghezza."""


def generate_outline(client: OpenAI, decomposed: dict, papers: list) -> dict:
    log.info("  Generazione indice strutturato (outline)...")
    # Outline: solo titolo + anno + subtopic. Niente abstract/concepts.
    papers_summary = json.dumps([
        {
            "id": p["_outline_id"],
            "title": p["title"],
            "year": p["year"],
            "subtopic_id": p["subtopic_id"],
        }
        for p in papers
    ], ensure_ascii=False)

    subtopics_min = [
        {"id": s["id"], "question": s["question"]}
        for s in decomposed["subtopics"]
    ]

    prompt = (
        f"MACRO-ARGOMENTO: {decomposed['macro_topic']}\n"
        f"SOTTO-DOMANDE:\n{json.dumps(subtopics_min, ensure_ascii=False)}\n\n"
        f"PAPER (id, title, year, subtopic_id):\n{papers_summary}\n\n"
        "Progetta la struttura del documento accademico (5-7 sezioni)."
    )
    raw = llm_chat(
        client,
        system=OUTLINE_SYSTEM,
        user=prompt,
        max_tokens=CONFIG.get("max_tokens_outline", 1500),
    ).strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"Nessun JSON trovato nella risposta outline: {raw[:200]}")
    outline = json.loads(match.group())
    log.info(f"  Indice generato: {len(outline['sections'])} sezioni")
    return outline


def write_section(client: OpenAI, section: dict, papers: list, macro_topic: str) -> str:
    log.info(f"  Scrittura sezione: '{section['title']}'")
    abs_chars = CONFIG["abstract_chars_synthesis"]
    papers_block = json.dumps([
        {
            "id": p["_outline_id"],
            "title": p["title"],
            "authors": p["authors"][:3],
            "year": p["year"],
            "abstract": p["abstract"][:abs_chars],
            "url": p["url"],
            "citations": p["citations"],
        }
        for p in papers
    ], ensure_ascii=False)

    prompt = (
        f"MACRO-ARGOMENTO: {macro_topic}\n"
        f"SEZIONE: {section['title']}\n"
        f"FOCUS: {section['focus']}\n\n"
        f"PAPER:\n{papers_block}\n\n"
        "Scrivi la sezione completa."
    )
    return llm_chat(
        client,
        system=SECTION_SYSTEM,
        user=prompt,
        max_tokens=CONFIG.get("max_tokens_section", 2500),
    )


def reflect_on_section(client: OpenAI, section_title: str, draft: str, papers: list) -> str:
    log.info(f"  Self-reflection su: '{section_title}'")
    citations_list = "\n".join(
        f"- [{(p['authors'][0].split()[-1] if p['authors'] else 'N.A.')} et al., {p['year']}]({p['url']})"
        for p in papers
    )
    prompt = (
        f"SEZIONE: {section_title}\n\n"
        f"CITAZIONI VALIDE:\n{citations_list}\n\n"
        f"BOZZA:\n{draft}\n\n"
        "Revisiona e migliora la sezione."
    )
    return llm_chat(
        client,
        system=REFLECT_SYSTEM,
        user=prompt,
        max_tokens=CONFIG.get("max_tokens_section", 2500),
    )


def synthesize_hierarchical(client: OpenAI, decomposed: dict, papers: list[Paper]) -> tuple[dict, str]:
    """Sintesi gerarchica: outline → scrittura per sezione → self-reflection → merge."""
    log.info("STEP 4 — Sintesi gerarchica...")

    # Bilancia i paper per sotto-argomento (come synthesize originale)
    top_per_subtopic: dict[str, list] = {}
    for p in papers:
        sid = p.get("subtopic_id", "unknown")
        top_per_subtopic.setdefault(sid, [])
        if len(top_per_subtopic[sid]) < 5:
            top_per_subtopic[sid].append(p)
    top_papers = [p for group in top_per_subtopic.values() for p in group]

    # Aggiunge un ID progressivo per il mapping dell'outline (campo temporaneo)
    indexed: list[dict] = []
    for i, p in enumerate(top_papers):
        p_copy = dict(p)
        p_copy["_outline_id"] = i + 1
        indexed.append(p_copy)
    log.info(f"  Paper usati: {len(indexed)}")

    # Step 4a: genera l'indice strutturato
    outline = generate_outline(client, decomposed, indexed)

    # Step 4b + 4c: scrivi e (opzionalmente) rifletti su ogni sezione
    paper_by_id = {p["_outline_id"]: p for p in indexed}
    section_drafts: list[tuple[str, str]] = []
    do_reflect = CONFIG.get("enable_reflection", False)
    for section in outline["sections"]:
        assigned = [paper_by_id[pid] for pid in section["paper_ids"] if pid in paper_by_id]
        if not assigned:
            log.warning(f"  Nessun paper assegnato a '{section['title']}', sezione saltata.")
            continue
        draft = write_section(client, section, assigned, decomposed["macro_topic"])
        if do_reflect:
            draft = reflect_on_section(client, section["title"], draft, assigned)
        section_drafts.append((section["title"], draft))

    # Step 4d: merge finale in Markdown
    parts = [f"## {title}\n\n{content}" for title, content in section_drafts]
    full_synthesis = "\n\n---\n\n".join(parts)
    return outline, full_synthesis


# ─────────────────────────────────────────────
# VALIDATORE CITAZIONI
# ─────────────────────────────────────────────

def validate_citations(synthesis_text: str) -> list[dict]:
    """
    Scansiona tutte le citazioni [testo](url) nel documento e le verifica:
    - Per URL con DOI/arXiv/SS: risoluzione via SS Graph API
    - Per altri URL (PubMed etc.): HTTP HEAD check
    Restituisce lista di {url, display, status, resolved_title}
    """
    pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
    results: list[dict] = []
    urls_seen: set[str] = set()

    for display, url in pattern.findall(synthesis_text):
        url = url.rstrip(".,;)")
        if url in urls_seen:
            continue
        urls_seen.add(url)

        lookup_id = _build_ss_paper_lookup_id(url)
        if lookup_id:
            ss_data = resolve_paper_ss(url)
            if ss_data and ss_data.get("title"):
                results.append({"url": url, "display": display, "status": "verified", "resolved_title": ss_data["title"]})
            else:
                results.append({"url": url, "display": display, "status": "ss_unresolved", "resolved_title": None})
        else:
            try:
                r = requests.head(url, timeout=5, allow_redirects=True)
                status = "ok" if r.status_code < 400 else f"http_{r.status_code}"
            except Exception:
                status = "unreachable"
            results.append({"url": url, "display": display, "status": status, "resolved_title": None})

    verified = sum(1 for r in results if r["status"] in ("verified", "ok"))
    bad = sum(1 for r in results if r["status"] not in ("verified", "ok"))
    log.info(f"  Citazioni: {len(results)} totali — {verified} verificate, {bad} non risolte")
    if bad:
        for r in results:
            if r["status"] not in ("verified", "ok"):
                log.warning(f"    ⚠ [{r['status']}] {r['url'][:80]}")
    return results


def citation_validation_md(results: list[dict]) -> str:
    """Formatta il report di validazione come blocco Markdown."""
    if not results:
        return ""
    lines = ["\n\n---\n\n## ⚠ Report Validazione Citazioni\n"]
    bad = [r for r in results if r["status"] not in ("verified", "ok")]
    if not bad:
        lines.append(f"Tutte le {len(results)} citazioni sono state verificate.\n")
    else:
        lines.append(f"**{len(bad)}/{len(results)} citazioni non risolte:**\n")
        for r in bad:
            lines.append(f"- `[{r['status']}]` {r['display']} → {r['url']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# STEP 5 — SALVATAGGIO RUN
# ─────────────────────────────────────────────

def save_run(thought: str, decomposed: dict, papers: list[Paper], synthesis: str,
             outline: Optional[dict] = None, citation_report: Optional[list] = None) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(CONFIG["output_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Trace completa JSONL
    trace = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "config": CONFIG,
        "input": thought,
        "decomposed": decomposed,
        "papers_count": len(papers),
        "top_papers": papers[:20],
        "llm_stats": LLM_STATS.to_dict(),
        "citation_report": citation_report or [],
    }
    with open(run_dir / "trace.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    # Tutti i paper recuperati
    with open(run_dir / "papers_all.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)

    # Outline strutturato (solo in modalità sintesi gerarchica)
    if outline is not None:
        with open(run_dir / "outline.json", "w", encoding="utf-8") as f:
            json.dump(outline, f, ensure_ascii=False, indent=2)

    # Output finale Markdown con metadati di generazione
    md_path = run_dir / "synthesis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {decomposed['macro_topic']}\n\n")
        f.write(f"*Run ID: {run_id} — Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("> **Metadati generazione**\n>\n")
        for line in LLM_STATS.summary_md(CONFIG["model"]).split("\n"):
            f.write(f"> {line}\n")
        f.write(f"> - **Paper analizzati**: {len(papers)}\n")
        f.write("\n---\n\n")
        f.write(synthesis)
        # Appendice validazione citazioni (se ci sono problemi)
        if citation_report:
            bad = [r for r in citation_report if r["status"] not in ("verified", "ok")]
            if bad:
                f.write(citation_validation_md(citation_report))

    log.info(f"  Run salvata in: {run_dir}")
    return md_path


# ─────────────────────────────────────────────
# PIPELINE PRINCIPALE
# ─────────────────────────────────────────────

def run_pipeline(thought: str) -> Path:
    """
    Entry point principale.
    Riceve il pensiero grezzo dell'utente e restituisce il path del documento finale.
    """
    log.info("=" * 60)
    log.info("AVVIO PIPELINE DI RICERCA")
    log.info(f"Input: {thought[:100]}...")
    log.info("=" * 60)
    LLM_STATS.start_run()

    # GitHub Copilot espone un'API OpenAI-compatibile.
    # Richiede un token OAuth GitHub (NON un PAT classico).
    # Ottienilo con: gh auth token  (richiede GitHub CLI)
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise SystemExit(
            "Errore: variabile GITHUB_TOKEN non impostata.\n"
            "Ottieni il token con: gh auth token\n"
            "Poi esegui: $env:GITHUB_TOKEN = (gh auth token)"
        )
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )

    # Step 1: decomposizione
    decomposed = decompose_thought(client, thought)

    # Step 2: retrieval con retry se risultati insufficienti
    all_papers: list[Paper] = []
    for cycle in range(CONFIG["max_retry_cycles"] + 1):
        papers = retrieve_all_sources(decomposed, client=client)
        all_papers.extend(papers)

        unique = deduplicate(all_papers)
        filtered = filter_by_year(unique)

        if len(filtered) >= CONFIG["min_total_papers"]:
            all_papers = filtered
            break

        if cycle < CONFIG["max_retry_cycles"]:
            log.warning(f"  Solo {len(filtered)} paper trovati, ciclo {cycle+1} — allargo le query...")
            CONFIG["min_year"] = max(0, CONFIG["min_year"] - 5)
    else:
        all_papers = deduplicate(all_papers)
        log.warning(f"  Risultati finali dopo retry: {len(all_papers)} paper")

    # Step 2b: Citation graph expansion (recupera referenze dei paper top)
    if CONFIG.get("enable_citation_graph", False) and all_papers:
        log.info("STEP 2b — Citation graph expansion...")
        # Prima uno scoring veloce per identificare i paper più importanti
        temp_scored = score_papers(list(all_papers), decomposed["macro_topic"])
        graph_papers = fetch_ss_citation_graph(temp_scored, subtopic_id="citation_graph")
        if graph_papers:
            combined = deduplicate(all_papers + graph_papers)
            combined = filter_by_year(combined)
            log.info(f"  Pool dopo citation graph: {len(combined)} paper (+{len(combined)-len(all_papers)})")
            all_papers = combined

    # Step 3: scoring + MMR
    scored_papers = score_papers(all_papers, decomposed["macro_topic"])
    if CONFIG.get("enable_mmr", False) and len(scored_papers) > CONFIG.get("mmr_top_k", 25):
        top_k = CONFIG["mmr_top_k"]
        lambda_mmr = CONFIG.get("mmr_lambda", 0.6)
        log.info(f"  MMR: selezione {top_k} paper diversificati (λ={lambda_mmr})...")
        scored_papers = mmr_select(scored_papers, k=top_k, lambda_mmr=lambda_mmr)

    # Step 4: sintesi
    outline: Optional[dict] = None
    if CONFIG.get("hierarchical_synthesis", False):
        outline, synthesis = synthesize_hierarchical(client, decomposed, scored_papers)
    else:
        synthesis = synthesize(client, decomposed, scored_papers)

    # Step 4b: validazione citazioni
    citation_report: Optional[list] = None
    if CONFIG.get("enable_citation_validation", False):
        log.info("STEP 4b — Validazione citazioni...")
        citation_report = validate_citations(synthesis)

    # Step 5: salvataggio
    output_path = save_run(thought, decomposed, scored_papers, synthesis,
                           outline=outline, citation_report=citation_report)

    log.info("=" * 60)
    log.info(f"PIPELINE COMPLETATA — Output: {output_path}")
    log.info(f"  LLM: {LLM_STATS.calls} chiamate, {LLM_STATS.total_tokens:,} token, {LLM_STATS.total_seconds:.1f}s")
    log.info(f"  Tempo totale: {LLM_STATS.wall_seconds:.1f}s")
    log.info("=" * 60)

    return output_path


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline deterministica di ricerca scientifica.",
        add_help=False,
    )
    parser.add_argument("--reflect", action="store_true",
                        help="Abilita self-reflection LLM su ogni sezione (raddoppia chiamate).")
    parser.add_argument("--no-hierarchical", action="store_true",
                        help="Usa la sintesi flat (1 sola chiamata LLM) invece di outline+sezioni.")
    parser.add_argument("--no-citation-graph", action="store_true",
                        help="Disabilita citation graph expansion.")
    parser.add_argument("--no-mmr", action="store_true",
                        help="Disabilita MMR (Maximal Marginal Relevance) per la selezione paper.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Disabilita la validazione delle citazioni post-synthesis.")
    parser.add_argument("--no-query-rewrite", action="store_true",
                        help="Disabilita il query rewriting per subtopic deficitari.")
    parser.add_argument("--list", action="store_true",
                        help="Mostra tutte le run esistenti con statistiche.")
    parser.add_argument("-h", "--help", action="store_true")
    args, remaining = parser.parse_known_args()

    if args.help:
        parser.print_help()
        print('\nUso: python research_pipeline.py [opzioni] "il tuo pensiero"')
        sys.exit(0)

    if args.list:
        import glob, textwrap
        runs_dir = Path(CONFIG["output_dir"])
        run_dirs = sorted(runs_dir.glob("*/trace.jsonl"), reverse=True)
        if not run_dirs:
            print("Nessuna run trovata.")
            sys.exit(0)
        print(f"\n{'Run ID':<20} {'Topic':<50} {'Paper':>7} {'Token':>8} {'Tempo':>6}")
        print("-" * 95)
        for t_path in run_dirs:
            try:
                trace = json.loads(t_path.read_text(encoding="utf-8").split("\n")[0])
                rid = trace.get("run_id", "?")
                topic = textwrap.shorten(trace.get("decomposed", {}).get("macro_topic", "?"), 49)
                n_papers = trace.get("papers_count", "?")
                stats = trace.get("llm_stats", {})
                tokens = f"{stats.get('total_tokens', 0):,}"
                wall = f"{stats.get('wall_seconds', 0):.0f}s"
                print(f"{rid:<20} {topic:<50} {str(n_papers):>7} {tokens:>8} {wall:>6}")
            except Exception:
                print(f"{t_path.parent.name:<20} (trace non leggibile)")
        sys.exit(0)

    if args.reflect:
        CONFIG["enable_reflection"] = True
    if args.no_hierarchical:
        CONFIG["hierarchical_synthesis"] = False
    if args.no_citation_graph:
        CONFIG["enable_citation_graph"] = False
    if args.no_mmr:
        CONFIG["enable_mmr"] = False
    if args.no_validate:
        CONFIG["enable_citation_validation"] = False
    if args.no_query_rewrite:
        CONFIG["enable_query_rewrite"] = False

    if remaining:
        user_thought = " ".join(remaining)
    else:
        print("Inserisci il pensiero da espandere (INVIO x2 per confermare):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        user_thought = " ".join(lines)

    if not user_thought.strip():
        print("Errore: nessun input fornito.")
        sys.exit(1)

    output = run_pipeline(user_thought)
    print(f"\n✓ Documento generato: {output}")
    print(f"  Aprilo con: code {output}")
