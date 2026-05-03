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
from datetime import datetime
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
    "hierarchical_synthesis": True,   # True = outline + sezioni + self-reflection
    "max_tokens_outline": 1500,        # token per la generazione dell'indice
    "max_tokens_section": 2500,        # token per la scrittura di ogni sezione
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


def _cache_get(conn: sqlite3.Connection, key: str) -> Optional[str]:
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
# STEP 1 — DECOMPOSIZIONE DEL PENSIERO
# ─────────────────────────────────────────────

DECOMPOSE_SYSTEM = """Sei un assistente di ricerca scientifica esperto nell'analisi di problemi complessi.

Usa i tag <analisi> per il tuo ragionamento interno prima di produrre il JSON:
<analisi>
Genera 3 approcci alternativi per scomporre il problema:
- APPROCCIO A: focalizzato sugli aspetti metodologici e tecnici
- APPROCCIO B: focalizzato sulle applicazioni pratiche e casi d'uso reali
- APPROCCIO C: focalizzato sui fondamenti teorici e framework concettuali
Seleziona l'approccio (o la combinazione) più adatta ai database accademici
(Arxiv, PubMed, Semantic Scholar, OpenAlex).
</analisi>

Le sotto-domande devono essere MECE (Mutually Exclusive, Collectively Exhaustive):
nessuna sovrapposizione, copertura completa del macro-argomento.

Dopo il tag </analisi>, scrivi SOLO l'oggetto JSON (nessun altro testo):
{
  "macro_topic": "stringa",
  "description": "stringa (2-3 frasi)",
  "subtopics": [
    {
      "id": "S1",
      "question": "stringa",
      "queries": ["query1", "query2", "query3"]
    }
  ]
}"""


def decompose_thought(client: OpenAI, thought: str) -> dict:
    log.info("STEP 1 — Decomposizione del pensiero...")
    response = client.chat.completions.create(
        model=CONFIG["model"],
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG["max_tokens_decompose"],
        messages=[
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user", "content": thought},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Estrae il JSON in modo robusto: ignora scratchpad <analisi>, backtick e testo discorsivo
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"Nessun oggetto JSON trovato nella risposta LLM: {raw[:200]}")
    result = json.loads(match.group())
    log.info(f"  Macro-argomento: {result['macro_topic']}")
    log.info(f"  Sotto-argomenti: {len(result['subtopics'])}")
    return result


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


def retrieve_all_sources(decomposed: dict) -> list[Paper]:
    log.info("STEP 2 — Retrieval parallelo da tutte le sorgenti...")
    all_papers = []

    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for subtopic in decomposed["subtopics"]:
            sid = subtopic["id"]
            queries = subtopic["queries"]
            tasks.append(executor.submit(fetch_arxiv,            queries, sid))
            tasks.append(executor.submit(fetch_pubmed,           queries, sid))
            tasks.append(executor.submit(fetch_semantic_scholar, queries, sid))
            tasks.append(executor.submit(fetch_openalex,         queries, sid))

        for future in as_completed(tasks):
            try:
                all_papers.extend(future.result())
            except Exception as e:
                log.warning(f"  Task fallito: {e}")

    log.info(f"  Paper grezzi recuperati: {len(all_papers)}")
    return all_papers


# ─────────────────────────────────────────────
# STEP 3 — DEDUPLICAZIONE + SCORING
# ─────────────────────────────────────────────

def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Rimuove duplicati basandosi su titolo normalizzato."""
    seen = {}
    for p in papers:
        key = "".join(c.lower() for c in p["title"] if c.isalnum())
        if key and key not in seen:
            seen[key] = p
        elif key in seen and p["citations"] > seen[key]["citations"]:
            # tieni quello con più citazioni (dati più ricchi)
            seen[key] = p
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
    papers_block = json.dumps([
        {
            "id": i + 1,
            "title": p["title"],
            "authors": p["authors"][:3],
            "year": p["year"],
            "abstract": p["abstract"][:400],
            "url": p["url"],
            "citations": p["citations"],
            "subtopic_id": p["subtopic_id"],
            "concepts": p.get("concepts", [])[:5],
            "affiliations": p.get("affiliations", [])[:3],
        }
        for i, p in enumerate(top_papers)
    ], ensure_ascii=False, indent=2)

    return f"""MACRO-ARGOMENTO: {decomposed['macro_topic']}
DESCRIZIONE: {decomposed['description']}

SOTTO-DOMANDE:
{json.dumps(decomposed['subtopics'], ensure_ascii=False, indent=2)}

PAPER DISPONIBILI:
{papers_block}

Produce il documento strutturato richiesto."""


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

    prompt = build_synthesis_prompt(decomposed, top_papers)
    response = client.chat.completions.create(
        model=CONFIG["model"],
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG["max_tokens_synthesize"],
        messages=[
            {"role": "system", "content": SYNTHESIZE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


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
    papers_summary = json.dumps([
        {
            "id": p["_outline_id"],
            "title": p["title"],
            "year": p["year"],
            "subtopic_id": p["subtopic_id"],
            "concepts": p.get("concepts", [])[:3],
        }
        for p in papers
    ], ensure_ascii=False, indent=2)

    prompt = (
        f"MACRO-ARGOMENTO: {decomposed['macro_topic']}\n"
        f"DESCRIZIONE: {decomposed['description']}\n\n"
        f"SOTTO-DOMANDE:\n{json.dumps(decomposed['subtopics'], ensure_ascii=False, indent=2)}\n\n"
        f"PAPER DISPONIBILI:\n{papers_summary}\n\n"
        "Progetta la struttura del documento accademico."
    )
    response = client.chat.completions.create(
        model=CONFIG["model"],
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG.get("max_tokens_outline", 1500),
        messages=[
            {"role": "system", "content": OUTLINE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"Nessun JSON trovato nella risposta outline: {raw[:200]}")
    outline = json.loads(match.group())
    log.info(f"  Indice generato: {len(outline['sections'])} sezioni")
    return outline


def write_section(client: OpenAI, section: dict, papers: list, macro_topic: str) -> str:
    log.info(f"  Scrittura sezione: '{section['title']}'")
    papers_block = json.dumps([
        {
            "id": p["_outline_id"],
            "title": p["title"],
            "authors": p["authors"][:3],
            "year": p["year"],
            "abstract": p["abstract"][:600],
            "url": p["url"],
            "citations": p["citations"],
            "concepts": p.get("concepts", [])[:5],
            "affiliations": p.get("affiliations", [])[:3],
        }
        for p in papers
    ], ensure_ascii=False, indent=2)

    prompt = (
        f"MACRO-ARGOMENTO: {macro_topic}\n"
        f"SEZIONE: {section['title']}\n"
        f"FOCUS: {section['focus']}\n\n"
        f"PAPER ASSEGNATI A QUESTA SEZIONE:\n{papers_block}\n\n"
        "Scrivi la sezione completa."
    )
    response = client.chat.completions.create(
        model=CONFIG["model"],
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG.get("max_tokens_section", 2500),
        messages=[
            {"role": "system", "content": SECTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def reflect_on_section(client: OpenAI, section_title: str, draft: str, papers: list) -> str:
    log.info(f"  Self-reflection su: '{section_title}'")
    citations_list = "\n".join(
        f"- [{(p['authors'][0].split()[-1] if p['authors'] else 'N.A.')} et al., {p['year']}]({p['url']})"
        for p in papers
    )
    prompt = (
        f"SEZIONE: {section_title}\n\n"
        f"CITAZIONI VALIDE PER QUESTA SEZIONE:\n{citations_list}\n\n"
        f"BOZZA DA REVISIONARE:\n{draft}\n\n"
        "Revisiona e migliora la sezione."
    )
    response = client.chat.completions.create(
        model=CONFIG["model"],
        temperature=CONFIG["temperature"],
        max_tokens=CONFIG.get("max_tokens_section", 2500),
        messages=[
            {"role": "system", "content": REFLECT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


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

    # Step 4b + 4c: scrivi e rifletti su ogni sezione
    paper_by_id = {p["_outline_id"]: p for p in indexed}
    section_drafts: list[tuple[str, str]] = []
    for section in outline["sections"]:
        assigned = [paper_by_id[pid] for pid in section["paper_ids"] if pid in paper_by_id]
        if not assigned:
            log.warning(f"  Nessun paper assegnato a '{section['title']}', sezione saltata.")
            continue
        draft = write_section(client, section, assigned, decomposed["macro_topic"])
        revised = reflect_on_section(client, section["title"], draft, assigned)
        section_drafts.append((section["title"], revised))

    # Step 4d: merge finale in Markdown
    parts = [f"## {title}\n\n{content}" for title, content in section_drafts]
    full_synthesis = "\n\n---\n\n".join(parts)
    return outline, full_synthesis


# ─────────────────────────────────────────────
# STEP 5 — SALVATAGGIO RUN
# ─────────────────────────────────────────────

def save_run(thought: str, decomposed: dict, papers: list[Paper], synthesis: str, outline: Optional[dict] = None) -> Path:
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

    # Output finale Markdown
    md_path = run_dir / "synthesis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {decomposed['macro_topic']}\n\n")
        f.write(f"*Run ID: {run_id} — Generato: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        f.write("---\n\n")
        f.write(synthesis)

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
    all_papers = []
    for cycle in range(CONFIG["max_retry_cycles"] + 1):
        papers = retrieve_all_sources(decomposed)
        all_papers.extend(papers)

        unique = deduplicate(all_papers)
        filtered = filter_by_year(unique)

        if len(filtered) >= CONFIG["min_total_papers"]:
            all_papers = filtered
            break

        if cycle < CONFIG["max_retry_cycles"]:
            log.warning(f"  Solo {len(filtered)} paper trovati, ciclo {cycle+1} — allargo le query...")
            # allarga il filtro anno
            CONFIG["min_year"] = max(0, CONFIG["min_year"] - 5)
    else:
        all_papers = deduplicate(all_papers)
        log.warning(f"  Risultati finali dopo retry: {len(all_papers)} paper")

    # Step 3: scoring
    scored_papers = score_papers(all_papers, decomposed["macro_topic"])

    # Step 4: sintesi
    outline: Optional[dict] = None
    if CONFIG.get("hierarchical_synthesis", False):
        outline, synthesis = synthesize_hierarchical(client, decomposed, scored_papers)
    else:
        synthesis = synthesize(client, decomposed, scored_papers)

    # Step 5: salvataggio
    output_path = save_run(thought, decomposed, scored_papers, synthesis, outline=outline)

    log.info("=" * 60)
    log.info(f"PIPELINE COMPLETATA — Output: {output_path}")
    log.info("=" * 60)

    return output_path


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Uso da CLI: python research_pipeline.py "il tuo pensiero"
        user_thought = " ".join(sys.argv[1:])
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
    print(f"  Aprilo con: open {output}")
