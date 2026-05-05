"""
expansion_pipeline.py
=====================
Pipeline di espansione per arricchire una synthesis.md esistente.

Per ogni sezione della sintesi:
  1. Estrae le citazioni presenti (DOI / Semantic Scholar / arXiv URL)
  2. Risolve ogni citazione → metadati via Semantic Scholar Graph API
  3. Recupera paper correlati via SS Recommendations API
  4. Recupera paper correlati anche via ricerca per titolo su tutte le sorgenti
     (Arxiv, PubMed, Semantic Scholar, OpenAlex) — stessa metodologia della pipeline
  5. Deduplicazione + filtro paper già citati nell'intera sintesi
  6. Scoring (titolo sezione come topic semantico)
  7. LLM espande la sezione con i nuovi paper trovati
  8. Self-reflection (critica e revisione)
  9. Salvataggio: {run_dir}_expanded/synthesis_expanded.md + expansion_trace.jsonl

Uso:
  python expansion_pipeline.py runs/20260503_191211/synthesis.md
  python expansion_pipeline.py runs/20260503_191211/synthesis.md --top-k 5
"""

import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ─── Riutilizzo integrale dell'infrastruttura della pipeline principale ───────
from research_pipeline import (
    CONFIG,
    LLM_STATS,
    Paper,
    _EMBEDDING_MODEL,           # noqa: F401  (importato per triggerare singleton)
    _SS_FIELDS,
    _SS_LOCK,
    _build_ss_paper_lookup_id,
    _cache_get,
    _cache_key,
    _cache_set,
    _http_get,
    _init_cache,
    _normalize_paper,
    _ss_paper_to_normalized,
    deduplicate,
    fetch_arxiv,
    fetch_openalex,
    fetch_pubmed,
    fetch_semantic_scholar,
    filter_by_year,
    get_ss_recommendations,
    llm_chat,
    mmr_select,
    reflect_on_section,
    resolve_paper_ss,
    score_papers,
)

log = logging.getLogger(__name__)

# ─── Configurazione espansione ────────────────────────────────────────────────
EXPANSION_CONFIG = {
    "top_k_per_section": 5,
    "ss_recommendations_limit": 10,
    "max_tokens_expand": 3000,
    # Self-reflection: raddoppia le chiamate LLM, default disattivato
    "enable_reflection": False,
    # Max citazioni per sezione da cui partire (riduce le chiamate API)
    "max_citations_per_section": 3,
    # Max titoli per la ricerca multi-source (se threshold non soddisfatta)
    "max_titles_for_search": 3,
    # Soglia: se SS Recommendations producono >= N candidati, salta multi-source
    "skip_multisource_threshold": 15,
    # Caratteri abstract per ogni paper inviato all'LLM
    "abstract_chars_expand": 300,
    # MMR per la selezione dei paper per sezione
    "enable_mmr": True,
    "mmr_lambda": 0.6,
    # Fetch delle REFERENZE (oltre alle raccomandazioni) via SS
    "enable_references_fetch": True,
    # Modalità diff: chiede all'LLM solo i nuovi paragrafi (riduce ~50% token output)
    "diff_mode": False,
}


# ─────────────────────────────────────────────
# UTILITÀ DETERMINISTICHE ANTI-RIPETIZIONE
# ─────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """
    Rimuove wrapper ```markdown ... ``` (o ```) dal testo.
    Alcune sezioni generate dalla pipeline principale vengono avvolte in
    blocchi di codice Markdown — vanno eliminati prima di passare il
    contenuto all'LLM per evitare che li riproduca.
    """
    # Rimuove apertura: ```markdown\n  oppure ```\n
    text = re.sub(r'^```(?:markdown)?\s*\n', '', text.strip())
    # Rimuove chiusura: \n```  alla fine
    text = re.sub(r'\n```\s*$', '', text.strip())
    return text.strip()


def _strip_leading_heading(text: str) -> str:
    """
    Rimuove eventuali righe di heading Markdown (##, ###, ecc.) che l'LLM
    aggiunge in cima alla risposta, duplicando il titolo già gestito da
    save_expansion_run con il separatore ## .
    Continua a rimuovere righe di heading consecutive fino al primo
    paragrafo di testo.
    """
    lines = text.lstrip().split("\n")
    while lines and re.match(r'^#{1,4}\s+', lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).lstrip()


# ─────────────────────────────────────────────
# STEP 1 — PARSING DELLA SINTESI
# ─────────────────────────────────────────────

def parse_synthesis(md_path: Path) -> list[dict]:
    """
    Divide synthesis.md in sezioni usando i titoli ## come separatori.
    Restituisce una lista di { section_id, title, content }.
    """
    text = md_path.read_text(encoding="utf-8")
    # Splitta su ogni ## header (non ###)
    parts = re.split(r'\n(?=## )', text)
    sections = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        raw_title = lines[0].lstrip("# ").strip()
        raw_content = lines[1].strip() if len(lines) > 1 else ""
        # Rimuove wrapper ```markdown``` prima di qualsiasi elaborazione
        content = _strip_code_fences(raw_content)
        # Salta header di metadati (run ID, linea orizzontale, titolo H1)
        if not raw_title or raw_title.startswith("*Run ID"):
            continue
        sections.append({
            "section_id": f"SEC{i + 1}",
            "title": raw_title,
            "content": content,
        })
    log.info(f"  Sezioni trovate: {len(sections)}")
    return sections


def extract_citations(text: str) -> list[dict]:
    """
    Estrae tutte le citazioni nel formato [testo](url) dal testo Markdown.
    Deduplica per URL. Restituisce [{ display_text, url }].
    """
    pattern = re.compile(r'\[([^\]]+)\]\((https?://[^\)]+)\)')
    seen: set[str] = set()
    citations = []
    for display_text, url in pattern.findall(text):
        url = url.rstrip(".,;)")
        if url not in seen:
            seen.add(url)
            citations.append({"display_text": display_text, "url": url})
    return citations


def collect_all_cited_urls(sections: list[dict]) -> set[str]:
    """Raccoglie tutti gli URL citati nell'intera sintesi (per filtrare duplicati cross-sezione)."""
    all_urls: set[str] = set()
    for sec in sections:
        for c in extract_citations(sec["content"]):
            all_urls.add(c["url"].lower())
    return all_urls


# ─────────────────────────────────────────────
# STEP 2 — FETCH REFERENZE VIA SS
# ─────────────────────────────────────────────

def fetch_ss_references_for_paper(ss_paper_id: str, subtopic_id: str = "expansion") -> list[Paper]:
    """
    Recupera la BIBLIOGRAFIA di un paper via SS /paper/{id}/references.
    I paper referenziati dall'autore originale sono semanticamente più vicini
    al tema della sezione rispetto alle semplici "raccomandazioni algoritmiche".
    """
    conn = _init_cache()
    cache_k = _cache_key("ss_references", ss_paper_id)
    cached = _cache_get(conn, cache_k, ttl_days=60)
    if cached:
        return json.loads(cached)

    ref_url = f"https://api.semanticscholar.org/graph/v1/paper/{ss_paper_id}/references"
    fields = "title,authors,year,abstract,externalIds,citationCount,url"
    headers = {}
    if CONFIG.get("semantic_scholar_api_key"):
        headers["x-api-key"] = CONFIG["semantic_scholar_api_key"]

    try:
        with _SS_LOCK:
            resp = _http_get(
                ref_url,
                params={"fields": fields, "limit": 20},
                headers=headers,
                timeout=15,
            )
            time.sleep(1.0)
        papers = []
        for item in resp.json().get("data", []):
            ref = item.get("citedPaper", {})
            if ref.get("title"):
                papers.append(_ss_paper_to_normalized(ref, subtopic_id))
        _cache_set(conn, cache_k, json.dumps(papers))
        log.debug(f"  SS References per {ss_paper_id[:12]}...: {len(papers)} paper")
        return papers
    except Exception as e:
        log.debug(f"  SS References fallita per {ss_paper_id}: {e}")
        _cache_set(conn, cache_k, json.dumps([]))
        return []


def resolve_papers_ss_batch(urls: list[str], subtopic_id: str = "expansion") -> list[Paper]:
    """
    Risolve più URL contemporaneamente via SS /paper/batch (fino a 500 ID per call).
    Molto più efficiente di N chiamate seriali; riduce drasticamente il rischio 429.
    """
    # Costruisce gli SS lookup IDs
    id_to_url: dict[str, str] = {}
    for url in urls:
        lid = _build_ss_paper_lookup_id(url)
        if lid:
            id_to_url[lid] = url

    if not id_to_url:
        return []

    # Controlla cache per ogni ID
    conn = _init_cache()
    results: list[Paper] = []
    ids_to_fetch: list[str] = []
    for lid in id_to_url:
        cache_k = _cache_key("ss_resolve", lid)
        cached = _cache_get(conn, cache_k, ttl_days=0)
        if cached:
            data = json.loads(cached)
            if data and data.get("title"):
                results.append(_ss_paper_to_normalized(data, subtopic_id))
        else:
            ids_to_fetch.append(lid)

    if not ids_to_fetch:
        return results

    # Batch call
    batch_url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    headers = {"Content-Type": "application/json"}
    if CONFIG.get("semantic_scholar_api_key"):
        headers["x-api-key"] = CONFIG["semantic_scholar_api_key"]

    try:
        with _SS_LOCK:
            resp = requests.post(
                batch_url,
                json={"ids": ids_to_fetch},
                params={"fields": _SS_FIELDS},
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            time.sleep(1.0)
        for lid, entry in zip(ids_to_fetch, resp.json()):
            cache_k = _cache_key("ss_resolve", lid)
            if entry and entry.get("title"):
                _cache_set(conn, cache_k, json.dumps(entry))
                results.append(_ss_paper_to_normalized(entry, subtopic_id))
            else:
                _cache_set(conn, cache_k, json.dumps({}))
        log.debug(f"  SS Batch resolve: {len(results)} paper risolti")
    except Exception as e:
        log.warning(f"  SS Batch resolve fallito: {e}")
        # Fallback: risoluzione seriale per i paper non ancora in cache
        for lid in ids_to_fetch:
            url = id_to_url.get(lid, "")
            data = resolve_paper_ss(url)
            if data and data.get("title"):
                results.append(_ss_paper_to_normalized(data, subtopic_id))

    return results


# ─────────────────────────────────────────────
# STEP 3 — MULTI-SOURCE RETRIEVAL PER SEZIONE
# ─────────────────────────────────────────────

def find_related_papers(
    citations: list[dict],
    section_title: str,
    already_cited_urls: set[str],
) -> list[Paper]:
    """
    Per ogni citazione della sezione (max N):
      (A) Batch resolve via SS /paper/batch
      (B) Per ogni paper risolto: SS Recommendations + SS References (se abilitato)
      (C) Solo se il pool è ancora insufficiente: ricerca multi-source per titolo

    Restituisce il pool grezzo prima di scoring/filtro.
    """
    pool: list[Paper] = []
    title_queries: list[str] = []

    max_cit = EXPANSION_CONFIG["max_citations_per_section"]
    citations_subset = citations[:max_cit]

    # ── Fase A: batch resolve via SS ─────────────────────────────────────────
    resolved_papers = resolve_papers_ss_batch(
        [c["url"] for c in citations_subset], subtopic_id="expansion"
    )
    pool.extend(resolved_papers)

    for ss_paper in resolved_papers:
        if ss_paper.get("title"):
            title_queries.append(ss_paper["title"])

        # Ricava SS paper ID dal URL per raccomandazioni + referenze
        url = ss_paper.get("url", "")
        ss_id = None
        # Prova a trovare il paperId dal resolve cache
        lookup_id = _build_ss_paper_lookup_id(url)
        if lookup_id:
            conn = _init_cache()
            cache_k = _cache_key("ss_resolve", lookup_id)
            cached = _cache_get(conn, cache_k, ttl_days=0)
            if cached:
                data = json.loads(cached)
                ss_id = data.get("paperId")

        if ss_id:
            # SS Recommendations (paper correlati algoritmicamente)
            recs = get_ss_recommendations(ss_id)
            pool.extend(recs)

            # SS References (BIBLIOGRAFIA del paper — alta qualità semantica)
            if EXPANSION_CONFIG.get("enable_references_fetch", True):
                refs = fetch_ss_references_for_paper(ss_id)
                pool.extend(refs)

    # Fallback: se nessuna citazione è risolvibile, usa il titolo della sezione
    if not title_queries:
        title_queries = [section_title]

    # ── Fase B: ricerca multi-source SOLO se pool insufficiente ─────────────
    skip_threshold = EXPANSION_CONFIG["skip_multisource_threshold"]
    if len(pool) < skip_threshold:
        max_titles = EXPANSION_CONFIG["max_titles_for_search"]
        subtopic_id = "expansion"
        tasks = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            for title_q in title_queries[:max_titles]:
                queries = [title_q]
                tasks.append(executor.submit(fetch_arxiv,            queries, subtopic_id))
                tasks.append(executor.submit(fetch_pubmed,           queries, subtopic_id))
                tasks.append(executor.submit(fetch_semantic_scholar, queries, subtopic_id))
                tasks.append(executor.submit(fetch_openalex,         queries, subtopic_id))
            for future in as_completed(tasks):
                try:
                    pool.extend(future.result())
                except Exception as e:
                    log.debug(f"  Task retrieval espansione fallito: {e}")
    else:
        log.info(f"  Pool SS sufficiente ({len(pool)} candidati) — skip ricerca multi-source")

    # ── Deduplicazione e filtraggio già-citati ───────────────────────────────
    unique = deduplicate(pool)
    unique = filter_by_year(unique)

    # Normalizza title+year come fingerprint aggiuntivo per dedup cross-sezione
    def _title_year_fp(p: Paper) -> str:
        title_key = "".join(c.lower() for c in p["title"] if c.isalnum())
        return f"{title_key}_{p.get('year', '')}"

    already_fps = {_title_year_fp({"title": u, "year": None}) for u in already_cited_urls}  # stub

    new_papers = [
        p for p in unique
        if p["url"].lower() not in already_cited_urls
        and (not p.get("doi") or f"https://doi.org/{p['doi']}".lower() not in already_cited_urls)
        and bool(p["title"])
    ]

    log.info(
        f"  [{section_title[:50]}] Pool grezzo: {len(pool)} | Unici nuovi: {len(new_papers)}"
    )
    return new_papers


# ─────────────────────────────────────────────
# STEP 5 — SCORING E SELEZIONE TOP-K
# ─────────────────────────────────────────────

def select_top_papers(new_papers: list[Paper], section_title: str) -> list[Paper]:
    """Scoring + MMR (se abilitato) per restituire i top-K paper più diversificati."""
    if not new_papers:
        return []
    scored = score_papers(new_papers, section_title)
    k = EXPANSION_CONFIG["top_k_per_section"]
    if EXPANSION_CONFIG.get("enable_mmr", True) and len(scored) > k:
        lambda_mmr = EXPANSION_CONFIG.get("mmr_lambda", 0.6)
        return mmr_select(scored, k=k, lambda_mmr=lambda_mmr)
    return scored[:k]


# ─────────────────────────────────────────────
# STEP 6 — ESPANSIONE LLM
# ─────────────────────────────────────────────

EXPAND_SYSTEM = """Sei un ricercatore accademico senior specializzato nella revisione e nel potenziamento di survey scientifiche.

Ricevi:
- Il titolo di una sezione esistente
- Il contenuto originale della sezione (già scritto)
- Una lista di nuovi paper scientifici non ancora citati

Il tuo compito:
1. Integra organicamente i nuovi paper nel testo esistente
2. Aggiungi un nuovo paragrafo o espandi i paragrafi esistenti dove i nuovi paper apportano valore
3. Ogni nuovo paper deve essere citato nel formato [Autore et al., Anno](URL)
4. Mantieni la coerenza stilistica con il testo originale (tono accademico formale)
5. Non rimuovere o alterare le citazioni esistenti
6. Restituisci la sezione completa espansa in Markdown

IMPORTANTE:
- Cita SOLO i paper presenti nella lista "nuovi paper". Non inventare riferimenti.
- NON inserire il titolo della sezione come intestazione (##, ###) all'inizio della risposta.
- Inizia DIRETTAMENTE con il testo del primo paragrafo, senza alcuna riga di heading.
- NON avvolgere la risposta in blocchi di codice (```markdown```)."""


EXPAND_DIFF_SYSTEM = """Sei un ricercatore accademico senior. Devi integrare nuovi paper in una sezione esistente.

COMPITO: Scrivi SOLO i nuovi paragrafi da aggiungere (NON riscrivere il testo esistente).
Per ogni nuovo paper, scrivi 1-3 frasi che lo integrino nel filo argomentativo della sezione.
Ogni citazione nel formato [Autore et al., Anno](URL).

Formato risposta:
[INSERISCI_DOPO_PARAGRAFO_N]
testo del nuovo paragrafo...

[INSERISCI_DOPO_PARAGRAFO_N]
testo del secondo nuovo paragrafo...

IMPORTANTE:
- Cita SOLO i paper nella lista. Non inventare riferimenti.
- Indica il numero del paragrafo ESISTENTE dopo cui inserire il nuovo testo (contando da 1).
- Se tutti i paper vanno alla fine, usa [INSERISCI_ALLA_FINE].
- NON riscrivere o ripetere il testo originale."""


def expand_section_content(
    client: OpenAI,
    section_title: str,
    original_content: str,
    new_papers: list[Paper],
) -> str:
    """LLM espande la sezione originale integrando i nuovi paper.
    Se diff_mode è abilitato, chiede solo i nuovi paragrafi (meno token output).
    """
    if EXPANSION_CONFIG.get("diff_mode", False):
        return _expand_section_diff(client, section_title, original_content, new_papers)
    return _expand_section_full(client, section_title, original_content, new_papers)


def _expand_section_full(
    client: OpenAI,
    section_title: str,
    original_content: str,
    new_papers: list[Paper],
) -> str:
    """Modalità standard: riscrive la sezione completa con i nuovi paper integrati."""
    abs_chars = EXPANSION_CONFIG["abstract_chars_expand"]
    papers_block = json.dumps(
        [{"id": i + 1, "title": p["title"], "authors": p["authors"][:3],
          "year": p["year"], "abstract": p["abstract"][:abs_chars],
          "url": p["url"], "citations": p["citations"]}
         for i, p in enumerate(new_papers)],
        ensure_ascii=False,
    )
    prompt = (
        f"SEZIONE: {section_title}\n\n"
        f"TESTO ORIGINALE:\n{original_content}\n\n"
        f"NUOVI PAPER:\n{papers_block}\n\n"
        "Espandi la sezione integrando i nuovi paper."
    )
    raw = llm_chat(client, system=EXPAND_SYSTEM, user=prompt,
                   max_tokens=EXPANSION_CONFIG["max_tokens_expand"])
    return _strip_leading_heading(_strip_code_fences(raw))


def _expand_section_diff(
    client: OpenAI,
    section_title: str,
    original_content: str,
    new_papers: list[Paper],
) -> str:
    """Modalità diff: chiede solo i nuovi paragrafi da inserire, poi li applica al testo originale.
    Risparmia ~50% token di output rispetto alla riscrittura completa.
    """
    abs_chars = EXPANSION_CONFIG["abstract_chars_expand"]
    papers_block = json.dumps(
        [{"id": i + 1, "title": p["title"], "authors": p["authors"][:3],
          "year": p["year"], "abstract": p["abstract"][:abs_chars],
          "url": p["url"], "citations": p["citations"]}
         for i, p in enumerate(new_papers)],
        ensure_ascii=False,
    )
    # Numerazione paragrafi nel testo originale
    paragraphs = [p.strip() for p in original_content.split("\n\n") if p.strip()]
    numbered_content = "\n\n".join(
        f"[P{i+1}] {p}" for i, p in enumerate(paragraphs)
    )
    prompt = (
        f"SEZIONE: {section_title}\n\n"
        f"TESTO ESISTENTE (paragrafi numerati):\n{numbered_content}\n\n"
        f"NUOVI PAPER DA INTEGRARE:\n{papers_block}\n\n"
        "Scrivi SOLO i nuovi paragrafi da aggiungere."
    )
    raw = llm_chat(client, system=EXPAND_DIFF_SYSTEM, user=prompt,
                   max_tokens=EXPANSION_CONFIG["max_tokens_expand"] // 2)
    # Applica il diff al testo originale
    return _apply_diff_to_content(original_content, raw)


def _apply_diff_to_content(original: str, diff_response: str) -> str:
    """Applica i nuovi paragrafi generati in modalità diff al testo originale."""
    paragraphs = [p.strip() for p in original.split("\n\n") if p.strip()]

    # Parsa le istruzioni [INSERISCI_DOPO_PARAGRAFO_N] e [INSERISCI_ALLA_FINE]
    insertions: dict[int, list[str]] = {}  # para_index → [new_paras]
    current_key: Optional[int] = None
    current_text: list[str] = []

    for line in diff_response.split("\n"):
        m_after = re.match(r'\[INSERISCI_DOPO_PARAGRAFO_(\d+)\]', line.strip(), re.IGNORECASE)
        m_end = re.match(r'\[INSERISCI_ALLA_FINE\]', line.strip(), re.IGNORECASE)
        if m_after or m_end:
            if current_key is not None and current_text:
                insertions.setdefault(current_key, []).append("\n".join(current_text).strip())
            current_key = int(m_after.group(1)) if m_after else len(paragraphs)
            current_text = []
        elif current_key is not None:
            current_text.append(line)

    if current_key is not None and current_text:
        insertions.setdefault(current_key, []).append("\n".join(current_text).strip())

    if not insertions:
        # Se il modello non ha seguito il formato, appendiamo il diff alla fine
        clean = _strip_leading_heading(_strip_code_fences(diff_response))
        return original + "\n\n" + clean

    # Ricostruisce il testo inserendo i nuovi paragrafi nelle posizioni giuste
    result: list[str] = []
    for i, para in enumerate(paragraphs):
        result.append(para)
        idx = i + 1  # 1-based
        for new_para in insertions.get(idx, []):
            if new_para:
                result.append(new_para)
    # Paragrafi "alla fine"
    for new_para in insertions.get(len(paragraphs), []):
        if new_para:
            result.append(new_para)

    return "\n\n".join(result)


# ─────────────────────────────────────────────
# STEP 7 — SALVATAGGIO
# ─────────────────────────────────────────────

def save_expansion_run(
    original_path: Path,
    expanded_sections: list[dict],
    trace: list[dict],
) -> Path:
    """
    Salva la sintesi espansa in {run_dir}_expanded/:
      - synthesis_expanded.md
      - expansion_trace.jsonl
    """
    run_dir = original_path.parent
    out_dir = run_dir.parent / f"{run_dir.name}_expanded"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ricostruisce il documento Markdown espanso
    header_lines: list[str] = []
    original_text = original_path.read_text(encoding="utf-8")
    for line in original_text.split("\n"):
        if line.startswith("# ") or line.startswith("*Run ID"):
            header_lines.append(line)
        elif line.strip() == "---" and len(header_lines) < 4:
            header_lines.append(line)
        else:
            break

    parts = ["\n".join(header_lines).strip(), ""]
    parts.append(f"*Espansa: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    parts.append("> **Metadati espansione**\n>")
    for line in LLM_STATS.summary_md(CONFIG["model"]).split("\n"):
        parts.append(f"> {line}")
    parts.append(f"> - **Sezioni espanse**: {sum(1 for s in expanded_sections if s.get('expanded_content'))}\n")

    for sec in expanded_sections:
        parts.append(f"## {sec['title']}\n")
        parts.append(sec["expanded_content"])
        parts.append("\n---\n")

    md_content = "\n".join(parts)
    md_path = out_dir / "synthesis_expanded.md"
    md_path.write_text(md_content, encoding="utf-8")

    # Trace JSONL
    trace_path = out_dir / "expansion_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        # Prima riga: metadati run
        f.write(json.dumps({
            "type": "run_metadata",
            "timestamp": datetime.utcnow().isoformat(),
            "model": CONFIG["model"],
            "expansion_config": EXPANSION_CONFIG,
            "llm_stats": LLM_STATS.to_dict(),
        }, ensure_ascii=False) + "\n")
        for entry in trace:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    log.info(f"  Output salvato in: {out_dir}")
    return md_path


# ─────────────────────────────────────────────
# PIPELINE PRINCIPALE
# ─────────────────────────────────────────────

def run_expansion_pipeline(synthesis_path: Path, top_k: int = 5) -> Path:
    """
    Entry point principale.
    Riceve il path di una synthesis.md esistente e restituisce
    il path della versione espansa.
    """
    EXPANSION_CONFIG["top_k_per_section"] = top_k

    log.info("=" * 60)
    log.info("AVVIO EXPANSION PIPELINE")
    log.info(f"Input: {synthesis_path}")
    log.info("=" * 60)
    LLM_STATS.start_run()

    if not synthesis_path.exists():
        raise FileNotFoundError(f"File non trovato: {synthesis_path}")

    # Client LLM (stesso della pipeline principale)
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

    # ── Fase 1: parsing ──────────────────────────────────────────────────────
    log.info("STEP 1 — Parsing della sintesi...")
    sections = parse_synthesis(synthesis_path)
    if not sections:
        raise ValueError("Nessuna sezione trovata in synthesis.md")

    # Raccoglie tutti gli URL già citati nell'intera sintesi
    already_cited_urls = collect_all_cited_urls(sections)
    log.info(f"  URL già citati nella sintesi: {len(already_cited_urls)}")

    # ── Processo per sezione ─────────────────────────────────────────────────
    expanded_sections: list[dict] = []
    trace: list[dict] = []

    for idx, section in enumerate(sections):
        log.info("-" * 50)
        log.info(f"SEZIONE {idx + 1}/{len(sections)}: {section['title'][:70]}")

        citations = extract_citations(section["content"])
        log.info(f"  Citazioni nella sezione: {len(citations)}")

        # ── Fase 2-3-4: retrieval correlato ─────────────────────────────────
        log.info("  Ricerca paper correlati...")
        related = find_related_papers(citations, section["title"], already_cited_urls)

        if not related:
            log.info("  Nessun paper nuovo trovato — sezione mantenuta invariata.")
            expanded_sections.append({
                "title": section["title"],
                "expanded_content": section["content"],
            })
            trace.append({
                "section_id": section["section_id"],
                "title": section["title"],
                "citations_found": len(citations),
                "new_papers_found": 0,
                "new_papers_used": 0,
                "expanded": False,
            })
            continue

        # ── Fase 5: scoring e selezione top-K ───────────────────────────────
        log.info(f"STEP 5 — Scoring {len(related)} paper correlati...")
        top_papers = select_top_papers(related, section["title"])
        log.info(f"  Top-{len(top_papers)} paper selezionati per espansione")

        # ── Fase 6: espansione LLM ───────────────────────────────────────────
        log.info("STEP 6 — Espansione sezione con LLM...")
        expanded_draft = expand_section_content(
            client, section["title"], section["content"], top_papers
        )

        # ── Fase 7: self-reflection (opzionale) ──────────────────────────────
        if EXPANSION_CONFIG.get("enable_reflection", False):
            log.info("STEP 7 — Self-reflection...")
            # Per la reflection: passa SIA i paper originali (estratti dalle citazioni
            # pre-esistenti) SIA i nuovi, altrimenti il revisore potrebbe rimuovere
            # citazioni valide pensandole inventate.
            original_citation_papers = [
                {
                    "title": c["display_text"],
                    "authors": [c["display_text"].split(" et al.")[0]] if "et al." in c["display_text"] else [],
                    "year": "",
                    "url": c["url"],
                }
                for c in citations
            ]
            all_papers_for_reflect = list(top_papers) + original_citation_papers
            raw_reflection = reflect_on_section(
                client, section["title"], expanded_draft, all_papers_for_reflect
            )
            expanded_final = _strip_leading_heading(_strip_code_fences(raw_reflection))
        else:
            expanded_final = expanded_draft

        # Aggiorna gli URL già citati per le sezioni successive
        new_urls = {c["url"].lower() for c in extract_citations(expanded_final)}
        already_cited_urls.update(new_urls)

        expanded_sections.append({
            "title": section["title"],
            "expanded_content": expanded_final,
        })

        trace.append({
            "section_id": section["section_id"],
            "title": section["title"],
            "citations_found": len(citations),
            "new_papers_found": len(related),
            "new_papers_used": len(top_papers),
            "expanded": True,
            "new_papers": [
                {
                    "title": p["title"],
                    "year": p["year"],
                    "url": p["url"],
                    "score": p.get("score", 0),
                    "score_detail": p.get("score_detail", {}),
                }
                for p in top_papers
            ],
        })

    # ── Salvataggio ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("STEP 8 — Salvataggio risultati...")
    output_path = save_expansion_run(synthesis_path, expanded_sections, trace)

    total_new = sum(e.get("new_papers_used", 0) for e in trace)
    expanded_count = sum(1 for e in trace if e.get("expanded"))
    log.info(f"  Sezioni espanse: {expanded_count}/{len(sections)}")
    log.info(f"  Nuovi paper integrati: {total_new}")
    log.info(f"  LLM: {LLM_STATS.calls} chiamate, {LLM_STATS.total_tokens:,} token, {LLM_STATS.total_seconds:.1f}s")
    log.info(f"  Tempo totale: {LLM_STATS.wall_seconds:.1f}s")
    log.info("=" * 60)
    log.info(f"EXPANSION COMPLETATA — Output: {output_path}")
    log.info("=" * 60)

    return output_path


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Espande una synthesis.md con paper correlati alle citazioni esistenti."
    )
    parser.add_argument(
        "synthesis",
        type=Path,
        help="Path al file synthesis.md da espandere (es. runs/20260503_191211/synthesis.md)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Numero massimo di nuovi paper da integrare per sezione (default: 5)",
    )
    parser.add_argument(
        "--reflect",
        action="store_true",
        help="Abilita self-reflection LLM su ogni sezione (raddoppia le chiamate al modello).",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Modalità diff: genera solo i nuovi paragrafi (risparmio ~50% token output).",
    )
    parser.add_argument(
        "--no-references",
        action="store_true",
        help="Disabilita il fetch delle referenze SS (usa solo raccomandazioni).",
    )
    args = parser.parse_args()

    EXPANSION_CONFIG["enable_reflection"] = args.reflect
    EXPANSION_CONFIG["diff_mode"] = args.diff
    if args.no_references:
        EXPANSION_CONFIG["enable_references_fetch"] = False

    output = run_expansion_pipeline(args.synthesis, top_k=args.top_k)
    print(f"\n✓ Sintesi espansa generata: {output}")
    print(f"  Aprila con: code {output}")
