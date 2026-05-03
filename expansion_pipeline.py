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
    Paper,
    _EMBEDDING_MODEL,       # noqa: F401  (importato per triggerare singleton)
    _SS_LOCK,
    _cache_get,
    _cache_key,
    _cache_set,
    _http_get,
    _init_cache,
    _normalize_paper,
    deduplicate,
    fetch_arxiv,
    fetch_openalex,
    fetch_pubmed,
    fetch_semantic_scholar,
    filter_by_year,
    reflect_on_section,
    score_papers,
)

log = logging.getLogger(__name__)

# ─── Configurazione espansione (sovrascrive solo i valori rilevanti) ──────────
EXPANSION_CONFIG = {
    # Paper nuovi da aggiungere per sezione (top-K dopo lo scoring)
    "top_k_per_section": 5,
    # Risultati massimi dall'API SS Recommendations
    "ss_recommendations_limit": 10,
    # Token LLM per l'espansione di ogni sezione
    "max_tokens_expand": 3000,
}

# Campi richiesti sia a SS Graph che a SS Recommendations
_SS_FIELDS = "title,authors,year,abstract,externalIds,citationCount,url"


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
        content = lines[1].strip() if len(lines) > 1 else ""
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
# STEP 2 — RISOLUZIONE CITAZIONI VIA SEMANTIC SCHOLAR
# ─────────────────────────────────────────────

def _extract_ss_id_from_url(url: str) -> Optional[str]:
    """
    Estrae l'ID Semantic Scholar da un URL semanticscholar.org.
    Formato: https://www.semanticscholar.org/paper/{Title}/{paperId}
              oppure  https://api.semanticscholar.org/.../{paperId}
    """
    m = re.search(r'semanticscholar\.org/paper/[^/]+/([a-f0-9]{40})', url)
    if m:
        return m.group(1)
    # Formato con solo l'ID hash finale
    m = re.search(r'semanticscholar\.org/paper/([a-f0-9]{40})', url)
    if m:
        return m.group(1)
    # ID alfanumerico corto (SS internal IDs)
    m = re.search(r'semanticscholar\.org/paper/[^/]+/([A-Za-z0-9]{6,})', url)
    if m:
        return m.group(1)
    return None


def _build_ss_paper_lookup_id(url: str) -> Optional[str]:
    """
    Costruisce l'identificatore per l'API SS Graph a partire da un URL.
    Restituisce una stringa nel formato accettato da /paper/{id}:
      - DOI:{doi}
      - ARXIV:{arxivId}
      - {ssId} (ID nativo SS)
    """
    url_lower = url.lower()

    # DOI
    if "doi.org/" in url_lower:
        doi = re.sub(r'^https?://doi\.org/', '', url, flags=re.IGNORECASE).strip()
        if doi:
            return f"DOI:{doi}"

    # arXiv
    arxiv_m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9v]+)', url, re.IGNORECASE)
    if arxiv_m:
        return f"ARXIV:{arxiv_m.group(1)}"

    # Semantic Scholar URL
    ss_id = _extract_ss_id_from_url(url)
    if ss_id:
        return ss_id

    # PubMed — non è direttamente supportato come lookup SS; restituisce None
    return None


def resolve_paper_ss(url: str) -> Optional[dict]:
    """
    Risolve un URL citato → metadati paper via Semantic Scholar Graph API.
    Usa la cache SQLite per evitare chiamate ripetute.
    Restituisce None se non riesce a risolvere.
    """
    lookup_id = _build_ss_paper_lookup_id(url)
    if not lookup_id:
        return None

    conn = _init_cache()
    cache_k = _cache_key("ss_resolve", lookup_id)
    cached = _cache_get(conn, cache_k)
    if cached:
        data = json.loads(cached)
        return data if data else None  # None serializzato come {}

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
        _cache_set(conn, cache_k, json.dumps({}))  # Cachea il fallimento
        return None


def _ss_paper_to_normalized(ss_data: dict, subtopic_id: str = "expansion") -> Paper:
    """Converte un paper SS Graph response al formato Paper normalizzato."""
    doi = (ss_data.get("externalIds") or {}).get("DOI", "")
    authors = [a.get("name", "") for a in (ss_data.get("authors") or [])]
    return _normalize_paper("semantic_scholar", {
        "title": ss_data.get("title", ""),
        "authors": authors,
        "year": ss_data.get("year"),
        "abstract": ss_data.get("abstract", "") or "",
        "url": ss_data.get("url", "") or f"https://doi.org/{doi}" if doi else "",
        "doi": doi,
        "citations": ss_data.get("citationCount", 0) or 0,
        "subtopic_id": subtopic_id,
    })


# ─────────────────────────────────────────────
# STEP 3 — SS RECOMMENDATIONS
# ─────────────────────────────────────────────

def get_ss_recommendations(ss_paper_id: str) -> list[Paper]:
    """
    Recupera paper correlati tramite SS Recommendations API.
    GET /recommendations/v1/papers/forpaper/{paperId}
    """
    conn = _init_cache()
    cache_k = _cache_key("ss_recommendations", ss_paper_id)
    cached = _cache_get(conn, cache_k)
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
                params={"fields": _SS_FIELDS, "limit": EXPANSION_CONFIG["ss_recommendations_limit"]},
                headers=headers,
                timeout=15,
            )
            time.sleep(1.0)
        data = resp.json()
        papers = []
        for p in data.get("recommendedPapers", []):
            if p.get("title"):
                papers.append(_ss_paper_to_normalized(p, subtopic_id="expansion"))
        _cache_set(conn, cache_k, json.dumps(papers))
        log.debug(f"  SS Recommendations per {ss_paper_id[:12]}...: {len(papers)} paper")
        return papers
    except Exception as e:
        log.debug(f"  SS Recommendations fallita per {ss_paper_id}: {e}")
        _cache_set(conn, cache_k, json.dumps([]))
        return []


# ─────────────────────────────────────────────
# STEP 4 — MULTI-SOURCE RETRIEVAL PER SEZIONE
# ─────────────────────────────────────────────

def find_related_papers(
    citations: list[dict],
    section_title: str,
    already_cited_urls: set[str],
) -> list[Paper]:
    """
    Per ogni citazione della sezione:
      (A) Risolve via SS API → ottiene SS paper ID → SS Recommendations
      (B) Usa il titolo del paper come query su tutte le 4 sorgenti

    Restituisce il pool grezzo prima di scoring/filtro.
    """
    pool: list[Paper] = []
    title_queries: list[str] = []

    # ── Fase A: risoluzione SS + raccomandazioni ─────────────────────────────
    for citation in citations:
        url = citation["url"]
        ss_data = resolve_paper_ss(url)
        if not ss_data or not ss_data.get("title"):
            continue

        # Aggiungi il paper stesso come candidato (potrebbe avere più citazioni di quanto noto)
        pool.append(_ss_paper_to_normalized(ss_data, subtopic_id="expansion"))

        # Raccolgo il titolo per la fase B (ricerca per titolo)
        if ss_data.get("title"):
            title_queries.append(ss_data["title"])

        # SS Recommendations dalla paper ID nativa
        ss_id = ss_data.get("paperId") or ss_data.get("externalIds", {}).get("DOI")
        if ss_id and not ss_id.startswith("DOI:"):
            recs = get_ss_recommendations(ss_id)
            pool.extend(recs)

    # Se non abbiamo risolto nulla, usa il titolo della sezione come query di fallback
    if not title_queries:
        title_queries = [section_title]

    # ── Fase B: ricerca multi-sorgente per titolo ────────────────────────────
    subtopic_id = "expansion"
    tasks = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for title_q in title_queries[:5]:  # Limita a 5 titoli per non sovraccaricare
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

    # ── Deduplicazione e filtraggio già-citati ───────────────────────────────
    unique = deduplicate(pool)
    unique = filter_by_year(unique)

    # Filtra paper già presenti nella sintesi originale (per URL normalizzato)
    new_papers = [
        p for p in unique
        if p["url"].lower() not in already_cited_urls
        and (p.get("doi", "") == "" or f"https://doi.org/{p['doi']}".lower() not in already_cited_urls)
        and bool(p["title"])
    ]

    log.info(
        f"  [{section_title[:50]}] Pool grezzo: {len(pool)} | "
        f"Unici nuovi: {len(new_papers)}"
    )
    return new_papers


# ─────────────────────────────────────────────
# STEP 5 — SCORING E SELEZIONE TOP-K
# ─────────────────────────────────────────────

def select_top_papers(new_papers: list[Paper], section_title: str) -> list[Paper]:
    """Esegue lo scoring usando il titolo della sezione come topic semantico e restituisce i top-K."""
    if not new_papers:
        return []
    scored = score_papers(new_papers, section_title)
    k = EXPANSION_CONFIG["top_k_per_section"]
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

IMPORTANTE: Cita SOLO i paper presenti nella lista "nuovi paper". Non inventare riferimenti."""


def expand_section_content(
    client: OpenAI,
    section_title: str,
    original_content: str,
    new_papers: list[Paper],
) -> str:
    """LLM espande la sezione originale integrando i nuovi paper."""
    papers_block = json.dumps(
        [
            {
                "id": i + 1,
                "title": p["title"],
                "authors": p["authors"][:3],
                "year": p["year"],
                "abstract": p["abstract"][:500],
                "url": p["url"],
                "citations": p["citations"],
                "concepts": p.get("concepts", [])[:5],
                "affiliations": p.get("affiliations", [])[:3],
            }
            for i, p in enumerate(new_papers)
        ],
        ensure_ascii=False,
        indent=2,
    )

    prompt = (
        f"SEZIONE: {section_title}\n\n"
        f"TESTO ORIGINALE:\n{original_content}\n\n"
        f"NUOVI PAPER DA INTEGRARE:\n{papers_block}\n\n"
        "Espandi la sezione integrando i nuovi paper."
    )

    response = client.chat.completions.create(
        model=CONFIG["model"],
        temperature=CONFIG["temperature"],
        max_tokens=EXPANSION_CONFIG["max_tokens_expand"],
        messages=[
            {"role": "system", "content": EXPAND_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


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
    parts.append(f"*Espansa: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

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

        # ── Fase 7: self-reflection ──────────────────────────────────────────
        log.info("STEP 7 — Self-reflection...")
        all_papers_for_reflect = top_papers  # include solo i nuovi per la validazione citazioni
        expanded_final = reflect_on_section(
            client, section["title"], expanded_draft, all_papers_for_reflect
        )

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
    args = parser.parse_args()

    output = run_expansion_pipeline(args.synthesis, top_k=args.top_k)
    print(f"\n✓ Sintesi espansa generata: {output}")
    print(f"  Aprila con: code {output}")
