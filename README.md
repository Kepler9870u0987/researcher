# Research Pipeline

Pipeline deterministico per espandere e formalizzare un pensiero utente attraverso ricerca scientifica accreditata.

## Setup

```bash
pip install -r requirements.txt

# Richiede GitHub CLI (https://cli.github.com)
gh auth login
$env:GITHUB_TOKEN = (gh auth token)   # PowerShell
# oppure
export GITHUB_TOKEN=$(gh auth token)  # bash/zsh
```

> **Nota:** viene usato l'endpoint [GitHub Models](https://docs.github.com/en/github-models) (`models.inference.ai.azure.com`), l'API pubblica ufficiale di GitHub per gli LLM. Funziona con token OAuth (`gh auth token`) o con un PAT classico.

Modifica `research_pipeline.py` → `CONFIG["pubmed_email"]` con la tua email (richiesto da NCBI).

## Utilizzo

```bash
# Da CLI con argomento diretto
python research_pipeline.py "l'impatto dei modelli linguistici sull'apprendimento umano"

# Interattivo (digita il pensiero, INVIO due volte per confermare)
python research_pipeline.py
```

## Output

Ogni run crea una cartella in `./runs/<timestamp>/`:

```
runs/
└── 20240503_143022/
    ├── synthesis.md      ← documento finale con citazioni
    ├── papers_all.json   ← tutti i paper recuperati e scored
    └── trace.jsonl       ← traccia completa della run (input, config, decomposed)
```

## Pipeline

```
[Input utente]
      │
      ▼
[1. Decomposizione]  ← gpt-4o (GitHub Copilot), temperature=0
    └─ macro_topic
    └─ sotto-domande (4-6)
    └─ query per sorgente
      │
      ▼ (parallelo)
[2. Retrieval]
    ├─ Arxiv API
    ├─ PubMed / Entrez
    ├─ Semantic Scholar API
    └─ OpenAlex API
      │
      ▼
[3. Deduplicazione + Scoring]
    ├─ citation count  (40%)
    ├─ recency         (20%)
    └─ semantic sim    (40%)  ← embedding all-MiniLM-L6-v2
      │
      ▼
[4. Sintesi]  ← gpt-4o (GitHub Copilot), temperature=0
    └─ documento Markdown con citazioni verificabili
      │
      ▼
[5. Salvataggio run]  ← JSONL tracciato + Markdown
```

## Determinismo

- `temperature=0` su tutti i call LLM
- Modello embedding fisso (`all-MiniLM-L6-v2`)
- Cache SQLite (`cache.db`) per le chiamate API
- Seed implicito nelle query (derivano deterministicamente dalla decomposizione)
- Ogni run è completamente riproducibile dai file in `trace.jsonl`

## Sorgenti accreditate

| Sorgente | Tipo | API |
|---|---|---|
| Arxiv | Preprint CS/Fisica/Math | Ufficiale, gratuita |
| PubMed | Biomedico | NCBI Entrez, gratuita |
| Semantic Scholar | Multi-dominio + citation graph | REST, gratuita |
| OpenAlex | Open access aggregator (250M+ works) | REST, gratuita |

## Configurazione avanzata

Modifica il dizionario `CONFIG` in cima allo script:

```python
CONFIG = {
    # Modelli disponibili via GitHub Copilot:
    # gpt-4o, gpt-4o-mini, o1, o3-mini, claude-3.5-sonnet, claude-3-haiku, ...
    "model": "gpt-4o",
    "max_results_per_source": 8,   # paper per query per sorgente
    "min_total_papers": 10,         # soglia prima di sintetizzare
    "max_retry_cycles": 2,          # retry se risultati insufficienti
    "weight_citations": 0.4,        # peso score finale
    "weight_recency": 0.2,
    "weight_semantic": 0.4,
    "min_year": 2015,               # filtra paper troppo vecchi
}
```
