# Guida Avanzata all'Ottimizzazione della Research Pipeline

Questo documento sintetizza l'analisi tecnica, le critiche costruttive e le strategie di evoluzione per la pipeline di ricerca scientifica deterministica definita nel progetto originale [cite: 1, 3].

---

## 1. Analisi della Pipeline Originale
La pipeline iniziale è strutturata come un sistema **Plan-and-Execute** [cite: 3]:
1.  **Decomposizione:** Claude trasforma un pensiero grezzo in query strutturate [cite: 3].
2.  **Retrieval:** Ricerca parallela su Arxiv, PubMed, Semantic Scholar e OpenAlex [cite: 3].
3.  **Scoring:** Ranking basato su citazioni, recency e similarità semantica (all-MiniLM-L6-v2) [cite: 3].
4.  **Sintesi:** Generazione di un documento Markdown con citazioni [cite: 3].

Sebbene solida, l'analisi ha evidenziato diverse aree di miglioramento per raggiungere uno standard professionale.

---

## 2. Punti Critici e Soluzioni Proposte

### 2.1 Concorrenza e Thread Safety
* **Problema:** L'uso di un unico oggetto di connessione SQLite tra più thread nel `ThreadPoolExecutor` può causare corruzione dei dati [cite: 3].
* **Soluzione:** Implementare una connessione separata per ogni thread worker o utilizzare un pool di connessioni.

### 2.2 Ingegneria dello Scoring
* **Caricamento Modelli:** Il caricamento di `SentenceTransformer` dentro la funzione di scoring rallenta l'esecuzione. Va inizializzato come Singleton.
* **Outliers nelle Citazioni:** La normalizzazione lineare penalizza i paper nuovi rispetto a "classici" con migliaia di citazioni.
* **Soluzione:** Utilizzare una scala logaritmica: `score = log(1 + citazioni) / log(1 + max_citazioni)`.

### 2.3 Robustezza dell'Interazione LLM
* **Parsing JSON:** I modelli possono aggiungere testo discorsivo fuori dai tag JSON.
* **Soluzione:** Utilizzare *Structured Outputs* (API native) o espressioni regolari (Regex) per estrarre il contenuto tra le parentesi graffe.

### 2.4 Gestione API e Rete
* **Rate Limiting:** Lo sleep statico è inefficiente.
* **Soluzione:** Implementare l'**Exponential Backoff** con librerie come `tenacity`.

---

## 3. Framework per la Decomposizione del Pensiero
La scomposizione del problema può essere potenziata integrando framework accademici e strategici:

* **Principio MECE:** "Mutually Exclusive, Collectively Exhaustive". Assicura che le sotto-domande coprano tutto l'argomento senza sovrapporsi.
* **Tree of Thoughts (ToT):** Il modello esplora più rami di ragionamento, valuta le opzioni e sceglie la via più rigorosa.
* **Graph of Thoughts (GoT):** Permette la combinazione di pensieri non lineari.
* **First Principles Thinking:** Decostruzione del problema fino alle verità fondamentali.

---

## 4. Implementazione del Framework "Tree of Thoughts" (ToT)

Per migliorare drasticamente lo Step 1 (Decomposizione), è stato proposto di modificare il prompt di sistema per includere una fase di "scratchpad" o analisi interna:

1.  **Analisi Interna:** Il modello genera 3 approcci diversi all'interno di tag `<analisi>`.
2.  **Valutazione:** Sceglie l'approccio più adatto ai database accademici.
3.  **Output Pulito:** Restituisce il JSON finale fuori dai tag.

*Nota tecnica: La funzione `decompose_thought` deve essere aggiornata con `re.search(r'\{.*\}', raw, re.DOTALL)` per estrarre il JSON in modo sicuro.*

---

## 5. Verso una Ricerca Professionale: Sintesi Gerarchica

Per superare la "piattezza" dei riassunti generati in un unico passaggio, è necessaria una **Generazione Gerarchica (Iterative Drafting)**:

### 5.1 Generazione dell'Indice (Outline)
Invece di scrivere subito, l'LLM progetta la struttura del paper (5-7 sezioni) e assegna i paper specifici a ogni capitolo.

### 5.2 Scrittura Modulare (Sezione per Sezione)
Il sistema esegue un loop sulle sezioni dell'indice:
* Invia all'LLM solo i dati relativi a quella sezione.
* Richiede un output lungo (800-1000 parole) per ogni capitolo.
* Questo garantisce una profondità di analisi impossibile con un approccio monolitico.

### 5.3 Self-Reflection (Critica e Revisione)
Aggiunta di un passaggio in cui un "agente critico" analizza la bozza di ogni sezione per verificare:
* Rigore del confronto tra fonti.
* Correttezza delle citazioni.
* Tono accademico.

### 5.4 Arricchimento dei Dati
Aumentare il contesto fornito all'LLM includendo l'abstract completo, i metadati di OpenAlex (concetti chiave) e le affiliazioni degli autori per permettere un'analisi critica basata sull'autorevolezza delle fonti.

---
*Documento generato per sintetizzare l'evoluzione della Research Pipeline.*
