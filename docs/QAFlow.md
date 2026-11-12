# QA Flow — Hybrid Retrieval Pipeline (HyDE + Keywords + Reranking)

## Overview

The `docQA` flow answers user questions against indexed documents (PDFs and web pages). It uses a 4-step pipeline with three complementary retrieval strategies to maximize recall: vector search with the original question (lexical similarity), vector search with a HyDE-generated passage (semantic similarity), and keyword search directly on the database file (exact term matching).

```
Question
  |
  v
[Step 1] HyDE — LLM generates hypothetical answer passage
  |
  v
[Step 2] Hybrid Retrieve
  |--- Vector search with original question (k=20)
  |--- Vector search with HyDE passage (k=20)
  |--- Keyword search on DB file (top 10)
  |--- Merge + deduplicate (max 25 unique)
  |
  v
[Step 3] Rerank — LLM scores each chunk 0-10 for relevance
  |--- Fallback: top-3 if reranker filters everything
  |
  v
[Step 4] Answer — top chunks become context, LLM streams final answer
```

## Step 1: HyDE (Hypothetical Document Embedding)

**Problem:** Embedding a short question like "Was kostet das Enterprise-Paket?" produces a vector in "question space" that is distant from the vector of a passage that states "Das Enterprise-Paket kostet 499 EUR pro Monat" in "passage space". This asymmetry causes relevant chunks to be missed.

**Solution:** Before retrieval, the LLM generates a hypothetical answer — a short, fact-rich paragraph (3-5 sentences) that *could* answer the question. This fabricated passage is never shown to the user; it only serves as a search query for the second retrieval path.

```
System prompt:
  "Schreibe einen kurzen, faktenreichen Absatz (3-5 Sätze),
   der die folgende Frage beantworten könnte.
   Erfinde plausible Details.
   Der Text wird nur zur Suche verwendet, nicht als Antwort."

Input:  "Was kostet das Enterprise-Paket?"
Output: "Das Enterprise-Paket wird typischerweise zu einem Preis von
         mehreren hundert Euro pro Monat angeboten. Es enthält erweiterte
         Funktionen wie dediziertes Support-Team, höhere API-Limits und
         individuelle SLAs. Die Preisgestaltung kann je nach Vertragslaufzeit
         und Nutzerzahl variieren."
```

The hypothetical passage's embedding lands much closer to actual document passages in vector space, because it has the same structure — declarative sentences with domain terminology.

**Fallback:** If the HyDE generation fails, the original question is used as the search query. The HyDE vector search path is skipped (no duplicate query).

**Logging:** `[docQA] hyde="..."` (first 100 chars of generated text)

## Step 2: Hybrid Retrieval

A single retrieval strategy can miss relevant documents — vector search depends on embedding quality, while keyword search depends on exact term overlap. The hybrid approach combines three complementary methods to maximize recall.

### 2a. Vector Search — Original Question (k=20)

The raw question is embedded and used to fetch **k=20** nearest neighbours from the localvec vector store. This path catches documents that share vocabulary or phrasing with the question.

### 2b. Vector Search — HyDE Passage (k=20)

If the HyDE passage differs from the original question (i.e., HyDE generation succeeded), it is embedded as a separate query to fetch another **k=20** candidates. This path catches documents that are semantically related but use different wording than the question.

### 2c. Keyword Search — Direct DB Scan (top 10)

The keyword search (`keywordSearch` in `search.go`) bypasses the embedding model entirely:

1. **Extract keywords** from the question: split on non-letter/digit boundaries, lowercase, filter stop words (German + English) and tokens shorter than 3 characters
2. **Read the localvec DB file** (`__db_docQA.json`) directly
3. **Score each document** by counting how many keywords appear (via `strings.Contains`) in the chunk text + parentText
4. **Return top 10** by match count

This path is critical for cases where embedding similarity fails but the question and document share the same terms — e.g., proper nouns, technical terms, or specific phrases that the embedding model doesn't represent well.

### Merge and Deduplication

Results from all three paths are merged and deduplicated by chunk content text. The combined pool is capped at **25 unique documents** for reranking. Documents from the original-question vector search come first (preserving vector similarity order), followed by HyDE results, followed by keyword results.

**Logging:** `[docQA] retrieved X docs (question) + Y docs (hyde) + Z docs (keyword), N unique after merge`

## Step 3: LLM Reranking

**Problem:** Vector similarity is a blunt instrument. Chunks about tangentially related topics can score high because they share vocabulary (e.g., a chunk about "Enterprise-Support" when the question asks about "Enterprise-Preise"). Keyword search can similarly surface chunks with matching terms but irrelevant content. Pure retrieval has no way to distinguish these.

**Solution:** The LLM reads each chunk (truncated to 500 chars) alongside the original question and assigns a relevance score from 0 to 10.

**Prompt format:**
```
Frage: Was kostet das Enterprise-Paket?

Bewerte jeden der folgenden Textabschnitte auf einer Skala von 0 bis 10,
wie relevant er für die Beantwortung der Frage ist.
Antworte NUR mit einer Zeile pro Abschnitt im Format: <nummer>:<score>
Beispiel: 1:8

--- Abschnitt 1 ---
Das Enterprise-Paket kostet 499 EUR pro Monat...

--- Abschnitt 2 ---
Unser Support-Team ist 24/7 erreichbar...
```

**Expected response:**
```
1:9
2:2
```

**Score parsing** (`parseRerankScores`):
- Parses `<number>:<score>` lines, ignoring non-numeric characters
- 1-based numbering from the LLM is converted to 0-based indices
- Unparseable lines or missing entries default to score 5
- Scores are capped at 10

**Filtering and sorting:**
- Chunks are sorted by score descending
- Chunks with score < 2 are discarded (clearly irrelevant)
- Only the top 10 are kept for the final context
- Deduplication by `parentText` still applies (same as before)

**Fallback — reranker filters everything:** If all candidates score below `minScore` (kept == 0), the top 3 documents from the original retrieval order are used regardless of score. This prevents the pipeline from returning "no results" when documents were retrieved but the reranker was too aggressive.

**Fallback — reranker call fails:** If the reranking LLM call fails entirely, the original retrieval order is preserved with a default score of 5 (no filtering by score).

**Logging:**
- `[docQA] rerank scores: [8 6 3 1 ...]`
- `[docQA] reranker filtered everything — using top-3 fallback` (when applicable)
- `[docQA] rerank failed, using original order (err=...)` (when applicable)
- `[docQA] kept=N docs for final answer`

## Step 4: Context Assembly and Streaming Answer

The surviving chunks are assembled into the context prompt:

- `parentText` (expanded ~3000 char context) is used when available, otherwise the chunk text itself
- Each chunk is prefixed with source metadata:
  - PDFs: `[Seite 3-5 | Kapitelname | Tag: Vertrag | Datum: 2025-01-15]`
  - Web: `[Quelle: https://... | Abschnitt: Products > Enterprise > Pricing | Tag: ... | Datum: ...]`
- Deduplicated by parentText content

The final prompt is sent to the LLM with streaming enabled. The system prompt instructs the LLM to cite sources and only answer from the provided context.

If no documents survive filtering (even after the top-3 fallback), the flow returns: "Es wurden keine relevanten Informationen in den indexierten Dokumenten gefunden."

## Ollama Model Configuration

The model is registered with `Type: "chat"`, which causes the Genkit Ollama plugin to use the `/api/chat` endpoint. This is important for two reasons:

1. **Correct prompt structure:** The dotprompt templates use `{{role "system"}}` and `{{role "user"}}`, which map directly to chat message roles
2. **Avoids scanner overflow:** The `/api/generate` endpoint includes a `context` token-ID array in its streaming response that can exceed the 64KB `bufio.Scanner` limit with large prompts. `/api/chat` does not have this issue.

## Cost and Latency

| Step | LLM Calls | Embed Calls | DB Reads | Typical Latency |
|------|-----------|-------------|----------|-----------------|
| HyDE | 1 (short generation) | 0 | 0 | 1-3s |
| Retrieve (question) | 0 | 1 | 0 | <1s |
| Retrieve (HyDE) | 0 | 1 | 0 | <1s |
| Keyword search | 0 | 0 | 1 (JSON file) | <0.1s |
| Rerank | 1 (structured scoring) | 0 | 0 | 2-5s |
| Answer | 1 (streamed) | 0 | 0 | varies |
| **Total** | **3** | **2** | **1** | **+3-8s overhead** |

The overhead of 2 extra LLM calls (HyDE + rerank) and a keyword search compared to a single-call approach is typically 3-8 seconds with a local Ollama model. The improvement in retrieval quality — especially for questions that use different vocabulary than the source documents — generally justifies this cost.

## Tuning Parameters

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| HyDE passage length | `prompts/hyde.prompt` | "3-5 Sätze" | Longer passages may match more diverse chunks |
| Retrieval k (per query) | Step 2, `"k": 20` | 20 | More candidates = better recall, slower rerank |
| Keyword search max results | Step 2, `keywordSearch(..., 10)` | 10 | More keyword matches in candidate pool |
| Merge cap | Step 2, `merged[:25]` | 25 | Max unique docs sent to reranker |
| Chunk truncation for rerank | Step 3, `text[:500]` | 500 chars | More text = better scoring, larger prompt |
| Min rerank score | Step 3, `minScore` | 2 | Lower = more permissive, higher = stricter |
| Max results after rerank | Step 3, `maxResults` | 10 | More = richer context, longer final prompt |
| Reranker fallback count | Step 3, fallback loop | 3 | Docs kept when reranker filters everything |
| Default score on parse failure | `parseRerankScores` | 5 | Affects chunks when rerank output is malformed |
| Keyword min length | `extractKeywords` | 3 chars | Shorter = more keywords, noisier matching |

## Files

| File | Responsibility |
|------|---------------|
| `main.go` | docQA flow (Steps 1-4), model/embedder/retriever setup |
| `search.go` | `keywordSearch`, `extractKeywords`, stop word list, localvec DB parsing |
| `prompts/hyde.prompt` | HyDE system prompt and input schema |
| `prompts/rerank.prompt` | Reranker system prompt, scoring format, input schema |
| `prompts/qa.prompt` | Final answer system prompt with citation rules |
| `document.go` | PDF parsing, paragraph splitting, `buildChunks` with parent-text expansion |
| `web.go` | HTML → structured paragraphs (headings, tables, lists) |
