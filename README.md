# ask-pdf — Document and Webpage RAG

A Retrieval-Augmented Generation (RAG) application built with **Go**, **Firebase Genkit**, and **PostgreSQL/pgvector**. It answers questions using **PDF documents and webpage content that has already been indexed** into its knowledge base. It supports university-hosted OpenAI-compatible models and local Ollama models, conversational follow-ups, document tagging, and source citations.

> **Grounding policy:** Technical and factual answers are based on indexed PDF/webpage content, not live web search. If the indexed evidence is insufficient, the assistant may return `NOT IN DOCUMENT` or an explanatory abstention. Simple conversational messages (such as greetings and thanks) are handled without retrieval.

## Features

- **PDF ingestion:** Extracts pages, structures paragraphs, builds overlapping chunks, and retains page/section metadata.
- **Webpage ingestion and crawling:** Converts HTML into structured text and indexes it with source URLs, headings, and breadcrumbs.
- **Hybrid retrieval:** Combines dense vector similarity with optional keyword search and rank/score fusion.
- **Query expansion or HyDE:** Optional additional retrieval queries to improve recall.
- **Automatic tag routing (P15D):** Uses retrieval evidence and tag profiles to select relevant document collections; supports explicit tag filtering.
- **LLM reranking and evidence selection:** Reranks candidates, removes duplicate contexts, and applies configurable evidence checks and heading/section matching.
- **Grounded answers and citations:** Associates inline evidence citations with indexed source records and checks answer support.
- **Conversation support (P20/P21):** Rewrites context-dependent follow-ups into standalone retrieval questions and bypasses retrieval for a limited set of recognized conversational messages.
- **Evaluation support:** Integrates Genkit evaluators, including custom faithfulness and answer-relevancy checks where configured.

## Architecture

```text
PDF uploads / indexed webpages / crawled pages
                 |
        Text extraction + chunking
                 |
              Embeddings
                 |
          PostgreSQL + pgvector
                 |
User message --> Conversational-message bypass (P21-A)
                 |
           Follow-up resolution (P20-A)
                 |
        Automatic tag selection (P15D)
                 |
      Vector retrieval + optional keyword search
                 |
       Optional HyDE / question expansion
                 |
      Fusion + heading/section boosts
                 |
         Optional LLM reranking
                 |
        Evidence/context selection
                 |
     Grounded LLM answer + source citations
```

The code can use university-hosted models for generation and embeddings, or Ollama for supported local-model experiments. The particular models in use depend on `.env`; they are **not automatically downloaded by this application** when using the university API.

## Prerequisites

- A compatible **Go** installation (use the version required by the repository's `go.mod`).
- **PostgreSQL with the `pgvector` extension**, configured for the application's schema.
- Access to an LLM and embedding service: the university API and its credentials, or an Ollama server with the required model(s).
- **Genkit CLI** (optional, for the Developer UI and evaluation workflows).
- Docker / Docker Compose (optional, when using the repository's container configuration).

For a university-hosted model, the inference endpoint must actually serve the selected model. Being listed at `/models` does not guarantee that `/embeddings` is operational.

## Setup

Clone the repository and enter the project directory:

```bash
git clone <your-gitlab-repository-url>
cd ask-pdf-clean

go mod download
```

Create a local `.env` file (or copy a provided example and edit it):

```bash
cp .env.example .env
```

If `.env.example` is not included in your checkout, create `.env` manually. **Do not commit real API keys or database passwords.**

### Example: university-hosted generation and embeddings

```dotenv
# Generation model (choose an ID available through your university API)
LLM_MODEL=university/gemma4:31b
UNI_API_BASE_URL=https://dbhage.prakinf.tu-ilmenau.de/api/v1
UNI_API_KEY=YOUR_UNIVERSITY_API_KEY

# Embeddings: use a model actually enabled on the university server
EMBEDDING_PROVIDER=university
EMBEDDING_MODEL=nomic-embed-text-v1.5
EMBEDDING_DIM=768

# PostgreSQL: adjust host, credentials and DB name to your deployment
DATABASE_URL=postgres://askpdf:YOUR_PASSWORD@localhost:5432/askpdf?sslmode=disable

# Retrieval options
AUTO_TAG_ROUTING=true
AUTO_TAG_MODE=p15d
USE_KEYWORD=true
USE_QUERY_EXPANSION=true
USE_RERANKER=true
USE_CONVERSATIONAL_RESOLUTION=true
CONVERSATION_HISTORY_MESSAGES=4
```

This is an **illustrative configuration**, not a guarantee that the university embedding service is currently available. Use the exact model IDs offered by your deployment and ensure the DB schema matches the embedding width. In the current code, `nomic-embed-text-v1.5` requires 768 dimensions, and `bge-m3` requires 1024 dimensions when provided by the university service.

### Example: local Ollama embeddings

If the university embedding endpoint is unavailable, the Go code also supports `EMBEDDING_PROVIDER=ollama`:

```dotenv
EMBEDDING_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=<your-installed-ollama-embedding-model>
EMBEDDING_DIM=<that-models-output-dimension>
```

Install the selected embedding model in Ollama before indexing. Model names and output dimensions must match the running Ollama instance. Switching embedding models or dimensions requires a compatible database schema and **re-indexing**, since embeddings from different models cannot be mixed.

**Note on Qwen3-Embedding-4B:** The university model registry reported `qwen3-embedding:4b` with a 2560-dimensional embedding width. Standard pgvector `vector` indexing has dimensional constraints (including a 2000-dimension HNSW limit); do not switch to 2560 dimensions without planning a compatible storage/index approach (such as `halfvec` or a supported lower-dimensional output).

## Running locally

```bash
go mod tidy
go run .
```

The Go application normally serves the web interface at **http://localhost:3400**. Override the binding with `LISTEN_ADDR` if needed; the default in `main.go` is `127.0.0.1:3400`.

For Genkit's Developer UI:

```bash
genkit start -- go run .
```

- Application: http://localhost:3400
- Genkit Developer UI: http://localhost:4000

If the first run reports missing `go.sum` entries for `compat_oai`, run `go mod tidy`, then `go build ./...`.

## Using the application

### Web UI

Open **http://localhost:3400** to upload PDFs, add webpage URLs, and ask questions about content already indexed in the database.

### HTTP endpoints

The following route names are registered in the Go application. Request-body details may depend on the corresponding handlers in `api.go`.

**Index a PDF:**

```bash
curl -X POST http://localhost:3400/addPDFDocument \
  -H 'Content-Type: application/json' \
  -d '{"path":"/absolute/path/to/file.pdf","tag":"amcn"}'
```

**Index a webpage:**

```bash
curl -X POST http://localhost:3400/addWebDocument \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/page","tag":"web"}'
```

**Crawl webpages:** The Go application also registers `POST /crawlWeb`. See its request type/handler for supported crawl options.

**Ask a question:**

```bash
curl -X POST http://localhost:3400/askPDF \
  -H 'Content-Type: application/json' \
  -d '{"data":{"question":"What is a Home Agent?"}}'
```

The underlying `docQA` flow accepts a `question` and optional `tag` and `history`. If your `api.go` adapter uses a different request envelope, follow that route's current handler or use the web UI / Genkit Developer UI.

Additional application routes may include listing sources, keyword search, and database reset, depending on the checked-out `api.go` version. A keyword-search endpoint is advertised by `main.go` as `/api/keyword-search`.

## Retrieval and grounding

The retrieval pipeline is configurable rather than fixed to one strategy:

1. Handle simple conversational messages directly (P21-A).
2. Optionally resolve history-dependent questions using recent user/assistant turns (P20-A).
3. Select relevant document tags automatically when no explicit tag is supplied (P15D).
4. Retrieve by vector similarity, optionally supplemented by question expansion **or** HyDE and keyword search.
5. Fuse and deduplicate results; optionally boost matching headings/sections and demote conflicting sibling sections.
6. Optionally rerank the candidate chunks with the LLM.
7. Select final context using rerank thresholds, overlap/evidence checks, and optional P17-E2 unsupported-fact validation.
8. Generate an answer from the selected evidence, validate inline citations, and return the associated sources.

A missing answer in the indexed material should not be replaced by an unsupported answer from the model's general knowledge. The application **does not perform live web search** simply because it can index webpages.

## Configuration reference

| Variable | Purpose |
|---|---|
| `LLM_MODEL` | Generation model, e.g. `university/gemma4:31b`, `university/qwen3.8:27b`, or an Ollama model with the `ollama/` prefix. |
| `UNI_API_BASE_URL` | OpenAI-compatible university API base URL. |
| `UNI_API_KEY` | University API bearer token; keep private. |
| `EMBEDDING_PROVIDER` | `university` or `ollama`. |
| `EMBEDDING_MODEL` | Embedding model ID. |
| `EMBEDDING_DIM` | Embedding size; must match the model and pgvector storage. |
| `OLLAMA_HOST` | Address of local Ollama service, when used. |
| `DATABASE_URL` | PostgreSQL connection string (used by the pgvector store). |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Chunking parameters for PDF and webpage indexing. |
| `USE_KEYWORD` | Enable optional lexical/keyword retrieval. |
| `USE_HYDE` | Enable hypothetical-document retrieval expansion. |
| `USE_QUERY_EXPANSION` | Enable question-expansion retrieval; the current flow uses this when HyDE is off. |
| `QUERY_EXPANSION_K`, `QUERY_EXPANSION_RETRIEVE_K` | Number of additional expansion queries and results per query. |
| `FUSION_METHOD` | `simple`, `score`, `rrf`, `borda`, `combmnz`, `combmnz_kw`, or `combmnz_rank_kw`. |
| `USE_RERANKER`, `RERANK_MIN_SCORE` | Enable LLM reranking and control its minimum score. |
| `FINAL_CONTEXT_K`, `MAX_CONTEXT_CHARS` | Final evidence count and optional context character cap. |
| `AUTO_TAG_ROUTING`, `AUTO_TAG_MODE` | Automatic tag selection and routing approach. |
| `ALLOWED_TAGS` | Optional explicit whitelist of indexed content tags. |
| `USE_PRERERANK_HEADING_BOOST` | Enable heading/section-aware pre-rerank boosting. |
| `USE_P17B_EXACT_SECTION_BOOST` | Boost direct section/title matches. |
| `USE_P17B_SIBLING_DEMOTION` | Demote competing sibling headings. |
| `USE_FINAL_HEADING_PROTECTION`, `USE_FINAL_HEADING_FORCE_INCLUDE` | Protect or force strong heading matches into final evidence. |
| `USE_P17E2_UNSUPPORTED_VALIDATION` | Narrow final support checks for selected precise-fact questions. |
| `USE_CONVERSATIONAL_RESOLUTION` | Enable follow-up question rewriting. Enabled by default in current `main.go`. |
| `CONVERSATION_HISTORY_MESSAGES` | Number of recent messages considered for follow-up rewriting (default 4; capped at 8). |
| `RAG_VERBOSE_LOGS` | Enable diagnostic logging. |
| `LISTEN_ADDR` | Application listen address (default `127.0.0.1:3400`). |

For available tuning thresholds and defaults, refer to the `.env` used for your experiment and the `setup()` function in `main.go`.

## Evaluation

The repository integrates Genkit evaluation capabilities, including a regex evaluator and custom quality evaluators where registered in `eval.go`.

Start the Genkit Developer UI:

```bash
genkit start -- go run .
```

Run an evaluation with **your own retained dataset** (historical experiment directories may have been removed from the final application repository):

```bash
genkit eval:flow docQA --input /path/to/your/eval_dataset.json
```

The evaluated questions must correspond to content already indexed in the target database. Evaluation results depend on the configured generation model, embedding model, retrieval settings, and indexed corpus. Store these settings alongside reported results to make comparisons reproducible.

## Project layout

```text
main.go           Genkit initialization; ingestion, crawl, and QA flows;
                  retrieval, routing, reranking, follow-up handling, citations
api.go            HTTP routes, upload endpoints, and request/response types
pgstore.go        PostgreSQL/pgvector storage, retrieval, and metadata filters
document.go       PDF extraction and chunking
web.go            HTML extraction and webpage chunking
search.go         Keyword retrieval helpers
eval.go           Evaluation integrations and custom metrics
prompts/          Genkit prompt templates (e.g., HyDE, reranking, QA)
static/           Browser UI
postgres/         Database initialization files, if present in the checkout
```

This describes the application's source layout. Historical phase-by-phase experiment results, datasets, and generated plots are **not required** in the final application repository. Keep thesis evidence and evaluation archives separately if needed for reproducibility.

## Database and re-indexing

Use the PostgreSQL/pgvector schema required by the current project. The vector dimensions must match the selected embedding model. Changing the embedding model, embedding width, or index-compatible storage representation requires checking the schema and typically rebuilding embeddings for the indexed corpus.

If your checkout includes `docker-compose.yml` / `compose.yaml`, use the actual services defined there (for example, `docker compose up -d postgres`). Do not assume the previous Ollama-based full-stack Compose configuration still matches a university-hosted deployment.

**Reset warning:** Database-reset operations remove indexed content. Back up valuable data before performing a reset, and re-upload/re-crawl your knowledge sources afterward.

## Security and GitLab

Keep `.env`, credentials, user-uploaded documents, and compiled binaries out of commits. A typical `.gitignore` includes:

```gitignore
.env
.env.*
!.env.example
/uploads/
/ask-pdf
```

If `.env` was ever committed previously, adding it to `.gitignore` does not remove existing Git history. Revoke or rotate any exposed credentials and remove the tracked file from the index with `git rm --cached .env`.
