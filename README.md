# AI Content Reference Checker (RAG-style)

Minimal prototype that accepts a text input, searches Semantic Scholar / CrossRef for related publications, computes semantic similarity between the input passages and candidate article abstracts/titles, and returns matches as `passage | article quoted` with scores and metadata.

Quick start

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the API server:

```bash
uvicorn app.main:app --reload
```

3. Open `http://127.0.0.1:8000` and use the simple frontend or POST to `/api/check`.

Notes
- This is a prototype. It searches Semantic Scholar (public API) and CrossRef for metadata and compares input passages against titles+abstracts using `sentence-transformers` embeddings. Full-text matching requires additional sources (publisher APIs / Unpaywall) and is not implemented here.

Overview
--------

This repository contains a minimal RAG-style application to help determine whether a piece of text closely matches or duplicates information in scholarly publications. It performs the following high-level steps:

- Split input text into smaller passages.
- Query public scholarly APIs (Semantic Scholar and CrossRef) to retrieve candidate publications (title, abstract, DOI, URL).
- Compute semantic embeddings for passages and article texts using `sentence-transformers`.
- Index candidate article embeddings in a FAISS vector store for efficient nearest-neighbor retrieval.
- Rank and score candidate articles using a weighted combination of passage similarity, title overlap, DOI presence, and OA (Unpaywall) signals.
- Return results grouped as `passage_matches` (best article per passage) and `global_matches` (top articles for full input).

Architecture & Key Files
------------------------

- `app/main.py`: FastAPI application; serves a tiny static frontend and exposes `/api/check`.
- `app/scholarly.py`: Helpers to query Semantic Scholar and CrossRef APIs.
- `app/unpaywall.py`: Optional Unpaywall client to fetch OA metadata and (best-effort) HTML excerpts.
- `app/embeddings.py`: Wraps `sentence-transformers` model loading and embedding computation.
- `app/vectordb.py`: `FaissIndexManager` — a small wrapper around FAISS for adding/searching vectors and storing metadata.
- `app/utils.py`: End-to-end pipeline that splits passages, retrieves candidates, computes embeddings, indexes candidates in FAISS, and computes ranking scores.
- `static/index.html`: Minimal UI to paste text and view results.

Setup
-----

1. Create and activate a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. (Optional) Configure Unpaywall to surface OA links and small text excerpts:

```bash
export UNPAYWALL_EMAIL="your.email@example.com"
export UNPAYWALL_FETCH=true   # optional: attempts to fetch HTML text from OA link
```

Run the server:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser to use the simple frontend.

API
---

- POST `/api/check` — body: `{ "text": "..." }`
	- Response: JSON with keys `query` and `matches`.
	- `matches` contains:
		- `passage_matches`: array of objects `{ passage, score, article: { title, authors, year, doi, url, excerpt, unpaywall } }` — best article per passage.
		- `global_matches`: array of top articles for the full input, with combined scores and metadata.

How matching & ranking works
---------------------------

1. Passage splitting: `app/utils.py::split_into_passages` performs a naive sentence split and groups sentences into passages of ~300 characters.
2. Candidate retrieval: queries are issued to Semantic Scholar and CrossRef using `app/scholarly.py`. Results are deduplicated by DOI/title.
3. Embeddings: `app/embeddings.py` uses `sentence-transformers` (`all-MiniLM-L6-v2`) to produce normalized embeddings.
4. Vector DB: candidate article embeddings are indexed in FAISS via `app/vectordb.py::FaissIndexManager` for efficient nearest-neighbor search. The index persists under `data/index.faiss` and metadata under `data/meta.json`.
5. Scoring: For each candidate we compute a combined score:

- passage similarity (best passage → article) — weighted heavily (default 0.7)
- title overlap (token overlap between input and title) — moderate weight (0.2)
- DOI presence and OA presence as small boosts (0.05 each)

The final ranking uses this weighted sum plus a global similarity measure to present `global_matches`.

Unpaywall / OA support
----------------------

When `UNPAYWALL_EMAIL` is set, the app will call Unpaywall for DOIs found in search results and will attach OA metadata (`is_oa`, `best_oa_location`, etc.) to articles. When `UNPAYWALL_FETCH=true`, the app attempts to fetch a short HTML text excerpt from the OA URL (best-effort) and includes it in the indexing step (improves matching when abstracts are missing).

Persistence & scaling notes
--------------------------

- The FAISS index provides quick vector similarity search. The current implementation appends candidates to the index on each run — in a production workflow you'd maintain a canonical corpus, deduplicate before indexing, and use an index sharding/cleanup strategy.
- For large-scale usage consider using a persistent vector DB service (Milvus, Pinecone, Weaviate) and a document ingestion pipeline.

Limitations & Legal / ToS
-------------------------

- This prototype relies on public APIs (Semantic Scholar, CrossRef, Unpaywall). Be mindful of their rate limits and terms of service.
- We do not scrape Google Scholar. The app does not include Google Scholar scraping because of ToS and reliability concerns. For comprehensive, publisher-level full-text checks you will need publisher agreements or paywalled access.
- Matching accuracy depends on the quality of abstracts/titles and may miss paraphrased or partial matches; full-text access improves accuracy.

Testing & Debugging
-------------------

- Unit tests are not included in this prototype. To smoke test locally:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
curl -s -X POST http://127.0.0.1:8000/api/check -H 'Content-Type: application/json' -d '{"text":"Transformers for NLP and neural networks."}' | jq
```

Troubleshooting
---------------

- If you get import errors from `sentence-transformers` or `huggingface_hub`, pin the compatible `huggingface_hub` version (this repo pins `0.13.4`).
- Large model downloads happen the first time embeddings are computed — be patient.
- If FAISS fails to install on your platform, try `pip install faiss-cpu` or use a hosted vector DB instead.

Next steps
----------

- Add persistent corpus ingestion and a deduplicated index build step.
- Improve passage splitting (semantic-aware chunking) and scoring (learned ranking).
- Add unit tests and a small CI configuration.
- Integrate publisher APIs or Unpaywall-expanded scraping only where permitted.

File map
--------

- `app/main.py` — API entrypoints and simple static frontend serving.
- `app/scholarly.py` — Semantic Scholar & CrossRef clients.
- `app/unpaywall.py` — Unpaywall client and HTML fetch helper.
- `app/embeddings.py` — Embedding model loader and encoder.
- `app/vectordb.py` — FAISS wrapper for indexing/searching embeddings.
- `app/utils.py` — Pipeline glue: retrieval, embedding, indexing, ranking.
- `static/index.html` — Tiny UI to paste text and view results.

Contact
-------

If you want, I can add tests, CI, or wire a persistent ingestion pipeline next.


Unpaywall (optional)

To surface open-access full-text links for DOIs, set the environment variable `UNPAYWALL_EMAIL` to a contact email and optionally set `UNPAYWALL_FETCH=true` to attempt fetching the OA HTML text (best-effort):

```bash
export UNPAYWALL_EMAIL="your.email@example.com"
export UNPAYWALL_FETCH=true
```

When enabled, the server will query Unpaywall for DOIs returned by Semantic Scholar / CrossRef and include OA metadata and an optional small text excerpt when available.
