# OCI Arabic RAG Toolkit

> Production Arabic retrieval-augmented generation on Oracle Cloud
> Infrastructure. Benchmarks, utilities, and a reference implementation
> for building enterprise AI in Arabic on OCI.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen.svg)](tests/)

## Why this exists

Arabic is the fifth most-spoken language in the world and a first-class
business language across MENA. But every team building Arabic RAG on OCI
keeps re-solving the same problems: which embedding model actually works
on Arabic? How do you chunk Arabic regulatory text without breaking
sentences? What's the right normalization pipeline for retrieval?

This toolkit ships the utilities and the benchmarks. It's designed as a
canonical starting point for Arabic enterprise AI projects on OCI —
banking, government services, healthcare, telecom.

## What's inside

- **Arabic-aware utilities** — RTL-safe chunking with section detection
  for regulatory text (`المادة 1:` …), diacritic normalization, alef and
  yaa folding, Arabic-Indic digit conversion.
- **Pluggable embedding adapters** — TF-IDF baseline (offline), OCI
  Generative AI (`cohere.embed-multilingual-v3.0`), and HuggingFace
  sentence-transformers.
- **Hybrid retrieval** — dense + BM25 with tunable α, so you can
  reproduce the trade-offs on your own corpus.
- **Reproducible benchmark** — Arabic eval set on synthetic regulatory
  text, with Precision@k, Recall@k, MRR, and nDCG.
- **Reference Q&A example** — full pipeline from raw text to
  OCI-Generative-AI-ready chat messages.

## Headline benchmark result

The same toolkit, run on two different Arabic corpora — banking
regulations and legal text — with no code changes between runs:

### Banking (13 docs · 25 queries)

| Retriever          | P@1   | R@5   | nDCG@5 | MRR   | Latency |
| ------------------ | :---: | :---: | :----: | :---: | :-----: |
| TF-IDF (char 3-5)  | 0.92  | 1.00  | 0.97   | 0.96  | 0.6 ms  |
| BM25               | 0.72  | 0.92  | 0.83   | 0.80  | 0.1 ms  |
| Hybrid (α=0.6)     | 0.88  | 1.00  | 0.95   | 0.93  | 0.8 ms  |

### Legal (13 docs · 25 queries)

| Retriever          |  P@1   | R@5   | nDCG@5 | MRR   | Latency |
| ------------------ | :----: | :---: | :----: | :---: | :-----: |
| TF-IDF (char 3-5)  | **0.96** | 1.00 | 0.98 | 0.97  | 0.6 ms  |
| BM25               | 0.56   | 0.84  | 0.70   | 0.66  | 0.1 ms  |
| Hybrid (α=0.6)     | 0.80   | 1.00  | 0.92   | 0.89  | 0.7 ms  |

**Two findings that matter for production:**

1. **TF-IDF on character n-grams beats BM25 word tokenization on Arabic.**
   Arabic morphology punishes word-level approaches because the same
   root (e.g. ك-ت-ب) appears in many surface forms (يكتب, كاتب, كتاب,
   مكتوب) that word tokenizers treat as unrelated.

2. **The gap widens on more morphologically rich domains.** Legal
   Arabic uses more classical, formal language with deeper morphological
   variation than banking — and the TF-IDF advantage grows from
   +20 points (banking) to +40 points (legal) at P@1.

This matters for production: don't assume English RAG defaults
transfer. Run the benchmark on your corpus before committing to a
retriever.

Reproduce locally:

```bash
pip install -r requirements.txt
python benchmarks/run_benchmark.py --corpus all
```

Full results in [`benchmarks/results/BENCHMARK.md`](benchmarks/results/BENCHMARK.md).

## Quickstart

```bash
git clone https://github.com/YOUR-USERNAME/oci-arabic-rag-toolkit
cd oci-arabic-rag-toolkit
pip install -r requirements.txt

# Run the unit tests
python tests/test_normalization.py

# Run the benchmark on synthetic Arabic banking regulations
python benchmarks/run_benchmark.py

# Run the full RAG pipeline end-to-end on a sample query
python examples/compliance_qa.py --query "ما هي نسبة تغطية السيولة المطلوبة؟"
```

To use the OCI Generative AI embedding model:

```bash
pip install oci
export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxxx"
python benchmarks/run_benchmark.py --retriever oci
```

## Library usage

```python
from arabicrag import (
    chunk_documents, normalize, TfidfEmbedder,
    VectorIndex, HybridRetriever, build_messages,
)

docs = [{"id": "policy_1", "text": "المادة 1: يلتزم البنك ..."}]

# Normalize and chunk
normalized = [{"id": d["id"], "text": normalize(d["text"])} for d in docs]
chunks = chunk_documents(normalized, mode="section", target_chars=500)

# Index and retrieve
index = HybridRetriever(VectorIndex(TfidfEmbedder()), alpha=0.6)
index.index(chunks)
results = index.search(normalize("ما هي متطلبات رأس المال؟"), k=5)

# Build OCI-Generative-AI-ready chat messages
messages = build_messages("ما هي متطلبات رأس المال؟", [r.chunk for r in results])
```

## Architecture

```
┌─────────────────┐
│  Arabic docs    │  (PDFs, DOCX, web — your ingest)
│  (regulatory)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Normalize      │  diacritics, alef/yaa, digits
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Section-aware  │  المادة 1: …  →  one chunk per article
│  chunker        │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Embed (TF-IDF | OCI | HF) + index  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐    ┌──────────────────────┐
│  Hybrid search  │ ── │  OCI Generative AI   │ ──► Arabic answer
│  (vec + BM25)   │    │  (cohere / llama /   │     with citations
└─────────────────┘    │   imported model)    │
                       └──────────────────────┘
```

For production: swap the in-memory `VectorIndex` for **Oracle Database
26ai AI Vector Search** (`VECTOR_DISTANCE`, `DBMS_VECTOR`) and replace
the local prompt-builder call with **OCI Generative AI Agents** for
multi-turn workflows. Reference SQL is included in
[`src/arabicrag/retrieval.py`](src/arabicrag/retrieval.py).

## Who this is for

- **Solutions Engineers** demoing Arabic AI to MENA customers
- **Data scientists** picking an embedding model for an Arabic corpus
- **Backend engineers** shipping Arabic RAG features on OCI

## Roadmap

- [ ] Oracle 26ai vector index adapter (replace in-memory index)
- [ ] OCI Generative AI Agents reference integration with tool calls
- [ ] Streamlit demo deployable in one command
- [ ] Dialect coverage: Egyptian, Maghrebi, Levantine in eval set
- [ ] Multilingual fine-grained eval (MSA vs dialect cross-retrieval)

## Project structure

```
.
├── src/arabicrag/         # the library
│   ├── normalization.py   # Arabic text normalization
│   ├── chunking.py        # sentence- and section-aware chunking
│   ├── embeddings.py      # TF-IDF, OCI, HuggingFace adapters
│   ├── retrieval.py       # vector + hybrid retrievers
│   └── prompts.py         # Arabic grounded-RAG prompt templates
├── benchmarks/
│   ├── data/              # corpus + eval set
│   ├── metrics.py         # P@k, R@k, MRR, nDCG
│   ├── run_benchmark.py   # CLI runner
│   └── results/           # generated outputs
├── examples/
│   └── compliance_qa.py   # end-to-end demo
├── tests/                 # unit tests (no network)
└── docs/                  # design notes, methodology
```

## Notes on the corpora

Both shipped corpora are **synthetic** — Arabic text written by the
project author in the style of banking regulations and legal statutes,
with `المادة N:` section markers. They illustrate the pipeline without
the licensing complications of redistributing real government or
regulator publications.

- `benchmarks/data/sample_corpus.json` — banking regulation style
  (capital adequacy, AML, KYC, liquidity, etc.)
- `benchmarks/data/legal_corpus.json` — legal style (employment,
  civil contracts, lease, inheritance, torts, etc.)

The legal corpus was added specifically to test whether the toolkit
generalizes across domains — and to mirror the domain of the
graduation project this toolkit grew out of (a RAG-based legal
assistant for Jordanian law offices, see Background above). The
results confirm both: same code, different domain, strong numbers,
and the morphological gap between TF-IDF and BM25 widens on the
more classically-worded legal text.

To benchmark on real publicly available Arabic documents, drop them
into `benchmarks/data/` as a JSON file matching the same schema and
add an entry to the `CORPORA` registry in `run_benchmark.py`.

## Contributing

PRs welcome. Especially interested in:
- Dialect coverage in the eval set (Egyptian, Gulf, Maghrebi)
- Additional embedding adapters
- An OCI 26ai vector index implementation

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
