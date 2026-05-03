"""
Benchmark runner.

Compares retrievers on Arabic eval sets across multiple domains.

Usage:
    python benchmarks/run_benchmark.py                       # banking (default)
    python benchmarks/run_benchmark.py --corpus legal        # legal corpus
    python benchmarks/run_benchmark.py --corpus all          # all corpora
    python benchmarks/run_benchmark.py --retriever bm25
    python benchmarks/run_benchmark.py --retriever oci       # requires creds

Results are written to benchmarks/results/ as JSON + a markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make the in-repo package importable when run from the repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from arabicrag import (  # noqa: E402
    BM25Retriever,
    Chunk,
    HybridRetriever,
    TfidfEmbedder,
    VectorIndex,
    chunk_documents,
    normalize,
)
from benchmarks.metrics import EvalRecord, aggregate  # noqa: E402

DATA_DIR = ROOT / "benchmarks" / "data"
RESULTS_DIR = ROOT / "benchmarks" / "results"

# Registered corpora — add new domains by dropping a {name}_corpus.json
# and {name}_eval_set.json into benchmarks/data/ and adding an entry here.
CORPORA = {
    "banking": ("sample_corpus.json", "eval_set.json"),
    "legal": ("legal_corpus.json", "legal_eval_set.json"),
}


def load_corpus(name: str = "banking") -> list[dict]:
    corpus_file, _ = CORPORA[name]
    with open(DATA_DIR / corpus_file, encoding="utf-8") as f:
        return json.load(f)


def load_eval_set(name: str = "banking") -> list[dict]:
    _, eval_file = CORPORA[name]
    with open(DATA_DIR / eval_file, encoding="utf-8") as f:
        return json.load(f)


def prep_chunks(corpus: list[dict]) -> list[Chunk]:
    """Normalize then chunk by section (regulatory text → use sections)."""
    normalized = [
        {"id": d["id"], "text": normalize(d["text"], fold_taa_marbuta=True)}
        for d in corpus
    ]
    return chunk_documents(normalized, mode="section", target_chars=500)


def run_tfidf(chunks: list[Chunk], queries: list[dict], k: int) -> tuple[list[EvalRecord], dict]:
    index = VectorIndex(TfidfEmbedder())
    t0 = time.time()
    index.index(chunks)
    index_time = time.time() - t0

    records, latencies = [], []
    for q in queries:
        t0 = time.time()
        results = index.search(normalize(q["query"]), k=k)
        latencies.append(time.time() - t0)
        records.append(
            EvalRecord(
                query_id=q["id"],
                relevant_docs=set(q["relevant_docs"]),
                retrieved_doc_ids=[r.chunk.doc_id for r in results],
            )
        )
    return records, {
        "retriever": "tfidf",
        "index_seconds": round(index_time, 3),
        "mean_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 2),
    }


def run_bm25(chunks: list[Chunk], queries: list[dict], k: int) -> tuple[list[EvalRecord], dict]:
    bm25 = BM25Retriever()
    t0 = time.time()
    bm25.fit([c.text for c in chunks])
    index_time = time.time() - t0

    records, latencies = [], []
    for q in queries:
        t0 = time.time()
        scores = bm25.score(normalize(q["query"]))
        ranked_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        latencies.append(time.time() - t0)
        records.append(
            EvalRecord(
                query_id=q["id"],
                relevant_docs=set(q["relevant_docs"]),
                retrieved_doc_ids=[chunks[i].doc_id for i in ranked_idx],
            )
        )
    return records, {
        "retriever": "bm25",
        "index_seconds": round(index_time, 3),
        "mean_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 2),
    }


def run_hybrid(chunks: list[Chunk], queries: list[dict], k: int, alpha: float = 0.6):
    vec_index = VectorIndex(TfidfEmbedder())
    hybrid = HybridRetriever(vec_index, alpha=alpha)
    t0 = time.time()
    hybrid.index(chunks)
    index_time = time.time() - t0

    records, latencies = [], []
    for q in queries:
        t0 = time.time()
        results = hybrid.search(normalize(q["query"]), k=k)
        latencies.append(time.time() - t0)
        records.append(
            EvalRecord(
                query_id=q["id"],
                relevant_docs=set(q["relevant_docs"]),
                retrieved_doc_ids=[r.chunk.doc_id for r in results],
            )
        )
    return records, {
        "retriever": f"hybrid(alpha={alpha})",
        "index_seconds": round(index_time, 3),
        "mean_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 2),
    }


def run_oci(chunks: list[Chunk], queries: list[dict], k: int):
    """Requires OCI credentials. Falls back gracefully if unavailable."""
    from arabicrag import OCIEmbedder

    compartment = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment:
        raise RuntimeError(
            "Set OCI_COMPARTMENT_ID environment variable to run OCI benchmark."
        )
    embedder = OCIEmbedder(compartment_id=compartment)
    index = VectorIndex(embedder)
    t0 = time.time()
    index.index(chunks)
    index_time = time.time() - t0

    records, latencies = [], []
    for q in queries:
        t0 = time.time()
        results = index.search(q["query"], k=k)  # OCI handles its own normalization
        latencies.append(time.time() - t0)
        records.append(
            EvalRecord(
                query_id=q["id"],
                relevant_docs=set(q["relevant_docs"]),
                retrieved_doc_ids=[r.chunk.doc_id for r in results],
            )
        )
    return records, {
        "retriever": embedder.name,
        "index_seconds": round(index_time, 3),
        "mean_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] * 1000, 2),
    }


RUNNERS = {
    "tfidf": run_tfidf,
    "bm25": run_bm25,
    "hybrid": run_hybrid,
    "oci": run_oci,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        choices=["tfidf", "bm25", "hybrid", "oci", "all"],
        default="all",
    )
    parser.add_argument(
        "--corpus",
        choices=[*CORPORA.keys(), "all"],
        default="banking",
        help="Which corpus to benchmark on.",
    )
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    corpus_names = list(CORPORA.keys()) if args.corpus == "all" else [args.corpus]
    retrievers = ["tfidf", "bm25", "hybrid"] if args.retriever == "all" else [args.retriever]

    all_results: dict = {}

    for corpus_name in corpus_names:
        corpus = load_corpus(corpus_name)
        queries = load_eval_set(corpus_name)
        chunks = prep_chunks(corpus)
        print(f"\n{'═' * 60}")
        print(f"  Corpus: {corpus_name.upper()}  "
              f"({len(corpus)} docs → {len(chunks)} chunks; {len(queries)} queries)")
        print(f"{'═' * 60}\n")

        all_results[corpus_name] = {}
        for r in retrievers:
            try:
                records, perf = RUNNERS[r](chunks, queries, args.k)
                metrics = aggregate(records)
                all_results[corpus_name][r] = {**metrics, **perf}
                print(f"=== {r.upper()} on {corpus_name} ===")
                for kk, vv in all_results[corpus_name][r].items():
                    if isinstance(vv, float):
                        print(f"  {kk}: {vv:.4f}")
                    else:
                        print(f"  {kk}: {vv}")
                print()
            except Exception as e:
                print(f"[skipped {r}: {e}]\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    write_markdown_table(all_results)
    print(f"Results written to {RESULTS_DIR}/")


def write_markdown_table(results_by_corpus: dict) -> None:
    if not results_by_corpus:
        return

    keys = ["retriever", "precision@1", "precision@5", "recall@5", "ndcg@5", "mrr", "mean_latency_ms"]
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"

    lines = [
        "# Benchmark results",
        "",
        "Cross-domain Arabic retrieval benchmarks. Same toolkit, different corpora.",
        "",
    ]

    for corpus_name, results in results_by_corpus.items():
        if not results:
            continue
        try:
            n_docs = len(load_corpus(corpus_name))
            n_queries = len(load_eval_set(corpus_name))
        except Exception:
            n_docs, n_queries = "?", "?"

        lines.append(f"## {corpus_name.title()}")
        lines.append("")
        lines.append(f"_{n_docs} synthetic Arabic documents · {n_queries} queries · k=10_")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        for row in results.values():
            cells = []
            for key in keys:
                v = row.get(key, "")
                if isinstance(v, float):
                    cells.append(f"{v:.4f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("Generated by `python benchmarks/run_benchmark.py --corpus all`.")

    with open(RESULTS_DIR / "BENCHMARK.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
