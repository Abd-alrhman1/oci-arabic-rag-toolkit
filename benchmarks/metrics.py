"""
Retrieval evaluation metrics.

Implements standard IR metrics: Precision@k, Recall@k, MRR, nDCG@k.
We use document-level relevance: a chunk is relevant if its source
doc_id is in the ground-truth relevant set. Because a single document
can produce multiple chunks, all metrics dedupe the retrieved doc_id
list (preserving first occurrence) before scoring — this gives true
document-level metrics that obey the [0, 1] bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class EvalRecord:
    query_id: str
    relevant_docs: set[str]
    retrieved_doc_ids: list[str]  # ordered, may contain duplicates from chunking


def _dedupe(ordered: list[str]) -> list[str]:
    """Return ordered list with duplicates removed, preserving first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for d in ordered:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def precision_at_k(record: EvalRecord, k: int) -> float:
    top_k = _dedupe(record.retrieved_doc_ids)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in record.relevant_docs)
    return hits / k


def recall_at_k(record: EvalRecord, k: int) -> float:
    if not record.relevant_docs:
        return 0.0
    top_k = _dedupe(record.retrieved_doc_ids)[:k]
    hits = sum(1 for d in top_k if d in record.relevant_docs)
    return hits / len(record.relevant_docs)


def mrr(record: EvalRecord) -> float:
    for i, d in enumerate(_dedupe(record.retrieved_doc_ids), start=1):
        if d in record.relevant_docs:
            return 1.0 / i
    return 0.0


def ndcg_at_k(record: EvalRecord, k: int) -> float:
    """Standard nDCG with binary relevance, on the deduped doc list."""
    top_k = _dedupe(record.retrieved_doc_ids)[:k]
    dcg = sum(
        (1.0 / math.log2(i + 2)) if d in record.relevant_docs else 0.0
        for i, d in enumerate(top_k)
    )
    ideal_hits = min(len(record.relevant_docs), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate(records: Sequence[EvalRecord], ks: Sequence[int] = (1, 3, 5, 10)) -> dict:
    """Compute mean metrics across a record set."""
    n = len(records) or 1
    out = {"n_queries": len(records)}
    for k in ks:
        out[f"precision@{k}"] = sum(precision_at_k(r, k) for r in records) / n
        out[f"recall@{k}"] = sum(recall_at_k(r, k) for r in records) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, k) for r in records) / n
    out["mrr"] = sum(mrr(r) for r in records) / n
    return out
