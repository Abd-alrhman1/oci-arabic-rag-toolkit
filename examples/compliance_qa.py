"""
End-to-end compliance Q&A example.

Demonstrates the full pipeline: load corpus → normalize → chunk → index
→ retrieve → build prompt. Generation step is sketched; plug in OCI
Generative AI's chat completion or any compatible LLM to actually
generate the answer.

Run from repo root:
    python examples/compliance_qa.py
    python examples/compliance_qa.py --query "ما هي حدود التعرض الكبير؟"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arabicrag import (  # noqa: E402
    HybridRetriever,
    TfidfEmbedder,
    VectorIndex,
    build_messages,
    chunk_documents,
    normalize,
)


def load_corpus():
    with open(ROOT / "benchmarks" / "data" / "sample_corpus.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="ما هي نسبة تغطية السيولة المطلوبة؟",
        help="Query in Arabic.",
    )
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    # 1. Load and normalize corpus
    corpus = load_corpus()
    normalized = [
        {"id": d["id"], "text": normalize(d["text"], fold_taa_marbuta=True)}
        for d in corpus
    ]

    # 2. Chunk by section (regulatory text)
    chunks = chunk_documents(normalized, mode="section", target_chars=500)
    print(f"[setup] {len(corpus)} docs → {len(chunks)} chunks\n")

    # 3. Index with hybrid retriever
    index = HybridRetriever(VectorIndex(TfidfEmbedder()), alpha=0.6)
    index.index(chunks)

    # 4. Retrieve
    query_norm = normalize(args.query)
    results = index.search(query_norm, k=args.k)

    print(f"Query: {args.query}\n")
    print(f"Top {args.k} retrieved chunks:")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        section = f" — {r.chunk.section}" if r.chunk.section else ""
        print(f"\n[{i}] {r.chunk.doc_id}{section}  (score={r.score:.4f})")
        print(f"    {r.chunk.text[:200]}{'...' if len(r.chunk.text) > 200 else ''}")
    print()

    # 5. Build the LLM prompt
    messages = build_messages(args.query, [r.chunk for r in results])

    print("=" * 60)
    print("Built chat messages (ready to pass to OCI Generative AI):")
    print("=" * 60)
    for m in messages:
        role = m["role"].upper()
        content = m["content"]
        if len(content) > 400:
            content = content[:400] + "..."
        print(f"\n[{role}]\n{content}")

    print("\n" + "=" * 60)
    print("To actually generate an answer, pipe `messages` to OCI Generative AI:")
    print("=" * 60)
    print("""
    import oci
    from oci.generative_ai_inference.models import ChatDetails, GenericChatRequest

    client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=oci.config.from_file()
    )
    response = client.chat(ChatDetails(
        compartment_id=COMPARTMENT_ID,
        serving_mode=OnDemandServingMode(model_id="cohere.command-r-plus"),
        chat_request=GenericChatRequest(
            messages=messages,
            max_tokens=600,
            temperature=0.2,
        ),
    ))
    print(response.data.chat_response.choices[0].message.content)
    """)


if __name__ == "__main__":
    main()
