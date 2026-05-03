"""
Interactive shell for the Arabic RAG toolkit.

Type Arabic questions, see what gets retrieved. No API keys required —
uses the TF-IDF baseline locally so you can play with the system before
wiring up OCI.

Run from repo root:
    python examples/interactive.py

Once it's loaded, type questions in Arabic. Type 'help' for commands.
"""

from __future__ import annotations

import json
import sys
import time
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


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          OCI Arabic RAG Toolkit — Interactive Shell              ║
║                                                                  ║
║  Type Arabic questions to see what the retriever finds.          ║
║  Commands: help, suggest, retriever <tfidf|hybrid>, k <n>, quit  ║
╚══════════════════════════════════════════════════════════════════╝
"""

SUGGESTED_QUERIES = [
    "ما هي نسبة تغطية السيولة المطلوبة؟",
    "ما هي متطلبات الحد الأدنى لرأس المال؟",
    "متى يجب الإبلاغ عن العمليات المشبوهة؟",
    "كم مدة الاحتفاظ بسجلات العملاء؟",
    "ما هي حدود التعرض لطرف واحد؟",
    "خلال كم ساعة يجب إبلاغ السلطة الرقابية بحادث سيبراني؟",
    "ما اللجان المنبثقة عن مجلس الإدارة؟",
    "ما الوثائق المطلوبة للتحقق من هوية العميل؟",
]

HELP = """
Commands:
  help                       Show this help
  suggest                    Show example queries you can copy/paste
  retriever <tfidf|hybrid>   Switch retriever (default: hybrid)
  k <n>                      Set how many results to show (default: 3)
  show <query>               Show full text of retrieved chunks
  prompt <query>             Show the LLM prompt that would be sent to OCI
  quit / exit                Leave

Anything else is treated as an Arabic query.
"""


def load_corpus():
    with open(ROOT / "benchmarks" / "data" / "sample_corpus.json", encoding="utf-8") as f:
        return json.load(f)


def setup_indices():
    """Build both indices once at startup so switching is instant."""
    corpus = load_corpus()
    titles = {d["id"]: d["title"] for d in corpus}
    normalized = [
        {"id": d["id"], "text": normalize(d["text"], fold_taa_marbuta=True)}
        for d in corpus
    ]
    chunks = chunk_documents(normalized, mode="section", target_chars=500)

    print(f"  Loaded {len(corpus)} documents → {len(chunks)} chunks")
    print("  Building TF-IDF index...", end=" ", flush=True)
    t0 = time.time()
    tfidf_index = VectorIndex(TfidfEmbedder())
    tfidf_index.index(chunks)
    print(f"({time.time() - t0:.2f}s)")

    print("  Building hybrid index...", end=" ", flush=True)
    t0 = time.time()
    hybrid = HybridRetriever(VectorIndex(TfidfEmbedder()), alpha=0.6)
    hybrid.index(chunks)
    print(f"({time.time() - t0:.2f}s)")

    return {"tfidf": tfidf_index, "hybrid": hybrid}, titles, chunks


def show_results(results, titles, full_text: bool = False):
    if not results:
        print("  (no results)")
        return
    print()
    for i, r in enumerate(results, 1):
        title = titles.get(r.chunk.doc_id, r.chunk.doc_id)
        section = f" — {r.chunk.section}" if r.chunk.section else ""
        print(f"  [{i}] {title}{section}")
        print(f"      score: {r.score:.4f}  |  doc_id: {r.chunk.doc_id}")
        text = r.chunk.text if full_text else r.chunk.text[:160] + (
            "..." if len(r.chunk.text) > 160 else ""
        )
        # Indent multi-line text for readability
        for line in text.splitlines():
            print(f"      {line}")
        print()


def show_prompt(query: str, results):
    messages = build_messages(query, [r.chunk for r in results])
    print("\n  ──── SYSTEM ────")
    for line in messages[0]["content"].splitlines():
        print(f"  {line}")
    print("\n  ──── USER ────")
    for line in messages[1]["content"].splitlines():
        print(f"  {line}")
    print()


def main():
    print(BANNER)
    print("Setting up...")
    indices, titles, chunks = setup_indices()
    print("\nReady. Type 'help' for commands, 'suggest' for example queries.\n")

    retriever_name = "hybrid"
    k = 3

    while True:
        try:
            line = input(f"[{retriever_name}, k={k}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not line:
            continue

        cmd = line.lower()

        if cmd in ("quit", "exit", "q"):
            print("bye.")
            break

        if cmd == "help":
            print(HELP)
            continue

        if cmd == "suggest":
            print("\n  Try one of these:")
            for q in SUGGESTED_QUERIES:
                print(f"    • {q}")
            print()
            continue

        if cmd.startswith("retriever "):
            choice = cmd.split(maxsplit=1)[1].strip()
            if choice in indices:
                retriever_name = choice
                print(f"  → switched to {retriever_name}")
            else:
                print(f"  unknown retriever '{choice}'. options: {list(indices)}")
            continue

        if cmd.startswith("k "):
            try:
                k = max(1, int(cmd.split()[1]))
                print(f"  → k = {k}")
            except (ValueError, IndexError):
                print("  usage: k <integer>")
            continue

        if cmd.startswith("show "):
            q = line[5:].strip()
            results = indices[retriever_name].search(normalize(q), k=k)
            show_results(results, titles, full_text=True)
            continue

        if cmd.startswith("prompt "):
            q = line[7:].strip()
            results = indices[retriever_name].search(normalize(q), k=k)
            show_results(results, titles, full_text=False)
            show_prompt(q, results)
            continue

        # Default: treat as a query
        t0 = time.time()
        results = indices[retriever_name].search(normalize(line), k=k)
        latency_ms = (time.time() - t0) * 1000
        show_results(results, titles)
        print(f"  ({latency_ms:.1f}ms)\n")


if __name__ == "__main__":
    main()
