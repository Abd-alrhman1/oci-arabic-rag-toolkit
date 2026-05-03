"""
Interactive shell with REAL Arabic answers via OCI Generative AI.

This is the production version — retrieves chunks AND generates a
grounded Arabic answer using OCI's hosted LLM.

Setup (one-time, ~30 minutes):
  1. Sign up at https://signup.cloud.oracle.com (Always Free tier)
  2. In OCI Console → Identity → Compartments, copy your compartment OCID
  3. In OCI Console → Profile → My Profile → API Keys, generate a key
     and download the config snippet → save to ~/.oci/config
  4. pip install oci

Then set your compartment ID and run:
  export OCI_COMPARTMENT_ID="ocid1.compartment.oc1..xxxx"
  python examples/interactive_oci.py

If OCI is not set up, this script tells you exactly what's missing and
falls back to printing the prompt instead of generating an answer.
"""

from __future__ import annotations

import json
import os
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


def load_corpus():
    with open(ROOT / "benchmarks" / "data" / "sample_corpus.json", encoding="utf-8") as f:
        return json.load(f)


def setup_oci_client():
    """Build an OCI Generative AI inference client. Returns None if not available."""
    try:
        import oci
    except ImportError:
        print("\n  ⚠  OCI SDK not installed. Run: pip install oci")
        return None, None

    compartment_id = os.environ.get("OCI_COMPARTMENT_ID")
    if not compartment_id:
        print("\n  ⚠  OCI_COMPARTMENT_ID not set.")
        print("     export OCI_COMPARTMENT_ID='ocid1.compartment.oc1..xxxx'")
        return None, None

    config_path = os.path.expanduser("~/.oci/config")
    if not os.path.exists(config_path):
        print(f"\n  ⚠  OCI config not found at {config_path}")
        print("     See: https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm")
        return None, None

    try:
        config = oci.config.from_file()
        client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240),
        )
        return client, compartment_id
    except Exception as e:
        print(f"\n  ⚠  OCI client setup failed: {e}")
        return None, None


def generate_answer(client, compartment_id: str, messages: list[dict], model_id: str) -> str:
    """Call OCI Generative AI's chat endpoint to generate an Arabic answer."""
    from oci.generative_ai_inference.models import (
        ChatDetails,
        GenericChatRequest,
        Message,
        OnDemandServingMode,
        TextContent,
    )

    oci_messages = [
        Message(
            role=m["role"].upper(),
            content=[TextContent(text=m["content"])],
        )
        for m in messages
    ]

    chat_details = ChatDetails(
        compartment_id=compartment_id,
        serving_mode=OnDemandServingMode(model_id=model_id),
        chat_request=GenericChatRequest(
            api_format="GENERIC",
            messages=oci_messages,
            max_tokens=600,
            temperature=0.2,
            top_p=0.9,
        ),
    )

    response = client.chat(chat_details)
    return response.data.chat_response.choices[0].message.content[0].text


def main():
    print("\n  OCI Arabic RAG — Interactive (with real LLM generation)\n")

    # Load and index the corpus
    corpus = load_corpus()
    titles = {d["id"]: d["title"] for d in corpus}
    normalized = [
        {"id": d["id"], "text": normalize(d["text"], fold_taa_marbuta=True)}
        for d in corpus
    ]
    chunks = chunk_documents(normalized, mode="section", target_chars=500)
    print(f"  Loaded {len(corpus)} documents → {len(chunks)} chunks")

    print("  Building hybrid index... ", end="", flush=True)
    index = HybridRetriever(VectorIndex(TfidfEmbedder()), alpha=0.6)
    index.index(chunks)
    print("ready.")

    # Try to set up OCI
    client, compartment_id = setup_oci_client()
    model_id = os.environ.get("OCI_MODEL_ID", "cohere.command-r-plus-08-2024")
    if client:
        print(f"  OCI Generative AI: connected ({model_id})")
        print("  → Real Arabic answers will be generated.\n")
    else:
        print("\n  → Running in retrieval-only mode (no LLM). The retrieved")
        print("    chunks and prompt will be shown, but no answer generated.\n")

    print("  Type a question in Arabic. Ctrl+C to exit.\n")

    while True:
        try:
            question = input("  ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye.")
            break
        if not question:
            continue

        # Retrieve
        t0 = time.time()
        results = index.search(normalize(question), k=4)
        retrieve_ms = (time.time() - t0) * 1000

        print(f"\n  Retrieved {len(results)} chunks ({retrieve_ms:.1f}ms):")
        for i, r in enumerate(results, 1):
            title = titles.get(r.chunk.doc_id, r.chunk.doc_id)
            section = f" — {r.chunk.section}" if r.chunk.section else ""
            print(f"    [{i}] {title}{section}  (score={r.score:.3f})")

        # Generate or fall back
        messages = build_messages(question, [r.chunk for r in results])

        if client:
            print("\n  Generating answer...", end=" ", flush=True)
            t0 = time.time()
            try:
                answer = generate_answer(client, compartment_id, messages, model_id)
                gen_ms = (time.time() - t0) * 1000
                print(f"({gen_ms:.0f}ms)\n")
                print("  ┌─ Answer ─" + "─" * 60)
                for line in answer.splitlines():
                    print(f"  │ {line}")
                print("  └" + "─" * 70 + "\n")
            except Exception as e:
                print(f"\n  ✗ generation failed: {e}\n")
        else:
            print("\n  (Skipping generation — OCI not configured.)\n")


if __name__ == "__main__":
    main()
