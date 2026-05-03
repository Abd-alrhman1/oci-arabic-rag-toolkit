"""
Arabic-aware chunking.

Naive character or token chunking destroys retrieval quality on Arabic
regulatory text because the boundary often lands mid-sentence or
mid-clause. This module respects Arabic sentence terminators (.، ؟ ! ؛)
and offers a section-aware mode for numbered regulatory documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Arabic sentence terminators + standard ASCII fallbacks
_SENTENCE_END = re.compile(r"(?<=[.؟!۔])\s+|(?<=[.\?!])\s+")

# Section-style headers: matches "المادة 12", "المادة (12)", "البند 3-4", etc.
# Anchored at start-of-string or after whitespace/newline so the marker is
# always a real section break, never a substring inside another word.
_SECTION_HEADER = re.compile(
    r"(?:(?<=\s)|^)(?:المادة|البند|الفقرة|الفصل|الباب)\s*\(?\d+(?:[-./]\d+)?\)?\s*[:.\-]?\s*"
)


@dataclass
class Chunk:
    """A retrieval-ready chunk of Arabic text."""

    text: str
    doc_id: str
    chunk_id: str
    section: str | None = None
    char_start: int = 0
    char_end: int = 0


def split_sentences(text: str) -> list[str]:
    """Split Arabic (and mixed) text into sentences."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_by_sentences(
    text: str,
    *,
    doc_id: str,
    target_chars: int = 800,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """Greedy sentence-aware chunker.

    Accumulates sentences until the chunk reaches target_chars, then
    starts a new chunk with overlap_chars of trailing context. This
    avoids splitting mid-sentence which is the #1 cause of bad
    retrieval on Arabic legal text.
    """
    sentences = split_sentences(text)
    chunks: list[Chunk] = []
    buf = ""
    buf_start = 0
    cursor = 0
    chunk_idx = 0

    for sent in sentences:
        sent_pos = text.find(sent, cursor)
        if sent_pos == -1:
            sent_pos = cursor
        cursor = sent_pos + len(sent)

        if not buf:
            buf = sent
            buf_start = sent_pos
        elif len(buf) + len(sent) + 1 <= target_chars:
            buf = f"{buf} {sent}"
        else:
            chunks.append(
                Chunk(
                    text=buf,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::{chunk_idx}",
                    char_start=buf_start,
                    char_end=buf_start + len(buf),
                )
            )
            chunk_idx += 1
            # carry overlap from end of previous chunk
            tail = buf[-overlap_chars:] if overlap_chars else ""
            buf = f"{tail} {sent}".strip() if tail else sent
            buf_start = max(0, buf_start + len(buf) - len(tail) - len(sent) - 1)

    if buf:
        chunks.append(
            Chunk(
                text=buf,
                doc_id=doc_id,
                chunk_id=f"{doc_id}::{chunk_idx}",
                char_start=buf_start,
                char_end=buf_start + len(buf),
            )
        )

    return chunks


def chunk_by_sections(
    text: str,
    *,
    doc_id: str,
    target_chars: int = 1200,
) -> list[Chunk]:
    """Section-aware chunker for regulatory text.

    Splits at headers like 'المادة 12' or 'البند 3' and emits each
    section as a chunk. Long sections are sub-chunked by sentence.
    Each chunk carries its section label, which is gold for citation.
    """
    matches = list(_SECTION_HEADER.finditer(text))
    if not matches:
        return chunk_by_sentences(text, doc_id=doc_id, target_chars=target_chars)

    chunks: list[Chunk] = []
    chunk_idx = 0

    for i, m in enumerate(matches):
        section_label = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if not section_text:
            continue

        if len(section_text) <= target_chars:
            chunks.append(
                Chunk(
                    text=section_text,
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}::{chunk_idx}",
                    section=section_label,
                    char_start=start,
                    char_end=end,
                )
            )
            chunk_idx += 1
        else:
            sub_chunks = chunk_by_sentences(
                section_text,
                doc_id=doc_id,
                target_chars=target_chars,
            )
            for sc in sub_chunks:
                chunks.append(
                    Chunk(
                        text=sc.text,
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}::{chunk_idx}",
                        section=section_label,
                        char_start=start + sc.char_start,
                        char_end=start + sc.char_end,
                    )
                )
                chunk_idx += 1

    return chunks


def chunk_documents(docs: Iterable[dict], *, mode: str = "sentence", **kwargs) -> list[Chunk]:
    """Chunk a collection of documents.

    Args:
        docs: iterable of dicts with at least 'id' and 'text' keys.
        mode: 'sentence' or 'section'.
        **kwargs: passed to the underlying chunker.
    """
    chunker = chunk_by_sections if mode == "section" else chunk_by_sentences
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunker(doc["text"], doc_id=doc["id"], **kwargs))
    return out
