"""Tests for arabic normalization and chunking."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from arabicrag import normalize  # noqa: E402
from arabicrag.chunking import chunk_by_sections, chunk_by_sentences, split_sentences  # noqa: E402
from arabicrag.normalization import (  # noqa: E402
    normalize_alef,
    normalize_digits,
    normalize_taa_marbuta,
    normalize_yaa,
    remove_diacritics,
)


# --- normalization ---------------------------------------------------------

def test_remove_diacritics():
    text = "اَلْعَرَبِيَّةُ"
    assert remove_diacritics(text) == "العربية"


def test_remove_tatweel():
    assert remove_diacritics("الـعـربـيـة") == "العربية"


def test_normalize_alef():
    assert normalize_alef("أحمد") == "احمد"
    assert normalize_alef("إسلام") == "اسلام"
    assert normalize_alef("آمنة") == "امنة"


def test_normalize_yaa():
    assert normalize_yaa("على") == "علي"


def test_normalize_taa_marbuta():
    assert normalize_taa_marbuta("مدرسة") == "مدرسه"


def test_normalize_digits():
    assert normalize_digits("سنة ٢٠٢٤") == "سنة 2024"


def test_pipeline():
    raw = "اَلْمَدْرَسَةُ الإِسْلَامِيَّةُ"
    assert normalize(raw) == "المدرسة الاسلامية"


def test_pipeline_with_taa_marbuta():
    raw = "اَلْمَدْرَسَةُ الإِسْلَامِيَّةُ"
    out = normalize(raw, fold_taa_marbuta=True)
    assert "ه" in out
    assert "ة" not in out


# --- chunking --------------------------------------------------------------

def test_split_sentences_basic():
    text = "هذه جملة. وهذه جملة أخرى؟ ونعم!"
    sents = split_sentences(text)
    assert len(sents) == 3


def test_chunk_by_sentences_no_split_mid_sentence():
    text = " ".join(["جملة قصيرة."] * 30)
    chunks = chunk_by_sentences(text, doc_id="d", target_chars=80)
    for c in chunks:
        # every chunk should end on a sentence terminator (. or ، etc.)
        assert c.text.rstrip().endswith((".", "؟", "!", "؛", "،"))


def test_chunk_by_sections_extracts_sections():
    text = (
        "المادة 1: نص المادة الأولى وهو قصير. "
        "المادة 2: نص المادة الثانية وهو أيضاً قصير. "
        "المادة 3: نص المادة الثالثة."
    )
    chunks = chunk_by_sections(text, doc_id="d", target_chars=500)
    assert len(chunks) == 3
    assert chunks[0].section.startswith("المادة 1")
    assert chunks[1].section.startswith("المادة 2")
    assert chunks[2].section.startswith("المادة 3")


def test_chunk_ids_unique():
    text = " ".join(f"المادة {i}: نص قصير جداً." for i in range(1, 6))
    chunks = chunk_by_sections(text, doc_id="d")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


if __name__ == "__main__":
    # simple test runner so you can do `python tests/test_normalization.py`
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except AssertionError:
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
            failed += 1
        except Exception:
            print(f"ERR  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
