"""
Arabic text normalization utilities.

Handles diacritics (tashkeel), character variants, tatweel, and dialect
markers. The defaults match what works best for retrieval over standard
Arabic regulatory and news text. Adjust if you're targeting dialect-heavy
content.
"""

from __future__ import annotations

import re
import unicodedata

# Tashkeel (diacritics): fatha, damma, kasra, sukun, shadda, tanween
_TASHKEEL = re.compile(r"[\u064B-\u0652\u0670\u0640]")

# Tatweel (kashida): used for visual elongation, never semantic
_TATWEEL = re.compile(r"\u0640")

# Non-Arabic punctuation that typically appears in mixed text
_PUNCT = re.compile(r"[^\w\s\u0600-\u06FF]", re.UNICODE)

# Arabic numerals → ASCII numerals (helpful for regulatory text with figures)
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def remove_diacritics(text: str) -> str:
    """Strip tashkeel and tatweel. Leaves base letters intact."""
    text = _TASHKEEL.sub("", text)
    text = _TATWEEL.sub("", text)
    return text


def normalize_alef(text: str, target: str = "ا") -> str:
    """Collapse all alef variants (أ إ آ ٱ) to a single form.

    Default target is bare alef (ا) — best for retrieval where the
    distinction between hamza-on/under-alef and bare alef is rarely
    semantically meaningful for the user's query.
    """
    return re.sub(r"[إأآٱ]", target, text)


def normalize_yaa(text: str, target: str = "ي") -> str:
    """Collapse alef maqsura (ى) into yaa (ي).

    This matches Egyptian/Levantine writing convention; switch the
    default if you're indexing Maghrebi text where ى is more strictly
    preserved.
    """
    return text.replace("ى", target)


def normalize_taa_marbuta(text: str, target: str = "ه") -> str:
    """Collapse taa marbuta (ة) into haa (ه) for retrieval.

    This is lossy semantically but improves recall significantly
    because users frequently type ه where the formal text uses ة.
    Disable if your downstream task is grammar-sensitive.
    """
    return text.replace("ة", target)


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic and Persian digits to ASCII digits."""
    return text.translate(_ARABIC_DIGITS).translate(_PERSIAN_DIGITS)


def remove_punctuation(text: str) -> str:
    """Remove all non-word, non-space, non-Arabic-letter chars."""
    return _PUNCT.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    """Squash runs of whitespace into single spaces; trim."""
    return re.sub(r"\s+", " ", text).strip()


def normalize(
    text: str,
    *,
    strip_diacritics: bool = True,
    fold_alef: bool = True,
    fold_yaa: bool = True,
    fold_taa_marbuta: bool = False,
    fold_digits: bool = True,
    strip_punctuation: bool = False,
    nfkc: bool = True,
) -> str:
    """Apply a configurable normalization pipeline.

    Default settings target enterprise retrieval over MSA (Modern Standard
    Arabic) text. For dialect-heavy or grammar-sensitive tasks, override
    the flags.

    Args:
        text: input string.
        strip_diacritics: remove tashkeel and tatweel.
        fold_alef: collapse hamza variants on alef.
        fold_yaa: collapse alef maqsura to yaa.
        fold_taa_marbuta: collapse taa marbuta to haa (off by default — lossy).
        fold_digits: convert Arabic-Indic digits to ASCII.
        strip_punctuation: remove punctuation entirely.
        nfkc: apply Unicode NFKC normalization first.

    Returns:
        The normalized string.
    """
    if nfkc:
        text = unicodedata.normalize("NFKC", text)
    if strip_diacritics:
        text = remove_diacritics(text)
    if fold_alef:
        text = normalize_alef(text)
    if fold_yaa:
        text = normalize_yaa(text)
    if fold_taa_marbuta:
        text = normalize_taa_marbuta(text)
    if fold_digits:
        text = normalize_digits(text)
    if strip_punctuation:
        text = remove_punctuation(text)
    text = collapse_whitespace(text)
    return text
