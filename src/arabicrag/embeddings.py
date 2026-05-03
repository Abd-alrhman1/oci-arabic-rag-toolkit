"""
Embedding model adapters.

Defines a uniform `Embedder` interface and three concrete implementations:

  * `TfidfEmbedder` — pure-sklearn baseline. Works offline, no GPU,
    no API keys. Useful for testing the pipeline and as the lower
    bound in the benchmark.

  * `OCIEmbedder` — OCI Generative AI service. Requires `oci` SDK
    and a configured tenancy. Default model is the multilingual
    Cohere embedding model on OCI.

  * `HuggingFaceEmbedder` — sentence-transformers wrapper for any
    Arabic-capable model on HF (e.g. paraphrase-multilingual,
    AraBERT-based encoders).

The TF-IDF baseline is implemented here as well to keep adapters
swappable for benchmarking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class Embedder(ABC):
    """Abstract embedder interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (N, dim) array of embeddings."""

    def fit(self, corpus: Sequence[str]) -> None:
        """Optional: fit-time vocabulary building (for TF-IDF, BM25)."""


class TfidfEmbedder(Embedder):
    """TF-IDF baseline. Offline, deterministic, no dependencies on APIs.

    Uses character n-grams (3-5) which works well across Arabic
    morphological variants without a heavy stemmer.
    """

    def __init__(self, ngram_range: tuple[int, int] = (3, 5), max_features: int = 50_000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=False,  # Arabic doesn't have case
        )
        self._fitted = False
        self._dim = 0

    @property
    def name(self) -> str:
        return "tfidf-char3-5"

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: Sequence[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True
        self._dim = len(self._vectorizer.get_feature_names_out())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit(corpus) before .embed(...) on TfidfEmbedder.")
        sparse = self._vectorizer.transform(texts)
        return sparse.toarray().astype(np.float32)


class BM25Retriever:
    """BM25 retriever — not strictly an embedder, but useful as a baseline.

    We expose this separately because BM25 doesn't produce vectors;
    it scores query-document pairs directly.
    """

    def __init__(self):
        from rank_bm25 import BM25Okapi  # lazy import

        self._BM25Okapi = BM25Okapi
        self._bm25 = None
        self._docs: list[str] = []

    @property
    def name(self) -> str:
        return "bm25"

    def fit(self, corpus: Sequence[str]) -> None:
        self._docs = list(corpus)
        tokenized = [doc.split() for doc in self._docs]
        self._bm25 = self._BM25Okapi(tokenized)

    def score(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            raise RuntimeError("Call .fit(corpus) before .score(query).")
        return self._bm25.get_scores(query.split())


class OCIEmbedder(Embedder):
    """OCI Generative AI embedding adapter.

    This is the production target. It calls OCI's hosted embedding
    endpoint (default: cohere multilingual). Requires:

        pip install oci

    plus a configured ~/.oci/config or instance principals.

    Example:
        embedder = OCIEmbedder(
            compartment_id="ocid1.compartment.oc1..xxx",
            model_id="cohere.embed-multilingual-v3.0",
            endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        )
    """

    def __init__(
        self,
        compartment_id: str,
        model_id: str = "cohere.embed-multilingual-v3.0",
        endpoint: str = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        config_profile: str = "DEFAULT",
        config_path: str | None = None,
    ):
        try:
            import oci
        except ImportError as e:
            raise ImportError(
                "OCIEmbedder requires the `oci` package. "
                "Install with: pip install oci"
            ) from e

        config = oci.config.from_file(file_location=config_path) if config_path \
            else oci.config.from_file(profile_name=config_profile)
        self._client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint=endpoint,
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=(10, 240),
        )
        self._compartment_id = compartment_id
        self._model_id = model_id
        self._dim = 1024  # Cohere multilingual-v3 dim; override if you swap models

    @property
    def name(self) -> str:
        return f"oci::{self._model_id}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import oci  # type: ignore
        from oci.generative_ai_inference.models import (
            EmbedTextDetails,
            OnDemandServingMode,
        )

        # OCI batch limit is 96 inputs per request for cohere embeddings
        batch_size = 96
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            details = EmbedTextDetails(
                inputs=batch,
                serving_mode=OnDemandServingMode(model_id=self._model_id),
                compartment_id=self._compartment_id,
                input_type="SEARCH_DOCUMENT",
                truncate="END",
            )
            resp = self._client.embed_text(details)
            all_vectors.extend(resp.data.embeddings)
        return np.asarray(all_vectors, dtype=np.float32)


class HuggingFaceEmbedder(Embedder):
    """sentence-transformers adapter.

    Use this for offline benchmarking with Arabic-capable models like:
      - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
      - intfloat/multilingual-e5-large
      - any AraBERT-based sentence encoder

    Requires:
        pip install sentence-transformers
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "HuggingFaceEmbedder requires `sentence-transformers`. "
                "Install with: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return f"hf::{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
