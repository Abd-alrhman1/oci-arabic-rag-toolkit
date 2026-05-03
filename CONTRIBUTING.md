# Contributing

Thanks for your interest. This project welcomes contributions in three areas:

1. **Eval set expansion** — more queries, dialect coverage, harder distractors.
2. **Embedding adapters** — additional OCI-available or HF models.
3. **Production adapters** — Oracle 26ai vector index, OCI Generative AI Agents.

## Development setup

```bash
git clone https://github.com/YOUR-USERNAME/oci-arabic-rag-toolkit
cd oci-arabic-rag-toolkit
pip install -e ".[dev]"
python tests/test_normalization.py   # should be 12/12 passing
python benchmarks/run_benchmark.py
```

## Pull requests

- One topic per PR.
- Add a test for any new behavior.
- Run `python tests/test_normalization.py` before pushing.
- For benchmark changes, include before/after numbers in the PR description.

## Code style

- Black-compatible formatting.
- Type hints encouraged but not strict.
- Public APIs need docstrings.

## Discussion

Open an issue first for large changes so we can align on approach.
