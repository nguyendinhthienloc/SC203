# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-11-18
### Added
- spaCy batching (`tokenize_and_pos_pipe`) with configurable `--batch-size` and `--n-process`.
- Input validation (`validate_inputs`) for column presence, empty text pruning, label normalization, id generation.
- Nominalization detection modes: `strict`, `balanced`, `lenient` plus configurable context window.
- Epsilon-stabilized PMI & log-odds with `collocation_min_count` parameter.
- Deterministic seeding (`--seed`) across Python & NumPy.
- Unified logging system with `--verbose` and `--debug` flags.
- Extended CLI flags (`--skip-keywords`, `--min-freq-keywords`, `--config`).
- Benchmark script (`benchmarks/benchmark_pipeline.py`).
- Additional tests: log-odds stability, cleaning citations, spaCy pipe equivalence, statistical edge cases, nominalization modes, end-to-end tiny pipeline.
- Packaging via `pyproject.toml` with optional `dev` extras.

### Changed
- Replaced per-row processing with efficient spaCy pipeline streaming.
- Hardened collocation and keyword extraction numeric stability (Haldane–Anscombe correction & safe divisions).

### Documentation
- README expanded with advanced flags, determinism, benchmarking, formatting instructions.

## [0.1.0] - 2024-12-01
### Added
- Initial implementation of ingestion, cleaning, POS tagging, lexical metrics, nominalization (single strategy), collocations, keywords, statistics, and visualization.
- Basic CLI and smoke tests.

---

Future plans: CI optimizations, caching strategies, expanded nominalization heuristics, multilingual model support.
