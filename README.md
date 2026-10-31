# tinygbm

> A tiny gradient-boosting machine built from first principles in C++, with a NumPy reference implementation, Python bindings, tests, and benchmarks.

`tinygbm` is a deliberately small systems-and-machine-learning project. The goal is not to compete with XGBoost or LightGBM. The goal is to make every important step visible: how residuals are formed, how a tree chooses a split, how predictions are updated, and what changes when the implementation moves from Python to optimized C++.

## Why this project exists

It brings together:

- **ML theory** — gradient boosting, residuals, loss functions, and greedy split selection
- **C++ engineering** — data layout, ownership, cache behavior, and performance-conscious implementation
- **Numerical computing** — a small NumPy implementation used as executable ground truth
- **Python interoperability** — bindings that make the C++ implementation easy to experiment with
- **Concurrency** — a natural place to explore safe parallelism in training and prediction
- **Benchmarking** — measured comparisons instead of intuition-driven optimization
- **Validation** — C++ and Python tests that keep the implementation honest

The same attention to invariants, complexity, memory, and edge cases applies throughout the project.

## Current project shape

```text
tinygbm/
├── include/tinygbm/   Public C++ headers
├── src/               C++ implementation
├── python/            Python bindings and package surface
├── reference/         Small NumPy implementation used as ground truth
├── tests/cpp/         GoogleTest unit tests
├── tests/python/      pytest tests against reference and sklearn
├── bench/             Benchmark scripts and results
└── docs/              Technical write-up and charts
```

The repository is intentionally being built in layers. Empty directories and placeholder files mark the planned boundaries before each component is implemented and tested.

## Learning path

1. Make the NumPy reference implementation correct and readable.
2. Implement the same behavior in C++ with explicit data structures.
3. Add a Python extension and verify parity with the reference.
4. Add tests for numerical correctness, edge cases, and model behavior.
5. Benchmark the hot paths and investigate memory/cache effects.
6. Add carefully scoped multithreading where measurements justify it.
7. Document the results, trade-offs, and limitations.

## Design principles

- Correctness before cleverness
- The reference implementation is the behavioral oracle
- Every optimization should have a benchmark
- Every behavior change should have a test
- Small APIs, explicit invariants, and readable code
- No claim of performance without reproducible measurements

## Status

Early-stage scaffold. The reference implementation is the first executable milestone; the C++ core, bindings, tests, benchmarks, and documentation will be added incrementally.

## License

To be decided when the project is ready for external use.
