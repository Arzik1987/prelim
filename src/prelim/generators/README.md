## Generators

This directory contains PRELIM data generators. A generator is a small class that learns from an input dataset and produces synthetic feature rows through a common interface.

### Interface

Each concrete generator should:
- inherit from `BaseGenerator`
- set a stable public name through `super().__init__(name, seed=...)`
- implement `fit(self, X, y=None, metamodel=None)` and return `self`
- implement `sample(self, n_samples=1)` and return a NumPy array of generated rows

The common conventions are:
- `X` is a 2D NumPy-like feature matrix
- `y` is optional and only required for generators that depend on labels
- `metamodel` is optional and only required for generators that depend on a fitted predictive model
- failures should raise Python exceptions such as `ValueError` or `RuntimeError`, not call `sys.exit(...)`
- `my_name()` should stay stable; do not mutate `self.name_` to signal internal fallback behavior

### Registration

If the generator should be available through the high-level `prelim(...)` API:
1. Add the class to `src/prelim/generators/__init__.py`.
2. Add an entry to `build_generator(...)` in the same file.
3. Use a short lowercase key such as `kde`, `smote`, or `vva`.
4. If the backend is optional or heavy, add a focused test that stubs the external library instead of training the real model in CI.
5. For dataframe-based tabular synthesizers such as `TabGAN`, `CTGAN`, `TVAE`, `CopulaGAN`, or `GaussianCopula`, convert internal NumPy arrays at the wrapper boundary and keep the public PRELIM interface NumPy-based.
6. If a generator needs an internal representation that differs from the public API, keep that translation inside the wrapper. `ForestDiffusion` and the Bayesian-network wrapper follow this rule in opposite directions: one delegates directly to a NumPy backend, the other discretizes numeric columns internally and reconstructs numeric samples before returning them.
7. If the official implementation is a standalone source repository instead of a pip-installable library, keep that integration explicit in the wrapper. `TabSyn` follows this pattern by calling the official CLI through a configured local checkout instead of reimplementing the training pipeline in PRELIM.
8. If a backend is pip-installable but still heavyweight because it downloads model weights or expects accelerator support, treat it as optional. `GReaT` follows this pattern: the wrapper is simple, but tests should stub the backend rather than fine-tuning a real transformer in CI.
9. If an official repository exposes importable train/sample functions but still expects a custom split dataset layout, keep the dataset translation inside the wrapper. `TabDDPM` follows this pattern by writing temporary split arrays and a helper binary target for the official code, then reconstructing PRELIM rows from the sampled arrays.

If the generator is only for direct imports in tests or experiments, exporting it from `__init__.py` is still preferred for discoverability.

### Tests

When adding a new generator:
1. Add direct behavior coverage in `test/test_generators_behavior.py`.
2. If it is exposed through `prelim(...)`, add or update coverage in `test/test_prelim_behavior.py`.
3. Keep tests deterministic by passing a fixed `seed`.

### Style

- Prefer explicit exceptions over process termination.
- Keep comments short and only where they explain non-obvious logic.
- Return `self` from `fit(...)`.
- Avoid hidden side effects outside generator state needed for sampling.
