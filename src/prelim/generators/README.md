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
5. For dataframe-based tabular synthesizers such as `TabGAN` or `CTGAN`, convert internal NumPy arrays at the wrapper boundary and keep the public PRELIM interface NumPy-based.

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
