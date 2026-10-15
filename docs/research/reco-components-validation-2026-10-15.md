# Reco-papers component validation

The implementation baseline is ModelEvoHarness commit `21e50362013ba6533227ad43bfae90ba7126527d`. The source-paper inventory is pinned to Reco-papers commit `2c617a1659c38f60848c051b377f3faef5df831f`. This change adds 25 component IDs in both PyTorch and TensorFlow, giving 69 method cards, 25 family guides and 140 framework entries including generic TwoTower.

## Executed checks

Local CPU runtime: Python 3.13.13, PyTorch 2.12.0, TensorFlow 2.21.0. The untouched baseline produced **181 passed, 17 skipped, 288 subtests passed**. After the extension and review fixes, `PYTHONPATH=src python -m pytest -q --disable-warnings -ra` produced **230 passed, 17 skipped, 318 subtests passed**. All 17 skips are existing CUDA/TensorFlow-GPU-only tests in explicit interaction and parallel composition modules; no new component test was skipped. The runtime emits existing gast/Python deprecation warnings during TensorFlow tracing.

New checks cover numeric graph proximity and relation attention, normalization, graph/pointer/history masks, causal prefix invariance, candidate conditioning, FM initialization, bilinear scoring, DPP singular kernels, RankNet/DeltaNDCG weighting, PAL's probability product and stable BCE, DICE mask direction, higher-order MetaEmbedding updates, four-stream MM-Rec attention, separate DICM queries, and detached teacher distillation. TensorFlow dynamic graph tests exercise variable sequence/candidate sizes. A same-weight TransAct comparison verifies the two framework encoders implement the same pre-LayerNorm equations.

Independent review identified and resolved masked infinite-score losses, inconsistent normalization order, empty pointer shapes, and an ambiguous DPP contract. DPP is explicitly a fixed-capacity selector, not an unconstrained-cardinality MAP solver. Regression tests retain those cases.

`python -m model_evo_harness.cli catalog-check` reports no missing files, guides, cards or upstream mappings, and validates all 115 PDF dispositions: 24 added-core papers, 18 existing-core papers, 14 shared-mechanism entries, 58 excluded entries and one duplicate. The 24 new paper entries map to 25 component IDs because RankNet and LambdaRank share one overview paper.

`python -m pip wheel . --no-deps --no-build-isolation` succeeds. A separate Python process imports the wheel without either DL framework, resolves all 138 method APIs, reads the source inventory and confirms that all ten added framework modules and the MIT third-party notice are packaged. Generic TwoTower adds the remaining two manifest entries. `git diff --check` is clean.

CI now installs CPU frameworks and runs pytest so framework tests and pytest-style contract tests are not silently omitted. Its configured Python version is 3.12; the local evidence above is Python 3.13.13. A remote CI outcome must be read separately from GitHub.

## Limits of the evidence

These checks validate computational components and Agent-facing contracts. They do not reproduce source-paper datasets or benchmark metrics, test the existing GPU-only paths, construct production data or indexes, or establish online improvement. TensorFlow code follows this project's current Keras API and is not TensorFlow 1.15 compatible. Sources without clear redistribution permission were not transplanted; the adapted DICE objective retains its MIT notice inside both extractable modules.
