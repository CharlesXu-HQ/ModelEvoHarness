# Reco-papers component validation

The implementation baseline is ModelEvoHarness commit `21e50362013ba6533227ad43bfae90ba7126527d`. The source-paper inventory is pinned to Reco-papers commit `2c617a1659c38f60848c051b377f3faef5df831f`. This change adds 25 component IDs in both PyTorch and TensorFlow, giving 69 method cards, 25 family guides and 140 framework entries including generic TwoTower.

## Executed checks

Local CPU runtime: Python 3.13.13, PyTorch 2.12.0, TensorFlow 2.21.0. The untouched baseline produced **181 passed, 17 skipped, 288 subtests passed**. After the extension and review fixes, `PYTHONPATH=src python -m pytest -q --disable-warnings -ra` produced **230 passed, 17 skipped, 318 subtests passed**. All 17 skips are existing CUDA/TensorFlow-GPU-only tests in explicit interaction and parallel composition modules; no new component test was skipped. The runtime emits existing gast/Python deprecation warnings during TensorFlow tracing.

New checks cover numeric graph proximity and relation attention, normalization, graph/pointer/history masks, causal prefix invariance, candidate conditioning, FM initialization, bilinear scoring, DPP singular kernels, RankNet/DeltaNDCG weighting, PAL's probability product and stable BCE, DICE mask direction, higher-order MetaEmbedding updates, four-stream MM-Rec attention, separate DICM queries, and detached teacher distillation. TensorFlow dynamic graph tests exercise variable sequence/candidate sizes. A same-weight TransAct comparison verifies the two framework encoders implement the same pre-LayerNorm equations.

Independent review identified and resolved masked infinite-score losses, inconsistent normalization order, empty pointer shapes, and an ambiguous DPP contract. DPP is explicitly a fixed-capacity selector, not an unconstrained-cardinality MAP solver. Regression tests retain those cases.

`python -m model_evo_harness.cli catalog-check` reports no missing files, guides, cards or upstream mappings, and validates all 115 PDF dispositions: 24 added-core papers, 18 existing-core papers, 14 shared-mechanism entries, 58 excluded entries and one duplicate. The 24 new paper entries map to 25 component IDs because RankNet and LambdaRank share one overview paper.

`python -m pip wheel . --no-deps --no-build-isolation` succeeds. A separate Python process imports the wheel without either DL framework, resolves all 138 method APIs, reads the source inventory and confirms that all ten added framework modules and the MIT third-party notice are packaged. Generic TwoTower adds the remaining two manifest entries. `git diff --check` is clean.

CI now installs CPU frameworks and runs pytest so framework tests and pytest-style contract tests are not silently omitted. Its configured Python version is 3.12; the local evidence above is Python 3.13.13. A remote CI outcome must be read separately from GitHub.

## GPU follow-up — 2026-10-16

After the test host reboot, all **17 previously skipped GPU test methods passed**, with zero failures, errors or skips. The tested source is commit `c3be5c4a431c8b25f8b936ebbb2c558f5051ad7d`, uploaded as an isolated `git archive` snapshot. Its archive SHA-256 is `fcabb880a68a93b429df7a8244defd2dcd9f18a81c8c3ae801d05d63919faf8e`, verified on both hosts. Hardware: NVIDIA GeForce RTX 5090 (32 GB), driver `580.178.04`.

| Environment | Explicit interactions | Parallel branches | Result |
| --- | --- | --- | --- |
| Python 3.12.3, PyTorch 2.14.1+cu130, CUDA 13.0 | 4 passed | 4 passed | 8 passed, 0 skipped; 0.478 s test execution |
| Python 3.11.15, TensorFlow 2.21.0, Keras 3.15.0, CUDA build 12.5.1 | 4 passed | 5 passed | 9 passed, 0 skipped; 15.168 s test execution |

The launcher required a visible GPU, verified a matrix multiplication on `cuda:0` / `GPU:0`, fixed each framework's random seed to 42, and failed on any skipped method or unexpected test count. Existing tests also assert output placement. They cover direct-expansion equivalence and gradients for mixed categorical/numeric fields, missing values, excluded fields, invalid input contracts, named branch routing, branch/gate gradients, shared parameter accumulation and TensorFlow training-context propagation under `tf.function`.

TensorFlow emitted a warning that this build uses PTX JIT compilation for the GPU's compute capability `12.0a`; the run completed successfully. Timing above excludes framework initialization and the initial device probe. Its existing environment uses Python 3.11 and imports the snapshot via `PYTHONPATH`; this is source-module execution evidence, not an installation check against the package's declared Python >=3.12 requirement.

To reproduce the selected classes, run these commands in the corresponding GPU environments with CUDA visible (no skips are acceptable):

```bash
PYTHONPATH=src:tests python -m unittest -v \
  test_explicit_models.PyTorchExplicitTests \
  test_parallel_blocks.PyTorchParallelBlocksTests
PYTHONPATH=src:tests python -m unittest -v \
  test_explicit_models.TensorFlowExplicitTests \
  test_parallel_blocks.TensorFlowParallelBlocksTests
```

Raw logs and result JSON, including framework metadata and SHA-256 hashes of both tested modules and test files, are retained on the GPU host in `modelevo-reco-gpu-c3be5c4-20261016/{pytorch,tensorflow}-gpu.log` and `{pytorch,tensorflow}-gpu-result.json`. The four selected classes exercise the existing explicit-interaction and composition modules; this follow-up does not establish GPU coverage for all 25 newly added components. The earlier CPU results remain a separate run.

## Limits of the evidence

These checks validate computational components and Agent-facing contracts. They do not reproduce source-paper datasets or benchmark metrics, construct production data or indexes, or establish online improvement. GPU evidence is limited to the 17 methods listed above. TensorFlow code follows this project's current Keras API and is not TensorFlow 1.15 compatible. Sources without clear redistribution permission were not transplanted; the adapted DICE objective retains its MIT notice inside both extractable modules.
