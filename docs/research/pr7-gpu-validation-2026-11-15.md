# PR #7 GPU validation

Validated on 2026-11-15 against PR #7 source commit
`9423be82ecc84eba4fff0d447029b1a4fe3be487`, with the dedicated GPU tests added
in this follow-up. No runtime implementation change was needed.

## Results

| Run | Result | Duration |
| --- | --- | --- |
| PyTorch GPU | **20 passed**, no failures or skips | 1.71 s |
| TensorFlow GPU | **20 passed**, no failures or skips | 5.47 s |
| Shared prediction diagnostics and catalog/source delivery | **67 passed**, no failures or skips | 0.15 s |

The third row is standard-library/metadata validation, not GPU execution.
Each GPU command deselected the other framework's 20 tests. JUnit results were
checked for the expected test count and zero failures, errors and skips.

Environment:

- NVIDIA GeForce RTX 5090, 32607 MiB; driver 580.178.04.
- PyTorch 2.11.0+cu128, Python 3.12.13, pytest 9.1.1.
- TensorFlow 2.21.0, Python 3.11.15, pytest 9.1.1; GPU memory growth enabled.
  A separate test virtual environment reused the existing TensorFlow packages.
  This is source-level GPU verification. Package installation still requires
  Python >=3.12 and is covered by the GitHub Python 3.12 workflow.

## What was verified

[`test_training_references_gpu.py`](../../tests/test_training_references_gpu.py)
checks actual GPU placement of outputs, model parameters and gradients:

- PEP soft-threshold outputs, analytic weight/threshold gradients and SGD
  updates for global, dimension, feature and feature-dimension thresholds.
- Fixed-mask rewind/retraining with momentum and weight decay, active zero
  recovery, excluded coordinates, padding and the all-pruned case.
- Same-mode search/retraining checkpoint recovery of weights and support;
  this does not assert full optimizer/training-resume recovery.
- Uniform, popularity and catalog-mixture proposals with correct per-row
  conditional probabilities after eligibility and positive exclusion.
- Local/stateless seed reproducibility without advancing the global stream.
- Extreme float32/float64 count ranges, large exponents, tiny uniform mixtures,
  and explicit rejection of undefined support.

`MODELEVO_REQUIRE_GPU=1` turns a missing framework or GPU into a failure.
The fixture may skip on ordinary CPU CI, but skipped cases are not counted as
GPU evidence. TensorFlow metadata and assertion operations may run on the CPU;
the tests check the relevant result/parameter/gradient devices rather than
claiming every internal operation is a GPU kernel.

An independent read-only review found no blocking issue in the device checks,
analytic expectations, required-GPU gate or checkpoint claims.

## Reproduction

Run the frameworks sequentially with their GPU-enabled Python interpreters:

```sh
MODELEVO_REQUIRE_GPU=1 PYTHONPATH=src python -m pytest \
  tests/test_training_references_gpu.py -k pytorch -q -rs --tb=short

MODELEVO_REQUIRE_GPU=1 PYTHONPATH=src TF_FORCE_GPU_ALLOW_GROWTH=true python -m pytest \
  tests/test_training_references_gpu.py -k tensorflow -q -rs --tb=short

PYTHONPATH=src python -m pytest \
  tests/test_prediction_diagnostics.py tests/test_training_reference_catalog.py -q
```

These are deterministic tensor/component checks, not dataset training or a
live-Agent optimization run. They establish neither uplift/business improvement
nor production-scale memory/latency performance. Mixed precision is outside this
run's scope. Dense masking still retains dense storage.

## Verified source hashes

The target's executed files matched the transferred SHA-256 manifest:

```text
abcf1159f7510a45cbd77a8c4d368954b8090278a171423cfce081bebb7616be  src/model_evo_harness/models/pytorch/embedding_pruning.py
2b0700c2e3145f15e2356f58a1c83bae34200f541e1e0e392b9c51438d0ee084  src/model_evo_harness/models/pytorch/negative_sampling.py
76edc821aa8fc21a52e23d989a3f8ad8fa446c016f67b42eafea8532fcc823fd  src/model_evo_harness/models/tensorflow/embedding_pruning.py
d5009181af2118f956c8f62e2310b2f0cb7bef136ce959cefd2dee1247f3fd1b  src/model_evo_harness/models/tensorflow/negative_sampling.py
df7de752dbd5954ccc7ece4bd4b0d122ddbca1495079eb75273a7c2c59c74f65  src/model_evo_harness/analysis/prediction_diagnostics.py
dbf06a5807fe770a8fa44dc2eb96323462a32df73d4d4020eb45c6a2a668b2bc  tests/test_training_references_gpu.py
dd57fe9d885b35b808210196bad6047d9d0a9eb88444188cf793db959e2eab81  tests/test_prediction_diagnostics.py
```
