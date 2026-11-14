# Research-guided components validation

> GPU follow-up: [PR #3 GPU validation](pr3-gpu-validation-2026-11-14.md) records the 2026-11-14 target-machine checks of these components and the 21 previously skipped GPU cases. The original CPU-only record below is preserved.

Date: 2026-11-11. Base: `c3b8ae520b17da3b66c3e023574de66ed7c1b16e`.
Scope: the four user-selected additions in ModelEvoHarness, represented by five
native components in each framework. No consumer project is changed.

## Result

Full repository command:

```sh
PYTHONPATH=src python -m pytest -q -ra
```

Result: **372 passed, 21 skipped, 340 subtests passed**, exit 0, 24.69 seconds.
The local runtime is Python 3.13.13. The run reports 3,094 warnings, predominantly
TensorFlow/gast Python AST deprecations; warnings were retained in the test output.
The local catalog loads 26 families, 82 method cards, 166 framework entries and
10 shared guides. Catalog/API inspection imports neither PyTorch nor TensorFlow.

## What was checked

- Entropy effective rank versus stable rank on orthogonal, duplicated, zero and
  masked matrices; population token variance; named-layer and aligned-branch
  observations; metrics detached from objective gradients.
- Gradient-times-activation saliency and pairwise-common-valid cosine redundancy;
  counts, deterministic ties, protected interaction groups and impossible budgets.
- Progressive masks are nested, reuse initial ranking, handle empty attention
  rows and preserve parameter counts across dense/fixed/progressive controls.
- Event context changes the encoding without reordering events; relational
  attention normalizes each relation independently and keeps candidates isolated.
- Masked NaN/Inf padding, empty histories, finite gradients, PyTorch/TensorFlow
  assigned-weight agreement, dynamic TensorFlow graphs and standalone source use.
- Capability gating, automatic source delivery to the completion callback and
  exposure of callable diagnostic APIs. Provider checks use a deterministic test
  callback; they are not a live LLM experiment or evidence of better proposals.

## Regressions found and fixed

1. Repeated decimal tokens could acquire artificial centered rank from mean
   roundoff. A failing regression in both frameworks was fixed by translating
   relative to a valid token before computing the centered mean.
2. A private Keras layer's default name produced an invalid `tf.function` scope.
   Explicit internal layer names fixed the failing graph-mode check.
3. `model_api` omitted custom public `__call__` operations. A failing catalog
   integration test now verifies the scoring API reaches the Agent.
4. Overlap counts initially used int64 GEMM, which lacks the required GPU kernel
   support. Count accumulation now uses float64 GEMM and casts the result back to
   int64 (exact integer counts through 2^53). A failing PyTorch dispatch-contract
   test and TensorFlow graph inspection were added before the fix. These tests
   execute the real diagnostic path on CPU; they do not establish GPU execution.
   Kernel references: [PyTorch CUDA BLAS](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Blas.cpp),
   [TensorFlow MatMul registrations](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/core/kernels/matmul_op_real.cc),
   [TensorFlow GPU number types](https://github.com/tensorflow/tensorflow/blob/v2.21.0/tensorflow/core/framework/register_types.h).

## GPU and effectiveness limits

The previously authorized GPU host could not be reached: SSH returned
`Host is down`. No GPU run or dataset training experiment was performed.
The 21 skipped existing tests require CUDA or a TensorFlow GPU: explicit field
components (8), parallel branches (9), QueryFormer (2), TAAC components (2).
All new analytic tests ran locally; no claim of measured GPU throughput or
task-level improvement follows from them.

This addition provides local mechanisms inspired by the cited papers, with
explicit simplifications. It does not reproduce full RankUp, TayFCS, MGDIN, SIF
or CCFormer systems. The last two user annotations asked for stage explanations;
new sample-bag losses, gradient-conflict optimizers and adaptive inference
policies are not included in this implementation.

## Follow-up: separate analysis from model components

The user's directory correction moves both native diagnostic modules unchanged
from `models/<framework>/` to `analysis/<framework>/`. The 82 method cards now
map to 162 model entries (80 methods plus TwoTower, each in both frameworks)
and 4 analysis entries (2 tools, each in both frameworks). Model and analysis
APIs and patterns are separate; complete-source reading and its hash ledger
support both. No consumer project was changed.

- Before implementation, all 10 initial boundary tests failed: no separate
  registry/API/patterns existed and an analysis ID could name an optimization trial.
- After the split, the full suite returned **382 passed, 21 skipped,
  340 subtests passed**, exit 0, in 25.64 seconds. The same 3,094 warnings and
  GPU-only skips remain.
- Two additional model-component-reference rejection cases were then added;
  the final targeted boundary suite returned **12 passed**, exit 0. The control
  model reference succeeds and each analysis reference is rejected.
- Coverage CLI reports no missing implementation files, guides or card/framework
  implementations. Both APIs and source reading remain independent of DL imports.
- Review verified separate Agent context, mixed model/analysis source reads and
  framework boundaries. Diagnostic tensor source code was moved without changes.

These are interface, numerical and catalog checks. This directory follow-up
does not add automatic activation hooks, a tool executor, GPU training or a live
Agent experiment. Analysis-only proposals are rejected before candidate evaluation;
screened-and-retrained models remain valid model/custom experiment proposals.
