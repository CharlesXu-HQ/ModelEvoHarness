# Discount-aware components validation — 2026-11-12

> GPU follow-up: the previously untested CUDA/mixed-precision and TensorFlow GPU paths were exercised on 2026-11-14. See [PR #3 GPU validation](pr3-gpu-validation-2026-11-14.md) for the executed cases, results and limits. The original CPU-only record below is preserved.

## Scope

Added `DiscountPreferenceBranch` and `DiscountAwareConversionHead` in native
PyTorch and TensorFlow. These are independently written DANet mechanism
references for response prediction, composed with the host's existing backbone.
The implementation choices and input contract are in
[the component guide](../danet-components.md). No author source was copied.

The Agent receives a marketing family guide, two method cards, structural
comparison guidance, a purchase-only auxiliary training pattern, callable APIs
and complete native source. Model components remain separate from analysis tools.

## Executed checks

Environment: Python 3.13.13, local CPU, installed PyTorch and TensorFlow.

```bash
PYTHONPATH=src python -m pytest -q -ra
```

Result: **469 passed, 21 skipped, 343 subtests passed**, exit 0, 26.98 seconds.
The run emitted 3,922 warnings, predominantly TensorFlow/gast AST deprecations.
All 21 skips are existing tests requiring a CUDA or TensorFlow GPU.

New coverage comprises 38 PyTorch tests, 38 TensorFlow tests, six catalog/Agent
tests and three cross-framework/source-extraction tests:

- Analytic real/imaginary FFT filtering, odd history lengths and residual outputs.
- Target-conditioned purchase attention, promotion-context gates and defined
  all-masked/zero-length purchase behavior.
- Sanitized NaN padding, finite gradients, explicit ratio and shape validation.
- Mean conversion BCE plus positive-and-observed-only discount MSE; ignored
  labels have no auxiliary influence, targets are detached, and an empty
  eligible set contributes differentiable zero.
- TensorFlow dynamic graph execution and scalar auxiliary weights; PyTorch
  CPU autocast with float32 FFT work.
- Matching independently assigned weights produce matching framework outputs,
  losses and input gradients within numerical tolerances.
- Extracted source runs without repository imports or model-wrapper libraries.
- Real-history and realized-label prerequisites are gated separately. The
  Agent proposal path delivers source before its final proposal and records
  its hash. These tests use a deterministic provider stub, not a live LLM.

`catalog-check` reports no missing guides, implementation files or per-card
framework implementations. The catalog now contains 27 families, 84 cards
(82 model methods and two analysis tools), 166 model implementation entries
(including generic TwoTower) and four analysis implementation entries.

## Failures found and fixed

The initial tests failed on the missing implementations and catalog entries.
Review subsequently reproduced a PyTorch autocast failure: the learned spectral
filter could produce bf16/fp16 values unsupported by complex construction or
inverse FFT. Its output is now restored to the FFT work dtype before that step;
odd-length CPU autocast forward/backward regression checks pass.

TensorFlow validation rejects complex labels and boolean/complex auxiliary
weights before casts can discard invalid information. Independent review also
checked dynamic batch/purchase lengths, masked-NaN gradients and dynamic alpha.
No remaining blocking issue was identified in that review.

The first full run had one failing inventory assertion: it assumed every
training pattern belonged to one older paper collection. It now checks that
each referenced method exists in the full catalog, allowing the new separately
attributed DANet pattern. The final full run above includes that correction.

## What this establishes

The component math, gradients, input contracts, standalone source delivery and
Agent-facing registration passed the stated checks. This is not a complete
reproduction of the paper or its production system. CUDA autocast, full-dataset
training, live Agent decisions and business gains were not tested in this task.
Conversion accuracy or auxiliary regression quality alone cannot establish
coupon uplift or net return; those require the host's causal policy evaluation.
