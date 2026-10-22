# TAAC tensor component validation

Implementation baseline: `d8af70197b69c2d8a4671a9101c82298067eb6c5`. This batch adds `rankmixer`, `unidot_conditioned_gate`, `unidot_multichannel_pool` and `unidot_fm_highway` in PyTorch and TensorFlow/Keras. The catalog now contains 73 method cards, 25 families and 148 framework entries including generic TwoTower. The 115-PDF Reco-papers inventory is unchanged. Exact interfaces and source/adaptation boundaries are in the [component map](../taac-components.md).

## Executed checks

Local CPU environment: Python 3.12.14, PyTorch 2.12.0, TensorFlow 2.21.0, NumPy 2.5.4 and pytest 9.1.1.

```bash
PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=2 python -m pytest -q --disable-warnings -ra
```

Result: **254 passed, 19 skipped, 330 subtests passed**. The 19 skips are GPU-only checks: 17 existing checks and two new tests that compose all four TAAC components and verify forward/backward device placement. All new CPU component and catalog tests passed.

The new tests cover:

- RankMixer's exact token/head permutation, post-residual LayerNorm and independent per-token FFN weights.
- Condition entering only the SwiGLU gate, sigmoid rather than softmax pooling, and numeric cross-layer/domain interaction readout.
- Padding invariance including masked NaN/Inf, zero-length histories, all-empty domains, learned affine biases and optional field-only Highway mode.
- Finite and connected input/parameter gradients, with masked-position gradients equal to zero.
- Same-weight PyTorch/TensorFlow output and input-gradient comparisons for all four components, plus TensorFlow graph execution with varying batch/history lengths.
- FP16 empty and valid-zero history forward/backward regression in both frameworks. L2 normalization uses float32 accumulation for half types and explicitly assigns zero normalization gradient to a zero vector.
- API/structure/guide registration, applicability without a sequence dependency for field-only components, retained attribution and standalone execution of extracted modules.

Independent review compared the three UniDot components against classes extracted from author commit `1399702006a1fc2f86329c8c6133f3f3ac1e6d0d`. For matched weights and legal nonempty inputs, maximum absolute output difference was zero. It checked RankMixer against paper equations 1 and 3–8 and verified full MIT notices against the pinned license. Review found the FP16 normalization issue above; both failing regressions were reproduced before the fix and passed afterward. Targeted review of the fix also checked float32, float16 and bfloat16 empty/zero-vector gradients and approved the change. These are implementation checks, not reproduction of source-paper experiment results.

## Catalog and package

```bash
PYTHONPATH=src python -m model_evo_harness.cli catalog-check
python -m pip wheel . --no-deps --no-build-isolation
git diff --check
```

All passed. Catalog validation reports no missing guides, implementation files, method implementations or upstream mappings. A separate Python process using `-S` loaded the built wheel, checked 73 cards and 148 entries, resolved all eight new method/framework APIs, retrieved and compiled the four standalone source modules, and verified the packaged UniDot notice. Neither DL framework was imported by the source API.

## GPU status and limits

The authorized target GPU host could not be reached over SSH on 2026-10-22; the final connection attempt timed out. No new remote GPU result is claimed. The previous RTX 5090 validation covered earlier explicit-interaction and parallel-branch modules, not these four components. On a host with a visible CUDA or TensorFlow GPU, the new component checks can be selected with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_taac_components.py -k gpu_forward_backward -ra
```

Each selected framework test must pass rather than skip before reporting GPU coverage. There is no new GPU throughput benchmark, paper-dataset result, production integration or online improvement claim. TensorFlow targets the repository's current Keras API, not TensorFlow 1.15. QueryFormer, LENS, CRAFT and the remaining research candidates are outside this implementation batch.
