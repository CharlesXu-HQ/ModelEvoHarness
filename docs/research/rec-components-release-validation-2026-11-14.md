# Recommendation component release validation

Date: 2026-11-14. Release: `0.3.0`. Implementation scope and usage are described in [component contracts](../rec-model-components.md); the preceding [source assessment](rec-models-assessment-2026-11-14.md) records the baseline rather than the final implementation status.

## Verified scope

- Native PyTorch and TensorFlow CoAction, ComiRec-SA and ComiRec-DR components.
- Explicit identity/ReLU CIN and optional per-hash normalized SDIM, with existing defaults retained.
- Framework-neutral COLD-inspired subset/retraining protocol, including protected interaction groups, comparable host evidence and quality/latency selection.
- Agent capability screening, complete local source delivery and model/analysis API separation.
- TensorFlow DeepCrossing declared-input-width regression and independent numeric crosschecks of existing overlapping structures.

## Test evidence

| Check | Environment | Result |
| --- | --- | --- |
| Full suite: `PYTHONPATH=src python -m pytest -q --tb=short` | macOS, Python 3.13, PyTorch 2.12.0, TensorFlow 2.21.0 | **998 passed, 35 skipped; 392 subtests passed**, 35.46 s |
| New GPU tests: `pytest tests/test_rec_components_gpu.py -k pytorch -q -rs` | NVIDIA GeForce RTX 5090, Python 3.12.13, PyTorch 2.11.0+cu128 | **7 passed, 7 deselected**, 1.21 s |
| New GPU tests: `pytest tests/test_rec_components_gpu.py -k tensorflow -q -rs` | Same GPU, Python 3.11.15, TensorFlow 2.21.0 | **7 passed, 7 deselected**, 15.08 s |
| `python -m model_evo_harness.cli catalog-check` | Local source | Exit 0; no missing implementation files, card implementations or local guides |
| `python -m build --no-isolation` | Local Python 3.13 | Wheel and sdist built; new sources, guides and registries present |
| Installed wheel, outside the source checkout | Local Python 3.13 | 131 cards load; both framework APIs and all four new method references resolve without importing DL runtimes |

The local skips are GPU-only tests. The 14 new GPU cases were subsequently executed on the target; older GPU cases were not rerun in this release check. GPU cases assert device placement of outputs and gradients, finite backpropagation, and zero contribution from masked NaN padding. Separate FP16 cases exercise normalized SDIM with zero and tiny valid bucket vectors. No GPU test silently falls back to CPU.

The TensorFlow GPU environment is a pre-existing Python 3.11 runtime used for source-level numeric verification. The published package requires Python >=3.12; package installation is covered separately by the Python 3.12 GitHub workflow. This GPU run does not establish package support for Python 3.11. The local run emits dependency AST deprecation warnings; these are not test failures.

## Review and corrections

Independent model-math review checked parameter layout, normalization axes, routing stop-gradient, empty/padded histories, numeric oracles and framework parity. It found an FP16 gradient overflow in the initial normalized SDIM implementation: a `1e-6` norm floor could produce a derivative too large for float16. Four regression cases reproduced the failure. Both implementations now use a common `1e-4` norm floor, normalize half inputs in float32, and pass the regressions and GPU checks. This correction affects only the new normalization option.

A separate review checked the COLD workflow and shared-source path boundary. It found no remaining blocker: protected groups survive, all candidates invoke host retraining, incomparable contexts are rejected, and selection uses measured feasible quality/latency results. A full-suite contract failure exposed a missing `trainer` requirement in COLD registry metadata; both framework entries were corrected before the passing full run above.

Final integration review found a stale journal package-version constant. A new test failed with `0.2.0 != 0.3.0`; the constant was aligned and the generated/persisted journal regression passed. The final full suite above includes this fix. Independent integration review found no other blocker.

## GPU source identity

The following SHA-256 values matched between the local release sources and the GPU work directory:

| Source relative to `src/model_evo_harness/` | SHA-256 |
| --- | --- |
| `models/pytorch/coaction.py` | `62f753a47d977548b86ed5c15637f3c9d74e755c227d109d546730c866994ab2` |
| `models/pytorch/multi_interest.py` | `bd3bafd7774afb56c5cc08ec0425028545fcacd2c1ec02460e4ed6e6927c18c7` |
| `models/pytorch/interactions.py` | `95eb4be38926f80180c3dc35adef4905da2343ec4d217771d431c8310407ae4f` |
| `models/pytorch/history_retrieval.py` | `0764ea11913caf9a12864bfda55a181d709b6870f8047684556cb718c0ece80f` |
| `models/tensorflow/coaction.py` | `bf5b5038a1db7d4ff036cc1ed8dffa26c4a35b454f1acd3308279b7c8341f448` |
| `models/tensorflow/multi_interest.py` | `e1e19a74b358c88d76b1f3494770dafd4b175ea725fad3791956b201e304c4ed` |
| `models/tensorflow/interactions.py` | `caaa999335c02d688636631ddb5f05da8f0b34a05521c4ca6dcb0744da6c3e41` |
| `models/tensorflow/history_retrieval.py` | `441e867cb3f6ad51a3bcd4d12e272fc34a6cf7705d606c7848a8ddaab9929e61` |
| `models/tensorflow/paper_ranking.py` | `56d088387b0754bc2769db00e0937924c5bfbbb7dab042d9d11d9cb56c427a6a` |

The GPU test file SHA-256 is `fd55deda2fbae3da7f666e8b8b8dd3bbb134d0a59b9feecbbafceef6b9849231`.

## Limits

These are component, protocol and integration checks. There was no new full-dataset model training, live DeepSeek experiment, paper-score reproduction or coupon/uplift improvement measurement. COLD tests validate callback enforcement and decisions; actual physical pruning, retraining artifacts and serving measurements remain the host's responsibility. The new reference components are available for subsequent Agent experiments under the existing dataset-scoped evidence rules.
