# QueryFormer query bridge validation

Baseline: `b05541fd581604815f2d253596e22906fefb4022`. This extension adds three independent tensor components and their single-view composition: `CoTransformerBlock`, `QuerySeqCrossAttn`, `SeqQueryCrossAttn`, and `QueryFormerQueryBridge`. Scope and explicit implementation choices are in the [component guide](../queryformer-components.md). Source: [QueryFormer v1, §4.2, Table 2 and Eq.2–6](https://arxiv.org/html/2609.16548v1#S4.SS2).

## Executed checks

Local CPU environment: Python 3.12.14, PyTorch 2.12.0, TensorFlow 2.21.0, NumPy 2.5.4, pytest 9.1.1. The full suite command was:

```bash
PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=2 python -m pytest -q --disable-warnings -ra
```

Result: **285 passed, 21 skipped, 334 subtests passed**. All skips require a CUDA or TensorFlow GPU; two are the new QueryFormer forward/backward tests. The remaining 19 skips predate this extension. No new GPU execution or speedup is claimed.

Tests were written and observed failing before adding implementations and catalog registrations. Executed checks cover:

- Hand-derived attention results: masked-mean item retrieval, mean/max probes querying context rather than history, and self-attention before the user-only gated cross update.
- Domain order, context concatenation, validity masks, masked NaN padding, zero-length user/item/history inputs, absent output protection after affine biases, and malformed input rejection.
- Finite input and parameter gradients with zero gradients at masked positions; same-weight PyTorch/TensorFlow output and input-gradient agreement.
- TensorFlow graph execution across dynamic batch and token lengths; FP16 score accumulation and finite forward/backward behavior.
- Discovery through method cards, applicability, structure/guide mappings and the source API, plus execution of extracted source without repository imports.

Independent review found no blocking issue. Its additional probes executed both complete source files directly, using float32, float16 and bfloat16 over five user/item/history length combinations, including all-empty and single-side-empty inputs. A fully masked NaN sample was included; outputs and all input/parameter gradients remained finite. A wording ambiguity about domain-specific attention parameters was corrected.

Catalog validation and `python -m pip wheel . --no-deps --no-build-isolation` passed. A separate Python process with `-S` loaded the wheel, verified 77 cards and 156 framework entries, and retrieved/compiled all four QueryFormer APIs from one complete module per framework without importing either DL runtime. The component guide's two Python examples executed successfully, local documentation links resolved, and `git diff --check` passed. The Reco-papers inventory remains 115 PDFs.

## Interpretation

The default two-layer Co-Transformer follows the paper's stated stage order and Table 2's user-to-item cross direction. The selected gate formula, residual/LayerNorm placement, masked-mean item aggregation and detailed parameterization are documented reference choices. The bridge covers Eq.2–4 and returns generated queries plus augmented context. It does not implement Eq.5's downstream sequence evolution and token mixing, H-view tokenization, packed execution, the prediction head or the competition training recipe.

This validates reusable tensor mechanisms and Agent-facing delivery. It does not establish exact equivalence to unpublished author code, reproduce competition metrics, demonstrate production gains or provide TensorFlow 1.15 compatibility.
