# PR #3 GPU follow-up validation

Date: 2026-11-14. [PR #3](https://github.com/CharlesXu-HQ/ModelEvoHarness/pull/3) was already merged; its GPU checks were pending because the target was offline. [PR #5](https://github.com/CharlesXu-HQ/ModelEvoHarness/pull/5) is also merged at `8f8d4e92356bd08a174b718ef43eeb6d0d0e2eb4`.

The eight PR #3 tensor source files are unchanged between its head `0888cd51c184a64da81d75e858d6671a4dc8b72d` and the current main tree. This follow-up tests those sources on the target GPU. No production implementation change was needed.

## Results

| GPU run | New PR #3 cases | Previously skipped cases | Result |
| --- | ---: | ---: | --- |
| PyTorch | 18 | 10 | **28 passed, 17 subtests passed**, 2.40 s |
| TensorFlow | 19 | 11 | **30 passed, 17 subtests passed**, 34.56 s |
| Total | **37** | **21** | **58 GPU tests passed; no skips or failures in the selected runs** |

The 21 skips mentioned in the original PR report were older explicit-interaction, parallel-composition, QueryFormer and TAAC GPU cases. They did not constitute GPU coverage for PR #3's new components. This follow-up adds dedicated coverage and executes all 21 older cases as well. Opposite-framework cases are deselected, not counted as passes.

Environment:

- NVIDIA GeForce RTX 5090, 32 GB.
- PyTorch 2.11.0+cu128, Python 3.12.13.
- TensorFlow 2.21.0, existing Python 3.11.15 environment, GPU memory growth enabled. This is source-level numeric verification; the package still requires Python >=3.12, with installation/regression checks in the Python 3.12 GitHub workflow.

## New coverage

[`test_research_components_gpu.py`](../../tests/test_research_components_gpu.py), 16 cases:

- Representation effective/stable rank, variance and branch similarity against analytic values; float16 input promotion, masked NaNs and empty history.
- Gradient screening from a real differentiable objective, exact overlap counts, redundancy and preservation of protected interaction-only groups.
- Progressive interaction edge budgets, deterministic ties, nested masks, finite parameter/input gradients and zero masked contribution.
- Contextual history plus independently normalized relation attention, float32/mixed float16, candidate independence, empty histories and all-empty sequences.

[`test_discount_aware_gpu.py`](../../tests/test_discount_aware_gpu.py), 21 cases:

- Complete branch → conversion/discount head → purchase-only objective → backward → optimizer step on GPU.
- Float32, float16 and bfloat16 compute policies with supported float32 parameters. PyTorch half-precision parameters remain outside the component contract.
- Odd FFT length 3, even non-power-of-two length 6, and zero-purchase length-5 history. Wrappers verify the actual rFFT/irFFT result devices and complex64/float32 working dtypes.
- Masked NaN histories/purchases, exact zero masked input gradients, all-masked rows, valid-purchase discount loss, detached labels and differentiable zero auxiliary loss when no eligible purchase exists.
- TensorFlow mixed_float16 graph execution with one dynamic signature across `(B=2,L=3)` and `(B=1,L=0)`, checking returned gradients and one trace.

Floating results and gradients are explicitly checked for GPU placement. TensorFlow shape/assert/integer metadata may legitimately execute on the host; these tests do not claim every internal operation is a GPU kernel. An independent read-only review found no blocking incorrect assertion or false-positive path in the new tests.

## Reproduction

Run the two framework suites separately using their GPU-enabled interpreters. Set `PYTHONPATH=src`; use `TF_FORCE_GPU_ALLOW_GROWTH=true` for TensorFlow.

```sh
python -m pytest tests/test_discount_aware_gpu.py tests/test_research_components_gpu.py \
  tests/test_explicit_models.py::PyTorchExplicitTests \
  tests/test_parallel_blocks.py::PyTorchParallelBlocksTests \
  'tests/test_queryformer_components.py::test_query_bridge_gpu_forward_backward[pytorch]' \
  'tests/test_taac_components.py::test_all_four_components_gpu_forward_backward[pytorch]' \
  -k pytorch -q -rs --tb=short

python -m pytest tests/test_discount_aware_gpu.py tests/test_research_components_gpu.py \
  tests/test_explicit_models.py::TensorFlowExplicitTests \
  tests/test_parallel_blocks.py::TensorFlowParallelBlocksTests \
  'tests/test_queryformer_components.py::test_query_bridge_gpu_forward_backward[tensorflow]' \
  'tests/test_taac_components.py::test_all_four_components_gpu_forward_backward[tensorflow]' \
  -k tensorflow -q -rs --tb=short
```

The new files also collect and skip cleanly on a machine without a GPU: **37 skipped**. A skipped GPU test is not execution evidence.

## Interpretation

No GPU compatibility or numerical failure was found in the tested cases. These are focused component/training-step checks using analytic tensors, not full-dataset experiments or a new live-Agent run. They do not establish model quality, coupon uplift, net return, throughput or peak-memory behavior for a production-sized workload.

## Verified source hashes

The following SHA-256 values are checked against the files actually executed on the target.

```text
29b69f9088bd48de8ca0c4bb6274433138c31898c04805b6c8b16863ce8fbb42  src/model_evo_harness/analysis/pytorch/research_diagnostics.py
779cb66b2c6db619564982de4d8986fd7a67e6cb404cb0ea90c3a86d57bb221d  src/model_evo_harness/analysis/tensorflow/research_diagnostics.py
ba2d696a7e4d43c2e2e121b3647407b4bedd7bacd3c0c9c520e9377ccaea9fdf  src/model_evo_harness/models/pytorch/progressive_interaction.py
dab9397a25bb3fe4b1159d8d31b2a487914f77aec42277309e9c7a395d454fc5  src/model_evo_harness/models/pytorch/contextual_history.py
a8eb961e8b5c56c832091431e31b0eecbce98fe388b7d836e48828a6744c4c6e  src/model_evo_harness/models/pytorch/discount_aware.py
308e8e8976e6cd4653d5d8cfb932f213cd52ddcc9919046df2a52e2064c2b2cb  src/model_evo_harness/models/tensorflow/progressive_interaction.py
a0305cfdf0b9afbfadef13e3cc9ccabf08d414cd35d7f06910829ef5fa07f5e0  src/model_evo_harness/models/tensorflow/contextual_history.py
c83856f7d643d9ab0985d3b05ca3aae7a185b15996fb53687f3c3ee412a97f9d  src/model_evo_harness/models/tensorflow/discount_aware.py
8d2ed1747fc18da90e9781f16f6e0eb941b842979b6622986d914474ff5f0f77  tests/test_discount_aware_gpu.py
928a3a726a38602479bfe839b8ba5f796f47e16693511215a749508cfc6b006a  tests/test_research_components_gpu.py
```
