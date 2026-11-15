# Native CTR refinements: verification, 2026-11-15

## Scope and environment

Base: ModelEvoHarness main `148f8936502c27dc9e6599fa8e9b7ed287f1e962`.
The changes implement the native components and boundaries described in
[native CTR refinements](../native-ctr-refinements.md). No DeepCTR dependency.

Target: NVIDIA RTX 5090, 32607 MiB, driver 580.178.04.
PyTorch runtime: Python 3.12.13, PyTorch 2.11.0+cu128.
TensorFlow runtime: TensorFlow 2.21.0 with a separate pytest venv over the existing
Python 3.11.15 GPU environment. TensorFlow source-level GPU verification does not
change the package's Python >=3.12 requirement; CI uses Python 3.12.
Existing shared environments were not modified. The target remains running.

## Final GPU result

| Framework | Passed | Failed/errors | Skipped | JUnit time |
| --- | ---: | ---: | ---: | ---: |
| pytorch | 60 | 0 | 0 | 1.85s |
| tensorflow | 62 | 0 | 0 | 8.54s |

Both runs used `MODELEVO_REQUIRE_GPU=1`, which makes missing GPU/runtime fail.
GPU outputs, parameters/gradients are asserted where the tests exercise them;
shape checks, NumPy formula oracles and packed-sequence lengths may execute on CPU.
This does not claim every operation executes on GPU.

```bash
MODELEVO_REQUIRE_GPU=1 PYTHONPATH=src <pytorch-python> -m pytest   tests/test_interaction_variants.py tests/test_sequence_inputs.py   tests/test_sequence_refinements.py -k pytorch -q -rs --tb=short   --junitxml=../pytorch-gpu.xml

MODELEVO_REQUIRE_GPU=1 PYTHONPATH=src TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 <tensorflow-test-python> -m pytest   tests/test_interaction_variants.py tests/test_sequence_inputs.py   tests/test_sequence_refinements.py -k tensorflow -q -rs --tb=short   --junitxml=../tensorflow-gpu.xml
```

Evidence remains under `/home/charles/modelevo-native-ctr-20261115/` on the target:
`pytorch-gpu.xml`, `tensorflow-gpu.xml`, their logs and `source-verification.log`.
All 216 transferred source/test hashes verified. JUnit was retrieved and parsed
independently: the final runs contain 60 and 62 tests and zero failures/errors/skips.

## Behavior covered

- CrossNetMix multilayer expert/routing equations and input finite differences;
  IFM joint first-order/FM weighting and DIFM distinct attention/MLP factors.
- ONN un-summed products, FiBiNET sharing/branch parameters, symmetric FEFM pairs.
- Dice valid-only statistics and frozen evaluation, local pooling modes,
  length/mask agreement, weighted reductions and event alignment.
- DIN/DIEN masked NaN forward/backward, DIEN explicit prefix rejection,
  empty/dynamic histories, four evolution variants and shifted auxiliary loss
  reaching extractor parameters. Auxiliary and regularization scales are fixed.
- DSIN learned bias plus padding/holes/empty-user invariance and finite gradients;
  BST history-only encoding with injected target pooling.

Before the fix, DIN NaN padding caused nonfinite gradients in both frameworks.
A separate review reproduced a pre-existing PyTorch DSIN padding sensitivity
(logit delta 0.00122386 in the regression fixture). DSIN now compacts valid
sessions before recurrence, preserves their original indices for bias encoding,
and restores output positions. Both cases now pass on the real GPU.

## Combined local verification

Python 3.13.13, PyTorch 2.12.0 and TensorFlow 2.21.0:

- `PYTHONPATH=src python -m pytest -q -rs --tb=short`:
  **1352 passed, 112 skipped, 402 subtests passed** in 32.98 seconds.
- All 112 skips require CUDA/TensorFlow GPU on the non-CUDA local host; new
  component GPU execution is the separate result above. This is not a rerun
  of every historical GPU suite.
- 4001 warnings are reported, principally Python 3.13/TensorFlow gast AST
  deprecations and existing framework warnings; no failures.
- `catalog-check`: no missing guides, source files, card implementations or
  source inventory mappings. New API/source-delivery tests passed.
- Wheel built without downloading dependencies; all six new native modules,
  the packaged guide and six method cards verified inside the artifact.
- Independent cross-review checked formulas, gradient reachability, masks,
  source delivery and the DSIN fix. No unresolved blocking findings.

These are implementation and integration checks, not a full dataset experiment,
live Agent run, benchmark speed claim or evidence of marketing uplift.

## Verified source hashes

The following hashes bind the behavior above to the implementation/tests:

```text
b94e405502d758f1019926051ba20424e9736b743adc196cfdc1aa77754b0089  src/model_evo_harness/models/pytorch/interaction_variants.py
a8178c11050f01fc598726e61773470f35620fd4936f0379d4640a88b2c82911  src/model_evo_harness/models/pytorch/sequence_attention.py
22328518b035f1829f47765207656337e7b8c2fd5da6115f5c4f94865d117311  src/model_evo_harness/models/pytorch/sequence_inputs.py
af62a6daa60342d4a45f890f565999200b9f79ddcc07805589fe65e4b0159192  src/model_evo_harness/models/pytorch/sequence.py
0c283131e24cfd43c59a42266294bcffbfbb48d7ba7b95713671e6148b51c389  src/model_evo_harness/models/pytorch/architectures.py
4b9b783e060cac063599f82a2af8b1e3808ad53c5d9e5bf3d9495b573942d7e9  src/model_evo_harness/models/pytorch/interactions.py
03738fc97c8409169f6d2a2b213ee5f4ab957d4e45296b1fb4f565d990c4c6e2  src/model_evo_harness/models/pytorch/field_products.py
923eea97a3cca4546b134563bed3a1472b87caf8a332529ac18500705b993973  src/model_evo_harness/models/pytorch/ctr_composition.py
f9dd66e849046ddcd04053a2ea102600c8c8873a982f99777451887774f5d116  src/model_evo_harness/models/pytorch/history_matching.py
fc0c912886daf98bf9ee4f03d03dc799a251a0ec64f05e53ed4fc4a84f4b6164  src/model_evo_harness/models/pytorch/ctr_training.py
25dd45316ea6b7801a6a5719b283afcdfd7f614a1542b1ce138bd4482310f96d  src/model_evo_harness/models/tensorflow/interaction_variants.py
95bfd05f0740668369bb1b0818e4bd7a845c535bb92d2c78300e5770d96375bc  src/model_evo_harness/models/tensorflow/sequence_attention.py
34a06901f6e0876dc8055e9d852a6690b4383bdab17355a361ea8d8325a85194  src/model_evo_harness/models/tensorflow/sequence_inputs.py
6e7a557d70b11db3de66d305e2915be7f635699904ece069e9cb1a21854cf5b9  src/model_evo_harness/models/tensorflow/sequence.py
e8fd325af0fbb990db7a0525004df00e6c69ebc92d8e1aa0182fdc6c4364c25f  src/model_evo_harness/models/tensorflow/architectures.py
6f3549b81821d7ca0fcaf6c2a3f3ed8767399773bd445ee5173828f63f2fb9fa  src/model_evo_harness/models/tensorflow/interactions.py
1ec5d65a4a2a81fc1aae573ea30ef94fe165285f042bf97da24c093652ea0a0f  src/model_evo_harness/models/tensorflow/field_products.py
f29483031d0c56f4adfde1a1962deadfbd3f423c68356e17b93145a40af0e8cf  src/model_evo_harness/models/tensorflow/ctr_composition.py
23414d9b72fe54655903337d04abff8ef30f902f1a4b23480ccf628865303ad9  src/model_evo_harness/models/tensorflow/history_matching.py
963d0107772d2c3fde8299cd2920014c742e04f9f4f320571e893b9c4a8bd037  src/model_evo_harness/models/tensorflow/ctr_training.py
340844188947e58e9d8a5f2015db0ad1eeeeacdb699e6b91117bf02f0c0cf55f  tests/test_interaction_variants.py
bb586ef090982373a1164bd6ac4bad051a0c4cc1bc8aeef81f6400b0851ca587  tests/test_sequence_inputs.py
bbdb92670d31fa698131015b888b392c7602b5aa9c4349ba52389b9bdf0666af  tests/test_sequence_refinements.py
3e49bafb97baef18742012c11625bec53b9dadffa32d294a195ee1dce080eb8e  tests/test_native_ctr_catalog.py
dede6aa6f98ca8f998fb29bb62fc2bcfc6608a59ca91d8ee5ef217760764a522  src/model_evo_harness/catalog.py
8a8eb4b4701d2cd5157739f7fb3606b05f4e11258d8bea15fbda1fd7b9e59591  src/model_evo_harness/embeddings.py
```
