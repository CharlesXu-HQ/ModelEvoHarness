# Horizontal composition validation — 2026-10-05

## Scope

This change adds a general research axis for parallel subnetworks inside a backbone, a declared topology contract, bounded source reading, and native PyTorch/TensorFlow composition helpers. Branch and fusion types are supplied by candidate code; they are not restricted to named catalog models.

The checks below establish contract and execution behavior. This run did not call a live LLM provider, train on a public dataset, or measure an uplift improvement. Tensor inputs are deterministic unit-test fixtures, not a substitute dataset or benchmark. Existing full-data experiment reports remain separate.

## GPU execution

Hardware: NVIDIA GeForce RTX 5090, driver 580.178.04. Both environments used the same source and tests, whose SHA-256 values are recorded below.

| Environment | Execution result |
| --- | --- |
| PyTorch 2.14.1+cu130 | Four GPU tensor tests and the independent source-import test passed. TensorFlow class skipped because this environment does not install TensorFlow. |
| TensorFlow 2.21.0 / Keras 3.15.0 | Five GPU tensor tests and the independent source-import test passed. PyTorch class skipped because this environment does not install PyTorch. |

All nine framework-specific tensor tests executed across the two GPU environments. They checked named routing with heterogeneous arguments, fusion into an existing head, each branch's effect on outputs, gradients to branches and a learned gate, shared versus independent parameter registration and gradient accumulation, and rejection of missing/extra branch inputs. TensorFlow additionally checked training-context propagation to branches and fusion under `tf.function`. Output device checks passed on CUDA / GPU:0.

The native example also executed in each environment. Both independent branches with concatenation/projection and tied branches with learned gating produced `(2, 1)` outputs through the same prediction head. This is an API execution check, not a trained-model result.

Reproduction (run each command in the corresponding GPU environment):

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_parallel_blocks.py -v
PYTHONPATH=src python examples/parallel_subnetworks.py pytorch
# In the TensorFlow environment:
PYTHONPATH=src python examples/parallel_subnetworks.py tensorflow
```

The tensor test classes explicitly skip when their framework has no GPU; a skipped class is not counted as a GPU pass. Logs were retained in `modelevo-horizontal-validation-20261005/{pytorch,tensorflow}-gpu.log` on the test host.

## Protocol and integration checks

The local regression suite covers required decisions, actual-field subsets, same-method multiple instances, graph references/cycles, sharing declarations, execution-path/output declarations, backward compatibility, and component inheritance. Reference tests confirm that a parallel proposal reads the correct native composition source and stores its actual hash before returning candidate code.

A two-round mock-host integration test persists an initial parallel recipe, records one assessment per branch and fusion, then changes a training component while preserving the existing graph and backbone identity. The next proposal receives these records through `composition_sources`. The mocked evaluator carries no model-quality evidence; component attribution stays `unverified`.

CouponEvo separately validates import availability and propagates field groups and this required proposal contract into provider and review contexts. Its review instruction requests runtime evidence for sharing/gradients where available; the declaration validator does not manufacture that evidence.

## Tested execution-source hashes

| File | SHA-256 |
| --- | --- |
| `src/model_evo_harness/models/pytorch/composition.py` | `2d0004a14e2b698874d27548f70c38490074ec6e509897b8aeaf15f4d3a6b27f` |
| `src/model_evo_harness/models/tensorflow/composition.py` | `bd059df18bfd46bea59e35dd0dffc5dbe2d81f4a9aa900164267d5cfbfdd119c` |
| `tests/test_parallel_blocks.py` | `a271b0a0f02c15a8094ef8e39f4a8937bbdbf8256c0605da6ad8559a67678d08` |
| `examples/parallel_subnetworks.py` | `19b1f56905dd099dac567ced2e593eaa026d242f527cc6af89dea0cbac69d7c2` |

Whether a live Agent chooses useful partitions, writes suitable representations and improves a real task remains an empirical question. Meaningful follow-up needs real available inputs, fixed controls and dataset-bound analysis; merely adding branches does not establish value.
