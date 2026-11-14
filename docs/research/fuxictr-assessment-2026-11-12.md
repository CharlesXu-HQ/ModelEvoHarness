# FuxiCTR: additions worth making to ModelEvoHarness

## Review scope

Inspected the official FuxiCTR source snapshot
[`183da1c`](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748),
including its 57-row model inventory and selected distinguishing implementations,
against Harness `0888cd5` (the development version including DANet). GitHub main
was still `c3b8ae5` when checked. This is a source-based assessment; no model
experiment or effectiveness comparison was performed.

Companion records:

- [Complete inventory dispositions](fuxictr-inventory-2026-11-12.md).
- [Model mechanism review](fuxictr-model-review-2026-11-12.md).
- [Training and input-contract review](fuxictr-training-review-2026-11-12.md).

## Conclusion

FuxiCTR provides useful reference implementations for local changes inside an
existing backbone: conditional masking, communication between branches,
field-pair parameterization, alternative multiplicative layers, and retrieval
before long-history encoding. It also makes representation choices and some
model-specific objectives explicit. These can expand the Agent's experiment
space while preserving the current task, dataset and evaluator.

The inventory contains substantial overlap. FM, DeepFM, DCN/DCNv2, DIN/DIEN,
APG, PEPNet-related gates and multitask experts already have local cores.
EDCN, FFM/ONN, FLEN, FGCNN/CCPM and BST have local design guidance without their
own code. A shared model name does not imply implementation parity, and a new
name does not imply a new primitive.

## Mechanism-level opportunities

| Direction | Actual increment and existing overlap | Evidence that should justify an experiment |
| --- | --- | --- |
| Instance-conditioned masks | [MaskNet](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/MaskNet/src/MaskNet.py) generates coordinate masks from the original embedding and supports serial/parallel blocks. Existing FiBiNET/PEPNet gates are useful controls, not identical implementations. | Supported differences in field relevance across contexts; compare masks, block count and sharing under the same fields and budget. |
| Intermediate branch communication | [EDCN](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/EDCN/src/EDCN.py) supplies cross/deep bridge and field regulation. [FinalMLP](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalMLP/src/FinalMLP.py) combines dual context gates with head-partitioned bilinear fusion. Local `ParallelBranches` routes tensors but does not implement these fusion rules; `PredictiveBilinear` already supplies a simpler related term. | Branch ablations show complementary residuals; isolate fusion/bridge from additional capacity and from input gating. |
| Gated and multiplicative cross layers | [GDCN](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/GDCN/src/GDCN.py) gates cross updates. [FinalNet](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalNet/src/FinalNet.py) and [QNN-alpha](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/QNN/src/QNN_alpha.py) offer local multiplicative compositions. FinalNet's sum form partly reduces to existing LatentCross, so its value includes recipes and objectives rather than a wholly new operation. | Evidence of a representation/interaction bottleneck after simple controls; hold depth, inputs and compute comparable. |
| Field-pair structure | FFM/ONN make representations depend on partner field or operation; [FmFM](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FmFM/src/FmFM.py) adds field-pair vector/matrix transformations; [FLEN](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FLEN/src/FLEN.py) separates within/across-group interactions. Local FwFM has scalar pair weights and local FiBiNET shares its bilinear transform. | Specific well-supported field pairs/groups underfit; check vocabulary, pair support and parameter cost rather than inventing new raw features. |
| Long-history retrieval and refinement | [LongCTR](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR) provides SIM/ETA/SDIM/TWIN/MIRRN implementations. Candidate-aware history pooling already exists locally; approximate selection, retrieved subsets and multi-granularity refinement provide additional hypotheses. | Actual long ordered pre-decision histories, a loss of signal from truncation or a measured attention-cost bottleneck. Declare masks, timestamps, retrieval budget and candidate availability. |
| Feature-graph and attention factorization | FiGNN, DESTINE and InterHAt provide more specialized field-message/attention decompositions. Existing GraphSAGE/KGAT operate on different graph contracts; AutoInt and progressive interaction are closer controls. | A specific missing field relation or interaction-order hypothesis that simpler local attention controls do not explain. |

These directions are not a closed menu or a mandatory order. For a rich fixed
tabular marketing dataset, the first four typically require fewer new input
contracts than long-history methods. Their practical order must follow measured
residuals, available fields, prior trials and the experiment budget.

## Representation and training additions

- Make field-role contracts explicit: shared vocabulary/embedding identity,
  field dimensions, numeric and sequence encoding, padding/OOV behavior, and
  trainable versus frozen pretraining. FuxiCTR's
  [embedding layer](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/pytorch/layers/embeddings/feature_embedding.py)
  is a concrete reference. This helps an Agent attribute a gain to representation
  changes instead of silently changing the inputs while changing architecture.
- Evaluate model-linked auxiliary objectives separately: FinalNet trains branch
  predictions toward a detached fused prediction; QNN-alpha uses consistency
  across forward passes. The existing teacher/student distillation objective
  covers a different task. Expose each loss term and gradient boundary, with an
  architecture-only control before claiming the loss helped.
- Complete existing sequence-method objectives where appropriate: the local
  DIEN reference omits next-behavior auxiliary supervision. Add it only with
  explicit positive/negative event alignment, sampling provenance and valid
  transition masks; the upstream DIEN/DMIN auxiliary implementations have
  source-visible shape/reduction concerns documented in the training review.
- Keep generic learning-rate, clipping, weighting and sampling strategies in
  the existing training system. FuxiCTR's grid tuner and trainer are useful
  operational references but do not supply the Agent's hypothesis/evidence loop.

## Recommended integration shape

1. Add small native PyTorch/TensorFlow components and explicit composition
   recipes; reuse local primitives where the computation is already present.
   Preserve an existing backbone ID for these local changes.
2. Register input/output contracts, applicable evidence, controlled ablations,
   known failure modes and complete source. Missing data declarations must
   distinguish genuinely absent business inputs from a model's optional needs.
3. Use existing horizontal composition for repeated branches. Make sharing and
   fusion deliberate choices; one working branch does not prove that replicating
   it across all field groups helps.
4. Keep trainable structures in `models/`, losses in training components,
   observation/screening tools in `analysis/`, and orchestration in the host.
5. Validate mathematical identities, masks, gradients, framework agreement and
   source extraction before a full-data GPU experiment. Retain the causal
   evaluator for coupon decisions; CTR benchmarks cannot establish uplift.

Upstream's README lists TensorFlow for only five of its 57 rows, and many model
classes import FuxiCTR's `BaseModel`, embedding layer and trainer APIs. Direct
wrapping therefore does not meet the local independent dual-framework contract.
The detailed reviews also record source-level edge cases; upstream implementation
availability alone is not sufficient validation for an Agent-facing reference.
