# FuxiCTR mechanism coverage

This delivery resolves all **57 rows** in the pinned FuxiCTR inventory to native components
or explicit composition recipes. It adds **43 method cards**, each with independent PyTorch
and TensorFlow code: **37 structural/sequence components, four objectives or penalties,
one pretrained-ID adapter and one LR/DNN control**. Existing components are reused.

This is mechanism coverage. Full named-model training pipelines and all paper variants are
not claimed. The host still owns data encoding, heads, sampling, training, and task evaluation.
Sources are bundled locally and use only their framework and the standard library; no
FuxiCTR/DeepCTR model dependency is required. Model and analysis directories stay separate.

## Complete scope

| Direction | Added mechanisms | Local modules, in both framework directories |
| --- | --- | --- |
| Local crossings and fusion | Gates, instance masks, independent bridge/regulation, multihead bilinear fusion, low-rank token products, field matrices/groups, factorized/quadratic products | `ctr_composition.py` |
| Context-conditioned parameters | Full/low-rank adaptive affine layers, prior-conditioned per-depth gates and detach boundaries | `ctr_conditioning.py` |
| Field parameterization/products | Partner-aware FFM, operation-aware ONN, HOFM/interaction-machine polynomials, holographic and Lorentz products, dense/sparse composition | `field_products.py` |
| Alternative algebra | Logarithmic, complex Euler and adaptive operation interactions | `field_algebra.py` |
| Field connectivity | Hierarchical aggregation, field graph messages, disentangled and multiplicative attention variants | `field_attention.py` |
| Local field patterns | Ordered k-max convolution and convolutional field generation | `field_convolution.py` |
| History relationships | Candidate-in-encoder BST, DMIN refinement, separate DMR matching paths | `history_matching.py` |
| Long history selection | Learned coarse/fine selection, hashing, collision pooling, headwise selection and multi-granularity refinement | `history_retrieval.py` |
| Representation input | Pretrained/learned ID init, sum and concat; raw-logit LR/MLP controls | `ctr_embeddings.py` |
| Training | Detached ensemble consistency, aligned next-event supervision, coarse/fine response losses, disjoint embedding/network penalties | `ctr_training.py` |

## Agent integration

The Agent receives applicable original family guides, method input contracts, structure
patterns and training patterns. `model_api` exposes actual constructors (including local
inherited constructors), calls, output shapes and limitations. `read_references` delivers
whole self-contained modules. The provider automatically supplies unread selected sources
before accepting a candidate. The 57-row map is validated by `load_catalog` and summarized
by `catalog-check`; missing native mappings fail validation.

Select a component from observed evidence inside the current compatible backbone. A longer
catalog is not a fixed exploration order. Preserve compatible previous components, justify
parallel branches with actual feature groups, and compare component, sharing and fusion
ablations. An objective-only change is a valid iteration. Derived tensor prerequisites
require host code when the underlying raw fields already exist; they do not justify an
immediate human data request. Feature requests and business lessons remain dataset-bound.

## Deliberate boundaries and source differences

- EDCN bridge attention uses independently parameterized gates; the pinned source calls
  the first gate twice. QNN restores field-coordinate order before the residual, rather
  than directly flattening head-major products.
- FFM/ONN require real partner-specific banks. Repeating one shared table is a control,
  not proof of field-aware parameterization. FmFM has pair matrices; it does not impose
  the symmetric constraint of FEFM. InterHAt starts from host-contextualized field tokens.
- FiGNN provides one graph update without GRU recurrence; CCPM/FGCNN provide one stage;
  BST provides one target-aware block. Their host stacks and readouts are explicit recipe
  choices. GDCN does not implement the paper's feature-dimension optimizer.
- HOFM and the interaction machine share elementary-symmetric polynomial algebra; their
  order/readout interfaces differ. AFN+/HFM+/AutoInt+ need a separately assessed host MLP
  path. PEPNet.py in the pinned source is a stub; the usable PPNet mechanism is addressed.
- Sequence masks mean True=valid. Invalid padding is sanitized. SIM/ETA/TWIN use dense
  selection over supplied histories, not a production ANN index. Hard top-k indices have
  no gradient; auxiliary coarse supervision trains a separate path. SDIM hash collisions
  and summed pooling differ from nearest-neighbor attention.
- MIRRN uses all retrieved masks and a full off-diagonal channel transform in the
  frequency path. This deliberately differs from ignored masks and diagonal-only `ndd`
  contraction in the pinned source. Recent-history membership is provided by the host.
- Next-event supervision takes already aligned states and positive/negative events,
  applies the mask before reduction and counts both classes. It does not add losses
  automatically to DIEN/DMIN/DMR. Expose the intended states and choose valid negatives.
- Ensemble consistency uses a detached teacher and sums branch losses; weights therefore
  depend on branch count. Explicit parameter penalties are not AdamW. Freezing pretrained
  vectors changes updates, not vocabulary/OOV coverage.

## Source-row recipes

Source snapshot: `183da1c944ba8930fcdcd54114837c081959d748`.
The [machine-readable inventory](../src/model_evo_harness/data/fuxictr_inventory.json)
records each recipe and the actual omitted mechanisms. The original [assessment](research/fuxictr-assessment-2026-11-12.md)
and companion reviews describe the pre-change baseline.

| Source entry | Native method IDs |
| --- | --- |
| [LR](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LR) | `response_mlp` |
| [FM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FM) | `fm` |
| [DSSM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DSSM) | `dssm` |
| [CCPM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/CCPM) | `ordered_field_convolution` |
| [FFM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FFM) | `ffm_pairs` |
| [DNN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DNN) | `response_mlp` |
| [Wide&Deep](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/WideDeep) | `wide_deep` |
| [PNN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/PNN) | `pnn` |
| [DeepCrossing](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DeepCrossing) | `deepcrossing` |
| [HOFM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/HOFM) | `higher_order_fm` |
| [DeepFM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DeepFM) | `deepfm` |
| [NFM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/NFM) | `nfm` |
| [AFM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/AFM) | `afm` |
| [DCN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DCN) | `dcn` |
| [FwFM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FwFM) | `fwfm` |
| [xDeepFM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/xDeepFM) | `xdeepfm` |
| [FiGNN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FiGNN) | `field_graph_interaction` |
| [AutoInt/AutoInt+](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/AutoInt) | `autoint` |
| [FiBiNET](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FiBiNET) | `fibinet` |
| [FGCNN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FGCNN) | `convolutional_field_generation` |
| [HFM/HFM+](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/HFM) | `holographic_interaction` |
| [DLRM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DLRM) | `dlrm_composition` |
| [ONN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/ONN) | `onn_operations` |
| [AFN/AFN+](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/AFN) | `logarithmic_interaction` |
| [LorentzFM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LorentzFM) | `lorentz_interaction` |
| [InterHAt](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/InterHAt) | `hierarchical_field_aggregation` |
| [FLEN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FLEN) | `grouped_field_interaction` |
| [DeepIM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DeepIM) | `deep_interaction_machine` |
| [FmFM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FmFM) | `field_matrix_interaction` |
| [DCN-V2](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DCNv2) | `dcn_v2` |
| [DESTINE](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DESTINE) | `disentangled_field_attention` |
| [EDCN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/EDCN) | `cross_deep_bridge`, `field_regulation` |
| [MaskNet](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/MaskNet) | `instance_mask` |
| [SAM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/SAM) | `sam_interaction` |
| [AOANet](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/AOANet) | `generalized_interaction` |
| [FinalMLP](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalMLP) | `context_feature_gate`, `multihead_bilinear_fusion` |
| [FinalNet](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalNet) | `factorized_interaction`, `ensemble_consistency` |
| [EulerNet](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/EulerNet) | `euler_interaction` |
| [GDCN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/GDCN) | `gated_cross` |
| [WuKong](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/WuKong) | `wukong_block` |
| [QNN-α](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/QNN) | `quadratic_interaction`, `ensemble_consistency` |
| [DIN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DIN) | `din` |
| [DIEN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DIEN) | `dien`, `next_behavior_auxiliary` |
| [BST](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/BST) | `bst_history_encoder` |
| [DMIN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DMIN) | `dmin_interest_refiner`, `next_behavior_auxiliary` |
| [DMR](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DMR) | `dmr_history_matching`, `next_behavior_auxiliary` |
| [TransAct](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/TransAct) | `transact` |
| [SIM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/SIM) | `sim_history_selector`, `coarse_fine_supervision` |
| [ETA](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/ETA) | `eta_history_selector` |
| [SDIM](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/SDIM) | `sdim_collision_pooling` |
| [TWIN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/TWIN) | `twin_headwise_selector` |
| [MIRRN](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/MIRRN) | `mirrn_history_refiner` |
| [APG](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/APG) | `apg`, `adaptive_linear` |
| [PPNet](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/PEPNet) | `pepnet`, `prior_conditioned_gate` |
| [ShareBottom](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/multitask/ShareBottom) | `shared_bottom` |
| [MMoE](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/multitask/MMoE) | `mmoe` |
| [PLE](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/multitask/PLE) | `ple` |

## Verification and practical limit

Tests cover independent source execution, reference delivery, manifest and guide links,
formulas, mapped-weight cross-framework results and gradients, masks/empty histories,
FFT/layout semantics, numerical extremes, shared vocabulary rejection, and frozen-table
updates. Synthetic tensors in these tests are analytic fixtures, not dataset experiments.
No live Agent experiment or GPU benchmark is implied; task improvement requires the host's
full-data experiments and independent evaluation under a fixed budget.

New component metadata separates raw/declared prerequisites in `requires` from candidate-produced inputs in `required_host_tensors`. The latter are exposed in `model_api` and source contracts and do not block discovery. Construct and verify them in the candidate before training. A field layout needs a stable declared axis; locality still needs a justified semantic hypothesis. Next-event negatives require an explicit eligible pool and sampler, not invented observed labels. A recent-N mask can use event order; a time-window mask additionally needs real event timestamps.


Verified locally: **780 tests and 388 subtests passed; 21 GPU-only tests skipped**.
Independent review and `catalog-check` passed. This result validates implementation and
integration contracts, not success rates of live Agent optimization on business datasets.
