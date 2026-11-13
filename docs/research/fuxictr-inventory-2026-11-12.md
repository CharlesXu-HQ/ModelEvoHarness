# FuxiCTR inventory comparison — 2026-11-12

## Scope and counting

Compared the [official model inventory](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/README.md) at `183da1c944ba8930fcdcd54114837c081959d748` with ModelEvoHarness at `0888cd51c184a64da81d75e858d6671a4dc8b72d`, including the pending DANet additions. This is source inspection, not a training benchmark.

The upstream README has 57 rows: 22 have a corresponding local native core, seven have local design guidance without dedicated code, two are generic baselines, and 26 have no dedicated local method. These are inventory dispositions, **not** 26 independent missing mechanisms or a claim of full parity for the 22 overlapping cores. For example, a scalar field-pair weight is not a matrix field-pair interaction, and APG/PPNet variants require finer inspection. The framework labels below come from upstream: only five README rows declare TensorFlow; they do not meet our full dual-framework requirement as-is.

Existing evidence: [model registry](../../src/model_evo_harness/data/model_implementations.json), [interaction cores](../../src/model_evo_harness/models/pytorch/interactions.py), [multi-task/scenario cores](../../src/model_evo_harness/models/pytorch/multitask.py), [structure extensions](../../src/model_evo_harness/knowledge/structure_extensions.md). The latter still says “44 method cards” in its opening paragraph; the current catalog has 84. That stale narrative should be corrected in a future implementation pass, and documented hypotheses must remain distinguishable from executable source.

| Upstream entry | Local disposition | Upstream framework labels | Evidence |
| --- | --- | --- | --- |
| LR | Generic baseline; not counted as a new research mechanism | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LR) |
| FM | Existing native core: `fm` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FM) |
| DSSM | Existing native core: `dssm` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DSSM) |
| CCPM | Local design guidance; no dedicated native model/card | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/CCPM) |
| FFM | Local design guidance; no dedicated native model/card | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FFM) |
| DNN | Generic baseline; not counted as a new research mechanism | `torch`, `tf` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DNN) |
| Wide&Deep | Existing native core: `wide_deep` (variant coverage needs separate inspection) | `torch`, `tf` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/WideDeep) |
| PNN | Existing native core: `pnn` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/PNN) |
| DeepCrossing | Existing native core: `deepcrossing` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DeepCrossing) |
| HOFM | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/HOFM) |
| DeepFM | Existing native core: `deepfm` (variant coverage needs separate inspection) | `torch`, `tf` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DeepFM) |
| NFM | Existing native core: `nfm` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/NFM) |
| AFM | Existing native core: `afm` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/AFM) |
| DCN | Existing native core: `dcn` (variant coverage needs separate inspection) | `torch`, `tf` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DCN) |
| FwFM | Existing native core: `fwfm` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FwFM) |
| xDeepFM | Existing native core: `xdeepfm` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/xDeepFM) |
| FiGNN | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FiGNN) |
| AutoInt/AutoInt+ | Existing native core: `autoint` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/AutoInt) |
| FiBiNET | Existing native core: `fibinet` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FiBiNET) |
| FGCNN | Local design guidance; no dedicated native model/card | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FGCNN) |
| HFM/HFM+ | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/HFM) |
| DLRM | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DLRM) |
| ONN | Local design guidance; no dedicated native model/card | `torch`, `tf` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/ONN) |
| AFN/AFN+ | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/AFN) |
| LorentzFM | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LorentzFM) |
| InterHAt | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/InterHAt) |
| FLEN | Local design guidance; no dedicated native model/card | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FLEN) |
| DeepIM | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DeepIM) |
| FmFM | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FmFM) |
| DCN-V2 | Existing native core: `dcn_v2` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DCNv2) |
| DESTINE | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DESTINE) |
| EDCN | Local design guidance; no dedicated native model/card | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/EDCN) |
| MaskNet | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/MaskNet) |
| SAM | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/SAM) |
| AOANet | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/AOANet) |
| FinalMLP | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalMLP) |
| FinalNet | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/FinalNet) |
| EulerNet | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/EulerNet) |
| GDCN | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/GDCN) |
| WuKong | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/WuKong) |
| QNN-α | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/QNN) |
| DIN | Existing native core: `din` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DIN) |
| DIEN | Existing native core: `dien` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DIEN) |
| BST | Local design guidance; no dedicated native model/card | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/BST) |
| DMIN | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DMIN) |
| DMR | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/DMR) |
| TransAct | Existing native core: `transact` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/TransAct) |
| SIM | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/SIM) |
| ETA | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/ETA) |
| SDIM | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/SDIM) |
| TWIN | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/TWIN) |
| MIRRN | No dedicated native method; deduplicate by mechanism before adding | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/LongCTR/MIRRN) |
| APG | Existing native core: `apg` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/APG) |
| PPNet | Existing native core: `pepnet` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/PEPNet) |
| ShareBottom | Existing native core: `shared_bottom` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/multitask/ShareBottom) |
| MMoE | Existing native core: `mmoe` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/multitask/MMoE) |
| PLE | Existing native core: `ple` (variant coverage needs separate inspection) | `torch` | [source](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748/model_zoo/multitask/PLE) |

Detailed mechanism and training conclusions are in the companion model and training reviews. No runtime source was changed by this evaluation.
