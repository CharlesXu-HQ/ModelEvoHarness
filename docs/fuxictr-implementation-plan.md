# Complete FuxiCTR mechanism coverage implementation plan

**Goal:** Complete the source-based assessment in one delivery, providing independent native PyTorch and TensorFlow mechanisms, composable recipes and Agent guidance for every useful gap in the 57-row FuxiCTR inventory.

**Architecture:** Keep the host backbone, dataset and evaluator. Expose tensor-first local layers, selectors, objectives and representation adapters. Existing matching cores remain canonical; new method IDs identify distinct mechanisms. Every inventory row must resolve to an existing implementation or a new executable component/recipe, with explicit limitations rather than a claim of full production reproduction.

**Tech stack:** Python, PyTorch and TensorFlow only for model modules; stdlib for metadata/contract checks. Native source must run when extracted from the package, with no FuxiCTR, DeepCTR or other model-framework imports.

**Spec:** User approval to merge the previous branch and implement all additions from [the assessment](research/fuxictr-assessment-2026-11-12.md), [structure review](research/fuxictr-model-review-2026-11-12.md), [training review](research/fuxictr-training-review-2026-11-12.md), and [57-row inventory](research/fuxictr-inventory-2026-11-12.md). These documents define the requested scope; this plan resolves routine interfaces without an additional approval round.

## Scope

- Gated crosses, instance masks, conditioned multi-head fusion, intermediate cross/deep bridges, WuKong token interactions, field-pair matrices/group interactions, factorized and quadratic layers, reusable adaptive parameters and prior-conditioned gates.
- Remaining feature mechanisms: field-aware and operation-aware pairs, higher-order FM, holographic and Lorentz interactions, log/exp and Euler interactions, feature-graph/attention aggregation, adaptive operations, convolutional feature generation, and dense/sparse interaction composition.
- Candidate-aware history encoding, refined multi-interest history, user/item matching views, learned/hash/collision/headwise history selection and multi-granularity history refinement.
- Pretrained/ID representation fusion, explicit field-role/shared-vocabulary/dimension contracts, detached ensemble consistency, next-behavior auxiliary supervision, coarse/fine head supervision and embedding/network regularization groups.
- Existing FM/DeepFM/DCN/DIN/APG/PEPNet/multitask implementations are reused. LR/DNN are represented as native baseline recipes. No duplicate wrapper models are needed to fill names.

## Global constraints

- New work is in ModelEvoHarness. Preserve its model/analysis directory separation and dataset-bound experience.
- Native components return representations, raw scores or explicitly documented dicts; the host owns embeddings unless the method specifically changes embedding parameterization, training, device placement and causal evaluation.
- Do not fabricate histories, ID meanings, negative eligibility or discount features. Data prerequisites describe actual fields/tensors.
- Masks are boolean, True=valid. Sanitize padding before arithmetic; define empty histories and all-masked outputs. Learned transformations must receive meaningful gradients.
- Mark source corrections and deliberate simplifications. Do not inherit upstream EDCN shared-gate mistakes, DIEN mask/reduction defects, or MIRRN discarded masks/diagonal-only matrix semantics accidentally.
- Explain why and where a local change belongs in a backbone. Horizontal copies require explicit inputs, sharing and fusion plus an ablation; method inventory is not a closed search menu.
- Local analytic tensor tests are verification, not dataset experiments. Any later model experiment uses the authorized GPU host and full data. This task makes no effectiveness claim without such evidence.

## Work lanes and ownership

### A. Composable crossing and conditioning

Owner writes only new `models/{pytorch,tensorflow}/ctr_*.py` modules assigned to this lane, its tests, and a metadata handoff JSON. Cover GDCN, MaskNet, FinalMLP, EDCN, WuKong, FmFM/FLEN, FinalNet, QNN-alpha, adaptive linear layers and prior gates. Use explicit formulas from the review. Suggested modules: `ctr_composition.py`, `ctr_conditioning.py`.

### B. Remaining field mechanisms

Owner writes only new `models/{pytorch,tensorflow}/field_*.py` modules, tests and handoff JSON. Cover CCPM, FGCNN, FFM, ONN, HOFM, HFM, DLRM, AFN, LorentzFM, InterHAt, DeepIM, FiGNN, DESTINE, SAM, AOANet and EulerNet. Preserve each discriminating computation instead of assigning many names to one generic MLP.

### C. Sequence mechanisms

Owner writes only new `models/{pytorch,tensorflow}/history_*.py` modules, tests and handoff JSON. Cover BST, DMIN, DMR, SIM, ETA, SDIM, TWIN and MIRRN. Separate retrieval/index tensors from downstream learned representations, and state dense-search versus ANN limitations. Do not overwrite the existing DIN/DIEN/TransAct modules.

### D. Root integration, representations and objectives

Root owns `models/{pytorch,tensorflow}/ctr_training.py`, `ctr_embeddings.py`, data registries, feature-role contract validation, knowledge guides, provider integration tests, executable inventory coverage, docs, README counts and release/GitHub work.

Each handoff JSON describes each method's ID, class, relative module, family, constructor/call signatures, requirements, outputs, mechanism, source path, comparison and limitations. This is build-time integration material, not a runtime plugin registry.

## Verification steps

- [x] Merge PR #3 and verify its exact merge commit.
- [x] Write formula/contract tests first in each lane and observe expected missing-component failures.
- [x] Implement independent frameworks and test formulas, gradients, shapes and declared edge cases. Include mapped-weight agreement or common analytic expected results, not just shape smoke tests.
- [x] Add Agent-visible method, source, structure and training registration with capability checks; prove source delivery and standalone execution.
- [x] Map all 57 upstream inventory rows to executable local references, separately recording variant limitations; reject broken paths/IDs/framework coverage.
- [x] Add embedding-role checks and training objectives; test vocabulary/shape mismatch and detached-target/invalid-transition behavior.
- [x] Update English/Chinese README, local original guides and provenance. Replace stale statements about absent implementations only where now implemented.
- [x] Run full tests, catalog coverage and focused independent review; fix concrete defects and rerun affected tests/full suite as needed.
- Release gate: publish the complete change as a reviewable PR with exact verification evidence and remaining experiment limits. The PR is the release-status record.

## Review focus

1. Masked NaNs, empty histories, sequence cutoff and retrieval indices must remain aligned.
2. Complex/FFT/log/exp operations need explicit real-valued semantics and numerical bounds, including odd lengths.
3. Losses must be scalar with detached soft targets where declared, correct valid denominators and no false supervision on missing transitions.
4. Shared embedding modules require shared vocabulary identity and compatible dimensions; freezing must not silently change OOV coverage.
5. A source extractor must deliver all code needed by a component; cross-framework coverage and recipe claims must match actual runnable references.


## Verification record

- Existing PR #3 merged first: `d104f5e53a712843ec6f7371af47b2aa9dcb9c2f`.
- Full suite: **780 passed, 21 skipped, 388 subtests passed**; 35.06 seconds locally.
  Skips are GPU-specific checks; this was CPU analytic verification, not a dataset benchmark.
  Dependency deprecation warnings were emitted by TensorFlow/gast.
- Catalog check: 57 FuxiCTR rows / 43 added methods, no missing source mappings or frameworks;
  all existing source inventories and packaged guide/implementation checks remain complete.
- Independent reviews fixed extreme-value reductions, disabled-penalty NaNs, TensorFlow
  float64 fusion, flaky random-ReLU assertions, objective key metadata and derived-tensor
  discoverability. Mapped framework formulas/gradients and standalone extraction passed.
- All new source attributions were checked against the pinned local upstream snapshot.
- No live Agent or GPU/full-data effectiveness experiment was run for this delivery.
