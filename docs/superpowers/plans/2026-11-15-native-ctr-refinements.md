# Native CTR Refinements Implementation Plan

> **For agentic workers:** Use supervised independent work on the owned modules; root integrates metadata, reviews and verifies the combined result.

**Goal:** Implement the approved mechanism and contract additions in both native frameworks.

**Architecture:** Existing classes retain defaults. Focused tensor components compose within host backbones; local source manifests deliver the actual implementation to the Agent.

**Tech Stack:** Python >=3.12, native PyTorch and TensorFlow, pytest; no model-wrapper dependency.

**Spec:** [Design](../specs/2026-11-15-native-ctr-refinements-design.md).

## Global constraints

- Only ModelEvoHarness changes; original implementations, no DeepCTR dependency.
- Preserve existing defaults and framework-neutral package imports.
- Keep model structures, training utilities and analysis tools in their existing categories.
- No real-data or Agent-effectiveness claim from component verification.

## Review focus

- Masked NaNs must not reach arithmetic or corrupt backward gradients.
- Empty histories and length/mask disagreement need explicit results/errors.
- Shared field factors must affect both first- and second-order terms.
- Dice train/eval state and loss reductions must be documented and testable.
- Extracted sources must include all required helpers without third-party model dependencies.

## Task 1: Cross and field variants

Files: paired `interaction_variants.py`, existing `interactions.py`, `field_products.py`,
`ctr_composition.py`; `tests/test_interaction_variants.py`.

- [x] Add failing formula tests for CrossNetMix and IFM/DIFM, plus parameterization tests for existing field operators.
- [x] Implement standalone blocks and opt-in variants, preserve default output contracts.
- [x] Verify analytic values, active parameter gradients and cross-framework mappings.

## Task 2: Sequence structures and auxiliary interface

Files: paired `sequence_attention.py`, DIN in `architectures.py`, DIEN/DSIN in
`sequence.py`, BST in `history_matching.py`; `tests/test_sequence_refinements.py`.

- [x] Reproduce mask/NaN issues before fixes; test pooling and evolution variants.
- [x] Implement Dice/local pooling, sanitize boundaries, expose DIEN state/auxiliary operations, add optional sequence variants.
- [x] Verify padding/empty invariance, training/evaluation state and auxiliary gradients.

## Task 3: Input and training contracts

Files: `embeddings.py`, paired `sequence_inputs.py` and `ctr_training.py` where needed;
`tests/test_sequence_inputs.py`, `tests/test_embedding_roles.py`.

- [x] Add failing metadata and runtime tests for sequence lengths, alignment and weighted pooling.
- [x] Implement optional role declarations and native pooling with precise normalization.
- [x] Verify hand-computed objective/regularization values without changing existing scale.

## Task 4: Harness integration and documentation

Files: existing JSON registries, packaged knowledge, `catalog.py` only if source delivery
requires companions, English/Chinese README, native refinement guide and integration tests.

- [x] Register methods and exact API contracts; extend existing pattern entries for variants.
- [x] Correct DIFM description and obsolete implementation limitations; provide source attribution.
- [x] Verify Agent receives local source code and input/ablation requirements in both frameworks.

## Task 5: Combined verification and publication

- [x] Independently review the combined diff and resolve actionable findings.
- [x] Run full pytest and catalog-check; verify package contents.
- [x] Run targeted PyTorch/TensorFlow GPU checks on RTX 5090; document exact scope.
- [ ] Commit, publish via gh, attach the PR and confirm CI state (publication gate tracked in the PR).
