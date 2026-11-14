# Evidence-driven Training References Implementation Plan

> **For agentic workers:** Use the assigned independent component boundaries; the primary agent integrates and reviews the final branch.

**Goal:** Fill three verified reference gaps with local components and actionable experimental guidance.

**Architecture:** Keep native tensor mechanisms under `models/{pytorch,tensorflow}` and prediction diagnostics under `analysis`. Register them in existing cards, patterns and manifests so Agent context and source retrieval expose their actual contracts.

**Tech Stack:** Python >=3.12; optional native PyTorch/TensorFlow; stdlib diagnostics.

**Spec:** [Design](../specs/2026-11-14-training-reference-gaps-design.md).

## Global Constraints

- Independent implementations; no upstream wrapper dependencies or copied trainers.
- Keep the host's data, objective, evaluation universe and final holdout fixed.
- No remote GPU access or dataset training while the target is off.
- Changes are confined to ModelEvoHarness; use a conventional `feat/` branch.

## Review Focus

- A fixed mask must survive optimizer updates and serialization; sparse cells are not a physically smaller embedding table.
- A zero-support negative pool must fail; returned probabilities must reflect positive exclusion, mixture and replacement semantics.
- Metric comparison requires identical ordered sample IDs; tied AUC scores, a single class and zero-weight observations have explicit semantics.
- Components must be discoverable, capability-gated and fully readable by the Agent, without analysis IDs becoming model trials.

## Tasks

- [x] Research pinned upstream code and document gaps versus existing implementations.
- [x] Add failing numerical tests, then implement `PrunableEmbedding` in both frameworks and verify masks, gradients, padding and fixed-mask retraining.
- [x] Add failing sampling tests, then implement `ConditionalNegativeSampler` in both frameworks and verify distributions, exclusions and reproducibility.
- [x] Add failing paired-metric tests, then implement stdlib `BinaryPredictionDiagnostics` and verify AUC/LogLoss decomposition and invalid provenance.
- [x] Register implementations/cards/patterns; extend existing local guides and add context/source-delivery tests.
- [x] Update English/Chinese README, component documentation and source attribution.
- [x] Run targeted tests, full regression, catalog check and package/source extraction checks; independently review the changed code.
- [ ] Commit and publish a reviewable PR with local evidence and explicit GPU/experiment limits.
