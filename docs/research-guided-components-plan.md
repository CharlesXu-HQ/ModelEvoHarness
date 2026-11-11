# Research-guided components implementation plan

## Approved scope

Implement the user's four selected additions in ModelEvoHarness only:

1. Representation diagnostics: effective/stable rank, output variance and branch similarity, callable on intermediate layers.
2. Gradient-based feature/cross screening with redundancy evidence, explicit protection of interaction-only fields, and ablation/retraining requirements.
3. Initial-affinity interaction masks that progressively open more edges with depth inside a backbone.
4. Context-rich historical event encoding and separately normalized user/history/target relations.

The user's fifth and sixth annotations ask which optimization stage they belong to. Explain training-data/training optimization and inference optimization respectively; no new loss, optimizer or inference-policy implementation is requested this turn.

## Design

Extend the existing native PyTorch/TensorFlow component and method-card pattern. Components consume explicit tensors and masks; they do not infer dataset semantics. They are independently extractable, using only stdlib and the named DL framework. They are mechanisms inspired by papers, not claims of full paper reproduction. Existing backbones, task metrics and dataset-bound experience remain the basis of experimentation.

Rejected alternatives: a new experiment runner would duplicate the existing engine; copying full paper systems introduces unrelated data/platform dependencies. Use small components and attach inputs, limitations, controlled comparisons and counterexamples to Agent-readable references.

## Files and ownership

- Diagnostics/selection worker: `models/{pytorch,tensorflow}/research_diagnostics.py`, `tests/test_research_diagnostics.py`.
- Interaction worker: `models/{pytorch,tensorflow}/progressive_interaction.py`, `tests/test_progressive_interaction.py`.
- History worker: `models/{pytorch,tensorflow}/contextual_history.py`, `tests/test_contextual_history.py`.
- Integrator: method/implementation/structure registry entries, shared knowledge, bilingual README, documentation, reference integration tests and final validation.

All `models/` paths are relative to `src/model_evo_harness/`. Workers do not edit shared registries.

## Verification

- [x] Write failing tests for new interfaces before implementation; preserve failure evidence.
- [x] Check analytic cases: orthogonal/duplicated/zero representations; known gradient saliency; protected interaction groups; monotonic nested edge masks; independently normalized relations.
- [x] Cover masked NaN padding, absent histories, malformed inputs, zero variance, ties and finite gradients.
- [x] Check PyTorch/TensorFlow numerical agreement and standalone source extraction.
- [x] Verify the catalog exposes every component, input eligibility and source material to Agent proposals without importing a DL runtime.
- [x] Run the repository suite and inspect failures/skips.
- [ ] Run actual GPU computations when the authorized GPU machine is reachable; currently blocked by `Host is down`.
- [x] Review changes and record exact evidence; do not equate component correctness with dataset improvement.

## Research boundaries

- Rank diagnostics are evidence, not an objective to maximize. Report matrix axes and mask/centering conventions.
- Gradient saliency is not causal importance. Selection uses training-only information; validation is reserved for controlled comparisons and the final holdout stays sealed.
- Preserve explicitly protected fields and all members of protected interactions; require ablation and retraining before removing data inputs.
- Progressive masks use a declared initial-affinity ranking and nondecreasing budgets. Record the interpretation where the source paper is ambiguous.
- Historical event context must be aligned and available before the prediction cutoff. This includes no RQ-VAE/SIF quantizer, full CCFormer hierarchy or online serving system.
- No automatic human feature-gap conclusion follows from one weak model or one saliency score; retain the expert-obvious missing-input exception.
