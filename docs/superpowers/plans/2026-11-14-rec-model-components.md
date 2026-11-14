# Recommendation mechanism additions implementation plan

> For agentic workers: use the existing project workflow with independent implementation owners and a separate final review. User authorized implementation and GitHub publication on 2026-11-14.

**Goal:** Publish the evaluated CAN, ComiRec-SA/DR, CIN/SDIM options and COLD feature-selection/retraining workflow as discoverable Harness capabilities.

**Architecture:** Native tensor components stay in framework-specific model folders. Cost selection/retraining orchestration is framework-neutral analysis code, separate from networks. Existing model defaults remain compatible; cards, contracts and local knowledge expose exact variants and host requirements to the Agent.

**Tech stack:** Python >=3.12, PyTorch, TensorFlow; no third-party DL model wrappers.

**Spec:** docs/research/rec-models-assessment-2026-11-14.md and docs/research/rec-models-gap-review-2026-11-14.md. The user approved implementation and publication of the remaining capabilities after that assessment.

## Global constraints and design decisions

- Implement independently; retain source attribution and distinguish paper/author-code variants.
- Tensor models have PyTorch and TensorFlow counterparts with matching semantics.
- Add components inside a backbone; do not force backbone switching or invent missing business fields.
- Real feature/sequence availability gates recommendations; task/dataset-scoped evidence remains required.
- Empty/padded history has finite zero output; invalid padding may contain NaN and must not leak into output/gradients.
- Preserve default CIN ReLU and unnormalized SDIM; explicit options enable identity and per-hash L2.
- COLD orchestration invokes host physical subset/retrain/evaluate callbacks, retains protected interaction groups, compares measured costs under one profile, and never represents gate scores as proof of feature value.
- Publish to GitHub main and a versioned GitHub release after review and CI. No PyPI upload or CouponEvo submodule update in this scope.

## Review focus

1. CAN batch-specific parameter layouts and multi-order/layer readout cannot silently mix samples.
2. ComiRec masking, routing normalization axes, positional value semantics and stop-gradient require hand-computed checks.
3. Variant defaults must produce unchanged outputs with mapped weights; zero vectors in SDIM L2 need finite gradients.
4. Cost search must reject mismatched evaluation/profile context, invalid metrics and un-retrained candidates; protected interaction-only groups must survive.
5. Agent discovery must serve local code and accurate tensor/input contracts for every new method in both frameworks.

## Tasks

- [x] A. Native CoAction module in models/{pytorch,tensorflow}/coaction.py, tests/test_coaction.py. `CoAction(input_dim, layer_dims, order=1, order_reduction="concat", use_bias=False)`, property parameter_dim/output_dim, `(parameters[B,P], feed[B,T,D], valid[B,T]) -> [B,T,C]`. Each order traverses the same per-sample MLP; tanh except final layer; read out every layer; explicit sum/concat across orders. Test numeric weights, batch separation, multiple orders, padding/empty sequence, parameter/feed gradients and invalid shapes.
- [x] B. Native ComiRecSA and ComiRecDR in models/{pytorch,tensorflow}/multi_interest.py, tests/test_multi_interest.py. Candidate-independent history/mask -> dict interests[B,K,D], interest_mask[B,K], assignments[B,K,T]. SA position affects scores only, values remain original history. DR supports explicit position-specific projections, capsule squash and stopped non-final routing votes. Test formulas, nonuniform assignments, masking, padding/empty, gradients, framework parity.
- [x] C. XDeepFM `cin_activation="relu"|"identity"` and SDIMCollisionPooling `normalize_per_hash=False|True`, tests/test_mechanism_variants.py. Preserve defaults and output keys; add independent formula and zero-gradient boundary checks.
- [x] D. Framework-neutral analysis/cost_aware_search.py with deterministic protected-group subset proposals and host callback execution of physical retrain/evaluation. Require provenance, exact retained groups and matching serving profile. Report all trials, feasible frontier and chosen feasible result; never select by raw saliency alone. Tests/test_cost_aware_search.py exercises actual callback invocation and decision logic without training a demo model.
- [x] E. Agent integration: cards, implementation registries, structure/analysis patterns and local guides; capability checks and integration tests. Update README English/Chinese and source/release notes with exact scope and examples; include the previous crosscheck/fix work.
- [ ] F. Final review, full tests and catalog-check; build/check distribution; GPU tensor validation where target available; merge prerequisite PR #4 and publish reviewed addition via PR and release. Record exact commits, tests and limitations.

Each component owner first writes a failing test, implements, then runs its focused tests. Root runs the final suite and owns all git operations and publication. Independent owners do not revert others' changes.

## Progress

Initial baseline: 5dd85dc (PR #4); previous assessment changes are intentionally included in this release. No new algorithm dataset effectiveness claim is part of this release.

Implementation and local verification: 998 passed, 35 skipped, 392 subtests; all 14 new GPU tests passed on RTX 5090. Independent math/cost reviews passed; wheel build and isolated API/source checks passed. Final GitHub CI, merge and release are tracked by their public PR/release records.
