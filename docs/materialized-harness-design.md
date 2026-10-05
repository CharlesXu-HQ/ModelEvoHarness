# Materialized research harness design

This design records the October 2026 expansion of ModelEvoHarness. The package remains an Agent-led experiment coordinator; it now carries original, local technical explanations and direct-framework model implementations. It does not mirror upstream repositories.

## Boundaries

- Every catalog family has a substantive local guide. A source URL and pinned revision provide attribution, while the local guide is the working material. Guides cover the mechanism, inputs, measurement conditions, controlled tests, and failure cases.
- Executable model references for all 44 method cards live in separate `models/pytorch` and `models/tensorflow` directories. They import only their respective framework and Python standard library. The manifest gives exact import paths; a ready data contract only means the host declared the inputs, not that a model is appropriate or has improved the task.
- Training changes are first-class experiment directions. Loss, sampling, hard-example mining, optimizer, regularization, and calibration choices follow the same measured-symptom, fixed-control, reject-if protocol as architecture changes. The catalog remains open: an Agent can propose a justified direction beyond its examples.
- A task adapter owns the frozen dataset, labels, splits, candidate execution, GPU selection, and final holdout. The Agent receives validation evidence and cannot change the evaluation protocol.

## Feature requests

An absent feature becomes a **future suggestion** after distinct viable experiments using present fields, or immediately when the host declares an explicit business prerequisite. A terminal request needs a concrete source, decision-time cutoff, validation plan, and either:

1. Multiple completed, distinct feature or architecture experiments with referenced trial IDs and measured evidence that a missing signal now limits progress; or
2. A domain requirement explicitly supplied by the host in the frozen task snapshot, with an ID and provenance. This permits an immediate request without pretending that the Agent inferred business semantics from a column name.

The validator checks these evidence paths. The host still decides whether a human should create the requested field and publishes a new dataset fingerprint afterward.

## Two kinds of experience

Each trial records technical experience (mechanism, control, observed result, uncertainty, next test) separately from business experience (a policy or user-behavior interpretation grounded in a host-supplied observation). Without a measured and semantically defined business observation, the second part is explicitly `not_observable`. Both are scoped to the task, dataset and evaluation protocol; final holdout evidence never enters the Agent loop.

## Verification

Tests cover bundled-guide coverage and wheel inclusion, framework model forward/gradient behavior, training-direction applicability, the two feature-request evidence paths, experience persistence and rejection of unsupported business claims. CouponEvo integration verifies the same constraints at the first consumer without changing its frozen evaluation contract.
