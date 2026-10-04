# ModelEvoHarness design

## Goal

ModelEvoHarness is a standalone, Apache-2.0 Python package that lets an Agent lead reproducible offline model iteration for recommendation, search, advertising, and marketing tasks. It uses FunRec as a *complete source inventory* for research mechanisms, not as copied code or a claim that every FunRec algorithm is bundled. CouponEvo is the first consumer and owns its own data, training sandbox, causal evaluator, and final test.

## Scope and evidence

Pin the FunRec repository tree used for the inventory. Account for every substantive upstream chapter page, core Python module and production-backend Python module in a machine-checked coverage map. Each mapped family says what it tests, which task stages and data capabilities it needs, how it can be compared, and which upstream examples have executable code versus documentation only. Each upstream model module also has one method card with mechanism, data contracts, controlled comparison, failure signals and implementation boundary. Families and method cards are research lenses, not a closed model whitelist. A task may declare additional families.

An experiment is valid only when it names a mechanism, evidence for trying it now, actual input fields, a control, an expected metric change, a falsifying result, and at least one considered alternative. Missing capability yields a concrete data request or a different hypothesis. Stage and capability screening supplies reasons; it never invents field semantics from dtype or name.

## Runtime boundary

The package owns run-scoped task history, proposal validation, action selection, trial scheduling, reflection capture, and immutable journal writes. A host adapter supplies a stable task snapshot, a baseline evaluation, and evaluation of an opaque candidate artifact. The Agent supplies `propose(context)` and `reflect(observation)` callbacks; the package includes an OpenAI-compatible JSON client and a simple command-line runner with importable task and Agent adapters. The host defines score, metric uncertainty, execution environment, holdout discipline, and candidate format. The harness never opens a host dataset or sends raw rows to a provider.

The journal identity includes the dataset/task fingerprint, frozen objective and constraints, package version, and catalog digest. Resume rejects identity changes. Historical lessons are associated with the exact task and dataset fingerprint. The package exposes trial results and reflection, but a host can keep its final holdout separate.

## First consumer

CouponEvo tracks this repository as a Git submodule, constructs the task profile from its training split, and uses the package's catalog and proposal validation in its existing sandboxed `search` loop. CouponEvo's candidate interface, CUDA requirement, data split, paired bootstrap, and final-test isolation remain its responsibility. The independently runnable loop is tested with an adapter fixture; the CouponEvo path is tested separately with its real evaluator.

## Acceptance criteria

1. Source inventory covers all substantive FunRec chapter pages, all 38 current model files, 54 supporting modules and 50 production-backend modules at one pinned upstream tree, with tests rejecting unmapped paths and missing method cards.
2. The independent package installs and runs a multi-trial Agent loop through a host adapter, records comparable hypotheses/results/reflections, rejects unknown input fields and unavailable capabilities, resumes deterministically, and never needs CouponEvo imports.
3. Public repo has an Apache-2.0 license, English and Chinese README, usage example, tests, and a GitHub URL under CharlesXu-HQ.
4. CouponEvo installs the package from its Git submodule, can opt into it without a copied catalog, and records the package/catalog identity in its task journal.
5. A fixed full-public-data A/B run compares CouponEvo with and without the new harness using the same split, objective, GPU task, Agent model, and trial budget. Report selection and paired confidence intervals; do not claim a lift unless the evidence supports it.

## Risks and boundaries

FunRec's documentation is licensed CC BY-NC-SA 4.0. This repository writes its own descriptions and implementations and links to upstream paths. Some FunRec chapters describe generative or production systems without runnable models; coverage means an explicit applicability/disposition record, not a working implementation of each named model. A coupon RCT lacks item universes, slates, and behavior sequences, so those families need other data for execution proof. Local tests use adapters; real performance evidence must come from a full public dataset and the GPU host.
