# ModelEvoHarness implementation plan

**Goal:** Publish a standalone cross-scenario Agent model-iteration harness, then validate its CouponEvo integration on a full public dataset.

**Architecture:** The independent package owns a pinned FunRec coverage catalog and a host-adapter experiment loop. CouponEvo consumes the published package through an opt-in adapter and retains training, GPU sandbox, data split and causal evaluation.

**Tech stack:** Python 3.12 standard library for the package; existing PyTorch and EconML stack only in CouponEvo.

**Spec:** [design.md](design.md)

## Tasks and checks

- [x] Capture every substantive FunRec chapter page, core Python module and production-backend Python module at one tree hash, map each exactly once to a research family, and add one method card per upstream model module. Check: `coverage_report` returns no unmapped, unknown, duplicate or misassigned paths.
- [x] Build the independent Agent loop and provider transport. Check: fake adapter tests exercise baseline → hypotheses → candidate evaluations → reflections → best selection, rejection before training, and resume after interrupted reflection.
- [ ] Publish package metadata, two READMEs, adapter contract, Apache license and CI. Check: clean install, unit tests, CLI catalog check and a GitHub clone from the new public repository.
- [ ] Integrate the package into CouponEvo without copying its catalog. Check: existing test suite and an opt-in search using a real full dataset.
- [ ] Run a fixed-budget A/B on the RTX 5090 host with the same public data, split, objective, Agent configuration and candidate budget. Check: store journals, compare selected candidates on a paired final evaluation, and report uncertainty and limitations.

## Review focus

- Missing sequence/slate/item inputs must produce explicit `needs_data` or `other_stage` rather than fabricated feasibility.
- A known family missing prerequisites must be blocked before training; a novel direction remains possible if its raw inputs exist.
- Agent failure after an expensive evaluation must not repeat the same trial on resume.
- External package/catalog changes must alter task identity, preventing old journal or experience reuse.
- No final holdout score, API key or raw user row may enter the Agent context or published repository.
