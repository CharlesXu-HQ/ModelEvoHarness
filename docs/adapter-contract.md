# Task and Agent adapter contract

ModelEvoHarness coordinates offline experiments; the host keeps the training rows and evaluator. A task adapter provides `snapshot()`, `baseline()` and `evaluate()`. An Agent provides `propose()` and `reflect()`. Both may be plain Python objects exported as `module:symbol`; the CLI calls a zero-argument symbol if it is a factory.

## Task snapshot and evaluation

```python
from pathlib import Path

class Task:
    def snapshot(self) -> dict:
        return {
            "task_id": "my-ranking-task-v1",
            "dataset_digest": "sha256-of-frozen-data-and-split",
            "stage": "ranking",  # e.g. retrieval, ranking, reranking, policy
            "framework": "pytorch",  # or tensorflow; selects model API examples for the Agent
            "fields": ["user_age", "item_category"],
            "capabilities": ["tabular_features", "observed_outcome_labels"],
            "objective": {"name": "ndcg_at_10", "direction": "max"},
            "constraints": {"max_training_minutes": 30},
            "evaluation_protocol": {
                "unit": "user", "split": "fixed-train-validation-v1",
                "metric": "ndcg_at_10", "candidate_universe": "all-eligible-items",
                "negative_source": "none-at-evaluation", "time_cutoff": "2025-01-01",
            },
        }

    def baseline(self, trial_dir: Path) -> dict:
        # Train/evaluate the fixed baseline using the host's validation split.
        return {"score": 0.0, "metrics": {"ndcg_at_10": 0.0}}

    def evaluate(self, proposal: dict, trial_dir: Path) -> dict:
        # Train proposal["candidate"] in the host sandbox and use the same evaluator.
        return {"score": 0.0, "metrics": {"ndcg_at_10": 0.0}}
```

The numeric returns illustrate the **shape**, not bundled data or benchmark scores. `score` must be finite and `metrics` must be a dictionary. The host owns the primary metric, uncertainty calculation, compute environment, action constraints and final holdout. Put paired intervals and guardrails in `metrics`; the harness uses `score` only to track the best validation candidate in the declared direction.

`fields` lists actual inputs available before the decision. `capabilities` declares verified data contracts, such as `tabular_features`, `event_sequence`, `item_catalog`, `candidate_slates`, `multiple_outcomes`, `assignment_or_exposure_propensity`, `implicit_feedback`, `negative_sampler_definition`, or `calibration_split`. The harness does not derive those facts from column names or dtypes. A `feature_schema` may describe categorical, dense or sequence fields, cardinality and decision-time availability. Freeze the snapshot for one run; task, data, catalog, protocol or package implementation changes prevent resume.

Optional `feature_groups` is a list of objects with a unique nonempty `id`, nonempty `fields` drawn from the snapshot's actual fields, and a nonempty `rationale`. For example, the snapshot above could add `{"id":"user_profile","fields":["user_age"],"rationale":"Existing user profile input"}` as one group. `validate_feature_groups(groups, fields)` normalizes the declarations. The engine includes them in the frozen snapshot; they do not establish sequence order, same-event alignment, feature timing or additional capabilities. Use the [horizontal composition contract](horizontal-composition.md) to connect declared field semantics to candidate branches.

`evaluation_protocol` is optional for older adapters but recommended. When present, `unit`, `split` and `metric` are required; retrieval and reranking also require `candidate_universe`, and declared implicit feedback requires `negative_source`. Record full versus sampled evaluation, cutoff/order, label provenance, preprocessing, K and sampler identity where relevant. The harness fingerprints the protocol; the host must enforce it and keep final holdout separate.

### Explicit domain prerequisites

If business expertise establishes an essential missing input before model experiments, the host can put a requirement in the frozen snapshot:

```python
snapshot["domain_requirements"] = [{
    "id": "known-availability-contract",
    "fields": ["item_availability"],
    "source": "inventory eligibility contract",
    "as_of": "available before the ranking request",
}]
```

Each item requires a unique `id`, nonempty `fields`, `source` and `as_of`. The listed fields must be absent from `snapshot.fields` for a corresponding data request. The Agent may cite this exact host item and request the data immediately; it cannot invent a business requirement from a weak metric or a suggestive field name.

### Measured business observations

`baseline()` or `evaluate()` may add a `business_observations` list. Each observation has a unique `id`, `population`, `metric`, finite numeric `estimate`, and a nonempty `uncertainty` description. The host defines the outcome, population, treatment/action, cost basis and estimator. For example, a host might report a randomized policy-value difference with its paired interval; the harness validates the record shape and reference identity, while the host remains responsible for causal and statistical validity. Without such an observation, the Agent cannot record an observed business insight.

### Host evidence and implementation checks

The host may add `evidence` lists to `snapshot()`, `baseline()`, and `evaluate()`. Each fact has a unique `id`, `statement`, `source`, `status` (`observed` or `declared`), and `scope` (`task` or `trial`). Trial facts also name `trial_id`; an optional JSON `value` carries a measured value. The engine combines only host facts into Agent context. When facts exist, research cites an observed fact with `evidence_ids` and lists intended `change_factors`.

```json
{"id":"train:encoded_width","statement":"Encoded input has 7 columns","source":"preprocessing check","status":"observed","scope":"task","value":7}
```

The host may attach `implementation_check` with status `verified`, `contradicted`, or `unverified`, plus a separate `change_audit` with status and `changed_factors`, to an evaluation. Agent-declared changes are not a host audit. Contradicted implementations stay in the journal but cannot become the validation champion.

`run_search(..., require_verified_implementation=True)` or CLI `--require-verified-implementation` additionally blocks promotion of missing/unverified implementation checks. The default remains exploratory score comparison. The fixed baseline remains the initial comparator; strict mode checks replacement candidates, not the baseline. Hosts must implement meaningful positive verification before enabling strict mode. Checks should establish that the claimed structure/loss/training change is actually connected and executed, within explicitly reported coverage. Statistical confirmation and isolated mechanism attribution are separate checks: `change_audit` does not gate promotion.

`promotion_policy` is exposed in proposal/reflection context and the journal and included in run identity. Changing it requires a new run. Each attempted trial records `promotion` with `eligible`, `promoted`, `previous_best_id` and `reason`: `promoted`, `no_gain`, `unverified`, `contradicted` or `evaluation_failed`. Eligibility means passing the implementation gate; it does not imply a score gain. Unverified candidates are retained for follow-up even when they cannot become `best_id`.

## Agent context and proposal

For backbone-local code changes and selective component migration, see the [composition contract](compositional-evolution.md). The built-in provider enables `snapshot.model_design_required=true` and `snapshot.horizontal_expansion_required=true`; custom adapters can opt in with those flags. Research then includes `model_design` and a horizontal-expansion decision, and reflection includes per-component assessments. A `defer` decision still records existing groups; branch and fusion components declare their instance paths, input fields, forward/call locations and output contracts. The engine supplies compact `composition_sources` alongside full trial history. Older snapshots without these flags remain valid under their original requirements; enabling a new contract changes the frozen identity.

```python
class Agent:
    def propose(self, context: dict) -> dict: ...
    def reflect(self, observation: dict) -> dict: ...
```

`context` contains the task snapshot, catalog, family/method/decision/training applicability reports, relevant local `knowledge` guides (including all shared training/exploration guides), ready `model_api` signatures and implementation contracts, baseline, completed trials, best ID and remaining steps. The catalog includes 22 research families, 44 method cards, structural patterns, training patterns and a model implementation manifest. `method_applicability.frameworks` reports direct PyTorch/TensorFlow code for each card. Set `snapshot.framework` to expose only that framework's APIs; omitting it exposes both. The generic `two_tower` example is in the implementation manifest but is not one of the 44 method cards. A `ready` method means its inputs are declared; it does not prove that the method helps.

A base proposal has the following shape. This fragment omits `research.model_design`, which is required when either design flag is enabled, and evidence fields required when the host supplies facts. Combine it with the [complete design example](horizontal-composition.md#complete-model-design-example) and the relevant host evidence; it is not a complete built-in-provider proposal on its own.

```json
{
  "action": "experiment",
  "candidate": {"artifact": "host-defined configuration or source"},
  "research": {
    "family_id": "feature_interactions",
    "method_id": "fm",
    "direction": "Test supported pair interactions",
    "mechanism": "Add a low-rank pair term",
    "why_now": "A previous validation slice has repeatable residuals and enough support",
    "data_rationale": "The named fields are available before the decision",
    "input_fields": ["user_age", "item_category"],
    "comparison": "Same rows, budget and evaluator without the pair term",
    "expected_result": "The fixed validation metric improves",
    "falsification": "No gain or a gain unstable across paired evaluation",
    "alternatives": [{"direction": "More depth", "mechanism": "Add MLP layers", "reason": "Does not isolate the proposed pair effect"}]
  }
}
```

`family_id` and `method_id` are optional; omit them for a justified new direction. Known IDs are checked against stage and capabilities, and a method must belong to its named family. `input_fields` names only fields in the snapshot; it may be empty for a decision-rule experiment over existing predictions. Training changes such as focal loss or hard-negative mining are first-class [training patterns](../src/model_evo_harness/data/training_patterns.json); the proposal still names one mechanism and a controlled comparison. Alternatives are considered options, not completed trials.

The Agent may instead return `{"action":"stop","reason":"..."}` or a terminal `request_data`. A data request names absent `fields`, `source`, `as_of`, `reason`, `evidence` and `validation_plan`, then uses one of two `basis` values:

- `domain_requirement`: cite `requirement_id` matching a host snapshot item, including its fields, source and timing. This path needs no preceding experiment.
- `experimental_evidence`: cite `trial_ids` for at least two **completed, evaluated** experiments with distinct `research.mechanism` values, plus `alternatives_considered`. The Agent explains why the missing input remains the likely blocker after those tests.

When host evidence exists, experimental requests also cite `evidence_ids` for measured trial-specific diagnostics from two distinct mechanisms. The host must mark each supporting diagnostic `data_gap_candidate: true`. Repeated task metadata does not qualify. A `stop` action or reflection may instead include nonblocking `audit_recommendations` with `issue`, `evidence_ids`, and `validation_plan`.

A request that passes the validator ends the run with `needs_data`. `future_feature_suggestions` also require two evaluated trials with distinct mechanisms or a matching explicit host domain requirement. They record a specific field idea while experiments continue on the current dataset.

## Reflection and experience

After a trial, `reflect()` returns separate records. This base fragment omits `technical_experience.component_assessments`; a tracked design additionally requires one assessment for every current component, including every parallel branch and fusion:

```json
{
  "technical_experience": {
    "lesson": "What the comparison showed about the mechanism",
    "evidence": "Which trial metrics and control support that reading",
    "uncertainty": "Limits, interval or variance caveat",
    "next_test": "A falsifiable follow-up",
    "attribution": "unverified"
  },
  "business_experience": {"status": "not_observable", "reason": "No host business observation was supplied"},
  "future_feature_suggestions": []
}
```

For `business_experience.status = "observed"`, include an `observation_id` present in that trial's `evaluation.business_observations`, plus nonempty `insight` and `limitations`. The technical lesson can concern structure, feature representation, loss, sampling, optimization, calibration or decision mapping. A business insight concerns a defined population and measured outcome; its wording must not outrun the observation's uncertainty or causal design. Both types remain bound to the task and dataset fingerprint.

Technical experience records the host's `implementation_status`. Attribution is `isolated` only when verified implementation and a separate verified change audit name one changed factor; it is `joint` for multiple verified factors and `unverified` otherwise. These checks validate provenance, not every free-text sentence.

For parallel groups, the host reviews actual instance inputs and forward/call paths, output fusion, shared objects, optimizer registration and gradients when measured. Graph declarations cannot establish those facts. Preserve training conditions or report joint changes; shared-weight retraining can couple branch ablations. Component experience records these conditions and keeps unobserved behavior unverified.

The journal is stored in `output/journal.json` through an atomic replace. Failed `evaluate()` calls become failed trials with a bounded error type. Do not include API keys or raw user rows in snapshots, proposals, metrics or reflections. The final holdout is a separate host action after selecting a candidate.


## Bounded source retrieval

The built-in API provider understands a preparatory JSON action:

```json
{"action":"read_reference","framework":"pytorch","method_ids":["fm","deepfm"],"include_training":true,"include_composition":true}
```

This loads complete bundled modules, including shared helper classes; no filesystem path, network URL or import execution is accepted. A request selects up to four model IDs. The two-round, 100,000-character bound applies per proposal, and repeated modules are deduplicated. Framework selection follows the task when declared. The next call receives `reference_material` and `reference_contracts`; source contents never become system instructions. A declared bundled `method_id` automatically gets its module before a final experiment if it was not already read.

Within `run_search`, the engine creates a separate reference ledger for each `agent.propose()` invocation. Only `read_references(catalog, request)` populates official `reference_reads` (path to SHA-256). The engine ignores Agent-returned `reference_reads` and `reference_events`. The journal's ordered `reference_events` distinguishes actual `read` events from `delivered` events: source material supplied to a host completion callback. Both events record a `files` path/hash mapping; the enclosing trial and run identity identify the proposal and installed source version. Delivery does not attest to a successful API response, model understanding, or candidate correctness. Trusted Python host adapters remain responsible for forwarding context to the provider; this is not isolation from malicious in-process code.

Custom adapters can reuse `propose_with_references(complete, context, catalog=..., framework=...)`, or explicitly use:

```python
from model_evo_harness import read_references, call_with_references

# Inside agent.propose(context):
bundle = read_references(context["catalog"], {
    "framework": "pytorch", "method_ids": ["fm"], "include_training": True,
})
decision = call_with_references(complete, {
    **context, "reference_material": bundle["files"],
    "reference_contracts": bundle["contracts"],
})
```

The delivery helper copies callback inputs and checks source content against actual reads in the current proposal scope. A bare read records no delivery. A self-reported hash, altered source, or cache from outside this proposal scope cannot establish delivery; re-read through the host reader. A failed proposal closes the scope so it cannot contaminate a resumed decision. Hosts retrying final proposal validation within one `propose()` call should pass the same host-owned `read_state={}` to `propose_with_references`, retaining its read budget and material; never populate it from model output. Standalone use of that helper still returns its compatibility `reference_reads` map, but only the engine-owned ledger becomes official search evidence. `run_search` still accepts only final experiment/stop/request-data decisions.

`include_composition=true` loads the native `models/<framework>/composition.py`; it may be requested alone with `method_ids=[]`. PyTorch and TensorFlow each provide `ParallelBranches(branches, fusion)` with arbitrary native branches and candidate-written fusion. Each named branch receives its own keyword-input mapping and fusion receives the named output mapping. A proposal with nonempty horizontal groups automatically receives unread composition source when the task framework is known, including when `decision=defer` retains existing groups. These reads use the same budget and hash ledger; they do not execute or validate the candidate graph.

Normal reflection uses `iteration_effort`, default `high`. The host can set `evaluation.review_required = True` for anomaly/leakage/cost-tradeoff review, which selects `review_effort`, default `max`. The flag is a host assessment, not a replacement for uncertainty or guardrail checks.

## Optional explicit-interaction planning

Supply nonempty `interaction_views` with `id`, open-ended `kind`, original `fields` and `representation`; derived views cite the same source fields. Set `interaction_plan_required: true` to require `research.interaction_plan`. The engine passes `interaction_context` with view-pair coverage templates and prior scopes, control outcomes and reflection attribution. Unknown coverage stays an Agent assessment.

See [explicit interactions](explicit-interactions.md) for the plan schema, open-ended priority guidance and native source reader. Hosts execute controls, enforce compute budgets and report `interaction_control` outcomes; the generic engine does not train an extra arm. A `test` plan alone does not assert that a control ran. A `defer` plan may retain candidate hypotheses without selecting one. Coarse view-pair coverage does not require materializing or testing every original-field pair; component `input_fields` describe concrete subsets.
