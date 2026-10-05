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

## Agent context and proposal

```python
class Agent:
    def propose(self, context: dict) -> dict: ...
    def reflect(self, observation: dict) -> dict: ...
```

`context` contains the task snapshot, catalog, family/method/decision/training applicability reports, relevant local `knowledge` guides (including all shared training/exploration guides), ready `model_api` signatures and implementation contracts, baseline, completed trials, best ID and remaining steps. The catalog includes 22 research families, 44 method cards, structural patterns, training patterns and a model implementation manifest. `method_applicability.frameworks` reports direct PyTorch/TensorFlow code for each card. Set `snapshot.framework` to expose only that framework's APIs; omitting it exposes both. The generic `two_tower` example is in the implementation manifest but is not one of the 44 method cards. A `ready` method means its inputs are declared; it does not prove that the method helps.

A proposal has this shape:

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

After a trial, `reflect()` returns separate records:

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

The journal is stored in `output/journal.json` through an atomic replace. Failed `evaluate()` calls become failed trials with a bounded error type. Do not include API keys or raw user rows in snapshots, proposals, metrics or reflections. The final holdout is a separate host action after selecting a candidate.


## Bounded source retrieval

The built-in API provider understands a preparatory JSON action:

```json
{"action":"read_reference","framework":"pytorch","method_ids":["fm","deepfm"],"include_training":true}
```

This loads complete bundled modules, including shared helper classes; no filesystem path, network URL or import execution is accepted. A request selects up to four model IDs. The two-round, 100,000-character bound applies per proposal, and repeated modules are deduplicated. Framework selection follows the task when declared. The next call receives `reference_material` and `reference_contracts`; source contents never become system instructions. A declared bundled `method_id` automatically gets its module before a final experiment if it was not already read. The final proposal stores `reference_reads` (path to host-computed SHA-256). Custom Agents can call `read_references(catalog, request)` or reuse `propose_with_references(complete, context, catalog=..., framework=...)`; `run_search` itself still accepts only final experiment/stop/request-data decisions. Hosts that retry final proposal validation must pass the same host-owned `read_state={}` across those retries, preserving the read budget and source ledger; never populate it from model output.

Normal reflection uses `iteration_effort`, default `high`. The host can set `evaluation.review_required = True` for anomaly/leakage/cost-tradeoff review, which selects `review_effort`, default `max`. The flag is a host assessment, not a replacement for uncertainty or guardrail checks.
