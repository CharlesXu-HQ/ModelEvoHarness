# Task and Agent adapter contract

ModelEvoHarness coordinates offline experiments and does not read your training rows. A task adapter exposes three methods. An Agent exposes two. Both can be plain Python objects available as `module:symbol`; the CLI calls a zero-argument symbol if it is a factory.

```python
from pathlib import Path

class Task:
    def snapshot(self) -> dict:
        return {
            "task_id": "my-ranking-task-v1",
            "dataset_digest": "sha256-of-frozen-dataset-and-split",
            "stage": "ranking",  # e.g. retrieval, ranking, reranking, generation, policy
            "fields": ["user_age", "item_category"],  # actual decision-time inputs
            "capabilities": ["tabular_features", "observed_outcome_labels", "item_catalog"],
            "objective": {"name": "ndcg_at_10", "direction": "max"},
            "constraints": {"max_training_minutes": 30},
        }

    def baseline(self, trial_dir: Path) -> dict:
        # Train/evaluate the fixed baseline on your validation split.
        return {"score": 0.0, "metrics": {"ndcg_at_10": 0.0}}

    def evaluate(self, proposal: dict, trial_dir: Path) -> dict:
        # Interpret proposal["candidate"], train in your sandbox, evaluate on
        # the same validation definition, and return the same score shape.
        return {"score": 0.0, "metrics": {"ndcg_at_10": 0.0}}
```

The numeric examples above describe the **return shape**, not a bundled dataset or benchmark result. Use the real evaluator and full eligible dataset for actual experiments. `score` must be finite. The task owns the target metric, direction, constraints, uncertainty estimates, execution environment and final holdout. Put any paired interval or guardrail outcomes in `metrics`; the harness uses `score` only to track the current best by the declared direction.

`fields` lists raw inputs the Agent may name. `capabilities` is an open set of data contracts established by the host, such as `tabular_features`, `observed_outcome_labels`, `event_sequence`, `interaction_graph`, `item_catalog`, `candidate_slates`, `multiple_outcomes`, `scenario_context`, `assignment_or_exposure_propensity`, or `generative_targets`. Do not infer a capability from a column dtype or invented business meaning. Freeze `snapshot()` for a run; changing the snapshot, dataset digest, package version or catalog digest prevents resume.

```python
class Agent:
    def propose(self, context: dict) -> dict: ...
    def reflect(self, observation: dict) -> dict: ...
```

`context` includes the task snapshot, catalog, per-family applicability, per-method applicability, baseline, completed trials, best ID and remaining step count. `observation` includes the new trial and its evaluation status so the Agent can compare the outcome with its prediction or react to a failed candidate evaluation. The provider included in this package implements the same two methods; a host can supply its own Agent.

An experiment proposal uses this JSON shape:

```json
{
  "action": "experiment",
  "candidate": {"artifact": "host-defined candidate configuration or source"},
  "research": {
    "family_id": "feature_interactions",
    "method_id": "fm",
    "direction": "Test explicit pair interactions",
    "mechanism": "Add a low-rank interaction term",
    "why_now": "Previous validation shows an underfit subgroup with sufficient support",
    "data_rationale": "The declared raw fields are available before the decision",
    "input_fields": ["user_age", "item_category"],
    "comparison": "Same data, training budget and evaluator without the interaction term",
    "expected_result": "Validation NDCG@10 increases under the fixed split",
    "falsification": "No improvement or unstable gain across paired evaluation",
    "alternatives": [{"direction": "More depth", "mechanism": "Add MLP layers", "reason": "Does not isolate the proposed interaction"}]
  }
}
```

`family_id` and `method_id` are optional. Known family IDs and method IDs are screened against task stage and capabilities; omit them for a new direction outside the catalog. If both are present, the method must belong to the family. `input_fields` may be empty when the experiment changes only a policy over existing predictions or needs no covariates. Alternatives are considered options, not completed trials. The Agent may return `{"action":"request_data","request":{...}}` to record a specific missing-input request, or `{"action":"stop","reason":"..."}` when it has no justified next experiment or the budget is exhausted.

The harness stores journal data in `output/journal.json`, writing through an atomic replace. A failed `evaluate()` call becomes a failed trial with a bounded error type so resume does not silently repeat the same candidate. Do not include API credentials or raw user rows in `snapshot()`, proposals, metrics or reflections; these are intended for review and can be committed selectively. Reserve final holdout evaluation for a separate host step after selecting a candidate.
