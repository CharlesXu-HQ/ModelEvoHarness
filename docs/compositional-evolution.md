# Compositional model evolution

## Research intent

Prefer informative changes inside a suitable backbone before replacing it. This is a research priority, not an obligatory trial count. A clearly unsuitable estimator or architecture can be replaced immediately. The Agent must explain the evidence, data fit, local alternative and comparison that makes the switch worth its budget.

Three different decisions are recorded:

- **Estimator**: the learning/estimation formulation, such as a causal meta-learner or a supervised ranking estimator.
- **Backbone**: the representation/predictor family used inside that formulation.
- **Components**: actual local code for representations, interactions, heads, losses, sampling, optimization, calibration or other mechanisms. This list of examples is open.

T/S/DR learner names do not, by themselves, specify the network inside them. Similarly, a network with a familiar name may contain useful original adaptations. The Agent may read local reference modules and write/adapt their internals inside a candidate, subject to the host's execution contract. It is not limited to instantiating whole catalog models.

## Proposal contract

The built-in `OpenAICompatibleAgent` advertises `requires_model_design=True`; the engine activates `task.model_design_required` for its searches. Custom adapters can declare that snapshot field to use the same contract. Older adapters without the flag may omit `model_design`; any design they do supply is validated. Enabling the contract changes the frozen run identity, so old runs are not silently resumed under a new protocol.

Add `research.model_design` alongside the existing falsifiable research fields:

```json
{
  "estimator": "t_learner",
  "backbone": "tabular_mlp",
  "change_scope": "initialize",
  "parent_trial_id": null,
  "rationale": "Begin a tracked recipe after inspecting the untracked seed",
  "data_fit": "The host declares existing pre-decision tabular fields; no sequence is assumed",
  "comparison_plan": "Same split, objective and compute budget as the seed",
  "components": [
    {
      "id": "pair_encoder",
      "mechanism": "An explicit pair term contributes to each arm's prediction",
      "code_sections": ["ArmNetwork.encode", "ArmNetwork.forward"],
      "input_fields": ["user_age", "item_category"],
      "required_capabilities": ["tabular_features"],
      "reference_method_id": "fm"
    }
  ],
  "inheritance": []
}
```

This is a schema example, not an evaluated recipe. Actual `input_fields`, capabilities and code locations must come from the task and candidate. Training-only components may have empty input/capability lists. `reference_method_id` is optional for original code; when present it must name a bundled implementation for the host framework, and triggers bounded source reading even without a top-level `method_id`. Reading an entire reference module does not make its full-model prerequisites optional.

`initialize` registers a recipe only when no prior tracked design exists. `local` keeps estimator/backbone IDs; `switch` changes at least one. These IDs are open strings; host code review must detect a misleading rename or an undeclared structural change.

Each local/switch proposal names its `parent_trial_id` and accounts for **every parent component**. Extra donors may be cited from the same host-supplied history. An inheritance item is:

```json
{
  "source_trial_id": "trial_003",
  "component_id": "pair_encoder",
  "decision": "adapt",
  "reason": "Reuse the tested encoding idea while changing its attachment to a shared encoder",
  "compatibility": "Named fields and timing remain available; target head dimensions and estimator fitting boundaries differ",
  "validation_plan": "Compare plain target backbone and adapted target at matched budget, then ablate this block",
  "target_component_id": "shared_pair_encoder"
}
```

The four decisions mean:

| Decision | Required interpretation |
| --- | --- |
| retain | Same declared mechanism, code locations, input fields, required capabilities and reference; the source must be eligible for reuse. Actual execution still needs checking. |
| adapt | Reuse an idea with explicitly changed interfaces or training semantics; provide compatible target declarations and a test. |
| drop | Exclude the source component with a reason; no target component mapping. |
| retest | Revisit an uncertain, failed or invalid idea as a new hypothesis; no inherited claim of success. |

Invalid, unevaluated, blocked or harmful source components, and components without a recorded assessment, can only be dropped or retested. An assessed but inconclusive component may be carried as an explicit hypothesis with its uncertainty; this does not establish a gain. Plans referencing invented source/component IDs, unavailable fields/capabilities, missing parent dispositions, or a switch disguised as a local change fail validation. This checks declarations and lineage; it does not statically prove that the code followed the plan. The host must compare donor/candidate source, actual prediction paths and runtime contracts.

## Reflection and evidence

A tracked recipe requires `technical_experience.component_assessments`, one entry per current component:

```json
{
  "component_id": "shared_pair_encoder",
  "outcome": "inconclusive",
  "evidence": "The combined candidate changed several factors; the component effect is unresolved",
  "compatibility_limits": "Tested only on this dataset/split and the declared tabular input",
  "next_test": "Remove the component while holding the target backbone and training budget fixed",
  "attribution": "unverified"
}
```

Outcomes are `promising`, `inconclusive`, `harmful` or `invalid`. Promising is exploratory. `unverified` attribution is always allowed. `joint` requires a verified multi-factor host change audit. `isolated` additionally requires the sole audited factor to equal this component ID. Merely winning as a combined candidate cannot upgrade all components to individually effective. Failed or contradicted implementations cannot supply a promising/harmful mechanism conclusion.

For custom host calls to `validate_reflection`, include the current research and `trial_status` in the evaluation view. The native engine supplies these; the host retains ownership of independent evidence.

## Practical comparison and limits

For a switch, preserve the previous recipe and compare a plain new backbone with the selectively composed new backbone under the same objective, split and compute budget. If the experiment budget cannot isolate every factor, choose the most informative comparison and retain uncertainty. No universal component-transfer benefit is assumed. Verify shape, field semantics, timing, output scale, objective, sampling and fitting/cross-fitting boundaries; do not transfer fitted state across splits.

Historical components belong to the frozen task/dataset. Cross-run storage and source loading remain host responsibilities. CouponEvo exposes a bounded set of prior recipe sources and preserves component assessments in same-task experience; unavailable donor code must remain an explicit limitation. None of these contract checks establishes higher model quality without real host experiments.
