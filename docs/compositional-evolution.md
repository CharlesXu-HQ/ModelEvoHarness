# Compositional model evolution

## Research intent

Prefer informative changes inside a suitable backbone before replacing it. This is a research priority, not an obligatory trial count. A clearly unsuitable estimator or architecture can be replaced immediately. The Agent must explain the evidence, data fit, local alternative and comparison that makes the switch worth its budget.

Three different decisions are recorded:

- **Estimator**: the learning/estimation formulation, such as a causal meta-learner or a supervised ranking estimator.
- **Backbone**: the representation/predictor family used inside that formulation.
- **Components**: actual local code for representations, interactions, heads, losses, sampling, optimization, calibration or other mechanisms. This list of examples is open.

T/S/DR learner names do not, by themselves, specify the network inside them. Similarly, a network with a familiar name may contain useful original adaptations. The Agent may read local reference modules and write/adapt their internals inside a candidate, subject to the host's execution contract. It is not limited to instantiating whole catalog models.

## Proposal contract

The built-in `OpenAICompatibleAgent` advertises `requires_model_design=True`; the engine activates `task.model_design_required` and `task.horizontal_expansion_required` for its searches. Custom adapters can declare these snapshot fields to use the same contract. Older snapshots without the flags may omit the corresponding declarations; supplied designs are still validated. Enabling the contract changes the frozen run identity, so old runs are not silently resumed under a new protocol.

Add `research.model_design` alongside the existing falsifiable research fields:

```json
{
  "estimator_id": "t_learner",
  "backbone_id": "tabular_mlp",
  "estimator": "Two independent outcome models",
  "backbone": "Tabular MLP with explicit interaction terms",
  "change_scope": "initialize",
  "parent_trial_id": null,
  "rationale": "Begin a tracked recipe after inspecting the untracked seed",
  "data_fit": "The host declares existing pre-decision tabular fields; no sequence is assumed",
  "comparison_plan": "Same split, objective and compute budget as the seed",
  "horizontal_expansion": {
    "decision": "defer",
    "rationale": "This candidate first isolates the explicit pair term within one encoder",
    "comparison_plan": "Next compare separate complementary encoders against this unsplit representation at matched capacity",
    "groups": []
  },
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

This is a complete `research.model_design` schema example, not an evaluated recipe or a complete proposal. Actual `input_fields`, capabilities and code locations must come from the task and candidate. Training-only components may have empty input/capability lists. `reference_method_id` is optional for original code; when present it must name a bundled implementation for the host framework, and triggers bounded source reading even without a top-level `method_id`. Reading an entire reference module does not make its full-model prerequisites optional.

Every enabled proposal assesses horizontal expansion using actual field semantics and a measurable bottleneck. Any suitable subnetwork can have multiple instances, including several references to the same method; fusion remains candidate-defined. `expand` records at least one group, `defer` records a specific reason and next comparison while keeping existing groups, and `not_applicable` has no groups. A later loss-only change must still describe the current branches. Optional `feature_groups` express semantic field subsets without granting sequence or alignment capabilities. See the [complete horizontal contract, source API and example](horizontal-composition.md).

`estimator_id` and `backbone_id` are stable identifiers. `estimator` and `backbone` describe the current recipe and may change when a branch is ablated, a loss changes, or a component is adapted. For `local`, omit IDs to inherit them from the parent, or supply the exact parent IDs. Description edits do not change identity; explicitly different IDs are rejected for a local edit. Existing records without IDs get deterministic IDs when read, without rewriting the original journal. New canonical records store the IDs, so subsequent description changes cannot alter them.

`initialize` registers a recipe only when no prior tracked design exists. `local` keeps the two IDs; `switch` changes at least one. IDs are declarations, not proof of the actual computation; host code review must still detect a misleading rename or an undeclared structural change. For an older switch proposal without IDs, unchanged descriptions preserve the corresponding parent ID and changed descriptions receive deterministic IDs.

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
| retain | Same declared mechanism, code locations, input fields, required capabilities, reference, instance path and output contract when present; the source must be eligible for reuse. Actual execution still needs checking. |
| adapt | Reuse an idea with explicitly changed interfaces or training semantics; provide compatible target declarations and a test. |
| drop | Exclude the source component with a reason; no target component mapping. |
| retest | Revisit an uncertain, failed or invalid idea as a new hypothesis; no inherited claim of success. |

Invalid, unevaluated, blocked or harmful source components, and components without a recorded assessment, can only be dropped or retested. An assessed but inconclusive component may be carried as an explicit hypothesis with its uncertainty; this does not establish a gain. Plans referencing invented source/component IDs, unavailable fields/capabilities, missing parent dispositions, or a switch disguised as a local change fail validation. This checks declarations and lineage; it does not statically prove that the code followed the plan. The host must compare donor/candidate source, actual prediction paths and runtime contracts.

For a parallel component, `retain` also preserves its branch/fusion role, peer branch paths, fusion path, and shared members/source locations. These relationships are compared by instance path, so component/group renames and rationale edits alone do not count as rewiring. Changing connections or sharing, or removing a group while keeping its components, requires `adapt` or `retest` for affected surviving components. Every branch and fusion has a distinct path, including aliases that point to one shared object.

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

Assessments include every parallel branch and fusion, with tested input conditions, sharing dependencies and the next discriminating control. Shared-weight retraining after a branch ablation can change the remaining branches; a one-branch source edit alone cannot establish an isolated effect. The host's audit must account for that coupling. Check actual instance routing, forward/call paths, fusion, parameter registration and gradients when observed. The graph and component metadata do not prove execution or benefit.

For custom host calls to `validate_reflection`, include the current research and `trial_status` in the evaluation view. The native engine supplies these; the host retains ownership of independent evidence.

## Practical comparison and limits

For a switch, preserve the previous recipe and compare a plain new backbone with the selectively composed new backbone under the same objective, split and compute budget. If the experiment budget cannot isolate every factor, choose the most informative comparison and retain uncertainty. No universal component-transfer benefit is assumed. Verify shape, field semantics, timing, output scale, objective, sampling and fitting/cross-fitting boundaries; do not transfer fitted state across splits.

Historical components belong to the frozen task/dataset. Cross-run storage and source loading remain host responsibilities. CouponEvo exposes a bounded set of prior recipe sources and preserves component assessments in same-task experience; unavailable donor code must remain an explicit limitation. None of these contract checks establishes higher model quality without real host experiments.
