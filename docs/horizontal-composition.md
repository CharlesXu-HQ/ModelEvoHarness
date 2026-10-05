# Horizontal subnetwork composition

Horizontal expansion asks whether complementary computations should run as branches and contribute through an explicit fusion step. It applies to any suitable encoder, interaction block, expert, head or other subnetwork. Multiple instances may use the same reference method, different methods or original code. The catalog, branch examples and fusion examples do not form a closed list.

## Start from input semantics

Inspect the host's real fields, timing, schema and measured bottleneck. Distinct behavior streams may justify separate encoders; aligned attributes of the same events may belong in one event representation. Different transformations of the same fields can also support a controlled branch experiment. Neither a field count nor a familiar model name decides the topology.

The optional snapshot fragment below declares subsets of existing fields. Each group needs a unique `id`, nonempty `fields` drawn from `snapshot.fields`, and a nonempty `rationale`. `validate_feature_groups(groups, fields)` normalizes these declarations without adding capabilities.

```json
{
  "fields": ["pre_visits", "pre_spend"],
  "feature_groups": [
    {"id": "activity", "fields": ["pre_visits"], "rationale": "Historical visit count before the decision"},
    {"id": "value", "fields": ["pre_spend"], "rationale": "Historical spending before the decision"}
  ]
}
```

These are semantic declarations, not observations of sequence order, event alignment, feature timing or independent streams. Anonymous scalar fields remain anonymous. A feature group need not map to exactly one branch, and overlapping groups do not establish independent evidence.

## Assess every proposal and record the current graph

The built-in provider's searches enable `horizontal_expansion_required=true` alongside `model_design_required=true`. Hosts using custom Agents can enable these snapshot flags explicitly. Every enabled experiment supplies `research.model_design.horizontal_expansion`:

| Decision | Current candidate and next comparison |
| --- | --- |
| `expand` | Record at least one parallel group and explain the testable benefit and comparison. |
| `defer` | Give the specific data, evidence or budget reason for prioritizing another test. Preserve all parallel groups that remain in the current candidate, even if this trial only changes its loss or training. |
| `not_applicable` | Use empty groups, explain why this candidate has no parallel group, and state the next discriminating comparison. |

Increasing width or switching a model name alone does not answer this assessment. A new group is not mandatory on every iteration. A graph declaration must describe the current candidate rather than silently erase unchanged branches. Removing a group also requires accounting for its parent components under the existing inheritance contract.

Older snapshots without the flag may omit this field; supplied declarations are still validated. Compatibility does not rewrite old proposals or resume them under a changed frozen protocol.

## Complete model-design example

This is a complete `research.model_design` value for an initial tracked design. It is a schema example, not an executed recipe or a complete experiment proposal. The enclosing proposal still needs the [research fields](adapter-contract.md#agent-context-and-proposal), host candidate artifact and any required evidence. The example assumes the two fields above and `tabular_features` are available; actual code paths must be supplied by the candidate.

```json
{
  "estimator_id": "supervised_regression",
  "backbone_id": "parallel_scalar_views",
  "estimator": "Supervised regression on observed outcomes",
  "backbone": "Two scalar encoders followed by concatenation and a prediction head",
  "change_scope": "initialize",
  "parent_trial_id": null,
  "rationale": "Test separate transformations of activity and spending inputs",
  "data_fit": "The host supplies two pre-decision scalar fields; no sequence capability is assumed",
  "comparison_plan": "Compare with an unsplit encoder at matched capacity, loss, split and training budget",
  "components": [
    {
      "id": "activity_encoder",
      "mechanism": "Encode historical visit counts",
      "instance_path": "model.branches.activity",
      "code_sections": ["ActivityEncoder.forward"],
      "input_fields": ["pre_visits"],
      "required_capabilities": ["tabular_features"],
      "output_contract": "Float representation [batch, 8], no temporal mask"
    },
    {
      "id": "value_encoder",
      "mechanism": "Encode historical spending",
      "instance_path": "model.branches.value",
      "code_sections": ["ValueEncoder.forward"],
      "input_fields": ["pre_spend"],
      "required_capabilities": ["tabular_features"],
      "output_contract": "Float representation [batch, 8], no temporal mask"
    },
    {
      "id": "fusion",
      "mechanism": "Concatenate the named representations and predict the outcome",
      "instance_path": "model.fusion",
      "code_sections": ["ConcatHead.forward"],
      "input_fields": ["pre_visits", "pre_spend"],
      "required_capabilities": ["tabular_features"],
      "output_contract": "Regression score [batch, 1] in the declared outcome scale"
    }
  ],
  "horizontal_expansion": {
    "decision": "expand",
    "rationale": "Separate scalar transformations may capture complementary response shapes",
    "comparison_plan": "Use the unsplit control, then remove each branch with matched retraining conditions",
    "groups": [
      {
        "id": "scalar_views",
        "branch_ids": ["activity_encoder", "value_encoder"],
        "fusion_id": "fusion",
        "parameter_sharing": []
      }
    ]
  },
  "inheritance": []
}
```

Every group names at least two distinct branch component IDs and a different fusion component ID. Each branch and fusion must declare `instance_path`, `output_contract`, actual `input_fields`, and `code_sections` containing its `forward` or `call` entry point. All branch and fusion instance paths are distinct, although different alias paths can reach a shared object. Group IDs and fusion IDs are unique; groups may nest when their branch-to-fusion graph is acyclic. Repeated `reference_method_id` values are allowed for distinct instances.

`retain` checks the component's graph relationships as well as its own fields: branch/fusion role, peer branch paths, fusion path, and shared members and source locations. It compares actual instance paths, ignoring group names, component-ID wording and rationale. Renaming IDs alone therefore preserves the relationship, while rewiring, changing sharing or removing a group requires `adapt` or `retest` for affected components that remain. Removed components use `drop`. A `defer` decision with the same graph can retain its components.

`parameter_sharing: []` declares independent parameters between the group's branches. If the candidate instead shares a projection, a **sharing-entry fragment** could be:

```json
{
  "component_ids": ["activity_encoder", "value_encoder"],
  "code_sections": ["model.shared_projection"],
  "rationale": "Both branch paths use this same projection object; their input transforms remain separate"
}
```

Put that entry in the appropriate group's `parameter_sharing` list and match the actual code. Each entry names at least two branches from that group. Sharing an embedding or projection does not imply sharing the entire encoder. Equal initial values or copied code do not establish shared parameters.

## Read and use native composition source

The complete preparatory request below reads only the generic composition module. Use `tensorflow` for the native Keras implementation; framework selection must match a declared task framework.

```json
{"action":"read_reference","framework":"pytorch","method_ids":[],"include_composition":true}
```

`include_composition` can be combined with method IDs and `include_training`. It shares the existing bounded reading budget and source-hash ledger. An experiment declaring nonempty groups automatically receives the relevant composition source before final proposal generation when the task framework is known and that source has not already been read. This also applies to `defer` with retained groups.

Both `model_evo_harness.models.pytorch.composition.ParallelBranches` and `model_evo_harness.models.tensorflow.composition.ParallelBranches` accept `(branches, fusion)`. Branches are a mapping from names to native `nn.Module` or Keras layer instances; fusion is a candidate-written native module/layer. At invocation, supply a matching mapping from each branch name to its keyword arguments. Each branch receives `branch(**branch_inputs[name])`; fusion receives the dictionary of named outputs. The wrapper requires matching branch keys and registers child modules/layers. Reusing the same object shares its parameters; distinct objects stay independent. PyTorch remains native PyTorch and TensorFlow remains native Keras. “Parallel” describes the computation graph, not concurrent CUDA streams.

The candidate controls preprocessing, encoders, output representations, fusion, loss and training. Scalar reference scores cannot silently substitute for vector representations: expose the required representation or declare score-level fusion. The wrapper does not select branches or fusion from an enum and does not provide a training loop.

## Check execution and retain conditional experience

Metadata validation checks references and graph consistency. The host still needs source and runtime evidence for input routing, `forward`/`call` execution, tensor shapes/scales/masks, contribution to the score/loss, object sharing, optimizer registration and measured gradient paths. Missing gradient evidence remains unverified. A class name, declared path or successful schema check does not prove these behaviors.

Use controls that distinguish the hypothesis: an unsplit or pooled view of the same inputs, a matched-capacity model where feasible, branch removal/replacement, shared versus independent parameters, or a fusion change. These are examples, not a required list. Keep data, split, sampler, loss, training schedule and budget fixed or declare their joint changes. Removing one branch and retraining shared weights can change surviving branches, so a one-branch code edit alone does not establish isolated attribution.

The existing `technical_experience.component_assessments` covers every branch and fusion. Record the tested input semantics, alignment, sharing dependencies, fusion interactions, uncertainty and the next control. Whole-recipe gains do not prove each component's value; failed recipes do not establish a missing-feature diagnosis. These records remain bound to the task and dataset and do not, by themselves, establish Agent behavior across multiple real searches or improved model metrics.
