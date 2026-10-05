# Horizontal subnetwork composition

This research principle applies to any suitable encoder, interaction block, expert, head or other trainable subnetwork. Model names do not limit the search space. After choosing a backbone, actively assess whether parallel components address complementary inputs or an observed bottleneck. A branch must have a testable role; merely replicating capacity does not establish a new mechanism.

## Input grouping

Start from the host's actual fields, optional `feature_groups`, time cutoffs and schema. Separate independent behavior streams from several attributes of the same events: event-aligned attributes may need joint event embeddings before one encoder. Overlapping windows, repeated fields or correlated groups are not independent evidence. Group declarations do not establish timestamps, sequence alignment, candidate identity, modality or causal provenance. Never invent a missing input or relabel anonymous statistics as behavioral sequences.

Multiple useful views of the same fields can also justify parallel branches. Explain the computational difference and compare against a capacity-matched control; an extra field group is not a requirement for every valid parallel design.

## Decide, then implement

Every enabled proposal records `model_design.horizontal_expansion` with a decision, rationale and controlled comparison. `expand` describes a candidate with at least one parallel group. `defer` explains why another experiment has higher information value and may retain existing groups. `not_applicable` explains why no parallel group belongs in this candidate. Decisions remain evidence-based rather than a fixed model sequence, branch count or quota.

Give each branch its own component ID and actual instance path, even if several reference the same method. Record its real input fields and output contract. Give fusion its own component ID and code location. Parameter sharing names precisely which module paths are tied; sharing embeddings does not imply sharing the entire encoder. Use an empty sharing list when branches have independent parameters. Shared modules may be reached through distinct instance paths.

For inheritance, `retain` also preserves each component's branch/fusion role, connected instance paths and group sharing declarations. Changing the graph or sharing requires `adapt` or `retest` for the affected group's components. Renaming a group or component ID with unchanged instance paths does not change those relationships.

The graph is a declaration that is checked for valid references and cycles. The host must still inspect source and execution. Neither an ID nor a class name proves a branch was used. Check input routing, temporal masks, forward/call execution, output shapes and scales, fusion, trainable-parameter registration, and gradient paths when measured. Unobserved gradients or weights must remain unverified.

## Local code available to the Agent

Read `models/pytorch/composition.py` or `models/tensorflow/composition.py` using `include_composition=true`. Both contain an independent native `ParallelBranches(branches, fusion)` implementation. It dispatches one dictionary of keyword inputs to each named branch and sends the named outputs to a custom fusion module. Passing the same module object to two branch names ties its weights; passing two instances keeps them independent. Fusion is supplied by the candidate, so its form is not restricted to a fixed list.

The candidate still owns encoders, preprocessing, loss, prediction heads and fit/predict semantics. If a reference exposes a scalar score but the proposed fusion needs vectors, adapt the reference to expose a representation or explicitly declare score-level fusion. Do not silently treat scores as embeddings. The wrapper registers modules; it does not certify useful learning, correct optimization or uplift identification.

## Controlled tests

Choose comparisons that discriminate the hypothesis: same-input unsplit or pooled control, matched-capacity control where feasible, branch removal/replacement, sharing versus independent parameters, and fusion ablations. These are examples, not a required enumeration. Keep training schedules, data, splits, losses and budgets fixed or declare joint changes. Existing groups should stay recorded even when a later experiment only changes the loss.

A branch ablation with shared weights can affect every branch after retraining. A fusion change can alter all incoming signals. Account for these dependencies before labeling an effect isolated. Record whole-recipe results separately from component-level conclusions. Better held-out prediction need not improve the host's policy objective.

## Dataset-bound experience

Use the existing component assessments to record every branch and fusion, input conditions, sharing dependencies, supported or refuted hypotheses, measured uncertainty and the next discriminating control. Record business observations only when host-defined metrics support them. A failed composition is not evidence that the dataset needs more fields; compare viable existing-input mechanisms first, subject to explicit domain prerequisites.
