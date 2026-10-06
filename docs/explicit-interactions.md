# Explicit feature interaction research

## Design

An embedding is a representation of an input field, not a competing source of information. The Agent should first inspect available field views and the current model's interaction coverage, then choose a falsifiable structural change. Numeric fields are not automatically preferred to categorical fields; neither a model name nor an untested field pair establishes value.

The host supplies `interaction_views`: each view has an `id`, an open-ended `kind`, original `fields`, and a `representation` description. A derived missingness view still cites its original fields. The contract supports semantic or statistical groups without inferring user/item/sequence semantics from anonymous columns. Groups describe available representations; they do not establish measured importance or pair support.

The Harness provides a compact pair-coverage template and prior interaction experiments. When `interaction_plan_required` is enabled, each proposal supplies `research.interaction_plan`:

- `coverage`: one entry per available unordered view pair (including within-view pairs with at least two fields), each with `left`, `right`, `status` and `basis`. Status is `implicit`, `explicit`, `mixed`, `none`, or `unknown`. These remain Agent assessments, not verified host observations.
- `decision`: `test` or `defer`, plus a `rationale`. Deferring permits loss, optimization, diagnosis and other work without forcing an interaction experiment.
- `candidates`: ranked hypotheses with `id`, `views`, `order`, `mechanism`, `priority`, `reason`, `cost`, `risks`, and host `evidence_ids`. Mechanisms and view kinds remain open-ended; the catalog is not a whitelist.
- `selected_id`: the highest-priority candidate when testing; null when deferring.
- `control`: for a test, a nonempty `config_patch.model` and `expected_effect`. The patch changes model configuration in the same source; the host owns execution and resource budgets.

Only one selected hypothesis is required. Multiple alternatives are useful when they distinguish explanations, but exhaustive original-field pair enumeration and exhaustive experiments are not requirements. Coverage is a view-level audit, not a demand to materialize every crossed feature. Existing host-backed fields can be crossed in code without asking a human to recreate the dataset. Requests for unavailable histories, timestamps or aggregates continue to follow the data-request contract.

## Priority and diagnosis

Before adding capacity, consider branch output scale, fusion, gradients, missingness, field support, redundant views and leakage. An MLP provides implicit coverage; DCN can provide explicit crosses involving both numeric and categorical inputs. Adding FM on those inputs is an incremental hypothesis, not automatically filling an uncovered relationship.

Prioritize expected information and feasible compute based on current evidence. A failed categorical FM recipe does not rule out numeric-category terms, different fusion or sharing choices. Conversely, an untested interaction does not automatically deserve priority. Separate the failed recipe, the suspected cause and the next discriminating control. Keep findings bound to the task and dataset identity.

## Native reference components

Read with `read_reference` and `include_interactions: true`:

- `NumericFieldEmbedding(num_fields, embedding_dim)` scales each numeric value by a learned field vector. An optional observed-value mask prevents imputed values from silently participating. Binary indicators can use the same representation; include or exclude them deliberately.
- `GroupedFM(field_count, groups, interactions)` consumes field vectors and returns one second-order interaction value per selected group pair. Group self-pairs use only distinct fields. Cross-group terms use products of group sums. Groups are disjoint within one instance and pairs are unique, so self-products and double counting are excluded. Separate instances permit other field partitions.

Both PyTorch and TensorFlow implementations are independent framework-native source. They avoid expanding every field pair into a large intermediate tensor. Returned group terms can be scaled, gated, concatenated or fused by a learned head inside the existing backbone. They are reference building blocks, not the only permitted structures.

Numeric standardization, bin boundaries, vocabularies and support estimates must be fitted on training data only. A raw feature and its missingness/bin view share a source: consider tautological or duplicate crosses before including both. Sparse Cartesian IDs, product/bilinear layers, higher-order crosses and learned selectors remain valid alternatives when supported by the task; define their cost, support and appropriate controls.

## Host execution and evidence

The generic Harness validates the proposal and supplies reference code; it does not infer that code implements the declared coverage. A host may execute the same-source control and return paired metrics. Report planned, completed and failed controls separately. A controlled recipe comparison is useful evidence but does not, by itself, establish isolated benefit for every component or every field pair.

CTREvo opts into the contract, declares numeric, categorical and missingness views, and reserves at most one additional full-data control training per interaction trial. Its example uses the existing embedding MLP with small numeric-field vectors and grouped second-order terms. The selected model and the control use identical source, seed, splits, batch size and epoch budget. No holdout labels enter the search.

Same seed and source do not guarantee shared initialization if a toggle changes module construction or random-number consumption. Prefer constructing the same modules and gating their outputs. Even then, training shared weights changes both branches; a control is a recipe comparison, not automatic isolated attribution.
