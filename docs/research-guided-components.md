# Diagnose and improve components inside a backbone

These five additions provide native PyTorch and TensorFlow mechanisms for four
research questions. They retain the current backbone and experiment protocol;
the Agent chooses a local intervention using measured evidence. The two analysis
tools are discoverable through `analysis_api`; the three model components use
`model_api`. Both support whole-module `read_references` and include input
contracts, controlled comparisons and explicit limitations.

| Method ID | Class | Purpose |
| --- | --- | --- |
| `representation_diagnostics` | `RepresentationDiagnostics` | Per-layer entropy effective rank, stable rank, token variance and aligned branch similarity. |
| `gradient_feature_selection` | `GradientFeatureSelector` | First-order field/cross saliency, overlap-aware redundancy and protected budget selection. |
| `progressive_interaction` | `ProgressiveInteraction` | Reuse initial-affinity edge order and progressively allow more interactions at successive layers. |
| `contextual_history_encoding` | `ContextualHistoryEncoder` | Preserve supplied per-event item and context information in historical tokens. |
| `relation_history_attention` | `RelationHistoryAttention` | Independently normalize user-to-history, target-to-history and target-to-user relations. |

The catalog retains 82 unified method cards: 80 model methods and 2 analysis
tools. Their implementation and pattern records are separate:

```text
src/model_evo_harness/
├── analysis/{pytorch,tensorflow}/research_diagnostics.py
├── models/{pytorch,tensorflow}/
│   ├── progressive_interaction.py
│   └── contextual_history.py
└── data/
    ├── model_implementations.json     # 162 entries, including both TwoTower implementations
    ├── analysis_implementations.json  # 4 entries for 2 tools in both frameworks
    ├── structure_patterns.json       # model changes: structural_change
    └── analysis_patterns.json        # measurements and screening: operation
```

The host calls analysis tools explicitly while diagnosing a baseline or completed
trial, using captured activations and training gradients. The Agent can use the
resulting evidence to propose a change; the host integrates model components in
the candidate's forward pass and trains and evaluates it. Discovery and source
reading install no hooks and execute no tools or training. An analysis tool ID
cannot serve as an experiment's standalone `research.method_id`.

## Measure before changing the network

```python
from model_evo_harness.analysis.pytorch.research_diagnostics import (
    RepresentationDiagnostics, GradientFeatureSelector,
)

probe = RepresentationDiagnostics(center=False)
observations = probe.inspect(
    {"input": input_tokens, "cross": cross_tokens},
    valid_masks={"input": field_mask, "cross": field_mask},
)
selector = GradientFeatureSelector()
screen = selector(field_activations, training_loss_gradients, field_mask)
kept = selector.select(
    screen["saliency"], budget=12,
    protected_indices=(0,), protected_groups=((2, 5), (7, 9)),
)
```

TensorFlow exposes the same tools under
`model_evo_harness.analysis.tensorflow.research_diagnostics`.
The caller captures actual activations and gradients. The diagnostics detach
their results and do not modify predictions. Ranks use each sample's token ×
channel matrix; output variance measures variation over valid tokens, averaged
over channels. Branch similarity requires aligned tensors. An all-zero matrix
has rank zero by convention; a constant nonzero uncentered matrix can have rank
one and zero variance. Neither rank nor saliency is a performance or causal metric.

Screen only training observations. Retaining a cross feature also requires its
source fields. `protected_groups` retains every supplied member; an insufficient
budget fails explicitly. Weak marginal saliency cannot rule out useful
interactions. The subsequent full-input versus screened comparison requires
matched retraining and group ablation, and the final holdout remains sealed.

## Change the interaction schedule

```python
from model_evo_harness.models.pytorch.progressive_interaction import ProgressiveInteraction

cross = ProgressiveInteraction(num_fields=8, dim=32, edge_budgets=(16, 32, 64))
cross_tokens, masks = cross(field_tokens, field_mask, return_masks=True)
```

The schedule counts global directed edges, including self edges, out of `F²`.
Descending scaled raw initial dot products determine the order; ties use stable
row-major order. Ranking is reused within the forward call, not recomputed from
mixed hidden states. Each layer still has learned current Q/K/V projections and
an FFN. Compare `(64,64,64)`, `(16,16,16)` and `(16,32,64)` with equal parameter
counts and the same downstream head, then inspect task quality and cost. Empty
attention rows have zero context; padded fields stay zero.

This is an original MGDIN-inspired local mechanism. It does not reproduce learned
initial Q0/K0, multigranularity branches, semantic IDs or the complete paper model.
The explicit edge-count convention resolves ambiguities in the source equations.

## Preserve historical context and separate relations

```python
from model_evo_harness.models.pytorch.contextual_history import (
    ContextualHistoryEncoder, RelationHistoryAttention,
)

encode = ContextualHistoryEncoder(item_dim=16, context_dim=8, dim=32)
history, valid = encode(event_item_features, event_context_features, history_mask)
relations = RelationHistoryAttention(dim=32, num_heads=4)(
    user_tokens, history, target_tokens, user_mask, valid, target_mask,
    return_attention=True,
)
```

`ContextualHistoryEncoder` applies `tanh(item_projection + context_projection)`
per event. Order is retained but no temporal or positional encoding is invented.
The caller provides aligned features available before the prediction cutoff.
Current labels and future event attributes are not valid history inputs.

`RelationHistoryAttention` returns `user_history`, `target_history`, `target_user`,
their availability masks and optional per-relation attention weights. Each
relation has separate Q/K/V/output parameters and normalization; candidates do not
attend to one another. Empty or fully padded sides return zero and false masks.
Gradients can flow into the encoders; read-only evidence is not implicit
stop-gradient. The caller supplies fusion and the task head.

These are contextual and relational cores inspired by SIF and CCFormer, not SIF
quantization/codebooks or the complete CCFormer history hierarchy. TensorFlow
exposes the same classes and methods under `models.tensorflow`.

## Agent use and optimization stages

The shared [evidence-driven guide](../src/model_evo_harness/knowledge/evidence_driven_iteration.md)
is loaded into the experiment context. It connects observed evidence, competing
explanations, one local intervention, an ablation, and dataset-bound lessons.
Methods without the needed hidden states, gradients or real history remain
`needs_data`; that status can mean missing instrumentation, not missing raw data.
Source delivery and declared capability do not prove correct use or task gain.

Sample-bag weighting and gradient-conflict handling act during **training**;
false-negative and feedback-maturity checks also affect **training-data preparation**.
Uncertainty-triggered extra paths or ensembles act during **inference**, generally
with frozen parameters and additional compute. Those options are explained in the
guide, not implemented as part of these four requested additions.

## Sources and validation boundaries

Original explanations and implementations informed by [RankUp](https://arxiv.org/html/2604.17878v3),
[TayFCS](https://arxiv.org/html/2507.03895v1),
[MGDIN](https://arxiv.org/html/2603.12586v1),
[Sample Is Feature](https://arxiv.org/html/2604.15650v3) and
[CCFormer](https://arxiv.org/html/2607.28070v2). Papers were discovered through
[step-by-step-learning-ai](https://github.com/lmxhappy/step-by-step-learning-ai/tree/fb3a35da6671a13ba71a074b68e20574c5525e98).
No paper prose or third-party model code is copied. Component algebra, masking,
gradients and framework agreement are separate from dataset-level effectiveness.
