# Use internal evidence to choose a local experiment

The component catalog is an open set of tools, not an ordered checklist. Start
from the current task, dataset version, existing backbone and measured residuals.
Preserve a promising backbone while testing a falsifiable local change. Diagnostics
do not justify switching families or requesting human feature work by themselves.

## Observe, distinguish, intervene

1. Identify the observed evidence and its source: checkpoint, split, layer/field
   mapping, valid counts, objective, centering and available gradient lineage.
   Use `representation_diagnostics` for named-layer rank/variance and aligned
   branch similarity; use `gradient_feature_selection` for first-order screening.
   No observations means unknown. Do not invent activations or claim layer
   collapse from a model class name.
2. Name competing explanations. Similar branches may share redundant inputs or
   legitimately encode the same sufficient statistic. Low saliency can be caused
   by saturation, insufficient learning or missing interaction context. Observed
   evidence should select a test, not settle the explanation in advance.
3. Propose one local intervention and an ablation with fixed data, head, budget
   and evaluator. For tokenized field interactions, compare full, fixed-sparse
   and progressive edge schedules. For history features, test existing context
   encoding and separately normalized relations. Novel compatible interventions
   remain valid; these examples are not a mandatory sequence.
4. Compare task quality, uncertainty, slices and resource cost. A probe that only
   reports extra fields changes no predictions and is not a performance candidate.
   An improved effective rank without task gain is not an accepted improvement.
5. Record technical evidence and a separate business interpretation with its
   limits. Reuse depends on the same task, dataset version and protocol, not just
   the same model name. Preserve failures and counterexamples.

## Protect features that act through interactions

Use training-only activation/gradient observations for screening. Repeated
estimates help distinguish stable signal from a single batch. Supply
`protected_indices` and `protected_groups` explicitly; every field in each
protected interaction must survive selection, or selection rejects the budget.
When scoring constructed crosses, preserve their source-field mapping as well.
Do not assume weak marginal saliency makes a field dispensable. Confirm a
screening decision with group ablation and full-input versus screened retraining.
Redundancy is evidence for an experiment, not a causal deletion rule.

Before a human feature request, distinguish an absent field from an available
field that the current encoding or backbone has not used effectively. Explore
plausible local mechanisms supported by the available data and budget. Follow the
existing feature-gap evidence contract: experiment-based requests require two
distinct evaluated mechanisms and the specified trial evidence; that minimum
alone does not prove a data gap. An explicit expert requirement can show
an essential input is absent immediately; document that requirement and the
pre-decision availability of the proposed field. Do not fabricate a sequence
from aggregate columns or insert current outcomes into historical context.

## Which stage changes

| Stage | Example change | Evidence and boundary |
| --- | --- | --- |
| Training-data preparation | False-negative filtering or feedback-maturity masks | Exposure, label timing and observation-window definitions; do not silently relabel unseen events as negatives. |
| Training | Sample-bag weighting, objective changes, task-gradient conflict handling | Real labels and gradients for the declared objectives; compare task and guardrail metrics under fixed training budgets. |
| Structure used in training and inference | Progressive interaction, contextual event encoding, separate relation attention | Valid input semantics, masks, parameter/latency cost and component ablation. |
| Inference | Uncertainty-triggered extra feature paths or ensembles | Fixed trained model, calibrated selection signal and measured extra forward/backward cost. This is not a free training improvement. |

This release adds diagnostics, protected screening and the structural components
above. The stage table explains other research options; it does not imply new
sample-bag losses, gradient-surgery optimizers or inference policies are bundled.

## Research lineage

See [RankUp](https://arxiv.org/html/2604.17878v3),
[TayFCS](https://arxiv.org/html/2507.03895v1),
[MGDIN](https://arxiv.org/html/2603.12586v1),
[Sample Is Feature](https://arxiv.org/html/2604.15650v3), and
[CCFormer](https://arxiv.org/html/2607.28070v2). The stage distinctions draw on
[VarBPR](https://arxiv.org/html/2503.11067v3) and
[UTTSI](https://arxiv.org/html/2605.24989v1); their mechanisms are not implemented
by this addition. Paper results are not evidence of improvement on this task.
