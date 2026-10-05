# Multiple outcomes and task dependence

Shared Bottom shares one representation across tasks; MMoE gates shared experts per task; PLE separates shared and task-specific experts. ESMM uses an observed ordered event relation, while AITM transfers information along an ordered task chain. These are ways to use several real labels, not ways to manufacture them. Treatment assignment plus a single outcome is not an ordered funnel.

For a genuine click-to-conversion funnel, `P(click and conversion | x) = P(click | x) × P(conversion | click, x)`. The identity follows from probability rules, but the model only benefits when both labels, denominators, and observation windows are defined consistently.

## Data and conditions

Define each outcome, observation window, missing-label process, task priority, and the unit on which tasks align. If click precedes conversion, distinguish conversion-per-click from conversion-per-impression. Sparse later-stage outcomes may benefit from earlier labels, but biased exposure can contaminate both. For coupon work, redemption, engagement, revenue, and cost must be defined separately when used.

## Controlled experiment

Compare a primary-task-only model, independent task heads, Shared Bottom, and one gated/specialized variant at matched data and capacity. Keep primary objective and selection metric fixed; report every task's calibration and the primary policy value. For ordered funnels, compare against a factorized probability baseline and verify the ordering empirically.

## Interpretation and feature needs

A secondary-task improvement with a primary-task loss is not success unless the declared utility says so. If labels are not observed or are observed only after selected exposures, document the gap before proposing a multitask network. Request new labels only when essential to the business decision or after tests show the present outcome set cannot discriminate the desired tradeoff.

## Method choices

| Method | What changes | First discriminating test |
| --- | --- | --- |
| Shared Bottom | All tasks share one backbone and use separate heads. | Compare with independent single-task models. |
| MMoE | A task gate mixes shared experts for each outcome. | Compare with Shared Bottom at matched capacity and label rows. |
| PLE | Adds task-specific and shared experts across extraction layers. | Test whether task conflict persists after a simpler gated mixture. |
| ESMM | Links click and post-click conversion through an ordered probability factorization. | Verify actual funnel labels and compare with independent heads. |
| AITM | Passes information along observed ordered task stages. | Ablate the transfer on later-task and calibration metrics. |
| M2M | Combines task, scenario and expert conditioning. | Use only when both multiple outcomes and supported scenarios exist; compare simpler two-axis controls. |

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_2_ranking/4.multi_objective/1.arch.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/4.multi_objective/1.arch.html), [docs/chapter_2_ranking/4.multi_objective/2.dependency_modeling.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/4.multi_objective/2.dependency_modeling.html), [src/funrec/models/shared_bottom.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/shared_bottom.py), [src/funrec/models/esmm.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/esmm.py).

Additional pinned source for `aitm`: [torch_rechub/models/multi_task/aitm.py](https://github.com/datawhalechina/torch-rechub/blob/beb8b46fb718ce486ea5feb6847ac3abf0c491d4/torch_rechub/models/multi_task/aitm.py).
