# Scenario and context specialization

A pooled model assumes the same parameters can serve every context. STAR uses scenario-conditioned components; HMoE routes contexts among experts; PEPNet and APG adapt representations or parameters. Such designs are useful when residuals differ across stable, pre-decision scenarios and each scenario has enough support. They can also overfit small segments and hide weak global calibration.

## Data and conditions

Define scenario keys from fields known before action, such as placement, market, or lifecycle stage. Record segment size, label rate, treatment coverage where relevant, and serving-time availability. A segment built from future outcomes is not a valid context feature. Costs and latency grow with specialization.

## Controlled experiment

Compare the pooled baseline, explicit scenario indicator, separate small heads, and one conditional/gated model. Use the same rows, fields, split and declared model-size budget. Report primary value and calibration per scenario, especially low-support segments; test whether gains persist across time.

## Interpretation and feature needs

If the explicit scenario indicator closes the gap, a complex expert network is unnecessary. If scenario effects are known by business rules but the scenario identifier is absent, request that identifier immediately. If all available scenario variants fail, first review support, label drift and split design before requesting new segmentation fields.

## Method choices

| Method | What changes | First discriminating test |
| --- | --- | --- |
| STAR | Shares a main network while allowing scenario-specific transforms and normalization. | Compare with pooled model plus scenario ID and small local heads. |
| HMoE | Gates shared experts into domain predictions. | Check whether expert specialization persists across time. |
| PEPNet | Gates input embeddings and intermediate layers from context. | Ablate input and layer gates separately. |
| APG | Generates compact context-conditioned layer parameters. | Compare quality and serving cost with simpler scenario indicators. |

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_2_ranking/5.multi_scenario/1.multi_tower.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/5.multi_scenario/1.multi_tower.html), [docs/chapter_2_ranking/5.multi_scenario/2.dynamic_weight.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_2_ranking/5.multi_scenario/2.dynamic_weight.html), [src/funrec/models/star.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/star.py), [src/funrec/models/apg.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/apg.py).
