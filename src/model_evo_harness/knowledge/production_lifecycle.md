# Offline-online system lifecycle

A candidate architecture becomes a useful experiment only when its input and output contracts can run in production. Feature freshness, serving latency, index versions, fallback behavior, monitoring, and rollback can erase or reverse an offline improvement. The offline experiment should encode these limits before selecting a winner.

## Data and conditions

Version data, preprocessing, model artifact, feature schema, candidate generator, policy rule, and evaluation code. State which features are available online and their maximum delay. Define deployment guardrails: latency, memory, cost, missing-feature behavior, eligibility, and rollback thresholds. For online A/B tests, specify assignment unit and possible interference.

## Controlled experiment

Replay the selected model with production-equivalent feature timestamps and serving interfaces, then compare its outputs with the offline predictions. Profile latency and memory under realistic batch and traffic shapes. Stage release behind shadow evaluation or an authorized online experiment. Monitor data drift, calibration, decision rate, cost, and primary business metric with guardrails.

## Interpretation and feature needs

If online joins cannot provide a field used offline, document a direct serving-data gap before more modeling. If the field exists but is stale, fix the freshness contract. A laboratory metric alone cannot prove online lift under feedback loops or shared budgets.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_10_projects/1.intro.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_10_projects/1.intro.html), [docs/chapter_10_projects/2.architecture.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_10_projects/2.architecture.html).
