# Cold-start and sparse entities

New users, new items, and rare contexts fail for different reasons. A collaborative ID embedding has little or no evidence for an unseen ID; content and context features can generalize if they are available before the decision. Cold start is an evaluation slice and a data contract, not a single architecture.

## Data and conditions

Define coldness by time and count: new-to-training, recent launch, low-history, or unknown-at-serving. Preserve first-seen timestamps and catalog entry times. Random row splits often leak future interactions for the same entity into training, obscuring the problem. Do not let a new-item attribute be computed using post-launch outcomes.

## Controlled experiment

Report head, tail, new, and low-history slices with confidence intervals. Compare popularity, ID-only embedding, content-only model, and a gated hybrid at the same catalog and cutoff. Track coverage, calibration, latency, and the fallback policy for truly unseen values. Keep cold definitions fixed across experiments.

## Interpretation and feature needs

If content-only works on new items, the gap may be ID dependence rather than missing data. If domain catalog facts such as category or price are known to exist but are absent from the dataset, a human data request is justified immediately. Otherwise, test available metadata and hybrid/fallback choices before declaring new feature collection necessary.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_4_trends/2.cold_start.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_4_trends/2.cold_start.html).
