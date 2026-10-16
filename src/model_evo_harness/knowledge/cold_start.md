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

## Reco-papers component map

These entries are independent computational components, not full training systems. Read each exact signature and limitation through `model_api` or `read_references`; use the same input tensors and evaluator when comparing them.

- `meta_embedding` / `MetaEmbedding`: Attribute-to-ID generator plus differentiable inner embedding update and cold/warm objective combination. Output: `generated_embedding`, `[B,D]`. Host owns support/query construction and nested autodiff. Generator alone trained by ordinary supervised loss is not meta-learning; no copying of unlicensed author code.

The machine-readable paper inventory links these methods to their PDF, family, structural pattern and training objective. Explicitly declared capabilities only establish input availability; they do not establish that an experiment will improve the target task.
