# Scaling and hardware-aware architecture

Longer histories and larger embeddings can improve capacity while raising memory, communication, and serving cost. HSTU-like sequence units are one architecture aimed at long behavior streams; batching, sparse embedding lookup, precision, and sequence truncation may matter just as much as the attention block. Scaling is worthwhile only if the quality-versus-cost frontier improves.

## Data and conditions

Record sequence-length distribution, item vocabulary, embedding table size, GPU memory, throughput, training time, latency, and the time cutoff. Very long histories may be mostly redundant; rare items may dominate memory without sufficient learning signal. A compute budget is part of the experiment contract.

## Controlled experiment

Build a learning curve over history length and model capacity for a simple baseline before switching architecture. Compare candidate blocks at matched FLOPs, parameter count or wall-clock budget and the same data windows. Measure quality on tail and long-history slices, training stability, peak memory, energy/cost if available, and serving P95 latency.

## Interpretation and feature needs

A gain that requires infeasible serving cost is not a deployable improvement. If truncation harms quality only for long-history users, consider compression or retrieval over histories. Request richer event semantics only when the existing events are known to omit relevant behavior or controlled models repeatedly cannot distinguish the observed sequence states.

## Method choice

| Method | What changes | First discriminating test |
| --- | --- | --- |
| HSTU | Causal sequence blocks use position/time information and gated projections. | Compare with a smaller causal sequence encoder at matched history length and cost. |

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_6_scaling/1.hstu.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_6_scaling/1.hstu.html), [docs/chapter_6_scaling/2.gen_rank.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_6_scaling/2.gen_rank.html), [src/funrec/models/hstu.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/hstu.py).
