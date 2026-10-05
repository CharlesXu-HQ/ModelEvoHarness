# Streaming retrieval and indexing

The retrieval model can be sound while the online index is stale. Streaming updates change item vectors, availability, and the searchable catalog between offline training and serving. This direction is about freshness and state consistency, not a new neural layer.

## Data and conditions

Record item publish and removal times, feature update timestamps, index build time, encoder version, and serving latency. Define what an eligible item is at each request time. A historical replay needs the catalog and index state that existed then; evaluating against today's catalog leaks future inventory and understates freshness errors.

## Controlled experiment

Keep the encoder fixed and compare index refresh cadences or incremental update rules. Evaluate recall, stale-item exposure, missing-new-item rate, latency, update backlog, and resource cost over temporal windows. Replay old requests with versioned states when available. A new encoder should be a separate experiment so freshness and representation effects are identifiable.

## Interpretation

If the model's offline vectors improve but replay quality falls as index age grows, prioritize index operations. If the business catalog has no versioned availability or update history, that fact directly blocks realistic freshness evaluation and justifies a data-instrumentation request. Do not mislabel an index issue as feature underfitting.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_1_retrieval/5.streaming_index/1.trinity.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/5.streaming_index/1.trinity.html), [docs/chapter_1_retrieval/5.streaming_index/2.streaming_vq.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/5.streaming_index/2.streaming_vq.html).
