# Generative representations and tokenization

A generative recommender maps a user history to output tokens representing items or actions. Tokenization can use raw IDs, semantic codes, or learned codebooks. The token scheme determines which items can be decoded, how unknown items appear, and whether similar items share prefixes. This is a different contract from scoring a fixed list of candidate IDs.

## Data and conditions

Require ordered histories, a versioned item catalog, item-to-token and token-to-item mappings, a legal-output decoder, and a target sequence. Record how token collisions, catalog additions, invalid outputs, and repeated recommendations are handled. If the action space is two coupon choices, a generative decoder needs a concrete advantage over a calibrated binary policy.

## Controlled experiment

Compare the chosen tokenization with a strong non-generative retriever or ranker on the same time split and candidate universe. Hold model size and inference budget explicit. Evaluate valid-ID rate, catalog coverage, recall/value, new-item behavior, and decoding cost. An ablation should isolate tokenization from a larger backbone.

## Interpretation and feature needs

A model that predicts plausible text but fails to map it to eligible items has not solved recommendation. Missing item-token alignment or catalog versioning is a direct data prerequisite. If the data contract exists but gains vanish under a fair latency budget, do not infer that more semantic fields are needed without representation tests.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_4_trends/3.generative.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_4_trends/3.generative.html), [docs/chapter_5_gr_basic/1.gr_intro.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_5_gr_basic/1.gr_intro.html).
