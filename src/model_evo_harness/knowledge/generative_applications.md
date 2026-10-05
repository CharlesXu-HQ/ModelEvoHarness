# End-to-end generation for recommendation, search and ads

Generation can emit an item code, query expansion, ad candidate, or a sequence of actions. This can remove a separate scorer for some tasks, but it introduces legal-output, constraint, and attribution questions. For search or ads, generated content or IDs must satisfy inventory, policy, and budget constraints; open-ended token likelihood is not the business objective.

## Data and conditions

Define the generated target, its tokenizer, catalog or ad inventory, eligibility rules, supervision provenance, and evaluation at the decision timestamp. Search requires query and relevance context; advertising requires auction and policy constraints; recommendation requires user history and item mapping. Keep protected and post-decision data out of prompts and training features.

## Controlled experiment

Compare with the incumbent retrieve-then-rank or direct classifier under the same requests, catalog, and latency budget. Measure target validity, recall/quality, utility, calibration where applicable, constraint violations, and decoding cost. Ablate the generation step and constrained decoder separately.

## Interpretation and feature needs

A language-model answer that cannot be converted into an eligible action is a failed candidate, even if its text looks relevant. If inventory and target mappings are missing, request them as prerequisites. If they exist, test simpler retrieval and ranking controls before claiming that generation needs additional business features.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_7_gr_e2e/1.recommendation.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_7_gr_e2e/1.recommendation.html), [docs/chapter_7_gr_e2e/2.search.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_7_gr_e2e/2.search.html).
