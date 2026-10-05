# Collaborative retrieval

Collaborative methods infer similarity from observed user-item interactions. UserCF and ItemCF aggregate neighbors; Swing discounts shared interactions that come from very popular or weakly informative users; matrix factorization learns low-dimensional user and item vectors. These mechanisms can recover repeat preference that content features omit, but they inherit exposure bias and cold-start limits.

## Data and conditions

Require stable user and item IDs, an item catalog, interaction timestamps, and a definition of positive feedback. For implicit data, an unobserved item is not automatically a disliked item. Measure graph sparsity, head/tail concentration, and new-user/item coverage. Serving requires either a maintained neighbor table or item embeddings plus a retrieval index.

## Controlled experiment

Compare popularity, ItemCF, and one factor model with the same temporal cutoff, candidate universe, seen-item filter, top-K, and downstream ranker. Tune only on validation. Evaluate full-catalog recall where feasible; a sampled candidate test may reward a different algorithm. Report cold and tail slices, catalog coverage, diversity, memory, and update cost.

## Interpretation and feature needs

A gain concentrated among dense-history users favors collaborative evidence; failure on unseen entities is expected and suggests a content or onboarding path. If available content features already cover cold entities, test a hybrid before asking humans for new fields. A request for new item attributes is justified when a missing attribute can be named from catalog knowledge or repeated, controlled failures show that existing representations cannot separate meaningful items.

## Method choices

| Method | What changes | First discriminating test |
| --- | --- | --- |
| UserCF | Neighbor users vote on a candidate item. | Compare with popularity and ItemCF on dense versus sparse users. |
| ItemCF | Similar items extend each user's history. | Hold seed history and K fixed against UserCF. |
| Swing | Downweights overly common user-pair co-occurrences. | Check whether gains survive on tail items, not only popular co-clicks. |
| FunkSVD | Fits a user-item latent dot product. | Compare against ItemCF with equal catalog coverage and cold fallback. |
| BiasSVD | Adds global, user and item intercepts to latent factors. | Test whether bias terms, rather than factor capacity, explain the gain. |

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_1_retrieval/1.cf/1.itemcf.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/1.cf/1.itemcf.html), [docs/chapter_1_retrieval/1.cf/2.swing.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_1_retrieval/1.cf/2.swing.html), [src/funrec/models/item_cf.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/item_cf.py), [src/funrec/models/user_cf.py](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/src/funrec/models/user_cf.py).
