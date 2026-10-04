# FunRec coverage and applicability

This is a pinned, independently written research map of [FunRec](https://github.com/datawhalechina/fun-rec), not a port of its code. The upstream [commit](https://github.com/datawhalechina/fun-rec/commit/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572) is `5f7bd84d4403b5f92b6bedbd31e3984ef00a4572` and its Git tree is `84a4e63da7d7cb0ca8ea7e47128174535f0ab40a`. The [inventory](../src/model_evo_harness/data/upstream_inventory.json) records 55 substantive chapter pages, 38 model modules, 54 supporting core modules and 50 production-backend Python modules. `coverage_report` checks that each maps to exactly one family. FunRec's chapter on bias and cold start is included alongside the newer outline in its top-level README. Frontend assets and image files are recorded by the upstream tree but have no model-iteration mechanism, so they are outside this code inventory.

| Research family | Task stages | Required capabilities | Pages | Models | Core support | Production backend |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Task framing and measurement (`task_framing`) | any | none | 2 | 0 | 14 | 0 |
| Collaborative retrieval (`collaborative_retrieval`) | retrieval | interaction_graph, item_catalog | 4 | 5 | 5 | 0 |
| Item embeddings and graph retrieval (`item_representation`) | retrieval | item_catalog, interaction_graph | 4 | 2 | 2 | 0 |
| Two-tower matching (`two_tower_matching`) | retrieval | item_catalog, user_features | 3 | 3 | 3 | 0 |
| Interest and sequence retrieval (`interest_retrieval`) | retrieval | event_sequence, item_catalog | 2 | 3 | 3 | 0 |
| Streaming retrieval and indexing (`streaming_index`) | retrieval | item_catalog, streaming_updates | 2 | 0 | 0 | 0 |
| Feature representation and interactions (`feature_interactions`) | ranking, policy | tabular_features | 3 | 10 | 12 | 0 |
| Behavior sequence ranking (`sequence_ranking`) | ranking, policy | event_sequence | 1 | 3 | 3 | 0 |
| Multiple outcomes and task dependence (`multiobjective`) | ranking, policy, advertising | multiple_outcomes | 3 | 5 | 5 | 0 |
| Scenario and context specialization (`multiscenario`) | ranking, policy, advertising | scenario_context | 2 | 4 | 4 | 0 |
| Slate reranking and diversity (`slate_reranking`) | reranking | candidate_slates | 2 | 2 | 2 | 0 |
| Feedback bias and causal support (`feedback_bias`) | retrieval, ranking, reranking, policy, advertising | assignment_or_exposure_propensity | 1 | 0 | 0 | 0 |
| Cold-start and sparse entities (`cold_start`) | retrieval, ranking, reranking, policy | cold_start_segments | 1 | 0 | 0 | 0 |
| Generative representations and tokenization (`generative_foundations`) | generation | item_tokens, event_sequence | 5 | 0 | 0 | 0 |
| Scaling and hardware-aware architecture (`scaling_architecture`) | ranking, generation | event_sequence, compute_budget | 5 | 1 | 1 | 0 |
| End-to-end generation for recommendation, search and ads (`generative_applications`) | generation, search, advertising | generative_targets | 3 | 0 | 0 | 0 |
| Semantic and reasoning recommendation (`reasoning_recommendation`) | generation, ranking | semantic_item_context | 3 | 0 | 0 | 0 |
| Diffusion augmentation and recommendation (`diffusion_recommendation`) | generation, ranking | event_sequence, item_catalog | 3 | 0 | 0 | 0 |
| Offline-online system lifecycle (`production_lifecycle`) | any | deployment_context | 6 | 0 | 0 | 50 |

The [catalog JSON](../src/model_evo_harness/data/catalog.json) links every family to its exact upstream paths. It also states a research question, controlled experiment and likely failure mode. Model modules are *upstream examples*, not bundled implementations. Families without a model module come from documentation. The harness can still propose a new direction outside these families.

The [method cards](../src/model_evo_harness/data/method_cards.json) add one independently written card for each of the 38 upstream model modules. A card records:

- `id`: the method name used in proposals, such as `fm` or `din`
- `family_id`: the parent research family
- `source_model`: the pinned upstream model path
- `mechanism`: the portable modeling idea to test
- `requires`: the host data contracts needed before execution
- `comparison`: the controlled ablation or baseline comparison
- `failure_signals`: evidence that should reject or weaken the hypothesis
- `implementation_boundary`: what the upstream example does and what the host must still provide

`coverage_report` checks that every upstream model path is represented by exactly one family entry and exactly one method card, and that every method card belongs to the family that lists the same source model.

## How coverage affects an experiment

A task adapter declares its stage and capabilities from actual available data. The harness marks each family and each method card `ready`, `needs_data`, or `other_stage` and gives a reason. `ready` means the inputs are present, not that a model is installed or that the hypothesis is sound. If a selected known family or method needs absent inputs, the Agent must request data or choose another experiment. For a new direction, it may omit `family_id` and `method_id`, then must still name existing input fields and a falsifiable comparison.

The mapping reflects FunRec's retrieval, ranking, reranking, bias/cold-start, generative, scaling, reasoning, diffusion and production chapters. A binary coupon dataset generally has no item catalog, sequence, slate or generative target, so CouponEvo cannot execute those categories. The independent package remains applicable when a different host adapter supplies those contracts and evaluators.

## Source and license

FunRec's README and [license metadata](https://github.com/datawhalechina/fun-rec/blob/master/pyproject.toml) specify CC BY-NC-SA 4.0 for its work. This project records source paths and writes its own guidance, method cards and code under Apache-2.0; it does not copy FunRec implementations or chapter prose.
