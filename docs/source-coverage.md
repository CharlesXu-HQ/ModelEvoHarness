# Research source coverage and applicability

This page audits the pinned source lineage. The working technical material is the [local, independently written knowledge library](knowledge/README.md). No FunRec prose or implementation is copied into this package. The upstream [commit](https://github.com/datawhalechina/fun-rec/commit/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572) is `5f7bd84d4403b5f92b6bedbd31e3984ef00a4572` and its Git tree is `84a4e63da7d7cb0ca8ea7e47128174535f0ab40a`. The [inventory](../src/model_evo_harness/data/upstream_inventory.json) records 55 substantive chapter pages, 38 model modules, 54 supporting core modules and 50 production-backend Python modules. `coverage_report` checks that each maps to exactly one family. FunRec's chapter on bias and cold start is included alongside the newer outline in its top-level README. Frontend assets and image files are recorded by the upstream tree but have no model-iteration mechanism, so they are outside this code inventory.

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

The table above describes the **19 FunRec-derived families**. Three additional local guides cover adaptive segmentation, prediction-to-decision mapping and typed knowledge-graph retrieval, for **22 families total**. Each family has a packaged Markdown guide with its mechanism, required data, controlled experiment, rejection signals and feature-gap interpretation. The 38 FunRec-derived method cards plus six supplemental cards give **44 method cards**; all 44 are discussed in their family guides. The complete pinned paths are in [source attribution](research/source-attribution.md). Paths are attribution and audit evidence, not the material an Agent must fetch before proposing an experiment.

The catalog's `local_guide` field resolves a packaged guide; `load_guide(family_id)` reads it. The Agent context carries guides for families whose declared task prerequisites are ready. The [training patterns](../src/model_evo_harness/data/training_patterns.json) also let the Agent investigate losses, sampling, hard-example mining, optimization, regularization, calibration and augmentation without inventing a new model name. Every pattern states evidence, a controlled comparison and a rejection signal.

All 44 method cards have independent [PyTorch and TensorFlow reference implementations](models.md). Generic TwoTower is an additional structure, not one of the cards. The [implementation manifest](../src/model_evo_harness/data/model_implementations.json) records the exact module and symbol for each framework. Framework modules import no other model library. A card's upstream `source_model` is an attribution path, never imported code or evidence of measured performance.

`coverage_report` still checks that every pinned FunRec document, model and support/project module maps to exactly one appropriate family, and every FunRec model path maps to one method card. This audit preserves completeness of the research map without reproducing upstream files.

## How coverage affects an experiment

A task adapter declares its stage and capabilities from actual available data. The harness marks each family and each method card `ready`, `needs_data`, or `other_stage` and gives a reason. `ready` means the host declared the inputs; it does not mean the hypothesis is sound. Method applicability lists reference frameworks, and the Agent receives the ready models' local API signatures. If a known family or method needs absent inputs, the Agent can choose another viable experiment; a terminal human data request must meet the evidence rules in the [adapter contract](adapter-contract.md). For a new direction, it may omit `family_id` and `method_id`, then must still name existing input fields and a falsifiable comparison.

The mapping reflects FunRec's retrieval, ranking, reranking, bias/cold-start, generative, scaling, reasoning, diffusion and production chapters. A binary coupon dataset generally has no item catalog, sequence, slate or generative target, so CouponEvo cannot execute those categories. The independent package remains applicable when a different host adapter supplies those contracts and evaluators.

## Source and license

FunRec's README and [license metadata](https://github.com/datawhalechina/fun-rec/blob/master/pyproject.toml) specify CC BY-NC-SA 4.0 for its work. ModelEvoHarness is Apache-2.0: its guides and direct-framework model code are independently written. The [attribution record](research/source-attribution.md) retains pinned origins while keeping upstream prose and code outside the package.
