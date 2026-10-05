# Local technical knowledge

These 22 substantive, independently written family guides are packaged with ModelEvoHarness. The Agent can read them without fetching upstream repositories. They explain the mechanism, data contract, controlled experiment, failure condition, and when a human feature request is justified. All 44 method cards also have direct PyTorch and TensorFlow [model cores](../models.md); a runnable structure still needs the host's preprocessing, loss, sampling and evaluator. Source links provide attribution, not the guide content or copied code.

## Catalog families

- [Task framing and measurement](../../src/model_evo_harness/knowledge/task_framing.md)
- [Collaborative retrieval](../../src/model_evo_harness/knowledge/collaborative_retrieval.md)
- [Item embeddings and graph retrieval](../../src/model_evo_harness/knowledge/item_representation.md)
- [Two-tower matching](../../src/model_evo_harness/knowledge/two_tower_matching.md)
- [Interest and sequence retrieval](../../src/model_evo_harness/knowledge/interest_retrieval.md)
- [Streaming retrieval and indexing](../../src/model_evo_harness/knowledge/streaming_index.md)
- [Feature representation and interactions](../../src/model_evo_harness/knowledge/feature_interactions.md)
- [Behavior sequence ranking](../../src/model_evo_harness/knowledge/sequence_ranking.md)
- [Multiple outcomes and task dependence](../../src/model_evo_harness/knowledge/multiobjective.md)
- [Scenario and context specialization](../../src/model_evo_harness/knowledge/multiscenario.md)
- [Slate reranking and diversity](../../src/model_evo_harness/knowledge/slate_reranking.md)
- [Feedback bias and causal support](../../src/model_evo_harness/knowledge/feedback_bias.md)
- [Cold-start and sparse entities](../../src/model_evo_harness/knowledge/cold_start.md)
- [Generative representations and tokenization](../../src/model_evo_harness/knowledge/generative_foundations.md)
- [Scaling and hardware-aware architecture](../../src/model_evo_harness/knowledge/scaling_architecture.md)
- [End-to-end generation for recommendation, search and ads](../../src/model_evo_harness/knowledge/generative_applications.md)
- [Semantic and reasoning recommendation](../../src/model_evo_harness/knowledge/reasoning_recommendation.md)
- [Diffusion augmentation and recommendation](../../src/model_evo_harness/knowledge/diffusion_recommendation.md)
- [Offline-online system lifecycle](../../src/model_evo_harness/knowledge/production_lifecycle.md)
- [Adaptive piecewise response models](../../src/model_evo_harness/knowledge/adaptive_segmentation.md)
- [Prediction-to-decision mapping](../../src/model_evo_harness/knowledge/decision_mapping.md)
- [Typed knowledge-graph retrieval](../../src/model_evo_harness/knowledge/knowledge_graph_retrieval.md)

## Cross-cutting training and interpretation

- [Training objectives and losses](../../src/model_evo_harness/knowledge/training_objectives.md)
- [Negative sampling and hard-example mining](../../src/model_evo_harness/knowledge/sampling_and_hard_examples.md)
- [Optimization, regularization, and training controls](../../src/model_evo_harness/knowledge/optimization_and_regularization.md)
- [When an Agent should request human feature work](../../src/model_evo_harness/knowledge/feature_gap_decisions.md)
- [From experiments to business insight](../../src/model_evo_harness/knowledge/business_insight_synthesis.md)

The complete original-source path mapping and reuse boundary are in [source attribution](../research/source-attribution.md). The [implementation manifest](../../src/model_evo_harness/data/model_implementations.json) lists exact model class paths in both frameworks. The catalog records machine-readable applicability; these guides carry the technical explanation and do not claim measured performance gains.

- [Choosing the next experiment](../../src/model_evo_harness/knowledge/exploration_strategy.md): symptom-driven exploration, stage-specific sampling, controlled comparisons and current implementation boundaries.

- [Additional structural hypotheses](../../src/model_evo_harness/knowledge/structure_extensions.md): non-overlapping field, sequence and interest mechanisms; explicitly marked design-only.
- [Randomized marketing experiments](../../src/model_evo_harness/knowledge/causal_policy_experiments.md): estimator vs network changes, cross-fitting, observable cost and policy contracts.
