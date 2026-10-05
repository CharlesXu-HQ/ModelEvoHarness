# ModelEvoHarness

[中文](README.zh-CN.md) · [Local technical knowledge](docs/knowledge/README.md) · [Model references](docs/models.md) · [Task adapter](docs/adapter-contract.md)

ModelEvoHarness is an **Agent-led offline experiment harness** for recommendation, search, advertising, and marketing. The Agent sees a frozen task, prior trials, and local technical knowledge; it proposes a falsifiable change, runs it through the host's evaluator, then records what the result supports. Model selection is only one direction. Losses, sampling, hard-example mining, optimization, calibration, feature representations, and decision rules can be tested under the same fixed protocol.

## What an algorithm engineer gets

| Component | Purpose |
| --- | --- |
| [22 local technical guides](docs/knowledge/README.md) | Independently written mechanisms, input contracts, controlled comparisons, failure signals, and feature-gap interpretation for all catalog families. The Agent reads the applicable guide from the installed package without fetching another repository. |
| 44 method cards and [structure patterns](docs/multi-source-guidance.md) | State *when* a method may address a measured bottleneck. A `ready` data contract is not a recommendation or an implementation claim. |
| [Training patterns](src/model_evo_harness/data/training_patterns.json) | Put loss, negative mining, sample weighting, regularization, calibration, and augmentation alongside architecture proposals. |
| [44 direct-framework model cores](docs/models.md) | Every method card has independently written PyTorch and TensorFlow code, with generic TwoTower as one additional structure. The [manifest](src/model_evo_harness/data/model_implementations.json) gives exact class paths. Outputs include logits, retrieval scores, probabilities and representations; the host supplies the appropriate loss and evaluator. No model wrapper is required. |
| Experiment journal | Keeps the task/dataset/protocol fingerprint, hypotheses, evaluated trials, failures, and separate technical and business experience. A changed dataset or protocol starts a new evidence context. |

The [source attribution](docs/research/source-attribution.md) records the upstream work that informed the research map. The explanations and model code in this repository are original; upstream prose and code are not mirrored here. Coverage is explained in [research source coverage](docs/source-coverage.md).

## The experiment loop

1. Read the task's available fields, objective, evaluation protocol, and previous trials. Diagnose the current failure using measured evidence and the applicable local guides.
2. Propose one mechanism, a stable control, an expected result, and a rejection condition. The host adapter trains the candidate and evaluates it on the same validation definition.
3. Compare prediction and decision metrics with uncertainty, then record **technical experience** separately from **business experience**. A business claim must cite a measured `business_observations` entry supplied by the host; otherwise it is `not_observable`.
4. Continue, stop, or request a human dataset change. The Agent records `future_feature_suggestions` after two distinct evaluated mechanisms, or immediately when the host declares an explicit business prerequisite. A terminal data request cites the completed trials and evidence, or the matching `domain_requirements` item.

The task adapter owns raw data, model training, GPU selection, splits, objective, uncertainty calculation, final holdout, and action constraints. The harness validates the experiment record and applicability; it does not infer business semantics from column names or assert a model gain. See the [adapter contract](docs/adapter-contract.md).

## Integrate a task

Requires Python 3.12 or newer. Install the package and inspect its catalog:

```bash
pip install git+https://github.com/CharlesXu-HQ/ModelEvoHarness.git
model-evo-harness catalog-check
```

Implement `snapshot()`, `baseline()` and `evaluate()` in your task adapter, then provide an Agent with `propose()` and `reflect()` or use the included OpenAI-compatible provider:

```bash
export MODEL_AGENT_API_KEY=...  # keep credentials out of repository files
model-evo-harness run \
  --task my_project.experiments:task \
  --agent-config agent.json \
  --output runs/my-task-v1 \
  --max-steps 4
```

`agent.json` defines `provider_url`, `model`, `api_key_env`, `iteration_effort` (`high`), and `review_effort` (`max`). Python hosts can call `run_search(task, agent, output=..., catalog=load_catalog(), max_steps=4)`. The framework does not require a particular model provider or training framework; the optional model references require PyTorch or TensorFlow only when imported.

[CouponEvo](https://github.com/CharlesXu-HQ/CouponEvo) is the first host integration. It supplies a randomized coupon task, candidate execution, and policy evaluation. The shared harness gives it an experiment protocol and technical knowledge. Adding this harness or its reference models does not itself demonstrate uplift or net-value improvement.

## License

ModelEvoHarness is [Apache-2.0](LICENSE). [Source attribution and reuse boundaries](docs/research/source-attribution.md) explain how its original material relates to FunRec, DeepCTR, Torch-RecHub, RecBole, and other research sources.
