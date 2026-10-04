# ModelEvoHarness

[中文](README.zh-CN.md) · [Research coverage](docs/source-coverage.md) · [Adapter contract](docs/adapter-contract.md)

ModelEvoHarness runs **Agent-led, falsifiable offline model experiments** across recommendation, search, advertising, and marketing. An Agent reads the fixed task and previous results, chooses a mechanism, states a control and predicted result, submits a candidate, then reflects on the measured outcome. The harness checks data prerequisites and records the experiment. Your task adapter trains the candidate and evaluates it under your own split, metrics and execution rules.

The research catalog maps 55 chapter pages, 38 model modules, 54 supporting modules and 50 production backend modules into 19 research families. Each model module has a method card describing its mechanism, required data contracts, controlled comparison, failure signals and implementation boundary. The Agent can choose a ready family or method card, or propose a direction outside the catalog. These are research references, not bundled model implementations.

## Why this exists

A model name is not an experiment. To tell whether a new architecture helps, the Agent must identify the bottleneck it expects to fix, compare against a stable control, use fields that exist before the decision, and name a result that would reject the hypothesis. ModelEvoHarness makes those steps part of the executable loop. It distinguishes a considered alternative from a completed trial and binds lessons to the exact task and dataset version.

| Boundary | Owner |
| --- | --- |
| Research coverage, family and method applicability checks, experiment design, Agent loop, journal and run-scoped task history | ModelEvoHarness |
| Dataset semantics, train/validation/test split, candidate execution, GPU use, objective and uncertainty calculation | Task adapter |
| Hypothesis, candidate artifact, data request and reflection | Agent |

## Use it

Requires Python 3.12 or newer. Install from the repository:

```bash
pip install git+https://github.com/CharlesXu-HQ/ModelEvoHarness.git
model-evo-harness catalog-check
```

Implement the small [task adapter](docs/adapter-contract.md), then run the loop with an importable task and an OpenAI-compatible provider:

```bash
export MODEL_AGENT_API_KEY=...  # set in your shell; do not put it in config or the repository
model-evo-harness run \
  --task my_project.experiments:task \
  --agent-config agent.json \
  --output runs/my-task-v1 \
  --max-steps 4
```

`agent.json` specifies `provider_url`, `model`, `api_key_env`, `iteration_effort` (`high`) and `review_effort` (`max`). A local Agent object can be supplied with `--agent module:symbol` instead. Python applications can call `run_search(task, agent, output=..., catalog=load_catalog(), max_steps=4)` directly. The library has no dependency on a specific model framework, provider, recommender dataset or CouponEvo.

Each task snapshot declares a stage, available capabilities and actual input fields. The catalog labels every family and method card `ready`, `needs_data` or `other_stage`, with a reason. For example, a tabular coupon trial can test some feature interaction methods while sequence ranking requires time-safe event histories. `ready` establishes input availability; it does not claim that a model implementation exists or will improve a metric.

The journal records baseline and trial scores, research designs, evaluation failures, reflections, selected best candidate, catalog digest and task identity. Resume fails if the dataset, objective, package implementation or catalog changes. History is scoped to the same run identity and dataset fingerprint. Keep final holdout data inside the task adapter; use validation results for iteration and reserve final evaluation for the selected candidate.

## First integration

[CouponEvo](https://github.com/CharlesXu-HQ/CouponEvo) is the first consumer. It supplies a randomized coupon task, PyTorch candidate sandbox and paired policy evaluation. ModelEvoHarness supplies the independent research catalog and design checks. The two projects are released and tested separately; CouponEvo's before/after results are reported as an integration benchmark, not as a guarantee of improvement.

## Source and license

The initial [research coverage](docs/source-coverage.md) is mapped to a pinned tree of [FunRec](https://github.com/datawhalechina/fun-rec). Its work is marked [CC BY-NC-SA 4.0](https://github.com/datawhalechina/fun-rec/blob/master/pyproject.toml). ModelEvoHarness independently writes its guidance, method cards and code under [Apache-2.0](LICENSE), with no upstream source code copied.
