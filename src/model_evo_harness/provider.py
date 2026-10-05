"""Small OpenAI-compatible JSON Agent; the host remains responsible for evaluation."""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_PROPOSE_INSTRUCTIONS = """You lead one offline model-research decision. Return one JSON object only.
Use the task, local knowledge, catalog families, method_cards, structure_patterns,
training_patterns, decision_checks, model_api and their applicability reports, baseline,
and measured step history in the user JSON. Local guides contain the working material;
source paths provide attribution. The model implementation manifest and model_api give
framework-specific import paths and signatures. Reuse those modules when the host's
candidate contract permits imports; inspect each model's input and output contract.
Keep the task's objective, constraints, split, and metric fixed. A proposal is a falsifiable
hypothesis, not a proven outcome. Do not claim an improvement before the host evaluates it.
The catalog is research guidance, not a closed model list; a new direction may omit family_id
and method_id. Method cards describe controlled ablations, not installed models.
Before changing model structure, connect a pattern's when_to_try to a measured bottleneck,
name the required data evidence, and use its controlled comparison and reject_if signal.
Do not choose a larger structure solely because it is available in the catalog.
Also consider loss, sampling, hard-example mining, optimization, regularization,
calibration and decision rules when their measured symptom better explains the gap.
Change one mechanism at a time and compare under the same protocol.
Preserve the host's evaluation_protocol when supplied: unit, split, metric, candidate universe,
label/negative provenance and time cutoff. Do not compare scores from different protocols.
Distinguish prediction quality from a decision rule over predictions; test one mechanism at a time.
Apply decision_checks only when their applicability status is ready; not_triggered means
the task has not declared the triggering condition, not that the data should be invented.
Do not assume that a field name or dtype proves its semantics. Use only task.fields as inputs.
With a rich fixed dataset, first test sound hypotheses using available pre-decision fields.
Record a missing feature as a future_feature_suggestion only after two completed trials
with distinct mechanisms, or when an explicit host domain requirement identifies it.
Use request_data immediately only when task.domain_requirements contains an explicit
host-supplied rule that names the missing field, source and timing. Otherwise a terminal
request needs at least two completed experiments testing distinct mechanisms and measured
evidence that the missing input blocks progress. Do not infer domain rules from column names.
Do not assign an unavailable family_id or method_id.
For an experiment return {"action":"experiment","research":{"direction":"...",
"mechanism":"...","why_now":"evidence from baseline/history","data_rationale":"...",
"comparison":"control at the same task and evaluation","expected_result":"...",
"falsification":"...","input_fields":["actual task field"],
"alternatives":[{"direction":"...","mechanism":"...","reason":"..."}],
"family_id":"optional available family","method_id":"optional available method"},
"candidate":{}}.
Candidate must follow any host candidate contract in the task snapshot. The comparison must
identify a control; the expected_result and falsification must be observable on this task.
If no sound experiment is possible, return {"action":"stop","reason":"..."} or
{"action":"request_data","request":{"basis":"experimental_evidence or domain_requirement",
"fields":["missing field"],"source":"...","as_of":"before the decision",
"reason":"...","evidence":"...","validation_plan":"...",
"trial_ids":["two evaluated IDs for experimental_evidence"],
"alternatives_considered":"for experimental_evidence",
"requirement_id":"host rule ID for domain_requirement"}}."""

_REFLECT_INSTRUCTIONS = """Review one completed offline trial. Return one JSON object only
with technical_experience {lesson, evidence, uncertainty, next_test} and
business_experience {status: observed or not_observable}. For observed business experience,
cite an observation_id from trial.evaluation.business_observations and supply insight and
limitations. If that list is absent or no valid business claim follows, use
{status: not_observable, reason: ...}. Do not invent a cohort or cost definition.
Optional future_feature_suggestions is a list of {field, source, as_of, evidence,
validation_plan}; it records a human data idea after two distinct evaluated mechanisms,
or an explicit host domain requirement, without stopping current-data experiments.
Use the task's fixed objective and the baseline, history, and trial in the user JSON.
Distinguish measured outcomes from hypotheses; do not call a lift significant unless the
observation includes uncertainty evidence. Do not infer unavailable field semantics or
claim success on an unseen final holdout."""


class OpenAICompatibleAgent:
    """Send task-level context to an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, provider_url: str, api_key: str, model: str, *,
                 thinking: str = "omit", iteration_effort: str = "high",
                 review_effort: str = "max", timeout: float = 120.0):
        if not isinstance(provider_url, str):
            raise ValueError("provider_url must be an HTTP(S) URL")
        try:
            parsed = urlsplit(provider_url)
            host = parsed.hostname
            parsed.port
        except ValueError:
            raise ValueError("provider_url must be an HTTP(S) URL") from None
        if (parsed.scheme not in ("http", "https") or not parsed.netloc or not host
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("provider_url must be an HTTP(S) URL without credentials or query")
        if parsed.scheme == "http" and host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("provider_url requires HTTPS except for loopback")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a nonempty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if thinking not in ("enabled", "omit"):
            raise ValueError("thinking must be enabled or omit")
        if iteration_effort not in ("high", "max") or review_effort not in ("high", "max"):
            raise ValueError("iteration_effort and review_effort must be high or max")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")

        self.url = provider_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self._api_key = api_key
        self.model = model
        self.thinking = thinking
        self.iteration_effort = iteration_effort
        self.review_effort = review_effort
        self.timeout = timeout

    def propose(self, context: dict) -> dict:
        return self._complete(_PROPOSE_INSTRUCTIONS, context, self.iteration_effort)

    def reflect(self, observation: dict) -> dict:
        return self._complete(_REFLECT_INSTRUCTIONS, observation, self.review_effort)

    def _complete(self, instructions: str, context: dict, effort: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False, sort_keys=True)},
            ],
            "response_format": {"type": "json_object"},
            "reasoning_effort": effort,
        }
        if self.thinking == "enabled":
            payload["thinking"] = {"type": "enabled"}
        request = Request(self.url, data=json.dumps(payload).encode("utf-8"), method="POST",
                          headers={"Authorization": f"Bearer {self._api_key}",
                                   "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = response.read()
        except HTTPError as error:
            raise RuntimeError(f"provider returned HTTP {error.code}") from None
        except Exception:
            raise RuntimeError("provider request failed") from None

        try:
            envelope = json.loads(data)
            choice = envelope["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                raise ValueError("provider response incomplete")
            content = choice["message"]["content"]
            answer = json.loads(content)
        except (ValueError, TypeError, KeyError, IndexError):
            raise ValueError("provider returned invalid JSON response") from None
        if not isinstance(answer, dict):
            raise ValueError("provider must return a JSON object")
        return answer
