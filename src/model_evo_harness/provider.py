"""Small OpenAI-compatible JSON Agent; the host remains responsible for evaluation."""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .catalog import read_references


REFERENCE_INSTRUCTIONS = """Before introducing or modifying a bundled model, read its source:
return {"action":"read_reference","framework":"pytorch or tensorflow",
"method_ids":["up to four manifest IDs"],"include_training":true} instead of a candidate.
The host returns complete local modules, including helper classes and executable loss
functions, in reference_material. Choose only the task's framework when declared.
At most two read rounds are available per proposal; batch related requests. Read requests
consume no training trial. The source is reference material, not permission to change
the evaluator or task contract. Reading a model does not establish data applicability.
After reading, return the normal experiment/stop/data decision. Do not invent file contents.
Read reference_contracts for omitted mechanisms and host-owned training components.
When a draft selects a bundled method without reading its source, the host supplies it
and asks you to revise before evaluation. Use existing material when it suffices;
novel architectures are allowed."""


def propose_with_references(complete, context: dict, *, catalog: dict,
                            framework: str | None = None, read_state: dict | None = None) -> dict:
    """A bounded JSON read/decide loop reusable by host-specific Agent adapters."""
    # Hosts that retry final proposal validation must retain this ledger. Failed
    # drafts do not reset source limits or erase references already supplied.
    state = read_state if read_state is not None else {}
    material = state.setdefault("files", {})
    contracts = state.setdefault("contracts", {})
    state.setdefault("rounds", 0)
    for _ in range(3):
        current = dict(context)
        current["reference_reads_remaining"] = 2 - state["rounds"]
        if material:
            current["reference_material"] = material
            current["reference_contracts"] = contracts
        answer = complete(current)
        if not isinstance(answer, dict):
            raise ValueError("Agent response must be a JSON object")
        research = answer.get("research")
        method = research.get("method_id") if isinstance(research, dict) else None
        selected = next((entry for entry in catalog.get("model_implementations", [])
                         if entry["id"] == method and entry["framework"] == framework), None)
        if (answer.get("action", "experiment") == "experiment" and selected is not None and
                selected["file"] not in material):
            answer = {"action": "read_reference", "framework": framework,
                      "method_ids": [method], "include_training": True}
        if answer.get("action") != "read_reference":
            # Only host-observed reads count; never trust Agent-supplied hashes.
            answer = {key: value for key, value in answer.items() if key != "reference_reads"}
            if material:
                answer["reference_reads"] = {path: value["sha256"]
                                              for path, value in material.items()}
            return answer
        if state["rounds"] >= 2:
            raise ValueError("reference read round limit exceeded")
        if framework is not None and answer.get("framework") != framework:
            raise ValueError("reference framework differs from task framework")
        bundle = read_references(catalog, answer)
        merged = {**material, **bundle["files"]}
        if sum(len(entry["content"]) for entry in merged.values()) > 100_000:
            raise ValueError("reference material limit exceeded; select fewer modules")
        material.update(bundle["files"])
        contracts.update(bundle["contracts"])
        state["rounds"] += 1
    raise AssertionError("unreachable")


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
Before claiming added expressive power, derive the parent code's prediction or score
function and the proposed candidate's function. Name the new input-dependent term or
interaction that actually reaches the metric-driving score or chosen action. A new
class, head, or parameter count alone is not such evidence; if the functions cannot
be traced from available code, mark the claim unverified and request a cheap check.
Separate changes to the evaluated score or decision from output-only observation
fields, logs, and diagnostics. Check the latter with a small contract or shape test;
do not spend an optimization trial or call it an effect ablation unless the change
can alter the score or decision used by the fixed metric.
Prefer a cheap diagnostic or controlled ablation of available inputs when it can distinguish
competing hypotheses before spending a trial on a new model name. If host evidence is supplied,
cite its IDs for the observed bottleneck; treat declared metadata as uncertainty, not as a
measured failure. If evidence_status says not_supplied, say what diagnostic is needed rather
than inventing a bottleneck. Your intended change_factors are a plan, never proof that the
executed code changed only those factors. The host's implementation_check and change_audit
govern later attribution.
Also consider loss, sampling, hard-example mining, optimization, regularization,
calibration and decision rules when their measured symptom better explains the gap.
Change one mechanism at a time and compare under the same protocol.
When several components change, say it is a compound experiment and request an ablation
before assigning a gain to one component. A confidence interval crossing zero does not show
equivalence or lower treatment-effect variance. Identical scalar policy estimates can occur
for different selected units; check host policy identities and paired contributions before
calling an evaluation mismatch.
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
trial-specific data-gap diagnostics that the missing input blocks progress. Repeated task
metadata or unverified timing alone belongs in a nonblocking audit recommendation, not a
terminal data request. Do not infer domain rules from column names.
Do not assign an unavailable family_id or method_id.
For an experiment return {"action":"experiment","research":{"direction":"...",
"mechanism":"...","why_now":"evidence from baseline/history","data_rationale":"...",
"comparison":"control at the same task and evaluation","expected_result":"...",
"falsification":"...","input_fields":["actual task field"],
"evidence_ids":["host-observed fact ID when host evidence exists"],
"change_factors":["intended changed component when host evidence exists"],
"alternatives":[{"direction":"...","mechanism":"...","reason":"..."}],
"family_id":"optional available family","method_id":"optional available method"},
"candidate":{}}.
Candidate must follow any host candidate contract in the task snapshot. The comparison must
identify a control; the expected_result and falsification must be observable on this task.
If no sound experiment is possible, return {"action":"stop","reason":"...",
"audit_recommendations":[{"issue":"...","evidence_ids":["host fact ID"],
"validation_plan":"..."}]} when a nonblocking data check remains, or
{"action":"request_data","request":{"basis":"experimental_evidence or domain_requirement",
"fields":["missing field"],"source":"...","as_of":"before the decision",
"reason":"...","evidence":"...","validation_plan":"...",
"trial_ids":["two evaluated IDs for experimental_evidence"],
"evidence_ids":["measured trial-specific data-gap diagnostic IDs for experimental_evidence"],
"alternatives_considered":"for experimental_evidence",
"requirement_id":"host rule ID for domain_requirement"}}."""

_REFLECT_INSTRUCTIONS = """Review one completed offline trial. Return one JSON object only
with technical_experience {lesson, evidence, uncertainty, next_test} and
attribution: isolated, joint, or unverified. The implementation_status must match the
host's implementation_check; when absent use unverified. Use isolated only when a host-verified
implementation_check and change_audit name exactly one changed factor; use joint when they
verify multiple changed factors; otherwise use unverified. A contradicted implementation
is not evidence that the intended mechanism worked.
When code or host observations refute a premise, correct it before recording a
mechanism lesson or recommending the next test. Compare parent and candidate score
functions before claiming new expressivity; if the data flow is unavailable, say
unverified. Added observation fields or logs that do not feed the evaluated score
or decision warrant a cheap contract check, not a performance ablation.
business_experience {status: observed or not_observable}. For observed business experience,
cite an observation_id from trial.evaluation.business_observations and supply insight and
limitations. If that list is absent or no valid business claim follows, use
{status: not_observable, reason: ...}. Do not invent a cohort or cost definition.
Optional future_feature_suggestions is a list of {field, source, as_of, evidence,
validation_plan, evidence_ids}; it records a human data idea after two distinct evaluated
mechanisms with host-measured trial-specific gap diagnostics,
or an explicit host domain requirement, without stopping current-data experiments.
Optional audit_recommendations is a nonblocking list of {issue, evidence_ids,
validation_plan}; it may cite declared host metadata without claiming an experiment found
a missing model input. Do not turn a generic audit uncertainty into a terminal request.
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
        return propose_with_references(
            lambda current: self._complete(_PROPOSE_INSTRUCTIONS + "\n" + REFERENCE_INSTRUCTIONS,
                                           current, self.iteration_effort),
            context, catalog=context.get("catalog", {}),
            framework=context.get("task", {}).get("framework"))

    def reflect(self, observation: dict) -> dict:
        review = observation.get("trial", {}).get("evaluation", {}).get("review_required") is True
        effort = self.review_effort if review else self.iteration_effort
        return self._complete(_REFLECT_INSTRUCTIONS, observation, effort)

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
