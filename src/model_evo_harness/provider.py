"""Small OpenAI-compatible JSON Agent; the host remains responsible for evaluation."""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .catalog import read_references


REFERENCE_INSTRUCTIONS = """Before introducing or modifying a bundled model, read its source:
return {"action":"read_reference","framework":"pytorch or tensorflow",
"method_ids":["up to four manifest IDs"],"include_training":true,"include_composition":true} instead of a candidate.
The host returns complete local modules, including helper classes and executable loss
functions, in reference_material. Use include_composition=true to read generic parallel
branch wiring without selecting a named model. Choose only the task's framework when declared.
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
        requested = [method] if isinstance(method, str) else []
        design = research.get("model_design") if isinstance(research, dict) else None
        if isinstance(design, dict) and isinstance(design.get("components"), list):
            requested.extend(component["reference_method_id"] for component in design["components"]
                             if isinstance(component, dict) and
                             isinstance(component.get("reference_method_id"), str))
        unread = [entry["id"] for entry in catalog.get("model_implementations", [])
                  if entry["id"] in requested and entry["framework"] == framework and
                  entry["file"] not in material]
        parallel = (design.get("horizontal_expansion") or {}) if isinstance(design, dict) else {}
        unread_composition = (isinstance(parallel, dict) and bool(parallel.get("groups")) and
                              framework in ("pytorch", "tensorflow") and
                              f"models/{framework}/composition.py" not in material)
        if answer.get("action", "experiment") == "experiment" and (unread or unread_composition):
            answer = {"action": "read_reference", "framework": framework,
                      "method_ids": unread[:4], "include_training": True,
                      "include_composition": unread_composition}
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



COMPOSITION_INSTRUCTIONS = """Prefer evidence-driven improvements inside the current compatible backbone:
read the local structure and training implementations, then edit/compose their actual code
inside the candidate's encoder, interaction layers, heads, objectives or training loop.
The estimator (e.g. a causal meta-learner) and the representation backbone are separate axes.
A catalog name is a reference, not the unit of progress; custom modules and combinations
are welcome. Few tabular fields do not prove that interaction or loss work is exhausted,
and neither cardinality nor field count establishes sequence, item or semantic inputs.
Do not impose a fixed number of local trials. Switch when evidence, data suitability or
expected information per budget favors it, and explain why a local alternative is weaker.

Horizontal expansion is a general research axis for every compatible subnetwork, not a
list of named models. Inspect task.feature_groups, typed fields and prior branch evidence.
Distinguish different behavior streams from aligned attributes of the same events; the
latter may belong in one event representation. Unknown semantics remain unknown.
Actively propose parallel instances when input groups, complementary representations or
measured bottlenecks support them. Different instances may use the same reference model
or different computations. Compare shared versus independent parameters and how branches
feed the existing prediction path. Do not default to a single instance or only wider layers.
The number and type of branches are open; do not require one branch per field or extra
branches without a testable rationale. Consider a small controlled expansion before a
backbone switch or a missing-feature request when current inputs support it.

When task.horizontal_expansion_required is true, include model_design.horizontal_expansion:
{decision:expand|defer|not_applicable, rationale, comparison_plan, groups:[{id,
branch_ids:[component IDs], fusion_id:component ID, parameter_sharing:[{
component_ids:[branch IDs], code_sections:[actual shared module paths], rationale}]}].
expand needs at least one group with two distinct branches; defer may keep existing groups;
not_applicable has no groups. Explain a concrete data/evidence/budget reason for deferral or
inapplicability and the next discriminating test. Do not silently skip this assessment.
Each branch and fusion is a separate component instance with instance_path and
output_contract (tensor shape, scale and mask where relevant), alongside input_fields and
code_sections naming its forward/call path. Bind each instance to its actual fields.
Distinct instance paths can alias a shared module; parameter_sharing must name the shared
part explicitly. An empty list declares no intended sharing between the listed branches.
Connections must be acyclic; a nested branch group may feed another group's fusion.
Keep existing groups in later recipes even when only the loss changes. Retain requires
unchanged branch/fusion roles, connected instances and parameter-sharing declarations;
rewiring or changing sharing needs adapt or retest for the affected group components.
These are declared connections, not proof of execution, tensor compatibility or shared weights.
Read the generic models/<framework>/composition.py with include_composition=true; it
supports arbitrary native branches and a host-written fusion module without a model whitelist.
If using representation-level fusion, expose the needed representations; logits are not
interchangeable with embeddings. Verify every branch reaches the score/loss, shared objects
are actually tied, independent objects are not accidentally tied, and trainable parameters
are registered in the optimizer. Missing gradient/runtime evidence remains unverified.

Compare against the same-input unsplit/pooled control, a matched-capacity control where
feasible, and branch/fusion ablations. Keep split, sampler, objective, training schedule and
budget fixed or declare their changes as joint factors. Removing a branch and retraining
shared weights can change all branches; that is not an isolated causal effect of one branch.
Assess each branch and fusion separately using the existing component_assessments, recording
input applicability, sharing, fusion interactions, uncertainty and the next test. A whole-model
gain does not establish every branch's value. Keep these lessons bound to task/dataset version.

When task.model_design_required is true, every experiment needs research.model_design:
{estimator_id, backbone_id, estimator, backbone, change_scope: initialize|local|switch, parent_trial_id,
rationale, data_fit, comparison_plan, components: [{id, mechanism, code_sections:[actual
candidate class/function names], input_fields:[task fields], required_capabilities:[task
capabilities], reference_method_id:optional bundled method ID}], inheritance:[{
source_trial_id, component_id, decision:retain|adapt|drop|retest, reason, compatibility,
validation_plan, target_component_id:required unless drop}]}.
estimator_id/backbone_id are short, stable identities, not architecture descriptions or
catalog restrictions. estimator/backbone are editable descriptions of the current recipe.
For local edits, omit both IDs to let the host inherit them, or copy the exact parent IDs;
update descriptions and components to explain the change. Removing a residual branch or
changing a loss does not by itself require a new backbone ID. Older records without IDs
receive deterministic identities when read; never invent or rewrite their history.
Use initialize only before a tracked design exists (parent_trial_id may be null for an
untracked seed); local keeps estimator_id/backbone_id, switch changes at least one identity.
Describe the whole current recipe, including accumulated
training and representation changes; code_sections must locate their implementation.
For local or switch, account for every parent component, and cite any other donor by its
host-supplied trial/component IDs. Read the donor code before claiming source reuse; metadata
alone is not copied code. Keep compatible improvements, adapt interfaces/objectives when
needed, drop incompatible or harmful parts with reasons, and retest uncertain or invalid
ideas as new hypotheses. A source without a matching component assessment may only be
dropped or retested; an inconclusive assessment never becomes proven by reuse. Reject invented data semantics and check shape, timing, output
scale, objective, sampling and fitting/cross-fitting boundaries. Do not carry fitted weights
or preprocessing state across splits. Transferring a design does not establish transfer gain.
Compare the previous recipe, a plain new-backbone control, and the selectively composed
candidate under the same protocol/budget when switching; if the budget cannot resolve all
factors, state the limitation and choose the most informative comparison. Joint gains do
not validate every component. Missing source/code evidence is uncertainty, not permission
to label an inherited component proven. Do not erase prior refinements by restarting with
initialize once tracked designs exist.

When reflecting on a model_design, include technical_experience.component_assessments,
one entry per current component: {component_id, outcome:promising|inconclusive|harmful|invalid,
evidence, compatibility_limits, next_test, attribution:unverified|joint|isolated}.
These are task/dataset-bound observations. Promising is exploratory. Attribution stays
unverified unless supported by the host change audit; isolated additionally requires that
the sole audited changed factor is that component ID. Failed/contradicted trials cannot
supply promising or harmful mechanism claims. Separate what happened to the whole recipe
from what has been established for any individual component."""

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

    requires_model_design = True
    requires_horizontal_expansion = True

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
            lambda current: self._complete(_PROPOSE_INSTRUCTIONS + "\n" + COMPOSITION_INSTRUCTIONS + "\n" + REFERENCE_INSTRUCTIONS,
                                           current, self.iteration_effort),
            context, catalog=context.get("catalog", {}),
            framework=context.get("task", {}).get("framework"))

    def reflect(self, observation: dict) -> dict:
        review = observation.get("trial", {}).get("evaluation", {}).get("review_required") is True
        effort = self.review_effort if review else self.iteration_effort
        return self._complete(_REFLECT_INSTRUCTIONS + "\n" + COMPOSITION_INSTRUCTIONS, observation, effort)

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
