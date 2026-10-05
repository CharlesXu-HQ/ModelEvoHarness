"""Small, task-owned offline experiment loop.

The host owns training, evaluation, and final holdout discipline. This module
only records validated Agent decisions and their observed offline results.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from .composition import composition_sources, validate_feature_groups
from .evidence import (cited_facts, validate_audit_recommendations,
                       validate_host_evidence)


PACKAGE_VERSION = "0.2.0"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_journal(path: Path, state: dict) -> None:
    payload = json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         indent=2).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".journal-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _snapshot(task: object) -> dict:
    snapshot = task.snapshot()
    if not isinstance(snapshot, dict):
        raise ValueError("task.snapshot() must return a dict")
    for key in ("task_id", "dataset_digest", "stage"):
        if not isinstance(snapshot.get(key), str) or not snapshot[key]:
            raise ValueError(f"task snapshot needs {key}")
    objective = snapshot.get("objective")
    if not isinstance(objective, dict) or objective.get("direction") not in ("max", "min"):
        raise ValueError("task objective needs direction 'max' or 'min'")
    if not isinstance(snapshot.get("fields"), list) or any(
            not isinstance(field, str) or not field for field in snapshot["fields"]):
        raise ValueError("task snapshot needs a list of field names")
    if not isinstance(snapshot.get("capabilities"), list) or any(
            not isinstance(capability, str) or not capability for capability in snapshot["capabilities"]):
        raise ValueError("task snapshot needs a list of capabilities")
    if "framework" in snapshot and snapshot["framework"] not in ("pytorch", "tensorflow"):
        raise ValueError("framework must be pytorch or tensorflow")
    if "feature_groups" in snapshot:
        snapshot = {**snapshot, "feature_groups": validate_feature_groups(
            snapshot["feature_groups"], snapshot["fields"])}
    requirements = snapshot.get("domain_requirements", [])
    if not isinstance(requirements, list):
        raise ValueError("domain_requirements must be a list")
    requirement_ids = set()
    for requirement in requirements:
        if not isinstance(requirement, dict) or any(
                not isinstance(requirement.get(key), str) or not requirement[key].strip()
                for key in ("id", "source", "as_of")) or not isinstance(
                    requirement.get("fields"), list) or not requirement["fields"] or any(
                    not isinstance(field, str) or not field.strip()
                    for field in requirement["fields"]):
            raise ValueError("domain requirement needs id, source, as_of, and fields")
        if requirement["id"] in requirement_ids:
            raise ValueError("duplicate domain requirement id")
        requirement_ids.add(requirement["id"])
    protocol = snapshot.get("evaluation_protocol")
    if "evaluation_protocol" in snapshot:
        required = ["unit", "split", "metric"]
        if snapshot["stage"] in ("retrieval", "reranking"):
            required.append("candidate_universe")
        if "implicit_feedback" in snapshot["capabilities"]:
            required.append("negative_source")
        if not isinstance(protocol, dict) or any(
                not isinstance(protocol.get(key), str) or not protocol[key].strip()
                for key in required):
            raise ValueError("evaluation_protocol needs nonempty " + ", ".join(required))
    if "evidence" in snapshot:
        for fact in validate_host_evidence(snapshot["evidence"]).values():
            if fact["scope"] != "task":
                raise ValueError("task snapshot evidence must have task scope")
    _json_bytes(snapshot)
    return snapshot


def _evaluation(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("evaluation must be a dict")
    score = value.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("evaluation needs a finite numeric score")
    if not isinstance(value.get("metrics"), dict):
        raise ValueError("evaluation needs a metrics dict")
    observations = value.get("business_observations", [])
    if not isinstance(observations, list):
        raise ValueError("business_observations must be a list")
    observed_ids = set()
    for observation in observations:
        if not isinstance(observation, dict) or any(
                not isinstance(observation.get(key), str) or not observation[key].strip()
                for key in ("id", "population", "metric", "uncertainty")) or isinstance(
                    observation.get("estimate"), bool) or not isinstance(
                    observation.get("estimate"), (int, float)) or not math.isfinite(
                    observation["estimate"]):
            raise ValueError("business observation needs id, population, metric, estimate, uncertainty")
        if observation["id"] in observed_ids:
            raise ValueError("duplicate business observation id")
        observed_ids.add(observation["id"])
    if "evidence" in value:
        validate_host_evidence(value["evidence"])
    check = value.get("implementation_check")
    if check is not None and (not isinstance(check, dict) or check.get("status") not in
                              ("verified", "contradicted", "unverified")):
        raise ValueError("implementation_check needs verified, contradicted, or unverified status")
    audit = value.get("change_audit")
    if audit is not None:
        if not isinstance(audit, dict) or audit.get("status") not in ("verified", "unverified"):
            raise ValueError("change_audit needs verified or unverified status")
        factors = audit.get("changed_factors")
        if audit["status"] == "verified" and (not isinstance(factors, list) or not factors or any(
                not isinstance(factor, str) or not factor.strip() for factor in factors)):
            raise ValueError("verified change_audit needs changed_factors")
    _json_bytes(value)
    return value


def _host_evidence(snapshot: dict, baseline: dict, steps: list[dict]) -> list[dict] | None:
    """Collect only facts supplied by the task and evaluator, never by Agent proposals."""
    sources = [snapshot, baseline]
    sources.extend(step["evaluation"] for step in steps if step.get("status") == "evaluated")
    if not any("evidence" in source for source in sources):
        return None
    evidence = [fact for source in sources for fact in source.get("evidence", [])]
    validate_host_evidence(evidence)
    for fact in baseline.get("evidence", []):
        if fact["scope"] != "task":
            raise ValueError("baseline evidence must have task scope")
    for step in steps:
        for fact in step.get("evaluation", {}).get("evidence", []):
            if fact["scope"] == "trial" and fact["trial_id"] != step["id"]:
                raise ValueError("trial evidence must match its evaluated trial")
    return evidence


def _trial_research(step: dict) -> dict:
    return step.get("proposal", {}).get("research", step.get("research", {}))


def _gap_trials(ids: object, evidence: list[dict], steps: list[dict]) -> set[str]:
    facts = cited_facts(ids, evidence)
    evaluated = {step["id"]: step for step in steps if step.get("status") == "evaluated"}
    return {fact["trial_id"] for fact in facts if fact["scope"] == "trial" and
            fact["status"] == "observed" and fact.get("data_gap_candidate") is True and
            fact["trial_id"] in evaluated}


def validate_data_request(request: dict, snapshot: dict, steps: list[dict], *,
                          evidence: list[dict] | None = None) -> dict:
    """Require experimental evidence or an explicit host-supplied domain rule."""
    if not isinstance(request, dict) or any(
            not isinstance(request.get(key), str) or not request[key].strip()
            for key in ("source", "as_of", "reason", "evidence", "validation_plan")):
        raise ValueError("data request needs source, as_of, reason, evidence, validation_plan")
    fields = request.get("fields")
    if not isinstance(fields, list) or not fields or any(
            not isinstance(field, str) or not field.strip() for field in fields):
        raise ValueError("data request needs nonempty fields")
    if set(fields) & set(snapshot["fields"]):
        raise ValueError("data request fields must be absent from the frozen dataset")
    basis = request.get("basis")
    if basis == "domain_requirement":
        requirement = next((item for item in snapshot.get("domain_requirements", [])
                            if item["id"] == request.get("requirement_id")), None)
        if (requirement is None or not set(fields) <= set(requirement["fields"]) or
                request["source"] != requirement["source"] or
                request["as_of"] != requirement["as_of"]):
            raise ValueError("data request needs a matching host domain requirement")
    elif basis == "experimental_evidence":
        trial_ids = request.get("trial_ids")
        if (not isinstance(trial_ids, list) or
                any(not isinstance(item, str) for item in trial_ids) or
                len(set(trial_ids)) < 2 or
                not isinstance(request.get("alternatives_considered"), str) or
                not request["alternatives_considered"].strip()):
            raise ValueError("experimental data request needs two distinct trials and alternatives")
        evaluated = {step["id"]: step for step in steps if step["status"] == "evaluated"}
        if any(trial_id not in evaluated for trial_id in trial_ids):
            raise ValueError("experimental data request must cite completed evaluated trials")
        mechanisms = {_trial_research(evaluated[trial_id])["mechanism"]
                      for trial_id in trial_ids}
        if len(mechanisms) < 2:
            raise ValueError("experimental data request needs two distinct tested mechanisms")
        if evidence is not None:
            gap_trials = _gap_trials(request.get("evidence_ids"), evidence, steps)
            supported = gap_trials & set(trial_ids)
            if len(supported) < 2 or len({_trial_research(evaluated[trial_id])["mechanism"]
                                          for trial_id in supported}) < 2:
                raise ValueError("experimental data request needs trial-specific measured gap evidence")
    else:
        raise ValueError("data request basis must be domain_requirement or experimental_evidence")
    if basis == "domain_requirement" and evidence is not None and "evidence_ids" in request:
        cited_facts(request["evidence_ids"], evidence)
    _json_bytes(request)
    return request


def validate_reflection(reflection: dict, evaluation: dict | None,
                        snapshot: dict, steps: list[dict], *,
                        evidence: list[dict] | None = None) -> dict:
    """Keep technical and business lessons distinct and cite observed business data."""
    if not isinstance(reflection, dict):
        raise ValueError("agent.reflect() must return a dict")
    technical = reflection.get("technical_experience")
    if not isinstance(technical, dict) or any(
            not isinstance(technical.get(key), str) or not technical[key].strip()
            for key in ("lesson", "evidence", "uncertainty", "next_test")):
        raise ValueError("technical_experience needs lesson, evidence, uncertainty, next_test")
    implementation = (evaluation or {}).get("implementation_check", {})
    audit = (evaluation or {}).get("change_audit", {})
    implementation_status = implementation.get("status", "unverified")
    if technical.setdefault("implementation_status", implementation_status) != implementation_status:
        raise ValueError("technical implementation_status must match the host check")
    if implementation.get("status") == "verified" and audit.get("status") == "verified":
        expected = "isolated" if len(audit["changed_factors"]) == 1 else "joint"
    else:
        expected = "unverified"
    attribution = technical.setdefault("attribution", "unverified")
    if attribution not in ("isolated", "joint", "unverified") or attribution != expected:
        raise ValueError(f"technical attribution must be {expected} under the host change audit")
    design = (evaluation or {}).get("research", {}).get("model_design")
    assessments = technical.get("component_assessments")
    if design is not None or assessments is not None:
        components = {item["id"] for item in (design or {}).get("components", [])}
        if not isinstance(assessments, list) or any(not isinstance(item, dict) for item in assessments):
            raise ValueError("component_assessments must cover the current model_design")
        ids = [item.get("component_id") for item in assessments]
        if any(not isinstance(identifier, str) for identifier in ids) or len(set(ids)) != len(ids) or set(ids) != components:
            raise ValueError("component_assessments must name every current component exactly once")
        invalid = (implementation_status == "contradicted" or
                   (evaluation or {}).get("trial_status") == "failed")
        for item in assessments:
            if item.get("outcome") not in ("promising", "inconclusive", "harmful", "invalid") or any(
                    not isinstance(item.get(key), str) or not item[key].strip()
                    for key in ("evidence", "compatibility_limits", "next_test")):
                raise ValueError("component_assessments need outcome, evidence, compatibility_limits, next_test")
            if invalid and item["outcome"] not in ("invalid", "inconclusive"):
                raise ValueError("invalid trials cannot establish component outcomes")
            allowed = {"unverified"}
            if expected == "joint":
                allowed.add("joint")
            if expected == "isolated" and audit.get("changed_factors") == [item["component_id"]]:
                allowed.add("isolated")
            if item.get("attribution", "unverified") not in allowed:
                raise ValueError("component attribution exceeds the host change audit")
            item.setdefault("attribution", "unverified")
    business = reflection.get("business_experience")
    if not isinstance(business, dict) or business.get("status") not in (
            "observed", "not_observable"):
        raise ValueError("business_experience needs observed or not_observable status")
    if business["status"] == "observed":
        known_ids = {item["id"] for item in (evaluation or {}).get("business_observations", [])}
        if (business.get("observation_id") not in known_ids or any(
                not isinstance(business.get(key), str) or not business[key].strip()
                for key in ("insight", "limitations"))):
            raise ValueError("observed business experience needs a measured business observation")
    elif not isinstance(business.get("reason"), str) or not business["reason"].strip():
        raise ValueError("not_observable business experience needs a reason")
    suggestions = reflection.get("future_feature_suggestions", [])
    if not isinstance(suggestions, list) or any(
            not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in ("field", "source", "as_of", "evidence", "validation_plan"))
            for item in suggestions):
        raise ValueError("future_feature_suggestions need field, source, as_of, evidence, validation_plan")
    tested_mechanisms = {_trial_research(step)["mechanism"] for step in steps
                         if step["status"] == "evaluated"}
    for suggestion in suggestions:
        if suggestion["field"] in snapshot["fields"]:
            raise ValueError("suggested feature is already in the frozen dataset")
        explicit = any(suggestion["field"] in requirement["fields"] and
                       suggestion["source"] == requirement["source"] and
                       suggestion["as_of"] == requirement["as_of"]
                       for requirement in snapshot.get("domain_requirements", []))
        if not explicit and len(tested_mechanisms) < 2:
            raise ValueError("future feature suggestion needs two distinct tested mechanisms or a host domain requirement")
        if evidence is not None and not explicit:
            gap_trials = _gap_trials(suggestion.get("evidence_ids"), evidence, steps)
            mechanisms = {_trial_research(step)["mechanism"] for step in steps
                          if step.get("id") in gap_trials and step.get("status") == "evaluated"}
            if len(mechanisms) < 2:
                raise ValueError("future feature suggestion needs trial-specific measured gap evidence")
    if "audit_recommendations" in reflection:
        validate_audit_recommendations(reflection["audit_recommendations"], evidence=evidence)
    _json_bytes(reflection)
    return reflection


def _proposal(value: object, snapshot: dict, catalog: dict, *,
              evidence: list[dict] | None = None, sources: list[dict] | None = None) -> dict:
    if not isinstance(value, dict):
        raise ValueError("agent.propose() must return a dict")
    if value.get("action") != "experiment":
        raise ValueError("expected an experiment proposal")
    if "candidate" not in value or value["candidate"] is None:
        raise ValueError("experiment needs a candidate")
    research = value.get("research")
    if not isinstance(research, dict):
        raise ValueError("experiment needs research")
    input_fields = research.get("input_fields")
    if not isinstance(input_fields, list):
        raise ValueError("research.input_fields needs a list of task field names")
    missing = [field for field in input_fields if field not in snapshot["fields"]]
    if missing:
        raise ValueError(f"research.input_fields absent from task: {', '.join(map(str, missing))}")
    from .catalog import validate_research

    checked_research = validate_research(value, snapshot, catalog, evidence=evidence, sources=sources)
    proposal = {"action": "experiment", "research": checked_research,
                "candidate": value["candidate"]}
    if "reference_reads" in value:
        proposal["reference_reads"] = value["reference_reads"]
    _json_bytes(proposal)
    return proposal


def _reflect_pending(agent: object, snapshot: dict, state: dict,
                     max_steps: int, journal_path: Path, *, structured: bool) -> None:
    step = state["steps"][-1]
    if "reflection" in step:
        return
    observation = {
        "task": snapshot,
        "baseline": state["baseline"],
        "steps": state["steps"][:-1],
        "trial": {key: value for key, value in step.items() if key != "reflection"},
        "best_id": state["best_id"],
        "remaining_steps": max(0, max_steps - len(state["steps"])),
    }
    evidence = _host_evidence(snapshot, state["baseline"], state["steps"])
    if evidence is not None:
        observation["evidence"] = evidence
    else:
        observation["evidence_status"] = "not_supplied"
    reflection = agent.reflect(json.loads(_json_bytes(observation)))
    if structured:
        evaluation = {**(step.get("evaluation") or {}), "research": step["proposal"]["research"],
                      "trial_status": step["status"]}
        reflection = validate_reflection(reflection, evaluation, snapshot,
                                         state["steps"], evidence=evidence)
    elif not isinstance(reflection, dict):
        raise ValueError("agent.reflect() must return a dict")
    else:
        _json_bytes(reflection)
    step["reflection"] = reflection
    _write_journal(journal_path, state)


def run_search(task: object, agent: object, *, output: Path, catalog: dict,
               max_steps: int, resume: bool = False) -> dict:
    """Run up to ``max_steps`` attempted experiments and return the journal.

    ``task`` supplies snapshot/baseline/evaluate; ``agent`` supplies
    propose/reflect. Only task and catalog identity, proposals, observations,
    and reflections are persisted. Agent/provider configuration is excluded.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if not isinstance(catalog, dict):
        raise ValueError("catalog must be a dict")

    from .catalog import (applicability, catalog_digest, common_knowledge, decision_applicability,
                          implementation_digest, load_guide, model_api, method_applicability,
                          training_applicability)

    snapshot = _snapshot(task)
    if getattr(agent, "requires_model_design", False):
        snapshot = {**snapshot, "model_design_required": True}
    if getattr(agent, "requires_horizontal_expansion", False):
        snapshot = {**snapshot, "model_design_required": True, "horizontal_expansion_required": True}
    applicability_report = applicability(snapshot, catalog)
    method_report = method_applicability(snapshot, catalog)
    decision_report = decision_applicability(snapshot, catalog)
    training_report = training_applicability(snapshot, catalog)
    knowledge = {item["family_id"]: load_guide(item["family_id"])
                 for item in applicability_report if item["status"] == "ready" and
                 any(family["id"] == item["family_id"] and family.get("local_guide")
                     for family in catalog["families"])}
    if catalog.get("experience_schema_version") == 2:
        knowledge.update(common_knowledge())
    ready_methods = {item["method_id"] for item in method_report if item["status"] == "ready"}
    frameworks = ((snapshot["framework"],) if "framework" in snapshot else
                  ("pytorch", "tensorflow"))
    model_apis = {framework: model_api(catalog, framework=framework,
                                      method_ids=ready_methods)
                  for framework in frameworks}
    structured = catalog.get("experience_schema_version") == 2
    identity = {
        "task_id": snapshot["task_id"],
        "dataset_digest": snapshot["dataset_digest"],
        "task_sha256": _digest(snapshot),
        "catalog_sha256": catalog_digest(catalog),
        "package_version": PACKAGE_VERSION,
        "implementation_sha256": implementation_digest(),
    }
    if "evaluation_protocol" in snapshot:
        identity["evaluation_protocol_sha256"] = _digest(snapshot["evaluation_protocol"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    journal_path = output / "journal.json"

    if resume:
        if not journal_path.exists():
            raise FileNotFoundError(f"cannot resume without {journal_path}")
        state = json.loads(journal_path.read_text(encoding="utf-8"))
        if (state.get("identity", {}).get("evaluation_protocol_sha256") !=
                identity.get("evaluation_protocol_sha256")):
            raise ValueError("journal evaluation protocol differs from this task")
        if state.get("identity") != identity:
            raise ValueError("journal identity differs from task, catalog, package version, or implementation")
        if state["status"] in ("stopped", "needs_data"):
            return state
        if state["steps"] and "reflection" not in state["steps"][-1]:
            _reflect_pending(agent, snapshot, state, max_steps, journal_path,
                             structured=structured)
    else:
        if journal_path.exists():
            raise FileExistsError(f"journal already exists: {journal_path}; use resume=True")
        baseline_dir = output / "baseline"
        baseline_dir.mkdir(exist_ok=True)
        baseline = _evaluation(task.baseline(baseline_dir))
        state = {"identity": identity, "baseline": baseline, "steps": [],
                 "best_id": "baseline", "status": "running"}
        _write_journal(journal_path, state)

    while len(state["steps"]) < max_steps:
        evidence = _host_evidence(snapshot, state["baseline"], state["steps"])
        context = {
            "task": snapshot,
            "catalog": catalog,
            "applicability": applicability_report,
            "method_applicability": method_report,
            "decision_applicability": decision_report,
            "training_applicability": training_report,
            "knowledge": knowledge,
            "model_api": model_apis,
            "baseline": state["baseline"],
            "steps": state["steps"],
            "best_id": state["best_id"],
            "remaining_steps": max_steps - len(state["steps"]),
            "composition_sources": composition_sources(state["steps"]),
        }
        if evidence is not None:
            context["evidence"] = evidence
        else:
            context["evidence_status"] = "not_supplied"
        # The Agent may mutate its input; the journal must retain observed history.
        decision = agent.propose(json.loads(_json_bytes(context)))
        if not isinstance(decision, dict):
            raise ValueError("agent.propose() must return a dict")
        action = decision.get("action")
        if action == "stop":
            reason = decision.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("stop action needs a reason")
            if "audit_recommendations" in decision:
                state["audit_recommendations"] = validate_audit_recommendations(
                    decision["audit_recommendations"], evidence=evidence)
            state["status"] = "stopped"
            state["stop_reason"] = reason
            _write_journal(journal_path, state)
            return state
        if action == "request_data":
            request = decision.get("request")
            if not isinstance(request, dict) or not request:
                raise ValueError("request_data action needs a concrete request")
            if structured:
                validate_data_request(request, snapshot, state["steps"], evidence=evidence)
            else:
                _json_bytes(request)
            state["status"] = "needs_data"
            state["data_request"] = request
            _write_journal(journal_path, state)
            return state
        if action != "experiment":
            raise ValueError("action must be experiment, stop, or request_data")

        proposal = _proposal(decision, snapshot, catalog, evidence=evidence,
                             sources=composition_sources(state["steps"]))
        trial_id = f"trial_{len(state['steps']) + 1:03d}"
        trial_dir = output / trial_id
        trial_dir.mkdir(exist_ok=True)
        step = {"id": trial_id, "proposal": proposal}
        try:
            raw_evaluation = task.evaluate(proposal, trial_dir)
        except Exception as error:
            # Exception messages may contain credentials or raw data. The type
            # gives the Agent a bounded failure signal without storing either.
            step.update(status="failed", error=f"evaluation failed: {type(error).__name__}"[:120])
        else:
            evaluation = _evaluation(raw_evaluation)
            step.update(status="evaluated", evaluation=evaluation)
            _host_evidence(snapshot, state["baseline"], [*state["steps"], step])
            current_score = (state["baseline"]["score"] if state["best_id"] == "baseline" else
                             next(previous["evaluation"]["score"] for previous in state["steps"]
                                  if previous["id"] == state["best_id"]))
            better = (evaluation["score"] > current_score if snapshot["objective"]["direction"] == "max"
                      else evaluation["score"] < current_score)
            if better and evaluation.get("implementation_check", {}).get("status") != "contradicted":
                state["best_id"] = trial_id
        state["steps"].append(step)
        state["status"] = "running"
        _write_journal(journal_path, state)
        _reflect_pending(agent, snapshot, state, max_steps, journal_path,
                         structured=structured)

    state["status"] = "completed"
    _write_journal(journal_path, state)
    return state
