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


PACKAGE_VERSION = "0.1.0"


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
    _json_bytes(value)
    return value


def _proposal(value: object, snapshot: dict, catalog: dict) -> dict:
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

    checked_research = validate_research(value, snapshot, catalog)
    proposal = {"action": "experiment", "research": checked_research,
                "candidate": value["candidate"]}
    _json_bytes(proposal)
    return proposal


def _reflect_pending(agent: object, snapshot: dict, state: dict,
                     max_steps: int, journal_path: Path) -> None:
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
    reflection = agent.reflect(json.loads(_json_bytes(observation)))
    if not isinstance(reflection, dict):
        raise ValueError("agent.reflect() must return a dict")
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

    from .catalog import applicability, catalog_digest, implementation_digest, method_applicability

    snapshot = _snapshot(task)
    applicability_report = applicability(snapshot, catalog)
    method_report = method_applicability(snapshot, catalog)
    identity = {
        "task_id": snapshot["task_id"],
        "dataset_digest": snapshot["dataset_digest"],
        "task_sha256": _digest(snapshot),
        "catalog_sha256": catalog_digest(catalog),
        "package_version": PACKAGE_VERSION,
        "implementation_sha256": implementation_digest(),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    journal_path = output / "journal.json"

    if resume:
        if not journal_path.exists():
            raise FileNotFoundError(f"cannot resume without {journal_path}")
        state = json.loads(journal_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise ValueError("journal identity differs from task, catalog, package version, or implementation")
        if state["status"] in ("stopped", "needs_data"):
            return state
        if state["steps"] and "reflection" not in state["steps"][-1]:
            _reflect_pending(agent, snapshot, state, max_steps, journal_path)
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
        context = {
            "task": snapshot,
            "catalog": catalog,
            "applicability": applicability_report,
            "method_applicability": method_report,
            "baseline": state["baseline"],
            "steps": state["steps"],
            "best_id": state["best_id"],
            "remaining_steps": max_steps - len(state["steps"]),
        }
        # The Agent may mutate its input; the journal must retain observed history.
        decision = agent.propose(json.loads(_json_bytes(context)))
        if not isinstance(decision, dict):
            raise ValueError("agent.propose() must return a dict")
        action = decision.get("action")
        if action == "stop":
            reason = decision.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("stop action needs a reason")
            state["status"] = "stopped"
            state["stop_reason"] = reason
            _write_journal(journal_path, state)
            return state
        if action == "request_data":
            request = decision.get("request")
            if not isinstance(request, dict) or not request:
                raise ValueError("request_data action needs a concrete request")
            _json_bytes(request)
            state["status"] = "needs_data"
            state["data_request"] = request
            _write_journal(journal_path, state)
            return state
        if action != "experiment":
            raise ValueError("action must be experiment, stop, or request_data")

        proposal = _proposal(decision, snapshot, catalog)
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
            current_score = (state["baseline"]["score"] if state["best_id"] == "baseline" else
                             next(previous["evaluation"]["score"] for previous in state["steps"]
                                  if previous["id"] == state["best_id"]))
            better = (evaluation["score"] > current_score if snapshot["objective"]["direction"] == "max"
                      else evaluation["score"] < current_score)
            if better:
                state["best_id"] = trial_id
        state["steps"].append(step)
        state["status"] = "running"
        _write_journal(journal_path, state)
        _reflect_pending(agent, snapshot, state, max_steps, journal_path)

    state["status"] = "completed"
    _write_journal(journal_path, state)
    return state
