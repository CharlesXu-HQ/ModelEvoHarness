"""Validate host-owned facts cited by Agent research decisions."""

from __future__ import annotations

import json


def validate_host_evidence(evidence: list[dict]) -> dict[str, dict]:
    """Return facts by ID; the host, never the Agent, supplies this list."""
    if not isinstance(evidence, list):
        raise ValueError("host evidence must be a list")
    facts = {}
    for item in evidence:
        if not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in ("id", "statement", "source")):
            raise ValueError("host evidence needs id, statement, and source")
        if item.get("status") not in ("observed", "declared") or item.get("scope") not in (
                "task", "trial"):
            raise ValueError("host evidence needs observed/declared status and task/trial scope")
        if item["scope"] == "trial":
            if not isinstance(item.get("trial_id"), str) or not item["trial_id"].strip():
                raise ValueError("trial evidence needs trial_id")
        elif "trial_id" in item:
            raise ValueError("task evidence cannot carry trial_id")
        gap = item.get("data_gap_candidate", False)
        if not isinstance(gap, bool) or (gap and (item["scope"] != "trial" or
                                            item["status"] != "observed")):
            raise ValueError("data_gap_candidate needs observed trial evidence")
        if item["id"] in facts:
            raise ValueError("duplicate host evidence id")
        json.dumps(item, allow_nan=False)
        facts[item["id"]] = item
    return facts


def cited_facts(ids: object, evidence: list[dict], *, nonempty: bool = True) -> list[dict]:
    if (not isinstance(ids, list) or (nonempty and not ids) or
            any(not isinstance(identifier, str) or not identifier.strip() for identifier in ids) or
            len(set(ids)) != len(ids)):
        raise ValueError("evidence_ids needs distinct nonempty IDs")
    facts = validate_host_evidence(evidence)
    unknown_ids = [identifier for identifier in ids if identifier not in facts]
    if unknown_ids:
        def preview(values: list[str]) -> str:
            shown = ", ".join(repr(identifier)[:96] for identifier in values[:6])
            return shown + (f" ... (+{len(values) - 6} more)" if len(values) > 6 else "")

        raise ValueError(f"evidence_ids must cite host evidence; unknown IDs: [{preview(unknown_ids)}]; "
                         f"available host IDs: [{preview(list(facts))}]. "
                         "Copy existing evidence IDs exactly; do not invent or rename them.")
    return [facts[identifier] for identifier in ids]


def validate_audit_recommendations(items: object, *, evidence: list[dict] | None) -> list[dict]:
    """Record data-quality checks without turning them into blocking data requests."""
    if not isinstance(items, list):
        raise ValueError("audit_recommendations must be a list")
    for item in items:
        if not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in ("issue", "validation_plan")):
            raise ValueError("audit recommendation needs issue and validation_plan")
        if evidence is None:
            raise ValueError("audit recommendation needs host evidence")
        cited_facts(item.get("evidence_ids"), evidence)
    return items
