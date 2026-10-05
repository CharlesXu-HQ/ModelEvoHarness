"""Track declared component lineage; declarations do not prove execution or gains."""

from __future__ import annotations

import hashlib


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _strings(value: object, name: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{name} must be a list" + (" with entries" if nonempty else ""))
    result = [_text(item, name) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {name}")
    return result


def model_design_identity(design: dict) -> dict[str, str]:
    """Read stable IDs, deriving deterministic IDs for older description-only records."""
    identity = {}
    for axis in ("estimator", "backbone"):
        key = f"{axis}_id"
        if key in design:
            identity[key] = _text(design[key], f"model_design.{key}")
        else:
            description = _text(design.get(axis), f"model_design.{axis}")
            digest = hashlib.sha256(description.encode()).hexdigest()[:16]
            identity[key] = f"{axis}-{digest}"
    return identity


def composition_sources(steps: list[dict]) -> list[dict]:
    """Expose compact lineage from core proposals or host-native trial records."""
    sources = []
    for step in steps:
        research = step.get("research") or step.get("proposal", {}).get("research", {})
        design = research.get("model_design")
        if design:
            research = {**research, "model_design": {**design, **model_design_identity(design)}}
        evaluation = step.get("evaluation") or {}
        analysis = step.get("analysis") or {}
        review = analysis.get("max") or analysis.get("high") or analysis
        check = (step.get("implementation_check") or evaluation.get("implementation_check") or
                 review.get("implementation_check") or {})
        sources.append({"id": step.get("id"), "status": step.get("status"),
                        "eligibility": step.get("eligibility"), "research": research,
                        "implementation_check": check, "reflection": step.get("reflection") or {}})
    return sources


def _components(items: object, snapshot: dict) -> dict[str, dict]:
    if not isinstance(items, list) or not items:
        raise ValueError("model_design.components must be a nonempty list")
    components = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("component must be an object")
        current: dict = {key: _text(item.get(key), f"component.{key}") for key in ("id", "mechanism")}
        current.update({key: _strings(item.get(key), f"component.{key}", nonempty=key == "code_sections")
                        for key in ("code_sections", "input_fields", "required_capabilities")})
        if not set(current["input_fields"]).issubset(snapshot.get("fields", [])):
            raise ValueError("component.input_fields must exist in snapshot.fields")
        if not set(current["required_capabilities"]).issubset(snapshot.get("capabilities", [])):
            raise ValueError("component.required_capabilities must exist in snapshot.capabilities")
        if "reference_method_id" in item:
            current["reference_method_id"] = _text(item["reference_method_id"], "reference_method_id")
        if current["id"] in components:
            raise ValueError("duplicate component id")
        components[current["id"]] = current
    return components


def validate_model_design(design: object, snapshot: dict, sources: list[dict]) -> dict:
    """Validate a local iteration or selective migration against recorded parents."""
    if not isinstance(design, dict):
        raise ValueError("model_design must be an object")
    result: dict = {key: _text(design.get(key), f"model_design.{key}")
              for key in ("estimator", "backbone", "rationale", "data_fit", "comparison_plan")}
    scope = design.get("change_scope")
    if scope not in ("initialize", "local", "switch"):
        raise ValueError("change_scope must be initialize, local, or switch")
    indexed = {}
    for item in composition_sources(sources):
        identifier = _text(item["id"], "source id")
        if identifier in indexed:
            raise ValueError("duplicate source id")
        indexed[identifier] = item
    parent_id = design.get("parent_trial_id")
    if parent_id is not None:
        parent_id = _text(parent_id, "parent_trial_id")
        if parent_id not in indexed:
            raise ValueError("parent_trial_id must reference a source")
    parent = indexed[parent_id]["research"].get("model_design") if parent_id is not None else None
    identity = model_design_identity(design)
    if parent:
        parent_identity = model_design_identity(parent)
        for axis in ("estimator", "backbone"):
            key = f"{axis}_id"
            if key not in design and (scope == "local" or result[axis] == parent.get(axis)):
                identity[key] = parent_identity[key]
    result.update(identity)
    if scope == "initialize":
        if any(item["research"].get("model_design") for item in indexed.values()):
            raise ValueError("initialize cannot reset tracked model_design history")
    else:
        if not parent:
            raise ValueError("local/switch needs a parent source with model_design")
        same = identity == model_design_identity(parent)
        if (scope == "local" and not same) or (scope == "switch" and same):
            raise ValueError("local preserves estimator_id/backbone_id (omit them to inherit the parent IDs); "
                             "switch must change at least one ID. Edit estimator/backbone descriptions freely.")
    components = _components(design.get("components"), snapshot)
    inheritance = design.get("inheritance")
    if not isinstance(inheritance, list):
        raise ValueError("model_design.inheritance must be a list")
    records, seen = [], set()
    for item in inheritance:
        if not isinstance(item, dict):
            raise ValueError("inheritance entry must be an object")
        entry: dict = {key: _text(item.get(key), f"inheritance.{key}") for key in
                 ("source_trial_id", "component_id", "reason", "compatibility", "validation_plan")}
        pair = (entry["source_trial_id"], entry["component_id"])
        if pair in seen:
            raise ValueError("duplicate inheritance source/component")
        seen.add(pair)
        source = indexed.get(pair[0])
        if source is None:
            raise ValueError("inheritance source_trial_id must reference a real source")
        previous = next((component for component in source["research"].get("model_design", {}).get("components", [])
                         if component.get("id") == pair[1]), None)
        if previous is None:
            raise ValueError("inheritance component_id must reference a source component")
        decision = item.get("decision")
        if decision not in ("retain", "adapt", "drop", "retest"):
            raise ValueError("inheritance decision must be retain, adapt, drop, or retest")
        eligibility = source.get("eligibility")
        if isinstance(eligibility, dict):
            eligibility = eligibility.get("status")
        reflection = source["reflection"]
        assessments = reflection.get("technical_experience", {}).get("component_assessments", [])
        reviewed = any(assessment.get("component_id") == pair[1] for assessment in assessments)
        invalid = (not reviewed or source.get("status") != "evaluated" or
                   eligibility in ("failed", "contradicted") or
                   (isinstance(eligibility, str) and eligibility.startswith("blocked_")) or
                   source["implementation_check"].get("status") == "contradicted" or
                   reflection.get("verdict") == "invalid" or
                   any(assessment.get("component_id") == pair[1] and
                       assessment.get("outcome") in ("invalid", "harmful") for assessment in assessments))
        if invalid and decision in ("retain", "adapt"):
            raise ValueError("invalid or unreviewed source components require drop or retest")
        entry["decision"] = decision
        target_id = item.get("target_component_id")
        if decision == "drop":
            if target_id is not None:
                raise ValueError("drop cannot map to a target component")
        else:
            target_id = _text(target_id, "target_component_id")
            if target_id not in components:
                raise ValueError("target_component_id must reference a current component")
            entry["target_component_id"] = target_id
            if decision == "retain":
                target = components[target_id]
                same = (_text(previous.get("mechanism"), "source mechanism") == target["mechanism"] and
                        previous.get("reference_method_id") == target.get("reference_method_id") and
                        all(set(previous.get(key, [])) == set(target[key]) for key in
                            ("code_sections", "input_fields", "required_capabilities")))
                if not same:
                    raise ValueError("retain must preserve mechanism, reference_method_id, code_sections, fields, and capabilities; use adapt/retest")
        records.append(entry)
    if parent:
        expected = {(parent_id, component["id"]) for component in parent["components"]}
        if not expected.issubset(seen):
            raise ValueError("every parent component needs an inheritance decision")
    result.update(change_scope=scope, parent_trial_id=parent_id,
                  components=list(components.values()), inheritance=records)
    return result
