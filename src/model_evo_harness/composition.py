"""Track declared component lineage; declarations do not prove execution or gains."""

from __future__ import annotations

import hashlib
import re


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


def validate_feature_groups(groups: object, fields: list[str]) -> list[dict]:
    """Normalize declared field subsets without inferring data capabilities."""
    if groups is None:
        return []
    if not isinstance(groups, list):
        raise ValueError("feature_groups must be a list")
    result, seen = [], set()
    for item in groups:
        if not isinstance(item, dict):
            raise ValueError("feature_groups entry must be an object")
        group = {key: _text(item.get(key), f"feature_groups.{key}") for key in ("id", "rationale")}
        group["fields"] = _strings(item.get("fields"), "feature_groups.fields", nonempty=True)
        if not set(group["fields"]).issubset(fields):
            raise ValueError("feature_groups.fields must exist in snapshot.fields")
        if group["id"] in seen:
            raise ValueError("duplicate feature_groups id")
        seen.add(group["id"])
        result.append(group)
    return result


def validate_horizontal_expansion(value: object, components: dict[str, dict]) -> dict:
    """Check declared topology, not execution, shared objects, shapes, or gradients."""
    if not isinstance(value, dict):
        raise ValueError("model_design.horizontal_expansion must be an object")
    result: dict = {key: _text(value.get(key), f"horizontal_expansion.{key}")
                    for key in ("decision", "rationale", "comparison_plan")}
    decision = result["decision"]
    if decision not in ("expand", "defer", "not_applicable"):
        raise ValueError("horizontal_expansion.decision must be expand, defer, or not_applicable")
    groups = value.get("groups")
    if not isinstance(groups, list):
        raise ValueError("horizontal_expansion.groups must be a list")
    if decision == "expand" and not groups:
        raise ValueError("expand requires at least one group")
    if decision == "not_applicable" and groups:
        raise ValueError("not_applicable requires empty groups")
    normalized, group_ids, fusion_ids, branch_ids = [], set(), set(), set()
    edges: dict[str, set[str]] = {}
    for item in groups:
        if not isinstance(item, dict):
            raise ValueError("horizontal group must be an object")
        group: dict = {key: _text(item.get(key), f"horizontal_group.{key}") for key in ("id", "fusion_id")}
        branches = _strings(item.get("branch_ids"), "horizontal_group.branch_ids", nonempty=True)
        if len(branches) < 2:
            raise ValueError("horizontal_group.branch_ids needs at least two distinct components")
        involved = set(branches) | {group["fusion_id"]}
        if not involved.issubset(components):
            raise ValueError("horizontal branch_ids and fusion_id must reference current components")
        if group["fusion_id"] in branches:
            raise ValueError("horizontal fusion_id cannot be one of its branch_ids")
        if group["id"] in group_ids:
            raise ValueError("duplicate horizontal group id")
        if group["fusion_id"] in fusion_ids:
            raise ValueError("duplicate horizontal fusion_id; combine its branches in one group")
        group_ids.add(group["id"])
        fusion_ids.add(group["fusion_id"])
        branch_ids.update(branches)
        for identifier in involved:
            component = components[identifier]
            for key in ("instance_path", "output_contract"):
                _text(component.get(key), f"component.{key}")
            if not any(re.search(r"(?:^|[.:/])(?:forward|call)(?:$|[.:/#(])", section)
                       for section in component["code_sections"]):
                raise ValueError("horizontal component.code_sections must locate a forward or call entry point")
        sharing = item.get("parameter_sharing")
        if not isinstance(sharing, list):
            raise ValueError("horizontal_group.parameter_sharing must be a list")
        declarations, shared = [], set()
        for declaration in sharing:
            if not isinstance(declaration, dict):
                raise ValueError("parameter_sharing entry must be an object")
            entry: dict = {"component_ids": _strings(declaration.get("component_ids"),
                                                     "parameter_sharing.component_ids", nonempty=True),
                           "code_sections": _strings(declaration.get("code_sections"),
                                                     "parameter_sharing.code_sections", nonempty=True),
                           "rationale": _text(declaration.get("rationale"), "parameter_sharing.rationale")}
            if len(entry["component_ids"]) < 2 or not set(entry["component_ids"]).issubset(branches):
                raise ValueError("parameter_sharing.component_ids must name at least two current group branches")
            signature = (frozenset(entry["component_ids"]), frozenset(entry["code_sections"]))
            if signature in shared:
                raise ValueError("duplicate parameter_sharing declaration")
            shared.add(signature)
            declarations.append(entry)
        group.update(branch_ids=branches, parameter_sharing=declarations)
        normalized.append(group)
        for branch in branches:
            edges.setdefault(branch, set()).add(group["fusion_id"])
    paths = [components[identifier]["instance_path"] for identifier in branch_ids | fusion_ids]
    if len(paths) != len(set(paths)):
        raise ValueError("distinct horizontal components must have distinct instance_path values")
    pending, visited = set(), set()

    def visit(identifier: str) -> None:
        if identifier in pending:
            raise ValueError("horizontal groups contain a cycle; branch-to-fusion topology must be acyclic")
        if identifier in visited:
            return
        pending.add(identifier)
        for target in edges.get(identifier, ()):
            visit(target)
        pending.remove(identifier)
        visited.add(identifier)

    for identifier in edges:
        visit(identifier)
    result["groups"] = normalized
    return result


def _horizontal_footprint(design: dict, component_id: str) -> set[tuple]:
    """Compare execution relationships by instance path, ignoring editorial IDs."""
    paths = {item['id']: item.get('instance_path', item['id']) for item in design['components']}
    footprint = set()
    for group in design.get('horizontal_expansion', {}).get('groups', []):
        branches, fusion = group['branch_ids'], group['fusion_id']
        if component_id not in branches and component_id != fusion:
            continue
        sharing = frozenset((frozenset(paths[key] for key in item['component_ids']),
                             frozenset(item['code_sections'])) for item in group['parameter_sharing'])
        footprint.add(('fusion' if component_id == fusion else 'branch',
                       frozenset(paths[key] for key in branches), paths[fusion], sharing))
    return footprint


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
        for key in ("reference_method_id", "instance_path", "output_contract"):
            if key in item:
                current[key] = _text(item[key], f"component.{key}")
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
    if "horizontal_expansion" in design:
        result["horizontal_expansion"] = validate_horizontal_expansion(design["horizontal_expansion"], components)
    elif snapshot.get("horizontal_expansion_required"):
        raise ValueError("model_design.horizontal_expansion is required by the snapshot")
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
                mismatches = []
                if _text(previous.get("mechanism"), "source mechanism") != target["mechanism"]:
                    mismatches.append("mechanism")
                if previous.get("reference_method_id") != target.get("reference_method_id"):
                    mismatches.append("reference_method_id")
                mismatches.extend(key for key in ("instance_path", "output_contract")
                                  if previous.get(key) != target.get(key))
                mismatches.extend(key for key in ("code_sections", "input_fields", "required_capabilities")
                                  if set(previous.get(key, [])) != set(target[key]))
                target_design = {**result, "components": list(components.values())}
                if (_horizontal_footprint(source['research']['model_design'], pair[1]) !=
                        _horizontal_footprint(target_design, target_id)):
                    mismatches.append('horizontal connections/parameter_sharing')
                if mismatches:
                    raise ValueError(f"retain mismatch for source_trial_id={repr(pair[0])[:96]}, "
                                     f"component_id={repr(pair[1])[:96]}, target_component_id={repr(target_id)[:96]}: "
                                     f"{', '.join(mismatches)}. For retain, copy those source fields exactly; "
                                     "intentional changes require adapt, uncertain reuse requires retest.")
        records.append(entry)
    if parent:
        expected = {(parent_id, component["id"]) for component in parent["components"]}
        if not expected.issubset(seen):
            raise ValueError("every parent component needs an inheritance decision")
    result.update(change_scope=scope, parent_trial_id=parent_id,
                  components=list(components.values()), inheritance=records)
    return result
