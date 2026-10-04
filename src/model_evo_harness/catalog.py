"""Research catalog and task-aware experiment checks.

The bundled inventory records what was reviewed upstream. The catalog is
original guidance; upstream source code is never imported or redistributed.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files


def _bundled(name: str) -> dict:
    return json.loads(files("model_evo_harness").joinpath("data", name).read_text())


def load_catalog(*, extra_families: list[dict] | None = None) -> dict:
    catalog = _bundled("catalog.json")
    catalog["families"].extend(_bundled("supplemental_families.json")["families"])
    catalog["method_cards"] = _bundled("method_cards.json")["method_cards"]
    catalog["method_cards"].extend(_bundled("supplemental_method_cards.json")["method_cards"])
    catalog["decision_checks"] = _bundled("decision_checks.json")["decision_checks"]
    catalog["structure_patterns"] = _bundled("structure_patterns.json")["structure_patterns"]
    if extra_families:
        catalog["families"].extend(extra_families)
    validate_catalog(catalog)
    return catalog


def load_inventory() -> dict:
    return _bundled("upstream_inventory.json")


def catalog_digest(catalog: dict) -> str:
    validate_catalog(catalog)
    payload = json.dumps(catalog, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def implementation_digest() -> str:
    """Identify installed decision logic as well as its bundled catalog."""
    package = files("model_evo_harness")
    digest = hashlib.sha256()
    for name in ("__init__.py", "catalog.py", "engine.py", "provider.py", "data/catalog.json",
                 "data/supplemental_families.json", "data/method_cards.json",
                 "data/supplemental_method_cards.json",
                 "data/decision_checks.json", "data/structure_patterns.json",
                 "data/upstream_inventory.json"):
        digest.update(name.encode())
        digest.update(package.joinpath(name).read_bytes())
    return digest.hexdigest()


def validate_catalog(catalog: dict) -> None:
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ValueError("catalog needs schema_version=1")
    families = catalog.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("catalog needs nonempty families")
    seen = set()
    for family in families:
        if not isinstance(family, dict):
            raise ValueError("family must be an object")
        for key in ("id", "name", "question", "experiment", "pitfalls"):
            if not isinstance(family.get(key), str) or not family[key].strip():
                raise ValueError(f"family needs nonempty {key}")
        if family["id"] in seen:
            raise ValueError(f"duplicate family id: {family['id']}")
        seen.add(family["id"])
        source_repo = family.get("source_repo", "funrec")
        if not isinstance(source_repo, str) or not source_repo.strip():
            raise ValueError(f"family {family['id']} needs nonempty source_repo")
        if source_repo != "funrec" and (not isinstance(family.get("source_url"), str) or
                                       not family["source_url"].startswith("https://")):
            raise ValueError(f"family {family['id']} needs HTTPS source_url")
        for key in ("stages", "requires", "source_docs", "source_models",
                    "source_support", "source_project"):
            value = family.get(key, []) if key.startswith("source_") else family.get(key)
            if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
                raise ValueError(f"family {family['id']} needs string list {key}")
    cards = catalog.get("method_cards", [])
    if not isinstance(cards, list):
        raise ValueError("method_cards must be a list")
    card_ids = set()
    for card in cards:
        if not isinstance(card, dict):
            raise ValueError("method card must be an object")
        for key in ("id", "family_id", "source_model", "mechanism", "comparison",
                    "failure_signals", "implementation_boundary"):
            if not isinstance(card.get(key), str) or not card[key].strip():
                raise ValueError(f"method card needs nonempty {key}")
        if card["id"] in card_ids:
            raise ValueError(f"duplicate method id: {card['id']}")
        card_ids.add(card["id"])
        if card["family_id"] not in seen:
            raise ValueError(f"method {card['id']} has unknown family_id")
        source_repo = card.get("source_repo", "funrec")
        if not isinstance(source_repo, str) or not source_repo.strip():
            raise ValueError(f"method {card['id']} needs nonempty source_repo")
        if source_repo != "funrec" and (not isinstance(card.get("source_url"), str) or
                                       not card["source_url"].startswith("https://")):
            raise ValueError(f"method {card['id']} needs HTTPS source_url")
        if not isinstance(card.get("requires"), list) or any(
                not isinstance(value, str) or not value for value in card["requires"]):
            raise ValueError(f"method {card['id']} needs string list requires")
    checks = catalog.get("decision_checks", [])
    if not isinstance(checks, list):
        raise ValueError("decision_checks must be a list")
    check_ids = set()
    for check in checks:
        if not isinstance(check, dict):
            raise ValueError("decision check must be an object")
        for key in ("id", "question", "guardrail", "source_url"):
            if not isinstance(check.get(key), str) or not check[key].strip():
                raise ValueError(f"decision check needs nonempty {key}")
        if check["id"] in check_ids:
            raise ValueError(f"duplicate decision check id: {check['id']}")
        check_ids.add(check["id"])
        if not check["source_url"].startswith("https://"):
            raise ValueError(f"decision check {check['id']} needs HTTPS source_url")
        for key in ("stages", "requires"):
            if not isinstance(check.get(key), list) or any(
                    not isinstance(value, str) or not value for value in check[key]):
                raise ValueError(f"decision check {check['id']} needs string list {key}")
    patterns = catalog.get("structure_patterns", [])
    if not isinstance(patterns, list):
        raise ValueError("structure_patterns must be a list")
    pattern_ids = set()
    linked_methods = set()
    for pattern in patterns:
        if not isinstance(pattern, dict):
            raise ValueError("structure pattern must be an object")
        for key in ("id", "name", "structural_change", "when_to_try",
                    "required_evidence", "controlled_comparison", "reject_if"):
            if not isinstance(pattern.get(key), str) or not pattern[key].strip():
                raise ValueError(f"structure pattern needs nonempty {key}")
        if pattern["id"] in pattern_ids:
            raise ValueError(f"duplicate structure pattern id: {pattern['id']}")
        pattern_ids.add(pattern["id"])
        method_ids = pattern.get("method_ids")
        if not isinstance(method_ids, list) or not method_ids or any(
                not isinstance(value, str) or value not in card_ids for value in method_ids):
            raise ValueError(f"structure pattern {pattern['id']} needs known method_ids")
        if linked_methods.intersection(method_ids) or len(set(method_ids)) != len(method_ids):
            raise ValueError("a method cannot belong to duplicate structure patterns")
        linked_methods.update(method_ids)


def coverage_report(catalog: dict, inventory: dict | None = None) -> dict:
    """Check explicit coverage of upstream topic pages and model modules."""
    validate_catalog(catalog)
    inventory = inventory or load_inventory()
    funrec_families = [family for family in catalog["families"]
                       if family.get("source_repo", "funrec") == "funrec"]
    documents = [path for family in funrec_families for path in family.get("source_docs", [])]
    models = [path for family in funrec_families for path in family.get("source_models", [])]
    support = [path for family in funrec_families for path in family.get("source_support", [])]
    project = [path for family in funrec_families for path in family.get("source_project", [])]
    expected_docs, expected_models = set(inventory["docs"]), set(inventory["models"])
    funrec_cards = [card for card in catalog.get("method_cards", [])
                    if card.get("source_repo", "funrec") == "funrec"]
    method_paths = [card["source_model"] for card in funrec_cards]
    family_models = {family["id"]: set(family["source_models"])
                     for family in funrec_families}
    return {
        "source_commit_sha": inventory["commit_sha"],
        "source_tree_sha": inventory["tree_sha"],
        "document_count": len(expected_docs),
        "model_count": len(expected_models),
        "support_count": len(inventory["support_modules"]),
        "project_count": len(inventory["production_project_modules"]),
        "unmapped_docs": sorted(expected_docs - set(documents)),
        "unmapped_models": sorted(expected_models - set(models)),
        "unknown_docs": sorted(set(documents) - expected_docs),
        "unknown_models": sorted(set(models) - expected_models),
        "duplicate_docs": sorted({path for path in documents if documents.count(path) > 1}),
        "duplicate_models": sorted({path for path in models if models.count(path) > 1}),
        "unmapped_method_models": sorted(expected_models - set(method_paths)),
        "unknown_method_models": sorted(set(method_paths) - expected_models),
        "duplicate_method_models": sorted({path for path in method_paths
                                            if method_paths.count(path) > 1}),
        "misassigned_method_models": sorted(card["source_model"] for card in funrec_cards
                                            if card["source_model"] not in
                                            family_models.get(card["family_id"], set())),
        "unmapped_support": sorted(set(inventory["support_modules"]) - set(support)),
        "unknown_support": sorted(set(support) - set(inventory["support_modules"])),
        "duplicate_support": sorted({path for path in support if support.count(path) > 1}),
        "unmapped_project": sorted(set(inventory["production_project_modules"]) - set(project)),
        "unknown_project": sorted(set(project) - set(inventory["production_project_modules"])),
        "duplicate_project": sorted({path for path in project if project.count(path) > 1}),
    }


def applicability(snapshot: dict, catalog: dict) -> list[dict]:
    """Explain prerequisites; stage and capability names remain extensible."""
    validate_catalog(catalog)
    stage = snapshot.get("stage")
    capabilities = set(snapshot.get("capabilities", []))
    result = []
    for family in catalog["families"]:
        stages = family["stages"]
        missing = sorted(set(family["requires"]) - capabilities)
        status = ("other_stage" if stages and stage not in stages else
                  "needs_data" if missing else "ready")
        result.append({"family_id": family["id"], "status": status,
                       "missing_capabilities": missing if status == "needs_data" else [],
                       "reason": (f"task stage {stage!r} is outside {stages}" if status == "other_stage" else
                                  f"requires {', '.join(missing)}" if missing else "requirements available")})
    return result


def method_applicability(snapshot: dict, catalog: dict) -> list[dict]:
    """Screen model-specific experiments against stage and declared data contracts."""
    family_status = {item["family_id"]: item for item in applicability(snapshot, catalog)}
    capabilities = set(snapshot.get("capabilities", []))
    result = []
    for card in catalog.get("method_cards", []):
        family = family_status[card["family_id"]]
        missing = sorted(set(card["requires"]) - capabilities)
        status = ("other_stage" if family["status"] == "other_stage" else
                  "needs_data" if family["status"] == "needs_data" or missing else "ready")
        missing = sorted(set(missing) | set(family["missing_capabilities"])) if status == "needs_data" else []
        result.append({"method_id": card["id"], "family_id": card["family_id"],
                       "status": status, "missing_capabilities": missing,
                       "reason": family["reason"] if status == "other_stage" else
                                 f"requires {', '.join(missing)}" if missing else "requirements available"})
    return result


def decision_applicability(snapshot: dict, catalog: dict) -> list[dict]:
    """Identify research checks relevant to the declared task data and stage."""
    validate_catalog(catalog)
    stage = snapshot.get("stage")
    capabilities = set(snapshot.get("capabilities", []))
    result = []
    for check in catalog.get("decision_checks", []):
        missing = sorted(set(check["requires"]) - capabilities)
        status = ("other_stage" if check["stages"] and stage not in check["stages"] else
                  "not_triggered" if missing else "ready")
        result.append({"check_id": check["id"], "status": status,
                       "missing_capabilities": missing if status == "not_triggered" else []})
    return result


def validate_research(proposal: dict, snapshot: dict, catalog: dict) -> dict:
    """Validate a falsifiable experiment without making families a whitelist."""
    research = proposal.get("research")
    fields = ("direction", "mechanism", "why_now", "data_rationale", "comparison",
              "expected_result", "falsification")
    if not isinstance(research, dict) or any(
            not isinstance(research.get(key), str) or not research[key].strip() for key in fields):
        raise ValueError("experiment research needs direction, mechanism, why_now, data_rationale, "
                         "comparison, expected_result and falsification")
    inputs = research.get("input_fields")
    available = set(snapshot.get("fields", []))
    if not isinstance(inputs, list) or any(not isinstance(name, str) or name not in available for name in inputs):
        raise ValueError("research.input_fields must name task fields; request absent data instead")
    alternatives = research.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives or any(
            not isinstance(item, dict) or any(not isinstance(item.get(key), str) or not item[key].strip()
                                             for key in ("direction", "mechanism", "reason"))
            for item in alternatives):
        raise ValueError("research.alternatives needs a direction, mechanism and reason")
    family_id = research.get("family_id")
    if family_id is not None:
        available_families = {entry["family_id"]: entry for entry in applicability(snapshot, catalog)}
        if family_id not in available_families:
            raise ValueError("unknown family_id; omit it for a new research direction")
        entry = available_families[family_id]
        if entry["status"] != "ready":
            raise ValueError(f"family {family_id} is {entry['status']}: {entry['reason']}")
    method_id = research.get("method_id")
    if method_id is not None:
        methods = {entry["method_id"]: entry for entry in method_applicability(snapshot, catalog)}
        if method_id not in methods:
            raise ValueError("unknown method_id; omit it for a new research direction")
        entry = methods[method_id]
        if entry["status"] != "ready":
            raise ValueError(f"method {method_id} is {entry['status']}: {entry['reason']}")
        if family_id is not None and entry["family_id"] != family_id:
            raise ValueError("method_id and family_id disagree")
    return {**{key: research[key].strip() for key in fields}, "input_fields": inputs,
            "alternatives": [{key: item[key].strip() for key in ("direction", "mechanism", "reason")}
                             for item in alternatives], **({"family_id": family_id} if family_id else {}),
            **({"method_id": method_id} if method_id else {})}
