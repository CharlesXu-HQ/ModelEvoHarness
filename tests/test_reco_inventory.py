"""Paper dispositions and every Agent-facing path must agree with bundled code."""
import copy
from importlib.resources import files

import pytest

from model_evo_harness import load_catalog, model_api, paper_coverage_report, read_references
from model_evo_harness.catalog import method_applicability, validate_catalog


def test_all_source_pdfs_have_explicit_dispositions():
    catalog = load_catalog()
    report = paper_coverage_report(catalog)
    assert report["pdf_count"] == 115
    assert report["added_method_count"] == 25
    assert report["dispositions"] == {"added_core": 24, "existing_core": 18,
                                       "shared_mechanism": 14, "excluded": 58, "duplicate": 1}
    papers = catalog["paper_inventory"]["papers"]
    assert {p["id"] for p in papers} == {f"reco-{i:03}" for i in range(1, 116)}
    assert papers[99]["duplicate_of"] == "reco-016"
    assert papers[72]["method_ids"] == ["hstu"]  # existing core retained
    assert papers[75]["exclusion_category"] == "index"  # TIGER is not simply called an LLM


def test_each_added_method_reaches_api_source_guide_and_training_maps():
    catalog = load_catalog()
    new = [c for c in catalog["method_cards"] if c.get("source_repo") == "reco-papers"]
    assert len(new) == 25
    for card in new:
        for framework in ("pytorch", "tensorflow"):
            api = model_api(catalog, framework=framework, method_ids={card["id"]})[card["id"]]
            source = read_references(catalog, {"framework": framework, "method_ids": [card["id"]]})
            assert api["symbol"].split(":")[1] in source["files"][api["file"]]["content"]
            assert source["contracts"][card["id"]]["paper_ids"] == card["paper_ids"]
            assert source["contracts"][card["id"]]["structure_pattern"]
            assert len(files("model_evo_harness").joinpath(api["local_guide"]).read_text().split()) > 150
    for pattern in catalog["training_patterns"]:
        for method_id in pattern.get("method_ids", []):
            assert any(card["id"] == method_id for card in new)


@pytest.mark.parametrize("mutation", ["unknown_method", "unknown_guide", "missing_pdf", "bad_disposition"])
def test_broken_inventory_fails_closed(mutation):
    catalog = copy.deepcopy(load_catalog())
    paper = catalog["paper_inventory"]["papers"][0]
    if mutation == "unknown_method":
        paper["method_ids"] = ["missing"]
    elif mutation == "unknown_guide":
        paper["local_guides"] = ["knowledge/missing.md"]
    elif mutation == "missing_pdf":
        catalog["paper_inventory"]["papers"].pop()
    else:
        paper["disposition"] = "fully_reproduced"
    with pytest.raises(ValueError):
        validate_catalog(catalog)


def test_pal_requires_logged_positions_but_not_a_fabricated_propensity():
    catalog = load_catalog()
    task = {"stage": "ranking", "capabilities": ["logged_feedback", "relevance_logits", "observed_outcome_labels"]}
    report = {r["method_id"]: r for r in method_applicability(task, catalog)}
    assert report["pal"]["status"] == "needs_data"
    assert report["pal"]["missing_capabilities"] == ["logged_positions"]
    task["capabilities"].append("logged_positions")
    report = {r["method_id"]: r for r in method_applicability(task, catalog)}
    assert report["pal"]["status"] == "ready"


def test_adapted_source_keeps_license_in_agent_extraction():
    catalog = load_catalog()
    for framework in ("pytorch", "tensorflow"):
        source = read_references(catalog, {"framework": framework, "method_ids": ["dice"]})
        text = next(iter(source["files"].values()))["content"]
        assert "Copyright (c) 2022 FIB LAB, Tsinghua University" in text
        assert "Permission is hereby granted" in text


def test_new_objectives_are_available_at_their_actual_task_stage():
    catalog = load_catalog()
    task = {"stage": "ranking", "capabilities": ["query_groups", "graded_relevance", "candidate_mask"]}
    report = {row["method_id"]: row for row in method_applicability(task, catalog)}
    assert report["ranknet"]["status"] == "ready"
    assert report["lambdarank"]["status"] == "ready"
    task = {"stage": "retrieval", "capabilities": ["multiple_outcomes", "teacher_logits", "student_logits", "impression_click_and_read_labels"]}
    report = {row["method_id"]: row for row in method_applicability(task, catalog)}
    assert report["dmtl_distillation"]["status"] == "ready"
