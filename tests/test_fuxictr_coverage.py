"""Coverage is executable source availability, never a paper reproduction claim."""
import copy
import pytest

from model_evo_harness.catalog import (
    coverage_report, load_catalog, load_guide, model_api, read_references, validate_catalog,
    method_applicability,
)


def test_all_57_source_entries_have_explicit_native_recipes():
    catalog = load_catalog()
    inventory = catalog["fuxictr_inventory"]
    assert len(inventory["entries"]) == 57
    assert len({row["name"] for row in inventory["entries"]}) == 57
    report = coverage_report(catalog)["fuxictr"]
    assert report["entry_count"] == 57
    assert report["new_method_count"] == 43
    assert report["unmapped_entries"] == []
    assert report["missing_frameworks"] == []
    assert all(row["recipe"] and row["limitations"] for row in inventory["entries"])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_new_methods_deliver_complete_source_and_contracts(framework):
    catalog = load_catalog()
    cards = [card for card in catalog["method_cards"] if card.get("source_repo") == "fuxictr"]
    assert len(cards) == 43
    methods = {card["id"] for card in cards}
    api = model_api(catalog, framework=framework, method_ids=methods)
    assert set(api) == methods
    for card in cards:
        entry = api[card["id"]]
        assert entry["constructor"] != "inherited"
        bundle = read_references(catalog, {"framework": framework, "method_ids": [card["id"]]})
        content = bundle["files"][entry["file"]]["content"]
        namespace = {}
        exec(compile(content, entry["file"], "exec"), namespace)
        assert callable(namespace[entry["symbol"].split(":")[1]])
        assert bundle["contracts"][card["id"]]["limitations"]
        assert card["id"] in load_guide(card["family_id"])


def test_inventory_rejects_missing_duplicate_or_unknown_recipe():
    catalog = load_catalog()
    for mutation in ("missing", "duplicate", "unknown"):
        broken = copy.deepcopy(catalog)
        rows = broken["fuxictr_inventory"]["entries"]
        if mutation == "missing":
            rows.pop()
        elif mutation == "duplicate":
            rows[-1] = rows[0]
        else:
            rows[0]["method_ids"] = ["invented_method"]
        with pytest.raises(ValueError, match="FuxiCTR"):
            validate_catalog(broken)


def test_local_tensor_construction_does_not_require_new_business_data():
    catalog = load_catalog()
    snapshot = {"stage": "policy", "capabilities": ["tabular_features", "binary_outcome_labels"]}
    status = {row["method_id"]: row for row in method_applicability(snapshot, catalog)}
    for method in ("gated_cross", "instance_mask", "wukong_block", "cross_deep_bridge",
                   "ffm_pairs", "euler_interaction", "ensemble_consistency", "grouped_regularization"):
        assert status[method]["status"] == "ready", method
    assert status["bst_history_encoder"]["status"] == "needs_data"
    assert status["pretrained_id_fusion"]["status"] == "needs_data"
    api = model_api(catalog, framework="pytorch", method_ids={"instance_mask"})
    assert "hidden_representation" in api["instance_mask"]["required_host_tensors"]


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_provider_delivers_new_structure_and_objective_sources(framework):
    from model_evo_harness.provider import propose_with_references
    calls = []
    def complete(context):
        calls.append(context)
        return {"action": "experiment", "research": {
            "method_id": "instance_mask", "model_design": {"components": [
                {"reference_method_id": "ensemble_consistency"}]}}}
    result = propose_with_references(complete, {}, catalog=load_catalog(), framework=framework)
    assert len(calls) == 2
    for module in ("ctr_composition", "ctr_training"):
        assert f"models/{framework}/{module}.py" in result["reference_reads"]
    assert "hidden_representation" in calls[1]["reference_contracts"]["instance_mask"]["required_host_tensors"]
