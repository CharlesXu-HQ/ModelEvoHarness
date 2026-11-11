"""Discount-response proposals require real input contracts and native source."""

import pytest

from model_evo_harness import analysis_api, load_catalog, model_api, read_references
from model_evo_harness.catalog import method_applicability, training_applicability
from model_evo_harness.provider import OpenAICompatibleAgent


METHODS = {"danet_discount_preference", "danet_conversion_head"}
BRANCH_INPUTS = ["item_discount_history", "regular_time_grid", "purchase_history",
                 "aligned_purchase_discount_price_category", "target_item_embedding",
                 "promotion_context", "sequence_mask", "predecision_feature_cutoff"]


def test_marketing_branch_requires_history_semantics_not_generic_tabular_fields():
    catalog = load_catalog()
    task = {"stage": "policy", "capabilities": ["tabular_features", "observed_outcome_labels"]}
    state = {entry["method_id"]: entry for entry in method_applicability(task, catalog)}
    assert METHODS <= state.keys()
    assert state["danet_discount_preference"]["status"] == "needs_data"
    assert set(BRANCH_INPUTS) <= set(state["danet_discount_preference"]["missing_capabilities"])
    task["capabilities"] += BRANCH_INPUTS
    state = {entry["method_id"]: entry for entry in method_applicability(task, catalog)}
    assert state["danet_discount_preference"]["status"] == "ready"
    assert state["danet_conversion_head"]["status"] == "needs_data"
    task["capabilities"] += ["interest_representation", "discount_preference_representation"]
    state = {entry["method_id"]: entry for entry in method_applicability(task, catalog)}
    assert all(state[method]["status"] == "ready" and state[method]["kind"] == "model"
               for method in METHODS)


def test_auxiliary_objective_requires_realized_discount_labels_independently_of_branch():
    catalog = load_catalog()
    task = {"stage": "policy", "capabilities": BRANCH_INPUTS + ["observed_outcome_labels",
                                                               "discount_prediction"]}
    state = {entry["pattern_id"]: entry for entry in training_applicability(task, catalog)}
    assert "purchase_discount_auxiliary" in state
    assert state["purchase_discount_auxiliary"]["status"] == "needs_data"
    assert "realized_discount_labels" in state["purchase_discount_auxiliary"]["missing_capabilities"]
    task["capabilities"].append("realized_discount_labels")
    state = {entry["pattern_id"]: entry for entry in training_applicability(task, catalog)}
    assert state["purchase_discount_auxiliary"]["status"] == "ready"


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_discount_methods_are_model_components_with_complete_source(framework):
    catalog = load_catalog()
    apis = model_api(catalog, framework=framework, method_ids=METHODS)
    assert set(apis) == METHODS
    assert analysis_api(catalog, framework=framework, method_ids=METHODS) == {}
    assert "objective" in apis["danet_conversion_head"]["methods"]
    references = read_references(catalog, {"framework": framework, "method_ids": sorted(METHODS)})
    assert set(references["files"]) == {f"models/{framework}/discount_aware.py"}
    for method in METHODS:
        assert apis[method]["local_guide"] == "knowledge/discount_response.md"
        assert references["contracts"][method]["output_contract"] == apis[method]["output_contract"]
        assert references["contracts"][method]["limitations"]


@pytest.mark.parametrize("method", sorted(METHODS))
def test_discount_source_is_delivered_before_final_proposal(method):
    catalog = load_catalog()
    agent = OpenAICompatibleAgent("https://example.test/v1", "test", "model")
    calls = []

    def complete(instruction, context, effort):
        calls.append(context)
        return {"action": "experiment", "research": {"method_id": method}, "candidate": {}}

    agent._complete = complete
    proposal = agent.propose({"catalog": catalog, "task": {"framework": "pytorch"}})
    assert len(calls) == 2
    path = "models/pytorch/discount_aware.py"
    assert path in calls[1]["reference_material"]
    assert len(proposal["reference_reads"][path]) == 64
