"""Research tools must reach the Agent with eligibility and honest scope."""

import ast

import pytest

from model_evo_harness import load_catalog, model_api, read_references
from model_evo_harness.catalog import common_knowledge, method_applicability
from model_evo_harness.provider import OpenAICompatibleAgent


METHODS = {
    "representation_diagnostics", "gradient_feature_selection",
    "progressive_interaction", "contextual_history_encoding",
    "relation_history_attention",
}


def test_research_components_are_registered_with_native_sources_and_boundaries():
    catalog = load_catalog()
    assert METHODS <= {card["id"] for card in catalog["method_cards"]}
    for framework in ("pytorch", "tensorflow"):
        apis = model_api(catalog, framework=framework, method_ids=METHODS)
        assert set(apis) == METHODS
        for method in sorted(METHODS):
            refs = read_references(catalog, {"framework": framework, "method_ids": [method]})
            api = apis[method]
            if method != "gradient_feature_selection":
                assert api["constructor"] != "inherited"
            if method in {"representation_diagnostics", "gradient_feature_selection"}:
                assert "__call__" in api["methods"]
            assert api["local_guide"] and api["structure_pattern"]
            assert refs["contracts"][method]["limitations"]
            assert refs["contracts"][method]["output_contract"] == api["output_contract"]
            for path, entry in refs["files"].items():
                assert path.startswith(f"models/{framework}/")
                ast.parse(entry["content"])


def test_missing_observations_and_histories_are_explicit_prerequisites():
    catalog = load_catalog()
    task = {"stage": "ranking", "capabilities": ["tabular_features", "field_tokens"]}
    state = {x["method_id"]: x for x in method_applicability(task, catalog)}
    assert METHODS <= state.keys()
    assert state["progressive_interaction"]["status"] == "ready"
    for method in METHODS - {"progressive_interaction"}:
        assert state[method]["status"] == "needs_data"
    assert "activation_gradients" in state["gradient_feature_selection"]["missing_capabilities"]
    assert "intermediate_representations" in state["representation_diagnostics"]["missing_capabilities"]
    assert "aligned_event_context" in state["contextual_history_encoding"]["missing_capabilities"]
    assert "event_sequence" in state["relation_history_attention"]["missing_capabilities"]
    task["capabilities"] += ["intermediate_representations", "activation_gradients",
        "event_sequence", "aligned_event_context", "sequence_mask", "user_tokens", "item_tokens"]
    state = {x["method_id"]: x for x in method_applicability(task, catalog)}
    assert all(state[method]["status"] == "ready" for method in METHODS)


def test_diagnostics_are_available_across_task_stages():
    catalog = load_catalog()
    for stage in ("retrieval", "ranking", "policy", "advertising", "generation", "search"):
        state = {x["method_id"]: x for x in method_applicability({"stage": stage,
            "capabilities": ["intermediate_representations"]}, catalog)}
        assert "representation_diagnostics" in state
        assert state["representation_diagnostics"]["status"] == "ready"


def test_agent_receives_diagnostic_reasoning_and_feature_protection_guidance():
    guides = common_knowledge()
    assert "evidence_driven_iteration" in guides
    guide = guides["evidence_driven_iteration"]
    for required in ("observed", "ablation", "protected_groups", "training",
                     "inference", "dataset", "expert"):
        assert required in guide


@pytest.mark.parametrize("method", sorted(METHODS))
def test_declared_method_supplies_real_source_before_final_proposal(method):
    catalog = load_catalog()
    agent = OpenAICompatibleAgent("https://example.test/v1", "test", "model")
    calls = []

    def complete(instruction, context, effort):
        calls.append(context)
        return {"action": "experiment", "research": {"method_id": method}, "candidate": {}}

    agent._complete = complete
    proposal = agent.propose({"catalog": catalog, "task": {"framework": "pytorch"}})
    assert len(calls) == 2
    expected = model_api(catalog, framework="pytorch", method_ids={method})[method]
    assert expected["file"] in calls[1]["reference_material"]
    assert len(proposal["reference_reads"][expected["file"]]) == 64
