"""Training references reach the Agent with prerequisites and source boundaries."""

import ast
import hashlib
import sys

import pytest

from model_evo_harness import analysis_api, load_catalog, model_api, read_references
from model_evo_harness.catalog import common_knowledge, method_applicability, training_applicability
from model_evo_harness.provider import OpenAICompatibleAgent


MODELS = {"prunable_embedding", "conditional_negative_sampling"}
METHODS = MODELS | {"binary_prediction_diagnostics"}
REQUIRES = {
    "prunable_embedding": {"id_vocabulary", "embedding_initialization", "observed_outcome_labels"},
    "conditional_negative_sampling": {"implicit_feedback", "item_catalog", "negative_sampler_definition"},
    "binary_prediction_diagnostics": {"paired_binary_predictions", "observed_outcome_labels"},
}


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_sources_and_apis_are_separate_complete_and_framework_native(framework):
    catalog = load_catalog()
    assert METHODS <= {card["id"] for card in catalog["method_cards"]}
    models = model_api(catalog, framework=framework, method_ids=METHODS)
    analysis = analysis_api(catalog, framework=framework, method_ids=METHODS)
    assert set(models) == MODELS
    assert set(analysis) == METHODS - MODELS
    references = read_references(catalog, {"framework": framework, "method_ids": sorted(METHODS)})
    for method, api in {**models, **analysis}.items():
        assert api["constructor"] != "inherited"
        assert api["limitations"] and api["training_support"]["requires_host"]
        assert api["output_contract"] == references["contracts"][method]["output_contract"]
        source = references["files"][api["file"]]
        assert source["sha256"] == hashlib.sha256(source["content"].encode()).hexdigest()
        allowed = sys.stdlib_module_names | ({"torch" if framework == "pytorch" else "tensorflow"}
                                            if method in MODELS else set())
        for node in ast.walk(ast.parse(source["content"])):
            if isinstance(node, ast.Import):
                assert {x.name.split(".")[0] for x in node.names} <= allowed
            elif isinstance(node, ast.ImportFrom):
                assert not node.level and node.module.split(".")[0] in allowed
    assert "from_mask" in models["prunable_embedding"]["methods"]
    assert "sample" in models["conditional_negative_sampling"]["methods"]
    assert analysis["binary_prediction_diagnostics"]["file"] == "analysis/prediction_diagnostics.py"


def test_missing_input_contracts_are_not_inferred_from_field_names():
    catalog = load_catalog()
    for method, capabilities in REQUIRES.items():
        for missing in capabilities:
            screen = {row["method_id"]: row for row in method_applicability(
                {"stage": "retrieval", "capabilities": sorted(capabilities - {missing}),
                 "fields": ["user_id", "label", "prediction"]}, catalog)}
            assert method in screen
            assert screen[method]["status"] == "needs_data"
            assert missing in screen[method]["missing_capabilities"]
        ready = {row["method_id"]: row for row in method_applicability(
            {"stage": "retrieval", "capabilities": sorted(capabilities)}, catalog)}[method]
        assert ready["status"] == "ready"
        assert ready["kind"] == ("model" if method in MODELS else "analysis")


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
@pytest.mark.parametrize("method", sorted(MODELS))
def test_training_component_source_is_delivered_before_proposal(framework, method):
    catalog = load_catalog()
    agent = OpenAICompatibleAgent("https://example.test/v1", "test", "model")
    contexts = []

    def complete(instruction, context, effort):
        contexts.append(context)
        return {"action": "experiment", "research": {"model_design": {"components": [
            {"id": "local", "reference_method_id": method}]}}, "candidate": {}}

    agent._complete = complete
    agent.propose({"catalog": catalog, "task": {"framework": framework}})
    assert len(contexts) == 2
    api = model_api(catalog, framework=framework, method_ids={method})[method]
    assert api["file"] in contexts[1]["reference_material"]


def test_new_controls_reach_existing_training_and_common_knowledge_contexts():
    catalog = load_catalog()
    patterns = {p["pattern_id"]: p for p in training_applicability(
        {"stage": "retrieval", "capabilities": sorted(set.union(*REQUIRES.values()))}, catalog)}
    assert patterns["conditional_negative_distribution"]["status"] == "ready"
    assert patterns["embedding_mask_retraining"]["status"] == "ready"
    guides = common_knowledge()
    assert "with replacement" in guides["sampling_and_hard_examples"]
