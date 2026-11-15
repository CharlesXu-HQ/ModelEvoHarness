"""Analysis sources are available to the Agent without becoming model components."""

import copy

import pytest

from model_evo_harness import catalog
from model_evo_harness.provider import OpenAICompatibleAgent


TOOLS = {"representation_diagnostics", "gradient_feature_selection", "cold_feature_search",
         "binary_prediction_diagnostics"}


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_analysis_has_separate_manifest_api_and_patterns(framework):
    data = catalog.load_catalog()
    assert "analysis_implementations" in data
    assert not TOOLS & {entry["id"] for entry in data["model_implementations"]}
    assert {entry["id"] for entry in data["analysis_implementations"]} == TOOLS
    assert catalog.model_api(data, framework=framework, method_ids=TOOLS) == {}
    api = catalog.analysis_api(data, framework=framework, method_ids=TOOLS)
    assert set(api) == TOOLS
    for method, entry in api.items():
        if method == "cold_feature_search":
            assert entry["file"] == "analysis/cost_aware_search.py"
            assert entry["symbol"] == "model_evo_harness.analysis.cost_aware_search:CostAwareFeatureSearch"
        elif method == "binary_prediction_diagnostics":
            assert entry["file"] == "analysis/prediction_diagnostics.py"
            assert entry["symbol"] == "model_evo_harness.analysis.prediction_diagnostics:BinaryPredictionDiagnostics"
        else:
            assert entry["file"].startswith(f"analysis/{framework}/")
            assert entry["symbol"].startswith(f"model_evo_harness.analysis.{framework}.")
        assert "__call__" in entry["methods"]
        assert "analysis_pattern" in entry
        assert "structure_pattern" not in entry
    assert not TOOLS & {method for pattern in data["structure_patterns"]
                        for method in pattern["method_ids"]}
    assert {method for pattern in data["analysis_patterns"]
            for method in pattern["method_ids"]} == TOOLS


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_analysis_sources_reach_provider_with_contracts_and_real_hashes(framework):
    data = catalog.load_catalog()
    agent = OpenAICompatibleAgent("https://example.test/v1", "test", "model")
    calls = []

    def complete(instruction, context, effort):
        assert "analysis_api" in instruction
        calls.append(context)
        if len(calls) == 1:
            return {"action": "read_reference", "framework": framework,
                    "method_ids": sorted(TOOLS)}
        return {"action": "stop", "reason": "Host activation capture is needed."}

    agent._complete = complete
    proposal = agent.propose({"catalog": data, "task": {"framework": framework}})
    path = f"analysis/{framework}/research_diagnostics.py"
    assert len(calls) == 2
    assert path in calls[1]["reference_material"]
    assert "analysis/prediction_diagnostics.py" in calls[1]["reference_material"]
    assert len(proposal["reference_reads"][path]) == 64
    assert calls[1]["reference_contracts"]["representation_diagnostics"][
        "analysis_pattern"] == "observed_representation_quality"


@pytest.mark.parametrize("path", ["analysis/../provider.py", "analysis/tensorflow/research_diagnostics.py",
                                  "knowledge/pytorch/research_diagnostics.py"])
def test_analysis_reader_keeps_framework_and_path_boundary(path):
    data = catalog.load_catalog()
    assert "analysis_implementations" in data
    data["analysis_implementations"][0]["file"] = path
    with pytest.raises(ValueError, match="reference file"):
        catalog.read_references(data, {"framework": "pytorch",
                                      "method_ids": ["representation_diagnostics"]})


def test_analysis_cannot_be_registered_twice_or_as_a_model():
    data = catalog.load_catalog()
    assert "analysis_implementations" in data
    data["model_implementations"].append(copy.deepcopy(data["analysis_implementations"][0]))
    with pytest.raises(ValueError):
        catalog.validate_catalog(data)


@pytest.mark.parametrize("method", sorted(TOOLS))
def test_observation_only_method_cannot_count_as_an_optimization_trial(method):
    data = catalog.load_catalog()
    task = {"stage": "ranking", "fields": [], "capabilities": [
        "intermediate_representations", "activation_gradients"]}
    research = {key: "Inspect existing activations" for key in (
        "direction", "mechanism", "why_now", "data_rationale", "comparison",
        "expected_result", "falsification")}
    research.update(method_id=method, input_fields=[], alternatives=[{
        "direction": "local change", "mechanism": "retrain a screened model",
        "reason": "must test the actual prediction change"}])
    with pytest.raises(ValueError, match="analysis tool"):
        catalog.validate_research({"research": research}, task, data)


@pytest.mark.parametrize("method", sorted(TOOLS))
def test_analysis_tool_cannot_be_used_as_a_model_component_reference(method):
    task = {"stage": "ranking", "framework": "pytorch", "fields": [], "capabilities": []}
    research = {key: "Compare a local model change" for key in (
        "direction", "mechanism", "why_now", "data_rationale", "comparison",
        "expected_result", "falsification")}
    component = {"id": "local", "mechanism": "Candidate interaction",
                 "code_sections": ["Candidate.forward"], "input_fields": [],
                 "required_capabilities": [], "reference_method_id": "fm"}
    research.update(input_fields=[], alternatives=[{
        "direction": "loss", "mechanism": "Change the loss", "reason": "Alternative hypothesis"}],
        model_design={"estimator": "prediction", "backbone": "MLP", "change_scope": "initialize",
                      "parent_trial_id": None, "rationale": "Track local changes",
                      "data_fit": "Host provides numeric fields", "comparison_plan": "Fixed protocol",
                      "components": [component], "inheritance": []})
    data = catalog.load_catalog()
    catalog.validate_research({"research": research}, task, data)
    component["reference_method_id"] = method
    with pytest.raises(ValueError, match="component reference_method_id"):
        catalog.validate_research({"research": research}, task, data)
