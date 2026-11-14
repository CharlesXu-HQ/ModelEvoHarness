"""New recommendation mechanisms must be discoverable with honest input contracts."""

import ast
import hashlib
import sys

import pytest

from model_evo_harness import analysis_api, load_catalog, model_api, read_references
from model_evo_harness.catalog import method_applicability
from model_evo_harness.provider import OpenAICompatibleAgent


NETWORKS = {"coaction", "comirec_sa", "comirec_dr"}
METHODS = NETWORKS | {"cold_feature_search"}
COLD_REQUIRES = {"training_feature_saliency", "feature_group_lineage",
                 "protected_interaction_groups", "retrain_evaluate_callback", "serving_profile"}


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_components_are_available_with_native_sources_and_composition_contracts(framework):
    catalog = load_catalog()
    assert METHODS <= {card["id"] for card in catalog["method_cards"]}
    models = model_api(catalog, framework=framework, method_ids=METHODS)
    analysis = analysis_api(catalog, framework=framework, method_ids=METHODS)
    assert set(models) == NETWORKS
    assert set(analysis) == {"cold_feature_search"}
    refs = read_references(catalog, {"framework": framework, "method_ids": sorted(METHODS)})
    assert set(refs["contracts"]) == METHODS
    for method, api in {**models, **analysis}.items():
        assert api["constructor"] != "inherited"
        assert api["local_guide"]
        assert api["training_support"]["requires_host"]
        assert refs["contracts"][method]["output_contract"] == api["output_contract"]
        source = refs["files"][api["file"]]
        assert source["sha256"] == hashlib.sha256(source["content"].encode()).hexdigest()
        tree = ast.parse(source["content"])
        assert any(isinstance(node, ast.ClassDef) and node.name == api["symbol"].split(":")[1]
                   for node in tree.body)
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not node.level
                imported_roots.add(node.module.split(".")[0])
        allowed_imports = sys.stdlib_module_names | ({"torch" if framework == "pytorch"
                                                   else "tensorflow"} if method in NETWORKS else set())
        assert imported_roots <= allowed_imports
        if method in NETWORKS:
            assert api["file"].startswith(f"models/{framework}/")
            assert "structure_pattern" in api
        else:
            assert api["file"] == "analysis/cost_aware_search.py"
            assert api["symbol"] == "model_evo_harness.analysis.cost_aware_search:CostAwareFeatureSearch"
            assert "__call__" in api["methods"]
            assert "analysis_pattern" in api and "structure_pattern" not in api
    assert models["coaction"]["output_contract"]["shape"] == "[B,T,C]"
    for method in ("comirec_sa", "comirec_dr"):
        assert models[method]["structure_pattern"] == "multiple_interest_retrieval"
        assert models[method]["output_contract"]["fields"] == {
            "interests": "[B,K,D]", "interest_mask": "bool [B,K]", "assignments": "[B,K,T]"}
        assert "target" not in next(value for key, value in models[method]["methods"].items()
                                    if key in ("forward", "call"))


def test_capability_screening_distinguishes_business_inputs_from_host_tensor_construction():
    catalog = load_catalog()

    def status(stage, capabilities):
        return {row["method_id"]: row for row in method_applicability(
            {"stage": stage, "capabilities": capabilities}, catalog)}

    ranking = status("ranking", ["tabular_features"])
    assert ranking["coaction"]["missing_capabilities"] == ["meaningful_feature_pairs"]
    ready = status("ranking", ["tabular_features", "meaningful_feature_pairs"])["coaction"]
    assert ready["status"] == "ready"
    assert ready["frameworks"] == ["pytorch", "tensorflow"]
    retrieval = status("retrieval", [])
    for method in ("comirec_sa", "comirec_dr"):
        assert set(retrieval[method]["missing_capabilities"]) == {"event_sequence", "item_catalog"}
        assert status("retrieval", ["event_sequence", "item_catalog"])[method]["status"] == "ready"
        assert ranking[method]["status"] == "other_stage"
    cold = ranking["cold_feature_search"]
    assert cold["kind"] == "analysis"
    assert set(cold["missing_capabilities"]) == COLD_REQUIRES
    for missing in COLD_REQUIRES:
        row = status("ranking", sorted(COLD_REQUIRES - {missing}))["cold_feature_search"]
        assert row["missing_capabilities"] == [missing]
    assert status("ranking", sorted(COLD_REQUIRES))["cold_feature_search"]["status"] == "ready"


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
@pytest.mark.parametrize("method", sorted(NETWORKS))
def test_provider_delivers_local_component_source_before_accepting_proposal(framework, method):
    catalog = load_catalog()
    contexts = []
    agent = OpenAICompatibleAgent("https://example.test/v1", "test", "model")

    def complete(instruction, context, effort):
        contexts.append(context)
        return {"action": "experiment", "research": {"model_design": {"components": [
            {"id": "local", "reference_method_id": method}]}}, "candidate": {}}

    agent._complete = complete
    proposal = agent.propose({"catalog": catalog, "task": {"framework": framework}})
    assert len(contexts) == 2
    api = model_api(catalog, framework=framework, method_ids={method})[method]
    material = contexts[1]["reference_material"][api["file"]]
    assert proposal["reference_reads"][api["file"]] == hashlib.sha256(
        material["content"].encode()).hexdigest()
    assert contexts[1]["reference_contracts"][method]["required_host_tensors"]


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_provider_delivers_shared_cost_search_as_analysis(framework):
    catalog = load_catalog()
    contexts = []
    agent = OpenAICompatibleAgent("https://example.test/v1", "test", "model")

    def complete(instruction, context, effort):
        contexts.append(context)
        if len(contexts) == 1:
            return {"action": "read_reference", "framework": framework,
                    "method_ids": ["cold_feature_search"]}
        return {"action": "stop", "reason": "Host must run the physical retraining callbacks."}

    agent._complete = complete
    proposal = agent.propose({"catalog": catalog, "task": {"framework": framework}})
    path = "analysis/cost_aware_search.py"
    assert len(contexts) == 2
    assert len(proposal["reference_reads"][path]) == 64
    contract = contexts[1]["reference_contracts"]["cold_feature_search"]
    assert "analysis_pattern" in contract and "structure_pattern" not in contract
    assert not any("cold_feature_search" in pattern["method_ids"]
                   for pattern in catalog["structure_patterns"])


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_existing_mechanism_options_are_visible_without_runtime_imports(framework):
    catalog = load_catalog()
    api = model_api(catalog, framework=framework,
                    method_ids={"xdeepfm", "sdim_collision_pooling"})
    for method, option, expected in (("xdeepfm", "cin_activation", "relu"),
                                     ("sdim_collision_pooling", "normalize_per_hash", False)):
        args = ast.parse("def " + api[method]["constructor"] + ": pass").body[0].args
        defaults = dict(zip((arg.arg for arg in args.args[-len(args.defaults):]), args.defaults))
        assert ast.literal_eval(defaults[option]) == expected
    assert "identity" in " ".join(api["xdeepfm"]["limitations"])
    assert "per_hash" in " ".join(api["sdim_collision_pooling"]["limitations"])
    assert "no optional per-hash" not in " ".join(api["sdim_collision_pooling"]["limitations"])
