"""The query bridge must be discoverable and extractable without a DL runtime."""

import pytest

from model_evo_harness import load_catalog, model_api, read_references
from model_evo_harness.catalog import method_applicability


METHODS = {"queryformer_co_transformer", "queryformer_history_retrieval",
           "queryformer_query_generator", "queryformer_query_bridge"}


def test_queryformer_source_and_contracts_reach_the_agent():
    catalog = load_catalog()
    assert METHODS <= {card["id"] for card in catalog["method_cards"]}
    for framework in ("pytorch", "tensorflow"):
        apis = model_api(catalog, framework=framework, method_ids=METHODS)
        assert set(apis) == METHODS
        refs = read_references(catalog, {"framework": framework, "method_ids": sorted(METHODS)})
        assert len(refs["files"]) == 1
        for method, api in apis.items():
            assert api["local_guide"] and api["structure_pattern"]
            assert refs["contracts"][method]["output_contract"] == api["output_contract"]
            compile(refs["files"][api["file"]]["content"], api["file"], "exec")


def test_queryformer_field_component_is_available_without_histories():
    task = {"stage": "ranking", "capabilities": ["tabular_features", "user_tokens", "item_tokens"]}
    states = {x["method_id"]: x for x in method_applicability(task, load_catalog())}
    assert METHODS <= states.keys()
    assert states["queryformer_co_transformer"]["status"] == "ready"
    for method in METHODS - {"queryformer_co_transformer"}:
        assert states[method]["status"] == "needs_data"
        assert "event_sequence" in states[method]["missing_capabilities"]


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_extracted_query_bridge_runs_without_repository_imports(framework):
    refs = read_references(load_catalog(), {"framework": framework,
                                          "method_ids": ["queryformer_query_bridge"]})
    namespace = {"__name__": "extracted_queryformer"}
    for path, record in refs["files"].items():
        exec(compile(record["content"], path, "exec"), namespace)
    if framework == "pytorch":
        import torch
        ones = lambda shape: torch.ones(shape)
        mask = lambda shape: torch.ones(shape, dtype=torch.bool)
    else:
        import tensorflow as tf
        ones = lambda shape: tf.ones(shape)
        mask = lambda shape: tf.ones(shape, tf.bool)
    bridge = namespace["QueryFormerQueryBridge"](4, 2, 1)
    result = bridge(ones((2, 2, 4)), ones((2, 1, 4)), [ones((2, 3, 4))],
                    mask((2, 2)), mask((2, 1)), [mask((2, 3))])
    assert tuple(result["queries"].shape) == (2, 1, 2, 4)
    assert tuple(result["context_tokens"].shape) == (2, 4, 4)
