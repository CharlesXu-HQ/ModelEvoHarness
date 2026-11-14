"""A framework-neutral workflow is delivered once without opening arbitrary paths."""
import pytest

from model_evo_harness.catalog import load_catalog, read_references, validate_catalog


def registered_shared_source():
    data = load_catalog()
    # Exercise the boundary with an existing analysis ID, independently of new card registration.
    entry = next(item for item in data["analysis_implementations"]
                 if item["framework"] == "pytorch")
    entry["file"] = "analysis/cost_aware_search.py"
    entry["symbol"] = "model_evo_harness.analysis.cost_aware_search:CostAwareFeatureSearch"
    return data, entry["id"]


def test_registered_shared_analysis_file_and_symbol_are_accepted():
    data, method = registered_shared_source()
    validate_catalog(data)
    bundle = read_references(data, {"framework": "pytorch", "method_ids": [method]})
    source = bundle["files"]["analysis/cost_aware_search.py"]
    assert "class CostAwareFeatureSearch" in source["content"]
    assert len(source["sha256"]) == 64


@pytest.mark.parametrize("path", ["analysis/../provider.py", "analysis/../../secrets.py",
                                   "analysis//cost_aware_search.py", "analysis/cost-aware-search.py",
                                   "models/cost_aware_search.py", "/analysis/cost_aware_search.py"])
def test_shared_analysis_rejects_paths_outside_source_boundary(path):
    data, method = registered_shared_source()
    entry = next(item for item in data["analysis_implementations"]
                 if item["id"] == method and item["framework"] == "pytorch")
    entry["file"] = path
    with pytest.raises(ValueError):
        validate_catalog(data)
    with pytest.raises(ValueError, match="reference file"):
        read_references(data, {"framework": "pytorch", "method_ids": [method]})


def test_shared_analysis_requires_exact_matching_symbol_module():
    data, method = registered_shared_source()
    entry = next(item for item in data["analysis_implementations"]
                 if item["id"] == method and item["framework"] == "pytorch")
    entry["symbol"] = "model_evo_harness.analysis.other_module:CostAwareFeatureSearch"
    with pytest.raises(ValueError):
        validate_catalog(data)
