"""The Agent must see what each installed model reference can actually do."""

import json
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "src/model_evo_harness/data"


def test_each_method_has_explicit_framework_boundaries():
    cards = json.loads((DATA / "method_cards.json").read_text())["method_cards"]
    cards += json.loads((DATA / "supplemental_method_cards.json").read_text())["method_cards"]
    implementations = json.loads((DATA / "model_implementations.json").read_text())[
        "model_implementations"]
    by_method = {}
    for entry in implementations:
        assert entry["reference_scope"] == "mechanism_reference"
        support = entry["training_support"]
        assert support["mode"] in {"host_trained", "closed_form"}
        assert isinstance(support["requires_host"], list)
        assert all(isinstance(value, str) and value for value in support["requires_host"])
        assert isinstance(entry["limitations"], list) and entry["limitations"]
        assert all(isinstance(value, str) and value for value in entry["limitations"])
        output = entry["output_contract"]
        assert isinstance(output["kind"], str) and output["kind"]
        assert isinstance(output["shape"], str) and output["shape"]
        by_method.setdefault(entry["id"], {})[entry["framework"]] = entry

    assert set(by_method) == {card["id"] for card in cards} | {"two_tower"}
    for method_id, frameworks in by_method.items():
        assert set(frameworks) == {"pytorch", "tensorflow"}, method_id
        assert frameworks["pytorch"]["output_contract"] == frameworks["tensorflow"][
            "output_contract"]
        for entry in frameworks.values():
            if method_id in {"item_cf", "user_cf", "swing"}:
                assert entry["training_support"]["mode"] == "closed_form"
                assert "trainer" not in entry["training_support"]["requires_host"]
            else:
                assert entry["training_support"]["mode"] == "host_trained"
                assert "trainer" in entry["training_support"]["requires_host"]


def test_known_paper_and_task_boundaries_are_not_hidden():
    implementations = json.loads((DATA / "model_implementations.json").read_text())[
        "model_implementations"]
    by_method = {entry["id"]: entry for entry in implementations
                 if entry["framework"] == "pytorch"}
    expected = {
        "dien": "auxiliary",
        "mind": "dynamic",
        "hstu": "end-to-end",
        "dcn_v2": "expert",
        "mlr": "causal",
        "esmm": "joint",
        "youtubednn": "sampled softmax",
    }
    for method_id, phrase in expected.items():
        assert phrase in " ".join(by_method[method_id]["limitations"]).lower()
