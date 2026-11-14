import importlib

import pytest


def roles():
    shared = {"module_path": "encoder.items", "id_domain": "item",
              "vocabulary_digest": "item-v1", "vocabulary_size": 10,
              "width": 8, "padding_id": 0, "oov_id": 1}
    return [{**shared, "id": "target", "kind": "categorical", "fields": ["item"]},
            {**shared, "id": "history", "kind": "sequence", "fields": ["history"]}]


def validator():
    return importlib.import_module("model_evo_harness.embeddings").validate_embedding_roles


def test_same_id_semantics_can_share_across_target_and_history_roles():
    original = roles()
    assert validator()(original, ["item", "history"]) == original
    assert original == roles()


@pytest.mark.parametrize("key,value", [("id_domain", "user"), ("vocabulary_digest", "item-v2"),
                                      ("width", 16), ("padding_id", 2), ("vocabulary_size", 20),
                                      ("oov_id", 2), ("freeze_pretrained", True)])
def test_shared_module_rejects_incompatible_representation_contract(key, value):
    bad = roles()
    bad[1][key] = value
    with pytest.raises(ValueError, match="shared"):
        validator()(bad, ["item", "history"])


@pytest.mark.parametrize("mutation", ["unknown_field", "duplicate_id", "zero_width", "invalid_oov"])
def test_embedding_roles_reject_invalid_mapping(mutation):
    bad = roles()
    if mutation == "unknown_field": bad[0]["fields"] = ["future_purchase"]
    elif mutation == "duplicate_id": bad[1]["id"] = "target"
    elif mutation == "zero_width": bad[0]["width"] = 0
    else: bad[0]["oov_id"] = 10
    with pytest.raises(ValueError):
        validator()(bad, ["item", "history"])


def test_snapshot_validates_embedding_roles_before_baseline():
    from model_evo_harness.engine import _snapshot
    class Task:
        def snapshot(self):
            bad = roles()
            bad[1]["vocabulary_digest"] = "wrong"
            return {"task_id": "coupon", "dataset_digest": "v1", "stage": "policy",
                    "fields": ["item", "history"], "capabilities": [],
                    "objective": {"direction": "max"}, "embedding_roles": bad}
    with pytest.raises(ValueError, match="shared"):
        _snapshot(Task())


def test_proposal_embedding_roles_are_checked_and_retained():
    from model_evo_harness import load_catalog, validate_research
    task = {"fields": ["item", "history"], "capabilities": [], "stage": "policy"}
    research = {"direction": "embedding sharing", "mechanism": "share item vocabulary",
                "why_now": "existing contract", "data_rationale": "same item domain",
                "comparison": "independent embeddings", "expected_result": "less sparse error",
                "falsification": "no gain", "input_fields": ["item", "history"],
                "alternatives": [{"direction": "keep", "mechanism": "independent", "reason": "control"}],
                "embedding_roles": roles()}
    result = validate_research({"research": research}, task, load_catalog())
    assert result["embedding_roles"] == roles()
    research["embedding_roles"][1]["width"] = 4
    with pytest.raises(ValueError, match="shared"):
        validate_research({"research": research}, task, load_catalog())
