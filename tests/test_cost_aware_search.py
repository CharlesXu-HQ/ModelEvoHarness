"""Contract tests with declared callback fixtures, not model effectiveness evidence."""

from copy import deepcopy
import importlib

import pytest


def _search(**kwargs):
    module = importlib.import_module("model_evo_harness.analysis.cost_aware_search")
    return module.CostAwareFeatureSearch(**kwargs)


def _context():
    return {
        "task_id": "click-prediction",
        "dataset_id": "snapshot-17",
        "train_split_id": "train-before-cutoff",
        "validation_split_id": "validation-after-cutoff",
        "quality_metric": "auc",
        "quality_direction": "maximize",
        "training_budget_id": "fixed-epochs-and-optimizer",
        "seed": 17,
        "serving_profile": {
            "hardware": "declared-test-device",
            "batch_size": 1,
            "candidate_count": 20,
            "concurrency": 4,
            "latency_statistic": "p95",
            "includes_feature_lookup": True,
        },
    }


def _callback(measurements, calls):
    # A host callback fixture supplies predetermined results. No training occurs.
    def evaluate(*, retained_indices, context):
        calls.append((retained_indices, deepcopy(context)))
        quality, latency = measurements[retained_indices]
        return {
            "retained_indices": retained_indices,
            "retrained": True,
            "physical_pruning": True,
            "implementation_id": "test-backbone-physical-subset-v1",
            "train_run_id": f"fixture-run-{len(calls)}",
            "context": context,
            "validation_quality": quality,
            "latency_ms": latency,
        }
    return evaluate


def _run(search, callback, **kwargs):
    arguments = {"context": _context(), "saliency_split": "train-before-cutoff"}
    arguments.update(kwargs)
    return search([1., 4., 3.], [2, 1], callback, **arguments)


def test_baseline_and_deduplicated_subsets_invoke_host_with_original_order():
    calls = []
    context = _context()
    callback = _callback({(0, 1, 2): (.9, 12.), (1, 2): (.89, 8.), (1,): (.8, 4.)}, calls)
    result = _search(latency_budget_ms=10, quality_floor=.85)(
        [1., 4., 3.], [3, 2, 2, 1, 3], callback, context=context,
        saliency_split=context["train_split_id"])
    assert [indices for indices, _ in calls] == [(0, 1, 2), (1, 2), (1,)]
    assert all(actual == context for _, actual in calls)
    assert result["baseline"]["retained_indices"] == (0, 1, 2)
    assert [trial["feasible"] for trial in result["trials"]] == [False, True, False]
    assert result["selected"]["retained_indices"] == (1, 2)


def test_every_member_of_overlapping_interaction_groups_is_protected():
    calls = []
    measurements = {(0, 1, 2, 3, 4): (.9, 12), (0, 2, 3, 4): (.89, 9),
                    (2, 3, 4): (.88, 7)}
    result = _search(latency_budget_ms=10, max_quality_drop=.03)(
        [9., 9., 0., 0., 0.], [4, 3], _callback(measurements, calls),
        context=_context(), saliency_split="train-before-cutoff",
        protected_groups=[(2, 3), (3, 4)])
    assert [indices for indices, _ in calls] == [(0, 1, 2, 3, 4), (0, 2, 3, 4), (2, 3, 4)]
    assert result["selected"]["retained_indices"] == (0, 2, 3, 4)


def test_protected_indices_and_all_protected_fields_deduplicate_with_baseline():
    calls = []
    result = _search(latency_budget_ms=10, quality_floor=.8)(
        [0., 0., 0.], [3, 3], _callback({(0, 1, 2): (.9, 9)}, calls),
        context=_context(), saliency_split="train-before-cutoff",
        protected_indices=[0], protected_groups=[(1, 2)])
    assert len(calls) == 1
    assert result["selected"]["is_baseline"] is True


@pytest.mark.parametrize("kwargs", [
    {"latency_budget_ms": 0, "quality_floor": .8},
    {"latency_budget_ms": -1, "quality_floor": .8},
    {"latency_budget_ms": float("inf"), "quality_floor": .8},
    {"latency_budget_ms": True, "quality_floor": .8},
    {"latency_budget_ms": 10},
    {"latency_budget_ms": 10, "quality_floor": float("nan")},
    {"latency_budget_ms": 10, "max_quality_drop": -1},
    {"latency_budget_ms": 10, "max_quality_drop": float("inf")},
])
def test_invalid_constraints_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _search(**kwargs)


@pytest.mark.parametrize("scores,budgets,extra", [
    ([], [0], {}),
    ([1., float("nan")], [1], {}),
    ([1., float("inf")], [1], {}),
    ([1., True], [1], {}),
    ([1., 2.], [-1], {}),
    ([1., 2.], [3], {}),
    ([1., 2.], [1.5], {}),
    ([1., 2.], [True], {}),
    ([1., 2.], [1], {"protected_indices": [2]}),
    ([1., 2.], [1], {"protected_indices": [True]}),
    ([1., 2.], [1], {"protected_groups": [(0, 1)]}),
])
def test_invalid_proposals_fail_before_any_host_work(scores, budgets, extra):
    calls = []
    with pytest.raises(ValueError):
        _search(latency_budget_ms=10, quality_floor=.8)(
            scores, budgets, _callback({}, calls), context=_context(),
            saliency_split="train-before-cutoff", **extra)
    assert calls == []


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop("task_id"),
    lambda c: c.update(dataset_id=""),
    lambda c: c.update(train_split_id=c["validation_split_id"]),
    lambda c: c.update(quality_direction="minimize"),
    lambda c: c.pop("training_budget_id"),
    lambda c: c.update(seed=True),
    lambda c: c["serving_profile"].pop("hardware"),
    lambda c: c["serving_profile"].update(batch_size=0),
    lambda c: c["serving_profile"].update(candidate_count=1.5),
    lambda c: c["serving_profile"].update(concurrency=True),
    lambda c: c["serving_profile"].update(latency_statistic=""),
    lambda c: c["serving_profile"].update(includes_feature_lookup="yes"),
])
def test_invalid_context_is_rejected_before_host_work(mutation):
    context = _context()
    mutation(context)
    calls = []
    with pytest.raises(ValueError):
        _run(_search(latency_budget_ms=10, quality_floor=.8), _callback({}, calls), context=context)
    assert calls == []


def test_validation_saliency_is_not_accepted_as_training_evidence():
    calls = []
    with pytest.raises(ValueError, match="saliency_split"):
        _run(_search(latency_budget_ms=10, quality_floor=.8), _callback({}, calls),
             saliency_split="validation-after-cutoff")
    assert calls == []


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(retrained=False),
    lambda r: r.update(retrained=1),
    lambda r: r.pop("retrained"),
    lambda r: r.update(physical_pruning=False),
    lambda r: r.update(physical_pruning=1),
    lambda r: r.update(implementation_id=""),
    lambda r: r.update(train_run_id=""),
    lambda r: r.update(retained_indices=(2, 1, 0)),
    lambda r: r.update(retained_indices=(0, 1)),
    lambda r: r.update(retained_indices=(False, 1, 2)),
    lambda r: r.update(validation_quality=float("nan")),
    lambda r: r.update(validation_quality=float("inf")),
    lambda r: r.update(validation_quality=True),
    lambda r: r.update(latency_ms=float("inf")),
    lambda r: r.update(latency_ms=0),
    lambda r: r.update(latency_ms=-1),
    lambda r: r["context"].update(task_id="another-task"),
    lambda r: r["context"].update(dataset_id="another-dataset"),
    lambda r: r["context"].update(validation_split_id="test"),
    lambda r: r["context"].update(quality_metric="gauc"),
    lambda r: r["context"].update(training_budget_id="extra-epochs"),
    lambda r: r["context"].update(seed=19),
    lambda r: r["context"]["serving_profile"].update(concurrency=8),
    lambda r: r["context"]["serving_profile"].update(hardware="another-device"),
    lambda r: r["context"]["serving_profile"].update(includes_feature_lookup=False),
])
def test_unretrained_or_incomparable_callback_evidence_is_rejected(mutation):
    calls = []
    base = _callback({(0, 1, 2): (.9, 10)}, calls)

    def invalid(**kwargs):
        result = base(**kwargs)
        mutation(result)
        return result

    context = _context()
    with pytest.raises(ValueError):
        _run(_search(latency_budget_ms=10, quality_floor=.8), invalid, context=context)
    assert len(calls) == 1
    assert context == _context()  # Host mutation must not alter the declared context.


def test_quality_best_feasible_selection_and_pareto_frontier_use_measured_cost():
    calls = []
    measurements = {(0, 1, 2, 3): (.94, 15), (0, 1, 2): (.93, 9),
                    (0, 1): (.92, 8), (0,): (.91, 9), (): (.2, 1)}
    result = _search(latency_budget_ms=10, quality_floor=.9, max_quality_drop=.03)(
        [8., 6., 4., 2.], [3, 2, 1, 0], _callback(measurements, calls),
        context=_context(), saliency_split="train-before-cutoff")
    assert result["quality_floor"] == pytest.approx(.91)
    assert result["selected"]["retained_indices"] == (0, 1, 2)
    assert [trial["retained_indices"] for trial in result["pareto_candidates"]] == [(0, 1, 2), (0, 1)]
    assert len(result["trials"]) == 5
    assert result["trials"][-1]["feasible"] is False


def test_no_feasible_result_is_explicit_without_falling_back_to_saliency():
    calls = []
    result = _run(_search(latency_budget_ms=5, max_quality_drop=.01), _callback(
        {(0, 1, 2): (.9, 12), (1, 2): (.89, 8), (1,): (.8, 4)}, calls))
    assert result["selected"] is None
    assert result["pareto_candidates"] == ()
    assert len(result["trials"]) == 3


def test_selection_ties_prefer_latency_then_smaller_subset_independent_of_budget_order():
    measurements = {(0, 1, 2, 3): (.9, 9), (0, 1, 2): (.9, 7), (0, 1): (.9, 7), (0,): (.8, 5)}
    for budgets in ([3, 2, 1], [1, 2, 3]):
        result = _search(latency_budget_ms=10, quality_floor=.8)(
            [1., 1., 1., 1.], budgets, _callback(measurements, []),
            context=_context(), saliency_split="train-before-cutoff")
        assert result["selected"]["retained_indices"] == (0, 1)


def test_callback_failures_propagate_without_manufacturing_measurements():
    def failed(**kwargs):
        raise RuntimeError("host retraining failed")

    with pytest.raises(RuntimeError, match="host retraining failed"):
        _run(_search(latency_budget_ms=10, quality_floor=.8), failed)


def test_reusing_baseline_training_evidence_for_a_subset_is_rejected():
    calls = []
    base = _callback({(0, 1, 2): (.9, 12), (1, 2): (.89, 8)}, calls)

    def reused(**kwargs):
        result = base(**kwargs)
        result["train_run_id"] = "same-run"
        return result

    with pytest.raises(ValueError, match="train_run_id"):
        _run(_search(latency_budget_ms=10, quality_floor=.8), reused)
    assert len(calls) == 2


def test_latency_excluding_feature_lookup_is_labeled_as_model_only():
    context = _context()
    context["serving_profile"]["includes_feature_lookup"] = False
    result = _run(_search(latency_budget_ms=10, quality_floor=.8), _callback(
        {(0, 1, 2): (.9, 12), (1, 2): (.89, 8), (1,): (.8, 4)}, []), context=context)
    assert result["latency_scope"] == "model_only"
