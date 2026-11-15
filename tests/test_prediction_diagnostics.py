"""Paired prediction algebra and invalid-data contracts; no model training."""

import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest


def _diagnostic(**kwargs):
    module = importlib.import_module("model_evo_harness.analysis.prediction_diagnostics")
    return module.BinaryPredictionDiagnostics(**kwargs)


def _compare(labels, baseline, candidate, *, eps=1e-15, **kwargs):
    ids = tuple(f"row-{i}" for i in range(len(labels)))
    arguments = {"sample_ids": ids, "baseline_sample_ids": ids,
                 "candidate_sample_ids": ids, "dataset_id": "snapshot-1",
                 "split_id": "validation-1"}
    arguments.update(kwargs)
    return _diagnostic(eps=eps)(labels, baseline, candidate, **arguments)


def test_rank_improvement_and_loss_regression_are_described_without_attribution():
    # Baseline wins three of four cross-class pairs; candidate wins all four,
    # but moves two positive probabilities much further from their labels.
    result = _compare([0, 0, 1, 1], [.1, .4, .3, .8], [.001, .002, .003, .004])
    assert result["baseline"]["auc"] == .75
    assert result["candidate"]["auc"] == 1.
    assert result["baseline"]["log_loss"] == pytest.approx(.5108256237659907)
    assert result["candidate"]["log_loss"] == pytest.approx(2.8334016027951328)
    assert result["delta"]["auc"] == .25
    assert result["delta"]["log_loss"] > 0
    assert result["comparison"] == {
        "auc_direction": "improved", "log_loss_direction": "worsened", "conflict": True}
    assert result["dataset_id"] == "snapshot-1"
    assert result["split_id"] == "validation-1"
    assert result["sample_ids"] == ("row-0", "row-1", "row-2", "row-3")
    assert result["paired_loss_changes"]["improved_count"] == 2
    assert result["paired_loss_changes"]["worsened_count"] == 2
    assert result["paired_loss_changes"]["unchanged_count"] == 0
    assert (result["paired_loss_changes"]["improvement_contribution"]
            + result["paired_loss_changes"]["regression_contribution"]
            == pytest.approx(result["delta"]["log_loss"]))


def test_class_conditional_losses_and_total_contributions_use_distinct_denominators():
    result = _compare([0, 0, 1], [.2, .4, .6], [.1, .3, .8], sample_weight=[1, 2, 3])
    baseline = result["baseline"]
    assert baseline["positive"]["log_loss"] == pytest.approx(.5108256237659907)
    assert baseline["negative"]["log_loss"] == pytest.approx(.414931599615397)
    assert baseline["positive"]["weight_fraction"] == .5
    assert baseline["negative"]["weight_fraction"] == .5
    assert baseline["positive"]["count"] == 1
    assert baseline["negative"]["count"] == 2
    assert baseline["positive"]["log_loss_contribution"] == pytest.approx(.25541281188299536)
    assert baseline["negative"]["log_loss_contribution"] == pytest.approx(.2074657998076985)
    for summary in (baseline, result["candidate"]):
        assert summary["log_loss"] == pytest.approx(
            summary["positive"]["log_loss_contribution"]
            + summary["negative"]["log_loss_contribution"])
    assert result["delta"]["positive_log_loss_contribution"] + result["delta"][
        "negative_log_loss_contribution"] == pytest.approx(result["delta"]["log_loss"])


def test_weighted_auc_assigns_half_credit_to_ties_and_ignores_zero_weight_rows():
    # Cross-class pair mass is 24. Wins carry 2, ties carry (6 + 4) / 2.
    result = _compare([0, 1, 0, 1, 1], [.1, .1, .9, .9, 0], [.5] * 5,
                      sample_weight=[2, 3, 4, 1, 0])
    assert result["baseline"]["auc"] == pytest.approx(7 / 24)
    assert result["candidate"]["auc"] == .5
    assert result["zero_weight_count"] == 1
    assert result["baseline"]["positive"]["active_count"] == 2
    assert sum(result["paired_loss_changes"][key] for key in (
        "improved_count", "worsened_count", "unchanged_count")) == 4


@pytest.mark.parametrize("weights", [[1e308, 1e308], [1e-300, 1e-300], [1e-300, 1e300]])
def test_weight_scaling_does_not_overflow_or_erase_auc_classes(weights):
    result = _compare([0, 1], [.2, .8], [.3, .7], sample_weight=weights)
    assert result["baseline"]["auc"] == 1.
    assert result["baseline"]["log_loss"] == pytest.approx(-math.log(.8))
    assert result["baseline"]["negative"]["log_loss"] == pytest.approx(-math.log(.8))
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("labels,weights", [([1, 1], None), ([0, 1], [0, 1])])
def test_single_effective_class_preserves_loss_and_reports_undefined_auc(labels, weights):
    result = _compare(labels, [.6, .6], [.8, .8], sample_weight=weights)
    assert result["baseline"]["auc"] is None
    assert result["candidate"]["auc"] is None
    assert result["delta"]["auc"] is None
    assert result["comparison"] == {
        "auc_direction": "undefined", "log_loss_direction": "improved", "conflict": None}
    assert result["candidate"]["negative"]["log_loss"] is None
    assert result["candidate"]["negative"]["log_loss_contribution"] == 0.
    assert result["candidate"]["log_loss"] == pytest.approx(-math.log(.8))


def test_correct_and_incorrect_endpoint_predictions_have_finite_declared_clipped_loss():
    result = _compare([0, 1], [0, 1], [1, 0], eps=.01)
    assert result["baseline"]["log_loss"] == pytest.approx(-math.log(.99))
    assert result["candidate"]["log_loss"] == pytest.approx(-math.log(.01))
    assert result["baseline"]["clipped_count"] == 2
    assert result["candidate"]["clipped_count"] == 2
    assert result["baseline"]["auc"] == 1.
    assert result["candidate"]["auc"] == 0.
    assert result["comparison"]["conflict"] is False
    json.dumps(result, allow_nan=False)


def test_tiny_eps_remains_finite_even_when_one_minus_eps_rounds_to_one():
    result = _compare([0, 1], [0, 1], [1, 0], eps=1e-300)
    assert result["candidate"]["log_loss"] == pytest.approx(690.7755278982137)
    assert result["baseline"]["log_loss"] == pytest.approx(1e-300, abs=0)
    assert result["candidate"]["clipped_count"] == 2
    json.dumps(result, allow_nan=False)


def test_probability_clipping_does_not_create_auc_ties():
    result = _compare([0, 1], [.0001, .0002], [.2, .4], eps=.01)
    assert result["baseline"]["clipped_count"] == 2
    assert result["baseline"]["auc"] == 1.
    assert result["candidate"]["auc"] == 1.
    assert result["comparison"]["auc_direction"] == "unchanged"
    assert result["comparison"]["conflict"] is False


def test_probabilities_exactly_at_clip_bounds_are_not_counted_as_clipped():
    result = _compare([0, 1], [.1, .9], [.1, .9], eps=.1)
    assert result["baseline"]["clipped_count"] == 0
    assert result["candidate"]["clipped_count"] == 0


@pytest.mark.parametrize("eps", [1e-15, .01, .1])
@pytest.mark.parametrize("label", [0, 1])
def test_at_and_above_representable_upper_clip_bound_have_identical_loss(eps, label):
    result = _compare([label, 1 - label], [1 - eps, .5], [1., .5], eps=eps)
    name = "positive" if label else "negative"
    assert result["baseline"][name]["log_loss"] == result["candidate"][name]["log_loss"]
    assert result["delta"]["log_loss"] == 0.
    assert result["comparison"]["log_loss_direction"] == "unchanged"
    assert result["paired_loss_changes"]["unchanged_count"] == 2
    assert result["baseline"]["clipped_count"] == 0
    assert result["candidate"]["clipped_count"] == 1


def test_virtual_upper_endpoint_preserves_monotonic_negative_loss_for_tiny_eps():
    result = _compare([0, 1], [math.nextafter(1., 0.), .5], [1., .5], eps=1e-300)
    assert result["candidate"]["negative"]["log_loss"] == pytest.approx(690.7755278982137)
    assert result["delta"]["log_loss"] > 0
    assert result["comparison"]["log_loss_direction"] == "worsened"
    json.dumps(result, allow_nan=False)


def test_unchanged_predictions_have_zero_deltas_and_no_conflict():
    result = _compare([0, 1], [.2, .8], [.2, .8])
    assert result["delta"]["auc"] == 0
    assert result["delta"]["log_loss"] == 0
    assert result["paired_loss_changes"]["unchanged_count"] == 2
    assert result["comparison"]["conflict"] is False


@pytest.mark.parametrize("eps", [0, -.1, .5, 1, float("inf"), float("nan"), True, "0.1"])
def test_invalid_eps_is_rejected(eps):
    with pytest.raises(ValueError, match="eps"):
        _diagnostic(eps=eps)


@pytest.mark.parametrize("extra", [
    {"baseline_sample_ids": ["row-1", "row-0"]},
    {"candidate_sample_ids": ["row-1", "row-0"]},
    {"sample_ids": ["same", "same"], "baseline_sample_ids": ["same", "same"],
     "candidate_sample_ids": ["same", "same"]},
    {"sample_ids": ["", "row-1"]},
    {"sample_ids": [True, "row-1"]},
    {"sample_ids": [[], "row-1"]},
    {"baseline_sample_ids": ["row-0"]},
    {"candidate_sample_ids": ["row-0", "other"]},
    {"dataset_id": " "}, {"dataset_id": 1}, {"split_id": ""},
])
def test_misaligned_or_invalid_identity_is_rejected(extra):
    with pytest.raises(ValueError):
        _compare([0, 1], [.2, .8], [.3, .7], **extra)


@pytest.mark.parametrize("labels,baseline,candidate,weights", [
    ([], [], [], None), ([0, 1], [.2], [.3, .7], None),
    ([0, 1], [.2, .8], [.3], None), ([0, 2], [.2, .8], [.3, .7], None),
    ([0, float("nan")], [.2, .8], [.3, .7], None),
    ([0, True], [.2, .8], [.3, .7], None),
    ([0, 1], [float("nan"), .8], [.3, .7], None),
    ([0, 1], [.2, .8], [float("inf"), .7], None),
    ([0, 1], [-.1, .8], [.3, .7], None), ([0, 1], [.2, .8], [.3, 1.1], None),
    ([0, 1], [False, .8], [.3, .7], None), ([0, 1], [".2", .8], [.3, .7], None),
    ([0, 1], [.2, .8], [.3, .7], [1]),
    ([0, 1], [.2, .8], [.3, .7], [0, 0]),
    ([0, 1], [.2, .8], [.3, .7], [-1, 2]),
    ([0, 1], [.2, .8], [.3, .7], [float("inf"), 1]),
    ([0, 1], [.2, .8], [.3, .7], [float("nan"), 1]),
    ([0, 1], [.2, .8], [.3, .7], [True, 1]),
    # Zero weight does not permit nonfinite predictions into a paired report.
    ([0, 1], [float("nan"), .8], [.3, .7], [0, 1]),
])
def test_invalid_prediction_data_is_rejected(labels, baseline, candidate, weights):
    with pytest.raises(ValueError):
        _compare(labels, baseline, candidate, sample_weight=weights)


def test_module_can_run_as_an_extracted_stdlib_source_without_site_packages(tmp_path):
    module = importlib.import_module("model_evo_harness.analysis.prediction_diagnostics")
    source = Path(module.__file__).read_text()
    standalone = tmp_path / "diagnostic.py"
    standalone.write_text(source + '''
result = BinaryPredictionDiagnostics()(
    [0, 1], [.2, .8], [.3, .7], sample_ids=["a", "b"],
    baseline_sample_ids=["a", "b"], candidate_sample_ids=["a", "b"],
    dataset_id="data", split_id="validation")
assert result["baseline"]["auc"] == 1.0
''')
    subprocess.run([sys.executable, "-I", "-S", str(standalone)], check=True)
