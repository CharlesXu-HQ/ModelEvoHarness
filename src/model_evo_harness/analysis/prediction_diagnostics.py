"""Stdlib observations of paired binary predictions, without model fitting.

AUC and logloss describe different properties. Their disagreement and the
reported loss contributions establish neither a cause nor statistical evidence
of improvement. The host owns prediction provenance and the fixed evaluator.
"""

from itertools import groupby
import math
from numbers import Real


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain finite real numbers, not booleans")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must contain finite real numbers") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must contain finite real numbers")
    return value


def _sequence(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    try:
        return tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence") from error


def _ids(values, name):
    values = _sequence(values, name)
    for value in values:
        if (isinstance(value, bool) or not isinstance(value, (str, int))
                or isinstance(value, str) and not value.strip()):
            raise ValueError(f"{name} must contain nonempty string or integer identifiers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique identifiers")
    return values


def _normalized(weights):
    # Scaling avoids overflowing a sum of finite weights. Normalize each class
    # separately for AUC and conditional losses to preserve tiny class masses.
    scale = max(weights, default=0.)
    if scale == 0:
        return [0.] * len(weights)
    scaled = [weight / scale for weight in weights]
    total = math.fsum(scaled)
    return [weight / total for weight in scaled]


def _auc(labels, probabilities, weights):
    by_class = [[weight if label == kind else 0.
                 for label, weight in zip(labels, weights)] for kind in (0, 1)]
    if not all(any(weight > 0 for weight in group) for group in by_class):
        return None
    negative, positive = map(_normalized, by_class)
    ordered = sorted(zip(probabilities, negative, positive))
    lower_negative = 0.
    contributions = []
    for _, group in groupby(ordered, key=lambda row: row[0]):
        group = tuple(group)
        negative_mass = math.fsum(row[1] for row in group)
        positive_mass = math.fsum(row[2] for row in group)
        contributions.append(positive_mass * (lower_negative + .5 * negative_mass))
        lower_negative += negative_mass
    return min(1., max(0., math.fsum(contributions)))


def _loss(label, probability, eps):
    # Branch on the label and endpoint instead of computing 0 * log(0).
    if probability < eps:
        return (-math.log(eps) if label else -math.log1p(-eps)), True
    upper = 1. - eps
    if upper == 1. and probability == 1.:
        # A virtual endpoint preserves finite loss when the bound rounds to 1.
        return (-math.log1p(-eps) if label else -math.log(eps)), True
    clipped = probability > upper
    probability = min(probability, upper)
    return (-math.log(probability) if label else -math.log1p(-probability)), clipped


def _measure(labels, probabilities, weights, normalized, eps):
    rows = [_loss(label, probability, eps) for label, probability in zip(labels, probabilities)]
    losses = [loss for loss, _ in rows]
    result = {
        "auc": _auc(labels, probabilities, weights),
        "log_loss": math.fsum(weight * loss for weight, loss in zip(normalized, losses)),
        "clipped_count": sum(clipped for _, clipped in rows),
    }
    for kind, name in ((0, "negative"), (1, "positive")):
        indices = [i for i, label in enumerate(labels) if label == kind]
        class_weights = [weights[i] for i in indices]
        class_normalized = _normalized(class_weights)
        active = sum(weight > 0 for weight in class_weights)
        result[name] = {
            "count": len(indices), "active_count": active,
            "weight_fraction": math.fsum(normalized[i] for i in indices),
            "log_loss": (math.fsum(weight * losses[i]
                                   for i, weight in zip(indices, class_normalized))
                         if active else None),
            "log_loss_contribution": math.fsum(normalized[i] * losses[i] for i in indices),
        }
    return result, losses


def _difference(baseline, candidate):
    return None if baseline is None or candidate is None else candidate - baseline


def _direction(delta, *, maximize):
    if delta is None:
        return "undefined"
    if delta == 0:
        return "unchanged"
    return "improved" if (delta > 0) == maximize else "worsened"


class BinaryPredictionDiagnostics:
    """Compare two probability vectors on the same identified binary outcomes.

    ``eps`` is a finite real number strictly between 0 and .5 (default 1e-15).
    Logloss uses natural logarithms with probabilities clipped to [eps, 1-eps].
    The upper bound uses the actual floating-point value of 1-eps; values at
    and above it have identical loss. If that bound rounds to 1, an exact p=1
    uses a virtual endpoint: loss is -log(eps) for label 0 and -log1p(-eps)
    for label 1, and counts as clipped. Other probabilities use the usual bound.
    AUC uses the original probabilities, exact score ties receive half credit,
    and sorted tie groups take O(n log n) time and O(n) memory.

    Call inputs are finite real sequences: labels exactly 0/1; baseline and
    candidate probabilities in [0,1]; optional shared nonnegative sample_weight.
    Booleans are rejected. At least one row and positive weight are required.
    All rows are validated, including zero-weight rows. Weights affect both
    metrics; positive/negative pair mass is the product of their row weights.
    Internal relative scaling avoids overflow from large finite weights.

    sample_ids identify labels; baseline_sample_ids and candidate_sample_ids
    must exactly match their order and length. IDs are unique nonempty strings
    or integers, not booleans. No joining, dropping or reordering is performed.
    dataset_id and split_id must be nonempty strings identifying one shared,
    immutable evaluation population. These are host declarations: matching IDs
    cannot prove row contents, label timing or correct prediction provenance.

    The report preserves IDs, dataset/split identity and eps. Each prediction
    summary contains auc, log_loss, clipped_count, and positive/negative
    summaries. Clipped counts and class count include zero-weight rows;
    active_count counts rows with positive original weight. Class log_loss is
    conditional on that class; log_loss_contribution uses the total population
    denominator, so the two contributions sum to overall log_loss. Absent or
    zero-weight classes have conditional loss None and contribution zero.
    Relative weight fractions below floating-point range may round to zero.

    With only one class of positive weight, auc and its delta are None and
    conflict is None. Otherwise conflict is true only when one metric improves
    and the other worsens. All deltas are candidate minus baseline; directions
    use exact numeric comparisons, without significance or tolerance claims.
    paired_loss_changes counts positive-weight rows only, reporting signed
    improvement (negative) and regression (positive) contributions to total
    loss change. Floating-point rounding may affect additive identities.

    This observation tool does not fit calibration, identify a cause, estimate
    uncertainty, select a model, or measure business utility. In particular,
    worse logloss alone does not establish miscalibration or overfitting.
    """

    def __init__(self, eps=1e-15):
        self.eps = _number(eps, "eps")
        if not 0 < self.eps < .5:
            raise ValueError("eps must be strictly between zero and .5")

    def __call__(self, labels, baseline, candidate, *, sample_ids,
                 baseline_sample_ids, candidate_sample_ids, dataset_id, split_id,
                 sample_weight=None):
        for name, value in (("dataset_id", dataset_id), ("split_id", split_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        ids = _ids(sample_ids, "sample_ids")
        if (_ids(baseline_sample_ids, "baseline_sample_ids") != ids
                or _ids(candidate_sample_ids, "candidate_sample_ids") != ids):
            raise ValueError("prediction sample IDs must exactly match label IDs and order")
        arrays = []
        for name, values in (("labels", labels), ("baseline", baseline), ("candidate", candidate)):
            values = tuple(_number(value, name) for value in _sequence(values, name))
            if not values or len(values) != len(ids):
                raise ValueError(f"{name} must be nonempty and have the same length as sample_ids")
            invalid = (any(value not in (0., 1.) for value in values) if name == "labels"
                       else any(not 0 <= value <= 1 for value in values))
            if invalid:
                raise ValueError(f"{name} must be binary labels or probabilities in [0,1]")
            arrays.append(values)
        labels, baseline, candidate = arrays
        weights = ((1.,) * len(ids) if sample_weight is None else tuple(
            _number(value, "sample_weight") for value in _sequence(sample_weight, "sample_weight")))
        if len(weights) != len(ids) or any(weight < 0 for weight in weights) or not any(weights):
            raise ValueError("sample_weight needs matching length, nonnegative values and positive total")
        normalized = _normalized(weights)
        previous, previous_losses = _measure(labels, baseline, weights, normalized, self.eps)
        current, current_losses = _measure(labels, candidate, weights, normalized, self.eps)
        delta = {key: _difference(previous[key], current[key]) for key in ("auc", "log_loss")}
        for name in ("positive", "negative"):
            for key in ("log_loss", "log_loss_contribution"):
                delta[f"{name}_{key}"] = _difference(previous[name][key], current[name][key])
        changes = [new - old for old, new in zip(previous_losses, current_losses)]
        active_changes = [change for weight, change in zip(weights, changes) if weight > 0]
        paired = {"improved_count": sum(change < 0 for change in active_changes),
                  "worsened_count": sum(change > 0 for change in active_changes),
                  "unchanged_count": sum(change == 0 for change in active_changes)}
        paired["improvement_contribution"] = math.fsum(
            weight * min(change, 0.) for weight, change in zip(normalized, changes))
        paired["regression_contribution"] = math.fsum(
            weight * max(change, 0.) for weight, change in zip(normalized, changes))
        auc_direction = _direction(delta["auc"], maximize=True)
        loss_direction = _direction(delta["log_loss"], maximize=False)
        conflict = (None if delta["auc"] is None else
                    {auc_direction, loss_direction} == {"improved", "worsened"})
        return {
            "dataset_id": dataset_id, "split_id": split_id, "sample_ids": ids,
            "n_samples": len(ids), "zero_weight_count": weights.count(0.), "eps": self.eps,
            "baseline": previous, "candidate": current, "delta": delta,
            "comparison": {"auc_direction": auc_direction,
                           "log_loss_direction": loss_direction, "conflict": conflict},
            "paired_loss_changes": paired,
        }
