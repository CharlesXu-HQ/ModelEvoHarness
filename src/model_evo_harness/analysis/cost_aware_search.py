"""COLD-inspired physical-subset/retraining protocol with host-measured costs.

This stdlib-only orchestrator neither trains models nor measures serving costs.
Saliency proposes subsets; only comparable host evaluation selects a candidate.
"""

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real
import operator


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _validate_context(context):
    if not isinstance(context, Mapping):
        raise ValueError("context must be a mapping")
    for key in ("task_id", "dataset_id", "train_split_id", "validation_split_id",
                "quality_metric", "training_budget_id"):
        _text(context.get(key), key)
    if context["train_split_id"] == context["validation_split_id"]:
        raise ValueError("training and validation split identifiers must differ")
    if context.get("quality_direction") != "maximize":
        raise ValueError("quality_direction must be 'maximize'")
    _integer(context.get("seed"), "seed")
    profile = context.get("serving_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("serving_profile must be a mapping")
    _text(profile.get("hardware"), "serving_profile.hardware")
    for key in ("batch_size", "candidate_count", "concurrency"):
        _integer(profile.get(key), f"serving_profile.{key}", 1)
    if profile.get("latency_statistic") not in ("mean", "p50", "p90", "p95", "p99"):
        raise ValueError("latency_statistic must be mean, p50, p90, p95 or p99")
    if not isinstance(profile.get("includes_feature_lookup"), bool):
        raise ValueError("includes_feature_lookup must be boolean")


class CostAwareFeatureSearch:
    """Propose protected subsets, invoke host retraining, and select measured evidence.

    Constructor arguments:
      latency_budget_ms: positive finite bound for the declared latency statistic.
      quality_floor: optional minimum validation quality (larger is better).
      max_quality_drop: optional nonnegative absolute drop from the full baseline.
        At least one quality constraint is required; when both are supplied the
        stricter floor applies. This is not a relative/percentage degradation.

    Call arguments:
      saliency: nonempty finite real scores for fields/feature groups 0..F-1,
        collected only on the training split. Scores are screening evidence,
        never estimates of quality or cost after removing a field.
      budgets: iterable of integer retained counts in [0,F]. Each proposal keeps
        every protected index and ALL members of protected_groups (including
        overlapping or zero-saliency interaction-only groups), then fills slots
        by descending saliency, ties by original index. An insufficient budget
        is rejected before any training. Retained indices stay in original order.
      retrain_evaluate: callable invoked once for the full baseline first, then
        each distinct proposed subset in budget order, with keyword arguments::

          retrain_evaluate(retained_indices=(0, 2, ...), context={...})

        The host must materialize exactly these inputs, remove omitted feature
        retrieval and model dimensions, rebuild and retrain with its real trainer,
        evaluate held-out validation quality, and measure the declared serving
        statistic. Multiplying omitted inputs by zero is not physical pruning.
        Even the full baseline must be rebuilt/retrained under the same budget.
      context: mapping requiring nonempty task_id, dataset_id, train_split_id,
        validation_split_id, quality_metric, training_budget_id; nonnegative
        integer seed; quality_direction='maximize'; and serving_profile mapping
        containing hardware (nonempty string), positive integer batch_size,
        candidate_count, concurrency, latency_statistic (mean/p50/p90/p95/p99),
        includes_feature_lookup (bool). Train and validation identifiers differ.
        IDs must resolve immutable task/data/split/training specifications;
        add model revision, dtype, software, timing protocol, and other relevant
        profile metadata as needed. Every context field must match every trial.
      saliency_split: must exactly equal context['train_split_id'].

    Each callback result is a mapping requiring::

      {"retained_indices": (0, 2, ...),  # exact indices and order requested
       "retrained": True, "physical_pruning": True,
       "implementation_id": "auditable model/code artifact identifier",
       "train_run_id": "unique auditable training record identifier",
       "context": {...},               # full unchanged context
       "validation_quality": 0.0,      # finite, maximize metric
       "latency_ms": 1.0}              # finite, positive, measured

    Booleans and identifiers are host attestations, not proof that training or
    physical removal was correct. The host must preserve auditable artifacts.
    Callback errors propagate; invalid/mismatched evidence raises ValueError.
    No failed measurement is replaced with synthetic quality or latency.

    Returns a dict with context, saliency_split, latency_budget_ms, quality_floor
    (effective floor), latency_scope ('with_feature_lookup' or 'model_only'),
    baseline, trials, pareto_candidates, selected. Trials is a tuple containing
    every normalized callback result plus is_baseline and feasible booleans.
    The feasible Pareto frontier maximizes quality and minimizes latency; equal
    quality/cost candidates are retained. Selection maximizes feasible quality,
    then minimizes latency, subset size, and retained-index tuple, in that order.
    selected is None and the frontier is empty when no result is feasible.

    Only latency is optimized. QPS, acquisition cost, test-set effectiveness,
    and production improvements are not inferred by this protocol.
    """

    def __init__(self, latency_budget_ms, *, quality_floor=None, max_quality_drop=None):
        self.latency_budget_ms = _finite(latency_budget_ms, "latency_budget_ms", positive=True)
        if quality_floor is None and max_quality_drop is None:
            raise ValueError("quality_floor or max_quality_drop is required")
        self.quality_floor = (None if quality_floor is None
                              else _finite(quality_floor, "quality_floor"))
        self.max_quality_drop = (None if max_quality_drop is None
                                 else _finite(max_quality_drop, "max_quality_drop"))
        if self.max_quality_drop is not None and self.max_quality_drop < 0:
            raise ValueError("max_quality_drop must be nonnegative")

    def __call__(self, saliency, budgets, retrain_evaluate, *, context, saliency_split,
                 protected_indices=(), protected_groups=()):
        _validate_context(context)
        context = deepcopy(context)
        if saliency_split != context["train_split_id"]:
            raise ValueError("saliency_split must equal train_split_id")
        if not callable(retrain_evaluate):
            raise ValueError("retrain_evaluate must be callable")
        scores = tuple(_finite(value, "saliency") for value in saliency)
        if not scores:
            raise ValueError("saliency must contain at least one field")
        protected = set()
        for group in (protected_indices, *protected_groups):
            for index in group:
                index = _integer(index, "protected index")
                if index >= len(scores):
                    raise ValueError("protected index is outside the field range")
                protected.add(index)
        ranked = sorted((i for i in range(len(scores)) if i not in protected),
                        key=lambda i: (-scores[i], i))
        full = tuple(range(len(scores)))
        proposals, seen = [full], {full}
        for budget in budgets:
            budget = _integer(budget, "budget")
            if budget > len(scores):
                raise ValueError("budget exceeds the field count")
            if budget < len(protected):
                raise ValueError("budget cannot contain all protected fields/groups")
            retained = tuple(sorted(protected.union(ranked[:budget - len(protected)])))
            if retained not in seen:
                proposals.append(retained)
                seen.add(retained)

        trials, run_ids = [], set()
        for retained in proposals:
            evidence = retrain_evaluate(retained_indices=retained, context=deepcopy(context))
            if not isinstance(evidence, Mapping):
                raise ValueError("callback result must be a mapping")
            _validate_context(evidence.get("context"))
            if evidence["context"] != context:
                raise ValueError("callback context must exactly match the declared context")
            actual = evidence.get("retained_indices")
            if not isinstance(actual, (tuple, list)):
                raise ValueError("retained_indices must contain the requested indices")
            actual = tuple(_integer(i, "retained index") for i in actual)
            if actual != retained:
                raise ValueError("retained_indices must exactly match the requested subset and order")
            for flag in ("retrained", "physical_pruning"):
                if evidence.get(flag) is not True:
                    raise ValueError(f"callback must attest {flag}=True")
            for key in ("implementation_id", "train_run_id"):
                _text(evidence.get(key), key)
            if evidence["train_run_id"] in run_ids:
                raise ValueError("train_run_id must be unique for each retrained subset")
            run_ids.add(evidence["train_run_id"])
            trials.append({
                "retained_indices": retained, "retrained": True, "physical_pruning": True,
                "implementation_id": evidence["implementation_id"],
                "train_run_id": evidence["train_run_id"], "context": deepcopy(context),
                "validation_quality": _finite(evidence.get("validation_quality"), "validation_quality"),
                "latency_ms": _finite(evidence.get("latency_ms"), "latency_ms", positive=True),
                "is_baseline": retained == full,
            })

        floors = [] if self.quality_floor is None else [self.quality_floor]
        if self.max_quality_drop is not None:
            floors.append(trials[0]["validation_quality"] - self.max_quality_drop)
        floor = max(floors)
        for trial in trials:
            trial["feasible"] = (trial["latency_ms"] <= self.latency_budget_ms
                                 and trial["validation_quality"] >= floor)
        feasible = [trial for trial in trials if trial["feasible"]]
        ordered = sorted(feasible, key=lambda t: (-t["validation_quality"], t["latency_ms"],
                                                  len(t["retained_indices"]), t["retained_indices"]))
        frontier = tuple(trial for trial in ordered if not any(
            other["validation_quality"] >= trial["validation_quality"]
            and other["latency_ms"] <= trial["latency_ms"]
            and (other["validation_quality"] > trial["validation_quality"]
                 or other["latency_ms"] < trial["latency_ms"])
            for other in ordered))
        return {
            "context": context, "saliency_split": saliency_split,
            "latency_budget_ms": self.latency_budget_ms, "quality_floor": floor,
            "latency_scope": ("with_feature_lookup" if context["serving_profile"]["includes_feature_lookup"]
                              else "model_only"),
            "baseline": trials[0], "trials": tuple(trials),
            "pareto_candidates": frontier, "selected": ordered[0] if ordered else None,
        }
