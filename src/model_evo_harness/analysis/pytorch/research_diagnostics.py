"""Standalone PyTorch evidence for representation quality and feature screening.

Mechanism-inspired utilities, not a RankUp or TayFCS reproduction. Metrics do not
establish collapse, causal importance, or downstream improvement. Collect scores
on training data; removing an input still requires ablation and retraining.
"""

import math
import operator

import torch


def _divide(numerator, denominator):
    return torch.where(denominator > 0,
                       numerator / torch.where(denominator > 0, denominator, 1), 0)


def _masked(values, valid_mask):
    if not isinstance(values, torch.Tensor) or values.ndim != 3:
        raise ValueError("values must have shape [B, T, D]")
    if values.shape[0] == 0 or values.shape[2] == 0 or not values.is_floating_point():
        raise ValueError("values need nonempty batch/channels and a real floating dtype")
    values = values.detach()
    if values.dtype != torch.float64:
        values = values.float()
    if valid_mask is None:
        valid_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
    if valid_mask.dtype != torch.bool or tuple(valid_mask.shape) != tuple(values.shape[:2]):
        raise ValueError("valid_mask must be boolean with shape [B, T]")
    valid_mask = valid_mask.to(device=values.device)
    values = torch.where(valid_mask[..., None], values, 0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("valid values must be finite")
    return values, valid_mask


def _unit_scale(matrix):
    # The appended zero also defines the maximum for an empty token axis.
    flat = matrix.abs().flatten(1)
    scale = torch.cat((flat, matrix.new_zeros((matrix.shape[0], 1))), 1).amax(1)
    return _divide(matrix, scale[:, None, None])


def _center(values, mask, count):
    # Shift first so mean rounding cannot invent rank for identical tokens.
    padded = torch.cat((values, values.new_zeros((values.shape[0], 1, values.shape[2]))), 1)
    first = torch.cat((mask, mask.new_zeros((mask.shape[0], 1))), 1).long().argmax(1)
    origin = padded[torch.arange(values.shape[0], device=values.device), first]
    shifted = torch.where(mask[..., None], values - origin[:, None, :], 0)
    mean = _divide(shifted.sum(1), count[:, None])
    return torch.where(mask[..., None], shifted - mean[:, None, :], 0)


class RepresentationDiagnostics:
    """Detached per-sample metrics for matrices with token rows and D columns.

    ``center=True`` subtracts each sample's valid-token channel mean before rank
    and branch similarity. The default uses uncentered matrices. Effective rank
    is exp(-sum(p*log(p))) with p=s/sum(s), using singular values (not squared
    values). Stable rank is sum(s**2)/max(s)**2. Both are zero for zero matrices.

    ``output_variance`` always means population variance across valid tokens,
    averaged over channels; this is not variance across the batch. Similarity
    is the signed Frobenius cosine of aligned, equally masked branch matrices.
    Zero-norm branches and samples with no valid tokens return zero. Half
    precision is promoted to float32; float64 inputs retain their precision.
    """

    def __init__(self, center=False):
        self.center = bool(center)

    def __call__(self, representations, valid_mask=None, reference=None):
        values, mask = _masked(representations, valid_mask)
        count = mask.sum(1)
        centered = _center(values, mask, count)
        variance = _divide(centered.square().sum((1, 2)), count * values.shape[2])
        matrix = _unit_scale(centered if self.center else values)
        # A zero row leaves the spectrum unchanged and supports T=0.
        padded = torch.cat((matrix, matrix.new_zeros((matrix.shape[0], 1, matrix.shape[2]))), 1)
        singular = torch.linalg.svdvals(padded)
        probability = _divide(singular, singular.sum(1, keepdim=True))
        log_probability = torch.log(torch.where(probability > 0, probability, 1))
        effective = torch.where(singular.sum(1) > 0,
                                torch.exp(-(probability * log_probability).sum(1)), 0)
        result = {
            "effective_rank": effective,
            "stable_rank": _divide(singular.square().sum(1), singular.amax(1).square()),
            "output_variance": variance,
            "valid_tokens": count,
        }
        if reference is not None:
            if reference.shape != representations.shape:
                raise ValueError("reference must have the same shape as representations")
            other, _ = _masked(reference, mask)
            if self.center:
                other = _center(other, mask, count)
            other = _unit_scale(other)
            norm = torch.sqrt(matrix.square().sum((1, 2)) * other.square().sum((1, 2)))
            result["branch_similarity"] = _divide((matrix * other).sum((1, 2)), norm).clamp(-1, 1)
        return result

    def inspect(self, layers, valid_masks=None, references=None):
        """Inspect caller-captured named tensors; no model hooks are installed.

        Optional mappings provide masks or aligned reference branches by layer
        name. Missing entries use all tokens or omit branch similarity.
        """
        valid_masks, references = valid_masks or {}, references or {}
        return {name: self(value, valid_masks.get(name), references.get(name))
                for name, value in layers.items()}


class GradientFeatureSelector:
    """Training-data screening evidence for fields OR explicit cross tensors.

    Inputs and caller-computed objective gradients both have shape [B, F, D].
    Scores are mean_valid_B(abs(sum_D(activation * gradient))). This is a local
    first-order sensitivity, depends on objective reduction and feature scale,
    and cannot discover missing interactions from marginal scores.

    Redundancy is absolute uncentered cosine of each field pair flattened across
    their jointly valid batch rows and channels. ``overlap_counts`` makes absent
    evidence explicit; zero-norm/no-overlap pairs return zero. Redundancy is
    reported, never silently used to delete correlated fields. Call this method
    separately on explicit cross activations to score interactions themselves.
    """

    def __call__(self, activations, gradients, valid_mask=None):
        if activations.shape != gradients.shape:
            raise ValueError("gradients must have the same shape as activations")
        values, mask = _masked(activations, valid_mask)
        gradient, _ = _masked(gradients, mask)
        if values.shape[1] == 0:
            raise ValueError("at least one field is required")
        counts = mask.sum(0)
        saliency = _divide((values * gradient).sum(2).abs().sum(0), counts)
        # Rescale each field before dot products; cosine is scale invariant.
        fields = _unit_scale(values.transpose(0, 1))
        flat = fields.flatten(1)
        pair_energy = fields.square().sum(2) @ mask.to(values.dtype)
        norm = torch.sqrt(pair_energy * pair_energy.T)
        redundancy = _divide((flat @ flat.T).abs(), norm).clamp(0, 1)
        # CUDA has no int64 GEMM. Float64 preserves integer counts up to 2**53.
        count_mask = mask.to(torch.float64)
        return {"saliency": saliency, "redundancy": redundancy, "valid_counts": counts,
                "overlap_counts": (count_mask.T @ count_mask).to(torch.int64)}

    def select(self, scores, budget, protected_indices=(), protected_groups=()):
        """Return an offline/eager proposal of original-order retained indices.

        Protected indices and ALL members of each protected interaction group
        are mandatory, even with zero marginal score. Overlapping groups form a
        union. Reject an insufficient budget; fill remaining slots by descending
        score, breaking ties by smaller index. No inputs are physically pruned.
        The caller must declare interaction-only fields/groups, then validate
        any proposed removal with ablation and retraining.
        """
        if scores.ndim != 1:
            raise ValueError("scores must have shape [F]")
        scores = scores.detach().cpu().tolist()
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("scores must be finite")
        if isinstance(budget, bool):
            raise ValueError("budget must be an integer between zero and field count")
        budget = operator.index(budget)
        if budget < 0 or budget > len(scores):
            raise ValueError("budget must be between zero and field count")
        protected = set()
        for group in (protected_indices, *protected_groups):
            for index in group:
                if isinstance(index, bool):
                    raise ValueError("protected indices must be integers")
                index = operator.index(index)
                if index < 0 or index >= len(scores):
                    raise ValueError("protected index is outside field range")
                protected.add(index)
        if len(protected) > budget:
            raise ValueError("budget cannot contain all protected fields and interaction groups")
        candidates = sorted((index for index in range(len(scores)) if index not in protected),
                            key=lambda index: (-scores[index], index))
        return tuple(sorted(protected.union(candidates[:budget - len(protected)])))
