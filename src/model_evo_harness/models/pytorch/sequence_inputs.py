"""Native sequence input contracts; True masks identify observed events."""
import torch
from torch import nn


def validate_sequence_mask(values, valid=None, lengths=None, require_prefix=False, max_length=None):
    """Return [B,T] mask; supplied lengths count valid positions.

    Length-only inputs create right-padded prefixes. An explicit mask may have
    gaps unless require_prefix=True. max_length bounds the padded time dimension.
    This checks layout, not feature meaning or event provenance.
    """
    if not isinstance(require_prefix, bool):
        raise ValueError("require_prefix must be boolean")
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[2] < 1:
        raise ValueError("sequence values must be [B,T,D] with positive B,D")
    batch, steps = values.shape[:2]
    if max_length is not None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if steps > max_length:
            raise ValueError("sequence exceeds max_length")
    if lengths is not None:
        lengths = torch.as_tensor(lengths, device=values.device)
        if (lengths.shape != (batch,) or lengths.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or not bool(((lengths >= 0) & (lengths <= steps)).all())):
            raise ValueError("lengths must be integer [B] in [0,T]")
    if valid is None:
        valid = (torch.ones((batch, steps), dtype=torch.bool, device=values.device) if lengths is None else
                 torch.arange(steps, device=values.device)[None, :] < lengths[:, None])
    else:
        valid = torch.as_tensor(valid, device=values.device)
        if valid.shape != (batch, steps) or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean [B,T] mask")
    if lengths is not None and not bool((valid.sum(-1) == lengths).all()):
        raise ValueError("lengths disagree with mask counts")
    if require_prefix and bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("valid must be a right-padded prefix mask")
    return valid


def validate_aligned_sequences(attributes, masks, event_ids=None):
    """Check declared same-event attributes, their masks and optional event IDs.

    Without event_ids only layout can be verified. The host remains responsible
    for joining attributes on the same event; independent histories must not be
    passed as a single alignment group. Padding event IDs are ignored.
    """
    if not isinstance(attributes, dict) or not attributes or set(attributes) != set(masks):
        raise ValueError("attributes and masks must have the same nonempty field keys")
    if event_ids is not None and set(event_ids) != set(attributes):
        raise ValueError("event_ids must identify every aligned attribute")
    reference_mask, reference_ids = None, None
    for name, value in attributes.items():
        if value.ndim < 2:
            raise ValueError("aligned attributes need batch and time dimensions")
        mask = torch.as_tensor(masks[name], device=value.device)
        if mask.dtype != torch.bool or mask.shape != value.shape[:2]:
            raise ValueError("aligned attribute mask must be boolean [B,T]")
        if reference_mask is not None and (mask.shape != reference_mask.shape or not torch.equal(mask, reference_mask)):
            raise ValueError("aligned attribute masks must match")
        reference_mask = mask
        if event_ids is not None:
            ids = torch.as_tensor(event_ids[name], device=value.device)
            if ids.shape != mask.shape:
                raise ValueError("event IDs must be [B,T]")
            if reference_ids is not None and not torch.equal(ids[mask], reference_ids[mask]):
                raise ValueError("aligned attributes have different valid event IDs")
            reference_ids = ids


class MaskedSequencePooling(nn.Module):
    """Pool [B,T,D] into [B,D], ignoring masked padding before arithmetic.

    With normalization='none', supplied weights must be nonnegative. Sum uses
    sum(w*x); mean uses sum(w*x)/sum(w); max uses coordinatewise max(w*x).
    Softmax normalization treats supplied weights as logits over valid events.
    Empty sequences and zero-total-weight rows return zeros. All valid vectors
    and weights must be finite; masked NaN/Inf padding is permitted.
    """
    def __init__(self, mode="mean", weight_normalization="none", require_prefix=False):
        super().__init__()
        if mode not in ("sum", "mean", "max"):
            raise ValueError("mode must be sum, mean or max")
        if weight_normalization not in ("none", "softmax"):
            raise ValueError("weight_normalization must be none or softmax")
        self.mode, self.weight_normalization, self.require_prefix = mode, weight_normalization, require_prefix

    def forward(self, values, valid=None, lengths=None, weights=None):
        valid = validate_sequence_mask(values, valid, lengths, self.require_prefix)
        if not values.is_floating_point():
            raise ValueError("values must be real floating tensors")
        values = torch.where(valid[..., None], values, torch.zeros_like(values))
        if not bool(torch.isfinite(values).all()):
            raise ValueError("valid sequence values must be finite")
        if weights is None:
            if self.weight_normalization == "softmax":
                raise ValueError("softmax normalization requires explicit weight logits")
            weights = valid.to(values.dtype)
        else:
            if weights.shape != valid.shape or not weights.is_floating_point() or weights.device != values.device:
                raise ValueError("weights must be floating [B,T] on the values device")
            weights = torch.where(valid, weights.to(values.dtype), torch.zeros_like(weights, dtype=values.dtype))
            if not bool(torch.isfinite(weights).all()):
                raise ValueError("valid sequence weights must be finite")
            if self.weight_normalization == "none" and bool((weights < 0).any()):
                raise ValueError("unnormalized weights must be nonnegative")
        if not values.shape[1]:
            return values.sum(1) + weights.sum(1, keepdim=True)*0
        if self.weight_normalization == "softmax":
            logits = weights.masked_fill(~valid, -torch.inf)
            logits = torch.where(valid.any(1, keepdim=True), logits, torch.zeros_like(logits))
            weights = torch.where(valid, torch.softmax(logits, dim=1), 0)
        if self.mode == "mean":
            # Rescale before summing so a representable weighted mean need not
            # overflow merely because many large weights were supplied.
            scale = weights.amax(1, keepdim=True)
            weights = weights / torch.where(scale > 0, scale, 1)
            total = weights.sum(1, keepdim=True)
            weights = weights / torch.where(total > 0, total, 1)
        weighted = values * weights[..., None]
        if self.mode != "max":
            return weighted.sum(1)
        result = weighted.masked_fill(~valid[..., None], -torch.inf).amax(1)
        return torch.where(valid.any(1, keepdim=True), result, 0)
