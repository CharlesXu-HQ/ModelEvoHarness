"""Independent DANet-inspired discount branch and positive-purchase objective.

Source: https://arxiv.org/html/2607.12578 . This local reference keeps the host
interest backbone. It does not reproduce the full DANet/data pipeline or its
private evaluation. The learned spectral MLP is not guaranteed to be low-pass.
The host supplies aligned, pre-decision features and a regular time grid; zero
filling missing slots for the FFT is an explicit approximation, not a claim
that a missing slot had zero discount. No labels enter forward prediction.
"""

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F


def _dimensions(*dimensions):
    if any(not isinstance(dim, Integral) or isinstance(dim, bool) or dim < 1
           for dim in dimensions):
        raise ValueError("dimensions must be positive integers")


def _floating(values, name, dtype):
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise ValueError(f"{name} must be a real floating tensor")
    return values.to(dtype=dtype)


def _finite(values, name):
    if not torch.isfinite(values).all():
        raise ValueError(f"observed {name} must be finite")


def _mask(mask, shape, name):
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != shape:
        raise ValueError(f"{name} must be bool with the matching input shape")


def _parameter_dtype(module):
    dtype = next(module.parameters()).dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("module parameters must use float32 or float64")
    return dtype


class DiscountPreferenceBranch(nn.Module):
    """Pre-decision discount signals -> a discount preference representation.

    Inputs, in order: history [B,N], bool history mask [B,N], target [B,D],
    purchase items [B,L,D], aligned attributes [B,L,A], bool purchase mask
    [B,L], promotion context [B,C]. N is history_length; L may be zero.
    Historical discount ratios must be in [0,1] under one host-chosen ratio
    convention. Attributes carry historical discount/category/price semantics;
    this component neither constructs these attributes nor infers their timing.

    rFFT -> concat(real, imaginary) -> MLP -> irFFT(n=N) gives filtered history.
    Residual history is the sanitized original minus that filtered signal.
    Target-conditioned attention pools purchase attributes; a ReLU projection
    and sigmoid context gate produce user_preference. Masked values are removed
    before arithmetic. Empty purchase/history components return exact zeros.

    Returns representation [user_preference; residual_history; filtered_history]
    of width hidden_dim+2N, the three named components, and attention [B,L].
    Features are cast to the module's float32 (default) or float64 parameters;
    half/bfloat16 inputs are safely promoted. Half/bfloat16 parameters are not
    supported. The host controls device placement, normalization and training.
    """

    def __init__(self, history_length: int, item_dim: int, attribute_dim: int,
                 context_dim: int, hidden_dim: int):
        super().__init__()
        _dimensions(history_length, item_dim, attribute_dim, context_dim, hidden_dim)
        self.history_length, self.item_dim = history_length, item_dim
        self.attribute_dim, self.context_dim = attribute_dim, context_dim
        self.hidden_dim = hidden_dim
        self.frequency_bins = history_length // 2 + 1
        self.spectral_filter = nn.Sequential(
            nn.Linear(2 * self.frequency_bins, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * self.frequency_bins))
        self.attribute_attention = nn.Sequential(
            nn.Linear(2 * item_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.user_projection = nn.Sequential(nn.Linear(attribute_dim, hidden_dim), nn.ReLU())
        self.context_gate = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.Sigmoid())

    def forward(self, item_discount_history: torch.Tensor, history_mask: torch.Tensor,
                target_item: torch.Tensor, purchase_items: torch.Tensor,
                purchase_attributes: torch.Tensor, purchase_mask: torch.Tensor,
                promotion_context: torch.Tensor) -> dict:
        dtype = _parameter_dtype(self)
        history = _floating(item_discount_history, "item_discount_history", dtype)
        target = _floating(target_item, "target_item", dtype)
        items = _floating(purchase_items, "purchase_items", dtype)
        attributes = _floating(purchase_attributes, "purchase_attributes", dtype)
        context = _floating(promotion_context, "promotion_context", dtype)
        if history.ndim != 2 or history.shape[1] != self.history_length:
            raise ValueError("item_discount_history must have shape [B,history_length]")
        batch = history.shape[0]
        if (target.shape != (batch, self.item_dim) or items.ndim != 3 or
                items.shape[0] != batch or items.shape[2] != self.item_dim or
                attributes.shape != (*items.shape[:2], self.attribute_dim) or
                context.shape != (batch, self.context_dim)):
            raise ValueError("incompatible target, purchase attribute/item or context shapes")
        _mask(history_mask, history.shape, "history_mask")
        _mask(purchase_mask, items.shape[:2], "purchase_mask")
        history = torch.where(history_mask, history, 0)
        items = torch.where(purchase_mask[..., None], items, 0)
        attributes = torch.where(purchase_mask[..., None], attributes, 0)
        for values, name in ((history, "history"), (target, "target"), (items, "purchase items"),
                             (attributes, "purchase attributes"), (context, "promotion context")):
            _finite(values, name)
        if ((history < 0) | (history > 1)).any():
            raise ValueError("observed historical discount ratios must lie in [0,1]")

        spectrum = torch.fft.rfft(history, n=self.history_length, dim=-1)
        filtered_coefficients = self.spectral_filter(torch.cat((spectrum.real, spectrum.imag), -1))
        # Host autocast may lower the MLP dtype; complex/odd-length FFT need full precision.
        filtered_coefficients = filtered_coefficients.to(history.dtype)
        real, imaginary = filtered_coefficients.split(self.frequency_bins, dim=-1)
        filtered = torch.fft.irfft(torch.complex(real, imaginary), n=self.history_length, dim=-1)
        filtered = torch.where(history_mask, filtered, 0)
        residual = history - filtered

        repeated_target = target[:, None].expand(-1, items.shape[1], -1)
        scores = self.attribute_attention(torch.cat((items, repeated_target), -1)).squeeze(-1)
        scores = scores.masked_fill(~purchase_mask, -torch.inf)
        present = purchase_mask.any(-1, keepdim=True)
        # A zero-valued sentinel is visible only when all real purchases are absent.
        sentinel = torch.where(present, scores.new_full((batch, 1), -torch.inf), 0)
        attention = torch.softmax(torch.cat((scores, sentinel), -1), dim=-1)[:, :-1]
        pooled = (attention[..., None] * attributes).sum(1)
        preference = torch.where(present, self.user_projection(pooled), 0)
        preference = preference * self.context_gate(context)
        return {"representation": torch.cat((preference, residual, filtered), -1),
                "user_preference": preference, "filtered_history": filtered,
                "residual_history": residual, "attention": attention}


class DiscountAwareConversionHead(nn.Module):
    """Attach conversion and discount heads to a host-owned interest backbone.

    Inputs interest [B,I] and discount [B,P] produce conversion_logit [B] and
    discount_prediction [B]. Conversion uses both representations; discount
    uses only the discount representation and a sigmoid output. Feature dtype
    follows float32/float64 module parameters as in DiscountPreferenceBranch.
    These predictions are response/discount estimates, not treatment effects.
    """

    def __init__(self, interest_dim: int, discount_dim: int, hidden_dim: int):
        super().__init__()
        _dimensions(interest_dim, discount_dim, hidden_dim)
        self.interest_dim, self.discount_dim = interest_dim, discount_dim
        self.conversion_head = nn.Sequential(nn.Linear(interest_dim + discount_dim, hidden_dim),
                                             nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.discount_head = nn.Sequential(nn.Linear(discount_dim, hidden_dim), nn.ReLU(),
                                           nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, interest_representation: torch.Tensor,
                discount_representation: torch.Tensor) -> dict:
        dtype = _parameter_dtype(self)
        interest = _floating(interest_representation, "interest_representation", dtype)
        discount = _floating(discount_representation, "discount_representation", dtype)
        if (interest.ndim != 2 or interest.shape[1] != self.interest_dim or
                discount.shape != (interest.shape[0], self.discount_dim)):
            raise ValueError("representations must have configured [B,I] and [B,P] shapes")
        _finite(interest, "interest_representation")
        _finite(discount, "discount_representation")
        return {"conversion_logit": self.conversion_head(torch.cat((interest, discount), -1)).squeeze(-1),
                "discount_prediction": self.discount_head(discount).squeeze(-1)}

    @staticmethod
    def objective(outputs: dict, conversion_labels: torch.Tensor,
                  discount_labels: torch.Tensor, discount_label_mask: torch.Tensor,
                  alpha: float = 1.0) -> dict:
        """Mean binary conversion loss + alpha * mean valid-purchase discount MSE.

        Labels have shape [B]; conversion labels are binary, discount labels
        use the history ratio convention in [0,1], and label_mask is bool.
        Only rows with conversion=1 and label_mask=True supervise discount.
        Irrelevant discount-label NaNs are sanitized, labels are detached,
        and no-valid-positive batches have differentiable zero auxiliary loss.
        Alpha is a finite nonnegative Python real scalar. Returns scalar loss,
        conversion_loss, discount_loss, and int64 discount_count.
        """
        if not isinstance(alpha, Real) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be a finite nonnegative real scalar")
        logits, prediction = outputs["conversion_logit"], outputs["discount_prediction"]
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 1 or logits.numel() == 0 or
                not isinstance(prediction, torch.Tensor) or prediction.shape != logits.shape):
            raise ValueError("outputs must have matching nonempty [B] shapes")
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        logits = _floating(logits, "conversion_logit", dtype)
        prediction = _floating(prediction, "discount_prediction", dtype)
        _finite(logits, "conversion_logit")
        _finite(prediction, "discount_prediction")
        if ((prediction < 0) | (prediction > 1)).any():
            raise ValueError("discount_prediction must lie in [0,1]")
        for labels in (conversion_labels, discount_labels):
            if not isinstance(labels, torch.Tensor) or labels.shape != logits.shape or labels.is_complex():
                raise ValueError("labels must be real tensors with shape [B]")
        _mask(discount_label_mask, logits.shape, "discount_label_mask")
        conversion_labels = conversion_labels.detach().to(dtype=dtype)
        _finite(conversion_labels, "conversion labels")
        if not ((conversion_labels == 0) | (conversion_labels == 1)).all():
            raise ValueError("conversion labels must be binary")
        eligible = (conversion_labels == 1) & discount_label_mask
        safe_labels = torch.where(eligible, discount_labels.detach().to(dtype=dtype), 0)
        _finite(safe_labels, "valid purchase discount labels")
        if ((safe_labels < 0) | (safe_labels > 1)).any():
            raise ValueError("valid purchase discount labels must lie in [0,1]")
        safe_prediction = torch.where(eligible, prediction, 0)
        count = eligible.sum()
        conversion_loss = F.binary_cross_entropy_with_logits(logits, conversion_labels)
        discount_loss = (safe_prediction - safe_labels).square().sum() / count.clamp_min(1)
        return {"loss": conversion_loss + alpha * discount_loss,
                "conversion_loss": conversion_loss, "discount_loss": discount_loss,
                "discount_count": count}
