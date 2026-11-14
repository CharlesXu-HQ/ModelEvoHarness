"""Independent field-product references from FuxiCTR 183da1c944ba8930fcdcd54114837c081959d748.

Sources: model_zoo/{FFM,ONN,HOFM,HFM,DLRM,LorentzFM,DeepIM} and
fuxictr/pytorch/layers/interactions/interaction_machine.py. No upstream code
or model-framework dependency is used. Host owns embeddings, heads and loss.
Pairs use i<j row-major order. Masks are bool True=valid; invalid NaN padding
is zeroed before arithmetic. Half inputs are promoted for polynomial/FFT work.
"""

from numbers import Integral
import torch
from torch import nn


def _dims(*dims):
    if any(not isinstance(d, Integral) or isinstance(d, bool) or d < 1 for d in dims):
        raise ValueError("dimensions must be positive integers")


def _tokens(x, mask, fields, dim):
    if x.ndim != 3 or x.shape[1:] != (fields, dim) or not x.is_floating_point():
        raise ValueError("tokens must be floating [B,num_fields,dim]")
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    if mask is None:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
        raise ValueError("valid_mask must be bool [B,num_fields]")
    x = torch.where(mask[..., None], x, 0)
    if not torch.isfinite(x).all():
        raise ValueError("observed tokens must be finite")
    return x, mask


class FieldAwareFactorization(nn.Module):
    """FFM pair dots from host-owned partner embeddings [B,F,F,D] -> [B,P].

    Entry (i,j) is field i embedded for interaction with field j. The host must
    actually use partner-specific parameters; repeating ordinary embeddings is
    only an FM control. Diagonal entries are unused by FFM. P=F(F-1)/2.
    """
    def __init__(self, num_fields, dim):
        super().__init__()
        _dims(num_fields, dim)
        if num_fields < 2:
            raise ValueError("pair interactions require at least two fields")
        self.num_fields, self.dim = num_fields, dim
        self.pairs = [(i, j) for i in range(num_fields) for j in range(i+1, num_fields)]

    def _prepare(self, embeddings, valid_mask):
        if embeddings.ndim != 4 or embeddings.shape[1:] != (self.num_fields, self.num_fields, self.dim):
            raise ValueError("partner embeddings must have shape [B,F,F,D]")
        if valid_mask is not None:
            if valid_mask.dtype != torch.bool or valid_mask.shape != embeddings.shape[:2]:
                raise ValueError("valid_mask must be bool [B,num_fields]")
            embeddings = torch.where(valid_mask[:,None,:,None],embeddings,0)
        flat, mask = _tokens(embeddings.flatten(2), valid_mask, self.num_fields, self.num_fields*self.dim)
        x = flat.reshape_as(embeddings)
        return torch.where(mask[:, None, :, None], x, 0), mask

    def forward(self, embeddings, valid_mask=None):
        x, _ = self._prepare(embeddings, valid_mask)
        return torch.stack([(x[:, i, j] * x[:, j, i]).sum(-1) for i, j in self.pairs], -1)


class OperationAwareInteraction(FieldAwareFactorization):
    """ONNv2 representation: diagonal copy embeddings, then partner pair dots.

    [B,F,F,D] -> [B,F*D+P]. Diagonal embeddings serve the copy operation and
    off-diagonal embeddings serve each pair operation. No DNN/head is added.
    """
    def forward(self, embeddings, valid_mask=None):
        x, _ = self._prepare(embeddings, valid_mask)
        diagonal = torch.stack([x[:, i, i] for i in range(self.num_fields)], 1).flatten(1)
        dots = torch.stack([(x[:, i, j] * x[:, j, i]).sum(-1) for i, j in self.pairs], -1)
        return torch.cat((diagonal, dots), -1)


class HigherOrderFactorization(nn.Module):
    """HOFM exact order-k elementary symmetric products, [B,F,D] -> [B,D].

    Dynamic programming sums products over distinct fields without enumerating
    tuples. One order and one supplied embedding bank; combine separate banks
    or orders in the host to reproduce independent per-order parameterization.
    """
    def __init__(self, num_fields, dim, order):
        super().__init__()
        _dims(num_fields, dim, order)
        if order > num_fields:
            raise ValueError("order cannot exceed num_fields")
        self.num_fields, self.dim, self.order = num_fields, dim, order

    def forward(self, tokens, valid_mask=None):
        x, _ = _tokens(tokens, valid_mask, self.num_fields, self.dim)
        terms = [torch.ones_like(x[:, 0])] + [torch.zeros_like(x[:, 0]) for _ in range(self.order)]
        for i in range(self.num_fields):
            terms = [terms[0]] + [terms[k] + x[:, i]*terms[k-1] for k in range(1, self.order+1)]
        return terms[-1]


class InteractionMachine(nn.Module):
    """DeepIM power-sum/Newton recurrence, concatenated orders 1..K -> [B,KD].

    Same elementary symmetric polynomials as HOFM, with an all-order readout.
    K<=5 follows the source; cancellation can occur at large magnitudes. No
    batch normalization, prediction head or parallel deep MLP is included.
    """
    def __init__(self, num_fields, dim, max_order):
        super().__init__()
        _dims(num_fields, dim, max_order)
        if max_order > min(5, num_fields):
            raise ValueError("max_order must be <= min(5,num_fields)")
        self.num_fields, self.dim, self.max_order = num_fields, dim, max_order

    def forward(self, tokens, valid_mask=None):
        x, _ = _tokens(tokens, valid_mask, self.num_fields, self.dim)
        powers = [None] + [x.pow(k).sum(1) for k in range(1, self.max_order+1)]
        terms = [torch.ones_like(x[:, 0])]
        for k in range(1, self.max_order+1):
            terms.append(sum((-1)**(j-1)*terms[k-j]*powers[j] for j in range(1, k+1))/k)
        return torch.cat(terms[1:], -1)


class HolographicInteraction(nn.Module):
    """HFM pair circular convolution or correlation, [B,F,D] -> [B,P,D].

    Correlation is irFFT(conj(rFFT(left))*rFFT(right)); convolution omits the
    conjugate. Explicit inverse length supports odd D. This is a circular
    coordinate interaction, not temporal convolution or an efficiency claim.
    """
    def __init__(self, num_fields, dim, mode="convolution"):
        super().__init__()
        _dims(num_fields, dim)
        if num_fields < 2 or mode not in ("convolution", "correlation"):
            raise ValueError("need at least two fields and convolution/correlation mode")
        self.num_fields, self.dim, self.mode = num_fields, dim, mode
        self.pairs = [(i,j) for i in range(num_fields) for j in range(i+1,num_fields)]

    def forward(self, tokens, valid_mask=None):
        x, _ = _tokens(tokens, valid_mask, self.num_fields, self.dim)
        spectrum = torch.fft.rfft(x, dim=-1)
        values = []
        for i,j in self.pairs:
            left = spectrum[:, i].conj() if self.mode == "correlation" else spectrum[:, i]
            values.append(torch.fft.irfft(left*spectrum[:, j], n=self.dim, dim=-1))
        return torch.stack(values, 1)


class DenseSparseInteraction(nn.Module):
    """DLRM composition: sparse tokens then one dense token; all pair dots + dense.

    dense [B,D], sparse [B,F,D], sparse mask [B,F] -> [B,(F+1)F/2+D].
    The host's bottom MLP must project its real dense features to D. This
    composition has no embedding tables, bottom/top MLP or new pair operator.
    """
    def __init__(self, num_fields, dim):
        super().__init__()
        _dims(num_fields, dim)
        self.num_fields, self.dim = num_fields, dim

    def forward(self, dense, sparse, valid_mask=None):
        sparse, _ = _tokens(sparse, valid_mask, self.num_fields, self.dim)
        dense, _ = _tokens(dense[:, None, :], None, 1, self.dim)
        if dense.shape[0] != sparse.shape[0]:
            raise ValueError("dense/sparse batches must match")
        x = torch.cat((sparse, dense), 1)
        pairs = [(x[:, i]*x[:, j]).sum(-1) for i in range(self.num_fields+1)
                 for j in range(i+1,self.num_fields+1)]
        return torch.cat((torch.stack(pairs, -1), dense[:, 0]), -1)


class LorentzInteraction(nn.Module):
    """LorentzFM triangle terms, spatial [B,F,D] -> pair values [B,P].

    Lift x to (sqrt(1+||x||²),x). For i<j return
    1+(1-dot(xi,xj)-ti-tj)/(ti*tj). Curvature is fixed at one. No manifold
    optimizer or field weights are supplied; keep embeddings in a finite
    numerical range. Missing pairs are zero, not a geometric observation.
    """
    def __init__(self, num_fields, dim):
        super().__init__()
        _dims(num_fields, dim)
        if num_fields < 2:
            raise ValueError("at least two fields are required")
        self.num_fields, self.dim = num_fields, dim

    def forward(self, tokens, valid_mask=None):
        x, mask = _tokens(tokens, valid_mask, self.num_fields, self.dim)
        # Float64 intermediates keep squared float32 embeddings from overflowing.
        work = x.double()
        time = torch.sqrt(1 + work.square().sum(-1))
        values = []
        for i in range(self.num_fields):
            for j in range(i+1,self.num_fields):
                score = 1+(1-(work[:, i]*work[:, j]).sum(-1)-time[:, i]-time[:, j])/(time[:, i]*time[:, j])
                values.append(torch.where(mask[:, i] & mask[:, j], score, 0))
        return torch.stack(values, -1).to(x.dtype)
