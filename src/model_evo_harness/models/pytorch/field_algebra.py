"""Independent native field mechanisms; FuxiCTR 183da1c944ba8930fcdcd54114837c081959d748.

Host owns data semantics, embeddings, heads and training. Boolean masks use
True=valid; masked nonfinite padding is sanitized before arithmetic. Sources
and deliberate mechanism boundaries are documented on each class.
"""

from numbers import Integral
import math
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


def _bounds(epsilon, max_log):
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(max_log) or not 0 < max_log <= 60:
        raise ValueError("epsilon must be positive finite; max_log must lie in (0,60]")


class LogarithmicInteraction(nn.Module):
    """AFN core: exp(clamp(sum_f orders[f,o]*log(max(abs(x),eps)))).

    [B,F,D] -> [B,O,D]. Sign is discarded as in AFN. Missing fields contribute
    zero log (multiplicative identity); fully absent samples return zero.
    Clipping bounds overflow but saturates gradients. This local operator
    omits upstream log/exp BatchNorm, ensemble DNN and prediction head.
    Source: model_zoo/AFN/src/AFN.py, logarithmic_net.
    """
    def __init__(self, num_fields, dim, out_fields, epsilon=1e-5, max_log=20.):
        super().__init__()
        _dims(num_fields, dim, out_fields)
        _bounds(epsilon, max_log)
        self.num_fields, self.dim, self.out_fields = num_fields, dim, out_fields
        self.epsilon, self.max_log = epsilon, max_log
        self.orders = nn.Parameter(torch.empty(num_fields, out_fields))
        nn.init.xavier_uniform_(self.orders)

    def forward(self, tokens, valid_mask=None):
        x, mask = _tokens(tokens, valid_mask, self.num_fields, self.dim)
        logs = torch.where(mask[..., None], x.abs().clamp_min(self.epsilon).log(), 0)
        exponent = torch.einsum("bfd,fo->bod", logs, self.orders)
        output = exponent.clamp(-self.max_log, self.max_log).exp()
        return torch.where(mask.any(1)[:, None, None], output, 0)


class EulerInteraction(nn.Module):
    """EulerNet explicit magnitude/phase orders plus shared implicit ReLU map.

    (real,imaginary) [B,F,D] each -> two [B,O,D] tensors. Same learned orders
    mix log-magnitudes and phases; each channel has a magnitude/phase bias.
    Implicit map is shared between real and imaginary flattened inputs.
    Zero coordinates have phase zero and floored magnitude with zero phase
    gradient; invalid fields are neutral in explicit products. Exponents are
    clipped, and all-absent outputs are zero. No initial cos/sin embedding
    conversion, dropout, normalization or head. Source: EulerNet.py layer.
    """
    def __init__(self, num_fields, dim, out_fields, epsilon=1e-8, max_log=20.):
        super().__init__()
        _dims(num_fields, dim, out_fields)
        _bounds(epsilon, max_log)
        self.num_fields, self.dim, self.out_fields = num_fields, dim, out_fields
        self.epsilon, self.max_log = epsilon, max_log
        self.orders = nn.Parameter(torch.empty(num_fields, out_fields))
        nn.init.xavier_uniform_(self.orders)
        self.log_bias = nn.Parameter(torch.zeros(out_fields, dim))
        self.phase_bias = nn.Parameter(torch.zeros(out_fields, dim))
        self.implicit = nn.Linear(num_fields*dim, out_fields*dim)

    def forward(self, real, imaginary, valid_mask=None):
        r, mask = _tokens(real, valid_mask, self.num_fields, self.dim)
        p, _ = _tokens(imaginary, mask, self.num_fields, self.dim)
        if p.shape != r.shape:
            raise ValueError("real/imaginary shapes must match")
        radius2 = r.square()+p.square()
        nonzero = radius2 > self.epsilon
        phase = torch.atan2(torch.where(nonzero, p, 0), torch.where(nonzero, r, 1))
        log_radius = torch.where(mask[..., None], .5*radius2.clamp_min(self.epsilon).log(), 0)
        exponent = torch.einsum("bfd,fo->bod", log_radius, self.orders)+self.log_bias
        angle = torch.einsum("bfd,fo->bod", phase, self.orders)+self.phase_bias
        magnitude = exponent.clamp(-self.max_log, self.max_log).exp()
        r_out = self.implicit(r.flatten(1)).relu().reshape(-1,self.out_fields,self.dim)+magnitude*angle.cos()
        p_out = self.implicit(p.flatten(1)).relu().reshape(-1,self.out_fields,self.dim)+magnitude*angle.sin()
        present = mask.any(1)[:, None, None]
        return torch.where(present,r_out,0), torch.where(present,p_out,0)


class GeneralizedInteraction(nn.Module):
    """AOANet generalized outer-product fusion, two explicit tensor axes.

    initial [B,F,D], current [B,S,D], optional masks [B,F]/[B,S] -> [B,O,D].
    Fusion[o,d,e]=sum_s,f alpha[s,f,o]*initial[f,d]*current[s,e], then multiply
    elementwise by W[o,d,e] and contract with h[o,e]. This keeps full channel
    matrices and learned field/subspace fusion; no parallel DNN or head.
    Source: model_zoo/AOANet/src/AOANet.py, GeneralizedInteraction.
    """
    def __init__(self, num_fields, input_subspaces, dim, output_subspaces):
        super().__init__()
        _dims(num_fields, input_subspaces, dim, output_subspaces)
        self.num_fields, self.input_subspaces, self.dim = num_fields,input_subspaces,dim
        self.alpha = nn.Parameter(torch.ones(input_subspaces,num_fields,output_subspaces))
        self.kernel = nn.Parameter(torch.eye(dim).repeat(output_subspaces,1,1))
        self.h = nn.Parameter(torch.ones(output_subspaces,dim))

    def forward(self, initial, current, field_mask=None, subspace_mask=None):
        x,_ = _tokens(initial,field_mask,self.num_fields,self.dim)
        y,_ = _tokens(current,subspace_mask,self.input_subspaces,self.dim)
        if x.shape[0] != y.shape[0]:
            raise ValueError("initial/current batches must match")
        fusion = torch.einsum("bfd,bse,sfo->bode",x,y,self.alpha)
        return (fusion*self.kernel*self.h[:,None,:]).sum(-1)
