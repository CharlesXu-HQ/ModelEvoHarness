"""Independent native field mechanisms; FuxiCTR 183da1c944ba8930fcdcd54114837c081959d748.

Host owns data semantics, embeddings, heads and training. Boolean masks use
True=valid; masked nonfinite padding is sanitized before arithmetic. Sources
and deliberate mechanism boundaries are documented on each class.
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


def _weights(logits, mask):
    logits = logits.masked_fill(~mask, -torch.inf)
    logits = torch.where(mask.any(-1, keepdim=True), logits, 0)
    return logits.softmax(-1)*mask


class HierarchicalInteractionAggregation(nn.Module):
    """InterHAt higher-order aggregation on already contextualized field tokens.

    u_k=attention_pool(X_k); X_(k+1)=X_k+u_k*X_initial. A second attention
    pools the K vectors -> [B,D]. [B,F,D] input is the host's contextual X1;
    upstream initial self-attention/FFN and classifier are deliberately omitted.
    Source: model_zoo/InterHAt/src/InterHAt.py forward recurrence.
    """
    def __init__(self,num_fields,dim,order=2):
        super().__init__()
        _dims(num_fields,dim,order)
        self.num_fields,self.dim,self.order = num_fields,dim,order
        self.scorers = nn.ModuleList([nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,1,bias=False))
                                      for _ in range(order)])
        self.order_scorer = nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,1,bias=False))

    def forward(self,tokens,valid_mask=None):
        initial,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        current, summaries = initial, []
        for score in self.scorers:
            weights = _weights(score(current).squeeze(-1),mask)
            summary = (weights[...,None]*current).sum(1)
            summaries.append(summary)
            current = current+summary[:,None,:]*initial
        stacked = torch.stack(summaries,1)
        available = mask.any(1)[:,None].expand(-1,self.order)
        weights = _weights(self.order_scorer(stacked).squeeze(-1),available)
        return (weights[...,None]*stacked).sum(1)


class FieldGraphInteraction(nn.Module):
    """One FiGNN field-graph propagation with per-field input/output matrices.

    [B,F,D] -> [B,F,D]; optional return_attention gives graph [B,F,F].
    g_ij=softmax_j(leaky_relu(a[xi;xj])), excluding i=j and invalid fields.
    output_i=xi+W_in[i] sum_j g_ij W_out[j] xj. Empty neighbor rows retain
    their residual. This is the source's non-GRU local variant, one step;
    no graph reuse across steps, extra initial residual or prediction module.
    Source: model_zoo/FiGNN/src/FiGNN.py, GraphLayer and graph construction.
    """
    def __init__(self,num_fields,dim):
        super().__init__()
        _dims(num_fields,dim)
        self.num_fields,self.dim = num_fields,dim
        self.scorer = nn.Linear(2*dim,1,bias=False)
        self.input_kernel = nn.Parameter(torch.empty(num_fields,dim,dim))
        self.output_kernel = nn.Parameter(torch.empty(num_fields,dim,dim))
        nn.init.xavier_uniform_(self.input_kernel)
        nn.init.xavier_uniform_(self.output_kernel)

    def forward(self,tokens,valid_mask=None,return_attention=False):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        left = x[:,:,None,:].expand(-1,-1,self.num_fields,-1)
        right = x[:,None,:,:].expand(-1,self.num_fields,-1,-1)
        logits = self.scorer(torch.cat((left,right),-1)).squeeze(-1)
        logits = torch.nn.functional.leaky_relu(logits,negative_slope=.01)
        pairs = mask[:,:,None] & mask[:,None,:] & ~torch.eye(self.num_fields,dtype=torch.bool,device=x.device)
        graph = _weights(logits,pairs)
        sent = torch.einsum("fij,bfj->bfi",self.output_kernel,x)
        received = graph@sent
        output = x+torch.einsum("fij,bfj->bfi",self.input_kernel,received)
        return (output,graph) if return_attention else output


class DisentangledFieldAttention(nn.Module):
    """DESTINE centered pair attention + independent unary attention.

    [B,F,D] -> [B,F,D], D divisible by H. Q/K are centered over VALID fields;
    pair softmax and unary-key softmax each sum to one, so combined weights
    sum to two. Empty/query padding yields zero. Uses correct [B,H,F,d]
    head transpose rather than a direct batch/head view. No residual/head.
    Source: model_zoo/DESTINE/src/DESTINE.py, DisentangledSelfAttention.
    """
    def __init__(self,num_fields,dim,num_heads=1):
        super().__init__()
        _dims(num_fields,dim,num_heads)
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_fields,self.dim,self.num_heads = num_fields,dim,num_heads
        self.query,self.key,self.value = [nn.Linear(dim,dim) for _ in range(3)]
        self.unary = nn.Linear(dim,num_heads)

    def forward(self,tokens,valid_mask=None,return_attention=False):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        width = self.dim//self.num_heads
        def heads(layer):
            return layer(x).reshape(-1,self.num_fields,self.num_heads,width).transpose(1,2)
        q,k,v = heads(self.query),heads(self.key),heads(self.value)
        valid = mask[:,None,:,None]
        count = valid.sum(2,keepdim=True).clamp_min(1)
        q = q-(q*valid).sum(2,keepdim=True)/count
        k = k-(k*valid).sum(2,keepdim=True)/count
        pairs = mask[:,None,:,None] & mask[:,None,None,:]
        pair_weights = _weights(q@k.transpose(-1,-2)/width**.5,pairs)
        unary_weights = _weights(self.unary(x).transpose(1,2),mask[:,None,:])
        weights = pair_weights+unary_weights[:,:,None,:]*valid
        output = (weights@v).transpose(1,2).reshape(-1,self.num_fields,self.dim)
        return (output,weights) if return_attention else output


class SelfAttentionInteraction(nn.Module):
    """SAM 2A/2E/3A/3E raw pair-vector interactions (no softmax).

    2: dot(xi,xj) multiplies learned pair vector (A) or xi*xj (E), producing
    [B,F,F,D]. 3 uses dot(xi,Kxj), sums j -> [B,F,D] and optionally adds Qxi.
    Self pairs are included; padding is zero. No dropout or output aggregation.
    Source: model_zoo/SAM/src/SAM.py, SAM2A/E and SAM3A/E.
    """
    def __init__(self,num_fields,dim,mode="2E",residual=True):
        super().__init__()
        _dims(num_fields,dim)
        if mode not in ("2A","2E","3A","3E"):
            raise ValueError("mode must be 2A, 2E, 3A or 3E")
        self.num_fields,self.dim,self.mode = num_fields,dim,mode
        self.pair_vectors = nn.Parameter(torch.ones(num_fields,num_fields,dim)) if mode.endswith("A") else None
        self.key = nn.Linear(dim,dim,bias=False) if mode.startswith("3") else None
        self.residual = nn.Linear(dim,dim,bias=False) if self.key is not None and residual else None

    def forward(self,tokens,valid_mask=None):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        key = self.key(x) if self.key is not None else x
        dots = x@key.transpose(-1,-2)
        vectors = self.pair_vectors if self.pair_vectors is not None else x[:,:,None,:]*x[:,None,:,:]
        output = dots[...,None]*vectors
        if self.key is not None:
            output = output.sum(2)
            if self.residual is not None:
                output = output+self.residual(x)
        return output
