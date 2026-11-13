"""Independent tensor-first CTR crossing, masking and interaction components.

Mechanism references: reczoo/FuxiCTR at
183da1c944ba8930fcdcd54114837c081959d748, model_zoo/{GDCN,MaskNet,
FinalMLP,EDCN,WuKong,FmFM,FLEN,FinalNet,QNN}. These are original native
implementations; the host owns embeddings, head, objective and optimizer.
All inputs here are fully defined floating point tensors, without padding.
EDCN uses distinct stream gates; QNN restores field-coordinate order after
head operations. Auxiliary consistency losses are deliberately separate.
"""

from itertools import combinations
from numbers import Integral
import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive(*values):
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError('dimensions and counts must be positive integers')


def _matrix(x, width):
    if x.ndim != 2 or x.shape[-1] != width:
        raise ValueError(f'expected [B,{width}] tensor')


def _tokens(x, fields, dim):
    if x.ndim != 3 or x.shape[1:] != (fields, dim):
        raise ValueError(f'expected [B,{fields},{dim}] tensor')


def _same_batch(x, y):
    if x.shape[0] != y.shape[0]:
        raise ValueError('inputs must have the same batch size')


class GatedCross(nn.Module):
    """[B,D] -> [B,D]; x += x0*(W(x)+b)*sigmoid(G(x)) at each depth.

    Matrix crossing plus coordinate gates; no FDO embedding optimization.
    """
    def __init__(self, input_dim: int, num_layers: int = 2):
        super().__init__()
        _positive(input_dim, num_layers)
        self.input_dim = input_dim
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim) for _ in range(num_layers))
        self.gate_layers = nn.ModuleList(nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers))

    def forward(self, x):
        _matrix(x, self.input_dim)
        initial, hidden = x, x
        for cross, gate in zip(self.cross_layers, self.gate_layers):
            hidden = hidden + initial * cross(hidden) * torch.sigmoid(gate(hidden))
        return hidden


class MaskBlock(nn.Module):
    """(original[B,I], hidden[B,H]) -> [B,O], using an unbounded mask.

    ReLU(LN(W(mask_MLP(original)*hidden))). Reuse original for serial
    composition, or independent blocks on the same hidden for parallel paths.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 mask_hidden_dim: int, layer_norm: bool = True):
        super().__init__()
        _positive(input_dim, hidden_dim, output_dim, mask_hidden_dim)
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.mask_in = nn.Linear(input_dim, mask_hidden_dim)
        self.mask_out = nn.Linear(mask_hidden_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim, eps=1e-5) if layer_norm else nn.Identity()

    def forward(self, original, hidden):
        _matrix(original, self.input_dim)
        _matrix(hidden, self.hidden_dim)
        _same_batch(original, hidden)
        mask = self.mask_out(F.relu(self.mask_in(original)))
        return F.relu(self.norm(self.projection(mask * hidden)))


class ContextFeatureGate(nn.Module):
    """(features[B,F], context[B,C]) -> [B,F], scaled by 2*sigmoid(MLP).

    Instantiate independently for distinct stream contexts; all context tensors
    are host-supplied. A constant context provides a global learned gate.
    """
    def __init__(self, feature_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        _positive(feature_dim, context_dim, hidden_dim)
        self.feature_dim, self.context_dim = feature_dim, context_dim
        self.gate_in = nn.Linear(context_dim, hidden_dim)
        self.gate_out = nn.Linear(hidden_dim, feature_dim)

    def forward(self, features, context):
        _matrix(features, self.feature_dim)
        _matrix(context, self.context_dim)
        _same_batch(features, context)
        return features * (2 * torch.sigmoid(self.gate_out(F.relu(self.gate_in(context)))))


class MultiHeadBilinearFusion(nn.Module):
    """Two [B,D] streams -> [B,O]; affine terms plus headwise bilinear terms.

    Head partitions are contiguous within each stream; cross-head bilinear
    parameters are excluded. No softmax attention or automatic branch creation.
    """
    def __init__(self, left_dim: int, right_dim: int, output_dim: int = 1, num_heads: int = 1):
        super().__init__()
        _positive(left_dim, right_dim, output_dim, num_heads)
        if left_dim % num_heads or right_dim % num_heads:
            raise ValueError('stream dimensions must be divisible by num_heads')
        self.left_dim, self.right_dim = left_dim, right_dim
        self.num_heads = num_heads
        self.left_linear = nn.Linear(left_dim, output_dim)
        self.right_linear = nn.Linear(right_dim, output_dim)
        self.interaction = nn.Parameter(torch.empty(num_heads, left_dim // num_heads,
                                                   right_dim // num_heads, output_dim))
        nn.init.xavier_uniform_(self.interaction)

    def forward(self, left, right):
        _matrix(left, self.left_dim)
        _matrix(right, self.right_dim)
        _same_batch(left, right)
        x = left.reshape(left.shape[0], self.num_heads, self.left_dim // self.num_heads)
        y = right.reshape(right.shape[0], self.num_heads, self.right_dim // self.num_heads)
        return self.left_linear(left) + self.right_linear(right) + torch.einsum('bhi,hijo,bhj->bo', x, self.interaction, y)


class CrossDeepBridge(nn.Module):
    """(cross[B,D], deep[B,D]) -> [B,D] between corresponding stream depths.

    mode: product, sum, concat (project+ReLU), attention (independent stream
    softmax gates). Corrects upstream attention1 reuse on the second stream.
    """
    def __init__(self, dim: int, mode: str = 'product'):
        super().__init__()
        _positive(dim)
        if mode not in ('product', 'sum', 'concat', 'attention'):
            raise ValueError('mode must be product, sum, concat or attention')
        self.dim, self.mode = dim, mode
        if mode == 'concat':
            self.projection = nn.Linear(2 * dim, dim)
        elif mode == 'attention':
            self.cross_gate = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.deep_gate = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, cross, deep):
        _matrix(cross, self.dim)
        _matrix(deep, self.dim)
        _same_batch(cross, deep)
        if self.mode == 'product':
            return cross * deep
        if self.mode == 'sum':
            return cross + deep
        if self.mode == 'concat':
            return F.relu(self.projection(torch.cat((cross, deep), -1)))
        return cross * self.cross_gate(cross).softmax(-1) + deep * self.deep_gate(deep).softmax(-1)


class FieldRegulation(nn.Module):
    """[B,F,D] -> two [B,F,D] streams with independent global field weights.

    Weights softmax over F at the specified temperature, shared over D.
    No sample-conditioned routing or implicit batch normalization.
    """
    def __init__(self, num_fields: int, embedding_dim: int, temperature: float = 1.0):
        super().__init__()
        _positive(num_fields, embedding_dim)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        self.num_fields, self.embedding_dim, self.temperature = num_fields, embedding_dim, temperature
        self.cross_logits = nn.Parameter(torch.zeros(num_fields))
        self.deep_logits = nn.Parameter(torch.zeros(num_fields))

    def forward(self, tokens):
        _tokens(tokens, self.num_fields, self.embedding_dim)
        cross = (self.cross_logits / self.temperature).softmax(0)[None, :, None]
        deep = (self.deep_logits / self.temperature).softmax(0)[None, :, None]
        return tokens * cross, tokens * deep


class WuKongBlock(nn.Module):
    """[B,F,D] -> [B,interaction_tokens+linear_tokens,D].

    FMB computes X(X^T P), normalizes its flattened F*rank entries and uses
    a two-layer ReLU MLP to create tokens. LCB linearly compresses fields.
    Concatenate FMB then LCB, add field-projected residual, optionally LN.
    Host chooses output token count to match its backbone; no scaling claim.
    """
    def __init__(self, num_fields: int, embedding_dim: int, linear_tokens: int = 2,
                 interaction_tokens: int = 2, rank: int = 2, hidden_dim: int = 16,
                 layer_norm: bool = True):
        super().__init__()
        _positive(num_fields, embedding_dim, linear_tokens, interaction_tokens, rank, hidden_dim)
        self.num_fields, self.embedding_dim = num_fields, embedding_dim
        self.interaction_tokens = interaction_tokens
        self.output_tokens = linear_tokens + interaction_tokens
        self.gram_projection = nn.Parameter(torch.empty(num_fields, rank))
        nn.init.xavier_uniform_(self.gram_projection)
        self.gram_norm = nn.LayerNorm(num_fields * rank, eps=1e-5)
        self.fmb_in = nn.Linear(num_fields * rank, hidden_dim)
        self.fmb_out = nn.Linear(hidden_dim, interaction_tokens * embedding_dim)
        self.linear_compression = nn.Linear(num_fields, linear_tokens, bias=False)
        self.residual_projection = nn.Linear(num_fields, self.output_tokens) if self.output_tokens != num_fields else nn.Identity()
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-5) if layer_norm else nn.Identity()

    def compressed_gram(self, tokens):
        """Projected pair affinities [B,F,rank], without forming full F*F Gram."""
        _tokens(tokens, self.num_fields, self.embedding_dim)
        return tokens @ (tokens.transpose(1, 2) @ self.gram_projection)

    def forward(self, tokens):
        gram = self.gram_norm(self.compressed_gram(tokens).flatten(1))
        fmb = F.relu(self.fmb_out(F.relu(self.fmb_in(gram))))
        fmb = fmb.reshape(tokens.shape[0], self.interaction_tokens, self.embedding_dim)
        lcb = self.linear_compression(tokens.transpose(1, 2)).transpose(1, 2)
        residual = self.residual_projection(tokens.transpose(1, 2)).transpose(1, 2)
        return self.norm(torch.cat((fmb, lcb), 1) + residual)


class FieldMatrixInteraction(nn.Module):
    """[B,F,D] -> [B,F*(F-1)/2] pair scores, lexicographic i<j order.

    matrix uses e_i^T M_ij e_j; vector uses sum(e_i*v_ij*e_j).
    Existing embeddings are supplied; no feature-ID table or classifier.
    """
    def __init__(self, num_fields: int, embedding_dim: int, interaction_type: str = 'matrix'):
        super().__init__()
        _positive(num_fields, embedding_dim)
        if num_fields < 2 or interaction_type not in ('matrix', 'vector'):
            raise ValueError('at least two fields and matrix/vector interaction_type required')
        self.num_fields, self.embedding_dim, self.interaction_type = num_fields, embedding_dim, interaction_type
        pairs = list(combinations(range(num_fields), 2))
        self.register_buffer('left_indices', torch.tensor([i for i, _ in pairs]))
        self.register_buffer('right_indices', torch.tensor([j for _, j in pairs]))
        shape = (len(pairs), embedding_dim, embedding_dim) if interaction_type == 'matrix' else (len(pairs), embedding_dim)
        self.pair_weight = nn.Parameter(torch.empty(shape))
        nn.init.xavier_uniform_(self.pair_weight)

    def forward(self, tokens):
        _tokens(tokens, self.num_fields, self.embedding_dim)
        left, right = tokens[:, self.left_indices], tokens[:, self.right_indices]
        if self.interaction_type == 'matrix':
            return torch.einsum('bpi,pij,bpj->bp', left, self.pair_weight, right)
        return (left * self.pair_weight * right).sum(-1)


class GroupedFieldInteraction(nn.Module):
    """[B,F,D] -> [B,D], weighted within-group FM + across-group products.

    field_groups supplies one contiguous nonnegative group ID per field.
    At least two nonempty groups required; singleton within terms are zero.
    Generalizes FLEN's three hardcoded source groups; host adds the deep head.
    """
    def __init__(self, field_groups, embedding_dim: int):
        super().__init__()
        _positive(embedding_dim)
        groups = tuple(field_groups)
        if not groups or any(not isinstance(g, Integral) or isinstance(g, bool) or g < 0 for g in groups):
            raise ValueError('field_groups must contain nonnegative integer IDs')
        count = max(groups) + 1
        if count < 2 or set(groups) != set(range(count)):
            raise ValueError('field_groups must cover contiguous IDs for at least two groups')
        self.num_fields, self.embedding_dim = len(groups), embedding_dim
        self.register_buffer('group_membership', F.one_hot(torch.tensor(groups), count).float())
        self.pairs = tuple(combinations(range(count), 2))
        self.within_weights = nn.Parameter(torch.ones(count))
        self.across_weights = nn.Parameter(torch.ones(len(self.pairs)))

    def forward(self, tokens):
        _tokens(tokens, self.num_fields, self.embedding_dim)
        membership = self.group_membership.to(tokens.dtype)
        pooled = torch.einsum('bfd,fg->bgd', tokens, membership)
        squared = torch.einsum('bfd,fg->bgd', tokens.square(), membership)
        within = (pooled.square() - squared) * .5
        across = torch.stack([pooled[:, i] * pooled[:, j] for i, j in self.pairs], 1)
        return (within * self.within_weights[None, :, None]).sum(1) + (across * self.across_weights[None, :, None]).sum(1)


class FactorizedInteraction(nn.Module):
    """[B,I] -> [B,O]; affine halves a,b give a+a*b or concat(a,a*b).

    Sum form is expressible with the installed LatentCross plus projection.
    A local convenience layer, without FinalNet's ensemble training losses.
    """
    def __init__(self, input_dim: int, output_dim: int, residual_type: str = 'sum'):
        super().__init__()
        _positive(input_dim, output_dim)
        if residual_type not in ('sum', 'concat') or (residual_type == 'concat' and output_dim % 2):
            raise ValueError('residual_type must be sum/concat; concat needs an even output_dim')
        self.input_dim, self.residual_type = input_dim, residual_type
        self.projection = nn.Linear(input_dim, 2 * output_dim if residual_type == 'sum' else output_dim)

    def forward(self, x):
        _matrix(x, self.input_dim)
        a, b = self.projection(x).chunk(2, -1)
        return a + a * b if self.residual_type == 'sum' else torch.cat((a, a * b), -1)


class QuadraticInteraction(nn.Module):
    """[B,F*D] -> same shape; QNN-alpha-style head/row projected products.

    Split each field's coordinates into H heads; project x to [H,R,F,D/H],
    apply ReLU, multiply by matching x head and sum R. Restore original F,D
    coordinate order before the residual. This corrects the source's direct
    flattening of head-major results. No batch norm, dropout or ensemble loss.
    """
    def __init__(self, input_dim: int, num_fields: int, num_heads: int = 1, num_rows: int = 2):
        super().__init__()
        _positive(input_dim, num_fields, num_heads, num_rows)
        if input_dim % num_fields or (input_dim // num_fields) % num_heads:
            raise ValueError('input_dim must divide into fields with embedding_dim divisible by heads')
        self.input_dim, self.num_fields, self.num_heads, self.num_rows = input_dim, num_fields, num_heads, num_rows
        self.embedding_dim = input_dim // num_fields
        self.projection = nn.Linear(input_dim, input_dim * num_rows)

    def forward(self, x):
        _matrix(x, self.input_dim)
        batch, channels = x.shape[0], self.embedding_dim // self.num_heads
        heads = x.reshape(batch, self.num_fields, self.num_heads, channels).permute(0, 2, 1, 3)
        rows = F.relu(self.projection(x)).reshape(batch, self.num_rows, self.num_fields, self.num_heads, channels).permute(0, 3, 1, 2, 4)
        products = (heads[:, :, None] * rows).sum(2)
        return x + products.permute(0, 2, 1, 3).reshape(batch, self.input_dim)
