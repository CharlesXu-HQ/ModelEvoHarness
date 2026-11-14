"""Independent SIM/ETA/SDIM/TWIN/MIRRN tensor mechanisms.

Reference: reczoo/FuxiCTR, commit 183da1c944ba8930fcdcd54114837c081959d748,
model_zoo/LongCTR/{SIM,ETA,SDIM,TWIN,MIRRN}. These are local mechanisms,
not full production models: host owns item embeddings, time-safe histories,
short-term branches, prediction heads, losses and evaluation. Hash search scans
supplied tensors; it is not an ANN index. True means valid. Invalid selections
have index -1, zero values and False masks; fixed K can exceed valid length.
"""
import math

import torch
from torch import nn


def _inputs(target, history, valid, dim):
    if (target.ndim != 2 or history.ndim != 3 or valid.ndim != 2
            or target.shape != (history.shape[0], dim)
            or history.shape[-1] != dim or valid.shape != history.shape[:2]
            or valid.dtype != torch.bool):
        raise ValueError("expected target [B,D], history [B,L,D], boolean valid [B,L]")
    return torch.where(valid[..., None], history, 0.)


def _weights(scores, valid):
    safe = torch.where(valid, scores, float("-inf"))
    dummy = torch.where(valid.any(-1, keepdim=True), float("-inf"), 0.)
    return torch.softmax(torch.cat([safe, dummy], -1), -1)[..., :-1] * valid


def _mean(history, valid):
    return history.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


def _select(history, valid, scores, k, chronological=False):
    batch, length, dim = history.shape
    values = torch.cat([history, history.new_zeros(batch, k, dim)], 1)
    mask = torch.cat([valid, valid.new_zeros(batch, k)], 1)
    score = torch.cat([torch.where(valid, scores, float("-inf")), scores.new_full((batch, k), float("-inf"))], 1)
    index = torch.argsort(score, dim=-1, descending=True, stable=True)[:, :k]
    selected_valid = torch.gather(mask, 1, index)
    if chronological:
        order = torch.argsort(torch.where(selected_valid, index, length + k), dim=-1, stable=True)
        index = torch.gather(index, 1, order)
        selected_valid = torch.gather(selected_valid, 1, order)
    selected = torch.gather(values, 1, index[..., None].expand(-1, -1, dim))
    selected = torch.where(selected_valid[..., None], selected, 0.)
    return selected, selected_valid, torch.where(selected_valid, index, -1)


class _TargetAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.scale = math.sqrt(dim)

    def forward(self, target, history, valid):
        scores = torch.einsum("bd,bld->bl", self.query(target), self.key(history)) / self.scale
        attention = _weights(scores, valid)
        return torch.einsum("bl,bld->bd", attention, self.value(history)), attention


class SIMHistorySelector(nn.Module):
    """Learned dot-product coarse sum and top-K target-attention fine readout.

    Returns coarse/fine [B,D], raw coarse_scores [B,L], selected [B,K,D],
    selected_mask/indices [B,K], attention [B,K], representation=fine.
    Coarse scores are unnormalized (including negatives), as in SIM. Selection
    masks invalid events before ranking, correcting zero-scored padding wins.
    The hard indices have no gradient; coarse supervision trains the selector.
    """
    def __init__(self, dim, topk=5):
        super().__init__()
        if dim < 1 or topk < 1:
            raise ValueError("dim and topk must be positive")
        self.dim, self.topk = dim, topk
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.fine_attention = _TargetAttention(dim)

    def forward(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim)
        scores = torch.einsum("bd,bld->bl", self.query(target), self.key(history))
        scores = torch.where(valid, scores, 0.)
        coarse = torch.einsum("bl,bld->bd", scores, history)
        selected, mask, indices = _select(history, valid, scores, self.topk)
        fine, attention = self.fine_attention(target, selected, mask)
        return dict(representation=fine, coarse=coarse, fine=fine, coarse_scores=scores,
                    selected=selected, selected_mask=mask, indices=indices, attention=attention)


class ETAHistorySelector(nn.Module):
    """Fixed random hyperplane Hamming top-K, then learned target attention.

    rotations [D,bits] are non-trainable and checkpointed. Stable ties prefer
    earlier input indices. Hashes use negative-projection bits (zero -> False).
    Returns selected values/mask/indices, hash_distance [B,L] (invalid=0),
    attention [B,K], and representation [B,D]. No stochastic rehashing.
    """
    def __init__(self, dim, topk=5, hash_bits=16):
        super().__init__()
        if min(dim, topk, hash_bits) < 1:
            raise ValueError("dimensions and topk must be positive")
        self.dim, self.topk = dim, topk
        self.register_buffer("rotations", torch.randn(dim, hash_bits))
        self.attention = _TargetAttention(dim)

    def forward(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim)
        distance = ((history @ self.rotations < 0) != (target @ self.rotations < 0)[:, None]).sum(-1)
        selected, mask, indices = _select(history, valid, -distance.to(history.dtype), self.topk)
        representation, attention = self.attention(target, selected, mask)
        return dict(representation=representation, selected=selected, selected_mask=mask,
                    indices=indices, hash_distance=torch.where(valid, distance, 0), attention=attention)


class SDIMCollisionPooling(nn.Module):
    """Sum exact target-bucket collisions per hash, then average hash outputs.

    Returns collision_mask [B,H,L], per_hash [B,H,D], representation [B,D].
    This preserves the unnormalized source option, without optional L2 norm.
    No learned parameters: gradients flow to selected history embeddings,
    while signs/collision membership are discrete and target is not trained.
    """
    def __init__(self, dim, num_hashes=2, hash_bits=4):
        super().__init__()
        if min(dim, num_hashes, hash_bits) < 1:
            raise ValueError("dimensions and hash counts must be positive")
        self.dim = dim
        self.register_buffer("rotations", torch.randn(num_hashes, dim, hash_bits))

    def forward(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim)
        history_bits = torch.einsum("bld,hdf->bhlf", history, self.rotations) < 0
        target_bits = torch.einsum("bd,hdf->bhf", target, self.rotations) < 0
        collision = (history_bits == target_bits[:, :, None]).all(-1) & valid[:, None]
        per_hash = torch.einsum("bhl,bld->bhd", collision.to(history.dtype), history)
        return dict(representation=per_hash.mean(1), per_hash=per_hash, collision_mask=collision)


class TWINHeadwiseSelector(nn.Module):
    """Each learned attention head retrieves its own top-K keys and values.

    Uses the same q/k scores for retrieval and fine normalization. Returns
    indices/selected_mask/attention [B,H,K], selected [B,H,K,D/H], heads
    [B,H,D/H] and projected representation [B,D]. Dense search, no ANN cache.
    """
    def __init__(self, dim, num_heads=2, topk=5):
        super().__init__()
        if min(dim, num_heads, topk) < 1 or dim % num_heads:
            raise ValueError("positive dimensions and dim divisible by heads required")
        self.dim, self.num_heads, self.topk = dim, num_heads, topk
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, target, history, valid):
        history = _inputs(target, history, valid, self.dim)
        b, length, _ = history.shape
        h, d = self.num_heads, self.dim // self.num_heads
        query = self.query(target).reshape(b, h, d)
        key = self.key(history).reshape(b, length, h, d).transpose(1, 2)
        value = self.value(history).reshape(b, length, h, d).transpose(1, 2)
        scores = torch.einsum("bhd,bhld->bhl", query, key) / math.sqrt(d)
        mask = valid[:, None].expand(b, h, length).reshape(b*h, length)
        selected, selected_mask, indices = _select(value.reshape(b*h, length, d), mask,
                                                   scores.reshape(b*h, length), self.topk)
        # The same selected scores must drive fine attention, including empty rows.
        padded_scores = torch.cat([scores.reshape(b*h, length), scores.new_zeros(b*h, 1)], -1)
        chosen = torch.gather(padded_scores, 1, torch.where(selected_mask, indices, length))
        attention = _weights(chosen, selected_mask)
        heads = (attention[..., None] * selected).sum(1).reshape(b, h, d)
        return dict(representation=self.out_proj(heads.reshape(b, self.dim)), heads=heads,
                    selected=selected.reshape(b, h, self.topk, d),
                    selected_mask=selected_mask.reshape(b, h, self.topk),
                    indices=indices.reshape(b, h, self.topk), attention=attention.reshape(b, h, self.topk))


class MIRRNHistoryRefiner(nn.Module):
    """Target/recent/global hash selectors followed by masked frequency refinement.

    recent_mask [B,L] declares actual recent events and must be a subset of
    valid. Each query selects K original history indices, restored to temporal
    order. Refinement uses rFFT/irFFT with explicit n=K (odd K supported),
    orthonormal transforms, full complex block-channel multiplication and a
    residual LayerNorm; selected masks are applied before FFT and after norm.
    Matrices are shared across frequencies. This intentionally fixes the source
    diagonal-only einsum and unused masks; no recency-position addition/dropout.
    Zero padding participates in the declared K-length transform. Returns
    interests [B,3,D], refined/selected [B,3,K,D], aligned indices/masks [B,3,K],
    interest_mask [B,3], attention [B,3], representation [B,D].
    """
    def __init__(self, dim, topk=5, hash_bits=16, num_blocks=2):
        super().__init__()
        if min(dim, topk, hash_bits, num_blocks) < 1 or dim % num_blocks:
            raise ValueError("positive sizes and dim divisible by blocks required")
        self.dim, self.topk, self.num_blocks = dim, topk, num_blocks
        self.register_buffer("rotations", torch.randn(dim, hash_bits))
        width = dim // num_blocks
        self.frequency_real = nn.Parameter(torch.randn(3, num_blocks, width, width) * .02)
        self.frequency_imag = nn.Parameter(torch.randn(3, num_blocks, width, width) * .02)
        self.norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-5) for _ in range(3)])
        self.attention = _TargetAttention(dim)

    def refine(self, selected, selected_mask, branch):
        values = torch.where(selected_mask[..., None], selected, 0.)
        b, length, _ = values.shape
        spectrum = torch.fft.rfft(values, dim=1, norm="ortho")
        spectrum = spectrum.reshape(b, length//2+1, self.num_blocks, self.dim//self.num_blocks)
        matrix = torch.complex(self.frequency_real[branch], self.frequency_imag[branch])
        mixed = torch.einsum("bfni,nij->bfnj", spectrum, matrix)
        filtered = torch.fft.irfft(mixed.reshape(b, length//2+1, self.dim), n=length, dim=1, norm="ortho")
        return torch.where(selected_mask[..., None], self.norms[branch](values + filtered), 0.)

    def forward(self, target, history, valid, recent_mask):
        history = _inputs(target, history, valid, self.dim)
        if recent_mask.shape != valid.shape or recent_mask.dtype != torch.bool or bool((recent_mask & ~valid).any()):
            raise ValueError("recent_mask must be a boolean subset of valid")
        queries = [target, _mean(torch.where(recent_mask[..., None], history, 0.), recent_mask), _mean(history, valid)]
        availability = [valid.any(1), recent_mask.any(1), valid.any(1)]
        history_hash = history @ self.rotations > 0
        selected_all, masks, indices_all, refined_all, interests = [], [], [], [], []
        for i, query in enumerate(queries):
            score = -(history_hash != (query @ self.rotations > 0)[:, None]).sum(-1).to(history.dtype)
            selected, mask, indices = _select(history, valid & availability[i][:, None], score,
                                              self.topk, chronological=True)
            refined = self.refine(selected, mask, i)
            selected_all.append(selected); masks.append(mask); indices_all.append(indices)
            refined_all.append(refined); interests.append(_mean(refined, mask))
        interests = torch.stack(interests, 1)
        interest_mask = torch.stack(availability, 1)
        representation, attention = self.attention(target, interests, interest_mask)
        return dict(representation=representation, interests=interests, interest_mask=interest_mask,
                    refined=torch.stack(refined_all, 1), selected=torch.stack(selected_all, 1),
                    selected_mask=torch.stack(masks, 1), indices=torch.stack(indices_all, 1), attention=attention)
