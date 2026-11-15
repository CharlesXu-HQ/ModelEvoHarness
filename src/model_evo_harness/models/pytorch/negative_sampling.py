"""Host-controlled conditional negative sampling, independent of model scoring."""
import math
from numbers import Real

import torch
from torch import nn


class ConditionalNegativeSampler(nn.Module):
    """Draw with replacement from a declared training candidate catalog.

    ``counts[N]`` must be nonnegative finite training-period counts, aligned with
    both boolean masks ``[B,N]``. The host supplies historical eligibility and
    known positives; a sampled non-interaction is not an observed dislike.

    For popularity, first form p = counts**alpha / sum(counts**alpha), then
    base = (1-uniform_mix)*p + uniform_mix/N. Uniform uses base = 1/N.
    Condition base on eligible_mask & ~positive_mask separately for each row.
    Thus uniform_mix is a catalog mixture weight, not its final per-row share.
    Alpha zero gives equal weights, but a used popularity branch still requires
    at least one positive count. Undefined distributions raise without fallback.

    Return item_ids[B,K] and sampling_probs[B,K], the conditional probability
    of each independent draw. These are neither exposure propensities nor
    without-replacement inclusion probabilities. This is not RALM replication.
    The dense conditional distribution uses O(B*N) memory; hosts should supply
    bounded catalogs or candidate pools. A local seeded generator leaves the
    global random stream unchanged; use a different seed for each fresh draw.
    Seed is a Python integer in [0, 2**63-1] in both frameworks.
    Probabilities are normalized in float64 log space before the output cast.
    """

    def __init__(self, distribution='uniform', alpha=.75, uniform_mix=0.):
        super().__init__()
        if distribution not in ('uniform', 'popularity'):
            raise ValueError('distribution must be uniform or popularity')
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')
        if (isinstance(uniform_mix, bool) or not isinstance(uniform_mix, Real)
                or not math.isfinite(uniform_mix) or not 0 <= uniform_mix <= 1):
            raise ValueError('uniform_mix must be finite and in [0,1]')
        self.distribution, self.alpha, self.uniform_mix = distribution, alpha, uniform_mix

    def forward(self, counts, eligible_mask, positive_mask, num_samples, seed=0):
        if type(num_samples) is not int or num_samples < 1:
            raise ValueError('num_samples must be a positive integer')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63-1]')
        if counts.ndim != 1 or counts.numel() == 0:
            raise ValueError('counts must have shape [N] with N > 0')
        if (eligible_mask.ndim != 2 or positive_mask.shape != eligible_mask.shape
                or eligible_mask.shape[1] != counts.numel()):
            raise ValueError('both masks must have matching shape [B,N]')
        if eligible_mask.dtype != torch.bool or positive_mask.dtype != torch.bool:
            raise ValueError('masks must be boolean')
        if counts.is_complex() or counts.dtype == torch.bool:
            raise ValueError('counts must be real numeric values, not complex or boolean')
        output_dtype = torch.float64 if counts.dtype == torch.float64 else torch.float32
        counts = counts.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
            raise ValueError('counts must be finite and nonnegative')
        log_uniform = torch.full_like(counts, -math.log(counts.numel()))
        log_base, scale = log_uniform, 1.
        if self.distribution == 'popularity' and self.uniform_mix < 1:
            if not bool((counts > 0).any()):
                raise ValueError('popularity requires positive catalog count mass')
            log_counts = counts.log() if self.alpha > 0 else torch.zeros_like(counts)
            if self.uniform_mix == 0:
                # The catalog normalizer cancels: center within each row before
                # scaling, so even huge alpha cannot erase all eligible support.
                log_base, scale = log_counts, self.alpha if self.alpha > 0 else 1.
            else:
                logits = (log_counts - log_counts.max()) * self.alpha if self.alpha > 0 else log_counts
                log_popularity = logits - torch.logsumexp(logits, dim=0)
                log_base = torch.logaddexp(log_popularity + math.log1p(-self.uniform_mix),
                                          log_uniform + math.log(self.uniform_mix))
        logits = log_base[None, :].expand(eligible_mask.shape).masked_fill(
            ~(eligible_mask & ~positive_mask), -math.inf)
        maximum = logits.max(dim=-1, keepdim=True).values
        if not bool(torch.isfinite(maximum).all()):
            raise ValueError('each query needs positive eligible non-positive probability mass')
        logits = (logits - maximum) * scale
        probabilities = (logits - torch.logsumexp(logits, dim=-1, keepdim=True)).exp()
        generator = torch.Generator(device=counts.device).manual_seed(seed)
        item_ids = torch.multinomial(probabilities, num_samples, replacement=True, generator=generator)
        return {'item_ids': item_ids, 'sampling_probs': probabilities.gather(1, item_ids).to(output_dtype)}

    def sample(self, counts, eligible_mask, positive_mask, num_samples, seed=0):
        return self(counts, eligible_mask, positive_mask, num_samples=num_samples, seed=seed)
