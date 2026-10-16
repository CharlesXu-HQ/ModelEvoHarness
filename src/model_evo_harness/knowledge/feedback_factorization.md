# Logged-feedback factorization

Position and item popularity affect what users observe and click. This family separates supported components of a prediction score without claiming that arbitrary observational logs identify causal user preferences. It contains PAL and DICE, with explicit data requirements and different mathematical assumptions. Neither method manufactures exposure labels or reconstructs the production training set.

## PAL

The PAL core multiplies a position-only examination probability by a position-free relevance probability. The host supplies relevance logits from its own backbone and logged positions for the joint binary cross-entropy objective. The implementation evaluates that probability product through a stable joint logit. For inference, use the relevance operation without assigning a fictitious position. The relevance tower must not receive position through an indirect feature path. Separate factors are not necessarily identifiable from biased logs, so calibration and held-out position slices matter.

## DICE

DICE uses separate user/item interest and conformity factors. Its objective accepts positive/negative pairs and a boolean mask indicating that the negative item is more popular. That mask changes the direction of the conformity comparison; it is not a propensity score. The host must explain its popularity cutoff and negative sampling. Optional negative L2 discrepancy encourages factor separation but is unbounded without regularization; the component therefore disables it by default. Curriculum, causal sample construction and distance-correlation variants are not implemented.

## Controlled experiment

Keep impressions or pairs, the time split and evaluation universe fixed. Compare PAL with a position-free relevance control; compare DICE total scores with ordinary factor scores, then ablate the masked auxiliary terms. Monitor position/popularity slices and calibration. Reject improvements that depend on unsupported labels, future popularity counts, unstable factor norms or changed exposure policy.

## Research lineage

PAL follows its probability factorization and joint loss equations. DICE's masked objective follows the pinned MIT author reference, with its copyright notice preserved in extracted source. Exact paper and repository links are available in the method cards and source inventory.
