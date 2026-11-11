# Discount-aware response inside a marketing backbone

## Question and input evidence

Use this direction when the current response model has measured residuals related
to historical discounts, purchase price/category or promotion context. Keep the
interest backbone and the task's evaluation protocol while testing a local
change. A method becoming `ready` means its inputs were declared, not that it is
the best next experiment. Compare the expected information and cost with simpler
feature interactions, time-domain pooling or an objective-only change.

The two native components are `danet_discount_preference`
(`DiscountPreferenceBranch`) and `danet_conversion_head`
(`DiscountAwareConversionHead`). Read their entire `discount_aware.py` module
through `read_references` before composing code. They belong to `model_api`;
representation probes and feature screening remain in `analysis_api`.

The branch requires a regular calendar of historical item discount ratios,
aligned purchase item embeddings and discount/category/price attribute vectors,
a target item embedding, promotion context and validity masks. Every observation
must precede the prediction cutoff. Historical calendar gaps stay at their real
positions; compressing the sequence changes frequency meaning. The implementation
zero-fills missing slots only for numerical filtering and masks them afterward.
Do not interpret these zeros as observed no-discount days. Check sensitivity to
coverage and, where supported, compare a missingness-aware time-domain control.

## Local mechanism and declared choices

The branch filters the supplied calendar with rFFT, a real/imaginary packed MLP
and an explicit-length inverse. It returns `filtered_history` and the residual
relative to sanitized input. A learned filter is not guaranteed to be low-pass.
An independent target-conditioned FFN attention pools purchase attributes; a
projected user preference is modulated by a sigmoid promotion-context gate.
The representation order is preference, residual history, filtered history.
The host can pass this representation into its current backbone's prediction
path, instantiate compatible branches for distinct semantic input groups, or
ablate one component. Repeated branches still require evidence of complementarity,
sharing/fusion choices and resource measurements.

The head accepts existing interest and discount representations. Its conversion
logit uses both; its bounded discount prediction uses the discount representation.
Its `objective` adds mean conversion BCE to a weighted auxiliary MSE over
purchased rows that have valid realized-discount labels. The denominator is the
count of those labels, and a batch without them contributes differentiable zero.
Labels are detached and ignored rows are sanitized before arithmetic. The
`purchase_discount_auxiliary` training pattern requires label availability
separately from the structural components. Use alpha=0 as the matched objective
control; absence of auxiliary labels does not prevent a conversion-only branch
experiment when its forward inputs are available.

## Experiment and rejection

Choose the intervention from observed evidence, not a mandatory sequence of
named models. Hold the parent interest representation, data, optimizer budget
and evaluator fixed while testing the branch; isolate auxiliary supervision
from forward changes. Compare a targeted history, user-preference or context
ablation with the full candidate before attributing a gain. Track positive-label
coverage, calibration, outcome slices and compute. Do not optimize auxiliary
MSE at the expense of the host's fixed business metric.

Current-order realized discount belongs only in the training target. Planned
offers known before decision are different features. Declare whether ratios
mean reduction proportion or paid/original proportion, use [0,1] consistently,
and never substitute coupon face value or assumed costs for realized labels.
Response accuracy and the discount head do not establish uplift or profitability.
For randomized coupon work, retain treatment/control support, held-out policy
comparisons, uncertainty and the host's cost/budget contract. Any adaptation of
the response objective to a causal estimator must be justified separately.

If specific histories are absent, mark this mechanism unavailable and consider
other supported local changes. Follow the existing feature-gap contract before
asking humans for new data; an expert-obvious input requirement may be documented
immediately, but unavailable DANet inputs alone do not prove the whole task needs
new features. Keep technical and business lessons bound to the task, dataset
version and protocol.

## Research lineage

These are independently written mechanism references informed by
[DANet](https://arxiv.org/html/2607.12578v1) and checked against its
[partial author implementation](https://github.com/tangrc/DANet). The paper and
public fragment disagree on FFT length and auxiliary-loss details. Our explicit
length, packed complex representation, single FFN softmax, sigmoid gate and
positive-count reduction are documented engineering choices. The host supplies
IPN/embeddings, data preparation, training and serving. There is no production
system reproduction, new uplift estimator or measured business-gain claim.
