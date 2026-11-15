# Native CTR mechanism refinements

## Intent and approved scope

Implement the DeepCTR/DeepCTR-Torch audit recommendations as original PyTorch and
TensorFlow tensor components, usable inside an existing backbone. The user has
authorized implementation with engineering choices delegated to the implementer.
Keep the package dependency-free; frameworks remain optional host dependencies.
CouponEvo and business-strategy guidance are outside this change.

## Approach

Extend existing components for small parameterization changes and add focused
modules for reusable cross, field-weighting, attention and pooling mechanisms.
This preserves existing defaults and avoids duplicate full-model wrappers.
Importing upstream model libraries would violate the user's dependency constraint;
copying entire upstream trainers would also duplicate the host's responsibilities.

## Contracts and behavior

- DIEN accepts contiguous-prefix valid histories, consistently validated in both
  frameworks. DIN accepts arbitrary Boolean masks. Invalid padding is sanitized
  before arithmetic; nonfinite valid values are rejected. Empty rows stay finite.
- Sequence role metadata is optional for backwards compatibility; when supplied,
  validate lengths/masks, event alignment, pooling and encoding identity explicitly.
- CrossNetMix adds low-rank nonlinear experts and per-example routing. IFM and DIFM
  generate field factors shared by the linear and FM terms; their normalization
  rules remain distinct.
- ONN vector products, FiBiNET sharing and symmetric field matrices are opt-in
  variants of existing components. Dice is an activation, distinct from DICE
  interest/conformity disentanglement.
- Expose DIEN states and wire the existing next-event objective through a public
  method, preserving its reduction. Add evolution variants, DSIN position/session
  bias and the history-only BST readout as explicit options.
- Keep train/eval statistics, auxiliary loss scaling and regularization conventions
  explicit. Inputs and labels must precede the host's decision cutoff.
- Register original guides, source contracts and controlled comparisons so Agent
  discovery and source delivery include the additions. No accuracy claims follow
  from availability or unit-test results.

## Acceptance

Analytic formula and gradient tests; padding/empty-history invariance; paired
framework checks with mapped weights; no third-party DL imports; catalog/provider
source-delivery tests; full local regression suite; targeted GPU checks on the
existing RTX 5090 host. Record failures and verification scope. Publish a focused
PR with the implementation, documentation and evidence.
