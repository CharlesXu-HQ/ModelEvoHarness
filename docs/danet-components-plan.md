# DANet components implementation plan

> **For agentic workers:** Use the existing implementation and test workflow; keep each framework's source independent.

**Goal:** Give the Agent composable discount-preference structure and a purchase-only auxiliary objective in native PyTorch and TensorFlow.

**Architecture:** Keep the host's interest backbone. A discount branch produces a representation from pre-decision item discount histories, aligned purchase attributes and promotion context. A dual head combines the existing interest representation with this branch and exposes a stable conversion-plus-discount objective. Both are model components under `models/`; neither is an analysis tool.

**Tech stack:** Python, PyTorch, TensorFlow; native tensor APIs and stdlib only.

**Spec:** The contract below implements the user's request to add DANet's discount preference branch and auxiliary task. Sources: [paper](https://arxiv.org/html/2607.12578), [author reference](https://github.com/tangrc/DANet). Code is independently written.

## Design and scope

A complete data-dependent IPN/trainer would duplicate the host backbone and require unavailable production inputs. Composable DPN and head references fit the existing harness and support local ablation. No new runner, dataset, uplift estimator or analysis component is added.

Paper/code ambiguities are explicit: the paper describes a 400-day history and elsewhere a 32-point FFT, while the public code reshapes to 97. Filtering details and gate normalization are underspecified; the public loss differs from the paper's positive-only objective. Use the mechanism and output contract below, without claiming exact reproduction.

### Shared API

`DiscountPreferenceBranch(history_length, item_dim, attribute_dim, context_dim, hidden_dim)`

- Call `(item_discount_history, history_mask, target_item, purchase_items, purchase_attributes, purchase_mask, promotion_context)`.
- Shapes: `[B,N]`, bool `[B,N]`, `[B,D]`, `[B,L,D]`, `[B,L,A]`, bool `[B,L]`, `[B,C]`. All dimensions constructor-positive, `N=history_length`, and `L` may be zero.
- Attributes encode the historical discount, category and price together; the host owns their semantics/embedding, regular time grid and pre-decision cutoff. No outcome label enters forward inputs.
- Mask invalid history/item/attribute values before arithmetic, including NaN padding. Reject nonfinite observed inputs and incompatible shapes/masks. Historical discount ratios are normalized to `[0,1]`; the host must use one declared ratio convention consistently.
- Apply rFFT over all N slots; concatenate real and imaginary parts, use Dense(2K,H)→ReLU→Dense(H,2K), then irFFT with explicit length N. This is a learned filter, with no guaranteed low-pass property. Mask its output; residual is sanitized input minus filtered output. No silent truncation to 32 or 97.
- Target attention uses concatenated purchase-item/target tokens, Dense(2D,H)→ReLU→Dense(H,1), masked softmax, then pools aligned purchase attributes. Project attributes with Dense(A,H)→ReLU, suppress empty-purchase output, and multiply by sigmoid(Dense(C,H)). Softmax and sigmoid are explicit engineering choices.
- Return a dict: `representation=[user_preference; residual_history; filtered_history]` with width `H+2N`, plus `user_preference`, `filtered_history`, `residual_history`, `attention`. Empty purchase/history signals yield zero for their respective components.

`DiscountAwareConversionHead(interest_dim, discount_dim, hidden_dim)`

- Call `(interest_representation[B,I], discount_representation[B,P])`.
- Independent heads: concat(interest,discount)→Dense(I+P,H)→ReLU→Dense(H,1) produces `conversion_logit[B]`; discount→Dense(P,H)→ReLU→Dense(H,1)→sigmoid produces `discount_prediction[B]`.
- `objective(outputs, conversion_labels, discount_labels, discount_label_mask, alpha=1.0)` returns scalar `loss`, `conversion_loss`, `discount_loss` and integer `discount_count`.
- Conversion loss is mean BCE with logits, over a nonempty batch with binary labels. Auxiliary MSE uses only purchased rows with valid discount labels, divided by their count. A no-valid-positive batch contributes differentiable zero. Mask unavailable/negative-row labels before arithmetic, so irrelevant NaNs are harmless. Valid labels lie in `[0,1]`; alpha is a finite nonnegative scalar. Labels never affect forward prediction.

## Global constraints

- Two self-contained `models/{pytorch,tensorflow}/discount_aware.py` modules; no third-party model wrappers or repository imports.
- Keep masks bool (`True=valid`); support gradient flow through the intended branch, and sanitize padding before tensor operations.
- Preserve the existing analysis/model separation. Host owns capture, tokenizer, training, evaluation and device placement.
- The predicted response/discount is not an individual treatment effect. Randomized policy evaluation remains necessary for coupon incrementality.

## Work and verification

1. Framework workers own their single native module and framework-specific tests. Write failing tests first; implement both classes and objective. Check masks, empty histories, target/context sensitivity, label gating and finite gradients.
2. Integrator owns method/family/structure/training registries, guide, bilingual README and docs. Add `discount_response`, `danet_discount_preference`, `danet_conversion_head`, and the auxiliary training pattern. Verify capability gating and actual source delivery.
3. Integrator checks independently assigned weights across frameworks, odd FFT lengths, tf.function, standalone extraction and detached-label safety. Use analytic tensor fixtures, not a dataset training claim.
4. Run targeted and full tests, catalog coverage and diff checks; independent review; push the existing PR with exact evidence and limits.

## Review focus

- Missing calendar slots are missing observations, never automatically zero-discount business facts; FFT on zero-filled slots is an explicit simplification.
- Negative rows and unavailable discount labels must have zero auxiliary gradient; no-positive batches must be finite.
- Positive labels are used only by the objective; pre-decision feature timing remains host-owned and must be exposed to the Agent.
- Empty/all-masked purchase sequences, masked NaNs and odd-length FFTs must preserve defined shapes and finite gradients.
- Filtered signal is not guaranteed to be low-frequency; no public dataset or business lift is claimed from tensor tests.

## Completion evidence

Implementation, catalog integration and independent review are complete. The
[validation record](research/danet-components-validation-2026-11-12.md) records
469 passing tests, 343 passing subtests and 21 GPU-only skips, along with the
remaining experiment boundaries.
