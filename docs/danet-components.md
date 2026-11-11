# Discount-aware marketing components

The [DANet paper](https://arxiv.org/html/2607.12578v1) motivates a discount branch
and an auxiliary regression task for conversion prediction. ModelEvoHarness
provides two independently written, composable references in native PyTorch and
TensorFlow. They plug into the host's existing interest backbone and live under
`models/`; the separate `analysis/` directory is unchanged.

| Method | Class | Contract |
| --- | --- | --- |
| `danet_discount_preference` | `DiscountPreferenceBranch` | Calendar discount filtering, target-conditioned purchase attribute pooling and promotion-context modulation. Returns a representation of width `hidden_dim + 2 * history_length`. |
| `danet_conversion_head` | `DiscountAwareConversionHead` | Conversion logit from interest plus discount representation; bounded discount prediction from the latter; optional purchase-only auxiliary objective. |

The Agent receives a [family guide](../src/model_evo_harness/knowledge/discount_response.md),
method prerequisites, a structural comparison pattern and the
`purchase_discount_auxiliary` training pattern. Branch eligibility and auxiliary
label eligibility are separate. `model_api` exposes both classes and `objective`;
`read_references` supplies the complete framework module and contracts.

## Compose with an existing backbone

```python
from model_evo_harness.models.pytorch.discount_aware import (
    DiscountPreferenceBranch, DiscountAwareConversionHead,
)

branch = DiscountPreferenceBranch(
    history_length=N, item_dim=D, attribute_dim=A,
    context_dim=C, hidden_dim=H,
)
head = DiscountAwareConversionHead(
    interest_dim=I, discount_dim=H + 2 * N, hidden_dim=H,
)
# Host owns device placement, embeddings, captures, batches and optimizer.
discount = branch(
    item_discount_history, history_mask, target_item,
    purchase_items, purchase_attributes, purchase_mask, promotion_context,
)
outputs = head(interest_representation, discount["representation"])
losses = head.objective(
    outputs, conversion_labels, discount_labels, discount_label_mask,
    alpha=auxiliary_weight,
)
losses["loss"].backward()
```

The TensorFlow classes use the same names and argument order under
`model_evo_harness.models.tensorflow.discount_aware`. Use `tf.GradientTape` and
host-managed optimization. No framework wrapper, tokenizer or trainer is imported.
PyTorch parameters use float32 by default or explicit float64; keep them at those
dtypes when using host autocast. FFT work remains float32/float64. Converting the
entire PyTorch module to half/bfloat16 is unsupported. TensorFlow follows its
Keras compute policy and promotes lower-precision FFT work to float32.

## Inputs, timing and missingness

| Input | Shape | Meaning |
| --- | --- | --- |
| `item_discount_history`, `history_mask` | `[B,N]` | Regular pre-decision item calendar, normalized ratio and boolean availability. |
| `target_item` | `[B,D]` | Current candidate's embedding. |
| `purchase_items` | `[B,L,D]` | Historical purchased-item embeddings. |
| `purchase_attributes`, `purchase_mask` | `[B,L,A]`, `[B,L]` | Event-aligned historical discount/category/price embeddings and validity. |
| `promotion_context` | `[B,C]` | Promotion features available at prediction time. |
| `interest_representation` | `[B,I]` | Existing host backbone output. |
| `conversion_labels`, `discount_labels`, `discount_label_mask` | `[B]` | Training-only binary purchase labels and actual realized ratios with explicit availability. |

Mask dtype is bool, with True meaning valid. Invalid padding is sanitized before
operations, including NaNs. All-masked histories/purchases contribute zero from
that component; `L=0` is supported. Missing calendar slots remain in place, are
zero-filled for the filter and remasked afterward. This numerical convention
must not be treated as a business observation of zero discount. The host owns
cutoff checks and ratio semantics; current-order realized labels are never
forward inputs.

## Auxiliary objective

Let `m = (conversion_labels == 1) & discount_label_mask`. The implementation uses

```text
conversion_loss = mean(BCEWithLogits(conversion_logit, conversion_labels))
discount_loss = sum(m * (discount_prediction - discount_labels)^2) / max(1, sum(m))
loss = conversion_loss + alpha * discount_loss
```

The implementation clears ignored labels before subtraction; it does not rely
on NaN multiplied by zero. Only valid positive-row discount labels receive an
auxiliary gradient path to predictions. Target tensors are detached. A batch
without valid positives has a differentiable zero auxiliary term. It also
returns `discount_count` so hosts can report supervision coverage. Alpha must be
finite and nonnegative; alpha=0 isolates conversion-only training.

## Fidelity and independent choices

This is a mechanism reference, not a complete reproduction. The paper mentions
400 daily slots and a 32-point FFT; its [public fragment](https://github.com/tangrc/DANet/blob/main/cvr_model_DANet_github/model/model.py)
reshapes to 97 and drops the spectrum's imaginary part. The released training
loss also differs from the paper's positive-only objective. We preserve complex
information through packed real/imaginary layers, use the declared full history
length, and expose filtered/residual outputs without claiming guaranteed low-pass
behavior. Single FFN masked-softmax attention, sigmoid context modulation and
positive-count normalization are explicit implementation choices. No source
code was copied from the author fragment, which depends on a private framework.

DANet is a response-model reference. For coupon incrementality, retain the
host's causal estimator and policy evaluation; conversion AUC or auxiliary MSE
alone cannot establish incremental revenue or net return. The modules add no
cost estimator, uplift labels or new feature-collection pipeline.
