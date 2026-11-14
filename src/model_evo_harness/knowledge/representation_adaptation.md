# Representation Adaptation

Do aligned pretrained representations improve supported cold or sparse slices under a fixed vocabulary contract?

## Research lineage

Original tensor implementations informed by the pinned FuxiCTR source inventory. No FuxiCTR model or training wrapper is imported.

## FuxiCTR mechanism components

These are independently implemented, composable PyTorch and TensorFlow mechanisms. Read the full local module and output contract before editing the current candidate. A paper name identifies a source of ideas; omitted towers, input adapters, samplers and objectives remain the host’s responsibility. Preserve the current compatible backbone and test one mechanism with its matched control. Multiple meaningful feature groups may use independent or shared instances; justify sharing and fusion from the actual fields. Do not force a complete named-model replacement or exhaust a fixed model list. Record mechanism evidence separately from task/dataset-bound business interpretations.

### `pretrained_id_fusion` — `PretrainedIDFusion`

Initialize, sum, or concatenate aligned pretrained ID vectors with a learned ID path and project to the declared output width.

**When:** A versioned aligned pretrained table and sparse/cold slice evidence exist. Test fusion and freezing independently with constant ID mapping.

**Input contract:** id_vocabulary, aligned_pretrained_vectors. See `model_api` for the framework constructor and calls. **Output:** `[B,...,output_dim]`.

**Controlled test:** Use the same vocabulary and pretrained table; isolate fusion mode and freezing, then compare cold/OOV and frequent-ID slices. Reject the explanation if its matched control accounts for the gain, the intended path has no gradients, or the supported target slice does not improve.

**Boundaries:** Host aligns IDs and validates vocabulary version, padding and OOV coverage. Inputs are native integer IDs. Freezing changes pretrained gradients only: it must not silently discard appended OOV vectors or switch the inference representation.

**Bundled code:** `models/pytorch/ctr_embeddings.py`, `models/tensorflow/ctr_embeddings.py`. [Research source](https://github.com/reczoo/FuxiCTR/blob/183da1c944ba8930fcdcd54114837c081959d748/fuxictr/pytorch/layers/embeddings/pretrained_embedding.py).

## Learned embedding sparsity and fixed-mask retraining

When observed table capacity or sparse-ID overfitting motivates a capacity experiment, `prunable_embedding` provides native `PrunableEmbedding` references. The learned representation is `sign(V) * relu(abs(V) - sigmoid(s))`. Thresholds may be global, per dimension, per vocabulary row or per cell. A thresholded zero cell is a parameter-selection observation, not proof that the associated input feature has no value.

Save the original embedding initialization. Train thresholds on training data, export a mask and use `from_mask(initial_weights, mask, padding_idx=...)` to create a separate retraining layer. Fix the mask while retraining; reset optimizer state and handle the rest of the backbone consistently. Compare against dense and matched random-mask controls with the same total search/retrain budget, splits and objective. Select sparsity on training/validation and keep the final holdout sealed. Check rare/OOV slices, protected padding and loss stability.

The implementation retains dense storage and the same output width. `nonzero_count()` measures active cells; physical sparse storage, field-dimension packing, memory savings and latency require a separate deployment implementation and measurements. A sigmoid threshold bounded by one is the specific PEP-inspired parameterization, not a general guarantee for every initialization scale. There is no automatic budget search or pruning schedule. Source: [Learnable Embedding Sizes for Recommender Systems](https://arxiv.org/abs/2101.07577).
