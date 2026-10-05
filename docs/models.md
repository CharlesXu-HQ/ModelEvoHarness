# Direct-framework model references

ModelEvoHarness includes an original implementation of **each of its 44 method cards in both PyTorch and TensorFlow**, plus a generic `TwoTower` example in both frameworks. The [implementation manifest](../src/model_evo_harness/data/model_implementations.json) maps every method ID to its class and file, `reference_scope`, `training_support`, `output_contract`, and `limitations`. These contracts reach the Agent through `model_api` and source retrieval. These are compact mechanism references for controlled experiments, not exact reproductions of every layer, loss, data pipeline or serving system in the original papers.

The frameworks are isolated under [`models/pytorch`](../src/model_evo_harness/models/pytorch/) and [`models/tensorflow`](../src/model_evo_harness/models/tensorflow/). Each module imports only its own deep-learning framework and Python's standard library; there is no DeepCTR, Torch-RecHub, RecBole or other model-wrapper dependency. Importing `model_evo_harness` itself does not import PyTorch or TensorFlow. Install the framework you intend to use separately.

## Find a model

The same method IDs have corresponding classes in both framework directories. Import from the named submodule; the manifest is the exact method-ID-to-symbol contract.

| Module in each framework | Method-card IDs | Input or modeling focus |
| --- | --- | --- |
| [`architectures.py`](../src/model_evo_harness/models/pytorch/architectures.py) | `fm`, `deepfm`, `dcn`, `din`, `mmoe` | Baseline interaction, behavior-attention and multi-task cores. Also contains generic `two_tower`, which is **not** a method card. |
| [`interactions.py`](../src/model_evo_harness/models/pytorch/interactions.py) | `afm`, `autoint`, `dcn_v2`, `fibinet`, `fwfm`, `nfm`, `pnn`, `wide_deep`, `xdeepfm` | Categorical pair or higher-order interactions; DCNv2 takes an already encoded dense vector. |
| [`retrieval.py`](../src/model_evo_harness/models/pytorch/retrieval.py) | `biassvd`, `funksvd`, `item_cf`, `user_cf`, `swing`, `eges`, `item2vec`, `dssm`, `fm_recall`, `youtubednn`, `youtube_sbc` | Neighbor retrieval, ID/side embeddings and independently encodable matching. |
| [`sequence.py`](../src/model_evo_harness/models/pytorch/sequence.py) | `mind`, `sasrec`, `sdm`, `narm`, `hstu`, `dien`, `dsin` | Interest vectors, ordered histories, session intent and target-aware behavior ranking. |
| [`multitask.py`](../src/model_evo_harness/models/pytorch/multitask.py) | `shared_bottom`, `esmm`, `ple`, `aitm`, `m2m`, `apg`, `hmoe`, `pepnet`, `star`, `prm`, `prs` | Multiple outcomes, scenario adaptation and slate interaction. |
| [`segmentation.py`](../src/model_evo_harness/models/pytorch/segmentation.py) | `mlr` | Soft mixture of local response functions. |
| [`training.py`](../src/model_evo_harness/models/pytorch/training.py) | Training helpers, not method cards | `focal_loss`, `bpr_loss`, `listwise_loss`, `hard_negative_indices`. |

The TensorFlow files use the same names under [`models/tensorflow`](../src/model_evo_harness/models/tensorflow/). Some multi-input Keras models accept multiple tensors through their `call` signature; generic Keras `TwoTower` takes `(user_features, item_features)` as one tuple, while PyTorch `TwoTower` takes two positional tensors.

## Output contract: choose the loss accordingly

A returned tensor is **not always a probability**. Check the model's docstring and the host's target before selecting a loss or threshold.

| Return type | Implementations | Host responsibility |
| --- | --- | --- |
| Raw binary logits `[B]` | FM/DeepFM/DCN and the feature-interaction classes; DIN/DIEN/DSIN; APG/HMoE/STAR | Apply a logits-aware binary loss and calibrate before using expected value. |
| Raw task logits `[B,T]` | MMoE, SharedBottom, PLE, AITM, M2M, PEPNet | Supply actual labels and missingness masks for every observed task; define primary metric and loss weights. |
| Pair or candidate scores | FunkSVD, BiasSVD, EGES, Item2Vec, DSSM, FMRecall, YouTubeDNN, YouTubeSBC, generic TwoTower | Define positive/negative provenance, candidate universe, sampling correction and ranking loss. DSSM is a cosine similarity; YouTubeSBC applies only the supplied item-sampling correction. |
| Neighbor score matrices | ItemCF, UserCF, Swing (`fit` then `scores`) | Provide positive interaction matrices, eligibility filtering and a production-scale neighbor/index strategy. These are direct tensor algorithms, not gradient-trained networks. |
| Interest or sequence features | MIND (`[B,K,D]`), SASRec/SDM/NARM (`[B,D]`), HSTU (`[B,T,D]`) | Add a task-specific item scorer or head, legal time cutoff, item catalog and loss. MIND also exposes `score(interests, items)`, `label_aware_weights(interests, target)` and `training_readout(history, mask, target)`; the latter two use the target only during supervised training. |
| Probabilities or distributions | ESMM (`click`, `post_click`, `joint`); MLR (`[B]`); PRM position softmax (`[B,S]`); PRS `ctr` and `continuation` (`[B,S]`) | Use probability-aware objectives and the correct denominators. PRM/PRS need logged slates and a valid slate evaluator; PRS `rerank` is a bounded beam search. |

For example, a binary interaction experiment in PyTorch can train on logits:

```python
from model_evo_harness.models.pytorch.interactions import FwFM
from model_evo_harness.models.pytorch.training import focal_loss

model = FwFM(cardinalities=[1000, 500], embedding_dim=16)
logits = model(field_ids)  # integer IDs [batch, 2]
loss = focal_loss(logits, observed_labels)
loss.backward()
```

The TensorFlow counterpart uses `model_evo_harness.models.tensorflow.interactions.FwFM` and `models.tensorflow.training.focal_loss` inside a `tf.GradientTape`. For a retrieval experiment, the host can select eligible difficult items while keeping the scoring model and candidate universe fixed:

```python
from model_evo_harness.models.pytorch.training import bpr_loss, hard_negative_indices

negative_ids = hard_negative_indices(query_vectors, item_vectors,
                                     allowed_negative_mask, k=8)
scores = query_vectors @ item_vectors.T
positive = scores.gather(1, positive_item_ids[:, None])
negative = scores.gather(1, negative_ids)
loss = bpr_loss(positive, negative)
```

The host constructs `allowed_negative_mask`, excludes positives, unexposed or future items as required by its feedback contract, and ensures at least `k` eligible negatives per query. TensorFlow exposes the same helper names in `models.tensorflow.training`. These functions select examples or compute a loss; they do not infer exposure probabilities or generate uplift labels. The [training objective](../src/model_evo_harness/knowledge/training_objectives.md) and [sampling](../src/model_evo_harness/knowledge/sampling_and_hard_examples.md) guides describe when such experiments are justified.

## Scope and failure boundaries

- All models assume the host has already defined labels, time-safe features, train/validation/test splits, candidate eligibility, negative sampling and final evaluation. A reference model is not a trainer, experiment adapter, ANN index or coupon policy.
- Categorical interaction examples use one integer ID per field; multi-hot, dense numerical and missing-value preprocessing belong to the host. DCN/DCNv2 and many scenario models take preprocessed dense inputs. Sequence models take pre-decision embeddings and explicit masks or timestamps. NARM and SDM require each recurrent history mask to be a contiguous prefix of valid events followed by padding; empty rows return zero vectors.
- ItemCF, UserCF and Swing use dense interaction/similarity calculations suitable for small studies; deployment needs pruning and scalable storage. Retrieval scores are not calibrated probabilities.
- The interaction implementations retain the defining core but omit some paper variants: FiBiNET uses a shared bilinear transform, XDeepFM omits split-half CIN routing, and DCNv2 includes a matrix or low-rank cross without expert mixture. Wide & Deep implements explicit pair crosses, not arbitrary feature-cross templates.
- Sequence and multi-task references isolate their main computational idea. SDM accepts `short_history, short_mask` and optional `long_history, long_mask, user_context`; with only two arguments it reuses the short history as long history and derives user context from its masked mean. Its short branch uses an LSTM and self-attention before user attention, then gates it with the long branch. DIEN omits the auxiliary next-behavior training objective; HSTU is a sequence block, not an end-to-end generative recommender; ordered-funnel methods require real ordered labels. PRS beam search is not a global optimum or an unbiased slate evaluator.
- MLR's soft regions are predictive segments, not causal treatment effects. No architecture here establishes coupon uplift or net-value improvement without the host's randomized evaluation and cost definition.

Every method has a [local technical guide](knowledge/README.md) explaining its data prerequisites, controlled comparison and rejection signals. The [source attribution record](research/source-attribution.md) lists pinned materials that informed those original explanations and implementations; upstream prose and code were not copied into this Apache-2.0 repository. No task-level performance improvement is claimed by having these implementations available.
