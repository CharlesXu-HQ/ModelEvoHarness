# Three training/reference gaps: source review and independent implementation

Baseline: ModelEvoHarness main `84b2f650645c849039b37bcd3dba564bdde1bbb0`.
The review concerns the three references selected after inspecting lmxhappy's
public repositories. It does not claim complete reproduction of their trainers
or datasets. No upstream model wrappers, notebooks or prediction arrays were copied.

## Findings and disposition

| Reference | Existing harness coverage | Non-overlapping addition |
| --- | --- | --- |
| Learnable Embedding Sizes / PEP | Pretrained ID fusion, role/vocabulary contracts and cross-network gates; no learned sparse support/retraining | Native PyTorch/TF soft-threshold embedding and fixed-mask rewind/retraining core |
| DSSM-Lookalike | Negative-sampling guidance and hard-score selection; no executable uniform/popularity conditional sampler | Native PyTorch/TF with-replacement proposals, explicit row exclusion and returned q |
| AUC/LogLoss notebook | Calibration guidance and representation diagnostics; no paired prediction analysis | Stdlib ID-paired weighted AUC/LogLoss and class/row loss decomposition in `analysis/` |

Implementation APIs and boundaries are in
[training-reference components](../training-reference-components.md).

## PEP: mechanism, not an upstream trainer

Sources:

- [Paper, equations 8–11 and algorithm 1](https://arxiv.org/html/2101.07577v2).
- [Author code](https://github.com/ssui-liu/learnable-embed-sizes-for-RecSys/tree/e4d07d2c5e4aaefa95b939b83600adfaf47d7a6f).
- [Inspected fork](https://github.com/lmxhappy/learnable-embed-sizes-for-RecSys/tree/9adfd6da0c4d487d935fa465f5e6d5f9f39395d7).

The formula subtracts a sigmoid threshold and creates exact zeros. Multiplying
an embedding by a sigmoid gate is not the same mechanism. After selecting a
mask, the local interface creates a new layer from original weights and fixes
the support. It retains the output width and dense storage; sparsity alone is
not a measured memory or latency saving.

Static crosschecks identified source-specific choices not reproduced here:

- The author's [embedding module](https://github.com/ssui-liu/learnable-embed-sizes-for-RecSys/blob/e4d07d2c5e4aaefa95b939b83600adfaf47d7a6f/models/pep_embedding.py)
  infers some field semantics from threshold shape. The independent component
  uses explicit sharing modes and has no field-offset inference.
- The [driver](https://github.com/ssui-liu/learnable-embed-sizes-for-RecSys/blob/e4d07d2c5e4aaefa95b939b83600adfaf47d7a6f/engine.py)
  uses test AUC in retraining stopping and saves an effective-table cache at
  its initial checkpoint location. The harness requires validation-only
  selection and a host-saved pristine initialization instead.
- Fixed-mask local retraining directly masks weights, so a retained cell at
  exactly zero can still receive gradients. Search has no monotone-sparsity or
  target-budget guarantee. Default threshold initialization is a finite starting
  point, not a reproduction of the author's tuned schedule.

## DSSM-Lookalike: explicit probability semantics

Pinned source: [07dd9442a8a721aa6db678c5dfc4aa65e5708e71](https://github.com/lmxhappy/DSSM-Lookalike/tree/07dd9442a8a721aa6db678c5dfc4aa65e5708e71).

In the inspected sampling code, the frequency mode uses inverse-square-root
counts, and weights produced in frequency order can be paired with a different
item iteration order. Its purported rank mode uses count values rather than
an explicit rank. Sampling is performed without replacement before positive
filtering, without replenishing removed draws. These observations concern that
source snapshot; they are not a verdict on the cited RALM method.

The independent sampler makes a different, explicit contract: counts align to
catalog rows, normalized count-power/uniform proposals are conditioned on
host-provided eligibility and positives, draws are with replacement, and each
draw returns its actual conditional probability. Empty support is an error.
No method is labeled a faithful RALM implementation. Existing hard-negative
helper docstrings now distinguish sampled unobserved items from logged
non-response; numerical selection behavior is unchanged.

## Prediction notebook: preserve the question, fix the measurement contract

Pinned source: [Notebook at a6aea3647affa9d952c5b975d0180c387c584c54](https://github.com/lmxhappy/why_loss_auc_unconsistent/blob/a6aea3647affa9d952c5b975d0180c387c584c54/why_loss_auc_unconsistent.ipynb).

The notebook compares earlier/later scores and positive/negative losses, but
does not define its referenced labels in the stored cells. Saved output
includes a non-finite later loss and endpoint-related warnings. Its custom
expression can evaluate `0 * log(0)`. Published comments are not substituted
for reproducible results.

The independent tool requires exact ordered sample IDs, binary labels,
probabilities, dataset/split identity and fixed optional weights. It computes
tie-aware weighted AUC from raw scores and finite LogLoss with explicit
clipping. It reports conditional class losses separately from contributions
to overall loss. Single-class AUC remains undefined; observations do not
diagnose a cause or establish significance, calibration failure or uplift.

## Integration and verification scope

The catalog registers two native model/training mechanisms and one shared
analysis tool, with capability requirements, controlled comparisons and
failure signals. The new prediction family contains packaged working knowledge.
Existing Agent paths provide APIs and complete local source before proposing
registered components. Analysis is explicitly invoked by the host and cannot
be used as a model component or count as an optimization result.

Local checks cover numerical mechanisms, gradients/masks, conditional sampling,
paired metrics, invalid inputs, source independence and Agent delivery. The
target GPU machine was shut down at the user's request. GPU kernels, mixed
precision and full-dataset experiments are outside this verification; no
optimization success rate or business-effect improvement is claimed.

### Local verification on 2026-11-14

Environment: Python 3.13.13, PyTorch 2.12.0, TensorFlow 2.21.0 and pytest 9.1.1;
CPU execution with CUDA disabled.

- Full regression: **1215 passed, 72 skipped, 396 subtests passed**. The suite
  emitted 3942 warnings; this is not a warning-free run. GPU-only tests were
  skipped on this host.
- `catalog-check`: exit 0; no missing guides, implementation files or card
  implementations, and no unmapped/unknown/duplicate source entries.
- Wheel and source distribution built successfully. Importing the built wheel,
  source
  retrieval for all three new methods in both frameworks returned complete
  files matching the working tree, including the shared analysis module.
- Independent review found two numerical boundary defects before publication:
  probability underflow could erase an eligible negative pool, and inconsistent
  upper clipping could reverse the apparent LogLoss change. Both received
  failing regression tests and fixes; the reviewer then independently reproduced
  correct conditional probabilities and finite, monotone boundary losses.
- The final review found no remaining blocking defects. This is component and
  integration verification, not a GPU training run or measured Agent success.
