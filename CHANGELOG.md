# Changelog

## 0.3.0

### Added

- Native PyTorch/TensorFlow CAN feature-pair co-action and ComiRec-SA/DR interest components, with complete Agent-readable source and explicit variant contracts.
- Configurable CIN identity/ReLU activation and optional SDIM per-hash L2 normalization; existing defaults retained.
- Framework-neutral COLD-inspired feature-subset/retraining workflow with protected groups, comparable host evidence, latency constraints and quality thresholds.
- Source crosschecks, independent formula/gradient tests and GPU device assertions.
- The previously prepared FuxiCTR mechanism coverage: 43 method additions and a 57-entry source inventory.

### Fixed

- TensorFlow DeepCrossing now enforces its declared feature width on the first call.
- The new SDIM normalization option keeps zero/tiny-bucket FP16 gradients finite with a shared norm floor.
- Agent documentation distinguishes paper structures from compact variants and recognizes bundled ComiRec source.
- Experiment journals record the published package version; a regression compares the generated identity with release metadata.

### Scope

- 131 cards: 128 model/training methods and 3 analysis tools. Both frameworks can read all methods; cost search is a single shared stdlib implementation.
- This release provides reusable mechanisms and protocols, not reproduced paper scores or measured coupon/uplift improvement. Host adapters own datasets, GPU training and independent evaluation.
