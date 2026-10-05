# Diffusion augmentation and recommendation

Diffusion methods learn to denoise corrupted representations or generate candidate behavior trajectories. They may augment sparse histories or model multimodal item distributions. The synthetic output is a training or retrieval hypothesis, not a substitute for observed counterfactual outcomes. In treatment-effect work, fabricated outcomes must never enter an evaluation set as if randomized.

## Data and conditions

Require a well-defined input representation, corruption/noise process, conditioning fields, catalog mapping, and an observed target distribution. Record synthetic-to-real ratio, random seed, and which split produced generator training. Generator training must not see validation or test outcomes.

## Controlled experiment

Compare no augmentation, a simple perturbation/oversampling control, and diffusion augmentation with the same downstream model and training budget. Evaluate on untouched real held-out data, including minority and cold slices. Report output validity, mode collapse, calibration, training time, and inference cost.

## Interpretation and feature needs

A lower denoising loss does not establish better recommendation value. Apparent gains only on synthetic data are non-evidence. If domain knowledge says a needed attribute is unobserved, generation cannot recover its true value per user; request the field. Otherwise, first test whether existing signals are underused by simpler augmentation.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [docs/chapter_9_diffusion/1.basics.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_9_diffusion/1.basics.html), [docs/chapter_9_diffusion/2.augmentation.html](https://github.com/datawhalechina/fun-rec/blob/5f7bd84d4403b5f92b6bedbd31e3984ef00a4572/docs/chapter_9_diffusion/2.augmentation.html).
