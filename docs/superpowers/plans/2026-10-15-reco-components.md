# Reco-papers portable components

Extend the existing dual-framework mechanism-reference library, without owning production data construction or serving infrastructure. A source-paper inventory will account for all 115 PDF entries at Reco-papers commit 2c617a1659c38f60848c051b377f3faef5df831f. Existing coverage, newly implemented tensor cores, shared mechanisms, duplicates and excluded system/LLM papers remain distinct.

1. Implement and numerically test graph/KG tensor cores: GraphSAGE, PinSage, SDNE, LINE, RippleNet, KGAT, CDL. Graph construction and negative sampling remain explicit caller inputs.
2. Implement ranking/matching and sequence/slate cores: DeepCrossing, FM-initialized FNN, NeuMF, LatentCross, multi-view DSSM, predictive bilinear, TransAct, causal PinnerFormer encoder, Seq2Slate pointer decoder, DPP greedy selection, RankNet/LambdaRank loss.
3. Implement PAL probability factorization, attribute-conditioned MetaEmbedding, and DICE scoring/objective with caller-supplied masks. Declare training procedures that the components do not reproduce.
4. Connect each addition to method cards, structural/training patterns, framework manifests, family guides, reference/API retrieval and the source-paper inventory. Record pinned source URLs, license decisions and precise implementation boundaries.
5. Validate numeric formulas, masking, gradients, both framework APIs, inventory completeness and the full existing suite. Review the changes, fix failures, then publish an isolated GitHub branch and pull request.

Acceptance: every accepted new component has executable PyTorch and TensorFlow code and an agent-visible contract; every PDF has an explicit disposition; no unlicensed code is transplanted; no claims of benchmark replication, production uplift or TensorFlow 1.15 compatibility are made for the current Keras-based project.
