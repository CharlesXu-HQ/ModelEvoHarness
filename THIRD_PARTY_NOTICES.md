# Third-party notices

The PEP-inspired embedding pruning, conditional negative sampling and paired
prediction diagnostics are independent implementations. Their pinned research
sources and deliberate differences are documented in
`docs/research/training-reference-gap-review-2026-11-14.md`. No torchfm/DeepCTR
wrappers, upstream trainer code, notebooks or data are incorporated in these
components. The independent additions use this project's Apache-2.0 license.

The DICE interest/conformity masked objective in both `models/*/debias.py` is adapted from [FIB LAB DICE](https://github.com/tsinghua-fib-lab/DICE/blob/591b883cd62bf0d88c906d12ed2ab88ff8fccc36/src/model.py). Changes include stable softplus, explicit tensor contracts, a TensorFlow port and an opt-in L2 discrepancy. Its MIT notice is also embedded in each source module so Agent source extraction preserves it.

MIT License

Copyright (c) 2022 FIB LAB, Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Other components, apart from the explicitly attributed adaptations in this document, are independently implemented from paper mechanisms. Consulted GitHub references, pinned commits, license findings and no-copy decisions are recorded on the corresponding method cards. In particular, GPL CDL code and repositories without a verified redistribution license were not copied. No pretrained weights or paper PDFs are redistributed.

## UniDot tensor components

The conditional SwiGLU, multi-channel sequence pool and interaction Highway in `models/*/paper_unified_sequence.py` and `models/*/paper_interactions.py` adapt [linrongc/kdd26_unidot](https://github.com/linrongc/kdd26_unidot/tree/1399702006a1fc2f86329c8c6133f3f3ac1e6d0d). Changes include standalone tensor interfaces, TensorFlow ports, True=valid masks, sanitized padding, zero empty-domain outputs, stable half-type L2 normalization with zero-direction gradients, and removal of temporal variants and training/data dependencies. Each extractable module includes the following notice. RankMixer in the same interaction modules is independently implemented from its paper.

MIT License

Copyright (c) 2026 Rongcheng Lin and the UniDot authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## FuxiCTR-informed mechanisms

The new `ctr_*`, `field_*`, and `history_*` modules are original tensor implementations
informed by [FuxiCTR](https://github.com/reczoo/FuxiCTR/tree/183da1c944ba8930fcdcd54114837c081959d748)
and the papers it identifies. FuxiCTR wrappers, feature processors, training infrastructure,
weights and datasets are not redistributed. The pinned source inventory, code-level
differences and boundaries are recorded in `docs/fuxictr-coverage.md`, the method cards
and packaged original guides. Source names provide attribution and do not imply endorsement
or full paper reproduction. The new independent code is covered by this project's Apache-2.0 license.
