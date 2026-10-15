# Third-party notices

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

Other new components are independently implemented from paper mechanisms. Consulted GitHub references, pinned commits, license findings and no-copy decisions are recorded on the corresponding method cards. In particular, GPL CDL code and repositories without a verified redistribution license were not copied. No pretrained weights or paper PDFs are redistributed.
