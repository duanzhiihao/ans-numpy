# Python interface for asymmetric numeral systems (ANS)

This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [ryg_rans](https://github.com/rygorous/ryg_rans) with minor modifications to make ANS easier to use for PyTorch models.

- The C++ function accepts NumPy inputs (can be easily converted from/to PyTorch tensors)
- Supports [CompressAI](https://github.com/InterDigitalInc/CompressAI) models with minimal changes
- Faster than the original CompressAI implementation (see `test-compressai.ipynb`)
