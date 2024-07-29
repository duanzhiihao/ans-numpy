# Python interface for asymmetric numeral systems (ANS)

This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [ryg_rans](https://github.com/rygorous/ryg_rans) with minor modifications to make ANS easier to use for PyTorch models.

- The C++ function accepts NumPy inputs (instead of Python lists)
- Supports CompressAI models with minimal changes
- Faster than the `encode_with_indexes()` and `decode_with_indexes()` in CompressAI


## Install
```bash
mkdir build
cd build

# Build the C++ extension
cmake ..
make
```

Then `test-compressai.ipynb` should work.


## Speed comparison

<p align="center">
<img src="figures/model-time.png">
</p>
