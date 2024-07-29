import torch
from compressai.entropy_models import GaussianConditional

from build import ansnp


class MyGaussianConditional(GaussianConditional):
    def compress(self, inputs, indexes, means=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self.quantize(inputs, "symbols", means)

        if len(inputs.size()) < 2:
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        # to cpu
        symbols = symbols.to(dtype=torch.int32, device='cpu')
        indexes = indexes.to(dtype=torch.int32, device='cpu')
        cdfs = self._quantized_cdf.cpu().numpy()
        cdf_lengths = self._cdf_length.to(dtype=torch.int32, device='cpu').reshape(-1).numpy()
        offsets = self._offset.to(dtype=torch.int32, device='cpu').reshape(-1).numpy()

        strings = []
        for i in range(symbols.size(0)):
            rv = ansnp.RansEncoder().encode_with_numpy(
                symbols[i].reshape(-1).numpy(),
                indexes[i].reshape(-1).numpy(),
                cdfs, cdf_lengths, offsets,
            )
            strings.append(rv)
        return strings

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        outputs = torch.empty_like(indexes)

        # to cpu
        indexes = indexes.to(dtype=torch.int32, device='cpu')
        cdfs = self._quantized_cdf.cpu().numpy()
        cdf_lengths = self._cdf_length.to(dtype=torch.int32, device='cpu').reshape(-1).numpy()
        offsets = self._offset.to(dtype=torch.int32, device='cpu').reshape(-1).numpy()

        for i, s in enumerate(strings):
            values = ansnp.RansDecoder().decode_with_numpy(
                s, indexes[i].reshape(-1).numpy(),
                cdfs, cdf_lengths, offsets,
            )
            import numpy as np
            assert isinstance(values, np.ndarray)
            outputs[i] = torch.from_numpy(values).to(outputs.device).reshape(outputs[i].shape)
        outputs = self.dequantize(outputs, means, dtype)
        return outputs
