import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t

from utils.types import Operations


class PoolLayer(nn.Module):
    """
    Pooling layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    """

    def __init__(
        self,
        pool_stride: _size_2_t = 2,
        pool_kernel: _size_2_t = 2,
        pool_type: Operations = Operations.max_pool,
        tensor_dim: int = 2,
    ):
        super(PoolLayer, self).__init__()

        if tensor_dim == 2:
            if pool_type == Operations.max_pool:
                self.pool = nn.MaxPool2d(pool_stride, pool_kernel)

            elif pool_type == Operations.avg_pool:
                self.pool = nn.AvgPool2d(pool_stride, pool_kernel)

        elif tensor_dim == 1:
            assert isinstance(pool_stride, int) and isinstance(
                pool_kernel, int
            ), "For 1d tensors, pool_stride and pool_kernel must be integers."
            if pool_type == Operations.max_pool:
                self.pool = nn.MaxPool1d(pool_stride, pool_kernel)

            elif pool_type == Operations.avg_pool:
                self.pool = nn.AvgPool1d(pool_stride, pool_kernel)

        else:
            raise ValueError(f"Only 1d and 2d tensors are supported. Got: {tensor_dim}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.pool(input)
        return output
