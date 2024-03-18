import torch
import torch.nn as nn

from utils.types import Operations


class PoolLayer(nn.Module):
    """
    Pooling layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    """

    def __init__(self, pool_stride=2, pool_kernel=2, pool_type: Operations = Operations.max_pool, tensor_type='2d'):
        super(PoolLayer, self).__init__()

        if tensor_type == '2d':
            if pool_type == Operations.max_pool:
                self.pool = nn.MaxPool2d(pool_stride, pool_kernel)

            elif pool_type == Operations.avg_pool:
                self.pool = nn.AvgPool2d(pool_stride, pool_kernel)

        else:
            raise ValueError("Only 2D pooling is supported")

    def forward(self, input):
        output = self.pool(input)
        return output
