import torch
import torch.nn as nn
import numpy as np


class ConvLayer(nn.Module):
    """
    Conv layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    Causal convolution implemented accoarding to:
    from https://gist.github.com/paultsw/7a9d6e3ce7b70e9e2c61bc9287addefc
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', stride=1,
                 upsample=None, dilation=1, tensor_type='2d', causal=False,
                 groups=1, separable=False):
        super(ConvLayer, self).__init__()
        self.dilation = dilation
        self.causal = causal
        self.kernel_size = kernel_size
        # 2D convolution
        if tensor_type == '2d':
            # Padding
            if not causal:
                if padding == 'SAME':
                    padding = ()
                    for ii in range(len(kernel_size)):
                        padding += (dilation[ii] * ((kernel_size[ii] - 1) // 2),)
                elif padding == 'VALID':
                    padding = (0,) * len(kernel_size)
            else:
                padding = (dilation[0] * ((kernel_size[0] - 1) // 2), dilation[1] * (kernel_size[1] - 1))

            # Full 2d conv
            if separable is False:
                self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=padding, dilation=dilation, stride=stride, groups=groups)
            # Separable conv
            else:
                conv_layer = []
                conv_layer.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                            padding=padding, dilation=dilation, stride=stride, groups=in_channels))
                conv_layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                            padding=0, dilation=1, stride=1))
                self.conv_layer = nn.Sequential(*conv_layer)

    def forward(self, input):
        output = self.conv_layer(input)
        if self.causal:
            output = output[..., self.dilation[-1] * (self.kernel_size[1] - 1):]

        return output
