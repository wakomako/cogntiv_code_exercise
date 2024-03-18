import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair


class MaskedConv2d(nn.Conv2d):
    """implementation taken from https://github.com/jzbontar/pixelcnn-pytorch/blob/14c9414602e0694692c77a5e0d87188adcded118/main.py"""

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {"A", "B"}
        self.register_buffer("mask", self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == "B") :] = 0
        self.mask[:, :, kH // 2 + 1 :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=1,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


def create_2dconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding="SAME",
    stride=1,
    upsample=None,
    dilation=1,
    causal: bool = False,
    groups=1,
    separable=False,
    bias: bool = True,
):

    if (
        causal
    ):  # implement a 2d causal convolution using a masked convolution as was done in Pixel Recurrent Neural Networks
        # https://proceedings.neurips.cc/paper_files/paper/2016/file/b1301141feffabac455e1f90a7de2054-Paper.pdf
        return MaskedConv2d(
            "B",
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )


class ConvLayer(nn.Module):
    """
    Conv layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    Causal convolution implemented accoarding to:
    from https://gist.github.com/paultsw/7a9d6e3ce7b70e9e2c61bc9287addefc
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding="SAME",
        stride=1,
        upsample=None,
        dilation=1,
        tensor_type="2d",
        causal=False,
        groups=1,
        separable=False,
        bias: bool = True,
    ):
        super(ConvLayer, self).__init__()
        self.dilation = dilation
        self.causal = causal
        self.kernel_size = kernel_size
        # 2D convolution
        if tensor_type == "2d":
            self.dilation = _pair(dilation)
            self.kernel_size = _pair(kernel_size)
            self.stride = _pair(stride)
            self.padding = _pair(padding)

            # Padding
            if not causal:
                if padding == "SAME":
                    padding = ()
                    for ii in range(len(kernel_size)):
                        padding += (dilation[ii] * ((kernel_size[ii] - 1) // 2),)
                elif padding == "VALID":
                    padding = (0,) * len(kernel_size)
            else:
                self.conv_layer = create_2dconv(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    upsample=upsample,
                    dilation=dilation,
                    causal=causal,
                    groups=groups,
                    separable=separable,
                    bias=bias,
                )
                return

                padding = (
                    dilation[0] * ((kernel_size[0] - 1) // 2),
                    dilation[1] * (kernel_size[1] - 1),
                )

            # Full 2d conv
            if separable is False:
                self.conv_layer = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    stride=stride,
                    groups=groups,
                )
            # Separable conv
            else:
                conv_layer = []
                conv_layer.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                        stride=stride,
                        groups=in_channels,
                    )
                )
                conv_layer.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        padding=0,
                        dilation=1,
                        stride=1,
                    )
                )
                self.conv_layer = nn.Sequential(*conv_layer)

        else:
            raise ValueError("Only 2D convolutions are supported")

    def forward(self, input):
        output = self.conv_layer(input)
        # if self.causal:
        #     output = output[..., self.dilation[-1] * (self.kernel_size[1] - 1) :]

        return output
