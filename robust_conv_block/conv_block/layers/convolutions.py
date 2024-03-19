import warnings
from typing import Union

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class MaskedConv2d(nn.Conv2d):
    """implementation taken from https://github.com/jzbontar/pixelcnn-pytorch/blob/14c9414602e0694692c77a5e0d87188adcded118/main.py"""

    def __init__(
        self,
        mask_type="B",
        *args,
        **kwargs,
    ):
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


def create_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    padding: Union[str, _size_2_t] = "same",
    stride: _size_2_t = 1,
    dilation: _size_2_t = 1,
    tensor_dim: int = 2,
    causal: bool = False,
    groups: int = 1,
    separable: bool = False,
    bias: bool = True,
) -> nn.Module:
    """
    Create a convolutional layer based on the tensor dimension and the parameters passed.
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of the kernel
    :param padding: padding type
    :param stride: convolution stride
    :param dilation: convolution dilation
    :param tensor_dim:  dimension of the tensor (currently only 1 or 2 are supported)
    :param causal: whether the convolution should be causal
    :param groups:  number of groups for the convolution
    :param separable: whether the convolution should be separable
    :param bias: whether to use bias in the convolution
    :return: convolutional layer
    """

    if tensor_dim == 2:
        conv_layer = ConvLayer2d
    elif tensor_dim == 1:
        conv_layer = ConvLayer1d
    else:
        raise ValueError("Only 1d and 2d convolutions are supported")

    if groups != 1 and separable:
        warnings.warn(
            "groups parameter has no affect when using separable convolutions"
        )

    conv = conv_layer(
        in_channels,
        out_channels,
        kernel_size,
        padding,
        stride,
        dilation,
        causal,
        groups,
        separable,
        bias,
    )

    return conv


def create_separable_conv_layer(
    in_channels: int,
    out_channels: int,
    tensor_dim: int,
    kernel_size: _size_2_t,
    padding: Union[str, _size_2_t],
    stride: _size_2_t,
    dilation: _size_2_t,
    causal: bool,
    bias: bool,
) -> nn.Module:
    """
    Create a separable convolutional layer based on the tensor dimension and the parameters passed.
    A separable convolution may also be Causal.
    """

    if tensor_dim == 1:
        depthwise = nn.Conv1d
        pointwise = nn.Conv1d

    elif tensor_dim == 2:

        depthwise = MaskedConv2d if causal else nn.Conv2d
        pointwise = nn.Conv2d

    else:
        raise ValueError("Only 1d and 2d convolutions are supported")

    conv_layer = []
    conv_layer.append(
        depthwise(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=in_channels,
            bias=bias,
        )
    )
    conv_layer.append(
        pointwise(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            dilation=1,
            stride=1,
            bias=bias,
        )
    )
    conv_layer = nn.Sequential(*conv_layer)

    return conv_layer


class ConvLayer1d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Union[str, int] = "same",
        stride: int = 1,
        dilation: int = 1,
        causal: bool = False,
        groups: int = 1,
        separable: bool = False,
        bias: bool = True,
    ):
        super(ConvLayer1d, self).__init__()

        assert isinstance(
            kernel_size, int
        ), "For 1d tensors, kernel_size must be an integer."
        assert isinstance(stride, int), "For 1d tensors, stride must be an integer."
        assert isinstance(dilation, int), "For 1d tensors, dilation must be an integer."
        assert isinstance(
            padding, (str, int)
        ), "For 1d tensors, padding must be an integer or a string in {‘valid’, ‘same’} ."

        self.padding = padding
        self.causal = causal
        self.conv_layer = self._create_1dconv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            causal=causal,
            groups=groups,
            separable=separable,
            bias=bias,
        )

    def _create_1dconv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        causal: bool,
        groups: int,
        separable: bool,
        bias: bool,
    ) -> nn.Module:

        if causal:
            self.padding = dilation * (kernel_size - 1)

        if separable:
            conv_layer = create_separable_conv_layer(
                in_channels,
                out_channels,
                1,
                kernel_size,
                self.padding,
                stride,
                dilation,
                causal,
                bias,
            )

        else:
            conv_layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                dilation=dilation,
                stride=stride,
                groups=groups,
                bias=bias,
            )

        return conv_layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv_layer(input)
        if (
            self.causal
        ):  # a 1D causal convolution discrads the last elements of the output
            output = output[..., : -self.padding]

        return output


class ConvLayer2d(nn.Module):
    """
    Conv layer abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case.
    Causal convolution implemented accoarding to:
    from https://gist.github.com/paultsw/7a9d6e3ce7b70e9e2c61bc9287addefc
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        padding: Union[str, _size_2_t] = "same",
        stride: _size_2_t = 1,
        dilation: _size_2_t = 1,
        causal: bool = False,
        groups: int = 1,
        separable: bool = False,
        bias: bool = True,
    ):
        super(ConvLayer2d, self).__init__()

        conv2d = self._create_2dconv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            causal=causal,
            groups=groups,
            separable=separable,
            bias=bias,
        )
        self.conv_layer = conv2d

    def _create_2dconv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        padding: Union[str, _size_2_t],
        stride: _size_2_t,
        dilation: _size_2_t,
        causal: bool,
        groups: int,
        separable: bool,
        bias: bool,
    ) -> nn.Module:

        if separable:

            return create_separable_conv_layer(
                in_channels,
                out_channels,
                2,
                kernel_size,
                padding,
                stride,
                dilation,
                causal,
                bias,
            )

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

        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
            groups=groups,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv_layer(input)
        return output
