import warnings
from functools import partial
from typing import Union, Sequence

import torch.nn as nn
from torch.nn.common_types import _size_2_t

from layers.activations import ActivationLayer
from layers.convolutions import create_conv_layer
from layers.pooling import PoolLayer
from utils.types import Operations, Activations, WeightInit


class ConvBlock(nn.Module):
    """Convolutional block abstraction. Input is assumed to be in canonical form: [Batch_size, channels, frames, features] for the 2d case."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_block: Sequence[Operations],
        kernel_size: _size_2_t = 3,
        padding: Union[str, _size_2_t] = "same",
        stride: _size_2_t = 1,
        dilation: _size_2_t = 1,
        tensor_dim: int = 2,
        causal: bool = False,
        groups: int = 1,
        activation: Activations = Activations.relu,
        separable: bool = False,
        pool_stride: _size_2_t = 1,
        pool_kernel: _size_2_t = 2,
        upsample: _size_2_t = 2,
        dropout: float = 0.0,
        conv_init: WeightInit = WeightInit.kaiming,
        bn_init: WeightInit = WeightInit.default,
    ):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param conv_block:  sequence of operations to be performed in the block
        :param kernel_size: size of the kernel
        :param padding: padding type
        :param stride: convolution stride
        :param dilation: convolution dilation
        :param tensor_dim:  dimension of the tensor (currently only 1 or 2 are supported)
        :param causal: whether the convolution should be causal
        :param groups:  number of groups for the convolution
        :param activation: activation function to be used
        :param separable: whether the convolution should be separable
        :param pool_stride: stride for the pooling operation
        :param pool_kernel: kernel size for the pooling operation
        :param upsample: upsample factor
        :param dropout: dropout probability
        :param conv_init: convolutional layer initialization
        :param bn_init: batch normalization layer initialization
        """

        super(ConvBlock, self).__init__()

        self.layers = self._create_layers(
            in_channels,
            out_channels,
            conv_block,
            kernel_size,
            padding,
            stride,
            dilation,
            tensor_dim,
            causal,
            groups,
            activation,
            separable,
            pool_stride,
            pool_kernel,
            upsample,
            dropout,
        )

        #  Initializing weights
        self.layers.apply(
            partial(
                self._init_weights,
                conv_init=conv_init,
                bn_init=bn_init,
                activation=activation,
            )
        )

    def forward(self, x):
        return self.layers(x)

    def _create_layers(
        self,
        in_channels: int,
        out_channels: int,
        conv_block: Sequence[Operations],
        kernel_size: _size_2_t,
        padding: Union[str, _size_2_t],
        stride: _size_2_t,
        dilation: _size_2_t,
        tensor_dim: int,
        causal: bool,
        groups: int,
        activation: Activations,
        separable: bool,
        pool_stride: _size_2_t,
        pool_kernel: _size_2_t,
        upsample: _size_2_t,
        dropout: float,
    ):
        cur_channels = in_channels

        layers = nn.Sequential()

        conv_exists: bool = False

        for i, op in enumerate(conv_block):
            if not isinstance(op, Operations):
                raise ValueError(f"Operation {op} not recognized.")
            if op == Operations.conv:
                conv = create_conv_layer(
                    in_channels=cur_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    dilation=dilation,
                    tensor_dim=tensor_dim,
                    causal=causal,
                    groups=groups,
                    separable=separable,
                )

                layers.add_module(f"conv_{i}", conv)
                cur_channels = out_channels
                conv_exists = True

            elif op == Operations.max_pool or op == Operations.avg_pool:
                layers.add_module(
                    f"{op.name}_{i}",
                    PoolLayer(
                        pool_stride,
                        pool_kernel,
                        pool_type=op,
                        tensor_dim=tensor_dim,
                    ),
                )

            elif op == Operations.bn:
                bn_channels = cur_channels
                if tensor_dim == 2:
                    layers.add_module(f"bn_{i}", nn.BatchNorm2d(bn_channels))
                elif tensor_dim == 1:
                    layers.add_module(f"bn_{i}", nn.BatchNorm1d(bn_channels))

            elif op == Operations.dropout:
                if tensor_dim == 2:
                    layers.add_module(f"dropout_{i}", nn.Dropout2d(p=dropout))
                elif tensor_dim == 1:
                    layers.add_module(f"dropout_{i}", nn.Dropout(p=dropout))

            elif op == Operations.upsample:
                layers.add_module(f"upsample_{i}", nn.Upsample(scale_factor=upsample))

            elif op == Operations.activation:
                layers.add_module(f"{activation.name}_{i}", ActivationLayer(activation))

        if not conv_exists:
            raise ValueError(
                "At least one convolutional layer must be present in the block."
            )

        return layers

    def _init_weights(
        self,
        m: nn.Module,
        conv_init: WeightInit,
        bn_init: WeightInit,
        activation: Activations,
    ):
        #  Initializing weights
        if isinstance(m, nn.Conv2d):
            if conv_init == WeightInit.kaiming:
                if activation not in [
                    Activations.relu,
                    Activations.elu,
                    Activations.lrelu,
                    Activations.relu6,
                ]:
                    warnings.warn(
                        f"Kaiming initialization is not recommended for {activation} activations."
                    )

                nn.init.kaiming_normal_(m.weight, nonlinearity=activation.name)

            elif conv_init == WeightInit.xavier:
                nn.init.xavier_normal_(m.weight)

            elif conv_init == WeightInit.zeros:
                nn.init.zeros_(m.weight)

            elif conv_init == WeightInit.ones:
                nn.init.ones_(m.weight)

            elif conv_init != WeightInit.default:
                raise ValueError(
                    f"Convolutional initialization {conv_init} not recognized."
                )

        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if bn_init == WeightInit.ones:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif bn_init == WeightInit.zeros:
                # Zero-initialize the last BN in each residual branch,
                # so that the residual branch starts with zeros, and each residual block behaves like an identity.
                # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
