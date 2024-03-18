import torch
import torch.nn as nn
import numpy as np
from layers.convolutions import ConvLayer
from layers.pooling import PoolLayer
from layers.activations import ActivationLayer
import re


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=3,
                 padding='SAME',
                 stride=1,
                 dilation=1,
                 tensor_type='2d',
                 conv_block='ConvBnActiv',
                 causal=False,
                 batch_norm=True,
                 groups=1,
                 activation='relu',
                 separable=False,
                 pool_stride=None,
                 pool_kernel=None,
                 upsample=None,
                 dropout=0.0,
                 conv_init='kaiming',
                 bn_init='default'):

        super(ConvBlock, self).__init__()

        op_list = re.findall('[A-Z][a-z]*', conv_block)
        if not batch_norm and 'Bn' in op_list:
            op_list.remove('Bn')
        dim = int(tensor_type[0])

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size,) * dim

        if type(dilation) is not tuple:
            dilation = (dilation,) * dim

        if type(stride) is not tuple:
            stride = (stride,) * dim

        self.layers = []
        for ii, layer_name in enumerate(op_list):

            if layer_name == 'Conv':
                self.layers.append(ConvLayer(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, padding=padding, stride=stride,
                                             upsample=upsample, dilation=dilation, tensor_type=tensor_type,
                                             causal=causal, groups=groups, separable=separable))  # assumes the stride and kernel are the same size
                nn.init.kaiming_normal_(self.layers[-1].conv_layer.weight)

            elif layer_name in ['Avg', 'Max']:
                self.layers.append(PoolLayer(pool_stride, pool_kernel, pool_type=layer_name, tensor_type=tensor_type))

            elif layer_name == 'Bn':
                if ii > op_list.index('Conv'):
                    bn_channels = out_channels
                else:
                    bn_channels = in_channels
                self.layers.append(nn.BatchNorm2d(bn_channels))
                nn.init.constant_(self.layers[-1].weight, 1)
                nn.init.constant_(self.layers[-1].bias, 0)

            elif layer_name == 'Activ':
                self.layers.append(ActivationLayer(activation))

            elif layer_name == 'Drop':
                self.layers.append(nn.Dropout2d(p=dropout))

            elif layer_name == 'Up':
                self.layers.append(nn.Upsample(scale_factor=upsample))

        self.layers = nn.Sequential(*self.layers)

        #  Initializing weights
        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                if conv_init == 'kaiming':
                    nonlinearity = activation
                    if isinstance(nonlinearity, str) and 'Activ' in op_list:
                        if activation == 'relu':
                            nn.init.zeros_(m.weight)
                        elif activation == 'elu':
                            nn.init.zeros_(m.weight)
                        elif activation == 'lrelu':
                            nn.init.zeros_(m.weight)
                        else:
                            nn.init.xavier_normal_(m.weight)
                    else:
                        nn.init.xavier_normal_(m.weight)
                elif conv_init == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif conv_init == 'zeros':
                    nn.init.constant_(m.weight, 0.)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if bn_init == 'ones':
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif bn_init == 'zeros':
                    # Zero-initialize the last BN in each residual branch,
                    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
                    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)
