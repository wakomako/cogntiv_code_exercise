from torchinfo import summary

from robust_conv_block.conv_block.conv_block import ConvBlock
from utils.types import Operations, Activations, WeightInit


def main():
    my_conv_block = ConvBlock(
        conv_block=(
            Operations.conv,
            Operations.bn,
            Operations.activation,
            Operations.max_pool,
            Operations.conv,
        ),
        in_channels=3,
        out_channels=1,
        kernel_size=3,
        padding=1,
        stride=1,
        dilation=1,
        tensor_dim=2,
        causal=True,
        groups=1,
        activation=Activations.relu,
        separable=True,
        pool_stride=2,
        pool_kernel=2,
        upsample=None,
        dropout=0.0,
        conv_init=WeightInit.kaiming,
        bn_init=WeightInit.default,
    )

    summary(my_conv_block, input_size=(1, 3, 28, 28))


if __name__ == "__main__":
    main()
