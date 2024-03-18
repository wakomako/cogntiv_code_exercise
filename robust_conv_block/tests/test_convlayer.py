import pytest
import torch
from torch import nn

from convolutions import ConvLayer


class TestConvLayer:

    @pytest.fixture
    def input_2d(self):
        return torch.ones(size=(1, 1, 5, 5))

    @pytest.fixture
    def causal_conv_2d_masked(self):
        conv = ConvLayer(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            stride=1,
            dilation=1,
            groups=1,
            separable=False,
            tensor_type="2d",
            causal=True,
            bias=False,
        )

        nn.init.ones_(conv.conv_layer.weight)

        return conv

    @pytest.fixture
    def expected_causal_conv_2d_output(self):
        return torch.tensor(
            [
                [
                    [
                        [1.0, 2.0, 2.0, 2.0, 2.0],
                        [3.0, 5.0, 5.0, 5.0, 4.0],
                        [3.0, 5.0, 5.0, 5.0, 4.0],
                        [3.0, 5.0, 5.0, 5.0, 4.0],
                        [3.0, 5.0, 5.0, 5.0, 4.0],
                    ]
                ]
            ]
        )

    @pytest.mark.parametrize(
        "input_t,conv,expected_output",
        [
            ("input_2d", "causal_conv_2d_masked", "expected_causal_conv_2d_output"),
        ],
    )
    def test_causal_conv(self, input_t, conv, expected_output, request):
        input_t = request.getfixturevalue(input_t)
        conv = request.getfixturevalue(conv)
        expected_output = request.getfixturevalue(expected_output)

        out_t = conv(input_t)

        assert torch.allclose(out_t, expected_output, atol=1e-4)
