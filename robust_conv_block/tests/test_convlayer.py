import pytest
import torch
from torch import nn

from convolutions import ConvLayer2d, ConvLayer1d

DEFAULT_CONV_PARAMS = dict(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    padding=1,
    stride=1,
    dilation=1,
    groups=1,
    separable=False,
    causal=True,
    bias=False,
)


class TestCausalConvLayer2d:

    @pytest.fixture
    def input_2d(self):
        return torch.ones(size=(1, 1, 5, 5))

    @pytest.fixture
    def causal_conv_2d_masked_padding1(self):
        conv = ConvLayer2d(**DEFAULT_CONV_PARAMS)

        nn.init.ones_(conv.conv_layer.weight)

        return conv

    @pytest.fixture
    def causal_conv_2d_masked_padding0(self):
        conv_params = DEFAULT_CONV_PARAMS.copy()
        conv_params["padding"] = 0
        conv = ConvLayer2d(**conv_params)

        nn.init.ones_(conv.conv_layer.weight)

        return conv

    @pytest.fixture
    def causal_conv_2d_masked_pad_same(self):
        conv_params = DEFAULT_CONV_PARAMS.copy()
        conv_params["padding"] = "same"
        conv = ConvLayer2d(**conv_params)

        nn.init.ones_(conv.conv_layer.weight)

        return conv

    @pytest.fixture
    def expected_causal_conv_2d_padding1_output(self):
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

    @pytest.fixture
    def expected_causal_conv_2d_padding0_output(self):
        return torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0],
                        [5.0, 5.0, 5.0],
                        [5.0, 5.0, 5.0],
                    ]
                ]
            ]
        )

    @pytest.mark.parametrize(
        "input_t,conv,expected_output",
        [
            (
                "input_2d",
                "causal_conv_2d_masked_padding1",
                "expected_causal_conv_2d_padding1_output",
            ),
            (
                "input_2d",
                "causal_conv_2d_masked_padding0",
                "expected_causal_conv_2d_padding0_output",
            ),
            (
                "input_2d",
                "causal_conv_2d_masked_pad_same",
                "expected_causal_conv_2d_padding1_output",
            ),
        ],
    )
    def test_causal_conv(self, input_t, conv, expected_output, request):
        input_t = request.getfixturevalue(input_t)
        conv = request.getfixturevalue(conv)
        expected_output = request.getfixturevalue(expected_output)

        out_t = conv(input_t)

        assert torch.allclose(out_t, expected_output, atol=1e-4)


class TestSeparableConvLayer2d:

    @pytest.fixture
    def input_2d_2c(self):
        return torch.ones(size=(1, 2, 5, 5))

    @pytest.fixture
    def separable_conv_2d(self):
        conv_params = DEFAULT_CONV_PARAMS.copy()
        conv_params.update(
            {"in_channels": 2, "padding": 0, "separable": True, "causal": False}
        )
        conv = ConvLayer2d(**conv_params)
        # init weights to ones:
        conv.conv_layer.apply(
            lambda x: nn.init.ones_(x.weight) if isinstance(x, nn.Conv2d) else None
        )

        return conv

    @pytest.fixture
    def separable_causal_conv_2d(self):
        conv_params = DEFAULT_CONV_PARAMS.copy()
        conv_params.update(
            {"in_channels": 2, "padding": 0, "separable": True, "causal": True}
        )
        conv = ConvLayer2d(**conv_params)
        # init weights to ones:
        conv.conv_layer.apply(
            lambda x: nn.init.ones_(x.weight) if isinstance(x, nn.Conv2d) else None
        )

        return conv

    @pytest.fixture
    def expected_separable_conv_2d_output(self):
        return torch.tensor(
            [
                [
                    [
                        [18.0, 18.0, 18.0],
                        [18.0, 18.0, 18.0],
                        [18.0, 18.0, 18.0],
                    ]
                ]
            ]
        )

    @pytest.fixture
    def expected_separable_causal_conv_2d_output(self):
        return torch.tensor(
            [
                [
                    [
                        [10.0, 10.0, 10.0],
                        [10.0, 10.0, 10.0],
                        [10.0, 10.0, 10.0],
                    ]
                ]
            ]
        )

    @pytest.mark.parametrize(
        "input_t,conv,expected_output",
        [
            (
                "input_2d_2c",
                "separable_conv_2d",
                "expected_separable_conv_2d_output",
            ),
            (
                "input_2d_2c",
                "separable_causal_conv_2d",
                "expected_separable_causal_conv_2d_output",
            ),
        ],
    )
    def test_separable_conv(self, input_t, conv, expected_output, request):
        input_t = request.getfixturevalue(input_t)
        conv = request.getfixturevalue(conv)
        expected_output = request.getfixturevalue(expected_output)

        out_t = conv(input_t)

        assert torch.allclose(out_t, expected_output, atol=1e-4)


class TestCausalConvLayer1d:
    @pytest.fixture
    def input_1d(self):
        return torch.ones(size=(1, 1, 5))

    @pytest.fixture
    def causal_conv_1d(self):
        conv = ConvLayer1d(**DEFAULT_CONV_PARAMS)

        nn.init.ones_(conv.conv_layer.weight)

        return conv

    @pytest.fixture
    def causal_conv_1d_dilation(self):
        conv_params = DEFAULT_CONV_PARAMS.copy()
        conv_params.update({"dilation": 2})
        conv = ConvLayer1d(**conv_params)

        # init weights to ones:
        conv.conv_layer.apply(
            lambda x: nn.init.ones_(x.weight) if isinstance(x, nn.Conv1d) else None
        )

        return conv

    @pytest.fixture
    def expected_casual_conv_1d_output(self):
        return torch.tensor([[[1.0, 2.0, 3.0, 3.0, 3.0]]])

    @pytest.fixture
    def expected_casual_conv_1d_dilation_output(self):
        return torch.tensor([[[1.0, 1.0, 2.0, 2.0, 3.0]]])

    @pytest.mark.parametrize(
        "input_t,conv,expected_output",
        [
            (
                "input_1d",
                "causal_conv_1d",
                "expected_casual_conv_1d_output",
            ),
            (
                "input_1d",
                "causal_conv_1d_dilation",
                "expected_casual_conv_1d_dilation_output",
            ),
        ],
    )
    def test_casual_conv1d(self, input_t, conv, expected_output, request):
        input_t = request.getfixturevalue(input_t)
        conv = request.getfixturevalue(conv)
        expected_output = request.getfixturevalue(expected_output)

        out_t = conv(input_t)

        assert torch.allclose(out_t, expected_output, atol=1e-4)


class TestSeparableConvLayer1d:
    @pytest.fixture
    def input_1d_2c(self):
        return torch.ones(size=(1, 2, 5))

    @pytest.fixture
    def separable_conv_1d(self):
        conv_params = DEFAULT_CONV_PARAMS.copy()
        conv_params.update(
            {"separable": True, "causal": False, "in_channels": 2, "padding": 0}
        )
        conv = ConvLayer1d(**conv_params)

        # init weights to ones:
        conv.conv_layer.apply(
            lambda x: nn.init.ones_(x.weight) if isinstance(x, nn.Conv1d) else None
        )

        return conv

    @pytest.fixture
    def expected_separable_conv_1d_output(self):
        return torch.tensor([[[6.0, 6.0, 6.0]]])

    @pytest.mark.parametrize(
        "input_t,conv,expected_output",
        [
            (
                "input_1d_2c",
                "separable_conv_1d",
                "expected_separable_conv_1d_output",
            ),
        ],
    )
    def test_separable_conv1d(self, input_t, conv, expected_output, request):
        input_t = request.getfixturevalue(input_t)
        conv = request.getfixturevalue(conv)
        expected_output = request.getfixturevalue(expected_output)

        out_t = conv(input_t)

        assert torch.allclose(out_t, expected_output, atol=1e-4)

    @pytest.fixture
    def separable_causal_conv_1d(self):
        conv_params = DEFAULT_CONV_PARAMS.copy()
        conv_params.update({"separable": True, "causal": True, "in_channels": 2})
        conv = ConvLayer1d(**conv_params)

        # init weights to ones:
        conv.conv_layer.apply(
            lambda x: nn.init.ones_(x.weight) if isinstance(x, nn.Conv1d) else None
        )

        return conv

    @pytest.fixture
    def expected_separable_causal_conv_1d_output(self):
        return torch.tensor([[[2.0, 4.0, 6.0, 6.0, 6.0]]])

    @pytest.mark.parametrize(
        "input_t,conv,expected_output",
        [
            (
                "input_1d_2c",
                "separable_causal_conv_1d",
                "expected_separable_causal_conv_1d_output",
            ),
        ],
    )
    def test_separable_causal_conv1d(self, input_t, conv, expected_output, request):
        input_t = request.getfixturevalue(input_t)
        conv = request.getfixturevalue(conv)
        expected_output = request.getfixturevalue(expected_output)

        out_t = conv(input_t)

        assert torch.allclose(out_t, expected_output, atol=1e-4)
