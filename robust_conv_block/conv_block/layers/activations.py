import torch.nn as nn

from utils.types import Activations


def ActivationLayer(act_type):
    """
    Get the activation layer that is requested

    :param act_type: activation type
    :return: Activation layer
    """
    if act_type == Activations.relu:
        return nn.ReLU()
    elif act_type == Activations.relu6:
        return nn.ReLU6()
    elif act_type == Activations.lrelu:
        return nn.lrelu()
    elif act_type == Activations.tanh:
        return nn.Tanh()
    elif act_type == Activations.sigmoid:
        return nn.Sigmoid()
    elif act_type == Activations.elu:
        return nn.ELU()

    else:
        raise ValueError(f"Activation type {act_type} not supported")
