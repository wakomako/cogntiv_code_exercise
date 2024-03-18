import torch.nn as nn


def ActivationLayer(act_type):
    """
    Get the activation layer that is requested

    :param act_type: activation type
    :return: Activation layer
    """
    if act_type == "relu":
        return nn.ReLU()
    elif act_type == "relu6":
        return nn.ReLU6()
    elif act_type == "lrelu":
        return nn.lrelu()
    elif act_type == "tanh":
        return nn.Tanh()
    elif act_type == "sigmoid":
        return nn.Sigmoid()
    elif act_type == "elu":
        return nn.ELU()
