import torch
import torch.nn.functional as F

from models import (DynamicsModel, DynamicsModelLSTM, RewardModel,
                    RewardModelLSTM)


def loss_function_factory(loss_type):
    if loss_type == "mse":
        return F.mse_loss
    elif loss_type == "l1":
        return F.l1_loss
    elif loss_type == "smooth_l1":
        return F.smooth_l1_loss
    elif loss_type == "huber":
        return F.huber_loss
    else:
        raise ValueError("Unknown loss: {}".format(loss_type))


def optimizer_factory(optimizer_type):
    if optimizer_type == "adam":
        return torch.optim.Adam
    elif optimizer_type == "rmsprop":
        return torch.optim.RMSprop
    elif optimizer_type == "adamw":
        return torch.optim.AdamW
    elif optimizer_type == "sgd":
        return torch.optim.SGD
    elif optimizer_type == "sparseadam":
        return torch.optim.SparseAdam
    elif optimizer_type == "asgd":
        return torch.optim.ASGD
    else:
        raise ValueError("Unknown optimizer: {}".format(optimizer_type))


def model_factory(model_type):
    if model_type == "dynamics":
        return DynamicsModel
    elif model_type == "dynamics-lstm":
        return DynamicsModelLSTM
    elif model_type == "reward":
        return RewardModel
    elif model_type == "reward-lstm":
        return RewardModelLSTM
    else:
        raise ValueError("Unknown model type: {}".format(model_type))
