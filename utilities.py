import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(1234)
np.random.seed(1234)


def gradients(outputs, inputs):
    # get dy/dx
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)


def to_device(input, device):
    # set input to device
    input = input.clone().detach().to(device)
    input.requires_grad_(True)
    return input


def to_numpy(input):
    # transform tensor to numpy
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or np.ndarray, but got {}'.format(type(input)))


def mean_squared_error(pred, exact):
    # mse
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return torch.mean(torch.square(pred - exact))


def relative_error(pred, exact):
    # l2 error
    if type(pred) is np.ndarray:
        return np.sqrt(np.mean(np.square(pred - exact)) / np.mean(np.square(exact)))
    return torch.sqrt(torch.mean(torch.square(pred - exact)) / torch.mean(torch.square(exact)))


def wn_linear(inputs_shape, outputs_shape):
    # full connect layer with weight nomination
    return torch.nn.utils.weight_norm(nn.Linear(inputs_shape, outputs_shape))