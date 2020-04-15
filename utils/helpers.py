"""Helper module."""

import argparse
import numpy as np
import os
import random
import shutil
import torch


def array(x):
    """Place tensor on cpu and convert to numpy.ndarray."""
    if x.device != torch.device('cpu'):
        x = x.cpu()

    return x.detach().numpy()


def get_device(is_gpu=True):
    """Return the correct device"""
    return torch.device('cuda' if torch.cuda.is_available() and is_gpu else 'cpu')


def get_model_device(model):
    """Return the device on which a model is."""
    return next(model.parameters()).device


def get_n_param(model):
    """Return the number of parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nParams = sum([np.prod(p.size()) for p in model_parameters])

    return nParams


def new_model_dir(directory):
    """Create a directory and archive the previous one if already existed."""
    if os.path.exists(directory):
        shutil.make_archive(directory, 'zip', directory)
        shutil.rmtree(directory)
    os.makedirs(directory)


def set_seed(seed):
    """Set all random seeds."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
