"""Main training module."""

import argparse
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from dynehr.model import Net
from utils.helpers import array, get_device, new_model_dir, set_seed


class EHR(Dataset):
    def __init__(self, X, Y, T=48.0):
        """PyTorch `Dataset` class with added target tiling.

        Parameters
        ----------
        X : torch.Tensor
            Input data

        Y : torch.Tensor
            Output data

        T : float, optional
            Total period of time
        """
        super(EHR, self).__init__()

        self.X = torch.tensor(X)
        # shape: (n_patients,) -> (n_patients, n_intervals)
        Y = np.tile(Y[:, None], (1, int(T)))
        self.Y = torch.tensor(Y).float()
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_epoch(model, device, train_loader, optimizer):
    model.train()
    epoch_loss = 0.

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output, kl_loss = model(data)
        loss = F.binary_cross_entropy(output, target) + kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    mean_epoch_loss = epoch_loss / len(train_loader)

    return mean_epoch_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0

    outputs = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target).item()
            outputs += [array(output)]

    mean_test_loss = test_loss / len(test_loader)

    outputs = np.concatenate(outputs)[:, -1]
    targets = test_loader.dataset.Y[:, -1]
    test_auroc = roc_auc_score(targets, outputs)
    test_auprc = average_precision_score(targets, outputs)

    return mean_test_loss, test_auroc, test_auprc


def parse_args(args_to_parse):
    description = 'PyTorch implementation of dynamic EHR prediction.'
    parser = argparse.ArgumentParser(description=description)

    # General options
    general = parser.add_argument_group('General options')

    # Not doing anything at the moment
    general.add_argument('name', type=str,
        help='Name of the model for storing and loading purposes.')
    general.add_argument('data', type=str,
        help='Path to array dictionary')
    general.add_argument('-s', '--seed', type=int, default=None,
        help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training options')
    training.add_argument('-e', '--epochs', type=int, default=10,
        help='Maximum number of epochs.')
    training.add_argument('-bs', '--batch-size', type=int, default=64,
        help='Batch size for training.')
    training.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
        help='Learning rate.')
    training.add_argument('-wd', '--weight_decay', type=float, default=0,
        help='Weight decay.')

    # Model options
    model = parser.add_argument_group('Model specific options')
    model.add_argument('-t', '--t_hours', type=int, default=48.0,
        help='ICU data time length.')
    model.add_argument('-n', '--n_bins', type=int, default=10,
        help='Number of bins per continuous variable.')
    model.add_argument('-z', '--emb-dim', type=int, default=32,
        help='Dimension of the token embedding.')
    model.add_argument('-r', '--rnn-dim', type=int, default=512,
        help='Dimension of the LSTM hidden state.')
    model.add_argument('-p', '--p-dropout', type=float, default=0.0,
        help='Dropout rate.')
    model.add_argument('-ln', '--layer-norm', action='store_true', default=False,
        help='Whether to use layer normalisation in the LSTM unit.')

    # Bayesian model options
    bayes = parser.add_argument_group('Bayesian model options')
    model.add_argument('--variational', action='store_true', default=False,
        help='Whether to make the embedding layer Bayesian.')

    args = parser.parse_args(args_to_parse)

    return args


def main():
    # Initialise
    config = parse_args(sys.argv[1:])
    device = get_device()
    set_seed(config.seed)

    # Data
    arrs = np.load(config.data, allow_pickle=True).item()
    train_data = EHR(arrs['X_train'], arrs['Y_train'])
    valid_data = EHR(arrs['X_valid'], arrs['Y_valid'])
    train_loader = DataLoader(train_data,
        batch_size=config.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_data,
        batch_size=128, shuffle=False, pin_memory=True)

    # Model
    n_tokens = int(arrs['X_train'].max()) + 1  # +1 to embedding size for 0 emb
    model = Net(n_tokens, config.emb_dim, config.rnn_dim,
                variational=config.variational,
                layer_norm=config.layer_norm).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.learning_rate,
                           weight_decay=config.weight_decay)

    # Train
    for epoch in range(config.epochs):
        t_loss = train_epoch(model, device, train_loader, optimizer)
        v_loss, v_auroc, v_auprc = test(model, device, valid_loader)

    # Save
    new_model_dir(os.path.join('results', config.name))
    torch.save(model.state_dict(), os.path.join(RES_DIR, "model.h5"))


if __name__ == '__main__':
    main()
