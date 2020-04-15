"""Module containing the main model class."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from math import e, pi
from scipy.stats import truncnorm
from torch.nn import Parameter
from torch import tensor, Tensor

from dynehr.lstms import LSTM
from utils.helpers import get_device


class Embedding(nn.Module):
    def __init__(self, n_tokens, emb_dim,
                 variational=False,
                 prior_std=0.25):
        """
        Embedding class with built-in variational option.

        Parameters
        ----------
        n_tokens : int
            Number of tokens to embed

        emb_dim : int
            Dimensionality embedding space

        variational : bool, optional
            Whether to use a variational embedding

        prior_std : float, optional
            Standard deviation of Gaussian prior for variational embedding
        """
        super(Embedding, self).__init__()

        self.n_tokens = n_tokens
        self.emb_dim = emb_dim
        self.variational = variational
        self.prior_std = prior_std

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            t = 1. / (self.n_tokens ** (1 / 2))
            W = truncnorm.rvs(-t, t, size=[self.n_tokens, self.emb_dim])
            if self.variational:
                mu = W
                logvar = np.log(self.prior_std * np.ones_like(W))
                W = np.concatenate((mu, logvar), axis=1)
            self.W = Parameter(tensor(W).float())

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            return mean

    def forward(self, x):
        x = F.embedding(x, self.W)

        if self.variational:
            mu = x[..., :self.emb_dim]
            logvar = x[..., self.emb_dim:]
            x = self.reparameterize(mu, logvar)
            return x, mu, logvar
        else:
            return x


class Aggregator(nn.Module):
    def __init__(self, n_tokens, emb_dim,
                 T=48.0):
        """
        Embedding aggregation class based in equal time intervals

        Parameters
        ----------
        n_tokens : int
            Number of tokens to embed

        emb_dim : int
            Dimensionality embedding space

        T : float, optional
            Total period of time
        """
        super(Aggregator, self).__init__()

        self.n_tokens = n_tokens
        self.emb_dim = emb_dim
        self.T = T
        self.device = get_device()

        self.embed = Embedding(n_tokens, emb_dim,
                               variational=False)

    def forward(self, X):
        T = X[:, :, 0]
        X = X[:, :, 1]

        T_mask = T < self.T
        n = T_mask.sum(dim=1).max()
        T = T[:, :n]
        X = X[:, :n].long()

        emb = self.embed(X)

        out = []
        for t in torch.arange(0, self.T, self.T/48, dtype=torch.float32).to(self.device):
            t_idx = ((t <= T) & (T < (t+self.T/48))).float().unsqueeze(2)
            X_t = t_idx * emb
            X_t = X_t.sum(dim=1, keepdim=True)
            out += [X_t]

        return torch.cat(out, dim=1)


class VariationalAggregator(nn.Module):
    def __init__(self, n_tokens, emb_dim,
                 T=48):
        """
        Embedding class with built-in variational option.

        Parameters
        ----------
        n_tokens : int
            Number of tokens to embed

        emb_dim : int
            Dimensionality embedding space

        T : float, optional
            Total period of time
        """
        super(VariationalAggregator, self).__init__()

        self.n_tokens = n_tokens
        self.emb_dim = emb_dim
        self.T = T
        self.device = get_device()

        self.embed = Embedding(n_tokens, emb_dim,
                               variational=True)

    def forward(self, X):
        T = X[:, :, 0]
        X = X[:, :, 1]

        T_mask = T < self.T
        n = T_mask.sum(dim=1).max()
        X = X[:, :n].long()

        emb, mu, logvar = self.embed(X)

        # KL divergence from p(z)~N(0,I) loss component
        kl_loss = _kl_normal_loss(mu, logvar)

        # Entropy of multivariate normal distribution:
        H = 0.5 * (self.emb_dim + self.emb_dim * torch.log(2 * pi) + logvar.sum(dim=2))
        H = H * T_mask[:, :n].float()
        H_cum = H.cumsum(dim=1)
        H_cum = H_cum / H_cum.max(dim=1, keepdim=True)[0]

        out = []
        for h in torch.arange(0, 1., 1./48, dtype=torch.float32).to(self.device):
            h_idx = ((h <= H_cum) & (H_cum < (h+1./48))).float().unsqueeze(2)
            X_h = h_idx * emb
            X_h_sum = X_h.sum(dim=1, keepdim=True)
            out += [X_h_sum]

        return torch.cat(out, dim=1), kl_loss


class Net(nn.Module):
    def __init__(self, n_tokens, emb_dim, rnn_dim,
                 T=48.0,
                 variational=False,
                 layer_norm=False):
        """
        Class defining network structure.

        Parameters
        ----------
        n_tokens : int
            Number of tokens to embed

        emb_dim : int
            Dimensionality embedding space

        rnn_dim : int
            Dimensionality of rnn space

        T : float, optional
            Total period of time

        variational : bool, optional
            Whether to use a variational embedding

        layer_norm : bool, optional
            Whether to use layer normalisation in the LSTM unit
        """
        super(Net, self).__init__()

        self.n_tokens = n_tokens
        self.emb_dim = emb_dim
        self.rnn_dim = rnn_dim
        self.T = T
        self.variational = variational
        self.layer_norm = layer_norm

        if self.variational:
            self.embedding = VariationalAggregator(n_tokens, emb_dim, T=T)
        else:
            self.embedding = Aggregator(n_tokens, emb_dim, T=T)
        self.lstm = LSTM(emb_dim, rnn_dim, layer_norm=layer_norm)
        self.fc = nn.Linear(rnn_dim, 1)

    def forward(self, x):
        emb = self.embedding(x)
        all_hidden, (final_hidden, _) = self.lstm(emb)
        output = self.fc(all_hidden).squeeze()

        return output.sigmoid()


def _kl_normal_loss(mean, logvar):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)
    """
    latent_dim = mean.size(1)
    # batch mean of kl for each latent dimension
    latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = latent_kl.sum()

    return total_kl
