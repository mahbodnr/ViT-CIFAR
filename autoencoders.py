import torch
from torch import nn
from torch.nn import functional as F

from nnmf.AutoNNMFLayer import AutoNNMFLayer
from nnmf.NNMFLinear import NNMFLinear


class NNMFSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        x_shape = x.shape
        x = nn.Flatten(0, -2)(x)
        x = super().forward(x)
        x = nn.Unflatten(0, x_shape[:-1])(x)
        return x


def linear(input_size, hidden_size, nnmf, nnmf_params={}, dropout=0.0):
    if nnmf:
        return NNMFSequential(
            NNMFLinear(
                number_of_input_neurons=input_size,
                number_of_neurons=hidden_size,
                **nnmf_params,
            ),
            nn.Dropout(dropout),
        )

    else:
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )


class Autoencoder(nn.Module):
    def __init__(
        self, input_size, hidden_size, nnmf, nnmf_params={}, dropout=0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.encoder = linear(
            input_size, hidden_size, nnmf, nnmf_params, dropout=dropout
        )
        self.decoder = linear(
            hidden_size, input_size, nnmf, nnmf_params, dropout=dropout
        )
        self.hidden_activity = None

    def forward(self, x):
        x = self.encoder(x)
        self.hidden_activity = x.clone()
        x = self.decoder(x)
        return x


class AutoencoderT(nn.Module):
    def __init__(self, seq_len, hidden_size, nnmf, nnmf_params={}, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.encoder = linear(seq_len, hidden_size, nnmf, nnmf_params, dropout=dropout)
        self.decoder = linear(hidden_size, seq_len, nnmf, nnmf_params, dropout=dropout)
        self.hidden_activity = None

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        self.hidden_activity = x.clone()
        x = self.decoder(x)
        x = x.transpose(-1, -2)
        return x


class AutoencoderH(nn.Module):
    def __init__(
        self, input_size, hidden_size, heads, nnmf, nnmf_params={}, dropout=0.0
    ):
        assert input_size % heads == 0, "input_size must be divisible by heads"
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.encoder = linear(
            input_size, hidden_size, nnmf, nnmf_params, dropout=dropout
        )
        self.decoder = linear(
            hidden_size, input_size, nnmf, nnmf_params, dropout=dropout
        )
        self.hidden_activity = None
        self.heads = heads

    def forward(self, x):
        if x.dim() == 3:
            b, n, f = x.size()
            x = x.reshape(b, n, self.heads, f // self.heads)  # (b, n, h, f/h)
            x = x.reshape(b, -1, x.size(-1))  # (b, n*h, f/h)
            x = x.transpose(-1, -2)  # (b, f/h, n*h)
            x = self.encoder(x)
            self.hidden_activity = x.clone()
            x = self.decoder(x)
            x = x.transpose(-1, -2)  # (b, n*h, f/h)
            x = x.reshape(b, n, self.heads, -1)  # (b, n, h, f/h)
            x = x.reshape(b, n, -1)  # (b, n, f)
        elif x.dim() == 4:
            b, n, n, f = x.size()
            x = x.reshape(b, n, n, self.heads, -1)  # (b, n, n, h, f/h)
            x = x.reshape(b, n, -1, x.size(-1))  # (b, n, n*h, f/h)
            x = x.transpose(-1, -2)  # (b, n, f/h, n*h)
            x = self.encoder(x)
            self.hidden_activity = x.clone()
            x = self.decoder(x)
            x = x.transpose(-1, -2)  # (b, n, n*h, f/h)
            x = x.reshape(b, n, n, self.heads, -1)  # (b, n, n, h, f/h)
            x = x.reshape(b, n, n, -1)  # (b, n, n, f)
        else:
            raise NotImplementedError
        return x


class Autoencoder2D(nn.Module):
    def __init__(
        self,
        order,
        seq,
        features,
        seq_hidden,
        features_hidden,
        nnmf,
        nnmf_params={},
        dropout=0.0,
    ):
        super().__init__()
        self.order = order
        self.seq = seq
        self.features = features
        self.seq_hidden = seq_hidden
        self.features_hidden = features_hidden

        self.dropout = dropout
        # encoder
        self.enc_features = linear(
            features, features_hidden, nnmf, nnmf_params, dropout=dropout
        )
        self.enc_seq = linear(seq, seq_hidden, nnmf, nnmf_params, dropout=dropout)
        # decoder
        self.dec_features = linear(
            features_hidden, features, nnmf, nnmf_params, dropout=dropout
        )
        self.dec_seq = linear(seq_hidden, seq, nnmf, nnmf_params, dropout=dropout)

        self.hidden_activity = None

    def forward(self, x):
        if self.order == "fsfs":
            x = self.enc_features(x)
            x = x.transpose(-1, -2)
            x = self.enc_seq(x)
            self.hidden_activity = x.clone()
            x = x.transpose(-1, -2)
            x = self.dec_features(x)
            x = x.transpose(-1, -2)
            x = self.dec_seq(x)
            x = x.transpose(-1, -2)
        elif self.order == "sffs":
            x = x.transpose(-1, -2)
            x = self.enc_seq(x)
            x = x.transpose(-1, -2)
            x = self.enc_features(x)
            self.hidden_activity = x.clone()
            x = self.dec_features(x)
            x = x.transpose(-1, -2)
            x = self.dec_seq(x)
            x = x.transpose(-1, -2)
        elif self.order == "sfsf":
            x = x.transpose(-1, -2)
            x = self.enc_seq(x)
            x = x.transpose(-1, -2)
            x = self.enc_features(x)
            self.hidden_activity = x.clone()
            x = x.transpose(-1, -2)
            x = self.dec_seq(x)
            x = x.transpose(-1, -2)
            x = self.dec_features(x)
        else:
            raise NotImplementedError
        return x


class AutoNNMF(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        number_of_iterations,
    ):
        super().__init__()
        assert len(input_size) == 2, "input_size must be 2D"
        self.autoencoder = AutoNNMFLayer(
            number_of_input_neurons=1,
            number_of_neurons=hidden_size,
            input_size=input_size,
            forward_kernel_size=[input_size[0], 1],
            number_of_iterations=number_of_iterations,
            w_trainable=True,
            device=torch.device("cuda"),
            default_dtype=torch.float32,
            dilation=[1, 1],
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            x = self.autoencoder(x)
            x = x.squeeze(1)
        elif x.dim() == 4:
            B, T, T, F = x.shape
            x = x.reshape(B * T, T, F)
            x = x.unsqueeze(1)
            x = self.autoencoder(x)
            x = x.squeeze(1)
            x = x.reshape(B, T, T, F)
        else:
            raise NotImplementedError
        return x
