import torch
import torch.nn as nn
import torch.nn.functional as F

from hamburger import get_hamburger

import argparse
from typing import Optional, Tuple, List

from autoencoders import Autoencoder, AutoencoderT, AutoencoderH, Autoencoder2D


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        features: int,
        mlp_hidden: int,
        head: int = 8,
        dropout: float = 0.0,
        use_mlp: bool = True,
        save_attn_map: bool = False,
    ):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(features)
        self.attention = MultiHeadSelfAttention(
            features, head=head, dropout=dropout, save_attn_map=save_attn_map
        )
        self.la2 = nn.LayerNorm(features)
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(features, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, features),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.mlp = None
        self._save_attn_map = save_attn_map

    def forward(self, x):
        out = self.attention(self.la1(x)) + x
        if self.mlp is not None:
            out = self.mlp(self.la2(out)) + out
        return out

    @property
    def save_attn_map(self):
        return self._save_attn_map

    @save_attn_map.setter
    def save_attn_map(self, value):
        self._save_attn_map = value
        self.attention.save_attn_map = value

    def get_attention_map(self):
        if self._save_attn_map:
            return self.attention.attn_map
        else:
            raise Exception(
                "Attention map was not saved. Set save_attn_map=True when initializing the model."
            )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        features: int,
        head: int = 8,
        dropout: float = 0.0,
        save_attn_map: bool = False,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.features = features
        self.sqrt_d = self.features**0.5

        self.Wq = nn.Linear(features, features)
        self.Wk = nn.Linear(features, features)
        self.Wv = nn.Linear(features, features)

        self.out_project = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)

        self.save_attn_map = save_attn_map

    def forward(self, x):
        B, T, _ = x.size()  # (#Batches, #Inputs, #Features)
        Q = self.Wq(x).view(B, T, self.head, self.features // self.head).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.head, self.features // self.head).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.head, self.features // self.head).transpose(1, 2)

        attn_map = F.softmax(
            torch.einsum("bhif, bhjf->bhij", Q, K) / self.sqrt_d, dim=-1
        )  # (b,h,n,n)
        if self.save_attn_map:
            self.attn_map = attn_map
        attn = torch.einsum("bhij, bhjf->bihf", attn_map, V)  # (b,n,h,f//h)
        output = self.dropout(self.out_project(attn.flatten(2)))
        return output


class AFTFull(nn.Module):
    def __init__(
        self,
        features: int,
        seq_len: int,
        factorize: bool = False,
        factorization_dimension: int = 128,
        head: int = 1,
        dropout: float = 0.0,
        query: bool = True,
    ):
        super().__init__()
        """
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        features: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full

        Number of heads is 1 as done in the paper
        w ∈ R^{T×T} is the learned pair-wise position biases
        we assume the query, key and value are the same dimension within each head,
        and the output dimension matches that of the input.
        """
        if head > 1:
            raise NotImplementedError
        self.query = query
        self.features = features
        self.Wk = nn.Linear(features, features)
        self.Wv = nn.Linear(features, features)
        if query:
            self.Wq = nn.Linear(features, features)
        self.factorize = factorize
        if factorize:
            self.u = nn.Parameter(torch.Tensor(seq_len, factorization_dimension))
            self.v = nn.Parameter(torch.Tensor(factorization_dimension, seq_len))
            nn.init.xavier_uniform_(self.u)
            nn.init.xavier_uniform_(self.v)
        else:
            self.w = nn.Parameter(torch.Tensor(seq_len, seq_len))
            nn.init.xavier_uniform_(self.w)

        self.out_project = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (#Batches, #Inputs, #Features)
        K = self.Wk(x)
        V = self.Wv(x)
        if self.factorize:
            w = (self.u @ self.v).unsqueeze(0)
        else:
            w = self.w.unsqueeze(0)
        # reduce the max value along arbitrary axis for stability reasons. The value will be cancelled out.
        exp_w = torch.exp(w - torch.max(w, dim=-1, keepdim=True)[0])
        exp_K = torch.exp(K - torch.max(K, dim=0, keepdim=True)[0])
        Yt = (exp_w @ torch.mul(exp_K, V)) / (exp_w @ exp_K)
        if self.query:
            Q = self.Wq(x)
            Yt = torch.mul(torch.sigmoid(Q), Yt)
        output = self.dropout(self.out_project(Yt))
        return output


class AFTSimple(nn.Module):
    def __init__(
        self,
        features: int,
        head: int = 1,
        dropout: float = 0.0,
        query: bool = True,
    ):
        super().__init__()
        """
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        features: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        """
        if head > 1:
            raise NotImplementedError
        self.query = query
        self.features = features
        self.Wk = nn.Linear(features, features)
        self.Wv = nn.Linear(features, features)
        if query:
            self.Wq = nn.Linear(features, features)

        self.out_project = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (#Batches, #Inputs, #Features)
        K = self.Wk(x)
        V = self.Wv(x)
        Yt = torch.sum(F.softmax(K, dim=1) * V, dim=1, keepdim=True)
        if self.query:
            Q = self.Wq(x)
            Yt = torch.mul(torch.sigmoid(Q), Yt)
        output = self.dropout(self.out_project(Yt))
        return output


class AttentionFreeTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        mode: str,
        features: int,
        seq_len: int,
        mlp_hidden: int,
        factorize: bool = False,
        factorization_dimension: int = 128,
        head: int = 1,
        dropout: float = 0.0,
        use_mlp: bool = True,
        query: bool = True,
    ):
        super(AttentionFreeTransformerEncoder, self).__init__(
            features, mlp_hidden, head, dropout, use_mlp
        )
        if mode == "full":
            self.attention = AFTFull(
                features,
                seq_len,
                factorize=factorize,
                factorization_dimension=factorization_dimension,
                head=head,
                dropout=dropout,
                query=query,
            )
        elif mode == "simple":
            self.attention = AFTSimple(features, head=head, dropout=dropout)
        elif mode == "local":
            raise NotImplementedError
        elif mode == "conv":
            raise NotImplementedError
        else:
            raise ValueError(f"mode must be one of 'full', 'local', 'conv'. Got {mode}")


class Hamburger(nn.Module):
    def __init__(
        self,
        version: str,
        in_c: int,
        depthwise: bool = False,
        ham_type: str = "NMF",
        MD_D: int = 512,
    ):
        super().__init__()
        hamburger_args = argparse.Namespace()
        hamburger_args.HAM_TYPE = ham_type
        hamburger_args.MD_D = MD_D
        hamburger_args.DEPTHWISE = depthwise
        self.model = get_hamburger(version)(in_c=in_c, args=hamburger_args)

    def forward(self, x):
        return self.model(x)


class HamburgerAttention(nn.Module):
    def __init__(
        self,
        burger: str,
        features: int,
        seq_len: int,
        depthwise: bool = False,
        dropout: float = 0.0,
        query: bool = True,
    ):
        super().__init__()
        """
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        features: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        """
        self.query = query
        self.hamburger = Hamburger(burger, seq_len, depthwise=depthwise)
        self.features = features
        # self.Wk = nn.Linear(features, features)
        self.Wv = nn.Linear(features, features)
        if query:
            self.Wq = nn.Linear(features, features)

        self.out_project = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (#Batches, #Inputs, #Features)
        # K = self.Wk(x)
        V = self.Wv(x)
        K = self.hamburger(x)
        Yt = torch.sum(F.softmax(K, dim=1) * V, dim=1, keepdim=True)
        if self.query:
            Q = self.Wq(x)
            Yt = torch.mul(torch.sigmoid(Q), Yt)
        output = self.dropout(self.out_project(Yt))
        return output


class HamburgerAttentionTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        burger: str,
        features: int,
        seq_len: int,
        depthwise: bool,
        mlp_hidden: int,
        dropout: float = 0.0,
        use_mlp: bool = True,
        query: bool = True,
    ):
        super(HamburgerAttentionTransformerEncoder, self).__init__(
            features, mlp_hidden, 1, dropout, use_mlp
        )
        self.attention = HamburgerAttention(
            burger,
            features,
            seq_len,
            depthwise=depthwise,
            dropout=dropout,
            query=query,
        )


class HamburgerTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        burger: str,
        features: int,
        seq_len: int,
        depthwise: bool,
        mlp_hidden: int,
        use_mlp: bool = True,
        dropout: float = 0.0,
    ):
        super(HamburgerTransformerEncoder, self).__init__(
            features, mlp_hidden, 1, dropout, use_mlp
        )
        self.attention = Hamburger(
            version=burger,
            in_c=seq_len,
            depthwise=depthwise,
        )


class GatedNNMF(nn.Module):
    def __init__(
        self,
        NNMF_type,
        seq_len,
        features,
        ffn_features,
        number_of_iterations,
        train_bases,
        local_learning,  # Only SbS
        depthwise,  # Only ham
    ):
        super().__init__()
        assert ffn_features % 2 == 0
        self.features = features
        self.ffn_features = ffn_features
        self.U = nn.Linear(features, ffn_features)
        self.V = nn.Linear(ffn_features // 2, features)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm([ffn_features // 2])

        self.NNMF_type = NNMF_type
        if NNMF_type == "ham":
            from hamburger.ham import NMF2D

            # add args to a argparser instance called NNMF_args
            NNMF_args = argparse.Namespace()
            NNMF_args.DEPTHWISE = depthwise
            NNMF_args.TRAIN_STEPS = number_of_iterations
            NNMF_args.EVAL_STEPS = number_of_iterations
            NNMF_args.RAND_INIT = not train_bases
            self.NNMF = NMF2D(NNMF_args)
        elif NNMF_type == "sbs":
            from nnmf.NNMFLayerSbSBP import NNMFConv2d

            if not train_bases:
                print("[Warning] SbS style NNMF called without trainable weights")
                # raise Exception("SbS style NNMF called without trainable weights")
            if depthwise:
                raise NotImplementedError
            self.NNMF = NNMFConv2d(
                number_of_input_neurons=1,  # input channes
                number_of_neurons=seq_len,  # output channels
                input_size=[seq_len, ffn_features // 2],
                forward_kernel_size=[seq_len, 1],
                number_of_iterations=number_of_iterations,
                w_trainable=train_bases,
                local_learning=local_learning,
                device=torch.device("cuda"),
                default_dtype=torch.float32,
                keep_last_grad_scale=True,
                disable_scale_grade=False,
            )
        elif NNMF_type == "sbsed":
            # from nnmf.NNMFLayerSbSBP import NNMFEncoderDecoder

            # if not train_bases:
            #     print("[Warning] SbS style NNMF called without trainable weights")
            #     # raise Exception("SbS style NNMF called without trainable weights")
            # if depthwise:
            #     raise NotImplementedError
            # self.NNMF = NNMFEncoderDecoder(
            #     number_of_input_neurons=1,  # input channes
            #     number_of_neurons=128,  # output channels
            #     input_size=[seq_len, ffn_features // 2],
            #     forward_kernel_size=[seq_len, ffn_features // 2],
            #     number_of_iterations=number_of_iterations,
            #     w_trainable=train_bases,
            #     local_learning=local_learning,
            #     device=torch.device("cuda"),
            #     default_dtype=torch.float32,
            #     keep_last_grad_scale = True,
            #     disable_scale_grade = False,
            # )
            from nnmf.AutoNNMFLayer import AutoNNMFLayer

            if not train_bases:
                print("[Warning] SbS style NNMF called without trainable weights")
            if depthwise:
                raise NotImplementedError
            self.NNMF = AutoNNMFLayer(
                number_of_input_neurons=1,  # input channes
                number_of_neurons=128,  # output channels
                input_size=[seq_len, ffn_features // 2],
                forward_kernel_size=[seq_len, ffn_features // 2],
                number_of_iterations=number_of_iterations,
                w_trainable=train_bases,
                local_learning=local_learning,
                device=torch.device("cuda"),
                default_dtype=torch.float32,
                keep_last_grad_scale=True,
                disable_scale_grade=False,
            )
        else:
            raise NotImplementedError(f"NNMF type {NNMF_type} not implemented")

    def forward(self, x):
        x = self.activation(self.U(x))
        z1, z2 = torch.chunk(x, 2, dim=-1)
        z2 = F.relu(self.norm(z2))
        if self.NNMF_type == "ham":
            z2 = self.NNMF(z2.unsqueeze(-1)).squeeze(-1)
        elif self.NNMF_type == "sbs":
            z2 = self.NNMF(z2.unsqueeze(1)).squeeze(-2)
        elif self.NNMF_type == "sbsed":
            z2 = self.NNMF(z2.unsqueeze(1)).squeeze(1)

        x = z1 * z2
        x = self.V(x)
        return x


class GatedNNMFTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        NNMF_type: str,
        seq_len: int,
        features: int,
        ffn_features: int,
        MD_iterations: int,
        train_bases: bool,
        local_learning: bool,
        depthwise: bool,
        mlp_hidden: int,
        dropout: float = 0.0,
        use_mlp: bool = True,
    ):
        super(GatedNNMFTransformerEncoder, self).__init__(
            features, mlp_hidden, 1, dropout, use_mlp
        )
        self.attention = GatedNNMF(
            NNMF_type,
            seq_len,
            features,
            ffn_features,
            MD_iterations,
            train_bases,
            local_learning,
            depthwise,
        )


class GatedMLP(nn.Module):
    def __init__(self, seq_len, features, ffn_features):
        super().__init__()
        assert ffn_features % 2 == 0
        self.features = features
        self.ffn_features = ffn_features
        self.U = nn.Linear(features, ffn_features)
        self.V = nn.Linear(ffn_features // 2, features)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm([ffn_features // 2])

        self.weight = nn.Parameter(
            torch.zeros(seq_len, seq_len).uniform_(-0.01, 0.01), requires_grad=True
        )
        self.bias = nn.Parameter(torch.ones(1, seq_len, 1), requires_grad=True)

    def forward(self, x):
        x = self.activation(self.U(x))
        z1, z2 = torch.chunk(x, 2, dim=-1)
        z2 = self.norm(z2)
        z2 = torch.einsum("ij,bjd->bid", self.weight, z2) + self.bias
        x = z1 * z2
        x = self.V(x)
        return x


class GatedMLPTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        seq_len: int,
        features: int,
        ffn_features: int,
        mlp_hidden: int,
        dropout: float = 0.0,
        use_mlp: bool = True,
    ):
        super(GatedMLPTransformerEncoder, self).__init__(
            features, mlp_hidden, 1, dropout, use_mlp
        )
        self.attention = GatedMLP(seq_len, features, ffn_features)


class WeightGatedMLP(nn.Module):
    def __init__(self, seq_len, features, ffn_features):
        super().__init__()
        assert ffn_features % 2 == 0
        self.features = features
        self.ffn_features = ffn_features
        self.U = nn.Linear(features, ffn_features)
        self.to_weight = nn.Linear(ffn_features // 2, seq_len)
        self.V = nn.Linear(ffn_features // 2, features)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm([ffn_features // 2])

    def forward(self, x):
        # x dimension: [batch, seq_len, features]
        x = self.activation(self.U(x))
        z1, z2 = torch.chunk(x, 2, dim=-1)  # [batch, seq_len, ffn_features // 2]
        z2 = self.norm(z2)
        z2 = self.to_weight(z2)  # [batch, seq_len, seq_len]
        x = torch.einsum("bij, bjf->bif", z2, z1)  # [batch, seq_len, ffn_features // 2]
        x = self.V(x)  # [batch, seq_len, features]
        return x


class WeightGatedMLPTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        seq_len: int,
        features: int,
        ffn_features: int,
        mlp_hidden: int,
        dropout: float = 0.0,
        use_mlp: bool = True,
    ):
        super(WeightGatedMLPTransformerEncoder, self).__init__(
            features, mlp_hidden, 1, dropout, use_mlp
        )
        self.attention = WeightGatedMLP(seq_len, features, ffn_features)


class LocalGlobalConvolution(nn.Module):
    def __init__(
        self,
        input_shapes,  # [channels, n_patches, n_patches]
        hidden_features,
        kernel_size: int = 1,
        use_cls_token: bool = True,
        normalization: str = "batch_norm",
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.kernel_size = kernel_size
        self.features = features = input_shapes[0]
        input_size = input_shapes[-1] * input_shapes[-2]
        self.local_conv_in = nn.Conv2d(
            features, hidden_features, kernel_size=kernel_size, padding="same"
        )
        self.local_conv_out = nn.Conv2d(
            hidden_features // 2, features, kernel_size=kernel_size, padding="same"
        )
        if use_cls_token:
            self.global_transform = nn.Linear(
                input_size + kernel_size**2, input_size + kernel_size**2
            )
        else:
            self.global_transform = nn.Linear(input_size, input_size)
        self.activation = nn.GELU()
        if normalization == "layer_norm":

            class Transpose(nn.Module):
                def forward(self, x):
                    return x.transpose(1, -1)

            self.norm = nn.Sequential(
                Transpose(),
                nn.LayerNorm(hidden_features // 2),
                Transpose(),
            )
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm2d(hidden_features // 2)

    def forward(self, x, cls_token):
        # x dimension: [batch, channels, n_patches, n_patches]
        x = self.activation(self.local_conv_in(x))
        z1, z2 = torch.chunk(x, 2, dim=1)  # split channels
        z2 = self.norm(z2)
        if self.use_cls_token:
            assert cls_token is not None, "cls_token is None"
            # CLS token dimension: [batch, channels, kernel_size, kernel_size]
            cls_token = self.activation(self.local_conv_in(cls_token))
            cls1, cls2 = torch.chunk(cls_token, 2, dim=1)
            cls2 = self.norm(cls2)
            z2_cls2 = torch.cat(
                [z2.flatten(-2), cls2.flatten(-2)], dim=-1
            )  # [batch, channels, n_patches ** 2 + kernel_size ** 2]
            z2_cls2 = self.global_transform(z2_cls2)
            z2, cls2 = z2_cls2[..., : -self.kernel_size**2].reshape_as(z2), z2_cls2[
                ..., -self.kernel_size**2 :
            ].reshape_as(cls2)
            cls_token = cls1 * cls2
            cls_token = self.local_conv_out(cls_token)
        else:
            z2 = self.global_transform(z2.flatten(-2)).reshape_as(z2)
        x = z1 * z2
        x = self.local_conv_out(x)

        if self.use_cls_token:
            return (x, cls_token)
        return x


class WeightLocalGlobalConvolution(nn.Module):
    def __init__(
        self,
        input_shapes,  # [channels, n_patches, n_patches]
        hidden_features,
        kernel_size: int = 1,
        use_cls_token: bool = True,
        normalization: str = "batch_norm",
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.kernel_size = kernel_size
        self.features = features = input_shapes[0]
        self.hidden_features = hidden_features
        input_size = input_shapes[-1] * input_shapes[-2]
        self.local_conv_in = nn.Conv2d(
            features, hidden_features, kernel_size=kernel_size, padding="same"
        )
        self.local_conv_out = nn.Conv2d(
            hidden_features // 2, features, kernel_size=kernel_size, padding="same"
        )
        if use_cls_token:
            self.global_transform = nn.Linear(input_size + kernel_size**2, features)
        else:
            self.global_transform = nn.Linear(input_size, input_size)
        self.activation = nn.GELU()
        if normalization == "layer_norm":

            class Transpose(nn.Module):
                def forward(self, x):
                    return x.transpose(1, -1)

            self.norm = nn.Sequential(
                Transpose(),
                nn.LayerNorm(hidden_features // 2),
                Transpose(),
            )
        elif normalization == "batch_norm":
            self.norm = nn.BatchNorm2d(hidden_features // 2)

    def forward(self, x, cls_token):
        if not self.use_cls_token:
            raise NotImplementedError
        assert cls_token is not None, "cls_token is None"
        # x dimension: [batch, channels (features), n_patches, n_patches]
        # CLS token dimension: [batch, channels, kernel_size, kernel_size]
        x = self.activation(
            self.local_conv_in(x)
        )  # [batch, channels (hidden_features), n_patches, n_patches]
        cls_token = self.activation(
            self.local_conv_in(cls_token)
        )  # [batch, channels (hidden_features), kernel_size, kernel_size]
        x_cls = torch.cat(
            [x.flatten(-2), cls_token.flatten(-2)], dim=-1
        )  # [batch, channels, n_patches ** 2 + kernel_size ** 2]
        z1, z2 = torch.chunk(x_cls, 2, dim=1)  # split channels
        z2 = self.norm(z2)
        z2 = self.global_transform(z2)  # [batch, channels, channels]

        x_cls = torch.einsum("bij, bjf->bif", z2, z1)
        x, cls_token = (
            x_cls[..., : -self.kernel_size**2].reshape(
                x.shape[0], self.hidden_features // 2, *x.shape[2:]
            ),
            x_cls[..., -self.kernel_size**2 :].reshape(
                cls_token.shape[0], self.hidden_features // 2, *cls_token.shape[2:]
            ),
        )
        x = self.local_conv_out(x)
        cls_token = self.local_conv_out(cls_token)

        return x, cls_token


class LocalGlobalConvolutionEncoder(nn.Module):
    def __init__(
        self,
        input_shapes: List[int],  # [channels, n_patches, n_patches]
        hidden_features: int,
        kernel_size: int,
        mlp_hidden: int,
        weight_gated: bool = False,
        dropout: float = 0.0,
        normalization: str = "batch_norm",
        use_cls_token: bool = True,
        use_mlp: bool = True,
    ):
        super().__init__()
        features = input_shapes[0]
        self.use_cls_token = use_cls_token
        if normalization == "layer_norm":

            class Transpose(nn.Module):
                def forward(self, x):
                    return x.transpose(1, -1)

            self.la1 = nn.Sequential(
                Transpose(),
                nn.LayerNorm(features),
                Transpose(),
            )
            self.la2 = nn.Sequential(
                Transpose(),
                nn.LayerNorm(features),
                Transpose(),
            )
            # self.la2 = nn.LayerNorm([features, input_shapes[-1] , input_shapes[-2]])
        elif normalization == "batch_norm":
            self.la1 = nn.BatchNorm2d(features)
            self.la2 = nn.BatchNorm2d(features)
        else:
            raise ValueError(f"normalization {normalization} not supported")
        if weight_gated:
            self.attention = WeightLocalGlobalConvolution(
                input_shapes=input_shapes,
                hidden_features=hidden_features,
                kernel_size=kernel_size,
                use_cls_token=use_cls_token,
                normalization=normalization,
            )
        else:
            self.attention = LocalGlobalConvolution(
                input_shapes=input_shapes,
                hidden_features=hidden_features,
                kernel_size=kernel_size,
                use_cls_token=use_cls_token,
                normalization=normalization,
            )
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Conv2d(
                    input_shapes[0], mlp_hidden, kernel_size=kernel_size, padding="same"
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(
                    mlp_hidden, input_shapes[0], kernel_size=kernel_size, padding="same"
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.mlp = None

    def forward(self, x):
        if self.use_cls_token:
            x, cls_token = x
        else:
            raise NotImplementedError(
                "'no CLS token' has not been implemented yet"
            )  # FIXME
        shortcut_x = x
        shortxut_cls_token = cls_token

        x = self.la1(x)
        cls_token = self.la1(cls_token)
        x, cls_token = self.attention(x, cls_token)

        x += shortcut_x
        cls_token += shortxut_cls_token

        if self.mlp is not None:
            x = self.mlp(self.la2(x)) + x
            cls_token = self.mlp(self.la2(cls_token)) + cls_token

        if self.use_cls_token:
            return (x, cls_token)
        return x


class AEAttention(nn.Module):
    def __init__(
        self,
        autoencoder,
        seq_len,
        features,
        ffn_features,
        AE_hidden,
        chunk,
        mask_type,
        save_attn_map=False,
    ):
        super().__init__()
        # assert ffn_features % 2 == 0
        self.features = features
        self.ffn_features = ffn_features
        self.U = nn.Linear(features, ffn_features)
        self.AE = autoencoder
        self.V = nn.Linear((ffn_features // 2 if chunk else ffn_features), features)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm([ffn_features // 2 if chunk else ffn_features])
        # save the input and output of the autoencoder in each forward pass
        self.AE_input = None
        self.AE_hidden = None
        self.AE_output = None
        # save attention map
        assert mask_type in ["zeros", "random"]
        self.mask_type = mask_type
        self.save_attn_map = save_attn_map
        self.chunk = chunk

        self.AE_optimizer = torch.optim.Adam(self.AE.parameters(), lr=0.001)

    def forward(self, x):
        # x dimension: [batch, seq_len, features]
        x = self.activation(self.U(x))
        # detach z2 from the graph
        if self.chunk:
            x, z = x.chunk(2, dim=-1)
            z = z.detach()
            z = self.norm1(z)
        else:
            z = x.detach()
            z = self.norm1(z)
        # Do a forward pass for the autoencoder with full unmasked data and save the input and output
        self.AE_input = z.clone()  # [batch, seq_len, ffn_features]
        self.AE_output = self.AE(z)
        self.AE_hidden = self.AE.hidden_activity.clone()
        # repeat z2, 'seq_len' times along dim=1
        z_mask = z.unsqueeze(1).repeat(
            1, z.shape[1], 1, 1
        )  # [batch, seq_len, seq_len, ffn_features]
        if self.mask_type == "zeros":
            # for each value in dim=1, keep one value in dim=2 and set the rest to 0
            z_mask = torch.eye(z.shape[1]).unsqueeze(-1).to(z.device) * z_mask
        elif self.mask_type == "random":
            # for each value in dim=1, keep one value in dim=2 and set the rest to be random
            mask = torch.eye(z.shape[1]).unsqueeze(-1).to(z.device)
            z_mask = mask * z_mask + (1 - mask) * (
                torch.randn_like(z_mask) * z_mask.std() + z_mask.mean()
            )
        # pass the result through AE
        AE_preds = self.AE(z_mask).reshape_as(
            z_mask
        )  # [batch, seq_len, seq_len, ffn_features]
        # calculate distance between the original z2 and the AE predictions
        # elementwise multiplication
        dist = (AE_preds * z.unsqueeze(1)).sum(dim=-1)
        # dist = AE_preds.sum(dim=-1)
        attn_map = F.softmax(
            dist, dim=-1
        ).detach()  # [batch, seq_len, seq_len] #TODO add without softmax
        if self.save_attn_map:
            self.attn_map = attn_map.clone()
        attn = torch.einsum(
            "bij, bjf->bif", attn_map, x
        )  # [batch, seq_len, ffn_features OR ffn_features // 2]
        x = self.V(attn)  # [batch, seq_len, features]
        return x

    def unsupervised_update(self):
        assert self.AE_input is not None

        AE_input = self.AE_input.detach().requires_grad_(True)
        AE_preds = self.AE(AE_input)
        # calculate the loss
        loss = F.mse_loss(AE_preds, AE_input)
        # zero the gradients
        self.AE_optimizer.zero_grad()
        # calculate the gradients
        loss.backward()
        # update the weights
        self.AE_optimizer.step()

        return loss.item()


class AEAttentionHeads(nn.Module):
    def __init__(
        self,
        heads,
        seq_len,
        features,
        ffn_features,
        AE_hidden,
        chunk,
        nnmf,
        nnmf_params,
        mask_type,
        save_attn_map=False,
    ):
        super().__init__()
        # assert ffn_features % 2 == 0
        self.features = features
        self.ffn_features = ffn_features
        self.heads = heads
        self.chunk = chunk
        self.U = nn.Linear(features, ffn_features)
        self.AE = AutoencoderT(seq_len * heads, AE_hidden, nnmf, nnmf_params)
        self.V = nn.Linear((ffn_features // 2 if chunk else ffn_features), features)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm([ffn_features // 2 if chunk else ffn_features])
        # save the input and output of the autoencoder in each forward pass
        self.AE_input = None
        self.AE_hidden = None
        self.AE_output = None
        # save attention map
        assert mask_type in ["zeros", "random"]
        self.mask_type = mask_type
        self.save_attn_map = save_attn_map

        self.AE_optimizer = torch.optim.Adam(self.AE.parameters(), lr=0.001)

    def forward(self, x):
        # x dimension: [batch, seq_len, features]
        x = self.activation(self.U(x))  # [batch, seq_len, ffn_features]
        if self.chunk:
            x, z = x.chunk(2, dim=-1)
            z = z.detach()  # [batch, seq_len, ffn_features // 2]
            z = self.norm1(z)
        else:
            x = self.norm1(x)
            z = x.detach()  # [batch, seq_len, ffn_features]
        x_heads = self._devide_to_heads(
            x
        )  # [batch, heads, seq_len, ffn_features(//2) // heads]
        # detach z2 from the graph
        z_heads = self._devide_to_heads(
            z
        )  # [batch, heads, seq_len, ffn_features(//2) // heads]
        # Do a forward pass for the autoencoder with full unmasked data and save the input and output
        self.AE_input = z_heads.reshape(
            z_heads.shape[0], z_heads.shape[1] * z_heads.shape[2], z_heads.shape[3]
        )  # [batch, seq_len * heads, ffn_features(//2) // heads]
        self.AE_output = self.AE(self.AE_input)
        self.AE_hidden = self.AE.hidden_activity.clone()
        # repeat z2, 'seq_len' times along dim=1
        z_mask = z.unsqueeze(1).repeat(
            1, z.shape[1], 1, 1
        )  # [batch, seq_len, seq_len, ffn_features(//2)]
        if self.mask_type == "zeros":
            # for each value in dim=1, keep one value in dim=2 and set the rest to 0
            z_mask = torch.eye(z.shape[1]).unsqueeze(-1).to(z.device) * z_mask
        elif self.mask_type == "random":
            # for each value in dim=1, keep one value in dim=2 and set the rest to be random
            mask = torch.eye(z.shape[1]).unsqueeze(-1).to(z.device)
            z_mask = mask * z_mask + (1 - mask) * (
                torch.randn_like(z_mask) * z_mask.std() + z_mask.mean()
            )
        # pass the result through AE
        z_mask_heads = self._devide_to_heads(
            z_mask
        )  # [batch, seq_len, heads, seq_len, ffn_features(//2) // heads]
        z_mask_heads_AE_input = z_mask_heads.reshape(
            z_mask_heads.shape[0],
            z_mask_heads.shape[1],
            z_mask_heads.shape[2] * z_mask_heads.shape[3],
            z_mask_heads.shape[4],
        )  # [batch, seq_len, heads * seq_len , ffn_features(//2) // heads]
        AE_preds = self.AE(z_mask_heads_AE_input).reshape_as(
            z_mask_heads
        )  # [batch, seq_len, heads, seq_len, ffn_features(//2) // heads]
        # calculate distance between the original z2 and the AE predictions
        # elementwise multiplication
        dist = (AE_preds * z_heads.unsqueeze(1)).sum(
            dim=-1
        )  # [batch, seq_len, heads, seq_len]
        # Baseline scenario
        # dist = AE_preds.sum(dim=-1)
        dist = dist.transpose(2, 1)  # [batch, heads, seq_len, seq_len]
        attn_map = F.softmax(
            dist, dim=-1
        ).detach()  # [batch, heads, seq_len, seq_len] #TODO add without softmax
        if self.save_attn_map:
            self.attn_map = attn_map.clone()
        attn = torch.einsum(
            "bhij, bhjf->bihf", attn_map, x_heads
        )  # [batch, heads, seq_len, ffn_features(//2) // heads]
        attn = attn.flatten(2)  # [batch, seq_len, ffn_features(//2)]
        x = self.V(attn)  # [batch, seq_len, features]
        return x

    def _devide_to_heads(self, x):
        """
        x: [..., seq_len, features]
        return: [..., heads, seq_len, features // heads]
        """
        new_x = x.reshape(x.shape[:-1] + (self.heads, x.shape[-1] // self.heads))
        new_x = new_x.transpose(-2, -3)
        return new_x

    def unsupervised_update(self):
        assert self.AE_input is not None

        AE_input = self.AE_input.detach().requires_grad_(True)
        AE_preds = self.AE(AE_input)
        # calculate the loss
        loss = F.mse_loss(AE_preds, AE_input)
        # zero the gradients
        self.AE_optimizer.zero_grad()
        # calculate the gradients
        loss.backward()
        # update the weights
        self.AE_optimizer.step()

        return loss.item()


class AEAttentionTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        AE_type: str,
        seq_len: int,
        features: int,
        ffn_features: int,
        AE_hidden_features: int,
        AE_hidden_seq_len: int,
        mlp_hidden: int,
        chunk: bool = False,
        order_2d: str = "sfsf",
        heads: int = 1,
        mask_type: str = "zeros",
        legacy_heads: bool = False,
        nnmf: bool = False,
        nnmf_params: dict = {},
        dropout: float = 0.0,
        use_mlp: bool = True,
        save_attn_map: bool = False,
    ):
        super(AEAttentionTransformerEncoder, self).__init__(
            features, mlp_hidden, heads, dropout, use_mlp, save_attn_map
        )
        if AE_type == "simple":
            if chunk:
                AE_input_size = ffn_features // 2
            else:
                AE_input_size = ffn_features
            autoencoder = Autoencoder(
                AE_input_size, AE_hidden_features, nnmf, nnmf_params
            )
            self.attention = AEAttention(
                autoencoder,
                seq_len,
                features,
                ffn_features,
                AE_hidden_features,
                chunk,
                mask_type,
                save_attn_map,
            )
        elif AE_type == "transpose":
            autoencoder = AutoencoderT(seq_len, AE_hidden_seq_len, nnmf, nnmf_params)
            self.attention = AEAttention(
                autoencoder,
                seq_len,
                features,
                ffn_features,
                AE_hidden_features,
                chunk,
                mask_type,
                save_attn_map,
            )
        elif AE_type == "heads":
            if legacy_heads:
                autoencoder = AutoencoderH(
                    seq_len * heads, AE_hidden_features, heads, nnmf, nnmf_params
                )
                self.attention = AEAttention(
                    autoencoder,
                    seq_len,
                    features,
                    ffn_features,
                    AE_hidden_features,
                    chunk,
                    mask_type,
                    save_attn_map,
                )
            else:
                self.attention = AEAttentionHeads(
                    heads,
                    seq_len,
                    features,
                    ffn_features,
                    AE_hidden_seq_len,
                    chunk,
                    nnmf,
                    nnmf_params,
                    mask_type,
                    save_attn_map,
                )
        elif AE_type == "2d":
            if chunk:
                AE_input_size = ffn_features // 2
            else:
                AE_input_size = ffn_features
            autoencoder = Autoencoder2D(
                order_2d,
                seq_len,
                AE_input_size,
                AE_hidden_seq_len,
                AE_hidden_features,
                nnmf,
                nnmf_params,
            )
            self.attention = AEAttention(
                autoencoder,
                seq_len,
                features,
                ffn_features,
                AE_hidden_features,
                chunk,
                mask_type,
                save_attn_map,
            )
        else:
            raise NotImplementedError(f"AE type {AE_type} not implemented")


class BaselineAEAttention(nn.Module):
    def __init__(self, seq_len, features, ffn_features, AE_hidden):
        super().__init__()
        assert ffn_features % 2 == 0
        self.features = features
        self.ffn_features = ffn_features
        self.U = nn.Linear(features, ffn_features)
        self.AE = Autoencoder(ffn_features // 2, AE_hidden)
        self.V = nn.Linear(ffn_features // 2, features)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm([ffn_features // 2])
        self.norm2 = nn.LayerNorm([ffn_features // 2])

    def forward(self, x):
        # x dimension: [batch, seq_len, features]
        x = self.activation(self.U(x))
        z1, z2 = torch.chunk(x, 2, dim=-1)  # [batch, seq_len, ffn_features // 2]
        z2 = self.norm1(z2)
        # repeat z2, 'seq_len' times along dim=1
        z2_mask = z2.unsqueeze(1).repeat(
            1, z2.shape[1], 1, 1
        )  # [batch, seq_len, seq_len, ffn_features // 2]
        # for each value in dim=1, keep one value in dim=2 and set the rest to 0
        z2_mask = torch.eye(z2.shape[1]).unsqueeze(-1).to(z2.device) * z2_mask
        # pass the result through AE
        AE_preds = self.norm2(
            self.AE(z2_mask)
        )  # [batch, seq_len, seq_len, ffn_features // 2]
        # calculate distance between the original z2 and the AE predictions
        # elementwise multiplication
        dist = (AE_preds * z2.unsqueeze(1)).sum(dim=-1)

        # multiply by the original z2 and sum along dim=1 to get the attention map
        attn_map = F.softmax(
            dist, dim=-1
        )  # [batch, seq_len, seq_len] #TODO add without softmax
        attn = torch.einsum(
            "bij, bjf->bif", attn_map, z1
        )  # [batch, seq_len, ffn_features // 2]
        x = self.V(attn)  # [batch, seq_len, features]
        return x


class BaselineAEAttentionTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        seq_len: int,
        features: int,
        ffn_features: int,
        AE_hidden: int,
        mlp_hidden: int,
        mask: bool,
        dropout: float = 0.0,
        use_mlp: bool = True,
    ):
        super(BaselineAEAttentionTransformerEncoder, self).__init__(
            features, mlp_hidden, 1, dropout, use_mlp
        )
        self.attention = BaselineAEAttention(seq_len, features, ffn_features, AE_hidden)


class LinearAttention(nn.Module):
    def __init__(self, seq_len, features, ffn_features):
        super().__init__()
        assert ffn_features % 2 == 0
        self.features = features
        self.ffn_features = ffn_features
        self.U = nn.Linear(features, ffn_features)
        self.to_weight1 = nn.Linear(ffn_features // 2, seq_len)
        self.to_weight2 = nn.Linear(seq_len, seq_len)
        self.V = nn.Linear(ffn_features // 2, features)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm([ffn_features // 2])

    def forward(self, x):
        x = self.activation(self.U(x))
        z1, z2 = torch.chunk(x, 2, dim=-1)
        z2 = self.norm(z2)
        z2 = F.relu(self.to_weight1(z2))  # [batch, seq_len, seq_len]
        z2 = self.to_weight2(z2)  # [batch, seq_len, seq_len]
        x = torch.einsum("bij, bjf->bif", z2, z1)  # [batch, seq_len, ffn_features // 2]
        x = self.V(x)
        return x


class LinearAttentionTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        seq_len: int,
        features: int,
        ffn_features: int,
        mlp_hidden: int,
        dropout: float = 0.0,
        use_mlp: bool = True,
    ):
        super(LinearAttentionTransformerEncoder, self).__init__(
            features, mlp_hidden, 1, dropout, use_mlp
        )
        self.attention = LinearAttention(seq_len, features, ffn_features)


class ANN(nn.Module):
    def __init__(self, layers, dropout=0.0, batchnorm=False, activation=nn.ReLU()):
        super(ANN, self).__init__()
        self.blocks = nn.ModuleList()
        for feature_idx in range(len(layers) - 1):
            self.blocks.append(nn.Linear(layers[feature_idx], layers[feature_idx + 1]))
            if batchnorm:
                self.blocks.append(nn.BatchNorm1d(layers[feature_idx + 1]))
            self.blocks.append(activation)
            if dropout:
                self.blocks.append(nn.Dropout(dropout))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class CNN(nn.Module):
    def __init__(
        self,
        features,
        kernel_size=3,
        batchnorm=True,
        activation=nn.ReLU(),
        pooling=True,
    ):
        super(CNN, self).__init__()
        self.features = features
        self.blocks = nn.ModuleList()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * (len(features) - 1)
        assert len(kernel_size) == len(features) - 1
        for feature_idx in range(len(features) - 1):
            self.blocks.append(
                nn.Conv2d(
                    features[feature_idx],
                    features[feature_idx + 1],
                    kernel_size=kernel_size[feature_idx],
                )
            )
            if batchnorm:
                self.blocks.append(nn.BatchNorm2d(features[feature_idx + 1]))
            self.blocks.append(activation)
            if pooling:
                self.blocks.append(nn.MaxPool2d(2, 2))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    from torchview import draw_graph

    input_size = (4, 16, 128)
    x = torch.randn(input_size)
    MHA = MultiHeadSelfAttention(input_size[-1], head=8)
    draw_graph(
        MHA,
        graph_name="Multy Head Attention",
        input_size=input_size,
        expand_nested=True,
        save_graph=True,
        directory="imgs",
    )
    net = TransformerEncoder(features=input_size[-1], mlp_hidden=256, head=8)
    draw_graph(
        net,
        graph_name="Transformer Encoder",
        input_size=input_size,
        expand_nested=True,
        save_graph=True,
        directory="imgs",
    )
    AFT = AttentionFreeTransformerEncoder(
        mode="full",
        features=input_size[-1],
        seq_len=input_size[1],
        mlp_hidden=256,
        head=1,
    )
    draw_graph(
        AFT,
        graph_name="Attention Free Transformer Encoder",
        input_size=input_size,
        expand_nested=True,
        save_graph=True,
        directory="imgs",
    )
    lgconv = LocalGlobalConvolutionEncoder(
        input_shapes=(378, 16, 16),
        hidden_features=128,
        kernel_size=1,
        mlp_hidden=200,
    )
    draw_graph(
        lgconv,
        graph_name="Local Global Convolution Encoder",
        input_data=[(torch.randn(1, 378, 16, 16), torch.randn(1, 378, 1, 1))],
        expand_nested=True,
        save_graph=True,
        directory="imgs",
    )
