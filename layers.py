import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(
        self, features: int, mlp_hidden: int, head: int = 8, dropout: float = 0.0
    ):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(features)
        self.attention = MultiHeadSelfAttention(features, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(features)
        self.mlp = nn.Sequential(
            nn.Linear(features, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, features),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.attention(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, features: int, head: int = 8, dropout: float = 0.0):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.features = features
        self.sqrt_d = self.features**0.5

        self.Wq = nn.Linear(features, features)
        self.Wk = nn.Linear(features, features)
        self.Wv = nn.Linear(features, features)

        self.out_project = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.size()  # (#Batches, #Inputs, #Features)
        Q = self.Wq(x).view(B, T, self.head, self.features // self.head).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.head, self.features // self.head).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.head, self.features // self.head).transpose(1, 2)

        attn_map = F.softmax(
            torch.einsum("bhif, bhjf->bhij", Q, K) / self.sqrt_d, dim=-1
        )  # (b,h,n,n)
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
        self.features = features
        self.Wq = nn.Linear(features, features)
        self.Wk = nn.Linear(features, features)
        self.Wv = nn.Linear(features, features)
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
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        Q_sig = torch.sigmoid(Q)
        if self.factorize:
            w = self.u @ self.v
        else:
            w = self.w
        # reduce the max value along arbitrary axis for stability reasons. The value will be cancelled out.
        exp_w = torch.exp(w - torch.max(w, dim=-1, keepdim=True)[0])
        exp_K = torch.exp(K - torch.max(K, dim=-1, keepdim=True)[0])
        weighted = (exp_w @ torch.mul(exp_K, V)) / (exp_w @ exp_K)
        Yt = torch.mul(Q_sig, weighted)
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
    ):
        super(AttentionFreeTransformerEncoder, self).__init__(
            features, mlp_hidden, head, dropout
        )
        if mode == "full":
            self.attention = AFTFull(
                features,
                seq_len,
                factorize=factorize,
                factorization_dimension=factorization_dimension,
                head=head,
                dropout=dropout,
            )
        elif mode == "local":
            raise NotImplementedError
        elif mode == "conv":
            raise NotImplementedError
        else:
            raise ValueError(f"mode must be one of 'full', 'local', 'conv'. Got {mode}")


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
