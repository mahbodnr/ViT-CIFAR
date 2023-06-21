import torch
import torch.nn as nn
import torch.nn.functional as F

from hamburger import get_hamburger
from hamburger.ham import NMF2D
from utils import Args


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
        # Yt = torch.einsum("tt, btf->btf", exp_w, exp_K * V) / torch.einsum("tt, btf->btf", exp_w, exp_K)
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
        query: bool = True,
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
        hamburger_args = Args()
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
        query: bool = True,
    ):
        super(HamburgerAttentionTransformerEncoder, self).__init__(
            features, mlp_hidden, 1, dropout
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
        dropout: float = 0.0,
    ):
        super(HamburgerTransformerEncoder, self).__init__(
            features, mlp_hidden, 1, dropout
        )
        self.attention = Hamburger(
            version=burger,
            in_c=seq_len,
            depthwise=depthwise,
        )


class GatedNNMF(nn.Module):
    def __init__(self, features, ffn_features, depthwise=True):
        super().__init__()
        assert ffn_features % 2 == 0
        self.features = features
        self.ffn_features = ffn_features
        self.U = nn.Linear(features, ffn_features)
        self.V = nn.Linear(ffn_features // 2, features)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm([features])
        self.norm2 = nn.LayerNorm([ffn_features // 2])

        NNMF_args = Args()
        NNMF_args.DEPTHWISE = depthwise
        self.NNMF = NMF2D(NNMF_args)

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm1(x)
        x = self.activation(self.U(x))
        z1, z2 = torch.chunk(x, 2, dim=-1)
        z2 = self.norm2(z2)
        z2 = self.NNMF(F.relu(z2).unsqueeze(-1)).squeeze(-1)
        x = z1 * z2
        x = self.V(x)
        return x


class GatedNNMFTransformerEncoder(TransformerEncoder):
    def __init__(
        self,
        features: int,
        ffn_features: int,
        depthwise: bool,
        mlp_hidden: int,
        dropout: float = 0.0,
    ):
        super(GatedNNMFTransformerEncoder, self).__init__(
            features, mlp_hidden, 1, dropout
        )
        self.attention = GatedNNMF(features, ffn_features, depthwise)


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
    hamburger = HamburgerTransformerEncoder(
        burger="V1",
        features=input_size[-1],
        seq_len=input_size[1],
        mlp_hidden=256,
    )
    draw_graph(
        hamburger,
        graph_name="Hamburger Transformer Encoder",
        input_size=input_size,
        expand_nested=True,
        save_graph=True,
        directory="imgs",
    )
    hamburger_attention = HamburgerAttentionTransformerEncoder(
        burger="V1",
        features=input_size[-1],
        seq_len=input_size[1],
        mlp_hidden=256,
    )
    draw_graph(
        hamburger_attention,
        graph_name="Hamburger Attention Transformer Encoder",
        input_size=input_size,
        expand_nested=True,
        save_graph=True,
        directory="imgs",
    )
