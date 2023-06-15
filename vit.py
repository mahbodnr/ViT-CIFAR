import torch
import torch.nn as nn
import torchsummary

from layers import TransformerEncoder, AttentionFreeTransformerEncoder


class ViT(nn.Module):
    def __init__(
        self,
        in_c: int = 3,
        num_classes: int = 10,
        img_size: int = 224,
        patch: int = 16,
        dropout: float = 0.0,
        num_layers: int = 12,
        hidden: int = 768,
        mlp_hidden: int = 768 * 4,
        head: int = 8,
        is_cls_token: bool = True,
    ):
        super(ViT, self).__init__()
        # hidden=384

        self.patch = patch  # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // self.patch
        f = (img_size // self.patch) ** 2 * 3  # 48 # patch vec length
        num_tokens = (self.patch**2) + 1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden)  # (b, n, f)
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        )
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        self.enc = nn.Sequential(*[
            TransformerEncoder(
                features=hidden, mlp_hidden=mlp_hidden, dropout=dropout, head=head
            )
            for _ in range(num_layers)
        ])
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, num_classes)  # for cls_token
        )

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class AttentionFreeViT(ViT):
    def __init__(
        self,
        mode,
        seq_len: int,
        factorize: bool = False,
        factorization_dimension: int = 128,
        in_c: int = 3,
        num_classes: int = 10,
        img_size: int = 224,
        patch: int = 16,
        dropout: float = 0.0,
        num_layers: int = 12,
        hidden: int = 768,
        mlp_hidden: int = 768 * 4,
        head: int = 1,
        is_cls_token: bool = True,
    ):
        super(AttentionFreeViT, self).__init__(
            in_c,
            num_classes,
            img_size,
            patch,
            dropout,
            num_layers,
            hidden,
            mlp_hidden,
            head,
            is_cls_token,
        )
        self.enc = nn.Sequential(
            *[
                AttentionFreeTransformerEncoder(
                    mode=mode,
                    features=hidden,
                    seq_len=seq_len,
                    mlp_hidden=mlp_hidden,
                    dropout=dropout,
                    head=head,
                    factorize=factorize,
                    factorization_dimension=factorization_dimension,
                )
                for _ in range(num_layers)
            ]
        )


if __name__ == "__main__":
    from torchview import draw_graph

    b, c, h, w = 4, 3, 32, 32
    input_size = (b, c, h, w)
    x = torch.randn(input_size)
    net = ViT(
        in_c=c,
        num_classes=10,
        img_size=h,
        patch=16,
        dropout=0.1,
        num_layers=7,
        hidden=384,
        head=12,
        mlp_hidden=384,
        is_cls_token=False,
    )
    draw_graph(
        net,
        graph_name="ViT",
        input_size=input_size,
        expand_nested=True,
        save_graph=True,
        directory="imgs",
    )
    AFViT = AttentionFreeViT(
        mode="full",
        seq_len=16*16,
        in_c=c,
        num_classes=10,
        img_size=h,
        patch=16,
        dropout=0.1,
        num_layers=7,
        hidden=384,
        head=1,
        mlp_hidden=384,
        is_cls_token=False,
    )
    draw_graph(
        AFViT,
        graph_name="AFViT",
        input_size=input_size,
        expand_nested=True,
        save_graph=True,
        directory="imgs",
    )
