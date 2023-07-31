import torch
import torch.nn as nn
from layers import LocalGlobalConvolutionEncoder


class LocalGlobalCNN(nn.Module):
    def __init__(
        self,
        weight_gated,
        num_layers,
        in_c,
        num_classes,
        n_channels,
        hidden_features,
        img_size,
        patch,
        kernel_size,
        use_cls_token,
        mlp_hidden,
        dropout,
        normalization,
        use_mlp,
    ):
        super().__init__()
        if not use_cls_token:
            raise NotImplementedError(
                "LocalGlobalCNN does not support not using cls token"
            )
        assert hidden_features % 2 == 0, "hidden_features must be divisible by 2"
        self.n_channels = n_channels
        self.patch = patch
        self.kernel_size = kernel_size
        self.patch_size = img_size // self.patch
        assert self.patch_size * self.patch == img_size, "img_size must be divisible by patch"
        self.use_cls_token = use_cls_token
        self.cls_token = (
            nn.Parameter(torch.randn(n_channels, kernel_size, kernel_size))
            if use_cls_token
            else None
        )
        self.in_c = in_c
        self.num_classes = num_classes
        self.emb = nn.Conv2d(in_c, n_channels, kernel_size=self.patch_size, stride=self.patch_size)
        self.enc = nn.Sequential(
            *[
                LocalGlobalConvolutionEncoder(
                    input_shapes=(n_channels, patch, patch),
                    hidden_features=hidden_features,
                    kernel_size=kernel_size,
                    use_cls_token=use_cls_token,
                    mlp_hidden=mlp_hidden,
                    weight_gated=weight_gated,
                    dropout=dropout,
                    normalization=normalization,
                    use_mlp=use_mlp,
                )
                for _ in range(num_layers)
            ]
        )
        if use_cls_token:
            output_token_size = n_channels * kernel_size**2
            self.fc = nn.Sequential(
                nn.LayerNorm(output_token_size),
                nn.Linear(output_token_size, num_classes),
            )
        else:
            raise NotImplementedError(
                "LocalGlobalCNN does not support not using cls token"
            )

    def forward(self, x):
        x = self.emb(x)
        # out = out + self.pos_emb
        cls_token = self.cls_token.repeat(
            x.size(0), 1, 1, 1
        )  # add batch dimension to cls token
        x, cls_token = self.enc((x, cls_token))
        if self.use_cls_token:
            return self.fc(cls_token.flatten(1))
        else:
            raise NotImplementedError(
                "LocalGlobalCNN does not support not using cls token"
            )


if __name__ == "__main__":
    from torchview import draw_graph

    num_layers = 2
    in_c = 3
    num_classes = 10
    n_channels = 64
    img_size = 32
    kernel_size = 1
    use_cls_token = True
    batch_size = 2

    input_size = [in_c, img_size, img_size]
    model = LocalGlobalCNN(
        num_layers, in_c, num_classes, n_channels, img_size, kernel_size, use_cls_token
    )
    draw_graph(
        model,
        graph_name="Local-Global CNN",
        input_size=[batch_size] + input_size,
        expand_nested=True,
        save_graph=True,
        directory="imgs",
    )


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def SparseAutoencoderLoss(autoencoder, input, model_output, l1_regularization=0.5, l1_all_layer_outputs=False):
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    mse = mse_loss(model_output, input)
    l1 = 0
    values = input
    if l1_all_layer_outputs:
        for layer in autoencoder.children():
            values = layer(values)
            l1 += l1_loss(values, torch.zeros_like(values))
    l1 += l1_loss(model_output, input)

    return mse + l1_regularization * l1