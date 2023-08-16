import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import argparse

import wandb

from nnmf.AutoNNMFLayer import AutoNNMFLayer


class Net(nn.Module):
    def __init__(
        self,
        regenerator_model: nn.Module,
        in_c: int = 3,
        img_size: int = 32,
        patch: int = 8,
        hidden: int = 384,
        is_cls_token: bool = True,
    ):
        super().__init__()
        # hidden=384
        self.patch = patch  # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // self.patch
        self.image_size = img_size
        assert (
            self.patch_size * self.patch == img_size
        ), "img_size must be divisible by patch"
        f = self.patch_size**2 * in_c
        num_tokens = (self.patch**2) + 1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden)  # (b, n, f)
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        )
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        self.regenerator = regenerator_model
        self.emb_transpose = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, f)  # for cls_token
        )
        self.mask = False
        self.mask_idx = None

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb

        self.regenerator_input = out.clone().detach()

        if self.mask:
            out = out.unsqueeze(1).repeat(1, out.shape[1], 1, 1)
            mask = torch.eye(out.shape[1]).unsqueeze(-1).to(out.device)
            out = out * mask

            # if self.mask_idx is not None:
            #     sample = out[0].clone().detach().cpu().numpy()
            #     for i, map in enumerate(sample):
            #         plt.title("I")
            #         plt.subplot(1, 17, i + 1)
            #         plt.imshow(map.transpose(1, 0))
            #         plt.axis("off")
            #     plt.show()
        out = self.regenerator(out)
        self.regenerator_output = out.clone().detach()

        # if self.mask and self.mask_idx is not None:
        #     sample = out[0].clone().detach().cpu().numpy()
        #     for i, map in enumerate(sample):
        #         plt.title("O")
        #         plt.subplot(1, 17, i + 1)
        #         plt.imshow(map.transpose(1, 0))
        #         plt.axis("off")
        #     plt.show()

        if self.mask:
            if self.mask_idx is None:
                return
            else:
                out = out[:, self.mask_idx, :]

        if self.is_cls_token:
            out = out[..., 1:, :]
        out = self.emb_transpose(out)
        out = self._to_img(out)
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

    def _to_img(self, x):
        """
        (b, n, f) -> (b, c, h, w)
        """
        out = F.fold(
            x.transpose(1, 2),
            output_size=(self.image_size, self.image_size),
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
        )
        return out


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
        self.hidden_activity = None

    def forward(self, x):
        x = self.encoder(x)
        self.hidden_activity = x.clone()
        x = self.decoder(x)
        return x


class AutoencoderT(nn.Module):
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
        self.hidden_activity = None

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        self.hidden_activity = x.clone()
        x = self.decoder(x)
        x = x.transpose(-1, -2)
        return x


class AutoencoderH(nn.Module):
    def __init__(self, input_size, hidden_size, heads, dropout=0.0):
        assert input_size % heads == 0, "input_size must be divisible by heads"
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
        self.hidden_activity = None
        self.heads = heads

    def forward(self, x):
        b, n, f = x.size()
        x = x.reshape(b, n, self.heads, -1)  # (b, n, h, f/h)
        x = x.reshape(b, -1, x.size(-1))  # (b, n*h, f/h)
        x = x.transpose(-1, -2)  # (b, f/h, n*h)
        x = self.encoder(x)
        self.hidden_activity = x.clone()
        x = self.decoder(x)
        x = x.transpose(-1, -2)  # (b, n*h, f/h)
        x = x.reshape(b, n, self.heads, -1)  # (b, n, h, f/h)
        x = x.reshape(b, n, -1)  # (b, n, f)
        return x


class Autoencoder2D(nn.Module):
    def __init__(self, order, seq, features, seq_hidden, features_hidden, dropout=0.0):
        super().__init__()
        self.order = order
        self.seq = seq
        self.features = features
        self.seq_hidden = seq_hidden
        self.features_hidden = features_hidden

        self.dropout = dropout
        # encoder
        self.enc_features = nn.Linear(features, features_hidden)
        self.enc_seq = nn.Linear(seq, seq_hidden)
        # decoder
        self.dec_features = nn.Linear(features_hidden, features)
        self.dec_seq = nn.Linear(seq_hidden, seq)

        self.hidden_activity = None

    def forward(self, x):
        if self.order == "fsfs":
            x = self.enc_features(x)
            x = F.relu(x)
            x = x.transpose(-1, -2)
            x = self.enc_seq(x)
            x = F.relu(x)
            self.hidden_activity = x.clone()
            x = x.transpose(-1, -2)
            x = self.dec_features(x)
            x = F.relu(x)
            x = x.transpose(-1, -2)
            x = self.dec_seq(x)
            x = F.relu(x)
            x = x.transpose(-1, -2)
        elif self.order == "sffs":
            x = x.transpose(-1, -2)
            x = self.enc_seq(x)
            x = F.relu(x)
            x = x.transpose(-1, -2)
            x = self.enc_features(x)
            x = F.relu(x)
            self.hidden_activity = x.clone()
            x = self.dec_features(x)
            x = F.relu(x)
            x = x.transpose(-1, -2)
            x = self.dec_seq(x)
            x = F.relu(x)
            x = x.transpose(-1, -2)
        elif self.order == "sfsf":
            x = x.transpose(-1, -2)
            x = self.enc_seq(x)
            x = F.relu(x)
            x = x.transpose(-1, -2)
            x = self.enc_features(x)
            x = F.relu(x)
            self.hidden_activity = x.clone()
            x = x.transpose(-1, -2)
            x = self.dec_seq(x)
            x = F.relu(x)
            x = x.transpose(-1, -2)
            x = self.dec_features(x)
            x = F.relu(x)
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
            dilation=[1,1],
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


def main(regenerator_model, regenerator_iterations):
    config = wandb.config

    if config.dataset == "cifar10":
        in_c = 3
        img_size = 32
    elif config.dataset == "mnist":
        in_c = 1
        img_size = 28
    else:
        raise NotImplementedError

    seq_len = config.patch**2 + (1 if config.is_cls_token else 0)

    model = Net(
        regenerator_model=regenerator_model,
        in_c=in_c,
        img_size=img_size,
        patch=config.patch,
        hidden=config.hidden,
        is_cls_token=config.is_cls_token,
    )

    model = model.cuda()
    for name, module in model.named_modules():
            if "nnmf" in name.lower():
                nnmf_layers.append(module)
    print("NNMF layers:", nnmf_layers)
    if len(nnmf_layers) > 0:
        Madam
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        regenerator_optimizer = optim.Adam(model.regenerator.parameters(), lr=config.lr)
    
    criterion = nn.MSELoss()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    if config.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif config.dataset == "mnist":
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=10, shuffle=False, num_workers=config.num_workers
    )
    test_images, _ = next(iter(testloader))
    test_images_plot = test_images / 2 + 0.5
    test_images = test_images.cuda()
    # wandb.log(
    #     {
    #         "test_images": wandb.Image(
    #             torchvision.utils.make_grid(test_images_plot, 10)
    #         )
    #     }
    # )

    for epoch in range(config.epochs):
        running_loss = 0.0
        regenerator_running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, _ = data
            inputs = inputs.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            new_loss = 0.0
            for _ in range(regenerator_iterations):
                # regenerator
                regenerator_optimizer.zero_grad()
                regenerator_output = model.regenerator(model.regenerator_input)
                regenerator_loss = criterion(
                    regenerator_output, model.regenerator_input
                )
                regenerator_loss.backward()
                regenerator_optimizer.step()
                new_loss += regenerator_loss.item()
            if regenerator_iterations > 0:
                regenerator_running_loss += new_loss / regenerator_iterations

            if i % config.log_interval == config.log_interval - 1:
                with torch.no_grad():
                    step = epoch * len(trainloader) + i
                    model.mask = True
                    model(inputs)
                    model.mask = False
                    regenerator_score = (
                        model.regenerator_input.unsqueeze(1) * model.regenerator_output
                    ).sum(-1)
                    # plotly table
                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

                    # normalize regenerator_score
                    regenerator_score /= (
                        model.regenerator_input.unsqueeze(1).norm(dim=-1)
                        * model.regenerator_output.norm(dim=-1)
                    ) + 1e-8
                    corr = regenerator_score.mean(0).detach().cpu().numpy()
                    img = axes[0, 0].imshow(corr)
                    axes[0, 0].set_title("regenerator_score")
                    plt.colorbar(img, ax=axes[0, 0])
                    # set diagonal to zero
                    regenerator_score_no_self_reconst = regenerator_score.clone()
                    regenerator_score_no_self_reconst[
                        :, torch.eye(regenerator_score_no_self_reconst.shape[1]).bool()
                    ] = 0
                    corr = (
                        regenerator_score_no_self_reconst.mean(0).detach().cpu().numpy()
                    )
                    img = axes[0, 1].imshow(corr)
                    axes[0, 1].set_title("regenerator_score(NSR)")
                    plt.colorbar(img, ax=axes[0, 1])
                    # MSE
                    mse = (
                        (
                            model.regenerator_input.unsqueeze(1)
                            - model.regenerator_output
                        )
                        .pow(2)
                        .mean(-1)
                    )
                    corr = mse.mean(0).detach().cpu().numpy()
                    img = axes[1, 0].imshow(corr)
                    axes[1, 0].set_title("MSE")
                    plt.colorbar(img, ax=axes[1, 0])
                    # set diagonal to zero
                    mse_no_self_reconst = mse.clone()
                    mse_no_self_reconst[
                        :, torch.eye(mse_no_self_reconst.shape[1]).bool()
                    ] = 0
                    corr = mse_no_self_reconst.mean(0).detach().cpu().numpy()
                    img = axes[1, 1].imshow(corr)
                    axes[1, 1].set_title("MSE(NSR)")
                    plt.colorbar(img, ax=axes[1, 1])

                    wandb.log({"scores": wandb.Image(fig)}, step=step)
                    # close figure
                    plt.close(fig)

                    wandb.log(
                        {
                            "loss": running_loss / config.log_interval,
                            "regenerator_score": regenerator_score.mean().item(),
                            "regenerator_score(NSR)": regenerator_score_no_self_reconst.mean().item(),
                            "regenerator_score(POS)": regenerator_score[
                                regenerator_score > 0
                            ]
                            .sum()
                            .item()
                            / 1000,
                            "regenerator_score(NSR)(POS)": regenerator_score_no_self_reconst[
                                regenerator_score_no_self_reconst > 0
                            ]
                            .sum()
                            .item()
                            / 1000,
                            "MSE": mse.mean().item(),
                            "MSE(NSR)": mse_no_self_reconst.mean().item(),
                        },
                        step=step,
                    )
                    if regenerator_iterations > 0:
                        wandb.log(
                            {
                                "regenerator_loss": regenerator_running_loss
                                / config.log_interval,
                            },
                            step=step,
                        )
                    running_loss = 0.0
                    regenerator_running_loss = 0.0

                    test_outputs = model(test_images).cpu()
                    # reverse transforms
                    plot_outputs = test_outputs / 2 + 0.5
                    # make output prepared for plotting
                    plot_outputs -= plot_outputs.min()
                    plot_outputs /= plot_outputs.max()
                    wandb.log(
                        {
                            "Network_reconstruct": wandb.Image(
                                torchvision.utils.make_grid(plot_outputs, 10)
                            )
                        },
                        step=step,
                    )

                    for mask in range(model.patch**2 + 1):
                        model.mask = True
                        model.mask_idx = mask
                        test_outputs = model(test_images).cpu()
                        # reverse transforms
                        plot_outputs = test_outputs / 2 + 0.5
                        # make output prepared for plotting
                        plot_outputs -= plot_outputs.min()
                        plot_outputs /= plot_outputs.max()
                        wandb.log(
                            {
                                f"mask_reconstruct_{mask}": wandb.Image(
                                    torchvision.utils.make_grid(plot_outputs, 10)
                                )
                            },
                            step=step,
                        )


if __name__ == "__main__":
    config_defaults = {
        "batch_size": 128,
        "dataset": "mnist",
        "epochs": 50,
        "lr": 0.001,
        "hidden": 400,
        "patch": 4,
        "is_cls_token": True,
        "log_interval": 100,
        "num_workers": 4,
    }
    seq_len = config_defaults["patch"] ** 2 + (
        1 if config_defaults["is_cls_token"] else 0
    )
    hidden = config_defaults["hidden"]

    # for regenerator in [
    #             "Autoencoder(hidden, 64)",
    #             "AutoencoderT(seq_len,8)",
    #             "AutoencoderH(seq_len*2, 16,2)",
    #             "AutoencoderH(seq_len*4, 16,4)",
    #             "AutoencoderH(seq_len*10, 16,10)",
    #             "Autoencoder2D('fsfs', seq_len, hidden, 8, 64)",
    #             "Autoencoder2D('sffs', seq_len, hidden, 8, 64)",
    #             "Autoencoder2D('sfsf', seq_len, hidden, 8, 64)",
    #         ]:
    #     config = config_defaults.copy()
    #     config["regenerator"] = regenerator
    #     regenerator_model = eval(regenerator)
    #     for regenerator_iterations in [0, 1, 5, 20, 100]:
    #         config["regenerator_iterations"] = regenerator_iterations
    #         with wandb.init(config=config, project="attention-regenerator"):
    #             main(regenerator_model, regenerator_iterations)

    NNMF_iterations = 20
    regenerator = AutoNNMF(
        input_size=[seq_len, hidden],
        hidden_size=8,
        number_of_iterations=NNMF_iterations,
    )
    regenerator_iterations = 0

    config = config_defaults.copy()
    config["regenerator"] = "AutoNNMF(seq_len, hidden, NNMF_iterations)"
    config["regenerator_iterations"] = regenerator_iterations
    with wandb.init(config=config, project="attention-regenerator"):
        main(regenerator, regenerator_iterations)
