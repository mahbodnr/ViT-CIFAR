import comet_ml
import pytorch_lightning as pl
import warmup_scheduler
import torchvision
import torch
from torchview import draw_graph


from da import CutMix, MixUp
from utils import get_model, get_criterion, get_layer_outputs


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        self.criterion = get_criterion(hparams)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.0)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.0)
        self.log_image_flag = hparams.comet_api_key is None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.hparams.weight_decay,
        )
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr
        )
        self.scheduler = warmup_scheduler.GradualWarmupScheduler(
            self.optimizer,
            multiplier=1.0,
            total_epoch=self.hparams.warmup_epoch,
            after_scheduler=self.base_scheduler,
        )
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        img, label = batch
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_ = self.cutmix((img, label))
            elif self.hparams.mixup:
                if torch.rand(1).item() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = (
                        img,
                        label,
                        torch.zeros_like(label),
                        1.0,
                    )
            out = self.model(img)
            loss = self.criterion(out, label) * lambda_ + self.criterion(
                out, rand_label
            ) * (1.0 - lambda_)
        else:
            out = self(img)
            loss = self.criterion(out, label)

        if not self.log_image_flag and not self.hparams.dry_run:
            self.log_image_flag = True
            self._log_image(img.clone().detach().cpu())

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)
        return loss

    def on_train_epoch_end(self):
        self.log(
            "lr", self.optimizer.param_groups[0]["lr"], on_epoch=True,
        )
        # check if there is any nan value in model parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"[ERROR] {name} has nan value. Training stopped.")
        # log weights and layer outputs
        try:
            if not self.hparams.dry_run:
                # log the output of each layer
                layer_outputs = get_layer_outputs(self.model, self.hparams.sample_input_data)
                for name, output in layer_outputs.items():
                    self.logger.experiment.log_histogram_3d(
                        output.detach().cpu().numpy(),
                        name=name + ".output",
                        epoch=self.current_epoch,
                    )
                # log weights
                for name, param in self.model.named_parameters():
                    self.logger.experiment.log_histogram_3d(
                        param.detach().cpu().numpy(),
                        name=name,
                        epoch=self.current_epoch,
                    )        
        except Exception as e:
            print(f"\n[ERROR] Failed to log weights and gradients. {e}")

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.criterion(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def _log_image(self, image):
        grid = torchvision.utils.make_grid(image, nrow=4)
        self.logger.experiment.log_image(grid.permute(1, 2, 0))
        draw_graph(
            self.model,
            graph_name=self.hparams.experiment_name,
            input_size=self.hparams.input_size,
            expand_nested=True,
            save_graph=True,
            directory="imgs",
        )
        self.logger.experiment.log_image(f"imgs/{self.hparams.experiment_name}.gv.png")
