import comet_ml
import pytorch_lightning as pl
import warmup_scheduler
import torchvision
import torch
from torchview import draw_graph

from da import CutMix, MixUp
from utils import get_model, get_criterion, get_layer_outputs, get_experiment_tags
from vit import ViT, AEViT
from cnn import LocalGlobalCNN
from criterions import AutoencoderCrossEntropyLoss

class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        self.save_hyperparameters(ignore=[key for key in self.hparams.keys() if key[0] == "_"])
        hparams._nnmf_params = {
                "number_of_iterations": hparams.md_iter,
                # epsilon_0: float = 1.0,
                # weight_noise_range: list[float] = [0.0, 1.0],
                "w_trainable": hparams.train_md_bases,
                # "device": self.device,
                "device": torch.device("cuda"),
                "default_dtype": torch.float32,
                # layer_id: int = -1,
                "local_learning": hparams.nnmf_local_learning,
                # output_layer: bool = False,
                # skip_gradient_calculation: bool = False,
                "keep_last_grad_scale": hparams.nnmf_scale_grade,
                "disable_scale_grade": not hparams.nnmf_scale_grade,
        }
        self.model, self.model_can_learn_unsupervised = get_model(hparams)
        if (
            not self.model_can_learn_unsupervised
            and self.hparams.unsupervised_steps > 0
        ):
            print(
                "[WARNING] Unsuperivsed steps is set to more than 0 but the model cannot learn unsupervised. Unsupervised steps will be ignored."
            )
        self.criterion = get_criterion(hparams)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.0)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.0)
        self.log_image_flag = hparams._comet_api_key is None
        # If there is any NNMF module in the model
        self.nnmf_layers = []
        # add all nnmf modules to nnmf_layers
        for name, module in self.model.named_modules():
            if "nnmf" in name.lower() or hasattr(module, "_weights"):
                self.nnmf_layers.append(module)
        print(f"Number of NNMF layers (modules): {len(self.nnmf_layers)}")

    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, out, label):
        if isinstance(self.criterion, AutoencoderCrossEntropyLoss):
            AE_modules = [
                module for module in self.model.modules() if isinstance(module, AEViT)
            ]
            loss = self.criterion(out, label, AE_modules)
        else:
            loss = self.criterion(out, label)

        return loss

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr,
                momentum=self.hparams.beta1,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "madam":
            from nnmf.optimizer import Madam

            nnmf_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if "nnmf" in name.lower() or "_weights" in name.lower():
                    nnmf_params.append(param)
                else:
                    other_params.append(param)
            print(
                f"Optimizer params:\nNNMF parameters: {len(nnmf_params)}\nOther parameters: {len(other_params)}"
            )
            self.optimizer = Madam(
                params=[
                    {"params": other_params, "lr": self.hparams.lr},
                    {
                        "params": nnmf_params,
                        "lr": self.hparams.lr_nnmf,
                        "nnmf": True,
                        "foreach": False,
                    },
                ],
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.hparams.optimizer}")
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

    def on_fit_start(self):
        summary = pl.utilities.model_summary.ModelSummary(
            self, max_depth=self.hparams.model_summary_depth
        )
        if hasattr(self.logger.experiment, "log_asset_data"):
            self.logger.experiment.log_asset_data(
                str(summary), file_name="model_summary.txt"
            )
        print(summary)

    def on_train_start(self):
        self.log(
            "trainable_params",
            float(sum(p.numel() for p in self.model.parameters() if p.requires_grad)),
        )
        self.log("total_params", float(sum(p.numel() for p in self.model.parameters())))
        # Get tags from hparams
        tags = self.hparams.tags.split(",")
        tags = [tag.strip() for tag in tags]
        if not (tags[0] == "" and len(tags) == 1):
            self.logger.experiment.add_tags(tags)
        # Get tags from experiment hyperparameters
        if hasattr(self.logger.experiment, "add_tags"):
            self.logger.experiment.add_tags(get_experiment_tags(self.hparams))

    def supervised_step(self, img, label):
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

            out = self(img)
            loss = (self.calculate_loss(out, label) * lambda_) + (
                self.calculate_loss(out, rand_label) * (1.0 - lambda_)
            )
        else:
            out = self(img)
            loss = self.calculate_loss(out, label)

        if self.model_can_learn_unsupervised and self.hparams.unsupervised_steps > 0:
            unsupervised_loss = 0
            for _ in range(self.hparams.unsupervised_steps):
                unsupervised_loss += self.model.unsupervised_update()
                # with torch.no_grad():
                #     out = self(img)
            self.log("unsupervised_loss", unsupervised_loss)
            
            # Do one more forward pass with the new autoencoder parameters
            # out = self(img)
            # loss = self.calculate_loss(out, label)

        return out, loss

    def unsupervised_step(self, unlabeled_batch):
        pass

    def training_step(self, batch, batch_idx):
        if self.hparams.semi_supervised:
            labeled_batch, unlabeled_batch = batch["labeled"], batch["unlabeled"]
            img, label = labeled_batch
            if self.model_can_learn_unsupervised:
                self.unsupervised_step(unlabeled_batch)
                out, loss = self.supervised_step(img, label)
            else:
                out, loss = self.supervised_step(img, label)
        else:
            img, label = batch
            out, loss = self.supervised_step(img, label)

        if not self.log_image_flag and not self.hparams.dry_run:
            self.log_image_flag = True
            self._log_image(img.clone().detach().cpu())

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)

        # DELETE later
        # if batch_idx == 0:
        #     self.log(
        #         "AE input mean", self.model.enc[0].attention.AE_input.mean()
        #     )
        #     self.log(
        #         "AE input std", self.model.enc[0].attention.AE_input.std()
        #     )


        return loss

    def on_train_epoch_end(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.log(f"lr_{i}", param_group["lr"], on_epoch=True)
        # check if there is any nan value in model parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"[ERROR] {name} has nan value. Training stopped.")
        # log weights and layer outputs
        if self.hparams.log_weights:
            if not self.hparams.dry_run and hasattr(self.logger.experiment, "log_histogram_3d"):
                # log the output of each layer
                layer_outputs = get_layer_outputs(
                    self.model, self.hparams._sample_input_data
                )
                for name, output in layer_outputs.items():
                    try:
                        self.logger.experiment.log_histogram_3d(
                            output.detach().cpu(),
                            name=name + ".output",
                            epoch=self.current_epoch,
                        )
                    except IndexError:
                        # Values closer than 1e-20 to zerro will lead to index error
                        positive_output = output[output > 0]
                        pos_min = (
                            positive_output.min().item()
                            if positive_output.numel() > 0
                            else float("inf")
                        )
                        negative_output = output[output < 0]
                        neg_min = (
                            abs(negative_output.max().item())
                            if negative_output.numel() > 0
                            else float("inf")
                        )
                        self.logger.experiment.log_histogram_3d(
                            output.detach().cpu(),
                            name=name + ".output",
                            epoch=self.current_epoch,
                            start=min(pos_min, neg_min),
                        )
                    except Exception as e:
                        raise e
                # log weights
                for name, param in self.model.named_parameters():
                    try:
                        self.logger.experiment.log_histogram_3d(
                            param.detach().cpu(),
                            name=name,
                            epoch=self.current_epoch,
                        )
                    except IndexError:
                        # Values closer than 1e-20 to zerro will lead to index error
                        positive_param = param[param > 0]
                        pos_min = (
                            positive_param.min().item()
                            if positive_param.numel() > 0
                            else float("inf")
                        )
                        negative_param = param[param < 0]
                        neg_min = (
                            abs(negative_param.max().item())
                            if negative_param.numel() > 0
                            else float("inf")
                        )
                        self.logger.experiment.log_histogram_3d(
                            param.detach().cpu(),
                            name=name,
                            epoch=self.current_epoch,
                            start=min(pos_min, neg_min),
                        )
                # log AE input
                for module in self.model.modules():
                    if hasattr(module, "AE_input"):
                        try:
                            self.logger.experiment.log_histogram_3d(
                                module.AE_input.detach().cpu(),
                                name=module.__class__.__name__ + ".AE_input",
                                epoch=self.current_epoch,
                            )
                        except IndexError:
                            # Values closer than 1e-20 to zerro will lead to index error
                            positive_input = module.AE_input[module.AE_input > 0]
                            pos_min = (
                                positive_input.min().item()
                                if positive_input.numel() > 0
                                else float("inf")
                            )
                            negative_input = module.AE_input[module.AE_input < 0]
                            neg_min = (
                                abs(negative_input.max().item())
                                if negative_input.numel() > 0
                                else float("inf")
                            )
                            self.logger.experiment.log_histogram_3d(
                                module.AE_input.detach().cpu(),
                                name=module.__class__.__name__ + ".AE_input",
                                epoch=self.current_epoch,
                                start=min(pos_min, neg_min),
                            )
                        except Exception as e:
                            raise e


    def on_before_optimizer_step(self, optimizer):
        if self.nnmf_layers:
            for layer in self.nnmf_layers:
                # if self.trainer.global_step == 0:
                #     layer.after_batch(True)
                # else:
                #     layer.after_batch(False)
                layer.update_pre_care()

        # log gradients once per epoch
        if (
            self.hparams.log_gradients
            and hasattr(self.logger.experiment, "log_histogram_3d")
            and not self.hparams.dry_run
            and self.trainer.global_step % self.hparams.log_gradients_interval == 0
        ):
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                try:
                    self.logger.experiment.log_histogram_3d(
                        param.grad.detach().cpu(),
                        name=name + ".grad",
                        epoch=self.current_epoch,
                        step=self.trainer.global_step,
                    )
                except IndexError:
                    # Values closer than 1e-20 to zerro will lead to index error
                    positive_grad = param.grad[param.grad > 0]
                    pos_min = (
                        positive_grad.min().item()
                        if positive_grad.numel() > 0
                        else float("inf")
                    )
                    negative_grad = param.grad[param.grad < 0]
                    neg_min = (
                        abs(negative_grad.max().item())
                        if negative_grad.numel() > 0
                        else float("inf")
                    )
                    self.logger.experiment.log_histogram_3d(
                        param.grad.detach().cpu(),
                        name=name + ".grad",
                        epoch=self.current_epoch,
                        step=self.trainer.global_step,
                        start=min(pos_min, neg_min),
                    )
                except Exception as e:
                    raise e

                # self.logger.experiment.log_metric(
                #     f"Max {name}.grad", param.grad.detach().cpu().max().item()
                # )

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.nnmf_layers:
            for layer in self.nnmf_layers:
                layer.update_after_care(
                    self.hparams.nnmf_learning_rate_threshold_w
                    / layer._number_of_input_neurons
                )

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.calculate_loss(out, label)
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
            input_size=image.shape,
            expand_nested=True,
            save_graph=True,
            directory="imgs",
        )
        self.logger.experiment.log_image(f"imgs/{self.hparams.experiment_name}.gv.png")
        if isinstance(self.model, ViT):
            draw_graph(
                self.model.enc[0],
                graph_name=self.hparams.experiment_name + "_encoder_block",
                input_size=(
                    1,
                    self.hparams.patch**2 + (1 if self.hparams.is_cls_token else 0),
                    self.hparams.hidden,
                ),
                expand_nested=True,
                save_graph=True,
                directory="imgs",
                depth=5,
            )
            self.logger.experiment.log_image(
                f"imgs/{self.hparams.experiment_name}_encoder_block.gv.png"
            )
        elif isinstance(self.model, LocalGlobalCNN):
            draw_graph(
                self.model.enc[0],
                graph_name=self.hparams.experiment_name + "_encoder_block",
                input_data=[
                    (
                        torch.randn(
                            1, self.model.n_channels, self.model.patch, self.model.patch
                        ),
                        torch.randn(
                            1,
                            self.model.n_channels,
                            self.model.kernel_size,
                            self.model.kernel_size,
                        ),
                    )
                ],
                expand_nested=True,
                save_graph=True,
                directory="imgs",
                depth=5,
            )
            self.logger.experiment.log_image(
                f"imgs/{self.hparams.experiment_name}_encoder_block.gv.png"
            )
        else:
            print("[WARNING] Failed to draw encoder graph.")

    def on_train_end(self):
        self.logger.experiment.end()
