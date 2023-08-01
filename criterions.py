import torch
import torch.nn as nn


class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class AutoencoderCrossEntropyLoss(nn.Module):
    def __init__(self, l1_regularization=0.5, l1_all_layer_outputs=False):
        super(AutoencoderCrossEntropyLoss, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.l1_regularization = l1_regularization
        self.l1_all_layer_outputs = l1_all_layer_outputs

    def forward(self, pred, target, AE_modules):
        assert AE_modules, "No AEViT modules found in model"
        loss = 0
        for AE_module in AE_modules:
            for AE_hidden, AE_input, AE_output in zip(
                AE_module.AE_hidden(), AE_module.AE_inputs(), AE_module.AE_outputs()
            ):
                loss += self.sparseAutoencoderLoss(
                    AE_hidden,
                    AE_input,
                    AE_output,
                    self.l1_regularization,
                    self.l1_all_layer_outputs,
                )

        loss += self.CrossEntropyLoss(pred, target)

        return loss

    @staticmethod
    def sparseAutoencoderLoss(
        autoencoder_hidden, input, model_output, l1_regularization, l1_outputs
    ):
        mse_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        mse = mse_loss(model_output, input)
        l1 = 0
        if l1_outputs:
            l1 += l1_loss(autoencoder_hidden, torch.zeros_like(autoencoder_hidden))
            l1 += l1_loss(model_output, torch.zeros_like(model_output))
        l1 += l1_loss(model_output, input)

        return mse + l1_regularization * l1
