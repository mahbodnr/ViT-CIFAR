import os
import argparse
from pprint import pprint

from network import Net
import torch
import pytorch_lightning as pl
import numpy as np

from utils import get_dataloader, get_experiment_name

parser = argparse.ArgumentParser()
parser.add_argument(
    "--comet-api-key", help="API Key for Comet.ml", dest="_comet_api_key"
)
parser.add_argument(
    "--dataset", default="c10", type=str, choices=["c10", "c100", "svhn"]
)
parser.add_argument(
    "--model-name",
    default="ae",
    type=str,
    choices=[
        "vit",
        "aftfull",
        "aftsimple",
        "hamburger",
        "hamburger_attention",
        "gnnmf_ham",
        "gnnmf_sbs",
        "gnnmf_sbsed",
        "gmlp",
        "wgmlp",
        "lgcnn",
        "wlgcnn",
        "ae",
        "ae_baseline",
        "linear",
    ],
)
parser.add_argument("--semi-supervised", action="store_true")
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=256, type=int)
parser.add_argument(
    "--optimizer", default="adam", type=str, choices=["adam", "sgd", "madam"]
)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--lr-nnmf", default=1e-2, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_false", dest="benchmark")
parser.add_argument("--max-epochs", default=100, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default="16-mixed", type=str)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--criterion", default="ce", type=str, choices=["ce", "aece"])
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument(
    "--depthwise",
    action="store_true",
    help="Apply depthwise operation in Matrix Decomposition (MD). This is equal to transpose the input matrix for MD.",
)
parser.add_argument(
    "--md-iter",
    default=7,
    type=int,
    help="Number of iterations in Matrix Decomposition (MD).",
)
parser.add_argument(
    "--train-md-bases",
    action="store_true",
    help="Train Matrix Decomposition (MD) bases. If False, generates random bases for each forward pass.",
)
parser.add_argument(
    "--local-learning",
    action="store_true",
    help="Enables local learning rule for SbS type NNMF instead of error backpropagation.",
)
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=1, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument(
    "--ffn-features",
    default=384 * 2,
    type=int,
    help="Number of featurtes in hidden part of the Gated models before spliting into two parts.",
)
parser.add_argument("--mlp-hidden", default=384, type=int)
# In original paper, mlp_hidden is set to: hidden*4
parser.add_argument(
    "--no-encoder-mlp",
    action="store_false",
    dest="use_encoder_mlp",
    help="Disable MLP in encoder blocks.",
)
parser.add_argument(
    "--kernel-size",
    default=1,
    type=int,
    help="Kernel size in Local-Global CNN model. Kernel-size=1 is similar to linear layers in ViT.",
)
parser.add_argument(
    "--unsupervised-steps",
    default=0,
    type=int,
    help="Number of unsupervised steps in a feedforward pass. Only applicable to models  that support unsupervised learning. This is different from semi-supervised learning.",
)
parser.add_argument("--mask-type", default="zeros", type=str, choices=["zeros", "random"])
parser.add_argument("--use-nnmf-layers", action="store_true")
parser.add_argument("--nnmf-local-learning", action="store_true")
parser.add_argument("--nnmf-scale-grade", action="store_true")
parser.add_argument("--chunk", action="store_true", help="Whether to chunk the input of the AE Attention models.")
parser.add_argument("--legacy-heads", action="store_true", help="Use legacy heads code in AEViT.")
parser.add_argument("--ae-type", default="simple", type=str, choices=["simple", "transpose", "heads", "2d"])
parser.add_argument(
    "--ae-hidden-features", default=128, type=int, help="Autoencoder hidden size in features dimension."
)
parser.add_argument("--ae-hidden-seq-len", default=8, type=int, help="Autoencoder hidden size in sequence dimension.")
parser.add_argument("--order-2d", type=str, default="sfsf", choices=["sfsf", "sffs"])
parser.add_argument("--ae-transpose", action="store_true", dest="AE_transpose")
parser.add_argument("--cnn-normalization", default="layer_norm", type=str)
parser.add_argument("--factorize", action="store_true")
parser.add_argument("--no-query", action="store_false", dest="query")
parser.add_argument("--no-pos-emb", action="store_false", dest="pos_emb")
parser.add_argument(
    "--burger-mode", default="V1", type=str, choices=["V1", "V2", "V2+", "Gated"]
)
parser.add_argument("--factorization-dimension", default=32, type=int)
parser.add_argument("--off-cls-token", action="store_false", dest="is_cls_token")
parser.add_argument(
    "--matmul-precision",
    default="medium",
    type=str,
    choices=["medium", "high", "highest"],
)
parser.add_argument("--log-gradients", action="store_true")
parser.add_argument("--log-gradients-interval", default=250, type=int)
parser.add_argument("--no-log-weights", action="store_false", dest="log_weights")
parser.add_argument("--model-summary-depth", default=-1, type=int)
parser.add_argument("--tags", default="", type=str, help="Comma separated tags.")
parser.add_argument("--seed", default=2045, type=int)  # Singularity is near
parser.add_argument("--project-name", default="Rethinking-Transformers", type=str)
parser.add_argument("--nnmf_learning_rate_threshold_w", default=1e-3, type=float)
parser.add_argument(
    "--aece_l1_regularization",
    default=0.0,
    type=float,
    help="L1 regularization for AutoencoderCrossEntropyLoss",
)
parser.add_argument(
    "--aece_l1_outputs",
    action="store_true",
    help="L1 regularization for all layer outputs in AutoencoderCrossEntropyLoss",
)
parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
parser.add_argument("--allow-download", action="store_true", dest="download_data")
args = parser.parse_args()

# args.semi_supervised = True #DELETE THIS LINE!!

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_float32_matmul_precision(args.matmul_precision)
args.gpus = torch.cuda.device_count()
args.num_workers = 4 * args.gpus if args.gpus else 8
if not args.gpus:
    args.precision = 32

args.num_classes = {
    "c10": 10,
    "c100": 100,
    "svhn": 10,
}[args.dataset]
args.seq_len = args.patch**2 + 1 if args.is_cls_token else args.patch**2

train_dl, test_dl = get_dataloader(args)
if args.semi_supervised:
    args._sample_input_data = next(iter(train_dl))["unlabeled"][0][0:10].to(
        "cuda" if args.gpus else "cpu"
    )
else:
    args._sample_input_data = next(iter(train_dl))[0][0:10].to(
        "cuda" if args.gpus else "cpu"
    )

if __name__ == "__main__":
    pprint({k: v for k, v in vars(args).items() if not k.startswith("_")})
    experiment_name = get_experiment_name(args)
    args.experiment_name = experiment_name
    print(f"Experiment: {experiment_name}")
    if args._comet_api_key:
        print("[INFO] Log with Comet.ml!")
        logger = pl.loggers.CometLogger(
            api_key=args._comet_api_key,
            save_dir="logs",
            project_name=args.project_name,
            experiment_name=experiment_name,
        )
    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(save_dir="logs", name=experiment_name)
    net = Net(args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=experiment_name + "-{epoch:03d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(
        precision=args.precision,
        fast_dev_run=args.dry_run,
        accelerator="auto",
        devices=args.gpus if args.gpus else "auto",
        benchmark=args.benchmark,
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        enable_model_summary=False,  # Implemented seperately inside the Trainer
    )
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)

    # Save model
    save_models_dir = "models"
    save_model_path = os.path.join(save_models_dir, experiment_name + ".ckpt")
    trainer.save_checkpoint(save_model_path)
    print(f"Model saved to {save_model_path}")
    # add model to comet
    if args._comet_api_key:
        urls = logger.experiment.log_model(
            experiment_name, save_model_path, overwrite=True
        )
        print(f"Model saved to comet: {urls}")