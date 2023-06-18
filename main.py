import argparse

from network import Net
import torch
import pytorch_lightning as pl
import numpy as np

from utils import get_dataset, get_experiment_name

parser = argparse.ArgumentParser()
parser.add_argument("--comet-api-key", help="API Key for Comet.ml")
parser.add_argument(
    "--dataset", default="c10", type=str, choices=["c10", "c100", "svhn"]
)
parser.add_argument(
    "--model-name", default="vit", type=str, choices=["vit", "aftfull", "aftsimple", "hamburger", "hamburger_attention"]
)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
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
parser.add_argument("--criterion", default="ce")
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=7, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument("--mlp-hidden", default=384 * 4, type=int)
parser.add_argument("--factorize", action="store_true")
parser.add_argument("--burger-mode", default="V1", type=str, choices=["V1", "V2", "V2+"])
parser.add_argument("--factorization-dimension", default=32, type=int)
parser.add_argument("--off-cls-token", action="store_false", dest="is_cls_token")
parser.add_argument(
    "--matmul-precision",
    default="medium",
    type=str,
    choices=["medium", "high", "highest"],
)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--project-name", default="VisionTransformer")
args = parser.parse_args()

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

if args.mlp_hidden != args.hidden * 4:
    print(
        f"[INFO] In original paper, mlp_hidden(CURRENT:{args.mlp_hidden}) is set to: {args.hidden*4}(={args.hidden}*4)"
    )

train_ds, test_ds = get_dataset(args)
train_dl = torch.utils.data.DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
test_dl = torch.utils.data.DataLoader(
    test_ds,
    batch_size=args.eval_batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
)


if __name__ == "__main__":
    experiment_name = get_experiment_name(args)
    print(experiment_name)
    if args.comet_api_key:
        print("[INFO] Log with Comet.ml!")
        logger = pl.loggers.CometLogger(
            api_key=args.comet_api_key,
            save_dir="logs",
            project_name=args.project_name,
            experiment_name=experiment_name,
        )
        refresh_rate = 0
    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(save_dir="logs", name=experiment_name)
        refresh_rate = 1
    net = Net(args)
    trainer = pl.Trainer(
        precision=args.precision,
        fast_dev_run=args.dry_run,
        accelerator="auto",
        devices=args.gpus if args.gpus else "auto",
        benchmark=args.benchmark,
        logger=logger,
        max_epochs=args.max_epochs,
    )
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)
    if not args.dry_run:
        model_path = f"weights/{experiment_name}.pth"
        torch.save(net.state_dict(), model_path)
        if args.comet_api_key:
            logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)
