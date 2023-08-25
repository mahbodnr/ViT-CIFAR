import random
import string
from datetime import datetime

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.utilities import CombinedLoader

from autoaugment import CIFAR10Policy, SVHNPolicy
from criterions import LabelSmoothingCrossEntropyLoss, AutoencoderCrossEntropyLoss
from da import RandomCropPaste

aft_models = {
    "aftfull": "full",
    "aftsimple": "simple",
}


def get_layer_outputs(model, input):
    layer_outputs = {}

    def hook(module, input, output):
        layer_name = f"{module.__class__.__name__}_{module.parent_name}"
        layer_outputs[layer_name] = output.detach()

    # Add parent name attribute to each module
    for name, module in model.named_modules():
        module.parent_name = name

    # Register the hook to each layer in the model
    for module in model.modules():
        module.register_forward_hook(hook)

    # Pass the input through the model
    _ = model(input)

    # Remove the hooks and parent name attribute
    for module in model.modules():
        module._forward_hooks.clear()
        delattr(module, "parent_name")

    return layer_outputs


def get_criterion(args):
    if args.criterion == "ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(
                args.num_classes, smoothing=args.smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.criterion == "aece":
        criterion = AutoencoderCrossEntropyLoss(
            args.aece_l1_regularization, args.aece_l1_outputs
        )

    else:
        raise ValueError(f"{args.criterion}?")

    return criterion


def get_model(args):
    can_learn_unsupervised = False # It can not do unsupervised learning... until it can.
    if args.model_name == "vit":
        from vit import ViT

        net = ViT(
            args.in_c,
            args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
        )
    elif args.model_name in aft_models:
        from vit import AttentionFreeViT

        net = AttentionFreeViT(
            mode=aft_models[args.model_name],
            seq_len=args.seq_len,
            factorize=args.factorize,
            factorization_dimension=args.factorization_dimension,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            query=args.query,
            pos_emb=args.pos_emb,
        )
    elif args.model_name == "hamburger_attention":
        from vit import HamburgerAttentionViT

        net = HamburgerAttentionViT(
            burger_mode=args.burger_mode,
            seq_len=args.seq_len,
            depthwise=args.depthwise,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            query=args.query,
            pos_emb=args.pos_emb,
        )
    elif args.model_name == "hamburger":
        from vit import HamburgerViT

        net = HamburgerViT(
            burger_mode=args.burger_mode,
            seq_len=args.seq_len,
            depthwise=args.depthwise,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            pos_emb=args.pos_emb,
        )
    elif args.model_name.startswith("gnnmf"):
        from vit import GatedNNMFViT

        nnmf_type = args.model_name.split("_")[1]
        net = GatedNNMFViT(
            NNMF_type=nnmf_type,
            seq_len=args.seq_len,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            ffn_features=args.ffn_features,
            MD_iterations=args.md_iter,
            train_bases=args.train_md_bases,
            local_learning=args.local_learning,
            depthwise=args.depthwise,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            pos_emb=args.pos_emb,
        )
    elif args.model_name == "gmlp":
        from vit import GatedMLPViT

        net = GatedMLPViT(
            seq_len=args.seq_len,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            ffn_features=args.ffn_features,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            pos_emb=args.pos_emb,
        )

    elif args.model_name == "wgmlp":
        from vit import WeightGatedMLPViT

        net = WeightGatedMLPViT(
            seq_len=args.seq_len,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            ffn_features=args.ffn_features,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            pos_emb=args.pos_emb,
        )

    elif args.model_name == "lgcnn":
        from cnn import LocalGlobalCNN

        net = LocalGlobalCNN(
            weight_gated=False,
            num_layers=args.num_layers,
            in_c=args.in_c,
            num_classes=args.num_classes,
            n_channels=args.hidden,  # Number of channels in CNN model is equivalent to the hidden embedding size in ViT
            hidden_features=args.ffn_features,  # Number of hidden features in CNN model is equivalent to the ffn features in GMLP
            img_size=args.size,
            patch=args.patch,
            kernel_size=args.kernel_size,
            use_cls_token=args.is_cls_token,
            mlp_hidden=args.mlp_hidden,
            dropout=args.dropout,
            normalization=args.cnn_normalization,
            use_mlp=args.use_encoder_mlp,
        )
    elif args.model_name == "wlgcnn":
        from cnn import LocalGlobalCNN

        net = LocalGlobalCNN(
            weight_gated=True,
            num_layers=args.num_layers,
            in_c=args.in_c,
            num_classes=args.num_classes,
            n_channels=args.hidden,  # Number of channels in CNN model is equivalent to the hidden embedding size in ViT
            hidden_features=args.ffn_features,  # Number of hidden features in CNN model is equivalent to the ffn features in GMLP
            img_size=args.size,
            patch=args.patch,
            kernel_size=args.kernel_size,
            use_cls_token=args.is_cls_token,
            mlp_hidden=args.mlp_hidden,
            dropout=args.dropout,
            normalization=args.cnn_normalization,
            use_mlp=args.use_encoder_mlp,
        )
    elif args.model_name == "ae":
        from vit import AEViT

        net = AEViT(
            AE_type=args.ae_type,
            seq_len=args.seq_len,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            ffn_features=args.ffn_features,
            AE_hidden_features=args.ae_hidden_features,
            AE_hidden_seq_len= args.ae_hidden_seq_len,
            chunk= args.chunk,
            legacy_heads = args.legacy_heads,
            order_2d=args.order_2d,
            depthwise=args.depthwise,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            pos_emb=args.pos_emb,
        )
        can_learn_unsupervised = True
    elif args.model_name == "ae_baseline":
        from vit import BaselineAEViT

        net = BaselineAEViT(
            seq_len=args.seq_len,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            ffn_features=args.ffn_features,
            AE_hidden=args.ae_hidden,
            depthwise=args.depthwise,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            pos_emb=args.pos_emb,
        )

    elif args.model_name == "linear":
        from vit import LinearAttentionViT

        net = LinearAttentionViT(
            seq_len=args.seq_len,
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            ffn_features=args.ffn_features,
            encoder_mlp=args.use_encoder_mlp,
            mlp_hidden=args.mlp_hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            pos_emb=args.pos_emb,
        )

    elif args.model_name == "cnn_baseline":
        from cnn import BaselineCNN

        net = BaselineCNN(
            input_shape= (3, 32, 32),
            cnn_features=[32],
            ann_layers=[1024, 10],
        )
        
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net, can_learn_unsupervised


def get_transform(args):
    train_transform = []
    test_transform = []
    train_transform += [transforms.RandomCrop(size=args.size, padding=args.padding)]
    if args.dataset != "svhn":
        train_transform += [transforms.RandomHorizontalFlip()]

    if args.autoaugment:
        if args.dataset == "c10" or args.dataset == "c100":
            train_transform.append(CIFAR10Policy())
        elif args.dataset == "svhn":
            train_transform.append(SVHNPolicy())
        else:
            print(f"No AutoAugment for {args.dataset}")

    train_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ]
    if args.rcpaste:
        train_transform += [RandomCropPaste(size=args.size)]

    test_transform += [
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ]

    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform


def get_dataloader(args):
    root = "data"
    if args.semi_supervised:
        if args.dataset == "c10":
            from datasets import CIFAR10SS

            args.in_c = 3
            args.num_classes = 10
            args.size = 32
            args.padding = 4
            args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            train_transforms, test_transforms = get_transform(args)
            labeled_set = CIFAR10SS(
                root=root,
                split="label",
                download=args.download_data,
                transform=train_transforms,
                boundary=0,
            )
            unlabeled_set = CIFAR10SS(
                root=root,
                split="unlabel",
                download=args.download_data,
                transform=train_transforms,
                boundary=0,
            )
            test_set = CIFAR10SS(
                root=root,
                split="test",
                download=args.download_data,
                transform=test_transforms,
                boundary=0,
            )

        elif args.dataset == "c100":
            raise NotImplementedError(
                "CIFAR100 is not implemented yet for semi-supervised."
            )

        elif args.dataset == "svhn":
            raise NotImplementedError(
                "SVHN is not implemented yet for semi-supervised."
            )

        else:
            raise NotImplementedError(
                f"{args.dataset} is not implemented yet for semi-supervised."
            )

        train_dl = CombinedLoader(
            {
                "labeled": DataLoader(
                    labeled_set,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                ),
                "unlabeled": DataLoader(
                    unlabeled_set,
                    batch_size=args.batch_size,
                    shuffle=args.shuffle,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                ),
            }
        )
        test_dl = DataLoader(
            test_set,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

    else:
        if args.dataset == "c10":
            args.in_c = 3
            args.num_classes = 10
            args.size = 32
            args.padding = 4
            args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            train_transform, test_transform = get_transform(args)
            train_ds = torchvision.datasets.CIFAR10(
                root,
                train=True,
                transform=train_transform,
                download=args.download_data,
            )
            test_ds = torchvision.datasets.CIFAR10(
                root,
                train=False,
                transform=test_transform,
                download=args.download_data,
            )

        elif args.dataset == "c100":
            args.in_c = 3
            args.num_classes = 100
            args.size = 32
            args.padding = 4
            args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
            train_transform, test_transform = get_transform(args)
            train_ds = torchvision.datasets.CIFAR100(
                root,
                train=True,
                transform=train_transform,
                download=args.download_data,
            )
            test_ds = torchvision.datasets.CIFAR100(
                root,
                train=False,
                transform=test_transform,
                download=args.download_data,
            )

        elif args.dataset == "svhn":
            args.in_c = 3
            args.num_classes = 10
            args.size = 32
            args.padding = 4
            args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
            train_transform, test_transform = get_transform(args)
            train_ds = torchvision.datasets.SVHN(
                root,
                split="train",
                transform=train_transform,
                download=args.download_data,
            )
            test_ds = torchvision.datasets.SVHN(
                root,
                split="test",
                transform=test_transform,
                download=args.download_data,
            )

        else:
            raise NotImplementedError(f"{args.dataset} is not implemented yet.")

        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

    return train_dl, test_dl


def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}_{args.num_layers}l"
    if not args.query:
        experiment_name += "_nq"
    if not args.use_encoder_mlp:
        experiment_name += "_nem"
    if args.autoaugment:
        experiment_name += "_aa"
    if args.label_smoothing:
        experiment_name += "_ls"
    if args.rcpaste:
        experiment_name += "_rc"
    if args.cutmix:
        experiment_name += "_cm"
    if args.mixup:
        experiment_name += "_mu"
    if not args.is_cls_token:
        experiment_name += "_gap"

    experiment_name += f"_{random_string(5)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return experiment_name

random_string = lambda n: ''.join([random.choice(string.ascii_lowercase) for i in range(n)]) 

def get_experiment_tags(args):
    tags = [args.model_name]
    if not args.query:
        tags.append("no-query")
    if not args.use_encoder_mlp:
        tags.append("no-encoder-mlp")
    return tags
