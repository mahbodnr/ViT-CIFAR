import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from autoaugment import CIFAR10Policy, SVHNPolicy
from criterions import LabelSmoothingCrossEntropyLoss
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
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion


def get_model(args):
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
            seq_len=args.patch**2 + 1 if args.is_cls_token else args.patch**2,
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
            seq_len=args.patch**2 + 1 if args.is_cls_token else args.patch**2,
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
            seq_len=args.patch**2 + 1 if args.is_cls_token else args.patch**2,
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
    elif args.model_name == "gnnmf":
        from vit import GatedNNMFViT

        net = GatedNNMFViT(
            in_c=args.in_c,
            num_classes=args.num_classes,
            img_size=args.size,
            patch=args.patch,
            dropout=args.dropout,
            num_layers=args.num_layers,
            hidden=args.hidden,
            ffn_features=args.ffn_features,
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
            seq_len=args.patch**2 + 1 if args.is_cls_token else args.patch**2,
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
    else:
        raise NotImplementedError(f"{args.model_name} is not implemented yet...")

    return net


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


def get_dataset(args):
    root = "data"
    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes = 10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(
            root, train=True, transform=train_transform, download=True
        )
        test_ds = torchvision.datasets.CIFAR10(
            root, train=False, transform=test_transform, download=True
        )

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes = 100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(
            root, train=True, transform=train_transform, download=True
        )
        test_ds = torchvision.datasets.CIFAR100(
            root, train=False, transform=test_transform, download=True
        )

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes = 10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(
            root, split="train", transform=train_transform, download=True
        )
        test_ds = torchvision.datasets.SVHN(
            root, split="test", transform=test_transform, download=True
        )

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")

    return train_ds, test_ds


def get_experiment_name(args):
    experiment_name = f"{args.model_name}_{args.dataset}"
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
    return experiment_name


class Args:
    pass