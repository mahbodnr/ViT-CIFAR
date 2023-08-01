from datasets import CIFAR10SS, CIFAR100SS


class SemiSupervisedDataset(LightningDataModule):
    def __init__(
        self,
        dataset_name,
        root="./data",
        train_transforms=None,
        train_target_transforms=None,
        val_transforms=None,
        val_target_transforms=None,
        test_transforms=None,
        test_target_transforms=None,
        predict_transforms=None,
        predict_target_transforms=None,
        download=False,
        train_batch_size=32,
        val_batch_size=128,
        test_batch_size=128,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        boundary=0,
        **kwargs
    ):
        super().__init__()
        if dataset_name.lower() == "cifar10":
            dataset = CIFAR10SS
        elif dataset_name.lower() == "cifar100":
            dataset = CIFAR100SS
        else:
            raise NotImplementedError("Dataset not implemented")

    # Define datasets
    self.labeled_set = dataset(
        root=root,
        split="labeled",
        download=download,
        transform=train_transforms,
        target_transform=train_target_transforms,
        boundary=boundary,
    )
    self.unlabeled_set = CIFAR10(
        root=root,
        split="unlabeled",
        download=download,
        transform=train_transforms,
        target_transform=train_target_transforms,
        boundary=boundary,
    )
    self.val_set = CIFAR10(
        root=root,
        split="test",
        download=download,
        transform=val_transforms,
        target_transform=val_target_transforms,
        boundary=boundary,
    )
    self.test_set = CIFAR10(
        root=root,
        split="test",
        download=download,
        transform=test_transforms,
        target_transform=test_target_transforms,
        boundary=boundary,
    )
    self.predict_set = CIFAR10(
        root=root,
        split="test",
        download=download,
        transform=predict_transforms,
        target_transform=predict_target_transforms,
        boundary=boundary,
    )

    # Define dataloaders
    self.labeled_dataloader = DataLoader(
        self.labeled_set,
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    self.unlabeled_dataloader = DataLoader(
        self.unlabeled_set,
        batch_size=train_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    self.val_dataloader = DataLoader(
        self.val_set,
        batch_size=val_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    self.test_dataloader = DataLoader(
        self.test_set,
        batch_size=test_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    self.predict_dataloader = DataLoader(
        self.predict_set,
        batch_size=test_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


    def train_dataloader(self):
        return {"labeled": self.labeled_dataloader, "unlabeled": self.unlabeled_dataloader}

    def val_dataloader(self):
        return self.val_dataloader

    def test_dataloader(self):
        return self.test_dataloader

    def predict_dataloader(self):
        return self.predict_dataloader
