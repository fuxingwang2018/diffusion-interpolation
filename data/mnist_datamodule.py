from __future__ import annotations
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str = "./data",
        batch_size: int = 256,
        num_workers: int = 8,
        pin_memory: bool = True,
        val_split: float = 0.1,
        download: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.download = download

        self.train_set = None
        self.val_set = None

        self.mean = 0.1307
        self.std = 0.3081
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((self.mean,), (self.std,)),
            ]
        )

    def prepare_data(self) -> None:
        MNIST(self.root, train=True, download=self.download)
        MNIST(self.root, train=False, download=self.download)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            full = MNIST(self.root, train=True, transform=self.transform)
            n_total = len(full)
            n_val = int(self.val_split * n_total)
            n_train = n_total - n_val
            self.train_set, self.val_set = random_split(full, [n_train, n_val])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
