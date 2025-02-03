from typing import Union, Optional
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from safetensors.torch import load_file
from stable_audio_tools.data.dataset import PreEncodedDataset, LocalDatasetConfig

"""
Not going to use........
just realized stable-audio-tools has a pre-encoded dataset class. 
"""


class LatentDataset(Dataset):
    def __init__(self, file_paths: list[Path], lazy_load: bool = False):
        self.file_paths = file_paths
        self.lazy_load = lazy_load
        self.cache = {}

        if not self.lazy_load:
            self.cache = {
                file_path: load_file(file_path)["mix"] for file_path in self.file_paths
            }

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        if file_path in self.cache:
            return self.cache[file_path]
        else:
            latents = load_file(file_path)["mix"]
            if self.lazy_load:
                self.cache[file_path] = latents
            return latents


class LatentDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[Path, str],
        batch_size: int = 1,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        lazy_load: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.lazy_load = lazy_load

    def setup(self, stage: Optional[str] = None):
        # self.file_paths = list(self.data_dir.glob("*.safetensors"))
        # dataset = LatentDataset(self.file_paths, lazy_load=self.lazy_load)
        dataset = PreEncodedDataset(
            [LocalDatasetConfig("latents", str(self.data_dir))],
            latent_crop_length=1024,
            latent_extension="safetensors",
            read_metadata=False,
            random_crop=True,
        )

        total_size = len(dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size

        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset, _ = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "validate":
            _, self.val_dataset, _ = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "test":
            _, _, self.test_dataset = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if stage == "predict":
            self.predict_dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
