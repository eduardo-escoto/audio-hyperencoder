from typing import Union, Optional
from pathlib import Path

from regex import compile as re_compile
from torch import Tensor, Generator
from lightning import LightningDataModule
from torchaudio import load as torchaudio_load
from torch.utils.data import Dataset, DataLoader, random_split

from .utils import group_paths_by_pattern, get_file_paths_by_pattern


class AudioDataset(Dataset):
    def __init__(
        self,
        file_paths: list[Path],
        lazy_load: bool = False,
    ):
        self.lazy_load: bool = lazy_load
        self.file_paths: list[Path] = list(file_paths)

        self.sample_rates: list[int] = []
        self.audio_tensors: Optional[list[Tensor]] = []
        self.valid_paths = []

        self.erred_idxs = []
        if not self.lazy_load:
            for idx, file_path in enumerate(self.file_paths):
                try:
                    self.__load_audio__(idx)
                except Exception as _:
                    print(f"Failed to load {file_path} at index {idx}")

        self.file_paths = self.valid_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        file_path = self.file_paths[idx]
        if self.lazy_load:
            try:
                audio, sample_rate = torchaudio_load(file_path)
                self.sample_rates.append(sample_rate)
                self.audio_tensors.append(audio)
                self.valid_paths.append(file_path)
                return audio
            except Exception as e:
                print(f"Failed to load {file_path} at index {idx}: {e}")
                self.erred_idxs.append(idx)
                return None
        else:
            return self.audio_tensors[idx]

    def __load_audio__(self, idx: int):
        file_path = self.file_paths[idx]
        audio, sample_rate = torchaudio_load(file_path)
        self.sample_rates.append(sample_rate)
        self.audio_tensors.append(audio)
        self.valid_paths.append(file_path)


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[Path, str],
        batch_size: int = 1,
        num_workers: int = 4,
        file_pattern: str = r".*\.wav$",
        group_pattern: Optional[str] = r"Track\d*",
        lazy_load: bool = False,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.file_pattern = re_compile(file_pattern)
        self.group_pattern = re_compile(group_pattern) if group_pattern else None
        self.lazy_load = lazy_load
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.file_paths = list(
            get_file_paths_by_pattern(self.data_dir, self.file_pattern)
        )

        if self.group_pattern is None:
            self.grouped_file_paths = {file.stem: [file] for file in self.file_paths}
        else:
            self.grouped_file_paths = group_paths_by_pattern(
                self.file_paths, self.group_pattern
            )

    def setup(self, stage: Optional[str] = None):
        self.file_paths = list(self.data_dir.glob("*.wav"))
        dataset = AudioDataset(self.file_paths, lazy_load=self.lazy_load)

        total_size = len(dataset)
        train_size = int(total_size * self.train_split)
        val_size = int(total_size * self.val_split)
        test_size = total_size - train_size - val_size

        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=Generator().manual_seed(self.seed),
            )

        if stage == "validate":
            _, self.val_dataset, _ = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=Generator().manual_seed(self.seed),
            )

        if stage == "test":
            _, _, self.test_dataset = random_split(
                dataset,
                [train_size, val_size, test_size],
                generator=Generator().manual_seed(self.seed),
            )

        if stage == "predict":
            self.prediction_dataset = dataset

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
            self.prediction_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
