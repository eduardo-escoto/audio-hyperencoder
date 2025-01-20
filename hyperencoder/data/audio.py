from typing import Union, Optional
from pathlib import Path
from collections.abc import Callable

from regex import Pattern
from regex import compile as re_compile
from torch import Tensor
from lightning import LightningDataModule
from torchaudio import load as torchaudio_load
from torch.utils.data import Dataset, DataLoader

from .utils import group_paths_by_pattern, get_file_paths_by_pattern


class AudioDataset(Dataset):
    def __init__(
        self,
        file_paths: list[Path],
        lazy_load: bool = False,
        preprocessor: Optional[Callable[[Tensor, int], Tensor]] = None,
    ):
        self.lazy_load: bool = False  # lazy_load
        self.file_paths: list[Path] = list(file_paths)
        self.preprocessor: Optional[Callable[[Tensor, int], Tensor]] = preprocessor

        self.sample_rates: list[int] = []
        self.audio_tensors: Optional[Tensor] = []
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

    def __getitem__(self, idx: int) -> Tensor:
        if self.lazy_load:
            return self.__load_audio__(idx)
        else:
            return self.audio_tensors[idx], self.sample_rates[idx]

    def __load_audio__(self, idx: int) -> Tensor:
        waveform, sample_rate = AudioDataset.load_audio(
            self.file_paths[idx], self.preprocessor
        )

        self.audio_tensors.append(waveform)
        self.sample_rates.append(sample_rate)
        self.valid_paths.append(self.file_paths[idx])

        return waveform, sample_rate, self.file_paths[idx]

    @staticmethod
    def load_audio(
        file_path: Path, preprocessor: Optional[Callable[[Tensor, int], Tensor]] = None
    ) -> tuple[Tensor, int]:
        waveform, sample_rate = torchaudio_load(file_path)

        if preprocessor is not None:
            waveform = preprocessor(waveform, sample_rate)

        return waveform, sample_rate


# class AudioTensorGroupType(Enum):
#     FILE = 1
#     DIRECTORY = 2


class AudioDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[Path, str],
        # tensor_group_type: AudioTensorGroupType,
        batch_size: int = 1,
        num_workers: int = 41,
        file_pattern: str = r".*\.wav$",
        group_pattern: Optional[str] = r"Track\d*",
        lazy_load: bool = True,
    ):
        super().__init__()

        # self.tensor_group_type: AudioTensorGroupType = tensor_group_type
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers
        self.data_dir: Path = data_dir if type(data_dir) is Path else Path(data_dir)
        self.file_pattern: Pattern = re_compile(file_pattern)
        self.group_pattern: Optional[Pattern] = (
            re_compile(group_pattern) if group_pattern is not None else None
        )

        self.lazy_load: bool = lazy_load

        self.file_paths = list(
            get_file_paths_by_pattern(self.data_dir, self.file_pattern)
        )

        if self.group_pattern is None:
            self.grouped_file_paths = {file.stem: [file] for file in self.file_paths}
        else:
            self.grouped_file_paths = group_paths_by_pattern(
                self.file_paths, self.group_pattern
            )

    def setup(self, stage: str):
        # only implementing predict for now
        if stage == "predict":
            self.prediction_dataset = AudioDataset(self.file_paths, lazy_load=False)
            self.file_paths = self.prediction_dataset.file_paths

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        return DataLoader(
            self.prediction_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
