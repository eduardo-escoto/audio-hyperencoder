from os import walk
from enum import Enum
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass

from torch import Generator
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from safetensors.torch import load_file

DEFAULT_FILE_SUFFIXES = [
    ".safetensors",
    ".json",
    "_reconstructed.wav",
    "_original_trimmed.wav",
]
DEFAULT_SUFFIX_KEYS = [
    "latents",
    "pre_encode_config",
    "reconstructed_audio",
    "original_audio",
]

DEFAULT_SUFFIX_MAPPING = dict(zip(DEFAULT_FILE_SUFFIXES, DEFAULT_SUFFIX_KEYS))


class LatentLoadStrategy(Enum):
    LAZY = "lazy"
    EAGER = "eager"
    LAZY_CACHED = "lazy_cached"


@dataclass(frozen=True)
class EncodedDirectoryInfo:
    root: str
    prefix: str
    latents: str
    pre_encode_config: str
    reconstructed_audio: str
    original_audio: str


"""
This is a dataset class that is used to load pre-encoded latent tensors
that were generated from the pre-encoding script. This is due to the specific 
folder structure, and file-naming conventions that were used to generate the
the ouptus. This is a bit of a hacky solution but ehh we balling brother.
"""


class PreEncodedLatentDataset(Dataset):
    def __init__(
        self,
        file_tuples: list[tuple[str, EncodedDirectoryInfo]],
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
    ):
        self.file_tuples = file_tuples
        self.loading_strategy = loading_strategy
        self.latent_dict = {}

        if loading_strategy == LatentLoadStrategy.EAGER:
            self.latent_dict = self.load_all(file_tuples)

    """
    This function gets the required filepaths from the pre-encoded
    base directory. The structure is determined by the outputs 
    from the pre-encoding script. This isn't the cleanest solution
    but it's the best I can do for now. The structure is as follows:
    - {prefix}_original_trimmed.wav
    - {prefix}_reconstructed.wav
    - {prefix}.json
    - {prefix}.safetensors

    The include stems parameter is used to include stems as part of the dataset,
    instead of just the overall mix.
    """

    @staticmethod
    def from_parent_dirs(
        parent_folders: list[str],
        prefix_filter: Optional[str] = None,
        suffix_mapping: Optional[dict[str, str]] = None,
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
    ):
        file_tuples = PreEncodedLatentDataset.collect_file_path_tuples(
            parent_folders, prefix_filter, suffix_mapping, loading_strategy
        )
        return PreEncodedLatentDataset(file_tuples)

    @staticmethod
    def collect_file_path_tuples(
        parent_folders: list[str],
        prefix_filter: Optional[str] = None,
        suffix_mapping: Optional[dict[str, str]] = None,
    ):
        if suffix_mapping is None:
            suffix_mapping = DEFAULT_SUFFIX_MAPPING

        if prefix_filter is not None:
            from re import compile

            prefix_filter_re = compile(prefix_filter)

        files_dicts = defaultdict(lambda: defaultdict(dict))
        tuples: list[tuple[str, EncodedDirectoryInfo]] = []

        def get_pref_from_suff(suff, file):
            return file.replace(suff, "")

        for path in parent_folders:
            for root, _, files in walk(path):
                for file in files:
                    for suff in suffix_mapping:
                        if file.endswith(suff):
                            pref = get_pref_from_suff(suff, file)
                            files_dicts[root][pref][suffix_mapping[suff]] = file

        for root, dir_dict in files_dicts.items():
            for pref, suff_dict in dir_dict.items():
                out_tuple = (
                    f"{root}/{suff_dict['latents']}",
                    EncodedDirectoryInfo(**{"root": root, "prefix": pref, **suff_dict}),
                )

                if prefix_filter is not None:
                    if prefix_filter_re.match(pref):
                        tuples.append(out_tuple)
                    else:
                        continue
                else:
                    tuples.append(out_tuple)

        return tuples

    def __len__(self):
        return len(self.file_tuples)

    @staticmethod
    def load_all(tuples):
        latent_dict = {}
        for idx in range(len(tuples)):
            latent, path, info = PreEncodedLatentDataset.load_item(tuples[idx])
            latent_dict[path] = {"latents": latent, "info": info}
        return latent_dict

    @staticmethod
    def load_item(latent_tuple):
        latents_path, info = latent_tuple
        latents_sf = load_file(latents_path)

        latents = latents_sf["latents"]
        info = {**info, **latents_sf}

        return latents, latents_path, info

    def __getitem__(self, idx):
        if self.loading_strategy == LatentLoadStrategy.EAGER:
            latent_path, info = self.file_tuples[idx]
            return self.latent_dict[latent_path]
        elif self.loading_strategy == LatentLoadStrategy.LAZY:
            latents, path, info = self.load_item(self.file_tuples[idx])
            return latents, info
        elif self.loading_strategy == LatentLoadStrategy.LAZY_CACHED:
            latent_path, info = self.file_tuples[idx]
            if latent_path in self.latent_dict:
                return self.latent_dict[latent_path]
            else:
                latents, path, info = self.load_item(self.file_tuples[idx])
                self.latent_dict[path] = {"latents": latents, "info": info}
                return self.latent_dict[path]


class PreEncodedLatentDataModule(LightningDataModule):
    def __init__(
        self,
        train_tuples: Optional[list[str, EncodedDirectoryInfo]],
        val_tuples: Optional[list[str, EncodedDirectoryInfo]],
        test_tuples: Optional[list[str, EncodedDirectoryInfo]],
        predict_tuples: Optional[list[str, EncodedDirectoryInfo]],
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_tuples = train_tuples
        self.val_tuples = val_tuples
        self.test_tuples = test_tuples
        self.predict_tuples = predict_tuples
        self.loading_strategy = loading_strategy
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def from_dirs_per_dataset(
        train_dirs: list[str],
        val_dirs: list[str],
        test_dirs: list[str],
        predict_dirs: list[str],
        batch_size: int = 32,
        num_workers: int = 4,
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
    ):
        train_tuples = PreEncodedLatentDataset.collect_file_path_tuples(train_dirs)
        val_tuples = PreEncodedLatentDataset.collect_file_path_tuples(val_dirs)
        test_tuples = PreEncodedLatentDataset.collect_file_path_tuples(test_dirs)
        predict_tuples = PreEncodedLatentDataset.collect_file_path_tuples(predict_dirs)

        return PreEncodedLatentDataModule(
            train_tuples=train_tuples,
            val_tuples=val_tuples,
            test_tuples=test_tuples,
            predict_tuples=predict_tuples,
            loading_strategy=loading_strategy,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @staticmethod
    def from_single_dataset_splits(
        datadirs: list[str],
        train_split_pct: float = 0.7,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.1,
        random_seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 4,
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
    ):
        latents_tuples = PreEncodedLatentDataset.collect_file_path_tuples(datadirs)

        train_tuples, val_tuples, test_tuples = random_split(
            latents_tuples,
            [
                int(len(latents_tuples) * train_split_pct),
                int(len(latents_tuples) * val_split_pct),
                int(len(latents_tuples) * test_split_pct),
            ],
            generator=Generator().manual_seed(random_seed),
        )

        return PreEncodedLatentDataModule(
            train_tuples=train_tuples,
            val_tuples=val_tuples,
            test_tuples=test_tuples,
            predict_tuples=latents_tuples,
            loading_strategy=loading_strategy,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = PreEncodedLatentDataset(
                self.train_tuples, loading_strategy=self.loading_strategy
            )
            self.val_dataset = PreEncodedLatentDataset(
                self.val_tuples, loading_strategy=self.loading_strategy
            )
        if stage == "validate":
            self.val_dataset = PreEncodedLatentDataset(
                self.val_tuples, loading_strategy=self.loading_strategy
            )
        if stage == "test":
            self.test_dataset = PreEncodedLatentDataset(
                self.test_tuples, loading_strategy=self.loading_strategy
            )
        if stage == "predict":
            self.predict_dataset = PreEncodedLatentDataset(
                self.predict_tuples, loading_strategy=self.loading_strategy
            )

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
