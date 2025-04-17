from typing import Union
from pathlib import Path
from collections import defaultdict
from collections.abc import Generator

from regex import Pattern
from torch import Tensor, stack, squeeze
from lightning import LightningDataModule

from .latent import LatentLoadStrategy, PreEncodedLatentDataModule


def collate_dicts(dicts: list[dict]):
    dict_types = {key: type(value) for key, value in dicts[0].items()}
    out_dict = {}
    for key, t in dict_types.items():
        collated = [
            squeeze(dict_item[key]) if t is Tensor else dict_item[key]
            for dict_item in dicts
        ]
        if t is Tensor:
            collated = stack(collated)
        out_dict[key] = collated

    return out_dict


def get_file_paths_by_pattern(
    directory: Union[Path, str], filename_pattern: Pattern
) -> Generator[Path]:
    if type(directory) is str:
        directory = Path(directory)

    for file in directory.rglob("*"):
        if filename_pattern.match(file.name):
            yield file


def group_paths_by_pattern(
    file_paths: list[Path], group_pattern: Pattern
) -> dict[str, list[Path]]:
    group_dict: dict[str, list] = defaultdict(list)

    for file_path in file_paths:
        group_key = group_pattern.search(str(file_path)).group()
        group_dict[group_key].append(file_path)

    return group_dict


def create_datamodule_from_config(
    dataset_config: dict,
    batch_size=32,
    num_workers=4,
    random_seed=42,
    loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
    persistent_workers=True,
) -> LightningDataModule:
    dataset_type = dataset_config.get("dataset_type", "latents_for_hyperencoder")
    split_type = dataset_config.get(
        "split_type", "auto"
    )  # auto or manual (manual must define the type in the configs)

    cfg_loading_strategy = dataset_config.get("loading_strategy")
    if cfg_loading_strategy is not None:
        loading_strategy = LatentLoadStrategy(cfg_loading_strategy)

    if dataset_type == "latents_for_hyperencoder":
        dir_configs = dataset_config.get("datasets")

        assert dir_configs is not None, (
            'Directory configuration must be specified in datasets["dataset"]'
        )

        if split_type == "auto":
            configs = []
            split_pcts = {
                "train_split_pct": dataset_config.get("train_split_pct", 0.8),
                "val_split_pct": dataset_config.get("val_split_pct", 0.1),
                "test_split_pct": dataset_config.get("test_split_pct", 0.1),
            }

        elif split_type == "manual":
            split_dict = {"train": [], "val": [], "test": [], "pred": []}

        for dir_config in dir_configs:
            dir_path = dir_config.get("path")
            dir_dataset_type = dir_config.get("dataset_type")
            assert dir_path is not None, (
                "Path must be set for local audio directory configuration"
            )

            if split_type == "auto":
                configs.append(dir_path)
            elif split_type == "manual":
                split_dict[dir_dataset_type].append()

        if split_type == "auto":
            return PreEncodedLatentDataModule.from_single_dataset_splits(
                configs,
                batch_size=batch_size,
                num_workers=num_workers,
                random_seed=random_seed,
                loading_strategy=loading_strategy,
                persistent_workers=persistent_workers,
                **split_pcts,
            )
        elif split_type == "manual":
            return PreEncodedLatentDataModule.from_manual_splits(
                split_dict["train"],
                split_dict["val"],
                split_dict["test"],
                split_dict["pred"],
                batch_size=batch_size,
                num_workers=num_workers,
                loading_strategy=loading_strategy,
                persistent_workers=persistent_workers,
            )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
