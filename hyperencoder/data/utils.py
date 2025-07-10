from typing import Any
from pathlib import Path
from collections import defaultdict
from collections.abc import Generator

from regex import Pattern
from torch import Tensor, stack, squeeze
from lightning import LightningDataModule

from hyperencoder.datamodels import DataConfig

from .latent import LatentLoadStrategy, PreEncodedLatentDataModule


def collate_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
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
    directory: Path | str, filename_pattern: Pattern[str]
) -> Generator[Path, None, None]:
    search_dir = Path(directory) if isinstance(directory, str) else directory

    for file in search_dir.rglob("*"):
        if filename_pattern.match(file.name):
            yield file


def group_paths_by_pattern(
    file_paths: list[Path], group_pattern: Pattern[str]
) -> dict[str, list[Path]]:
    group_dict: dict[str, list[Path]] = defaultdict(list)

    for file_path in file_paths:
        search_res = group_pattern.search(str(file_path))
        if search_res is not None:
            group_key = search_res.group()
            group_dict[group_key].append(file_path)
        else:
            raise FileNotFoundError()

    return group_dict


def create_datamodule_from_config(config: DataConfig) -> LightningDataModule:
    """Create a Lightning DataModule from a Pydantic data configuration.

    Args:
        config: DataConfig containing all data loading parameters

    Returns:
        Configured LightningDataModule instance

    Examples:
        >>> from hyperencoder.datamodels import DataConfig
        >>> config = DataConfig()  # Uses all defaults
        >>> datamodule = create_datamodule_from_config(config)
        >>>
        >>> # Or with custom parameters
        >>> config = DataConfig(batch_size=64, num_workers=16)
        >>> datamodule = create_datamodule_from_config(config)
    """
    # Convert loading strategy from string to enum
    loading_strategy = LatentLoadStrategy(config.loading_strategy)

    if config.dataset_type == "latents_for_hyperencoder":
        assert config.datasets is not None and len(config.datasets) > 0, (
            "Dataset entries must be specified for latents_for_hyperencoder"
        )

        if config.split_type == "auto":
            configs = []
            for dataset_entry in config.datasets:
                d_config = {"path": str(dataset_entry.path)}
                configs.append(d_config)

            return PreEncodedLatentDataModule.from_single_dataset_splits(
                configs,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                random_seed=config.random_seed,
                loading_strategy=loading_strategy,
                persistent_workers=config.persistent_workers,
                crop_config=config.crop_config.model_dump()
                if config.crop_config
                else None,
                train_split_pct=config.train_split_pct,
                val_split_pct=config.val_split_pct,
                test_split_pct=config.test_split_pct,
            )
        else:
            # Manual split - handle differently since datasets need split assignment
            # For now, we'll implement this when we have examples of manual split usage
            raise NotImplementedError(
                "Manual split not yet implemented for Pydantic DataConfig"
            )
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")


def create_datamodule(
    dataset_type: str = "latents_for_hyperencoder",
    split_type: str = "auto",
    loading_strategy: str = "lazy",
    train_split_pct: float = 0.8,
    val_split_pct: float = 0.1,
    test_split_pct: float = 0.1,
    datasets: list[dict[str, Any]] | None = None,
    crop_config: dict[str, Any] | None = None,
    batch_size: int = 32,
    num_workers: int = 8,
    random_seed: int = 42,
    persistent_workers: bool = False,
) -> LightningDataModule:
    """Create a Lightning DataModule with programmatic parameters.

    This is a convenience function for users who want to create data modules
    programmatically without using configuration files. All parameters
    use the same defaults as defined in the DataConfig Pydantic model.

    Args:
        dataset_type: Type of dataset to load
        split_type: How to split the data into train/val/test sets
        loading_strategy: Strategy for loading data into memory
        train_split_pct: Percentage of data to use for training
        val_split_pct: Percentage of data to use for validation
        test_split_pct: Percentage of data to use for testing
        datasets: List of dataset configurations
        crop_config: Optional crop configuration
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducibility
        persistent_workers: Whether to keep workers alive between epochs

    Returns:
        Configured LightningDataModule instance

    Examples:
        >>> # Use all defaults
        >>> datamodule = create_datamodule()
        >>>
        >>> # Custom batch size
        >>> datamodule = create_datamodule(batch_size=64)
        >>>
        >>> # Custom datasets
        >>> datamodule = create_datamodule(
        ...     datasets=[{"path": "/path/to/data"}]
        ... )
    """
    # Create a DataConfig with the provided parameters
    config_dict: dict[str, Any] = {
        "dataset_type": dataset_type,
        "split_type": split_type,
        "loading_strategy": loading_strategy,
        "train_split_pct": train_split_pct,
        "val_split_pct": val_split_pct,
        "test_split_pct": test_split_pct,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "random_seed": random_seed,
        "persistent_workers": persistent_workers,
    }

    if datasets is not None:
        from hyperencoder.datamodels import DatasetEntry

        config_dict["datasets"] = [DatasetEntry(path=d["path"]) for d in datasets]

    if crop_config is not None:
        config_dict["crop_config"] = crop_config

    # Create DataConfig and delegate to config-based factory
    data_config = DataConfig(**config_dict)
    return create_datamodule_from_config(data_config)
