"""
Data utility functions for hyperencoder.

This module provides utility functions for creating data loaders and data modules
for hyperencoder training and evaluation.
"""

from typing import Any, Dict, List, Optional, Union
from omegaconf import DictConfig
from lightning import LightningDataModule


def create_datamodule_from_config(config: DictConfig) -> LightningDataModule:
    """Create a Lightning DataModule from a Hydra configuration.

    Args:
        config: DictConfig containing all data loading parameters

    Returns:
        Configured LightningDataModule instance

    Examples:
        >>> from omegaconf import DictConfig
        >>> config = DictConfig({"batch_size": 32, "num_workers": 4})
        >>> datamodule = create_datamodule_from_config(config)
        >>> print(f"Batch size: {datamodule.batch_size}")
    """
    # Import here to avoid circular imports
    from .latent import PreEncodedLatentDataModule
    
    # Create the datamodule with config parameters
    return PreEncodedLatentDataModule(
        train_tuples=None,  # Will be populated based on config
        val_tuples=None,
        test_tuples=None,
        predict_tuples=None,
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        # Add other parameters as needed
    )


def create_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    **kwargs,
):
    """Create a data loader for hyperencoder data.

    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle the data
        pin_memory: Whether to pin memory for GPU transfer
        persistent_workers: Whether to keep workers alive between epochs
        **kwargs: Additional arguments passed to the data loader

    Returns:
        Configured DataLoader instance

    Examples:
        >>> loader = create_dataloader("/path/to/dataset", batch_size=64)
        >>> print(f"Batch size: {loader.batch_size}")
    """
    from torch.utils.data import DataLoader
    from .latent import PreEncodedLatentDataset
    
    # Create dataset
    dataset = PreEncodedLatentDataset.from_parent_dirs([dataset_path], **kwargs)
    
    # Create data loader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def create_train_val_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2,
    random_seed: int = 42,
    **kwargs,
):
    """Create training and validation data loaders.

    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducible splits
        **kwargs: Additional arguments

    Returns:
        Tuple of (train_loader, val_loader)

    Examples:
        >>> train_loader, val_loader = create_train_val_dataloaders("/path/to/dataset")
        >>> print(f"Train batches: {len(train_loader)}")
        >>> print(f"Val batches: {len(val_loader)}")
    """
    from torch.utils.data import DataLoader, random_split
    from .latent import PreEncodedLatentDataset
    import torch
    
    # Set random seed for reproducible splits
    torch.manual_seed(random_seed)
    
    # Create dataset
    dataset = PreEncodedLatentDataset.from_parent_dirs([dataset_path], **kwargs)
    
    # Calculate split sizes
    total_size = len(dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    
    return train_loader, val_loader


def create_split_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
):
    """Create data loaders from separate train/validation datasets.

    Args:
        train_path: Path to the training dataset
        val_path: Path to the validation dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        **kwargs: Additional arguments

    Returns:
        Tuple of (train_loader, val_loader)

    Examples:
        >>> train_loader, val_loader = create_split_dataloaders(
        ...     "/path/to/train", "/path/to/val"
        ... )
    """
    from torch.utils.data import DataLoader
    from .latent import PreEncodedLatentDataset
    
    # Create datasets
    train_dataset = PreEncodedLatentDataset.from_parent_dirs([train_path], **kwargs)
    val_dataset = PreEncodedLatentDataset.from_parent_dirs([val_path], **kwargs)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    
    return train_loader, val_loader


def create_datamodule(
    dataset_config: Dict[str, Any],
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs,
) -> LightningDataModule:
    """Create a Lightning DataModule from configuration parameters.

    This is a convenience function for users who want to create data modules
    programmatically without using configuration files. All parameters
    use sensible defaults.

    Args:
        dataset_config: Configuration dictionary for the dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        **kwargs: Additional arguments

    Returns:
        Configured LightningDataModule instance

    Examples:
        >>> config = {"dataset_path": "/path/to/data", "crop_length": 32768}
        >>> datamodule = create_datamodule(config, batch_size=64)
    """
    # Create DictConfig and delegate to config-based factory
    from omegaconf import DictConfig
    config_dict = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        **dataset_config,
        **kwargs,
    }
    data_config = DictConfig(config_dict)
    return create_datamodule_from_config(data_config)
