from typing import Any

from pydantic import Field, BaseModel


class DatasetConfig(BaseModel):
    type: str = Field(description="The type of the dataset")
    config: dict[str, Any] = Field(description="The configuration for the dataset")


class DataloaderConfig(BaseModel):
    num_workers: int = Field(description="The number of workers for the dataloader")
    batch_size: int = Field(description="The batch size for the dataloader")
    persistent_workers: bool = Field(
        description="Whether to use persistent workers for the dataloader"
    )

class DataConfig(BaseModel):
    dataset: DatasetConfig = Field(description="The configuration for the dataset")
    dataloader: DataloaderConfig = Field(
        description="The configuration for the dataloader"
    )
