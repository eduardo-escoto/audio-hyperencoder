from typing import Any, Literal

from pydantic import Field, BaseModel

from hyperencoder.datamodels.data import DataConfig
from hyperencoder.datamodels.modeling import ModelingConfig
from hyperencoder.datamodels.training import TrainingConfig
from hyperencoder.datamodels.experiment_logging import ExperimentLoggingConfig


class TaskConfig(BaseModel):
    seed: int = Field(description="The seed for the task")
    task: Literal["train", "pre_encode"] = Field(description="The type of the task")
    data: DataConfig = Field(description="The dataset configuration for the task")
    modeling: ModelingConfig = Field(description="The model configuration for the task")


class TrainingTaskConfig(TaskConfig):
    training: TrainingConfig = Field(
        description="The training configuration for the task"
    )
    experiment_logging: ExperimentLoggingConfig = Field(
        description="The logging configuration for the task"
    )


class PreEncodeTaskConfig(TaskConfig):
    pass
