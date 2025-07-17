from typing import Any, Literal

from pydantic import Field, BaseModel


class ExperimentLoggingConfig(BaseModel):
    type: Literal["wandb"] = Field(description="The type of the experiment logging")
    config: dict[str, Any] = Field(
        description="The configuration for the experiment logging"
    )
