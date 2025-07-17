from typing import Any

from pydantic import Field, BaseModel


class SubModelConfig(BaseModel):
    type: str = Field(description="The type of the model")
    config: dict[str, Any] = Field( # pyright: ignore[reportExplicitAny]
        description="The model configuration"
    )

class ModelConfig(BaseModel):
    encoder: SubModelConfig | None = Field(
        description="The encoder configuration", 
    )
    decoder: SubModelConfig | None = Field(
        description="The decoder configuration", 
    )
    bottleneck: SubModelConfig | None = Field(
        description="The bottleneck configuration", 
    )
    latent_dim: int = Field(description="The latent dimension")
    in_channels: int = Field(description="The input channels")
    out_channels: int = Field(description="The output channels")

class ModelingConfig(BaseModel):
    model_type: str = Field(description="The type of the model")
    model: ModelConfig = Field(description="The model configuration")