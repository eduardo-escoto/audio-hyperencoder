"""
Model configuration models for hyperencoder.

This module defines the Pydantic models for model architecture configurations,
replacing the previous JSON-based model configs.
"""

from pydantic import Field, field_validator

from .base import BaseConfig
from .auxiliary_heads import AuxiliaryHeadConfig


class OptimizerConfig(BaseConfig):
    """Configuration for optimizer settings."""

    target_: str = Field(
        default="torch.optim.AdamW",
        alias="_target_",
        description="Target optimizer class",
        examples=["torch.optim.AdamW", "torch.optim.Adam", "torch.optim.SGD"],
    )

    lr: float = Field(default=1e-4, ge=1e-8, le=1.0, description="Learning rate")

    betas: list[float] = Field(
        default=[0.9, 0.999], description="Beta parameters for Adam-based optimizers"
    )

    weight_decay: float = Field(
        default=1e-3, ge=0.0, description="Weight decay (L2 regularization)"
    )

    eps: float = Field(
        default=1e-8, ge=1e-12, description="Epsilon for numerical stability"
    )


class SchedulerConfig(BaseConfig):
    """Configuration for learning rate scheduler."""

    target_: str = Field(
        default="stable_audio_tools.training.lr_schedulers.InverseLR",
        alias="_target_",
        description="Target scheduler class",
        examples=[
            "stable_audio_tools.training.lr_schedulers.InverseLR",
            "torch.optim.lr_scheduler.CosineAnnealingLR",
            "torch.optim.lr_scheduler.StepLR",
        ],
    )

    inv_gamma: int | None = Field(
        default=1000000,
        ge=1,
        description="Inverse gamma parameter for InverseLR scheduler",
    )

    power: float | None = Field(
        default=0.5, ge=0.0, description="Power parameter for InverseLR scheduler"
    )

    warmup: float | None = Field(
        default=0.99, ge=0.0, le=1.0, description="Warmup ratio for scheduler"
    )

    # Additional scheduler parameters
    step_size: int | None = Field(
        default=None, ge=1, description="Step size for StepLR scheduler"
    )

    gamma: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Gamma parameter for StepLR scheduler"
    )


class OptimizerSchedulerConfig(BaseConfig):
    """Combined optimizer and scheduler configuration."""

    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig, description="Optimizer configuration"
    )

    scheduler: SchedulerConfig | None = Field(
        default_factory=SchedulerConfig,
        description="Learning rate scheduler configuration",
    )


class TrainingConfig(BaseConfig):
    """Training-specific configuration for the model."""

    optimizer_configs: dict[str, OptimizerSchedulerConfig] = Field(
        default_factory=dict,
        description="Optimizer configurations for different model components",
        examples=[
            {
                "hyperencoder": {
                    "optimizer": {"lr": 5e-5},
                    "scheduler": {"inv_gamma": 1000000},
                }
            }
        ],
    )


class EncoderConfig(BaseConfig):
    """Configuration for encoder architecture."""

    # target_: str = Field(
    #     default="hyperencoder.models.encoders.OobleckEncoder",
    #     alias="_target_",
    #     description="Target encoder class",
    #     examples=[
    #         "hyperencoder.models.encoders.OobleckEncoder",
    #         "hyperencoder.models.encoders.ResNetEncoder",
    #     ],
    # )

    type: str = Field(default="oobleck", description="Type of encoder")

    config: dict = Field(default_factory=dict, description="Configuration for encoder")

    # in_channels: int = Field(default=64, ge=1, description="Number of input channels")

    # channels: int = Field(default=4, ge=1, description="Base number of channels")

    # latent_dim: int = Field(default=4, ge=1, description="Dimension of latent space")

    # c_mults: list[int] = Field(
    #     default=[16, 8, 4, 2, 2], description="Channel multipliers for each layer"
    # )

    # strides: list[int] = Field(
    #     default=[8, 8, 4, 4, 1], description="Stride values for each layer"
    # )

    # use_snake: bool = Field(
    #     default=False, description="Whether to use Snake activation"
    # )

    # @field_validator("c_mults", "strides")
    # @classmethod
    # def validate_equal_lengths(cls, v, info):
    #     """Ensure c_mults and strides have the same length."""
    #     if info.field_name == "strides" and info.data.get("c_mults"):
    #         c_mults = info.data["c_mults"]
    #         if len(v) != len(c_mults):
    #             raise ValueError(
    #                 f"c_mults and strides must have the same length, got {len(c_mults)} and {len(v)}"
    #             )
    #     return v


class DecoderConfig(BaseConfig):
    type: str = Field(default="oobleck", description="Type of encoder")

    config: dict = Field(default_factory=dict, description="Configuration for encoder")
    """Configuration for decoder architecture."""

    # target_: str = Field(
    #     default="hyperencoder.models.decoders.OobleckDecoder",
    #     alias="_target_",
    #     description="Target decoder class",
    #     examples=[
    #         "hyperencoder.models.decoders.OobleckDecoder",
    #         "hyperencoder.models.decoders.ResNetDecoder",
    #     ],
    # )

    # out_channels: int = Field(default=64, ge=1, description="Number of output channels")

    # channels: int = Field(default=4, ge=1, description="Base number of channels")

    # latent_dim: int = Field(default=4, ge=1, description="Dimension of latent space")

    # c_mults: list[int] = Field(
    #     default=[16, 8, 4, 2, 2], description="Channel multipliers for each layer"
    # )

    # strides: list[int] = Field(
    #     default=[8, 8, 4, 4, 1], description="Stride values for each layer"
    # )

    # use_snake: bool = Field(
    #     default=False, description="Whether to use Snake activation"
    # )

    # final_tanh: bool = Field(
    #     default=False, description="Whether to apply tanh activation at the end"
    # )

    # @field_validator("c_mults", "strides")
    # @classmethod
    # def validate_equal_lengths(cls, v, info):
    #     """Ensure c_mults and strides have the same length."""
    #     if info.field_name == "strides" and info.data.get("c_mults"):
    #         c_mults = info.data["c_mults"]
    #         if len(v) != len(c_mults):
    #             raise ValueError(
    #                 f"c_mults and strides must have the same length, got {len(c_mults)} and {len(v)}"
    #             )
    #     return v


class BottleneckConfig(BaseConfig):
    """Configuration for bottleneck architecture."""

    # target_: str = Field(
    #     default="hyperencoder.models.bottlenecks.FSQBottleneck",
    #     alias="_target_",
    #     description="Target bottleneck class",
    #     examples=[
    #         "hyperencoder.models.bottlenecks.FSQBottleneck",
    #         "hyperencoder.models.bottlenecks.VQBottleneck",
    #         "hyperencoder.models.bottlenecks.NoBottleneck",
    #     ],
    # )
    type: str = Field(default="rvq_vae", description="Type of bottleneck")
    config: dict = Field(default_factory=dict, description="Configuration for bottleneck")
    # # FSQ-specific parameters
    # levels: list[int] | None = Field(
    #     default=[8, 5, 5, 5], description="Quantization levels for FSQ bottleneck"
    # )

    # # VQ-specific parameters
    # num_quantizers: int | None = Field(
    #     default=None, ge=1, description="Number of quantizers for VQ bottleneck"
    # )

    # codebook_size: int | None = Field(
    #     default=None, ge=1, description="Size of codebook for VQ bottleneck"
    # )

    # commitment_loss_weight: float | None = Field(
    #     default=None, ge=0.0, description="Weight for commitment loss in VQ"
    # )


class DemoConfig(BaseConfig):
    """Configuration for demo/evaluation settings."""

    demo_every: int = Field(default=2000, ge=1, description="Generate demo every N steps")

    max_demos: int = Field(
        default=8, ge=1, description="Maximum number of demos to generate"
    )

    sample_rate: int = Field(
        default=44100, ge=1, description="Sample rate for audio demos"
    )

    demo_length: int | None = Field(
        default=None, ge=1, description="Length of demo sequences"
    )

    save_demos: bool = Field(default=True, description="Whether to save demo outputs")


class AuxiliaryHeadsConfig(BaseConfig):
    """Configuration for auxiliary prediction heads for multi-task learning.
    
    This configuration enables auxiliary heads that predict MIDI metadata
    from latent representations to improve semantic learning.
    
    Examples:
        >>> # Basic song-level features
        >>> from hyperencoder.modules.auxiliary_heads import AuxiliaryHeadConfig
        >>> config = AuxiliaryHeadsConfig(
        ...     enabled=True,
        ...     heads=[
        ...         AuxiliaryHeadConfig(
        ...             name="tempo_predictor",
        ...             target_key="tempo_bpm",
        ...             head_type="regression",
        ...             loss_type="mse",
        ...             loss_weight=0.1
        ...         )
        ...     ]
        ... )
    """
    
    enabled: bool = Field(
        default=False,
        description="Whether to enable auxiliary heads during training"
    )
    
    heads: list[AuxiliaryHeadConfig] = Field(
        default_factory=list,
        description="List of auxiliary head configurations"
    )
    
    validation_enabled: bool = Field(
        default=True,
        description="Whether to compute auxiliary metrics during validation"
    )
    
    logging_interval: int = Field(
        default=100,
        ge=1,
        description="Log auxiliary metrics every N training steps"
    )
    
    @field_validator("heads")
    @classmethod
    def validate_unique_head_names(cls, v):
        """Ensure all head names are unique."""
        names = [head.name for head in v]
        if len(names) != len(set(names)):
            raise ValueError("All auxiliary head names must be unique")
        return v
    
    @field_validator("heads")
    @classmethod
    def validate_unique_target_keys(cls, v):
        """Ensure all target keys are unique."""
        targets = [head.target_key for head in v]
        if len(targets) != len(set(targets)):
            raise ValueError("All auxiliary head target keys must be unique")
        return v


class ModelConfig(BaseConfig):
    """
    Configuration for hyperencoder model architecture.

    This replaces the previous JSON-based model configurations with
    a modern, type-safe, validated configuration system.
    """

    # Core model configuration
    # target_: str = Field(
    #     default="hyperencoder.models.hyperencoder.HyperEncoder",
    #     alias="_target_",
    #     description="Target model class to instantiate",
    #     examples=[
    #         "hyperencoder.models.hyperencoder.HyperEncoder",
    #         "hyperencoder.models.hyperencoder.BasicHyperEncoder",
    #     ],
    # )
    model_type: str = Field(default="hyperencoder", description="Type of model")

    # Architecture components
    encoder: EncoderConfig = Field(
        default_factory=EncoderConfig, description="Encoder architecture configuration"
    )

    decoder: DecoderConfig = Field(
        default_factory=DecoderConfig, description="Decoder architecture configuration"
    )

    bottleneck: BottleneckConfig = Field(
        default_factory=BottleneckConfig,
        description="Bottleneck architecture configuration",
    )

    # Model dimensions
    latent_dim: int = Field(
        default=4, ge=1, description="Dimension of the latent space"
    )

    in_channels: int = Field(default=64, ge=1, description="Number of input channels")

    out_channels: int = Field(default=64, ge=1, description="Number of output channels")

    # Training configuration
    training: TrainingConfig | None = Field(
        default_factory=TrainingConfig, description="Training-specific configuration"
    )

    # Demo/evaluation settings
    demo: DemoConfig | None = Field(
        default_factory=DemoConfig, description="Demo and evaluation settings"
    )
    
    # Auxiliary heads for multi-task learning
    auxiliary_heads: AuxiliaryHeadsConfig | None = Field(
        default_factory=AuxiliaryHeadsConfig,
        description="Auxiliary prediction heads configuration"
    )

    # Model-specific parameters
    sample_rate: int | None = Field(
        default=None, ge=1, description="Sample rate for audio processing"
    )

    # Validation
    @field_validator("encoder", "decoder")
    @classmethod
    def validate_encoder_decoder_consistency(cls, v, info):
        """Ensure encoder and decoder have consistent dimensions."""
        if info.field_name == "decoder" and info.data.get("encoder"):
            encoder = info.data["encoder"]
            if hasattr(encoder, "latent_dim") and hasattr(v, "latent_dim"):
                if encoder.latent_dim != v.latent_dim:
                    raise ValueError(
                        f"Encoder and decoder latent_dim must match: {encoder.latent_dim} != {v.latent_dim}"
                    )
        return v

    @field_validator("latent_dim")
    @classmethod
    def validate_latent_dim_consistency(cls, v, info):
        """Ensure latent_dim matches encoder/decoder configurations."""
        # Guard against validation being called with wrong type
        if not isinstance(v, int):
            return v
            
        if info.data.get("encoder") and hasattr(info.data["encoder"], "latent_dim"):
            if info.data["encoder"].latent_dim != v:
                raise ValueError(
                    f"Model latent_dim must match encoder latent_dim: {v} != {info.data['encoder'].latent_dim}"
                )
        return v


# Rebuild model to resolve forward references
def _rebuild_models():
    """Rebuild models to resolve forward references."""
    try:
        from ..modules.auxiliary_heads import AuxiliaryHeadConfig
        AuxiliaryHeadsConfig.model_rebuild()
        ModelConfig.model_rebuild()
    except ImportError:
        # If auxiliary_heads is not available, skip rebuilding
        pass

# Call rebuild when module is imported
_rebuild_models()
