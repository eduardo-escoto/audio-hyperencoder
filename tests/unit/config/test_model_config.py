"""
Unit tests for ModelConfig class and related configuration classes.

Tests the model configuration classes with their validation rules,
constraints, and integration with BaseConfig.
"""

import pytest
from typing import Any, Dict

from pydantic import ValidationError
from hyperencoder.config import (
    ModelConfig,
    EncoderConfig,
    DecoderConfig,
    BottleneckConfig,
    OptimizerConfig,
    SchedulerConfig,
    OptimizerSchedulerConfig,
    DemoConfig,
)


class TestOptimizerConfig:
    """Test suite for OptimizerConfig class."""

    def test_optimizer_config_defaults(self):
        """Test that OptimizerConfig has correct default values."""
        config = OptimizerConfig()
        
        assert config.target_ == "torch.optim.AdamW"
        assert config.lr == 1e-4
        assert config.betas == [0.9, 0.999]
        assert config.weight_decay == 1e-3
        assert config.eps == 1e-8

    def test_optimizer_config_custom_values(self):
        """Test OptimizerConfig with custom values."""
        config = OptimizerConfig(
            target_="torch.optim.Adam",
            lr=5e-5,
            betas=[0.9, 0.99],
            weight_decay=1e-4,
            eps=1e-7
        )
        
        assert config.target_ == "torch.optim.Adam"
        assert config.lr == 5e-5
        assert config.betas == [0.9, 0.99]
        assert config.weight_decay == 1e-4
        assert config.eps == 1e-7

    def test_lr_validation(self):
        """Test learning rate validation."""
        # Valid learning rates
        valid_lrs = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
        for lr in valid_lrs:
            config = OptimizerConfig(lr=lr)
            assert config.lr == lr
        
        # Invalid learning rates
        with pytest.raises(ValidationError):
            OptimizerConfig(lr=0)
        
        with pytest.raises(ValidationError):
            OptimizerConfig(lr=-1e-4)
        
        with pytest.raises(ValidationError):
            OptimizerConfig(lr=2.0)


class TestSchedulerConfig:
    """Test suite for SchedulerConfig class."""

    def test_scheduler_config_defaults(self):
        """Test that SchedulerConfig has correct default values."""
        config = SchedulerConfig()
        
        assert config.target_ == "stable_audio_tools.training.lr_schedulers.InverseLR"
        assert config.inv_gamma == 1000000
        assert config.power == 0.5
        assert config.warmup == 0.99
        assert config.step_size is None
        assert config.gamma is None

    def test_scheduler_config_custom_values(self):
        """Test SchedulerConfig with custom values."""
        config = SchedulerConfig(
            target_="torch.optim.lr_scheduler.StepLR",
            step_size=10,
            gamma=0.1
        )
        
        assert config.target_ == "torch.optim.lr_scheduler.StepLR"
        assert config.step_size == 10
        assert config.gamma == 0.1


class TestEncoderConfig:
    """Test suite for EncoderConfig class."""

    def test_encoder_config_defaults(self):
        """Test that EncoderConfig has correct default values."""
        config = EncoderConfig()
        
        assert config.target_ == "hyperencoder.models.encoders.OobleckEncoder"
        assert config.in_channels == 64
        assert config.channels == 4
        assert config.latent_dim == 4
        assert config.c_mults == [16, 8, 4, 2, 2]
        assert config.strides == [8, 8, 4, 4, 1]
        assert config.use_snake is False

    def test_encoder_config_custom_values(self):
        """Test EncoderConfig with custom values."""
        config = EncoderConfig(
            in_channels=32,
            channels=8,
            latent_dim=8,
            c_mults=[8, 4, 2],
            strides=[4, 4, 2],
            use_snake=True
        )
        
        assert config.in_channels == 32
        assert config.channels == 8
        assert config.latent_dim == 8
        assert config.c_mults == [8, 4, 2]
        assert config.strides == [4, 4, 2]
        assert config.use_snake is True

    def test_c_mults_strides_length_validation(self):
        """Test that c_mults and strides have the same length."""
        # Valid - same length
        config = EncoderConfig(
            c_mults=[8, 4, 2],
            strides=[4, 4, 2]
        )
        assert len(config.c_mults) == len(config.strides)
        
        # Invalid - different lengths should raise validation error
        with pytest.raises(ValidationError):
            EncoderConfig(
                c_mults=[8, 4, 2],
                strides=[4, 4]  # Different length
            )


class TestDecoderConfig:
    """Test suite for DecoderConfig class."""

    def test_decoder_config_defaults(self):
        """Test that DecoderConfig has correct default values."""
        config = DecoderConfig()
        
        assert config.target_ == "hyperencoder.models.decoders.OobleckDecoder"
        assert config.out_channels == 64
        assert config.channels == 4
        assert config.latent_dim == 4
        assert config.c_mults == [16, 8, 4, 2, 2]
        assert config.strides == [8, 8, 4, 4, 1]
        assert config.use_snake is False
        assert config.final_tanh is False

    def test_decoder_config_custom_values(self):
        """Test DecoderConfig with custom values."""
        config = DecoderConfig(
            out_channels=32,
            final_tanh=True
        )
        
        assert config.out_channels == 32
        assert config.final_tanh is True


class TestBottleneckConfig:
    """Test suite for BottleneckConfig class."""

    def test_bottleneck_config_defaults(self):
        """Test that BottleneckConfig has correct default values."""
        config = BottleneckConfig()
        
        assert config.target_ == "hyperencoder.models.bottlenecks.FSQBottleneck"
        assert config.levels == [8, 5, 5, 5]
        assert config.num_quantizers is None
        assert config.codebook_size is None
        assert config.commitment_loss_weight is None

    def test_bottleneck_config_fsq(self):
        """Test BottleneckConfig for FSQ bottleneck."""
        config = BottleneckConfig(
            target_="hyperencoder.models.bottlenecks.FSQBottleneck",
            levels=[8, 5, 5, 5]
        )
        
        assert config.target_ == "hyperencoder.models.bottlenecks.FSQBottleneck"
        assert config.levels == [8, 5, 5, 5]

    def test_bottleneck_config_vq(self):
        """Test BottleneckConfig for VQ bottleneck."""
        config = BottleneckConfig(
            target_="hyperencoder.models.bottlenecks.VQBottleneck",
            num_quantizers=4,
            codebook_size=1024,
            commitment_loss_weight=0.25
        )
        
        assert config.target_ == "hyperencoder.models.bottlenecks.VQBottleneck"
        assert config.num_quantizers == 4
        assert config.codebook_size == 1024
        assert config.commitment_loss_weight == 0.25


class TestDemoConfig:
    """Test suite for DemoConfig class."""

    def test_demo_config_defaults(self):
        """Test that DemoConfig has correct default values."""
        config = DemoConfig()
        
        assert config.demo_every == 20
        assert config.max_demos == 10
        assert config.demo_length is None
        assert config.save_demos is True

    def test_demo_config_custom_values(self):
        """Test DemoConfig with custom values."""
        config = DemoConfig(
            demo_every=50,
            max_demos=5,
            demo_length=100,
            save_demos=False
        )
        
        assert config.demo_every == 50
        assert config.max_demos == 5
        assert config.demo_length == 100
        assert config.save_demos is False


class TestOptimizerSchedulerConfig:
    """Test suite for OptimizerSchedulerConfig class."""

    def test_optimizer_scheduler_config_defaults(self):
        """Test that OptimizerSchedulerConfig has correct default values."""
        config = OptimizerSchedulerConfig()
        
        assert isinstance(config.optimizer, OptimizerConfig)
        assert isinstance(config.scheduler, SchedulerConfig)
        assert config.optimizer.target_ == "torch.optim.AdamW"
        assert config.scheduler.target_ == "stable_audio_tools.training.lr_schedulers.InverseLR"

    def test_optimizer_scheduler_config_custom(self):
        """Test OptimizerSchedulerConfig with custom values."""
        optimizer = OptimizerConfig(lr=1e-3)
        scheduler = SchedulerConfig(power=0.8)
        
        config = OptimizerSchedulerConfig(
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        assert config.optimizer.lr == 1e-3
        assert config.scheduler.power == 0.8


class TestModelConfig:
    """Test suite for ModelConfig class."""

    def test_model_config_defaults(self):
        """Test that ModelConfig has correct default values."""
        config = ModelConfig()
        
        # Core model configuration
        assert config.target_ == "hyperencoder.models.hyperencoder.HyperEncoder"
        
        # Architecture components
        assert isinstance(config.encoder, EncoderConfig)
        assert isinstance(config.decoder, DecoderConfig)
        assert isinstance(config.bottleneck, BottleneckConfig)
        
        # Model dimensions
        assert config.latent_dim == 4
        assert config.in_channels == 64
        assert config.out_channels == 64
        
        # Training and demo configurations
        assert config.training is not None
        assert config.demo is not None
        
        # Model-specific parameters
        assert config.sample_rate is None

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        encoder = EncoderConfig(latent_dim=8, in_channels=32)
        decoder = DecoderConfig(latent_dim=8, out_channels=32)
        bottleneck = BottleneckConfig(levels=[4, 4, 4, 4])
        demo = DemoConfig(demo_every=10, max_demos=5)
        
        config = ModelConfig(
            target_="custom.model.CustomHyperEncoder",
            encoder=encoder,
            decoder=decoder,
            bottleneck=bottleneck,
            latent_dim=8,
            in_channels=32,
            out_channels=32,
            demo=demo,
            sample_rate=44100
        )
        
        assert config.target_ == "custom.model.CustomHyperEncoder"
        assert config.encoder.latent_dim == 8
        assert config.decoder.latent_dim == 8
        assert config.bottleneck.levels == [4, 4, 4, 4]
        assert config.latent_dim == 8
        assert config.in_channels == 32
        assert config.out_channels == 32
        assert config.demo.demo_every == 10
        assert config.sample_rate == 44100

    def test_model_config_from_dict(self):
        """Test ModelConfig creation from dictionary."""
        config_dict = {
            "target_": "custom.model.HyperEncoder",
            "latent_dim": 8,
            "in_channels": 32,
            "out_channels": 32,
            "encoder": {
                "latent_dim": 8,
                "in_channels": 32
            },
            "decoder": {
                "latent_dim": 8,
                "out_channels": 32
            },
            "bottleneck": {
                "levels": [4, 4, 4, 4]
            },
            "demo": {
                "demo_every": 10,
                "max_demos": 5
            }
        }
        
        config = ModelConfig.from_dict(config_dict)
        
        assert config.target_ == "custom.model.HyperEncoder"
        assert config.latent_dim == 8
        assert config.in_channels == 32
        assert config.out_channels == 32
        assert config.encoder.latent_dim == 8
        assert config.decoder.out_channels == 32
        assert config.bottleneck.levels == [4, 4, 4, 4]
        assert config.demo.demo_every == 10

    def test_model_config_yaml_serialization(self, temp_dir):
        """Test ModelConfig YAML serialization and deserialization."""
        # Create encoder and decoder with matching latent_dim
        encoder = EncoderConfig(latent_dim=8)
        decoder = DecoderConfig(latent_dim=8)
        
        config = ModelConfig(
            target_="custom.model.HyperEncoder",
            encoder=encoder,
            decoder=decoder,
            latent_dim=8,
            sample_rate=44100
        )
        
        # Save to YAML
        yaml_path = temp_dir / "model_config.yaml"
        config.save_yaml(yaml_path)
        
        # Load from YAML
        loaded_config = ModelConfig.from_yaml(yaml_path)
        
        assert loaded_config.target_ == "custom.model.HyperEncoder"
        assert loaded_config.latent_dim == 8
        assert loaded_config.sample_rate == 44100

    def test_model_config_comprehensive(self):
        """Test a comprehensive ModelConfig configuration."""
        # Create detailed component configurations
        encoder = EncoderConfig(
            target_="hyperencoder.models.encoders.OobleckEncoder",
            in_channels=64,
            channels=4,
            latent_dim=4,
            c_mults=[16, 8, 4, 2, 2],
            strides=[8, 8, 4, 4, 1],
            use_snake=False
        )
        
        decoder = DecoderConfig(
            target_="hyperencoder.models.decoders.OobleckDecoder",
            out_channels=64,
            channels=4,
            latent_dim=4,
            c_mults=[16, 8, 4, 2, 2],
            strides=[8, 8, 4, 4, 1],
            use_snake=False,
            final_tanh=False
        )
        
        bottleneck = BottleneckConfig(
            target_="hyperencoder.models.bottlenecks.FSQBottleneck",
            levels=[8, 5, 5, 5]
        )
        
        demo = DemoConfig(
            demo_every=20,
            max_demos=10,
            save_demos=True
        )
        
        config = ModelConfig(
            target_="hyperencoder.models.hyperencoder.HyperEncoder",
            encoder=encoder,
            decoder=decoder,
            bottleneck=bottleneck,
            latent_dim=4,
            in_channels=64,
            out_channels=64,
            demo=demo,
            sample_rate=44100
        )
        
        # Verify all components
        assert config.target_ == "hyperencoder.models.hyperencoder.HyperEncoder"
        assert config.encoder.target_ == "hyperencoder.models.encoders.OobleckEncoder"
        assert config.decoder.target_ == "hyperencoder.models.decoders.OobleckDecoder"
        assert config.bottleneck.target_ == "hyperencoder.models.bottlenecks.FSQBottleneck"
        assert config.latent_dim == 4
        assert config.in_channels == 64
        assert config.out_channels == 64
        assert config.demo.demo_every == 20
        assert config.sample_rate == 44100

    def test_field_descriptions_present(self):
        """Test that important fields have descriptions."""
        config = ModelConfig()
        schema = config.model_json_schema()
        
        # Check that key fields have descriptions
        properties = schema["properties"]
        assert "description" in properties["_target_"]  # Uses alias in schema
        assert "description" in properties["latent_dim"]
        assert "description" in properties["in_channels"]
        assert "description" in properties["out_channels"] 