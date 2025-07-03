# JSON Schemas for Hyperencoder Configuration

This directory contains JSON schemas generated from Pydantic configuration models.
These schemas enable IDE validation and autocomplete for YAML configuration files.

## Available Schemas

- **BaseConfig**: [`base_config.schema.json`](./base_config.schema.json)
  - Base configuration class that all other config classes inherit from.

- **TrainingConfig**: [`training_config.schema.json`](./training_config.schema.json)
  - Configuration for training hyperencoder models.

- **PreEncodeConfig**: [`pre_encode_config.schema.json`](./pre_encode_config.schema.json)
  - Configuration for pre-encoding audio files to latents.

- **DataConfig**: [`data_config.schema.json`](./data_config.schema.json)
  - Configuration for hyperencoder data loading and processing.

- **ModelConfig**: [`model_config.schema.json`](./model_config.schema.json)
  - Configuration for hyperencoder model architecture.

## Usage in YAML Files

Add the following line at the top of your YAML configuration files:

```yaml
# yaml-language-server: $schema=../schemas/your_config.schema.json
```

## Auto-generated

⚠️ **These files are auto-generated.** Do not edit them manually.
Run `python scripts/generate_schemas.py` to regenerate them.
