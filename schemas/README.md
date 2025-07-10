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

- **HydraConfig**: [`hydra_config.schema.json`](./hydra_config.schema.json)
  - Configuration for Hydra framework settings (hydra/default.yaml).

- **TrainTaskConfig**: [`train_task_config.schema.json`](./train_task_config.schema.json)
  - Configuration for train.yaml - complete training task configuration.

- **PreEncodeTaskConfig**: [`pre_encode_task_config.schema.json`](./pre_encode_task_config.schema.json)
  - Configuration for pre_encode.yaml - complete pre-encoding task configuration.

- **TaskConfig**: [`task_config.schema.json`](./task_config.schema.json)
  - DEPRECATED: Legacy task configuration. Use TrainTaskConfig or PreEncodeTaskConfig instead.

- **TrainingTaskConfig**: [`training_task_config.schema.json`](./training_task_config.schema.json)
  - DEPRECATED: Legacy training task configuration. Use TrainTaskConfig instead.

- **ExperimentConfig**: [`experiment_config.schema.json`](./experiment_config.schema.json)
  - DEPRECATED: Legacy experiment configuration. Use TrainTaskConfig instead.

- **EncoderConfig**: [`encoder_config.schema.json`](./encoder_config.schema.json)
  - Configuration for encoder architecture.

- **DecoderConfig**: [`decoder_config.schema.json`](./decoder_config.schema.json)
  - Configuration for decoder architecture.

- **BottleneckConfig**: [`bottleneck_config.schema.json`](./bottleneck_config.schema.json)
  - Configuration for bottleneck architecture.

- **OptimizerConfig**: [`optimizer_config.schema.json`](./optimizer_config.schema.json)
  - Configuration for optimizer settings.

- **SchedulerConfig**: [`scheduler_config.schema.json`](./scheduler_config.schema.json)
  - Configuration for learning rate scheduler.

- **OptimizerSchedulerConfig**: [`optimizer_scheduler_config.schema.json`](./optimizer_scheduler_config.schema.json)
  - Combined optimizer and scheduler configuration.

- **DemoConfig**: [`demo_config.schema.json`](./demo_config.schema.json)
  - Configuration for demo/evaluation settings.

- **CropConfig**: [`crop_config.schema.json`](./crop_config.schema.json)
  - Configuration for data cropping/augmentation.

- **DatasetEntry**: [`dataset_entry.schema.json`](./dataset_entry.schema.json)
  - Configuration for a single dataset entry.

## Usage in YAML Files

Add the following line at the top of your YAML configuration files:

```yaml
# yaml-language-server: $schema=../schemas/your_config.schema.json
```

## Auto-generated

⚠️ **These files are auto-generated.** Do not edit them manually.
Run `uv run hyperencoder-utils generate schemas` to regenerate them.
