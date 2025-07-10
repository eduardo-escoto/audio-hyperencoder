# Feature: Auxiliary Classification and Regression Heads for Hyperencoder

## Overview
Add configurable auxiliary heads to the hyperencoder system that can predict MIDI metadata from latent representations. This will enable multi-task learning where the hyperencoder learns better semantic representations by jointly optimizing reconstruction loss and auxiliary prediction tasks.

## Technical Analysis

### Current State Analysis
- **HyperEncoderTrainingWrapper**: Uses `stable_audio_tools.training.losses.MultiLoss` to combine multiple losses
- **Loss System**: Creates `gen_loss_modules` list that gets passed to `MultiLoss` for weighted combination
- **MIDI Metadata**: Rich metadata available in `MidiMetadata` class with regression and classification targets
- **Training Flow**: `training_step` processes `outer_latents` and `info` dict, encodes to `inner_latents`, then reconstructs
- **Integration Point**: Auxiliary heads should take `inner_latents` as input and predict metadata from `info` dict

### Available Prediction Targets (Basic Song-Level Focus)
From `MidiMetadata`, we'll focus on basic song-level characteristics:

**Primary Regression Targets:**
- `tempo_bpm`: Tempo in beats per minute (1-1000) - **Core musical feature**
- `duration_seconds`: Total duration (0+) - **Basic song structure**
- `average_velocity`: Average note velocity (0-127) - **Dynamic intensity**
- `note_density`: Average notes per second (0+) - **Rhythmic complexity**

**Primary Classification Targets:**
- `time_signature_numerator`: Time signature numerator (1-32) - **Rhythmic structure**
- `key_signature`: Key signature (-7 to 7) - **Harmonic center**
- `num_tracks`: Number of tracks (0+) - **Arrangement complexity**

**Secondary Targets (for future extension):**
- `beat_density`: Average notes per beat (0+)
- `velocity_std`: Standard deviation of velocities (0+)
- `note_range_span`: Range of notes (0-127)
- `unique_programs`: List of instrument programs (multi-label)

## Implementation Plan

### 1. Auxiliary Head Architecture

#### Create `hyperencoder/modules/auxiliary_heads.py`
```python
from typing import Literal, Any
from torch import nn, Tensor
from pydantic import BaseModel, Field

class AuxiliaryHeadConfig(BaseModel):
    """Configuration for a single auxiliary head."""
    
    name: str = Field(description="Name of the auxiliary head")
    target_key: str = Field(description="Key in info dict to predict")
    
    head_type: Literal["regression", "classification", "multi_label"] = Field(
        description="Type of prediction head"
    )
    
    # Architecture config
    hidden_dims: list[int] = Field(
        default_factory=lambda: [512, 256], 
        description="Hidden layer dimensions"
    )
    dropout_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    activation: str = Field(default="relu", description="Activation function")
    
    # Loss configuration
    loss_type: str = Field(description="Loss function type")
    loss_weight: float = Field(default=1.0, gt=0.0, description="Weight for this loss")
    
    # Classification specific
    num_classes: int | None = Field(default=None, description="Number of classes for classification")
    
    # Target preprocessing
    target_min: float | None = Field(default=None, description="Min value for normalization")
    target_max: float | None = Field(default=None, description="Max value for normalization")
    log_transform: bool = Field(default=False, description="Apply log transform to target")

class AuxiliaryHead(nn.Module):
    """A single auxiliary prediction head."""
    
    def __init__(self, config: AuxiliaryHeadConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # Build the network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        if config.head_type == "regression":
            output_dim = 1
        elif config.head_type == "classification":
            output_dim = config.num_classes
        elif config.head_type == "multi_label":
            output_dim = config.num_classes
        else:
            raise ValueError(f"Unknown head type: {config.head_type}")
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the head."""
        # Global average pooling if input has spatial dimensions
        if x.dim() > 2:
            x = x.mean(dim=tuple(range(2, x.dim())))
        
        return self.network(x)
    
    def preprocess_target(self, target: Any) -> Tensor:
        """Preprocess target value for training."""
        if isinstance(target, (list, tuple)):
            if self.config.head_type == "multi_label":
                # Convert list of programs to multi-hot encoding
                target_tensor = torch.zeros(self.config.num_classes)
                for idx in target:
                    if 0 <= idx < self.config.num_classes:
                        target_tensor[idx] = 1.0
                return target_tensor
            else:
                # Take first value for single predictions
                target = target[0] if target else 0.0
        
        target = float(target)
        
        if self.config.log_transform:
            target = torch.log(torch.tensor(target + 1e-8))
        
        if self.config.target_min is not None and self.config.target_max is not None:
            target = (target - self.config.target_min) / (self.config.target_max - self.config.target_min)
        
        return torch.tensor(target, dtype=torch.float32)
```

### 2. Auxiliary Loss Functions

#### Create `hyperencoder/modules/auxiliary_losses.py`
```python
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any, Optional
from stable_audio_tools.training.losses import LossWithTarget

class AuxiliaryLoss(LossWithTarget):
    """Loss function for auxiliary heads."""
    
    def __init__(
        self,
        auxiliary_head: AuxiliaryHead,
        latent_key: str = "inner_latents",
        info_key: str = "info",
        weight: float = 1.0,
        name: str = "auxiliary_loss"
    ):
        self.auxiliary_head = auxiliary_head
        self.latent_key = latent_key
        self.info_key = info_key
        self.weight = weight
        self.name = name
        
        # Initialize loss function based on head type
        if auxiliary_head.config.head_type == "regression":
            if auxiliary_head.config.loss_type == "mse":
                self.loss_fn = nn.MSELoss()
            elif auxiliary_head.config.loss_type == "mae":
                self.loss_fn = nn.L1Loss()
            elif auxiliary_head.config.loss_type == "huber":
                self.loss_fn = nn.HuberLoss()
            else:
                self.loss_fn = nn.MSELoss()
        
        elif auxiliary_head.config.head_type == "classification":
            if auxiliary_head.config.loss_type == "ce":
                self.loss_fn = nn.CrossEntropyLoss()
            elif auxiliary_head.config.loss_type == "focal":
                self.loss_fn = FocalLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        
        elif auxiliary_head.config.head_type == "multi_label":
            if auxiliary_head.config.loss_type == "bce":
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()
    
         def forward(self, loss_info: Dict[str, Any]) -> Tensor:
         """Compute auxiliary loss."""
         latents = loss_info[self.latent_key]
         info = loss_info.get(self.info_key, loss_info)
         
         # Extract MIDI metadata (guaranteed to exist since dataset errors without it)
         midi_metadata = info["midi_metadata"]
         
         # Get target value
         target_value = midi_metadata.get(self.auxiliary_head.config.target_key)
         if target_value is None:
             # Log warning if specific target is missing but don't fail
             logging.warning(f"Target key '{self.auxiliary_head.config.target_key}' not found in MIDI metadata")
             return torch.tensor(0.0, device=latents.device)
         
         # Preprocess target
         target = self.auxiliary_head.preprocess_target(target_value)
         target = target.to(latents.device)
         
         # Forward pass through auxiliary head
         prediction = self.auxiliary_head(latents)
         
         # Compute loss
         if self.auxiliary_head.config.head_type == "regression":
             loss = self.loss_fn(prediction.squeeze(), target)
         elif self.auxiliary_head.config.head_type == "classification":
             loss = self.loss_fn(prediction, target.long())
         elif self.auxiliary_head.config.head_type == "multi_label":
             loss = self.loss_fn(prediction, target)
         
         return loss * self.weight

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

### 3. Configuration Integration

#### Update `hyperencoder/datamodels/model_config.py`
```python
class AuxiliaryHeadsConfig(BaseConfig):
    """Configuration for auxiliary prediction heads."""
    
    enabled: bool = Field(default=False, description="Enable auxiliary heads")
    
    heads: list[AuxiliaryHeadConfig] = Field(
        default_factory=list,
        description="List of auxiliary head configurations"
    )
    
         # Global settings
     latent_key: str = Field(
         default="inner_latents",
         description="Key in loss_info dict containing latents to use as input"
     )
    
    # Common presets
    @classmethod
    def create_tempo_and_velocity_preset(cls) -> "AuxiliaryHeadsConfig":
        """Create preset with tempo and velocity prediction heads."""
        return cls(
            enabled=True,
            heads=[
                AuxiliaryHeadConfig(
                    name="tempo_predictor",
                    target_key="tempo_bpm",
                    head_type="regression",
                    loss_type="mse",
                    loss_weight=0.1,
                    target_min=60.0,
                    target_max=200.0,
                    hidden_dims=[512, 256, 128]
                ),
                AuxiliaryHeadConfig(
                    name="velocity_predictor",
                    target_key="average_velocity",
                    head_type="regression",
                    loss_type="mae",
                    loss_weight=0.05,
                    target_min=0.0,
                    target_max=127.0,
                    hidden_dims=[256, 128]
                )
            ]
        )
    
         @classmethod
     def create_basic_song_level_preset(cls) -> "AuxiliaryHeadsConfig":
         """Create preset with basic song-level musical feature prediction."""
         return cls(
             enabled=True,
             heads=[
                 # Core regression heads
                 AuxiliaryHeadConfig(
                     name="tempo_predictor",
                     target_key="tempo_bpm",
                     head_type="regression",
                     loss_type="mse",
                     loss_weight=0.1,
                     target_min=60.0,
                     target_max=200.0,
                     hidden_dims=[512, 256, 128]
                 ),
                 AuxiliaryHeadConfig(
                     name="duration_predictor",
                     target_key="duration_seconds",
                     head_type="regression",
                     loss_type="mse",
                     loss_weight=0.05,
                     log_transform=True,
                     hidden_dims=[256, 128]
                 ),
                 AuxiliaryHeadConfig(
                     name="velocity_predictor",
                     target_key="average_velocity",
                     head_type="regression",
                     loss_type="mae",
                     loss_weight=0.05,
                     target_min=0.0,
                     target_max=127.0,
                     hidden_dims=[256, 128]
                 ),
                 AuxiliaryHeadConfig(
                     name="note_density_predictor",
                     target_key="note_density",
                     head_type="regression",
                     loss_type="mse",
                     loss_weight=0.1,
                     log_transform=True,
                     hidden_dims=[256, 128]
                 ),
                 
                 # Core classification heads
                 AuxiliaryHeadConfig(
                     name="time_signature_predictor",
                     target_key="time_signature_numerator",
                     head_type="classification",
                     loss_type="ce",
                     loss_weight=0.1,
                     num_classes=16,  # Common time signatures: 2, 3, 4, 6, 8, 12, etc.
                     hidden_dims=[256, 128]
                 ),
                 AuxiliaryHeadConfig(
                     name="key_signature_predictor",
                     target_key="key_signature",
                     head_type="classification",
                     loss_type="ce",
                     loss_weight=0.1,
                     num_classes=15,  # -7 to +7 mapped to 0-14
                     hidden_dims=[256, 128]
                 ),
                 AuxiliaryHeadConfig(
                     name="track_count_predictor",
                     target_key="num_tracks",
                     head_type="classification",
                     loss_type="ce",
                     loss_weight=0.05,
                     num_classes=10,  # 0-9 tracks (can be extended)
                     hidden_dims=[256, 128]
                 )
             ]
         )

# Update ModelConfig
class ModelConfig(BaseConfig):
    # ... existing fields ...
    
    auxiliary_heads: AuxiliaryHeadsConfig | None = Field(
        default=None,
        description="Configuration for auxiliary prediction heads"
    )
```

### 4. Integration with Training Wrapper

#### Update `hyperencoder/training/hyperencoder.py`
```python
from ..modules.auxiliary_heads import AuxiliaryHead, AuxiliaryHeadConfig
from ..modules.auxiliary_losses import AuxiliaryLoss

class HyperEncoderTrainingWrapper(LightningModule):
    def __init__(
        self,
        hyperencoder: HyperEncoder,
        loss_config: dict[str, Any] | None = None,
        optimizer_configs: dict[str, Any] | None = None,
        auxiliary_heads_config: AuxiliaryHeadsConfig | None = None,
        lr: float = 1e-4,
        clip_grad_norm: float = 0.0,
    ):
        super().__init__()
        # ... existing initialization ...
        
        self.auxiliary_heads_config = auxiliary_heads_config
        self.auxiliary_heads = nn.ModuleDict()
        
        # Initialize auxiliary heads if configured
        if auxiliary_heads_config and auxiliary_heads_config.enabled:
            self._setup_auxiliary_heads()
        
        # ... rest of existing initialization ...
    
    def _setup_auxiliary_heads(self):
        """Initialize auxiliary heads and their losses."""
        if not self.auxiliary_heads_config:
            return
        
        # Get input dimension from hyperencoder
        input_dim = self.hyperencoder.latent_dim
        
        # Create auxiliary heads
        for head_config in self.auxiliary_heads_config.heads:
            auxiliary_head = AuxiliaryHead(head_config, input_dim)
            self.auxiliary_heads[head_config.name] = auxiliary_head
            
            # Create loss for this head
            auxiliary_loss = AuxiliaryLoss(
                auxiliary_head=auxiliary_head,
                latent_key=self.auxiliary_heads_config.latent_key,
                weight=head_config.loss_weight,
                name=f"{head_config.name}_loss"
            )
            
            # Add to loss modules
            self.gen_loss_modules.append(auxiliary_loss)
    
         def training_step(self, batch, batch_idx):
         outer_latents, info = batch
         
         # Since we require MIDI metadata for all batches, we can assume it's present
         # The dataset will error if metadata is missing, so no need to handle that case
         
         # ... rest of existing training step ...
         
         # loss_info now includes info dict which contains midi_metadata
         loss_info = {"outer_latents": outer_latents, "info": info}
         
         # ... rest of existing code ...
         
         reconstructed_outer_latents, encode_info = self.__reconstruct__(
             outer_latents, return_info=True
         )
         loss_info.update(encode_info)
         loss_info["decoder_output"] = reconstructed_outer_latents
         loss_info["reconstructed_outer_latents"] = reconstructed_outer_latents
         
         # Add auxiliary head predictions to logging
         if self.auxiliary_heads_config and self.auxiliary_heads_config.enabled:
             with torch.no_grad():
                 self._log_auxiliary_predictions(loss_info, log_dict)
         
         # ... rest of existing training step ...
    
     def validation_step(self, batch, batch_idx):
         outer_latents, info = batch
         
         # ... existing validation code ...
         
         # Add auxiliary head evaluation for validation
         if self.auxiliary_heads_config and self.auxiliary_heads_config.enabled:
             with no_grad():
                 val_loss_dict.update(self._evaluate_auxiliary_heads(loss_info))
         
         # ... rest of existing validation step ...
    
     def _evaluate_auxiliary_heads(self, loss_info: dict) -> dict:
         """Evaluate auxiliary heads during validation."""
         aux_metrics = {}
         latents = loss_info[self.auxiliary_heads_config.latent_key]
         info = loss_info.get("info", {})
         midi_metadata = info.get("midi_metadata", {})
         
         for head_name, head in self.auxiliary_heads.items():
             try:
                 prediction = head(latents)
                 target_value = midi_metadata.get(head.config.target_key)
                 
                 if target_value is not None:
                     target = head.preprocess_target(target_value)
                     target = target.to(latents.device)
                     
                     if head.config.head_type == "regression":
                         # Compute MAE and MSE for regression
                         mae = torch.abs(prediction.squeeze() - target).mean()
                         mse = torch.pow(prediction.squeeze() - target, 2).mean()
                         aux_metrics[f"aux_val/{head_name}_mae"] = mae.item()
                         aux_metrics[f"aux_val/{head_name}_mse"] = mse.item()
                         
                     elif head.config.head_type == "classification":
                         # Compute accuracy for classification
                         pred_class = torch.argmax(prediction, dim=-1)
                         accuracy = (pred_class == target.long()).float().mean()
                         aux_metrics[f"aux_val/{head_name}_accuracy"] = accuracy.item()
                         
                     elif head.config.head_type == "multi_label":
                         # Compute F1 score for multi-label
                         pred_binary = (torch.sigmoid(prediction) > 0.5).float()
                         tp = (pred_binary * target).sum()
                         fp = (pred_binary * (1 - target)).sum()
                         fn = ((1 - pred_binary) * target).sum()
                         f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
                         aux_metrics[f"aux_val/{head_name}_f1"] = f1.item()
                         
             except Exception as e:
                 logging.warning(f"Failed to evaluate auxiliary head {head_name}: {e}")
         
         return aux_metrics
    
    def _log_auxiliary_predictions(self, loss_info: dict, log_dict: dict):
        """Log auxiliary head predictions for monitoring."""
        latents = loss_info[self.auxiliary_heads_config.latent_key]
        info = loss_info.get("info", {})
        midi_metadata = info.get("midi_metadata")
        
        if not midi_metadata:
            return
        
        for head_name, head in self.auxiliary_heads.items():
            try:
                prediction = head(latents)
                
                # Log prediction statistics
                log_dict[f"aux/{head_name}_pred_mean"] = prediction.mean().item()
                log_dict[f"aux/{head_name}_pred_std"] = prediction.std().item()
                
                # Log target vs prediction if available
                target_value = midi_metadata.get(head.config.target_key)
                if target_value is not None:
                    target = head.preprocess_target(target_value)
                    
                    if head.config.head_type == "regression":
                        log_dict[f"aux/{head_name}_target"] = target.item()
                        log_dict[f"aux/{head_name}_pred"] = prediction.squeeze().mean().item()
                        log_dict[f"aux/{head_name}_error"] = abs(
                            prediction.squeeze().mean().item() - target.item()
                        )
                    
            except Exception as e:
                logging.warning(f"Failed to log auxiliary prediction for {head_name}: {e}")
```

### 5. Factory Function Integration

#### Update `hyperencoder/training/__init__.py`
```python
def create_he_training_wrapper_from_config(
    model_config: ModelConfig,
    hyperencoder: HyperEncoder,
) -> HyperEncoderTrainingWrapper:
    """Create HyperEncoderTrainingWrapper from model configuration."""
    
    # Extract auxiliary heads config
    auxiliary_heads_config = None
    if model_config.auxiliary_heads:
        auxiliary_heads_config = model_config.auxiliary_heads
    
    return HyperEncoderTrainingWrapper(
        hyperencoder=hyperencoder,
        loss_config=model_config.training.loss_config if model_config.training else None,
        optimizer_configs=model_config.training.optimizer_configs if model_config.training else None,
        auxiliary_heads_config=auxiliary_heads_config,
        lr=model_config.training.lr if model_config.training else 1e-4,
        clip_grad_norm=model_config.training.clip_grad_norm if model_config.training else 0.0,
    )
```

### 6. Configuration Files

#### Create `configs/auxiliary_heads/tempo_velocity.yaml`
```yaml
enabled: true
latent_key: "inner_latents"

heads:
  - name: "tempo_predictor"
    target_key: "tempo_bpm"
    head_type: "regression"
    loss_type: "mse"
    loss_weight: 0.1
    target_min: 60.0
    target_max: 200.0
    hidden_dims: [512, 256, 128]
    dropout_rate: 0.1
    activation: "relu"
    
  - name: "velocity_predictor"
    target_key: "average_velocity"
    head_type: "regression"
    loss_type: "mae"
    loss_weight: 0.05
    target_min: 0.0
    target_max: 127.0
    hidden_dims: [256, 128]
    dropout_rate: 0.1
    activation: "relu"
```

#### Create `configs/auxiliary_heads/basic_song_level.yaml`
```yaml
enabled: true
latent_key: "inner_latents"

heads:
  # Core regression heads
  - name: "tempo_predictor"
    target_key: "tempo_bpm"
    head_type: "regression"
    loss_type: "mse"
    loss_weight: 0.1
    target_min: 60.0
    target_max: 200.0
    hidden_dims: [512, 256, 128]
    
  - name: "duration_predictor"
    target_key: "duration_seconds"
    head_type: "regression"
    loss_type: "mse"
    loss_weight: 0.05
    log_transform: true
    hidden_dims: [256, 128]
    
  - name: "velocity_predictor"
    target_key: "average_velocity"
    head_type: "regression"
    loss_type: "mae"
    loss_weight: 0.05
    target_min: 0.0
    target_max: 127.0
    hidden_dims: [256, 128]
    
  - name: "note_density_predictor"
    target_key: "note_density"
    head_type: "regression"
    loss_type: "mse"
    loss_weight: 0.1
    log_transform: true
    hidden_dims: [256, 128]
    
  # Core classification heads
  - name: "time_signature_predictor"
    target_key: "time_signature_numerator"
    head_type: "classification"
    loss_type: "ce"
    loss_weight: 0.1
    num_classes: 16
    hidden_dims: [256, 128]
    
  - name: "key_signature_predictor"
    target_key: "key_signature"
    head_type: "classification"
    loss_type: "ce"
    loss_weight: 0.1
    num_classes: 15
    hidden_dims: [256, 128]
    
  - name: "track_count_predictor"
    target_key: "num_tracks"
    head_type: "classification"
    loss_type: "ce"
    loss_weight: 0.05
    num_classes: 10
    hidden_dims: [256, 128]
```

#### Update `configs/model/default.yaml`
```yaml
# ... existing model config ...

auxiliary_heads:
  enabled: false
  latent_key: "inner_latents"
  heads: []
```

### 7. Usage Examples

#### Example Model Configuration with Auxiliary Heads
```yaml
# configs/model/hyperencoder_with_aux.yaml
defaults:
  - default

auxiliary_heads:
  enabled: true
  latent_key: "inner_latents"
  heads:
    - name: "tempo_predictor"
      target_key: "tempo_bpm"
      head_type: "regression"
      loss_type: "mse"
      loss_weight: 0.1
      target_min: 60.0
      target_max: 200.0
      hidden_dims: [512, 256, 128]
      dropout_rate: 0.1
      activation: "relu"
    - name: "time_signature_predictor"
      target_key: "time_signature_numerator"
      head_type: "classification"
      loss_type: "ce"
      loss_weight: 0.1
      num_classes: 16
      hidden_dims: [256, 128]
      dropout_rate: 0.1
      activation: "relu"
```

#### Training with Auxiliary Heads
```bash
# Train with auxiliary heads
uv run python -m hyperencoder.cli.main \
  model=hyperencoder_with_aux \
  data.midi_metadata.enabled=true \
  data.midi_metadata.midi_dir="/path/to/midi/files"

# Use basic song-level preset
uv run python -m hyperencoder.cli.main \
  model.auxiliary_heads=basic_song_level \
  data.midi_metadata.enabled=true
```

## Design Decisions (Resolved)

1. **Latent Input**: ✅ Auxiliary heads take `inner_latents` (post-encoder, pre-decoder) as input
2. **Batching**: ✅ All batches must have MIDI metadata - dataset errors out if missing
3. **Validation**: ✅ Auxiliary heads are evaluated during validation with appropriate metrics (MAE/MSE for regression, accuracy for classification, F1 for multi-label)
4. **Presets**: ✅ Focus on basic song-level MIDI metadata (tempo, duration, velocity, time signature, key signature)
5. **Architecture**: ✅ Heads are completely independent - no layer sharing

## Dependencies

1. **MIDI Metadata System**: Requires the MIDI metadata injection system to be working
2. **Data Configuration**: Requires `data.midi_metadata.enabled=true` in training configuration
3. **Loss System**: Builds on existing `MultiLoss` system from `stable_audio_tools`

## Success Criteria

1. **Configurable Heads**: Users can specify arbitrary combinations of regression/classification heads targeting basic song-level MIDI metadata
2. **Multi-loss Training**: Auxiliary losses are properly weighted and combined with reconstruction loss using stable_audio_tools.MultiLoss
3. **Validation Evaluation**: Auxiliary heads are evaluated during validation with appropriate metrics (MAE/MSE, accuracy, F1)
4. **Comprehensive Monitoring**: Training logs include auxiliary loss values, prediction accuracy, and target vs. prediction comparisons
5. **Backward Compatibility**: Existing training configurations continue to work without auxiliary heads
6. **Easy Configuration**: Preset configurations (basic_song_level, tempo_velocity) make it easy to add common auxiliary tasks
7. **Independent Architecture**: Each auxiliary head is completely independent with no layer sharing

## Risks & Considerations

1. **Training Stability**: Multiple losses might destabilize training - need careful weight tuning for each head
2. **Computational Overhead**: Additional forward passes through heads during training and validation
3. **Memory Usage**: Additional parameters from auxiliary heads (independent architectures mean no parameter sharing)
4. **Data Dependency**: Requires MIDI metadata to be available for ALL training data (dataset will error if missing)
5. **Hyperparameter Sensitivity**: Loss weights and architectures may need task-specific tuning for optimal performance
6. **Validation Metrics**: Need to monitor auxiliary head performance to ensure they're learning meaningful representations

## Next Steps

1. **Implement Core Modules**: Create `auxiliary_heads.py` and `auxiliary_losses.py` modules
2. **Update Training Wrapper**: Add auxiliary head support to `HyperEncoderTrainingWrapper`
3. **Configuration Integration**: Update `ModelConfig` and add preset configurations
4. **Testing**: Test with basic song-level preset using existing MIDI metadata
5. **Monitoring**: Set up proper logging and validation metrics for auxiliary heads

This system provides a flexible framework for adding auxiliary prediction tasks to guide hyperencoder training while maintaining the existing architecture and training pipeline. The focus on basic song-level MIDI metadata ensures meaningful musical representations are learned.

---

## Appendix: Additional MIDI Metadata Targets for Future Implementation

### Additional Regression Targets

**Statistical Features:**
- `velocity_std`: Standard deviation of note velocities (0+) - **Dynamic variation**
- `note_range_span`: Range of notes (0-127) - **Pitch range complexity**
- `note_range_min`: Lowest MIDI note number (0-127) - **Bass register**  
- `note_range_max`: Highest MIDI note number (0-127) - **Treble register**
- `beat_density`: Average notes per beat (0+) - **Rhythmic density relative to tempo**

**Temporal Features:**
- `total_notes`: Total number of notes (0+, could be log-transformed or binned) - **Overall activity**

### Additional Classification Targets

**Musical Structure:**
- `time_signature_denominator`: Time signature denominator (1-32) - **Note value basis**
- `num_channels`: Number of MIDI channels used (0-16) - **Polyphonic complexity**

**Binned Continuous Variables:**
- `duration_seconds` (binned): Short (0-60s), Medium (60-180s), Long (180s+) - **Song length category**
- `tempo_bpm` (binned): Slow (60-90), Moderate (90-120), Fast (120-160), Very Fast (160+) - **Tempo category**
- `total_notes` (binned): Few (0-100), Some (100-500), Many (500-1000), Very Many (1000+) - **Activity level**

### Multi-Label Classification Targets

**Instrument Analysis:**
- `unique_programs`: List of unique MIDI program numbers (instruments) - **Multi-hot encoding for 128 possible instruments**
- `dominant_programs`: Most frequent instruments (top 5) - **Primary instrumentation**

**Temporal Program Analysis:**
- `program_changes`: Time-based instrument changes - **Could be converted to binary features like "has_program_changes", "frequent_program_changes"**

### Advanced Harmonic/Rhythmic Features (Optional)

**Harmonic Analysis** (if `extract_harmony=True`):
- `chord_progressions`: Detected chord progressions - **Could be converted to categorical/multi-label for common progressions**
- `key_centers`: Detected key centers - **Multi-label for modulating pieces**

**Rhythmic Analysis** (if `extract_rhythm=True`):
- `rhythmic_patterns`: Detected rhythmic patterns - **Could extract features like syncopation level, swing factor, etc.**

### Example Advanced Configuration

```yaml
# Future comprehensive auxiliary heads configuration
auxiliary_heads:
  enabled: true
  latent_key: "inner_latents"
  
  heads:
    # Core song-level (current focus)
    - name: "tempo_predictor"
      target_key: "tempo_bpm"
      head_type: "regression"
      # ... config
    
    # Advanced statistical features
    - name: "velocity_variation_predictor"
      target_key: "velocity_std"
      head_type: "regression"
      loss_type: "mse"
      loss_weight: 0.05
      hidden_dims: [256, 128]
      
    - name: "pitch_range_predictor"
      target_key: "note_range_span"
      head_type: "regression"
      loss_type: "mse"
      loss_weight: 0.05
      target_min: 0.0
      target_max: 127.0
      hidden_dims: [256, 128]
      
    # Advanced classification
    - name: "activity_level_predictor"
      target_key: "total_notes"
      head_type: "classification"
      loss_type: "ce"
      loss_weight: 0.05
      num_classes: 4  # Few, Some, Many, Very Many
      hidden_dims: [256, 128]
      
    - name: "tempo_category_predictor"
      target_key: "tempo_bpm"
      head_type: "classification"
      loss_type: "ce"
      loss_weight: 0.05
      num_classes: 4  # Slow, Moderate, Fast, Very Fast
      hidden_dims: [256, 128]
      
    # Multi-label instrument prediction
    - name: "instrument_ensemble_predictor"
      target_key: "unique_programs"
      head_type: "multi_label"
      loss_type: "bce"
      loss_weight: 0.1
      num_classes: 128  # All MIDI programs
      hidden_dims: [512, 256]
      
    # Binary features
    - name: "has_program_changes_predictor"
      target_key: "program_changes"  # Would need preprocessing to binary
      head_type: "classification"
      loss_type: "bce"
      loss_weight: 0.02
      num_classes: 2  # Has changes or not
      hidden_dims: [128, 64]
```

### Target Preprocessing Strategies

**For Binned Continuous Variables:**
```python
def preprocess_tempo_category(tempo_bpm: float) -> int:
    """Convert tempo to category."""
    if tempo_bpm < 90:
        return 0  # Slow
    elif tempo_bpm < 120:
        return 1  # Moderate  
    elif tempo_bpm < 160:
        return 2  # Fast
    else:
        return 3  # Very Fast

def preprocess_activity_level(total_notes: int) -> int:
    """Convert note count to activity level."""
    if total_notes < 100:
        return 0  # Few
    elif total_notes < 500:
        return 1  # Some
    elif total_notes < 1000:
        return 2  # Many
    else:
        return 3  # Very Many
```

**For Multi-label Instruments:**
```python
def preprocess_instruments(unique_programs: list[int]) -> torch.Tensor:
    """Convert program list to multi-hot encoding."""
    instrument_vector = torch.zeros(128)
    for program in unique_programs:
        if 0 <= program < 128:
            instrument_vector[program] = 1.0
    return instrument_vector
```

### Research Applications

**Musical Style Classification:**
- Combine tempo category, instrument ensemble, and rhythmic patterns for genre prediction
- Use harmonic features for classical vs. jazz vs. pop discrimination

**Arrangement Complexity:**
- Combine num_tracks, num_channels, instrument diversity for complexity scoring
- Use note density and velocity variation for dynamic range assessment

**Compositional Analysis:**
- Program changes and harmonic progressions for structural complexity
- Tempo stability and rhythmic patterns for groove analysis

This comprehensive set of targets provides a rich foundation for learning detailed musical representations from latent spaces. 