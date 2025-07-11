"""
Lightning training module for hyperencoder.

This module implements the PyTorch Lightning training loop for hyperencoder models,
including loss computation, optimization, and logging.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..models.hyperencoder import HyperEncoder
from ..factories.model_factory import create_hyperencoder_from_config
from ..factories.auxiliary_head_factory import create_auxiliary_heads_from_config
from ..modules.auxiliary_losses import AuxiliaryLoss


class HyperEncoderLightningModule(LightningModule):
    """
    Lightning module for training hyperencoder models.
    
    This module handles the complete training pipeline including:
    - Model initialization from configuration
    - Loss computation (reconstruction + auxiliary losses)
    - Optimization and learning rate scheduling
    - Logging and monitoring
    - Inference and evaluation
    """
    
    def __init__(
        self,
        model_config: DictConfig,
        training_config: DictConfig,
        demo_config: Optional[DictConfig] = None,
    ):
        """
        Initialize the hyperencoder training module.
        
        Args:
            model_config: Configuration for the hyperencoder model
            training_config: Configuration for training parameters
            demo_config: Configuration for demo/evaluation settings
        """
        super().__init__()
        
        # Store configurations
        self.model_config = model_config
        self.training_config = training_config
        self.demo_config = demo_config or DictConfig({})
        
        # Initialize logger
        self.logger_instance = logging.getLogger(__name__)
        
        # Initialize model from configuration
        self.model = self._create_model()
        
        # Initialize auxiliary heads if configured
        self.auxiliary_heads = None
        self.auxiliary_losses = []
        if model_config.get("auxiliary_heads", {}).get("enabled", False):
            self._setup_auxiliary_heads()
        
        # Initialize loss weights
        self.reconstruction_weight = training_config.get("reconstruction_weight", 1.0)
        self.auxiliary_weight = training_config.get("auxiliary_weight", 0.1)
        
        # Store training metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # Log model configuration
        self.logger_instance.info(f"Initialized hyperencoder with config: {dict(model_config)}")
    
    def _create_model(self) -> HyperEncoder:
        """Create the hyperencoder model from configuration."""
        try:
            model = create_hyperencoder_from_config(self.model_config)
            self.logger_instance.info(
                f"Created hyperencoder model with {sum(p.numel() for p in model.parameters())} parameters"
            )
            return model
        except Exception as e:
            self.logger_instance.error(f"Failed to create model: {e}")
            raise
    
    def _setup_auxiliary_heads(self) -> None:
        """Set up auxiliary prediction heads if configured."""
        try:
            auxiliary_config = self.model_config.get("auxiliary_heads", {})
            latent_dim = self.model_config.get("latent_dim", 4)
            
            self.auxiliary_heads, self.auxiliary_losses = create_auxiliary_heads_from_config(
                auxiliary_config, latent_dim
            )
            
            self.logger_instance.info(
                f"Created {len(self.auxiliary_heads)} auxiliary heads"
            )
        except Exception as e:
            self.logger_instance.error(f"Failed to setup auxiliary heads: {e}")
            raise
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the hyperencoder.
        
        Args:
            x: Input tensor (batch_size, channels, time)
            
        Returns:
            Dictionary containing model outputs
        """
        # Encode to inner latents
        inner_latents, encode_info = self.model.encode(x, return_info=True)
        
        # Decode back to outer latents
        reconstructed = self.model.decode(inner_latents)
        
        outputs = {
            "input": x,
            "inner_latents": inner_latents,
            "reconstructed": reconstructed,
            "encode_info": encode_info,
        }
        
        # Add auxiliary predictions if available
        if self.auxiliary_heads is not None:
            aux_predictions = {}
            for head_name, head in self.auxiliary_heads.items():
                aux_predictions[head_name] = head(inner_latents)
            outputs["auxiliary_predictions"] = aux_predictions
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute the total loss including reconstruction and auxiliary losses.
        
        Args:
            outputs: Model outputs from forward pass
            batch: Batch data containing targets
            
        Returns:
            Dictionary containing loss components
        """
        losses = {}
        
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(
            outputs["reconstructed"], 
            outputs["input"]
        )
        losses["reconstruction_loss"] = reconstruction_loss
        
        # Auxiliary losses
        total_aux_loss = torch.tensor(0.0, device=self.device)
        if self.auxiliary_losses:
            for aux_loss in self.auxiliary_losses:
                aux_loss_value = aux_loss(outputs, batch)
                losses[aux_loss.name] = aux_loss_value
                total_aux_loss += aux_loss_value
        
        losses["auxiliary_loss"] = total_aux_loss
        
        # Total weighted loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.auxiliary_weight * total_aux_loss
        )
        losses["total_loss"] = total_loss
        
        return losses
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step for one batch."""
        try:
            # Get input latents
            input_latents = batch["latents"]
            
            # Forward pass
            outputs = self(input_latents)
            
            # Compute losses
            losses = self.compute_loss(outputs, batch)
            
            # Log training metrics
            self.log("train_loss", losses["total_loss"], on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_reconstruction_loss", losses["reconstruction_loss"], on_step=True, on_epoch=True)
            
            if "auxiliary_loss" in losses:
                self.log("train_auxiliary_loss", losses["auxiliary_loss"], on_step=True, on_epoch=True)
            
            # Store outputs for epoch end
            self.training_step_outputs.append({
                "loss": losses["total_loss"].detach(),
                "reconstruction_loss": losses["reconstruction_loss"].detach(),
                "auxiliary_loss": losses.get("auxiliary_loss", torch.tensor(0.0)).detach(),
            })
            
            return losses["total_loss"]
            
        except Exception as e:
            self.logger_instance.error(f"Training step failed at batch {batch_idx}: {e}")
            raise
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step for one batch."""
        try:
            # Get input latents
            input_latents = batch["latents"]
            
            # Forward pass
            outputs = self(input_latents)
            
            # Compute losses
            losses = self.compute_loss(outputs, batch)
            
            # Log validation metrics
            self.log("val_loss", losses["total_loss"], on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_reconstruction_loss", losses["reconstruction_loss"], on_step=False, on_epoch=True)
            
            if "auxiliary_loss" in losses:
                self.log("val_auxiliary_loss", losses["auxiliary_loss"], on_step=False, on_epoch=True)
            
            # Store outputs for epoch end
            self.validation_step_outputs.append({
                "loss": losses["total_loss"].detach(),
                "reconstruction_loss": losses["reconstruction_loss"].detach(),
                "auxiliary_loss": losses.get("auxiliary_loss", torch.tensor(0.0)).detach(),
            })
            
            return losses["total_loss"]
            
        except Exception as e:
            self.logger_instance.error(f"Validation step failed at batch {batch_idx}: {e}")
            raise
    
    def on_training_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        if self.training_step_outputs:
            # Calculate average metrics
            avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
            avg_recon_loss = torch.stack([x["reconstruction_loss"] for x in self.training_step_outputs]).mean()
            avg_aux_loss = torch.stack([x["auxiliary_loss"] for x in self.training_step_outputs]).mean()
            
            # Log epoch metrics
            self.logger_instance.info(
                f"Epoch {self.current_epoch} - "
                f"Train Loss: {avg_loss:.4f}, "
                f"Reconstruction: {avg_recon_loss:.4f}, "
                f"Auxiliary: {avg_aux_loss:.4f}"
            )
            
            # Clear stored outputs
            self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of each validation epoch."""
        if self.validation_step_outputs:
            # Calculate average metrics
            avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
            avg_recon_loss = torch.stack([x["reconstruction_loss"] for x in self.validation_step_outputs]).mean()
            avg_aux_loss = torch.stack([x["auxiliary_loss"] for x in self.validation_step_outputs]).mean()
            
            # Log epoch metrics
            self.logger_instance.info(
                f"Epoch {self.current_epoch} - "
                f"Val Loss: {avg_loss:.4f}, "
                f"Reconstruction: {avg_recon_loss:.4f}, "
                f"Auxiliary: {avg_aux_loss:.4f}"
            )
            
            # Clear stored outputs
            self.validation_step_outputs.clear()
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        try:
            # Get optimization configuration
            optimizer_config = self.training_config.get("optimizer", {})
            scheduler_config = self.training_config.get("scheduler", {})
            
            # Create optimizer
            optimizer_type = optimizer_config.get("type", "Adam")
            learning_rate = optimizer_config.get("lr", 1e-4)
            weight_decay = optimizer_config.get("weight_decay", 0.0)
            
            if optimizer_type == "Adam":
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    betas=optimizer_config.get("betas", (0.9, 0.999))
                )
            elif optimizer_type == "AdamW":
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    betas=optimizer_config.get("betas", (0.9, 0.999))
                )
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
            
            # Create scheduler if configured
            scheduler_type = scheduler_config.get("type")
            if scheduler_type is None:
                return optimizer
            
            if scheduler_type == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=scheduler_config.get("factor", 0.5),
                    patience=scheduler_config.get("patience", 10),
                    verbose=True
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch",
                    },
                }
            elif scheduler_type == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.get("step_size", 30),
                    gamma=scheduler_config.get("gamma", 0.1)
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler,
                }
            else:
                raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
            
        except Exception as e:
            self.logger_instance.error(f"Failed to configure optimizers: {e}")
            raise
    
    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        input_latents = batch["latents"]
        outputs = self(input_latents)
        return outputs
