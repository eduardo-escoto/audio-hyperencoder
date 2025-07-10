import logging
from os import path, makedirs
from copy import deepcopy
from json import dump
from typing import Any

import torch
from torch import Tensor, no_grad
from torch import save as torch_save
from torch import int16 as torch_int16
from torch import float32 as torch_float32
from torch.nn import ModuleDict
from lightning import Callback, LightningModule
from torchaudio import save as ta_save
from torch.nn.utils import clip_grad_norm_
from safetensors.torch import save_file as st_save_file
from safetensors.torch import save_model as st_save_model
from lightning.pytorch.utilities import rank_zero_only
from stable_audio_tools.training.utils import (
    log_metric,
    create_optimizer_from_config,
    create_scheduler_from_config,
)
from stable_audio_tools.training.losses import (
    L1Loss,
    MSELoss,
    MultiLoss,
    HubertLoss,
    LossWithTarget,
)
from stable_audio_tools.training.autoencoders import (
    AudioAutoencoder,
    create_loss_modules_from_bottleneck,
)

from ..data import collate_dicts
from ..datamodels import ModelConfig, TrainingConfig, DemoConfig
from ..models.hyperencoder import HyperEncoder
from ..modules.auxiliary_heads import AuxiliaryHead, AuxiliaryHeadConfig
from ..modules.auxiliary_losses import AuxiliaryLoss

# Extract default values from configuration models
_DEFAULT_LEARNING_RATE = TrainingConfig.model_fields['learning_rate'].default
_DEFAULT_DEMO_EVERY = DemoConfig.model_fields['demo_every'].default
_DEFAULT_SAMPLE_RATE = DemoConfig.model_fields['sample_rate'].default
_DEFAULT_MAX_DEMOS = DemoConfig.model_fields['max_demos'].default


class HyperEncoderTrainingWrapper(LightningModule):
    def __init__(
        self,
        hyperencoder: HyperEncoder,
        loss_config: dict[str, Any] | None = None,
        optimizer_configs: dict[str, Any] | None = None,
        lr: float = _DEFAULT_LEARNING_RATE,
        clip_grad_norm: float = 0.0,
        auxiliary_heads_config: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.loss_config = loss_config
        self.hyperencoder = hyperencoder
        self.automatic_optimization = False
        self.clip_grad_norm = clip_grad_norm
        self.gen_loss_modules = []
        self.validation_step_outputs = []
        
        # Initialize auxiliary heads
        self.auxiliary_heads = ModuleDict()
        self.auxiliary_losses = []
        self.auxiliary_heads_config = auxiliary_heads_config or {}
        
        self.logger_inst = logging.getLogger(__name__)

        if optimizer_configs is None:
            optimizer_configs = {
                "hyperencoder": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {"lr": lr, "betas": (0.8, 0.99)},
                    }
                }
            }

        self.optimizer_configs = optimizer_configs

        if loss_config is None:
            loss_config = {
                "time": {
                    "type": "time",
                    "config": {},
                    "weights": {
                        "l1": 0.1,
                        "l2": 0.1,
                    },
                },
            }

        self.loss_config = loss_config

        if "hubert" in loss_config:
            hubert_weight = loss_config["hubert"]["weights"]["hubert"]
            if hubert_weight > 0:
                hubert_cfg = loss_config["hubert"].get("config", dict())
                self.hubert = HubertLoss(weight=1.0, **hubert_cfg)

                self.gen_loss_modules.append(
                    LossWithTarget(
                        self.hubert,
                        target_key="outer_latents",
                        input_key="reconstructed_outer_latents",
                        name="hubert_loss",
                        weight=hubert_weight,
                        decay=loss_config["hubert"].get("decay", 1.0),
                    )
                )

        if (
            "l1" in loss_config["time"]["weights"]
            and self.loss_config["time"]["weights"]["l1"] > 0.0
        ):
            self.gen_loss_modules.append(
                L1Loss(
                    key_a="outer_latents",
                    key_b="reconstructed_outer_latents",
                    weight=self.loss_config["time"]["weights"]["l1"],
                    name="l1_time_loss",
                    decay=self.loss_config["time"].get("decay", 1.0),
                )
            )

        if (
            "l2" in loss_config["time"]["weights"]
            and self.loss_config["time"]["weights"]["l2"] > 0.0
        ):
            self.gen_loss_modules.append(
                MSELoss(
                    key_a="outer_latents",
                    key_b="reconstructed_outer_latents",
                    weight=self.loss_config["time"]["weights"]["l2"],
                    name="l2_time_loss",
                    decay=self.loss_config["time"].get("decay", 1.0),
                )
            )

        if self.hyperencoder.bottleneck is not None:
            self.gen_loss_modules += create_loss_modules_from_bottleneck(
                self.hyperencoder.bottleneck, self.loss_config
            )

        # Initialize auxiliary heads if configured
        self._setup_auxiliary_heads()

        self.losses_gen = MultiLoss(self.gen_loss_modules)
        self.eval_losses = ModuleDict()
        # self.save_hyperparameters()

    def _setup_auxiliary_heads(self):
        """Initialize auxiliary heads and losses based on configuration."""
        from ..factories import create_auxiliary_heads_from_config
        
        self.auxiliary_heads, self.auxiliary_losses = create_auxiliary_heads_from_config(
            self.auxiliary_heads_config or {},
            self.hyperencoder.latent_dim,
        )
        
        # Add auxiliary losses to gen_loss_modules
        self.gen_loss_modules.extend(self.auxiliary_losses)
            
        if self.auxiliary_heads:
            self.logger_inst.info(f"Initialized {len(self.auxiliary_heads)} auxiliary heads")

    def forward(self, outer_latents):
        return self.__reconstruct__(outer_latents)

    def __reconstruct__(self, outer_latents, return_info=False):
        if return_info:
            inner_latents, info = self.hyperencoder.encode(
                outer_latents, return_info=True
            )
            info["inner_latents"] = inner_latents

            reconstructed_outer_latents = self.hyperencoder.decode(inner_latents)
            return reconstructed_outer_latents, info
        else:
            inner_latents = self.hyperencoder.encode(outer_latents, return_info=False)
            reconstructed_outer_latents = self.hyperencoder.decode(inner_latents)
            return reconstructed_outer_latents

    def validation_step(self, batch, batch_idx):
        outer_latents, info = batch

        loss_info = {}
        loss_info["outer_latents"] = outer_latents
        loss_info["info"] = info
        encoder_input = outer_latents

        loss_info["encoder_input"] = encoder_input

        with no_grad():
            reconstructed_outer_latents, info = self.__reconstruct__(
                outer_latents, return_info=True
            )
            loss_info.update(info)
            loss_info["decoder_output"] = reconstructed_outer_latents
            loss_info["reconstructed_outer_latents"] = reconstructed_outer_latents

            # Run evaluation metrics.
            val_loss_dict = {}
            for eval_key, eval_fn in self.eval_losses.items():
                loss_value = eval_fn(reconstructed_outer_latents, outer_latents)
                if eval_key == "sisdr":
                    loss_value = -loss_value
                if isinstance(loss_value, Tensor):
                    loss_value = loss_value.item()

                val_loss_dict[eval_key] = loss_value
                
            # Evaluate auxiliary heads if enabled
            if self.auxiliary_heads_config.get("validation_enabled", True):
                aux_metrics = self._evaluate_auxiliary_heads(loss_info, batch)
                val_loss_dict.update(aux_metrics)

        self.validation_step_outputs.append(val_loss_dict)
        return val_loss_dict

    def _evaluate_auxiliary_heads(self, loss_info: dict, batch: Any) -> dict:
        """Evaluate auxiliary heads during validation."""
        if not self.auxiliary_heads:
            return {}
            
        aux_metrics = {}
        outer_latents, info = batch
        
        # Ensure we have MIDI metadata
        if "midi_metadata" not in info:
            return aux_metrics
            
        midi_metadata = info["midi_metadata"]
        inner_latents = loss_info["inner_latents"]
        
        for head_name, head in self.auxiliary_heads.items():
            try:
                # Get target value
                target_key = head.config.target_key
                if target_key not in midi_metadata:
                    continue
                    
                target_value = midi_metadata[target_key]
                target = head.preprocess_target(target_value)
                target = target.to(inner_latents.device)
                
                # Get prediction
                prediction = head(inner_latents)
                
                # Compute appropriate metrics based on head type
                if head.config.head_type == "regression":
                    # Compute MAE and MSE
                    if prediction.dim() > 1:
                        prediction = prediction.squeeze()
                    if target.dim() > 0 and target.numel() == 1:
                        target = target.squeeze()
                    
                    # Expand target to match batch size if needed
                    if target.dim() == 0 and prediction.dim() == 1:
                        target = target.expand_as(prediction)
                    
                    mae = torch.nn.functional.l1_loss(prediction, target)
                    mse = torch.nn.functional.mse_loss(prediction, target)
                    
                    aux_metrics[f"aux/{head_name}_mae"] = mae.item()
                    aux_metrics[f"aux/{head_name}_mse"] = mse.item()
                    
                    # Denormalize for interpretable metrics
                    if head.config.target_min is not None and head.config.target_max is not None:
                        pred_denorm = head.denormalize_prediction(prediction)
                        target_denorm = head.denormalize_prediction(target)
                        mae_denorm = torch.nn.functional.l1_loss(pred_denorm, target_denorm)
                        aux_metrics[f"aux/{head_name}_mae_denorm"] = mae_denorm.item()
                        
                elif head.config.head_type == "classification":
                    # Compute accuracy
                    target = target.long()
                    if target.dim() == 0:
                        target = target.expand(prediction.size(0))
                    
                    pred_classes = torch.argmax(prediction, dim=1)
                    accuracy = (pred_classes == target).float().mean()
                    aux_metrics[f"aux/{head_name}_accuracy"] = accuracy.item()
                    
                elif head.config.head_type == "multi_label":
                    # Compute F1 score
                    target = target.float()
                    if target.dim() == 1 and prediction.dim() == 2:
                        target = target.unsqueeze(0).expand_as(prediction)
                    
                    pred_binary = torch.sigmoid(prediction) > 0.5
                    target_binary = target > 0.5
                    
                    tp = (pred_binary & target_binary).float().sum()
                    fp = (pred_binary & ~target_binary).float().sum()
                    fn = (~pred_binary & target_binary).float().sum()
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    
                    aux_metrics[f"aux/{head_name}_f1"] = f1.item()
                    aux_metrics[f"aux/{head_name}_precision"] = precision.item()
                    aux_metrics[f"aux/{head_name}_recall"] = recall.item()
                    
            except Exception as e:
                self.logger_inst.warning(f"Failed to evaluate auxiliary head {head_name}: {e}")
                
        return aux_metrics

    def on_validation_epoch_end(self):
        sum_loss_dict = {}
        for loss_dict in self.validation_step_outputs:
            for key, value in loss_dict.items():
                if key not in sum_loss_dict:
                    sum_loss_dict[key] = value
                else:
                    sum_loss_dict[key] += value

        for key, value in sum_loss_dict.items():
            val_loss = value / len(self.validation_step_outputs)
            val_loss = self.all_gather(val_loss)
            if hasattr(val_loss, "mean"):
                val_loss = val_loss.mean().item()
            else:
                val_loss = val_loss
            log_metric(self.logger, f"val/{key}", val_loss)

        self.validation_step_outputs.clear()  # free memory

    def training_step(self, batch, batch_idx):
        outer_latents, info = batch

        log_dict = {}
        loss_info = {"outer_latents": outer_latents, "info": info}

        encoder_input = outer_latents
        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        # Log detailed batch info periodically
        if batch_idx % 100 == 0:
            self.logger_inst.debug(f"🔄 Training step {batch_idx}: batch_shape={outer_latents.shape}, data_std={data_std:.4f}")

        reconstructed_outer_latents, info = self.__reconstruct__(
            outer_latents, return_info=True
        )
        loss_info.update(info)
        loss_info["decoder_output"] = reconstructed_outer_latents
        loss_info["reconstructed_outer_latents"] = reconstructed_outer_latents

        opt_gen = self.optimizers()
        sched_gen = self.lr_schedulers()

        loss, losses = self.losses_gen(loss_info)

        # Log loss breakdown periodically
        if batch_idx % 100 == 0:
            loss_breakdown = ", ".join([f"{k}: {v.item():.4f}" for k, v in losses.items()])
            self.logger_inst.debug(f"📊 Loss breakdown: {loss_breakdown}")

        opt_gen.zero_grad()
        self.manual_backward(loss)
        if self.clip_grad_norm > 0.0:
            clip_grad_norm_(self.hyperencoder.parameters(), self.clip_grad_norm)
        opt_gen.step()

        if sched_gen is not None:
            sched_gen.step()

        log_dict["train/loss"] = loss.detach().item()
        log_dict["train/latent_std"] = info["inner_latents"].std().detach().item()
        log_dict["train/data_std"] = data_std.detach().item()
        log_dict["train/gen_lr"] = opt_gen.param_groups[0]["lr"]

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach().item()

        # Log auxiliary head outputs if available
        if self.auxiliary_heads and batch_idx % 100 == 0:
            self.logger_inst.debug(f"🧠 Auxiliary heads active: {len(self.auxiliary_heads)} heads")

        self.log_dict(log_dict, prog_bar=True, on_step=True)

        return loss

    def configure_optimizers(self):
        gen_params = list(self.hyperencoder.parameters())

        opt_gen = create_optimizer_from_config(
            self.optimizer_configs["hyperencoder"]["optimizer"], gen_params
        )

        if "scheduler" in self.optimizer_configs["hyperencoder"]:
            sched_gen = create_scheduler_from_config(
                self.optimizer_configs["hyperencoder"]["scheduler"], opt_gen
            )
            return [opt_gen], [sched_gen]
        return [opt_gen]

    def export_model(self, path, use_safetensors=True):
        model = self.hyperencoder

        if use_safetensors:
            st_save_model(model, path)
        else:
            torch_save({"state_dict": model.state_dict()}, path)


def reload_he_training_wrapper_from_config_and_ckpt(
    config: ModelConfig, model: HyperEncoder, ckpt_path: str
):
    """Reload a HyperEncoder training wrapper from configuration and checkpoint.

    Args:
        config: ModelConfig containing training parameters
        model: HyperEncoder model instance
        ckpt_path: Path to checkpoint file

    Returns:
        HyperEncoderTrainingWrapper instance loaded from checkpoint
    """
    if not hasattr(config, "training") or config.training is None:
        raise ValueError("training config must be specified in model config")

    # Convert optimizer configs to the format expected by the training wrapper
    optimizer_configs = {}
    if config.training.optimizer_configs:
        for key, opt_config in config.training.optimizer_configs.items():
            optimizer_configs[key] = {
                "optimizer": opt_config.optimizer.model_dump(),
                "scheduler": opt_config.scheduler.model_dump()
                if opt_config.scheduler
                else None,
            }

    return HyperEncoderTrainingWrapper.load_from_checkpoint(
        ckpt_path,
        hyperencoder=model,
        loss_config=None,  # Will use default loss config
        optimizer_configs=optimizer_configs,
        # lr parameter will use default from TrainingConfig
    )


def create_he_training_wrapper_from_config(config: ModelConfig, model: HyperEncoder):
    """Create a HyperEncoder training wrapper from configuration.

    Args:
        config: ModelConfig containing training parameters
        model: HyperEncoder model instance

    Returns:
        HyperEncoderTrainingWrapper instance

    Examples:
        >>> from hyperencoder.datamodels import ModelConfig
        >>> config = ModelConfig()
        >>> model = create_hyperencoder_from_config(config)
        >>> wrapper = create_he_training_wrapper_from_config(config, model)
    """
    if not hasattr(config, "training") or config.training is None:
        raise ValueError("training config must be specified in model config")

    # Convert optimizer configs to the format expected by the training wrapper
    optimizer_configs = {}
    if config.training.optimizer_configs:
        for key, opt_config in config.training.optimizer_configs.items():
            optimizer_configs[key] = {
                "optimizer": opt_config.optimizer.model_dump(),
                "scheduler": opt_config.scheduler.model_dump()
                if opt_config.scheduler
                else None,
            }

    # Convert auxiliary heads config
    auxiliary_heads_config = None
    if hasattr(config, "auxiliary_heads") and config.auxiliary_heads is not None:
        auxiliary_heads_config = config.auxiliary_heads.model_dump()

    return HyperEncoderTrainingWrapper(
        model,
        loss_config=None,  # Will use default loss config
        optimizer_configs=optimizer_configs,
        lr=config.training.learning_rate,
        clip_grad_norm=config.training.gradient_clip_val,
        auxiliary_heads_config=auxiliary_heads_config,
    )


def create_training_wrapper(
    model: HyperEncoder,
    loss_config: dict[str, Any] | None = None,
    optimizer_configs: dict[str, Any] | None = None,
    lr: float = _DEFAULT_LEARNING_RATE,
    clip_grad_norm: float = 0.0,
    auxiliary_heads_config: dict[str, Any] | None = None,
) -> HyperEncoderTrainingWrapper:
    """Create a training wrapper with programmatic parameters.

    This is a convenience function for users who want to create training wrappers
    programmatically without using configuration files.

    Args:
        model: HyperEncoder model instance
        loss_config: Optional loss configuration dictionary
        optimizer_configs: Optional optimizer configuration dictionary
        lr: Learning rate
        clip_grad_norm: Gradient clipping value
        auxiliary_heads_config: Optional auxiliary heads configuration dictionary

    Returns:
        HyperEncoderTrainingWrapper instance

    Examples:
        >>> model = create_hyperencoder()
        >>> wrapper = create_training_wrapper(model, lr=5e-5)
    """
    return HyperEncoderTrainingWrapper(
        model,
        loss_config=loss_config,
        optimizer_configs=optimizer_configs,
        lr=lr,
        clip_grad_norm=clip_grad_norm,
        auxiliary_heads_config=auxiliary_heads_config,
    )


class AutoencoderDemoCallback(Callback):
    def __init__(
        self,
        demo_dl,
        pre_trained_autoencoder,
        demo_every=_DEFAULT_DEMO_EVERY,
        sample_rate=_DEFAULT_SAMPLE_RATE,
        max_demos=_DEFAULT_MAX_DEMOS,
    ):
        super().__init__()
        from itertools import cycle

        self.demo_every = demo_every
        self.demo_dl = cycle(deepcopy(demo_dl))
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.max_demos = max_demos
        self.pre_trained_autoencoder = pre_trained_autoencoder

    @rank_zero_only
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        log = logging.getLogger()

        log.debug(
            f"on_train_batch_end triggered at global_step={trainer.global_step}, "
            f"batch_idx={batch_idx}"
        )

        if (
            trainer.global_step - 1
        ) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            log.debug(
                f"Skipping demo generation at global_step={trainer.global_step}. "
                f"Last demo step: {self.last_demo_step}, demo_every: {self.demo_every}"
            )
            return

        self.last_demo_step = trainer.global_step
        module.eval()

        try:
            demo_outer_latents, info = next(self.demo_dl)
        except StopIteration:
            log.debug(
                "Caught StopIteration on the Demo DataLoader, seems we're still running into stale dataloader issue."
            )
            demo_outer_latents = None
            # enumerate(self.demo_dl)
            # demo_outer_latents, info = next(self.demo_dl)

        try:
            # Limit the number of demo samples
            if demo_outer_latents.shape[0] > self.max_demos:
                demo_outer_latents = demo_outer_latents[: self.max_demos, ...]
                info = info[: self.max_demos]

            log.debug(
                f"Prepared demo data with {demo_outer_latents.shape[0]} samples "
                f"at global_step={trainer.global_step}"
            )

            info = collate_dicts(info)

            encoder_input = demo_outer_latents
            encoder_input = encoder_input.to(module.device)

            demo_outer_latents = demo_outer_latents.to(module.device)

            pt_ae_model = self.pre_trained_autoencoder.to(module.device)
            with no_grad():
                inner_latents, inner_info = module.hyperencoder.encode(
                    encoder_input, return_info=True
                )
                reconstructed_outer_latents = module.hyperencoder.decode(inner_latents)
                reconstructed_audio = pt_ae_model.decode(reconstructed_outer_latents)

            reals_key = None
            if "cropped_reals" in info:
                reals_key = "cropped_reals"
                decoded_reals_key = "cropped_decoded_reals"
            else:
                reals_key = "trimmed_input_reals"
                decoded_reals_key = "decoded_reals"

            real_audio = info[reals_key].to(module.device)
            decoded_audio = info[decoded_reals_key].to(module.device)

            out_dict = {
                "demo_encoded_pre_bottleneck_latents": inner_info[
                    "pre_bottleneck_inner_latents"
                ]
                .contiguous()
                .cpu(),
                "demo_encoded_inner_latents": inner_latents.contiguous().cpu(),
                "demo_real_outer_latents": demo_outer_latents.contiguous().cpu(),
                "demo_reconstructed_outer_latents": reconstructed_outer_latents.contiguous().cpu(),
                "original_audio": real_audio.contiguous().cpu(),
                "sao_reconstructed_audio": decoded_audio.contiguous().cpu(),
                "hyperencoder_reconstructed_audio": reconstructed_audio.contiguous().cpu(),
            }

            out_infos = {
                "crop_start_pcts": info["crop_start_pct"],
                "crop_end_pcts": info["crop_end_pct"],
            }

            dict_data_path = path.join(
                trainer.logger.save_dir,
                trainer.logger.experiment.project,
                trainer.logger.experiment.id,
                "media",
                "demo_dicts",
            )

            log.debug("Saving demo dictionary")
            makedirs(dict_data_path, exist_ok=True)
            st_save_file(
                out_dict,
                path.join(
                    dict_data_path, f"demo_dict_{trainer.global_step:08}.safetensors"
                ),
            )

            with open(
                path.join(dict_data_path, f"demo_infos_{trainer.global_step:08}.json"),
                "w",
            ) as info_file:
                dump(out_infos, info_file)

            log.debug(
                f"Saved demo dictionary to {path.join(dict_data_path, f'demo_dict_{trainer.global_step:08}.safetensors')}"
            )

            for i in range(len(info["prefix"])):
                try:
                    direct_parent = info["root"][i].split("/")[-1]
                    prefix = info["prefix"][i]
                    data_dir = path.join(
                        trainer.logger.save_dir,
                        trainer.logger.experiment.project,
                        trainer.logger.experiment.id,
                        "media",
                        direct_parent,
                        prefix,
                    )
                    makedirs(data_dir, exist_ok=True)

                    def filename_gen(name, f_type, p):
                        return path.join(p, f"{name}_{trainer.global_step:08}.{f_type}")

                    og_pref = "real_audio"
                    pt_recon_pref = "pre_trained_ae_recon"
                    he_recon_pref = "hyperencoder_recon"
                    if "cropped_reals" in info:
                        crop_start_pct = info["crop_start_pct"][i]
                        crop_end_pct = info["crop_end_pct"][i]

                        og_pref += f"crop_{crop_start_pct}_{crop_end_pct}"
                        pt_recon_pref += f"crop_{crop_start_pct}_{crop_end_pct}"
                        he_recon_pref += f"crop_{crop_start_pct}_{crop_end_pct}"

                    og_filename = filename_gen(og_pref, "wav", data_dir)
                    pt_recon_filename = filename_gen(pt_recon_pref, "wav", data_dir)
                    hyper_recon_filename = filename_gen(he_recon_pref, "wav", data_dir)

                    def save_audio(filename, audio, sample_rate):
                        audio = (
                            audio.to(torch_float32)
                            .clamp(-1, 1)
                            .mul(32767)
                            .to(torch_int16)
                            .cpu()
                        )
                        ta_save(filename, audio, sample_rate)

                    save_audio(og_filename, real_audio[i], self.sample_rate)
                    save_audio(pt_recon_filename, decoded_audio[i], self.sample_rate)
                    save_audio(
                        hyper_recon_filename, reconstructed_audio[i], self.sample_rate
                    )

                    log.debug(
                        f"Saved audio files for demo sample {i} at global_step={trainer.global_step}: "
                        f"real_audio={og_filename}, pre_trained_recon={pt_recon_filename}, "
                        f"hyperencoder_recon={hyper_recon_filename}"
                    )

                except Exception as e:
                    log.error(
                        f"Error during demo generation at global_step={trainer.global_step}: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise e

        except Exception as e:
            log.error(
                f"Error during demo generation at global_step={trainer.global_step}: "
                f"{type(e).__name__}: {e}"
            )
            raise e
        finally:
            module.train()
