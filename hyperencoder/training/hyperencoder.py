import logging
from os import path, makedirs
from copy import deepcopy
from typing import Optional

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
from ..models.hyperencoder import HyperEncoder


def create_he_training_wrapper_from_config(model_config, model):
    model_type = model_config.get("model_type", None)
    assert model_type is not None, "model_type must be specified in model config"

    training_config = model_config.get("training", None)
    assert (
        training_config is not None
    ), "training config must be specified in model config"

    if model_type == "hyperencoder":
        return HyperEncoderTrainingWrapper(
            model,
            loss_config=training_config.get("loss_configs", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
            lr=training_config.get("learning_rate", None),
        )


class HyperEncoderTrainingWrapper(LightningModule):
    def __init__(
        self,
        hyperencoder: HyperEncoder,
        loss_config: Optional[dict] = None,
        optimizer_configs: Optional[dict] = None,
        lr: float = 1e-4,
        clip_grad_norm=0.0,
    ):
        super().__init__()
        self.loss_config = loss_config
        self.hyperencoder = hyperencoder
        self.automatic_optimization = False
        self.clip_grad_norm = clip_grad_norm
        self.gen_loss_modules = []
        self.validation_step_outputs = []

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

        self.losses_gen = MultiLoss(self.gen_loss_modules)
        self.eval_losses = ModuleDict()

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
        outer_latents, _ = batch

        loss_info = {}
        loss_info["outer_latents"] = outer_latents
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

        self.validation_step_outputs.append(val_loss_dict)
        return val_loss_dict

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
            val_loss = self.all_gather(val_loss).mean().item()
            log_metric(self.logger, f"val/{key}", val_loss)

        self.validation_step_outputs.clear()  # free memory

    def training_step(self, batch, batch_idx):
        outer_latents, _ = batch

        log_dict = {}
        loss_info = {"outer_latents": outer_latents}

        encoder_input = outer_latents
        loss_info["encoder_input"] = encoder_input

        data_std = encoder_input.std()

        reconstructed_outer_latents, info = self.__reconstruct__(
            outer_latents, return_info=True
        )
        loss_info.update(info)
        loss_info["decoder_output"] = reconstructed_outer_latents
        loss_info["reconstructed_outer_latents"] = reconstructed_outer_latents

        opt_gen = self.optimizers()
        sched_gen = self.lr_schedulers()

        loss, losses = self.losses_gen(loss_info)

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


class AutoencoderDemoCallback(Callback):
    def __init__(
        self,
        demo_dl,
        pre_trained_autoencoder: AudioAutoencoder,
        demo_every=2000,
        sample_rate=44100,
        max_demos=8,
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_dl = iter(deepcopy(demo_dl))
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

            real_audio = info["trimmed_input_reals"].to(module.device)
            decoded_audio = info["decoded_reals"].to(module.device)

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

                    og_filename = filename_gen("real_audio", "wav", data_dir)
                    pt_recon_filename = filename_gen(
                        "pre_trained_ae_recon", "wav", data_dir
                    )
                    hyper_recon_filename = filename_gen(
                        "hyperencoder_recon", "wav", data_dir
                    )

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
