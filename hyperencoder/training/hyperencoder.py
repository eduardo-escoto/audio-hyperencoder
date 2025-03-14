from os import path, makedirs
from copy import deepcopy
from typing import Optional

from torch import Tensor, no_grad
from torch import save as torch_save
from torch import int16 as torch_int16
from torch import float32 as torch_float32
from einops import rearrange
from torch.nn import ModuleDict
from lightning import Callback, LightningModule
from torchaudio import save as ta_save
from torch.nn.utils import clip_grad_norm_
from safetensors.torch import save_model as st_save_model
from models.hyperencoder import HyperEncoder
from lightning.pytorch.utilities import rank_zero_only
from stable_audio_open.training.utils import (
    log_audio,
    log_image,
    log_metric,
    log_point_cloud,
    logger_project_name,
    create_optimizer_from_config,
    create_scheduler_from_config,
)
from stable_audio_open.interface.aeiou import (
    audio_spectrogram_image,
    tokens_spectrogram_image,
)
from stable_audio_open.training.losses import (
    L1Loss,
    MSELoss,
    Multiloss,
    HubertLoss,
    LossWithTarget,
)
from stable_audio_open.training.autoencoders import (
    AudioAutoencoder,
    create_loss_modules_from_bottleneck,
)


class HyperEncoderTrainingWrapper(LightningModule):
    def __init__(
        self,
        hyperencoder: HyperEncoder,
        loss_config: Optional[dict] = None,
        optimizer_configs: Optional[dict] = None,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.loss_config = loss_config
        self.hyperencoder = hyperencoder

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

        self.losses_gen = Multiloss(self.gen_loss_modules)
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
            self.optimizer_configs["hyperencoder"], gen_params
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
        sample_size=65536,
        sample_rate=44100,
        max_demos=8,
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_samples = sample_size
        self.demo_dl = iter(deepcopy(demo_dl))
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.max_demos = max_demos

    @rank_zero_only
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if (
            trainer.global_step - 1
        ) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        self.last_demo_step = trainer.global_step
        module.eval()

        try:
            demo_outer_latents, info = next(self.demo_dl)

            # Limit the number of demo samples
            if demo_outer_latents.shape[0] > self.max_demos:
                demo_outer_latents = demo_outer_latents[: self.max_demos, ...]
                info = info[:, self.max_demos]

            encoder_input = demo_outer_latents
            encoder_input = encoder_input.to(module.device)

            demo_outer_latents = demo_outer_latents.to(module.device)

            with no_grad():
                inner_latents = module.hyperencoder.encode(encoder_input)
            # reconstructed_outer_latents = module.hyperencoder.decode(inner_latents)

            # log_dict = {}

            # Interleave reals and fakes
            reals_fakes = []
            # reals_fakes = rearrange([demo_reals, fakes], "i b d n -> (b i) d n")
            # Put the demos together
            reals_fakes = rearrange(reals_fakes, "b d n -> d (b n)")

            try:
                data_dir = path.join(
                    trainer.logger.save_dir,
                    logger_project_name(trainer.logger),
                    trainer.logger.experiment.id,
                    "media",
                )
                makedirs(data_dir, exist_ok=True)
                filename = path.join(data_dir, f"recon_{trainer.global_step:08}.wav")
            except Exception:
                filename = f"recon_{trainer.global_step:08}.wav"

            reals_fakes = (
                reals_fakes.to(torch_float32)
                .clamp(-1, 1)
                .mul(32767)
                .to(torch_int16)
                .cpu()
            )
            ta_save(filename, reals_fakes, self.sample_rate)

            log_audio(trainer.logger, "recon", filename, self.sample_rate)
            log_point_cloud(trainer.logger, "embeddings_3dpca", inner_latents)
            log_image(
                trainer.logger,
                "embeddings_spec",
                tokens_spectrogram_image(inner_latents),
            )
            log_image(
                trainer.logger,
                "recon_melspec_left",
                audio_spectrogram_image(reals_fakes),
            )
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            raise e
        finally:
            module.train()
