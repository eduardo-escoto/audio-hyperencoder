from typing import Any, final

from lightning import Trainer, LightningModule
from typing_extensions import override
from lightning.pytorch.callbacks import Callback

from hyperencoder.datamodels.modeling import ModelingConfig


class ExceptionCallback(Callback):
    @override
    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ):
        print(f"{type(exception).__name__}: {exception}")


@final
class ModelConfigEmbedderCallback(Callback):
    def __init__(self, model_config: ModelingConfig): 
        self.model_config = model_config

    @override
    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any] # pyright: ignore[reportExplicitAny]
    ):  
        checkpoint["model_config"] = self.model_config
