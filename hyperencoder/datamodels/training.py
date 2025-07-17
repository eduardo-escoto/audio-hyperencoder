from typing import Any

from pydantic import Field, BaseModel

strategy_kwargs_defaults = {
    "stage": 2,
    "contiguous_gradients": True,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 5e8,
    "allgather_bucket_size": 5e8,
    "load_full_weights": True,
}


class TrainingConfig(BaseModel):
    strategy: str = Field(description="The strategy for the training")
    strategy_kwargs: dict[str, Any] = Field(  # pyright: ignore[reportExplicitAny]
        description="The kwargs for the strategy", default=strategy_kwargs_defaults
    )

    num_gpus: int = Field(description="The number of GPUs for the training")
    # num_nodes: int = Field(description="The number of nodes for the training")
    # num_workers: int = Field(description="The number of workers for the training")
    # persistent_workers: bool = Field(description="Whether to use persistent workers for the training")
    pretrained_ckpt_path: str | None = Field(
        description="The path to the pretrained checkpoint", default=None
    )
    checkpoint_every: int = Field(
        description="The number of epochs between checkpoints"
    )
    save_top_k: int = Field(description="The number of checkpoints to save")
