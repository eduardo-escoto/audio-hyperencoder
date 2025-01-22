from json import dump as json_dump
from json import load as json_load
from pathlib import Path

from stable_audio_tools import get_pretrained_model


def get_model_config(
    model_name, cache_dir="./model_cache", force_download=False, hf_token=None
):
    # huggingface names usually have a slash to indicate repo + model name
    sanitized_model_name = model_name.replace("/", "-")
    config_path = Path(f"{cache_dir}/{sanitized_model_name}.json")

    model_config = None

    # load from cache or download and create cache
    if config_path.exists() and not force_download:
        with open(config_path) as config_file:
            model_config = json_load(config_file)
    else:
        from huggingface_hub import login

        # login to huggingface if token is provided, else it'll use env_variable
        if hf_token is not None:
            login(token=hf_token)

        # download model + model_config from huggingface
        _, model_config = get_pretrained_model(model_name)

        # create the file
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.touch()

        # dump model config into cache
        with open(config_path, "w") as config_file:
            json_dump(model_config, config_file, indent=4)

    return model_config
