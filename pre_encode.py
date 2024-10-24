import os
import pathlib
import json
import torch
import torchaudio
import safetensors.torch
from typing import List
from argparse import ArgumentParser

from stable_audio_tools.models.factory import create_pretransform_from_config
from stable_audio_tools import get_pretrained_model


parser = ArgumentParser(
    prog='audio_pre_encoder',
    description='pre-encodes audio to latents using stable audio'
)

parser.add_argument('-i', '--input-dir', required=True, help='data directory of files to pre-encode')
parser.add_argument('-t', '--token', required=False, help='token for hugging face login')
parser.add_argument('-o', '--output-dir', required=True, help='directory to output to')
parser.add_argument('-d', '--device', help='cuda device number')

def get_device(device_num = 0):
    return f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"

def login_to_hf(token: str = None):
    from huggingface_hub import login
    if token is None:
        login()
    else:
        login(token)

def load_model_config(path):
    with open(path) as f:
        pretransform_config = json.load(f)
        return pretransform_config

def load_model(device, model_config):
    # Download model
    model, pretrained_model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    # sample_rate = pretrained_model_config["sample_rate"]
    # sample_size = pretrained_model_config["sample_size"]

    model = model.to(device)

    # Save pretransform
    pretransform = model.pretransform
    pretransform_state_dict = model.pretransform.state_dict()

    file_path = 'pretransform.safetensors'
    safetensors.torch.save_file(pretransform_state_dict, file_path)

    # Create the pretransform model from the configuration
    reload_pretransform = create_pretransform_from_config(model_config, sample_rate=model_config["sample_rate"])

    # delete original model to free up space
    import gc
    model.to('cpu')
    del model
    gc.collect()
    torch.cuda.empty_cache() 


    reload_pretransform = reload_pretransform.to(device)

    # Check if the original pretransform and the reloaded pretransform are of the same type
    assert type(pretransform) == type(reload_pretransform)

    # Apply the state dictionary to the pretransform model
    state_dict = safetensors.torch.load_file(file_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('model.', '')  # 'model.'
        new_state_dict[new_key] = value

    reload_pretransform.load_state_dict(new_state_dict)
    
    return reload_pretransform


def get_input_files(input_dir, file_name='mix.wav'):
    input_path: pathlib.Path = pathlib.Path(input_dir)

    return list(input_path.glob(f"**/{file_name}"))

def process_files(reload_pretransform, file_paths, output_dir_path ):
    for file in file_paths:
        print(f'Processing: {file}')
        waveform, sample_rate = torchaudio.load(file)

        waveform = waveform.to(device=device)
        preprocessed_audio = reload_pretransform.model.preprocess_audio_for_encoder(waveform, sample_rate)
        # print(preprocessed_audio.shape)
        latent = reload_pretransform.encode(preprocessed_audio)
        output_file_name = f"{str(file.parents[0]).split('/')[-1].lower()}_latent.npy"
        out_path = output_dir_path /  output_file_name
        print(f'Outputting: {out_path}')
        with open(out_path.absolute(), 'wb') as outfile:
            latent.cpu().numpy().tofile(outfile)


if __name__ == '__main__':
    args = parser.parse_args()
    device = get_device(device_num=args.device) if args.device is not None else get_device()
    
    login_to_hf(token=args.token)
    
    input_files = get_input_files(args.input_dir)
    model_config = load_model_config('./model_config.json')
    model = load_model(device, model_config)

    output_path = pathlib.Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir()

    process_files(model, input_files, output_path)






