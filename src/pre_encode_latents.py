import os
import pathlib
import json
import torch
import torchaudio
import safetensors.torch
import warnings

from safetensors.torch import save_file as sf_save_file

from argparse import ArgumentParser
from tqdm import tqdm

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
parser.add_argument('-n', '--njobs', help='number of jobs')
parser.add_argument('-j', '--job', help='job number')
parser.add_argument('-p', '--pathfile', help='pathfile')

def get_device(device_num = 0):
    return torch.device(f"cuda:{device_num}") if torch.cuda.is_available() else "cpu"

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

def load_model(device, model_config = None):
    # Download model
    model, pretrained_model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    model = model.to(device)
    
    return model.pretransform


def get_input_files(input_dir, file_name='*.wav', batch_pattern = r"Track\d*", batched = True, path_file = None):
    if path_file is not None:
        file_paths = ''
        with open(path_file, 'r') as path_file:
            file_paths = path_file.read()
        file_paths = file_paths.split('\n')
        return file_paths
    
    input_path: pathlib.Path = pathlib.Path(input_dir)

    files =  [p.resolve() for p in sorted(list(input_path.glob(f"**/{file_name}")))]
    from collections import defaultdict
    batch_dict = defaultdict(list)
    import regex 
    r = regex.compile(batch_pattern)

    for file in files:
        batch_folder = r.findall(str(file))[0]
        if batch_folder !=  '':
            batch_dict[batch_folder].append(file)
    return batch_dict


def get_path_up_to_n_parents(path, n):
    """Gets the file path up to a certain number of parent directories."""
    out_path = "/"
    for _ in range(n):
        path = path.parent
        out_path = "/" + path.name + out_path
    return pathlib.Path(out_path)

def get_path_up_to_regex(path, regex_str = r"Track\d*"):
    """Gets the file path up to a certain number of parent directories."""
    import regex
    r = regex.compile(regex_str)
    
    full_path = path.resolve()
    iter_path = path.resolve()

    out_path = ""
    while out_path != str(full_path.parent) and r.search(out_path) is None:
        iter_path = iter_path.parent
        out_path = "/" + iter_path.name + out_path
    
    return pathlib.Path(out_path)

def process_batches(device, reload_pretransform, batches, output_dir_path: pathlib.Path, loop_offset = None, n_jobs = None, parent_level = 1):
    with open(output_dir_path / f'failures.log', 'w') as fail_file:
        for batch_name, file_paths in tqdm(batches.items(), total = len(list(batches.keys()))):
            print(f'Processing: {batch_name}')
            
            batch_tensors = []
            sample_rates = []

            for file in file_paths:
                try:
                    waveform, sample_rate = torchaudio.load(file)
                    batch_tensors.append(waveform)
                    sample_rates.append(sample_rate)
                except Exception as e: 
                    print('File Failed')
                    print(e)
                    fail_file.write(f'{str(file)}\r\n')
            
            batch_size = 1
            sub_batches = []
            for i in range(0, len(batch_tensors), batch_size):
                sub_batch_tensors = batch_tensors[i:i+batch_size if i+batch_size < len(batch_tensors) else len(batch_tensors)]
                sub_sample_rates = sample_rates[i:i+batch_size if i+batch_size < len(batch_tensors) else len(batch_tensors)]

                preprocessed_audio = reload_pretransform.model.preprocess_audio_list_for_encoder(sub_batch_tensors, sub_sample_rates) 
                preprocessed_audio = preprocessed_audio.to(device)            

                latents = reload_pretransform.model.encode_audio(preprocessed_audio)
                cpu_latents = latents.to('cpu')
                torch.cuda.empty_cache()

                sub_batches.append(cpu_latents)

            cpu_latents = torch.cat(sub_batches, dim=0)

            out_dict = {}
            for i in range(cpu_latents.shape[0]):
                out_dict[file_paths[i].stem] = cpu_latents[i]

            output_fp = pathlib.Path(f"{batch_name}_latent.safetensors")
            out_path = pathlib.Path(f"{str(output_dir_path.resolve())}/{str(output_fp)}")

            print(f'Outputting: {out_path.resolve()}')
            if not out_path.parent.exists():
                print(f'Creating directory:{out_path.parent}')
                out_path.parent.mkdir(parents=True, exist_ok=True)

            sf_save_file(out_dict, out_path.absolute())



if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    args = parser.parse_args()
    device = get_device(device_num=args.device) if args.device is not None else get_device()
    
    from os import environ as get_env

    token = args.token
    if args.token is None:
        token = get_env.get('HF_TOKEN')

    login_to_hf(token=token)
    
    print("getting input files")
    batches = get_input_files(args.input_dir, batched = True, path_file=args.pathfile)
    
    print('loading model')
    model = load_model(device)
    
    output_path = pathlib.Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir()
    print(output_path.resolve())

    njobs = int(args.njobs) if args.njobs is not None else None
    loop_offset=int(args.job) if args.job is not None else None

    process_batches(device, model, batches, output_path, loop_offset=args.job, n_jobs=njobs)