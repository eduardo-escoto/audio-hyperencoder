from argparse import ArgumentParser

from .pre_encode_audio import main

parser = ArgumentParser(
    prog="audio_pre_encoder",
    description="pre-encodes audio to latents using stable audio",
)

parser.add_argument(
    "-t", "--token", required=False, help="token for hugging face login"
)
parser.add_argument(
    "-i", "--input-dir", required=True, help="data directory of files to pre-encode"
)
parser.add_argument("-o", "--output-dir", required=True, help="directory to output to")
parser.add_argument(
    "-n", "--n-devices", required=False, help="number of devices to use"
)

if __name__ == "__main__":
    main(parser.parse_args())
