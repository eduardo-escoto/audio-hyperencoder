# Running the Hyperencoder Trainig

## Vanilla Python

To run: 
```sh
python -m hyperencoder.train
```

To run with a specific wandb name:

```sh
python -m hyperencoder.train --config-file  --name run_name
```

To run with a specific training config:
```sh
python -m hyperencoder.train --config-file ./config.ini
```

Recovering a run:
```sh
python -m hyperencoder.train --ckpt-path ./path_to_ckpt --run-id wandb-runid
```

## Running with UV

To run: 
```sh
uv run --env-file .env python -m hyperencoder.train
```

To run with a specific wandb name:

```sh
uv run --env-file .env python -m hyperencoder.train --config-file  --name run_name
```

To run with a specific training config:
```sh
uv run --env-file .env python -m hyperencoder.train --config-file ./config.ini
```

Recovering a run:
```sh
uv run --env-file .env python -m hyperencoder.train --ckpt-path ./path_to_ckpt --run-id wandb-runid
```