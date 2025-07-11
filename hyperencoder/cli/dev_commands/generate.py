"""
Generate commands for development CLI.

This module provides utilities for generating configuration files and other
development resources for the hyperencoder project.
"""

import logging
from pathlib import Path
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def generate_sample_config(output_dir: Path = Path("configs")) -> None:
    """Generate a sample configuration file.
    
    Args:
        output_dir: Directory to write the sample config to
    """
    sample_config = {
        "model": {
            "latent_dim": 4,
            "in_channels": 64,
            "out_channels": 64,
            "encoder": {
                "type": "basic",
                "hidden_dims": [256, 128, 64]
            },
            "decoder": {
                "type": "basic", 
                "hidden_dims": [64, 128, 256]
            }
        },
        "training": {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "max_epochs": 100,
            "optimizer": {
                "type": "Adam",
                "betas": [0.9, 0.999],
                "weight_decay": 0.0
            }
        },
        "data": {
            "dataset_path": "data/latents",
            "batch_size": 32,
            "num_workers": 4
        }
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "sample_config.yaml"
    
    import yaml
    with open(output_file, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    console.print(f"✅ Generated sample config: {output_file}")


def generate_readme_template(output_dir: Path = Path(".")) -> None:
    """Generate a README template.
    
    Args:
        output_dir: Directory to write the README template to
    """
    readme_content = """# Audio Hyperencoder

A PyTorch Lightning implementation of hyperencoder models for audio representation learning.

## Installation

```bash
pip install -e .
```

## Usage

### Training

```bash
python -m hyperencoder.cli.ml_tasks.train
```

### Pre-encoding

```bash
python -m hyperencoder.cli.ml_tasks.pre_encode
```

## Configuration

Configuration is managed through Hydra. See `configs/` directory for examples.

## Development

This project uses modern Python development practices:

- **uv** for dependency management
- **PyTorch Lightning** for training
- **Hydra** for configuration management
- **Rich** for beautiful CLI output

## License

MIT License
"""
    
    output_file = output_dir / "README_template.md"
    
    with open(output_file, 'w') as f:
        f.write(readme_content)
    
    console.print(f"✅ Generated README template: {output_file}")


def show_generation_options() -> None:
    """Display available generation options."""
    table = Table(title="Available Generation Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    
    table.add_row("config", "Generate sample configuration files")
    table.add_row("readme", "Generate README template")
    
    console.print(table)


def generate_command(target: str, output_dir: str = ".") -> None:
    """Main generate command dispatcher.
    
    Args:
        target: What to generate ('config', 'readme', etc.)
        output_dir: Directory to write generated files to
    """
    output_path = Path(output_dir)
    
    if target == "config":
        generate_sample_config(output_path)
    elif target == "readme":
        generate_readme_template(output_path)
    elif target == "help":
        show_generation_options()
    else:
        console.print(f"❌ Unknown generation target: {target}")
        console.print("Available targets: config, readme, help")
        return
    
    console.print(Panel(
        f"Generation completed successfully!\nOutput directory: {output_path}",
        title="✅ Success",
        style="green"
    ))
