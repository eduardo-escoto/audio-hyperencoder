"""
Configuration generation for hyperencoder CLI.

This module generates default configuration files from Pydantic datamodels,
ensuring that the models are the single source of truth for all configurations.
"""

import logging
from typing import Any, Union, TypedDict
from pathlib import Path

import yaml
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, TextColumn, SpinnerColumn

# Import all the Pydantic datamodels that are our single source of truth
from hyperencoder.datamodels import (
    DataConfig,
    HydraConfig,
    ModelConfig,
    TrainingConfig,
    PreEncodeConfig,
    TrainTaskConfig,
    PreEncodeTaskConfig,
)

# Type for Hydra defaults - can be a string (like "_self_") or a dict (like {"data": "hyperencoder"})
HydraDefault = Union[str, dict[str, str]]

logger = logging.getLogger(__name__)
console = Console()


class CustomYAMLDumper(yaml.SafeDumper):
    """Custom YAML dumper that handles Hydra defaults lists properly."""

    def represent_list(self, data):
        """Custom representation for lists to handle defaults properly."""
        # Convert to list if needed and check if this is a defaults list
        data_list = list(data) if not isinstance(data, list) else data
        if data_list and isinstance(data_list[0], (str, dict)):
            # Format as proper YAML list for Hydra
            return self.represent_sequence(
                "tag:yaml.org,2002:seq", data_list, flow_style=False
            )
        return super().represent_list(data)

    def represent_dict(self, data):
        """Custom representation for dictionaries."""
        return self.represent_mapping("tag:yaml.org,2002:map", data, flow_style=False)


# Register custom representers
CustomYAMLDumper.add_representer(list, CustomYAMLDumper.represent_list)
CustomYAMLDumper.add_representer(dict, CustomYAMLDumper.represent_dict)


def get_bundled_configs_path() -> Path:
    """Get the path to bundled configs within the package.

    Returns:
        Path to the bundled configs directory within the CLI package
    """
    # Use the CLI configs directory as the proper package location
    cli_configs_path = Path(__file__).parent / "configs"
    cli_configs_path.mkdir(exist_ok=True)
    return cli_configs_path


class ConfigGenerationResult(TypedDict):
    """Result of config generation operation."""

    config_name: str
    output_path: str
    success: bool
    error: str | None


class ConfigGenerationSummary(TypedDict):
    """Summary of all config generation operations."""

    total_configs: int
    successful: int
    failed: int
    results: list[ConfigGenerationResult]


def get_config_definitions() -> dict[str, dict[str, Any]]:
    """Get all configuration definitions from Pydantic models.

    Uses the Pydantic models as the single source of truth by instantiating
    them with their default values and dumping to dictionaries.

    Returns:
        Dictionary mapping config names to their definitions
    """
    return {
        # Main task configs (top-level entry points)
        "train": TrainTaskConfig().model_dump(),
        "pre_encode": PreEncodeTaskConfig().model_dump(),
        # Config groups
        "hydra/default": HydraConfig().model_dump(),
        "data/default": DataConfig().model_dump(),
        "model/default": ModelConfig().model_dump(),
        "training/default": TrainingConfig().model_dump(),
        "pre_encode/default": PreEncodeConfig().model_dump(),
    }


def apply_naming_convention(config_name: str, config_dir: Path) -> str:
    """Apply naming convention to config files.

    Args:
        config_name: Name of the config (e.g., "data/default")
        config_dir: Directory containing the config files

    Returns:
        Appropriate filename following the naming convention
    """
    # Special case: main Hydra config stays as config.yaml
    if config_name == "config":
        return "config.yaml"

    # Special case: train config stays as train.yaml
    if config_name == "train":
        return "train.yaml"

    # Extract the base name from the config path
    base_name = Path(config_name).name

    # For nested configs, use the base name as the filename
    return f"{base_name}.yaml"


def generate_single_config(
    config_name: str, config_data: dict[str, Any], output_dir: Path
) -> ConfigGenerationResult:
    """Generate a single configuration file.

    Args:
        config_name: Name of the configuration
        config_data: Configuration data to write
        output_dir: Output directory for the config file

    Returns:
        Result of the generation operation
    """
    try:
        # Determine the output path
        if "/" in config_name:
            # Nested config (e.g., "data/default")
            parent_dir = output_dir / Path(config_name).parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            filename = apply_naming_convention(config_name, output_dir)
            output_path = parent_dir / filename
        else:
            # Top-level config
            filename = apply_naming_convention(config_name, output_dir)
            output_path = output_dir / filename

        # Convert any Path objects to strings for YAML compatibility
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj

        yaml_data = convert_paths(config_data)

        # Write the YAML file using custom dumper
        with open(output_path, "w") as f:
            yaml.dump(
                yaml_data,
                f,
                Dumper=CustomYAMLDumper,
                default_flow_style=False,
                indent=2,
            )

        return ConfigGenerationResult(
            config_name=config_name,
            output_path=str(output_path),
            success=True,
            error=None,
        )

    except Exception as e:
        logger.error(f"Failed to generate config {config_name}: {e}")
        return ConfigGenerationResult(
            config_name=config_name, output_path="", success=False, error=str(e)
        )


def generate_all_configs(
    output_dir: Path | None = None,
    console: Console | None = None,
    show_progress: bool = True,
) -> ConfigGenerationSummary:
    """Generate all configuration files from Pydantic models.

    Args:
        output_dir: Directory to write configuration files to. If None, uses package configs directory
        console: Optional console for output
        show_progress: Whether to show progress bar

    Returns:
        Summary of the generation process
    """
    # Use provided console or create new one
    if console is None:
        console = Console()

    # Use current working directory configs folder if no output dir specified
    if output_dir is None:
        output_dir = Path("./configs")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all config definitions
    config_definitions = get_config_definitions()

    console.print(
        f"🏗️ Generating {len(config_definitions)} configurations from Pydantic models...",
        style="bold blue",
    )
    console.print(f"📁 Output directory: {output_dir}", style="cyan")

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Generating configs...", total=len(config_definitions))

        for config_name, config_data in config_definitions.items():
            progress.update(task, description=f"Generating {config_name}...")

            result = generate_single_config(config_name, config_data, output_dir)
            results.append(result)

            progress.advance(task)

    # Calculate summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    summary = ConfigGenerationSummary(
        total_configs=len(results),
        successful=successful,
        failed=failed,
        results=results,
    )

    # Display results
    _display_generation_results(summary, console)

    return summary


def _display_generation_results(
    summary: ConfigGenerationSummary, console: Console
) -> None:
    """Display the results of config generation."""

    if summary["successful"] == summary["total_configs"]:
        console.print(
            f"✅ [green]Successfully generated all {summary['total_configs']} configurations![/green]"
        )
    else:
        console.print(
            f"⚠️ [yellow]Generated {summary['successful']}/{summary['total_configs']} configurations ({summary['failed']} failed)[/yellow]"
        )

        # Show failed configs
        failed_configs = [r for r in summary["results"] if not r["success"]]
        if failed_configs:
            console.print("\n❌ [red]Failed configurations:[/red]")
            for result in failed_configs:
                console.print(f"  - {result['config_name']}: {result['error']}")

    # Show success table
    if summary["successful"] > 0:
        table = Table(title="📋 Generated Configuration Files")
        table.add_column("Config Name", style="cyan")
        table.add_column("Output Path", style="magenta")
        table.add_column("Status", style="green")

        for result in summary["results"]:
            if result["success"]:
                table.add_row(
                    result["config_name"], result["output_path"], "✅ Success"
                )

        console.print(table)


def detect_generated_config_files(directory: Path) -> list[Path]:
    """Detect generated configuration files for cleaning purposes.

    Args:
        directory: Directory to search for generated config files

    Returns:
        List of generated config file paths
    """
    generated_files = []

    if not directory.exists():
        return generated_files

    # Main config file
    main_config = directory / "config.yaml"
    if main_config.exists():
        generated_files.append(main_config)

    # Task config files (main entry points)
    train_config = directory / "train.yaml"
    if train_config.exists():
        generated_files.append(train_config)

    pre_encode_config = directory / "pre_encode.yaml"
    if pre_encode_config.exists():
        generated_files.append(pre_encode_config)

    # Find all default.yaml files in subdirectories
    for pattern in ["**/default.yaml"]:
        generated_files.extend(directory.glob(pattern))

    return generated_files
