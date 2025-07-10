"""Schema generation module for the hyperencoder CLI.

This module provides reusable functions for generating JSON schemas from
Pydantic configuration models, with enhanced features for CLI integration
including progress tracking, custom output directories, and rich console output.
"""

import json
from typing import TypedDict
from pathlib import Path

from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    SpinnerColumn,
    TaskProgressColumn,
)

# Import all config classes that need schema generation
from hyperencoder.datamodels import (
    BaseConfig,
    # Import additional configs from data_config
    CropConfig,
    DataConfig,
    DemoConfig,
    # Import additional configs from hydra_config
    TaskConfig,
    HydraConfig,
    ModelConfig,
    DatasetEntry,
    DecoderConfig,
    # Import additional configs from model_config
    EncoderConfig,
    TrainingConfig,
    OptimizerConfig,
    PreEncodeConfig,
    SchedulerConfig,
    TrainTaskConfig,
    BottleneckConfig,
    ExperimentConfig,
    TrainingTaskConfig,
    PreEncodeTaskConfig,
    OptimizerSchedulerConfig,
)


class SchemaGenerationResults(TypedDict):
    """Results from schema generation process."""

    total: int
    successful: int
    failed: int
    schemas: list[str]
    errors: list[str]


def get_config_classes() -> list[tuple[type[BaseConfig], str]]:
    """Get all configuration classes that need schema generation.

    Returns:
        List of tuples containing (config_class, schema_filename)
    """
    return [
        # Base configuration
        (BaseConfig, "base_config.schema.json"),
        # Core config groups
        (TrainingConfig, "training_config.schema.json"),
        (PreEncodeConfig, "pre_encode_config.schema.json"),
        (DataConfig, "data_config.schema.json"),
        (ModelConfig, "model_config.schema.json"),
        (HydraConfig, "hydra_config.schema.json"),
        # Task configurations (main entry points)
        (TrainTaskConfig, "train_task_config.schema.json"),
        (PreEncodeTaskConfig, "pre_encode_task_config.schema.json"),
        # Additional hydra configs
        (TaskConfig, "task_config.schema.json"),
        (TrainingTaskConfig, "training_task_config.schema.json"),
        (ExperimentConfig, "experiment_config.schema.json"),
        # Model sub-configurations
        (EncoderConfig, "encoder_config.schema.json"),
        (DecoderConfig, "decoder_config.schema.json"),
        (BottleneckConfig, "bottleneck_config.schema.json"),
        (OptimizerConfig, "optimizer_config.schema.json"),
        (SchedulerConfig, "scheduler_config.schema.json"),
        (OptimizerSchedulerConfig, "optimizer_scheduler_config.schema.json"),
        (DemoConfig, "demo_config.schema.json"),
        # Data sub-configurations
        (CropConfig, "crop_config.schema.json"),
        (DatasetEntry, "dataset_entry.schema.json"),
    ]


def generate_schema(
    config_class: type[BaseConfig], output_path: Path, console: Console | None = None
) -> bool:
    """Generate JSON schema for a Pydantic model.

    Args:
        config_class: The Pydantic model class to generate schema for
        output_path: Path where the schema will be saved
        console: Optional rich console for output

    Returns:
        True if schema generation was successful, False otherwise
    """
    if console is None:
        console = Console()

    try:
        # Generate the schema
        schema = config_class.model_json_schema()

        # Add custom properties for better IDE support
        schema["$id"] = f"https://hyperencoder.ai/schemas/{output_path.name}"
        schema["$comment"] = (
            f"Generated from {config_class.__module__}.{config_class.__name__}"
        )

        # Add title if not present
        if "title" not in schema:
            schema["title"] = config_class.__name__

        # Add description if not present
        if (
            "description" not in schema
            and hasattr(config_class, "__doc__")
            and config_class.__doc__
        ):
            schema["description"] = config_class.__doc__.strip()

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write schema with pretty formatting
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, sort_keys=True, ensure_ascii=False)

        console.print(f"✅ Generated schema: {output_path.name}", style="green")
        return True

    except Exception as e:
        console.print(
            f"❌ Error generating schema for {config_class.__name__}: {e}", style="red"
        )
        return False


def create_schema_summary(
    schemas_dir: Path,
    configs: list[tuple[type[BaseConfig], str]],
    console: Console | None = None,
) -> bool:
    """Create a summary README file listing all generated schemas.

    Args:
        schemas_dir: Directory containing the schemas
        configs: List of config classes and their schema filenames
        console: Optional rich console for output

    Returns:
        True if summary creation was successful, False otherwise
    """
    if console is None:
        console = Console()

    try:
        summary_path = schemas_dir / "README.md"

        content = """# JSON Schemas for Hyperencoder Configuration

This directory contains JSON schemas generated from Pydantic configuration models.
These schemas enable IDE validation and autocomplete for YAML configuration files.

## Available Schemas

"""

        for config_class, filename in configs:
            content += f"- **{config_class.__name__}**: [`{filename}`](./{filename})\n"
            if hasattr(config_class, "__doc__") and config_class.__doc__:
                doc_lines = config_class.__doc__.strip().split("\n")
                if doc_lines:
                    content += f"  - {doc_lines[0]}\n"
            content += "\n"

        content += """## Usage in YAML Files

Add the following line at the top of your YAML configuration files:

```yaml
# yaml-language-server: $schema=../schemas/your_config.schema.json
```

## Auto-generated

⚠️ **These files are auto-generated.** Do not edit them manually.
Run `uv run hyperencoder-utils generate schemas` to regenerate them.
"""

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(content)

        console.print(
            f"✅ Generated schema summary: {summary_path.name}", style="green"
        )
        return True

    except Exception as e:
        console.print(f"❌ Error creating schema summary: {e}", style="red")
        return False


def generate_all_schemas(
    output_dir: Path | None = None,
    console: Console | None = None,
    show_progress: bool = True,
) -> SchemaGenerationResults:
    """Generate all JSON schemas for configuration models.

    Args:
        output_dir: Directory to save schemas in. If None, uses package schemas directory
        console: Optional rich console for output
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with generation results including success count and details
    """
    if console is None:
        console = Console()

    # Use current working directory schemas folder if no output dir specified
    if output_dir is None:
        schemas_dir = Path("./schemas")
    else:
        schemas_dir = output_dir

    schemas_dir.mkdir(parents=True, exist_ok=True)

    # Get all config classes
    configs = get_config_classes()

    console.print(
        "🚀 Generating JSON schemas for hyperencoder configuration models...",
        style="bold blue",
    )

    results: SchemaGenerationResults = {
        "total": len(configs),
        "successful": 0,
        "failed": 0,
        "schemas": [],
        "errors": [],
    }

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating schemas...", total=len(configs))

            for config_class, filename in configs:
                try:
                    output_path = schemas_dir / filename
                    progress.update(task, description=f"Generating {filename}...")

                    if generate_schema(config_class, output_path, console):
                        results["successful"] += 1
                        results["schemas"].append(filename)
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"Failed to generate {filename}")

                    progress.update(task, advance=1)

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(f"Error with {filename}: {str(e)}")
                    progress.update(task, advance=1)
    else:
        # Generate without progress bar
        for config_class, filename in configs:
            try:
                output_path = schemas_dir / filename
                console.print(f"Generating {filename}...", style="cyan")

                if generate_schema(config_class, output_path, console):
                    results["successful"] += 1
                    results["schemas"].append(filename)
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to generate {filename}")

            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"Error with {filename}: {str(e)}")

    # Create summary file
    if create_schema_summary(schemas_dir, configs, console):
        console.print(f"✅ Schema summary created in {schemas_dir}", style="green")

    # Display results table
    results_table = Table(title="📋 Schema Generation Results")
    results_table.add_column("Schema", style="cyan")
    results_table.add_column("Status", style="magenta")

    for filename in results["schemas"]:
        results_table.add_row(filename, "✅ Success")

    for error in results["errors"]:
        results_table.add_row(error, "❌ Failed")

    console.print(results_table)

    # Summary message
    if results["successful"] == results["total"]:
        console.print(
            Panel.fit(
                f"[green]🎉 Successfully generated {results['successful']}/{results['total']} JSON schemas![/green]\n"
                f"[blue]📁 Schemas saved to: {schemas_dir}[/blue]",
                title="✅ Schema Generation Complete",
            )
        )
    else:
        console.print(
            Panel.fit(
                f"[yellow]⚠️ Generated {results['successful']}/{results['total']} schemas[/yellow]\n"
                f"[red]{results['failed']} schemas failed to generate[/red]\n"
                f"[blue]📁 Schemas saved to: {schemas_dir}[/blue]",
                title="⚠️ Schema Generation Completed with Errors",
            )
        )

    return results


def validate_schema_file(schema_path: Path, console: Console | None = None) -> bool:
    """Validate a JSON schema file.

    Args:
        schema_path: Path to the schema file
        console: Optional rich console for output

    Returns:
        True if schema is valid, False otherwise
    """
    if console is None:
        console = Console()

    try:
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        # Basic validation - check required fields
        required_fields = ["$schema", "type"]
        for field in required_fields:
            if field not in schema:
                console.print(f"❌ Schema missing required field: {field}", style="red")
                return False

        console.print(f"✅ Schema validation passed: {schema_path.name}", style="green")
        return True

    except json.JSONDecodeError as e:
        console.print(f"❌ Invalid JSON in schema {schema_path.name}: {e}", style="red")
        return False
    except Exception as e:
        console.print(
            f"❌ Error validating schema {schema_path.name}: {e}", style="red"
        )
        return False


def list_generated_schemas(
    schemas_dir: Path, console: Console | None = None
) -> list[Path]:
    """List all generated schema files in a directory.

    Args:
        schemas_dir: Directory containing schemas
        console: Optional rich console for output

    Returns:
        List of schema file paths
    """
    if console is None:
        console = Console()

    if not schemas_dir.exists():
        console.print(f"❌ Schemas directory not found: {schemas_dir}", style="red")
        return []

    schema_files = list(schemas_dir.glob("*.schema.json"))

    if not schema_files:
        console.print(f"ℹ️  No schema files found in {schemas_dir}", style="yellow")
        return []

    console.print(f"📋 Found {len(schema_files)} schema files:", style="blue")
    for schema_file in schema_files:
        console.print(f"  - {schema_file.name}", style="cyan")

    return schema_files
