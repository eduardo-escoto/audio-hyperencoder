"""Config conversion commands for the development utilities CLI.

This module provides commands for converting configuration files from
JSON/INI formats to YAML format.
"""

import json
import configparser
from pathlib import Path

import yaml
import typer
from rich.panel import Panel
from rich.progress import Progress, TextColumn, SpinnerColumn

from ..dev_cli import app, console


@app.command()
def convert(
    config_type: str = typer.Argument(..., help="Config type to convert (json|ini)"),
    input_path: Path = typer.Argument(..., help="Input config file path"),
    output_path: Path | None = typer.Option(None, help="Output YAML file path"),
) -> None:
    """🔄 Convert JSON/INI configs to YAML format."""

    if not input_path.exists():
        console.print(f"❌ [red]Input file not found: {input_path}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Converting config...", total=None)

        try:
            if config_type == "json":
                # Convert JSON to YAML
                with open(input_path) as f:
                    config_data = json.load(f)
            elif config_type == "ini":
                # Convert INI to YAML
                config = configparser.ConfigParser()
                config.read(input_path)
                config_data = {
                    section: dict(config.items(section))
                    for section in config.sections()
                }
            else:
                console.print(f"❌ [red]Unsupported config type: {config_type}[/red]")
                raise typer.Exit(1)

            progress.update(task, description="Writing YAML file...")

            if output_path is None:
                output_path = input_path.with_suffix(".yaml")

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

            progress.update(task, description="✅ Complete!")

        except Exception as e:
            console.print(f"❌ [red]Conversion failed: {str(e)}[/red]")
            raise typer.Exit(1)

    # Beautiful success message
    console.print(
        Panel.fit(
            f"[green]✅ Successfully converted {input_path} to {output_path}[/green]",
            title="🎉 Conversion Complete",
        )
    )
