"""Generation commands for the development utilities CLI.

This module provides commands for generating configuration files and JSON schemas.
"""

from pathlib import Path

import typer
from rich.table import Table

from ..dev_cli import console, generate_app


@generate_app.command("configs")
def generate_configs(
    output_dir: Path | None = typer.Option(
        None, help="Output directory (defaults to package configs)"
    ),
) -> None:
    """📄 Generate default configuration files."""
    # Generate configs using the comprehensive config generation module
    from ..core.config_generation import generate_all_configs

    console.print("🏗️ Generating configuration files...", style="bold blue")

    try:
        # Generate all configs - use package directory if no output specified
        results = generate_all_configs(
            output_dir=output_dir, console=console, show_progress=True
        )

        # Exit with error code if any configs failed
        if results["failed"] > 0:
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"❌ [red]Failed to import config generation module: {e}[/red]")
        raise typer.Exit(1)


@generate_app.command("schemas")
def generate_schemas(
    output_dir: Path | None = typer.Option(
        None, help="Output directory (defaults to package schemas)"
    ),
) -> None:
    """📋 Generate JSON schemas for configuration validation."""
    # Generate JSON schemas using the comprehensive schema generation module
    from ..core.schema_generation import generate_all_schemas

    console.print("🏗️ Generating JSON schemas...", style="bold blue")

    try:
        # Use schemas subdirectory if output_dir is specified
        schema_output_dir = output_dir / "schemas" if output_dir else None

        # Generate all schemas with progress tracking
        results = generate_all_schemas(
            output_dir=schema_output_dir, console=console, show_progress=True
        )

        # Exit with error code if any schemas failed
        if results["failed"] > 0:
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"❌ [red]Failed to import schema generation module: {e}[/red]")
        raise typer.Exit(1)


@generate_app.command("all")
def generate_all(
    output_dir: Path | None = typer.Option(
        None, help="Output directory for both configs and schemas"
    ),
) -> None:
    """🚀 Generate both configuration files and JSON schemas."""
    # Generate both configs and schemas
    from ..core.config_generation import generate_all_configs
    from ..core.schema_generation import generate_all_schemas

    console.print("🚀 Generating both configurations and schemas...", style="bold blue")

    config_results = None
    schema_results = None
    overall_success = True

    try:
        # Generate configs first
        if output_dir:
            configs_dir = output_dir / "configs"
            console.print(
                f"\n📄 Step 1: Generating configuration files to {configs_dir}...",
                style="bold cyan",
            )
        else:
            configs_dir = None
            console.print(
                "\n📄 Step 1: Generating configuration files to package directory...",
                style="bold cyan",
            )

        config_results = generate_all_configs(
            output_dir=configs_dir, console=console, show_progress=True
        )

        if config_results["failed"] > 0:
            overall_success = False
            console.print(
                f"⚠️ Config generation had {config_results['failed']} errors",
                style="yellow",
            )
        else:
            console.print(
                "✅ Configuration generation completed successfully!", style="green"
            )

    except Exception as e:
        console.print(f"❌ [red]Config generation failed: {e}[/red]")
        overall_success = False

    try:
        # Generate schemas second
        if output_dir:
            schemas_dir = output_dir / "schemas"
            console.print(
                f"\n📋 Step 2: Generating JSON schemas to {schemas_dir}...",
                style="bold cyan",
            )
        else:
            schemas_dir = None
            console.print(
                "\n📋 Step 2: Generating JSON schemas to package directory...",
                style="bold cyan",
            )

        schema_results = generate_all_schemas(
            output_dir=schemas_dir, console=console, show_progress=True
        )

        if schema_results["failed"] > 0:
            overall_success = False
            console.print(
                f"⚠️ Schema generation had {schema_results['failed']} errors",
                style="yellow",
            )
        else:
            console.print("✅ Schema generation completed successfully!", style="green")

    except Exception as e:
        console.print(f"❌ [red]Schema generation failed: {e}[/red]")
        overall_success = False

    # Display overall summary
    if config_results and schema_results:
        overall_table = Table(title="🎯 Complete Generation Summary")
        overall_table.add_column("Component", style="cyan")
        overall_table.add_column("Files Generated", style="green")
        overall_table.add_column("Status", style="magenta")

        config_status = (
            "✅ Success"
            if config_results["failed"] == 0
            else f"⚠️ {config_results['failed']} errors"
        )
        schema_status = (
            "✅ Success"
            if schema_results["failed"] == 0
            else f"⚠️ {schema_results['failed']} errors"
        )

        overall_table.add_row(
            "Configuration Files",
            f"{config_results['successful']}/"
            f"{config_results['successful'] + config_results['failed']}",
            config_status,
        )
        overall_table.add_row(
            "JSON Schemas",
            f"{schema_results['successful']}/{schema_results['total']}",
            schema_status,
        )

        console.print("\n")
        console.print(overall_table)

    if not overall_success:
        console.print(
            "\n⚠️ [yellow]Generation completed with some errors. "
            "Check the output above for details.[/yellow]"
        )
        raise typer.Exit(1)
    else:
        console.print(
            "\n🎉 [green]All generation tasks completed successfully![/green]"
        )
