"""Setup commands for IDE integration and development tools.

This module provides commands for setting up and tearing down IDE integrations
such as VS Code JSON schema validation.
"""

import json
import shutil
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from ..dev_cli import console, setup_app


@setup_app.command("vscode")
def setup_vscode(
    schemas_dir: Path | None = typer.Option(
        None, help="Directory containing JSON schemas (defaults to './schemas')"
    ),
    dry_run: bool = typer.Option(False, help="Preview changes without writing files"),
) -> None:
    """🔧 Setup VS Code with JSON schema validation for YAML configs."""

    # Default schemas directory
    if schemas_dir is None:
        schemas_dir = Path("./schemas")

    # Ensure schemas directory exists
    if not schemas_dir.exists():
        console.print(f"❌ [red]Schema directory not found: {schemas_dir}[/red]")
        console.print(
            f"💡 [yellow]Run 'uv run hyperencoder-utils generate schemas "
            f"--output-dir {schemas_dir.parent}' first[/yellow]"
        )
        raise typer.Exit(1)

    # Check for existing schema files
    schema_files = list(schemas_dir.glob("*.schema.json"))
    if not schema_files:
        console.print(f"❌ [red]No schema files found in {schemas_dir}[/red]")
        console.print(
            f"💡 [yellow]Run 'uv run hyperencoder-utils generate schemas "
            f"--output-dir {schemas_dir.parent}' first[/yellow]"
        )
        raise typer.Exit(1)

    console.print(
        f"🔧 Setting up VS Code integration with {len(schema_files)} schemas...",
        style="bold blue",
    )

    # Create .vscode directory if it doesn't exist
    vscode_dir = Path(".vscode")
    if not dry_run:
        vscode_dir.mkdir(exist_ok=True)

    settings_file = vscode_dir / "settings.json"

    # Load existing settings or create new ones
    existing_settings = {}
    if settings_file.exists():
        try:
            with open(settings_file) as f:
                existing_settings = json.load(f)
        except json.JSONDecodeError:
            console.print(
                f"⚠️ [yellow]Invalid JSON in {settings_file}, creating backup[/yellow]"
            )
            if not dry_run:
                backup_file = settings_file.with_suffix(".json.backup")
                settings_file.rename(backup_file)
                console.print(f"💾 [cyan]Backup saved to {backup_file}[/cyan]")

    # Create schema mappings
    schema_mappings = existing_settings.get("yaml.schemas", {})

    # Map each schema to appropriate file patterns
    for schema_file in schema_files:
        schema_name = schema_file.stem.replace(".schema", "")
        relative_schema_path = str(schemas_dir / schema_file.name)

        # Create file patterns based on schema type
        patterns = _get_schema_patterns(schema_name)
        schema_mappings[relative_schema_path] = patterns

    # Update settings
    updated_settings = existing_settings.copy()
    updated_settings["yaml.schemas"] = schema_mappings

    # Display what will be changed
    console.print(
        f"\n📋 Schema mappings to be {'applied' if not dry_run else 'previewed'}:",
        style="bold cyan",
    )

    schema_table = Table()
    schema_table.add_column("Schema", style="cyan")
    schema_table.add_column("File Patterns", style="green")

    for schema_path, patterns in schema_mappings.items():
        schema_name = Path(schema_path).stem.replace(".schema", "")
        schema_table.add_row(schema_name, ", ".join(patterns))

    console.print(schema_table)

    if dry_run:
        console.print(
            "\n🔍 [yellow]Dry run mode - no files would be modified[/yellow]"
        )
        console.print(f"💾 [cyan]Settings would be written to: {settings_file}[/cyan]")
    else:
        # Write updated settings
        with open(settings_file, "w") as f:
            json.dump(updated_settings, f, indent=2, sort_keys=True)

        console.print("\n✅ [green]VS Code settings updated successfully![/green]")
        console.print(f"💾 [cyan]Settings saved to: {settings_file}[/cyan]")

        # Show helpful next steps
        console.print(
            Panel.fit(
                f"[green]🎉 VS Code integration configured![/green]\n"
                f"[cyan]📋 {len(schema_files)} JSON schemas mapped to YAML "
                f"patterns[/cyan]\n"
                f"[yellow]💡 VS Code will now provide validation and autocomplete "
                f"for your config files[/yellow]\n"
                f"[blue]📝 Edit YAML files in VS Code to see schema validation "
                f"in action[/blue]\n"
                f"[blue]🔧 Install 'YAML' extension by Red Hat if not already "
                f"installed[/blue]",
                title="🚀 VS Code Ready",
            )
        )


@setup_app.command("teardown-vscode")
def teardown_vscode(
    schemas_dir: Path | None = typer.Option(
        None,
        help="Directory containing schemas to remove mappings for "
        "(defaults to './schemas')",
    ),
    dry_run: bool = typer.Option(False, help="Preview changes without writing files"),
    backup: bool = typer.Option(True, help="Create backup before modifying settings"),
) -> None:
    """🔧 Remove VS Code schema mappings for hyperencoder configs."""

    # Default schemas directory
    if schemas_dir is None:
        schemas_dir = Path("./schemas")

    vscode_dir = Path(".vscode")
    settings_file = vscode_dir / "settings.json"

    # Check if settings file exists
    if not settings_file.exists():
        console.print(
            f"ℹ️ [blue]VS Code settings file not found: {settings_file}[/blue]"
        )
        console.print("💡 [yellow]No VS Code integration to remove[/yellow]")
        return

    console.print(
        "🔧 Removing VS Code schema mappings for hyperencoder configs...",
        style="bold blue",
    )

    # Load existing settings
    try:
        with open(settings_file) as f:
            settings = json.load(f)
    except json.JSONDecodeError:
        console.print(f"❌ [red]Invalid JSON in {settings_file}[/red]")
        console.print("💡 [yellow]Cannot parse settings file[/yellow]")
        raise typer.Exit(1)

    # Get current schema mappings
    schema_mappings = settings.get("yaml.schemas", {})
    if not schema_mappings:
        console.print(
            "ℹ️ [blue]No yaml.schemas section found in VS Code settings[/blue]"
        )
        console.print("💡 [yellow]No schema mappings to remove[/yellow]")
        return

    # Find mappings that point to our schemas directory
    schemas_dir_str = str(schemas_dir)
    hyperencoder_mappings = {}
    other_mappings = {}

    for schema_path, patterns in schema_mappings.items():
        if schema_path.startswith(schemas_dir_str) and schema_path.endswith(
            ".schema.json"
        ):
            hyperencoder_mappings[schema_path] = patterns
        else:
            other_mappings[schema_path] = patterns

    if not hyperencoder_mappings:
        console.print(
            f"ℹ️ [blue]No hyperencoder schema mappings found for {schemas_dir}[/blue]"
        )
        console.print("💡 [yellow]No mappings to remove[/yellow]")
        return

    # Show what will be removed
    console.print(
        f"\n📋 Schema mappings to be "
        f"{'removed' if not dry_run else 'previewed for removal'}:",
        style="bold cyan",
    )

    removal_table = Table()
    removal_table.add_column("Schema", style="cyan")
    removal_table.add_column("File Patterns", style="yellow")

    for schema_path, patterns in hyperencoder_mappings.items():
        schema_name = Path(schema_path).stem.replace(".schema", "")
        removal_table.add_row(schema_name, ", ".join(patterns))

    console.print(removal_table)

    if other_mappings:
        console.print(
            "\n📋 Other schema mappings will be preserved:", style="bold green"
        )

        preserve_table = Table()
        preserve_table.add_column("Schema", style="cyan")
        preserve_table.add_column("File Patterns", style="green")

        for schema_path, patterns in other_mappings.items():
            schema_name = Path(schema_path).name
            preserve_table.add_row(schema_name, ", ".join(patterns))

        console.print(preserve_table)

    if dry_run:
        console.print(
            "\n🔍 [yellow]Dry run mode - no files would be modified[/yellow]"
        )
        console.print(f"💾 [cyan]Settings would be updated in: {settings_file}[/cyan]")
        if not other_mappings:
            console.print(
                "⚠️ [yellow]yaml.schemas section would be removed entirely[/yellow]"
            )
    else:
        # Create backup if requested
        if backup:
            backup_file = settings_file.with_suffix(".json.backup")
            try:
                shutil.copy2(settings_file, backup_file)
                console.print(f"💾 [cyan]Backup created: {backup_file}[/cyan]")
            except Exception as e:
                console.print(f"⚠️ [yellow]Failed to create backup: {e}[/yellow]")

        # Update settings
        updated_settings = settings.copy()

        if other_mappings:
            # Keep other mappings, only remove hyperencoder ones
            updated_settings["yaml.schemas"] = other_mappings
        else:
            # Remove yaml.schemas section entirely if no other mappings
            updated_settings.pop("yaml.schemas", None)

        # Write updated settings
        with open(settings_file, "w") as f:
            json.dump(updated_settings, f, indent=2, sort_keys=True)

        console.print(
            "\n✅ [green]VS Code schema mappings removed successfully![/green]"
        )
        console.print(f"💾 [cyan]Settings updated: {settings_file}[/cyan]")

        # Show helpful summary
        if other_mappings:
            console.print(
                Panel.fit(
                    f"[green]🎉 Hyperencoder schema mappings removed![/green]\n"
                    f"[cyan]🗑️ Removed {len(hyperencoder_mappings)} hyperencoder "
                    f"schema mappings[/cyan]\n"
                    f"[blue]📋 Preserved {len(other_mappings)} other schema "
                    f"mappings[/blue]\n"
                    f"[yellow]💡 VS Code will no longer validate hyperencoder "
                    f"YAML configs[/yellow]",
                    title="🔧 Teardown Complete",
                )
            )
        else:
            console.print(
                Panel.fit(
                    f"[green]🎉 All schema mappings removed![/green]\n"
                    f"[cyan]🗑️ Removed {len(hyperencoder_mappings)} schema "
                    f"mappings[/cyan]\n"
                    f"[blue]📋 yaml.schemas section removed from VS Code "
                    f"settings[/blue]\n"
                    f"[yellow]💡 VS Code schema validation is now disabled[/yellow]",
                    title="🔧 Teardown Complete",
                )
            )


def _get_schema_patterns(schema_name: str) -> list[str]:
    """Get file patterns for a given schema name.

    Args:
        schema_name: Name of the schema without .schema suffix

    Returns:
        List of file patterns that should use this schema
    """
    pattern_map = {
        "train_task_config": ["configs/train.yaml", "*/train.yaml", "train.yaml"],
        "pre_encode_task_config": [
            "configs/pre_encode.yaml",
            "*/pre_encode.yaml",
            "pre_encode.yaml",
        ],
        "data_config": ["configs/data/*.yaml", "*/data/*.yaml", "data/*.yaml"],
        "model_config": ["configs/model/*.yaml", "*/model/*.yaml", "model/*.yaml"],
        "training_config": [
            "configs/training/*.yaml",
            "*/training/*.yaml",
            "training/*.yaml",
        ],
        "pre_encode_config": [
            "configs/pre_encode/*.yaml",
            "*/pre_encode/*.yaml",
            "pre_encode/*.yaml",
        ],
        "hydra_config": ["configs/hydra/*.yaml", "*/hydra/*.yaml", "hydra/*.yaml"],
    }

    if schema_name in pattern_map:
        return pattern_map[schema_name]

    # Generic pattern for other schemas
    base_name = schema_name.replace("_config", "")
    return [f"configs/{base_name}/*.yaml", f"*/{base_name}/*.yaml"]
