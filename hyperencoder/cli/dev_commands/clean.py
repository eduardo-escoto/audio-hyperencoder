"""Cleaning commands for removing generated files and artifacts.

This module provides commands for cleaning generated configuration files,
JSON schemas, and other development artifacts.
"""

from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from ..dev_cli import console, clean_app


@clean_app.command("schemas")
def clean_schemas(
    schemas_dir: Path | None = typer.Option(
        None, help="Directory containing schemas to clean (defaults to './schemas')"
    ),
    dry_run: bool = typer.Option(
        False, help="Preview what would be deleted without actually deleting"
    ),
) -> None:
    """🧹 Clean generated JSON schema files."""

    if schemas_dir is None:
        schemas_dir = Path("./schemas")

    if not schemas_dir.exists():
        console.print(f"ℹ️ [blue]Schema directory doesn't exist: {schemas_dir}[/blue]")
        return

    # Find all schema files
    schema_files = list(schemas_dir.glob("*.schema.json"))
    readme_file = schemas_dir / "README.md"

    files_to_delete = schema_files.copy()
    if readme_file.exists():
        files_to_delete.append(readme_file)

    if not files_to_delete:
        console.print(f"ℹ️ [blue]No schema files found in {schemas_dir}[/blue]")
        return

    console.print(
        f"🧹 Found {len(files_to_delete)} files to clean in {schemas_dir}...",
        style="bold blue",
    )

    # Display files to be deleted
    delete_table = Table(title="Files to Delete")
    delete_table.add_column("File", style="cyan")
    delete_table.add_column("Type", style="magenta")

    for file in files_to_delete:
        file_type = "Schema" if file.suffix == ".json" else "Documentation"
        delete_table.add_row(str(file.name), file_type)

    console.print(delete_table)

    if dry_run:
        console.print("\n🔍 [yellow]Dry run mode - no files would be deleted[/yellow]")
    else:
        # Delete files
        deleted_count = 0
        for file in files_to_delete:
            try:
                file.unlink()
                deleted_count += 1
                console.print(f"🗑️ Deleted: {file.name}", style="dim")
            except Exception as e:
                console.print(f"❌ [red]Failed to delete {file.name}: {e}[/red]")

        # Remove empty directory if all files were deleted
        if deleted_count == len(files_to_delete):
            try:
                if not any(schemas_dir.iterdir()):  # Check if directory is empty
                    schemas_dir.rmdir()
                    console.print(
                        f"📁 Removed empty directory: {schemas_dir}", style="dim"
                    )
            except Exception:
                pass  # Directory not empty or other issue

        console.print(f"\n✅ [green]Cleaned {deleted_count} schema files[/green]")


@clean_app.command("configs")
def clean_configs(
    configs_dir: Path | None = typer.Option(
        None, help="Directory containing configs to clean (defaults to './configs')"
    ),
    dry_run: bool = typer.Option(
        False, help="Preview what would be deleted without actually deleting"
    ),
) -> None:
    """🧹 Clean generated configuration files."""

    if configs_dir is None:
        configs_dir = Path("./configs")

    if not configs_dir.exists():
        console.print(f"ℹ️ [blue]Config directory doesn't exist: {configs_dir}[/blue]")
        return

    # Find generated config files based on our known patterns
    from ..core.config_generation import detect_generated_config_files

    files_to_delete = detect_generated_config_files(configs_dir)

    if not files_to_delete:
        console.print(
            f"ℹ️ [blue]No generated config files found in {configs_dir}[/blue]"
        )
        return

    console.print(
        f"🧹 Found {len(files_to_delete)} generated config files to clean...",
        style="bold blue",
    )

    # Display files to be deleted
    delete_table = Table(title="Generated Config Files to Delete")
    delete_table.add_column("File", style="cyan")
    delete_table.add_column("Type", style="magenta")

    for file in files_to_delete:
        relative_path = file.relative_to(configs_dir)
        config_type = (
            "Task Config"
            if file.name in ["train.yaml", "pre_encode.yaml"]
            else "Config Group"
        )
        delete_table.add_row(str(relative_path), config_type)

    console.print(delete_table)

    if dry_run:
        console.print("\n🔍 [yellow]Dry run mode - no files would be deleted[/yellow]")
    else:
        # Confirm deletion
        if not typer.confirm(
            f"Are you sure you want to delete {len(files_to_delete)} "
            "generated config files?"
        ):
            console.print("❌ [yellow]Operation cancelled[/yellow]")
            return

        # Delete files
        deleted_count = 0
        for file in files_to_delete:
            try:
                file.unlink()
                deleted_count += 1
                console.print(
                    f"🗑️ Deleted: {file.relative_to(configs_dir)}", style="dim"
                )
            except Exception as e:
                console.print(f"❌ [red]Failed to delete {file.name}: {e}[/red]")

        # Clean up empty directories
        for subdir in ["data", "model", "training", "pre_encode", "hydra"]:
            subdir_path = configs_dir / subdir
            if subdir_path.exists() and not any(subdir_path.iterdir()):
                try:
                    subdir_path.rmdir()
                    console.print(f"📁 Removed empty directory: {subdir}", style="dim")
                except Exception:
                    pass

        # Remove main config directory if empty
        if not any(configs_dir.iterdir()):
            try:
                configs_dir.rmdir()
                console.print(f"📁 Removed empty directory: {configs_dir}", style="dim")
            except Exception:
                pass

        console.print(
            f"\n✅ [green]Cleaned {deleted_count} generated config files[/green]"
        )


@clean_app.command("all")
def clean_all(
    base_dir: Path | None = typer.Option(
        None, help="Base directory to clean (defaults to current directory)"
    ),
    dry_run: bool = typer.Option(
        False, help="Preview what would be deleted without actually deleting"
    ),
) -> None:
    """🧹 Clean all generated files (configs and schemas)."""

    if base_dir is None:
        base_dir = Path(".")

    configs_dir = base_dir / "configs"
    schemas_dir = base_dir / "schemas"

    console.print("🧹 Cleaning all generated files...", style="bold blue")

    # Clean schemas first
    if schemas_dir.exists():
        console.print("\n📋 Cleaning schemas...", style="bold cyan")
        # Call the clean schemas command
        clean_schemas(schemas_dir=schemas_dir, dry_run=dry_run)

    # Clean configs second
    if configs_dir.exists():
        console.print("\n📄 Cleaning configs...", style="bold cyan")
        clean_configs(configs_dir=configs_dir, dry_run=dry_run)

    if not dry_run:
        console.print(
            Panel.fit(
                "[green]🎉 All generated files cleaned![/green]\n"
                "[cyan]🧹 Removed all auto-generated configs and schemas[/cyan]\n"
                "[yellow]💡 Run 'uv run hyperencoder-utils generate all' to "
                "regenerate[/yellow]",
                title="🧹 Cleanup Complete",
            )
        )
