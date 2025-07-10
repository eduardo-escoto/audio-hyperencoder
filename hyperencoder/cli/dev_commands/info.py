"""Information commands for package and configuration details.

This module provides commands for displaying information about the package
structure and available configurations.
"""

from pathlib import Path

from rich.table import Table

from ..dev_cli import app, console


@app.command()
def info() -> None:
    """ℹ️ Show information about bundled configs and package structure."""

    from ..core.config_generation import get_bundled_configs_path

    bundled_configs = get_bundled_configs_path()

    info_table = Table(title="📦 Package Information")
    info_table.add_column("Item", style="cyan")
    info_table.add_column("Path", style="magenta")
    info_table.add_column("Status", style="green")

    info_table.add_row(
        "Bundled Configs",
        str(bundled_configs),
        "✅ Available" if bundled_configs.exists() else "❌ Missing",
    )
    info_table.add_row(
        "Package Root", str(Path(__file__).parent.parent.parent), "📁 Package"
    )

    console.print(info_table)

    if bundled_configs.exists():
        console.print(
            "\n[blue]💡 Use 'generate configs' to copy these to your working "
            "directory[/blue]"
        )
