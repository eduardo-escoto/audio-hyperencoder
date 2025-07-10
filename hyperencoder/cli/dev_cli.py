"""Development utilities CLI for Audio Hyperencoder.

This module provides the main development utilities CLI app using typer
for a beautiful and organized command-line interface.
"""

import typer
from rich.console import Console

# Create the main app
app = typer.Typer(
    name="hyperencoder-dev",
    help="🔧 Development utilities for Audio Hyperencoder",
    rich_markup_mode="rich",
)

# Create subcommands for better organization
generate_app = typer.Typer(help="🏗️ Generate configuration files and schemas")
setup_app = typer.Typer(help="🔧 Setup IDE integrations and development tools")
clean_app = typer.Typer(help="🧹 Clean generated files and artifacts")

# Add subcommands to main app
app.add_typer(generate_app, name="generate")
app.add_typer(setup_app, name="setup")
app.add_typer(clean_app, name="clean")

# Shared console instance for consistent output
console = Console()


def main() -> None:
    """Main entry point for the development utilities CLI."""
    app()


# Import all command modules to register them with the app
# This ensures all @app.command() and @subapp.command() decorators are executed
from . import dev_commands  # noqa: F401

if __name__ == "__main__":
    main()
