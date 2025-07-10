"""Development command modules for the hyperencoder CLI.

This package contains all the command modules that implement the various
development utilities like config generation, schema creation, IDE setup,
and file cleaning.
"""

# Import all command modules to register them with the typer app
# The @app.command() and @subapp.command() decorators are executed during import
from . import (
    info,  # noqa: F401
    clean,  # noqa: F401
    setup,  # noqa: F401
    convert,  # noqa: F401
    generate,  # noqa: F401
)

__all__ = ["convert", "generate", "setup", "clean", "info"]
