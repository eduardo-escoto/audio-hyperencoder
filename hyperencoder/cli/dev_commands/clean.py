"""
Clean commands for development CLI.

This module provides utilities for cleaning up generated files and build artifacts.
"""

import logging
import shutil
from pathlib import Path
from typing import List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)


def clean_pycache(root_dir: Path = Path(".")) -> int:
    """Clean up __pycache__ directories.
    
    Args:
        root_dir: Root directory to search for __pycache__ directories
        
    Returns:
        Number of directories cleaned
    """
    count = 0
    for pycache_dir in root_dir.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)
            count += 1
            logger.debug(f"Removed {pycache_dir}")
    
    return count


def clean_build_artifacts(root_dir: Path = Path(".")) -> int:
    """Clean up build artifacts.
    
    Args:
        root_dir: Root directory to search for build artifacts
        
    Returns:
        Number of artifacts cleaned
    """
    count = 0
    patterns = [
        "build",
        "dist",
        "*.egg-info",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
    ]
    
    for pattern in patterns:
        for artifact in root_dir.glob(pattern):
            if artifact.is_dir():
                shutil.rmtree(artifact)
                count += 1
                logger.debug(f"Removed directory {artifact}")
            elif artifact.is_file():
                artifact.unlink()
                count += 1
                logger.debug(f"Removed file {artifact}")
    
    return count


def clean_logs(root_dir: Path = Path(".")) -> int:
    """Clean up log files.
    
    Args:
        root_dir: Root directory to search for log files
        
    Returns:
        Number of log files cleaned
    """
    count = 0
    log_patterns = ["*.log", "logs/**/*.log", "*.log.*"]
    
    for pattern in log_patterns:
        for log_file in root_dir.glob(pattern):
            if log_file.is_file():
                log_file.unlink()
                count += 1
                logger.debug(f"Removed log file {log_file}")
    
    return count


def clean_temp_files(root_dir: Path = Path(".")) -> int:
    """Clean up temporary files.
    
    Args:
        root_dir: Root directory to search for temporary files
        
    Returns:
        Number of temporary files cleaned
    """
    count = 0
    temp_patterns = [
        "*.tmp",
        "*.temp",
        "*.swp",
        "*.bak",
        "*~",
        ".DS_Store",
        "Thumbs.db",
    ]
    
    for pattern in temp_patterns:
        for temp_file in root_dir.rglob(pattern):
            if temp_file.is_file():
                temp_file.unlink()
                count += 1
                logger.debug(f"Removed temp file {temp_file}")
    
    return count


def show_cleanup_options() -> None:
    """Display available cleanup options."""
    table = Table(title="Available Cleanup Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    
    table.add_row("pycache", "Remove __pycache__ directories")
    table.add_row("build", "Remove build artifacts")
    table.add_row("logs", "Remove log files")
    table.add_row("temp", "Remove temporary files")
    table.add_row("all", "Remove all of the above")
    
    console.print(table)


def clean_command(target: str, root_dir: str = ".") -> None:
    """Main clean command dispatcher.
    
    Args:
        target: What to clean ('pycache', 'build', 'logs', 'temp', 'all')
        root_dir: Root directory to search for files to clean
    """
    root_path = Path(root_dir)
    
    if not root_path.exists():
        console.print(f"❌ Directory does not exist: {root_path}")
        return
    
    total_cleaned = 0
    
    if target == "pycache":
        count = clean_pycache(root_path)
        total_cleaned += count
        console.print(f"✅ Cleaned {count} __pycache__ directories")
    
    elif target == "build":
        count = clean_build_artifacts(root_path)
        total_cleaned += count
        console.print(f"✅ Cleaned {count} build artifacts")
    
    elif target == "logs":
        count = clean_logs(root_path)
        total_cleaned += count
        console.print(f"✅ Cleaned {count} log files")
    
    elif target == "temp":
        count = clean_temp_files(root_path)
        total_cleaned += count
        console.print(f"✅ Cleaned {count} temporary files")
    
    elif target == "all":
        pycache_count = clean_pycache(root_path)
        build_count = clean_build_artifacts(root_path)
        logs_count = clean_logs(root_path)
        temp_count = clean_temp_files(root_path)
        
        total_cleaned = pycache_count + build_count + logs_count + temp_count
        
        console.print(f"✅ Cleaned {pycache_count} __pycache__ directories")
        console.print(f"✅ Cleaned {build_count} build artifacts")
        console.print(f"✅ Cleaned {logs_count} log files")
        console.print(f"✅ Cleaned {temp_count} temporary files")
    
    elif target == "help":
        show_cleanup_options()
        return
    
    else:
        console.print(f"❌ Unknown cleanup target: {target}")
        console.print("Available targets: pycache, build, logs, temp, all, help")
        return
    
    console.print(Panel(
        f"Cleanup completed successfully!\nTotal items cleaned: {total_cleaned}",
        title="✅ Success",
        style="green"
    ))
