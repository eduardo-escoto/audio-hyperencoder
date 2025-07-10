"""Development utilities CLI for Audio Hyperencoder.

This module provides development utilities including config conversion,
config generation, and schema generation using typer and rich for a
beautiful CLI experience.
"""

import typer
import shutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
import json
import yaml
from typing import Optional

app = typer.Typer(
    name="hyperencoder-utils",
    help="🔧 Development utilities for Audio Hyperencoder",
    rich_markup_mode="rich"
)
console = Console()

def get_bundled_configs_path() -> Path:
    """Get the path to bundled configs within the package."""
    return Path(__file__).parent.parent / "configs"

@app.command()
def convert(
    config_type: str = typer.Argument(..., help="Config type to convert (json|ini)"),
    input_path: Path = typer.Argument(..., help="Input config file path"),
    output_path: Optional[Path] = typer.Option(None, help="Output YAML file path")
):
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
                with open(input_path, 'r') as f:
                    config_data = json.load(f)
            elif config_type == "ini":
                # Convert INI to YAML
                import configparser
                config = configparser.ConfigParser()
                config.read(input_path)
                config_data = {section: dict(config.items(section)) for section in config.sections()}
            else:
                console.print(f"❌ [red]Unsupported config type: {config_type}[/red]")
                raise typer.Exit(1)
            
            progress.update(task, description="Writing YAML file...")
            
            if output_path is None:
                output_path = input_path.with_suffix('.yaml')
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            progress.update(task, description="✅ Complete!")
            
        except Exception as e:
            console.print(f"❌ [red]Conversion failed: {str(e)}[/red]")
            raise typer.Exit(1)
    
    # Beautiful success message
    console.print(Panel.fit(
        f"[green]✅ Successfully converted {input_path} to {output_path}[/green]",
        title="🎉 Conversion Complete"
    ))

@app.command()
def generate(
    target: str = typer.Argument(..., help="What to generate (configs|schemas)"),
    output_dir: Path = typer.Option("./hyperencoder-configs", help="Output directory")
):
    """🏗️ Generate default configs or JSON schemas."""
    
    console.print(f"🏗️ Generating {target} in {output_dir}...")
    
    if target == "configs":
        # Copy bundled configs to user directory for customization
        bundled_configs = get_bundled_configs_path()
        
        if not bundled_configs.exists():
            console.print(f"❌ [red]Bundled configs not found at {bundled_configs}[/red]")
            raise typer.Exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Copying config files...", total=None)
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all config files
            shutil.copytree(bundled_configs, output_dir, dirs_exist_ok=True)
            
            progress.update(task, description="✅ Complete!")
        
        # Show what was generated
        configs_table = Table(title="📄 Generated Config Files")
        configs_table.add_column("File", style="cyan")
        configs_table.add_column("Description", style="magenta")
        
        configs_table.add_row("config.yaml", "Main configuration with task selection")
        configs_table.add_row("train.yaml", "Training configuration")
        configs_table.add_row("pre_encode.yaml", "Pre-encoding configuration")
        configs_table.add_row("model/", "Model configurations directory")
        configs_table.add_row("data/", "Data configurations directory")
        configs_table.add_row("experiment/", "Experiment configurations directory")
        
        console.print(configs_table)
        
        console.print(Panel.fit(
            f"[green]✅ Default configs copied to {output_dir}[/green]\n"
            f"[yellow]💡 Edit these files to customize your experiments[/yellow]\n"
            f"[blue]📖 Use with: uv run hyperencoder --config-path {output_dir}[/blue]",
            title="🎉 Config Generation Complete"
        ))
        
    elif target == "schemas":
        # Generate JSON schemas from Pydantic models
        schemas_table = Table(title="📋 Generated JSON Schemas")
        schemas_table.add_column("Schema", style="cyan")
        schemas_table.add_column("Description", style="magenta")
        
        try:
            from hyperencoder.config import ModelConfig, DataConfig, TrainingConfig
            
            # Create schemas directory
            schemas_dir = output_dir / "schemas"
            schemas_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate schemas
            schemas = [
                (ModelConfig, "model_config.schema.json", "Model configuration schema"),
                (DataConfig, "data_config.schema.json", "Data configuration schema"), 
                (TrainingConfig, "training_config.schema.json", "Training configuration schema"),
            ]
            
            for config_class, filename, description in schemas:
                schema_path = schemas_dir / filename
                schema = config_class.model_json_schema()
                
                with open(schema_path, 'w') as f:
                    json.dump(schema, f, indent=2)
                
                schemas_table.add_row(filename, description)
            
            console.print(schemas_table)
            console.print(f"[green]✅ Schemas generated in {schemas_dir}[/green]")
            
        except ImportError as e:
            console.print(f"❌ [red]Failed to import config classes: {e}[/red]")
            raise typer.Exit(1)
    
    else:
        console.print(f"❌ [red]Unknown target: {target}. Use 'configs' or 'schemas'[/red]")
        raise typer.Exit(1)

@app.command()
def info():
    """ℹ️ Show information about bundled configs and package structure."""
    
    bundled_configs = get_bundled_configs_path()
    
    info_table = Table(title="📦 Package Information")
    info_table.add_column("Item", style="cyan")
    info_table.add_column("Path", style="magenta")
    info_table.add_column("Status", style="green")
    
    info_table.add_row("Bundled Configs", str(bundled_configs), "✅ Available" if bundled_configs.exists() else "❌ Missing")
    info_table.add_row("Package Root", str(Path(__file__).parent.parent), "📁 Package")
    
    console.print(info_table)
    
    if bundled_configs.exists():
        console.print(f"\n[blue]💡 Use 'generate configs' to copy these to your working directory[/blue]")

if __name__ == "__main__":
    app() 