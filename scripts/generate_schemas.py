#!/usr/bin/env python3
"""
Generate JSON schemas from Pydantic configuration models.

This script creates JSON schemas for all configuration models in the hyperencoder
project, enabling IDE validation and autocomplete for YAML configuration files.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Type, List, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hyperencoder.config import (
    BaseConfig,
    TrainingConfig,
    PreEncodeConfig,
    DataConfig,
    ModelConfig,
)


def generate_schema(config_class: Type[BaseConfig], output_path: Path) -> None:
    """
    Generate JSON schema for a Pydantic model.
    
    Args:
        config_class: The Pydantic model class to generate schema for
        output_path: Path where the schema will be saved
    """
    try:
        # Generate the schema
        schema = config_class.model_json_schema()
        
        # Add custom properties for better IDE support
        schema["$id"] = f"https://hyperencoder.ai/schemas/{output_path.name}"
        schema["$comment"] = f"Generated from {config_class.__module__}.{config_class.__name__}"
        
        # Add title if not present
        if "title" not in schema:
            schema["title"] = config_class.__name__
            
        # Add description if not present
        if "description" not in schema and hasattr(config_class, "__doc__") and config_class.__doc__:
            schema["description"] = config_class.__doc__.strip()
        
        # Write schema with pretty formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, sort_keys=True, ensure_ascii=False)
        
        print(f"✅ Generated schema: {output_path}")
        
    except Exception as e:
        print(f"❌ Error generating schema for {config_class.__name__}: {e}")
        raise


def create_schema_summary(schemas_dir: Path, configs: List[Tuple[Type[BaseConfig], str]]) -> None:
    """Create a summary file listing all generated schemas."""
    summary_path = schemas_dir / "README.md"
    
    content = """# JSON Schemas for Hyperencoder Configuration

This directory contains JSON schemas generated from Pydantic configuration models.
These schemas enable IDE validation and autocomplete for YAML configuration files.

## Available Schemas

"""
    
    for config_class, filename in configs:
        schema_path = schemas_dir / filename
        relative_path = schema_path.relative_to(project_root)
        content += f"- **{config_class.__name__}**: [`{filename}`](./{filename})\n"
        if hasattr(config_class, "__doc__") and config_class.__doc__:
            doc_lines = config_class.__doc__.strip().split('\n')
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
Run `python scripts/generate_schemas.py` to regenerate them.
"""
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Generated schema summary: {summary_path}")


def main():
    """Generate all JSON schemas."""
    print("🚀 Generating JSON schemas for hyperencoder configuration models...")
    
    # Create output directory
    schemas_dir = project_root / "schemas"
    schemas_dir.mkdir(exist_ok=True)
    
    # Define configurations to generate schemas for
    configs = [
        (BaseConfig, "base_config.schema.json"),
        (TrainingConfig, "training_config.schema.json"),
        (PreEncodeConfig, "pre_encode_config.schema.json"),
        (DataConfig, "data_config.schema.json"),
        (ModelConfig, "model_config.schema.json"),
    ]
    
    # Generate schemas
    success_count = 0
    for config_class, filename in configs:
        try:
            output_path = schemas_dir / filename
            generate_schema(config_class, output_path)
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to generate schema for {config_class.__name__}: {e}")
    
    # Create summary file
    create_schema_summary(schemas_dir, configs)
    
    # Print summary
    print(f"\n🎉 Successfully generated {success_count}/{len(configs)} JSON schemas!")
    print(f"📁 Schemas saved to: {schemas_dir}")
    
    if success_count < len(configs):
        print("⚠️  Some schemas failed to generate. Check the errors above.")
        sys.exit(1)
    else:
        print("✅ All schemas generated successfully!")


if __name__ == "__main__":
    main() 