# Audio Hyperencoder Development Makefile
# 
# This Makefile provides convenient developer commands that leverage the 
# hyperencoder-utils CLI for common development tasks.

.PHONY: help generate setup clean dev install test lint format

# Default target
help: ## Show this help message
	@echo "🚀 Audio Hyperencoder Development Commands"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "📖 For more details: https://github.com/your-repo/audio-hyperencoder"

# Installation and setup
install: ## Install dependencies and setup development environment
	@echo "📦 Installing dependencies..."
	uv sync --all-extras
	@echo "🔧 Installing pre-commit hooks..."
	uv run pre-commit install
	@echo "✅ Installation complete!"

dev-setup: install generate-all setup-vscode ## Complete development environment setup
	@echo "🎉 Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  - Edit configs in ./configs/"
	@echo "  - Run 'make train' to start training"
	@echo "  - Run 'make clean-all' to clean generated files"

# Generation commands
generate-configs: ## Generate default configuration files
	@echo "📄 Generating configuration files..."
	uv run hyperencoder-utils generate configs

generate-package-configs: ## Generate configuration files in package directory
	@echo "📄 Generating configuration files in package directory..."
	uv run hyperencoder-utils generate configs --output-dir ./hyperencoder/cli/configs

generate-schemas: ## Generate JSON schemas for configuration validation
	@echo "📋 Generating JSON schemas..."
	uv run hyperencoder-utils generate schemas

generate-all: ## Generate both configs and schemas
	@echo "🏗️  Generating all configuration files and schemas..."
	uv run hyperencoder-utils generate all

# Setup commands
setup-vscode: ## Setup VS Code with schema validation
	@echo "🔧 Setting up VS Code integration..."
	uv run hyperencoder-utils setup vscode

teardown-vscode: ## Remove VS Code schema mappings for hyperencoder configs  
	@echo "🔧 Removing VS Code integration..."
	uv run hyperencoder-utils setup teardown-vscode

setup: setup-vscode ## Setup all IDE integrations
	@echo "✅ All IDE integrations configured!"

teardown: teardown-vscode ## Remove all IDE integrations
	@echo "✅ All IDE integrations removed!"

# Cleaning commands
clean-configs: ## Clean generated configuration files
	@echo "🧹 Cleaning generated configuration files..."
	uv run hyperencoder-utils clean configs

clean-schemas: ## Clean generated JSON schemas
	@echo "🧹 Cleaning generated JSON schemas..."
	uv run hyperencoder-utils clean schemas

clean-all: ## Clean all generated files (configs and schemas)
	@echo "🧹 Cleaning all generated files..."
	uv run hyperencoder-utils clean all

clean-outputs: ## Clean training outputs and logs
	@echo "🧹 Cleaning training outputs..."
	rm -rf outputs/
	rm -rf logs/
	@echo "✅ Training outputs cleaned!"

clean-cache: ## Clean Python cache and build artifacts
	@echo "🧹 Cleaning Python cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ Python cache cleaned!"

clean: clean-cache clean-outputs ## Clean cache and outputs (keep configs/schemas)
	@echo "✅ Basic cleaning complete!"

clean-deep: clean-all clean-outputs clean-cache ## Deep clean everything
	@echo "🧹 Deep cleaning complete!"

# Development commands
refresh: clean-all generate-all ## Refresh all generated files
	@echo "🔄 Refreshing all generated files..."
	@echo "✅ Refresh complete!"

# Training commands
train: ## Start training with default configuration
	@echo "🚀 Starting training..."
	uv run hyperencoder --config-path ./configs

train-debug: ## Start training in debug mode
	@echo "🐛 Starting training in debug mode..."
	uv run hyperencoder --config-path ./configs training=debug

# Pre-encoding commands
pre-encode: ## Run pre-encoding with default configuration
	@echo "🔄 Starting pre-encoding..."
	uv run hyperencoder --config-path ./configs --config-name pre_encode

# Code quality commands
lint: ## Run linting with ruff
	@echo "🔍 Running linter..."
	uv run ruff check hyperencoder/

format: ## Format code with ruff
	@echo "✨ Formatting code..."
	uv run ruff format hyperencoder/

fix: ## Fix linting issues automatically
	@echo "🔧 Fixing linting issues..."
	uv run ruff check hyperencoder/ --fix

# Testing commands
test: ## Run tests
	@echo "🧪 Running tests..."
	uv run pytest

test-verbose: ## Run tests with verbose output
	@echo "🧪 Running tests (verbose)..."
	uv run pytest -v

# Utility commands
info: ## Show project information
	@echo "ℹ️  Project information:"
	uv run hyperencoder-utils info

convert-json: ## Convert JSON config to YAML (usage: make convert-json CONFIG=path/to/config.json)
	@if [ -z "$(CONFIG)" ]; then echo "❌ Usage: make convert-json CONFIG=path/to/config.json"; exit 1; fi
	@echo "🔄 Converting $(CONFIG) to YAML..."
	uv run hyperencoder-utils convert json $(CONFIG)

convert-ini: ## Convert INI config to YAML (usage: make convert-ini CONFIG=path/to/config.ini)
	@if [ -z "$(CONFIG)" ]; then echo "❌ Usage: make convert-ini CONFIG=path/to/config.ini"; exit 1; fi
	@echo "🔄 Converting $(CONFIG) to YAML..."
	uv run hyperencoder-utils convert ini $(CONFIG)

# Documentation commands
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	@echo "ℹ️  Documentation generation not yet implemented"

# Development workflow combinations
quick-start: dev-setup ## Quick start for new developers
	@echo "🎯 Quick start complete!"
	@echo ""
	@echo "You can now:"
	@echo "  - Run 'make train' to start training"
	@echo "  - Run 'make pre-encode' to pre-encode audio"
	@echo "  - Edit configs in ./configs/"

reset: clean-deep install generate-all setup ## Reset development environment
	@echo "🔄 Development environment reset complete!" 