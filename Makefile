# Audio Hyperencoder Development Makefile
# 
# This Makefile provides convenient developer commands for common development tasks.

.PHONY: help install clean dev lint format test train pre-encode

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

# Cleaning commands
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

clean: clean-cache clean-outputs ## Clean cache and outputs
	@echo "✅ Cleaning complete!"

# Training commands
train: ## Start training with default configuration
	@echo "🚀 Starting training..."
	uv run hyperencoder

train-debug: ## Start training in debug mode
	@echo "🐛 Starting training in debug mode..."
	uv run hyperencoder training=debug

# Pre-encoding commands
pre-encode: ## Run pre-encoding with default configuration
	@echo "🔄 Starting pre-encoding..."
	uv run hyperencoder --config-name pre_encode

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

# Development workflow
dev: install ## Setup development environment (alias for install)
	@echo "🎉 Development environment ready!"

reset: clean install ## Reset development environment
	@echo "🔄 Development environment reset complete!" 