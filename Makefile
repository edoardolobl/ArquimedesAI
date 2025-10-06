.PHONY: help install setup clean index discord status test lint format

help:
	@echo "ArquimedesAI - Makefile Commands"
	@echo "================================="
	@echo "setup        - Create .env from template"
	@echo "install      - Install Python dependencies"
	@echo "clean        - Remove cache and generated files"
	@echo "index        - Build vector index from data/"
	@echo "discord      - Start Discord bot"
	@echo "status       - Show system status"
	@echo "test         - Run tests"
	@echo "lint         - Run code linters"
	@echo "format       - Format code with ruff"

setup:
	@echo "Creating .env file..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ .env created from .env.example"; \
		echo "⚠️  Edit .env and add your Discord token"; \
	else \
		echo "⚠️  .env already exists"; \
	fi

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	@echo "✓ Cleaned"

index:
	@echo "Building vector index..."
	python cli.py index

index-rebuild:
	@echo "Rebuilding vector index (deleting existing)..."
	python cli.py index --rebuild

discord:
	@echo "Starting Discord bot..."
	python cli.py discord

status:
	@echo "Checking system status..."
	python cli.py status

test:
	@echo "Running tests..."
	pytest -v

lint:
	@echo "Running linters..."
	ruff check .
	mypy .

format:
	@echo "Formatting code..."
	ruff format .
	@echo "✓ Code formatted"
