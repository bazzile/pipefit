.PHONY: install run clean

# Default target
all: install

# Install dependencies using uv sync
install:
	uv sync

# Run jupyter lab using uv
run:
	uv run --with jupyter jupyter lab --no-browser --NotebookApp.token='' 

# Clean Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

help:
	@echo "Available commands:"
	@echo "  make install - Install dependencies using uv sync"
	@echo "  make run     - Run jupyter lab using uv"
	@echo "  make clean   - Clean Python cache files"
	@echo "  make help    - Show this help message"