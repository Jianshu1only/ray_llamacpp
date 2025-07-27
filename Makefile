# Makefile for Ray Llama Deployment

.PHONY: help install test test-mock test-scale demo clean lint format

# Default target
help:
	@echo "Ray Llama Deployment - Makefile Commands"
	@echo "========================================"
	@echo ""
	@echo "Installation:"
	@echo "  install     Install dependencies"
	@echo "  install-dev Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test        Run all tests"
	@echo "  test-mock   Test with mock servers (4 instances)"
	@echo "  test-scale  Test scalability (16 instances)"
	@echo "  test-perf   Performance test (32 instances)"
	@echo "  demo        Run demo scenarios"
	@echo ""
	@echo "Development:"
	@echo "  lint        Run code linting"
	@echo "  format      Format code"
	@echo "  clean       Clean temporary files"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-mock Deploy mock servers for testing"
	@echo "  stop        Stop all Ray processes"

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install black flake8 pytest

# Testing targets
test: test-mock test-scale

test-mock:
	@echo "Running mock server test..."
	python3 ray_llama_flexible.py --mock --num-instances 4 --test-only

test-scale:
	@echo "Running scalability test..."
	python3 ray_llama_flexible.py --mock --num-instances 16 --test-only

test-perf:
	@echo "Running performance test..."
	python3 ray_llama_flexible.py --mock --num-instances 32 --test-only --base-port 9000

demo:
	@echo "Running demo scenarios..."
	python3 examples/demo.py

# Development targets
lint:
	flake8 *.py examples/*.py --max-line-length=120 --ignore=E501,W503

format:
	black *.py examples/*.py --line-length=120

clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f /tmp/mock_llama_server_*.py
	rm -f /tmp/llama_server_*.py
	ray stop 2>/dev/null || true

# Deployment targets
deploy-mock:
	@echo "Deploying mock servers (8 instances)..."
	python3 ray_llama_flexible.py --mock --num-instances 8 --base-port 8000

stop:
	@echo "Stopping Ray processes..."
	ray stop

# Quick test commands
quick-test:
	python3 ray_llama_flexible.py --mock --num-instances 4 --test-only

client-test:
	python3 examples/test_client.py --base-port 8000 --num-instances 4 --parallel-requests 10

# Development workflow
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to verify installation"

# Package building
build:
	python3 setup.py sdist bdist_wheel

# Documentation generation (if needed)
docs:
	@echo "Opening documentation..."
	@echo "Main README: README.md"
	@echo "Usage Guide: docs/USAGE_GUIDE.md"
	@echo "Changelog: CHANGELOG.md"