#!/usr/bin/env python3
"""
Configuration file for Ray-based llama.cpp deployment
"""

import os

# Model Configuration
MODEL_PATH = ""  # Set this to your Qwen2.5 0.5B GGUF model path
# Example: MODEL_PATH = "/path/to/qwen2.5-0.5b-instruct-q4_0.gguf"

# Deployment Configuration
NUM_INSTANCES = 64
BASE_PORT = 8000
CONTEXT_SIZE = 2048
THREADS_PER_INSTANCE = 1
GPU_LAYERS = 32  # Adjust based on your GPU memory and model size

# Paths
LLAMA_CPP_PATH = "/mnt/weka/home/jianshu.she/jianshu/llama.cpp/build/bin/llama-cli"

# Resource Configuration (per Ray actor)
CPU_PER_INSTANCE = 1
GPU_PER_INSTANCE = 0.015625  # 1/64 of GPU per instance

# Performance Tuning
MAX_TOKENS_DEFAULT = 512
TEMPERATURE_DEFAULT = 0.7
REPEAT_PENALTY_DEFAULT = 1.1
REQUEST_TIMEOUT = 30  # seconds

# Health Check Configuration
HEALTH_CHECK_INTERVAL = 60  # seconds
STARTUP_TIMEOUT = 30  # seconds

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Model Download URLs (optional)
MODEL_URLS = {
    "qwen2.5-0.5b-instruct-q4_0": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf",
    "qwen2.5-0.5b-instruct-q8_0": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
}

def validate_config():
    """Validate configuration settings"""
    if not MODEL_PATH:
        raise ValueError("MODEL_PATH must be set to a valid GGUF model file")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    if not os.path.exists(LLAMA_CPP_PATH):
        raise FileNotFoundError(f"llama-cli executable not found: {LLAMA_CPP_PATH}")
    
    if NUM_INSTANCES <= 0:
        raise ValueError("NUM_INSTANCES must be positive")
    
    if BASE_PORT < 1024 or BASE_PORT > 65535 - NUM_INSTANCES:
        raise ValueError("BASE_PORT is invalid or doesn't leave enough room for all instances")
    
    return True