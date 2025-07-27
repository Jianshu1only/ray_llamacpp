# Ray Llama.cpp Deployment - Usage Guide

## Quick Start

### 1. Test with Mock Servers (No Model Required)

```bash
# Test with 4 mock instances
python3 ray_llama_flexible.py --mock --num-instances 4 --test-only

# Run 16 mock instances and keep them running
python3 ray_llama_flexible.py --mock --num-instances 16 --base-port 8000

# Test the running deployment
python3 test_client.py --base-port 8000 --num-instances 16 --parallel-requests 32
```

### 2. Deploy with Real Models

```bash
# Deploy 64 instances with your Qwen2.5 model
python3 ray_llama_flexible.py \
  --num-instances 64 \
  --model-path /path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
  --base-port 8000 \
  --context-size 2048 \
  --gpu-layers 32 \
  --threads 1

# Test with fewer instances first
python3 ray_llama_flexible.py \
  --num-instances 8 \
  --model-path /path/to/qwen2.5-0.5b-instruct-q4_0.gguf \
  --test-only
```

## Command Line Options

### Core Configuration
- `--num-instances N`: Number of model instances (default: 4)
- `--base-port PORT`: Starting port number (default: 8000)
- `--mock`: Use mock servers instead of real models
- `--test-only`: Run tests and exit (don't keep running)

### Model Configuration (Real Mode Only)
- `--model-path PATH`: Path to GGUF model file (required for real mode)
- `--llama-path PATH`: Path to llama-cli executable
- `--context-size SIZE`: Context size (default: 2048)
- `--threads N`: Threads per instance (default: 1)
- `--gpu-layers N`: GPU layers (default: 32)

### Runtime Options
- `--log-level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Example Commands

### Testing Scenarios

```bash
# Quick test with 4 mock instances
python3 ray_llama_flexible.py --mock --test-only

# Scale test with 32 mock instances
python3 ray_llama_flexible.py --mock --num-instances 32 --test-only

# Performance test with custom port range
python3 ray_llama_flexible.py --mock --num-instances 16 --base-port 9000 --test-only
```

### Production Deployment

```bash
# Small deployment (8 instances)
python3 ray_llama_flexible.py \
  --num-instances 8 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf \
  --base-port 8000

# Medium deployment (32 instances)
python3 ray_llama_flexible.py \
  --num-instances 32 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf \
  --base-port 8000 \
  --gpu-layers 20

# Full deployment (64 instances)
python3 ray_llama_flexible.py \
  --num-instances 64 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf \
  --base-port 8000 \
  --context-size 1024 \
  --gpu-layers 16 \
  --threads 1
```

### Client Testing

```bash
# Test all instances individually
python3 test_client.py --base-port 8000 --num-instances 64

# Stress test with parallel requests
python3 test_client.py \
  --base-port 8000 \
  --num-instances 64 \
  --parallel-requests 100

# Custom prompt testing
python3 test_client.py \
  --base-port 8000 \
  --num-instances 8 \
  --prompt "Write a short poem about AI"
```

## API Usage

Each instance exposes an OpenAI-compatible API:

### Chat Completions
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Python Client Example
```python
import requests

# Send request to any instance
response = requests.post('http://localhost:8000/v1/chat/completions', json={
    "messages": [{"role": "user", "content": "What is AI?"}],
    "max_tokens": 100
})

print(response.json())
```

## Performance Tuning

### GPU Memory Optimization
```bash
# For limited GPU memory, reduce layers
python3 ray_llama_flexible.py \
  --num-instances 64 \
  --gpu-layers 8 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf

# For more GPU memory, increase layers for speed
python3 ray_llama_flexible.py \
  --num-instances 32 \
  --gpu-layers 64 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf
```

### CPU Optimization
```bash
# More threads per instance (if you have many CPU cores)
python3 ray_llama_flexible.py \
  --num-instances 32 \
  --threads 2 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf
```

### Context Size Tuning
```bash
# Smaller context for more instances
python3 ray_llama_flexible.py \
  --num-instances 64 \
  --context-size 1024 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf

# Larger context for longer conversations
python3 ray_llama_flexible.py \
  --num-instances 32 \
  --context-size 4096 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf
```

## Monitoring

### Real-time Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor CPU and memory
htop

# Monitor network ports
netstat -tlnp | grep :80

# Monitor Ray dashboard
# Open http://127.0.0.1:8265 in browser
```

### Log Monitoring
```bash
# Enable debug logging
python3 ray_llama_flexible.py \
  --mock \
  --num-instances 8 \
  --log-level DEBUG

# Monitor specific instances
curl http://localhost:8000/health
curl http://localhost:8001/health
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Use different base port
   python3 ray_llama_flexible.py --mock --base-port 9000
   ```

2. **GPU out of memory**
   ```bash
   # Reduce GPU layers or instances
   python3 ray_llama_flexible.py \
     --num-instances 32 \
     --gpu-layers 16 \
     --model-path ./model.gguf
   ```

3. **Model file not found**
   ```bash
   # Test with mock mode first
   python3 ray_llama_flexible.py --mock --test-only
   
   # Check model path
   ls -la /path/to/your/model.gguf
   ```

4. **Ray initialization fails**
   ```bash
   # Kill existing Ray processes
   ray stop
   
   # Start fresh
   python3 ray_llama_flexible.py --mock --test-only
   ```

### Performance Issues

1. **Slow responses**
   - Increase `--gpu-layers`
   - Reduce `--context-size`
   - Use smaller model (q4_0 instead of q8_0)

2. **High memory usage**
   - Reduce `--num-instances`
   - Reduce `--context-size`
   - Increase `--gpu-layers` to use GPU memory instead of RAM

3. **Failed instances**
   - Check logs with `--log-level DEBUG`
   - Verify model file accessibility
   - Ensure sufficient system resources

## Production Deployment

### Recommended Settings for Different Scales

#### Small Scale (8 instances)
```bash
python3 ray_llama_flexible.py \
  --num-instances 8 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf \
  --base-port 8000 \
  --context-size 2048 \
  --gpu-layers 32 \
  --threads 1
```

#### Medium Scale (32 instances)
```bash
python3 ray_llama_flexible.py \
  --num-instances 32 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf \
  --base-port 8000 \
  --context-size 1536 \
  --gpu-layers 24 \
  --threads 1
```

#### Large Scale (64 instances)
```bash
python3 ray_llama_flexible.py \
  --num-instances 64 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf \
  --base-port 8000 \
  --context-size 1024 \
  --gpu-layers 16 \
  --threads 1
```

### Load Balancing
Use a reverse proxy like nginx to distribute requests:

```nginx
upstream llama_backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
    # ... add all your instances
}

server {
    listen 80;
    location / {
        proxy_pass http://llama_backend;
    }
}
```

## Advanced Usage

### Custom Model Configurations
```bash
# Different models for different instance groups
# This would require script modification, but the framework supports it

# High-performance instances (fewer, more resources)
python3 ray_llama_flexible.py \
  --num-instances 16 \
  --model-path ./qwen2.5-0.5b-instruct-q8_0.gguf \
  --base-port 8000 \
  --context-size 4096 \
  --gpu-layers 64 \
  --threads 2

# Efficiency instances (more, fewer resources)
python3 ray_llama_flexible.py \
  --num-instances 64 \
  --model-path ./qwen2.5-0.5b-instruct-q4_0.gguf \
  --base-port 9000 \
  --context-size 1024 \
  --gpu-layers 8 \
  --threads 1
```

### Integration with Other Services
The deployment provides standard HTTP endpoints that can be integrated with:
- Load balancers (nginx, HAProxy)
- API gateways
- Monitoring systems (Prometheus, Grafana)
- Container orchestration (Docker, Kubernetes)

## Support and Debugging

### Enable Debug Mode
```bash
python3 ray_llama_flexible.py \
  --mock \
  --num-instances 4 \
  --log-level DEBUG \
  --test-only
```

### Ray Dashboard
Access the Ray dashboard at http://127.0.0.1:8265 to monitor:
- Resource usage
- Task execution
- Actor status
- System metrics

### Manual Testing
```bash
# Test individual instance
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "test"}]}'

# Test health
curl http://localhost:8000/health
```