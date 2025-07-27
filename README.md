# Ray-based Llama.cpp Deployment for 64 Qwen2.5 0.5B Models

This project provides a Ray-based solution to deploy 64 instances of Qwen2.5 0.5B models using llama.cpp on a single node with efficient resource management.

## Features

- **Scalable Deployment**: Deploy 64 model instances in parallel using Ray
- **HTTP API**: Each instance exposes an OpenAI-compatible HTTP API
- **Resource Management**: Efficient GPU and CPU resource allocation
- **Health Monitoring**: Built-in health checks and monitoring
- **Load Balancing**: Round-robin request distribution
- **Fault Tolerance**: Automatic instance recovery and error handling

## Prerequisites

1. **CUDA-enabled GPU**: Required for running 64 model instances efficiently
2. **llama.cpp**: Built with CUDA support
3. **Python 3.8+**
4. **Qwen2.5 0.5B GGUF model file**

## Installation

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify llama.cpp build**:
   ```bash
   ls -la /mnt/weka/home/jianshu.she/jianshu/llama.cpp/build/bin/llama-cli
   ```

3. **Download Qwen2.5 0.5B model** (if not already available):
   ```bash
   # Example download command
   wget https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf
   ```

## Configuration

Edit `config.py` to set your model path:

```python
# Set this to your Qwen2.5 0.5B GGUF model path
MODEL_PATH = "/path/to/your/qwen2.5-0.5b-instruct-q4_0.gguf"

# Optionally adjust other parameters
NUM_INSTANCES = 64
BASE_PORT = 8000
CONTEXT_SIZE = 2048
GPU_LAYERS = 32
```

## Usage

### 1. Deploy All Instances

```bash
# Set your model path in config.py first
python ray_llama_server_deployment.py
```

This will:
- Initialize Ray cluster
- Deploy 64 server instances on ports 8000-8063
- Perform health checks
- Run example inference requests
- Keep running until interrupted

### 2. Test the Deployment

In another terminal:

```bash
python client_example.py
```

This will:
- Check health of all instances
- Send test requests
- Run parallel benchmarks
- Show performance metrics

### 3. Manual API Testing

Send requests to any instance:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "What is AI?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

Health check:
```bash
curl http://localhost:8000/health
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Ray Head      │    │  Instance 0     │    │  Instance 63    │
│                 │    │  Port 8000      │    │  Port 8063      │
│  - Coordination │    │  - Flask Server │    │  - Flask Server │
│  - Health Checks│    │  - llama-cli    │    │  - llama-cli    │
│  - Load Balance │    │  - 1/64 GPU     │    │  - 1/64 GPU     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Client        │
                    │  - Round Robin  │
                    │  - Health Check │
                    │  - Benchmarking │
                    └─────────────────┘
```

## Performance Considerations

### GPU Memory Management
- Each instance uses ~1/64 of GPU memory
- Adjust `GPU_LAYERS` in config based on your GPU memory
- Monitor GPU utilization: `nvidia-smi`

### CPU Allocation
- Each instance uses 1 CPU core
- Total: 64 CPU cores recommended
- Adjust `THREADS_PER_INSTANCE` if needed

### Network Resources
- Uses ports 8000-8063 by default
- Ensure firewall allows these ports
- Monitor network bandwidth for high throughput

## Monitoring

### Health Checks
The deployment automatically monitors:
- Process health
- HTTP endpoint availability
- Response time
- Memory usage

### Logs
Monitor deployment logs:
```bash
tail -f ray_deployment.log
```

### Resource Monitoring
```bash
# GPU usage
nvidia-smi -l 1

# CPU and memory
htop

# Network
iftop
```

## Troubleshooting

### Common Issues

1. **Model file not found**:
   - Verify `MODEL_PATH` in `config.py`
   - Check file permissions

2. **Port conflicts**:
   - Adjust `BASE_PORT` in config
   - Check for other services using ports

3. **GPU memory errors**:
   - Reduce `GPU_LAYERS`
   - Reduce `NUM_INSTANCES`
   - Check available GPU memory

4. **Instance startup failures**:
   - Check llama.cpp build
   - Verify CUDA installation
   - Review error logs

### Debug Mode

Enable debug logging:
```python
# In config.py
LOG_LEVEL = "DEBUG"
```

## API Compatibility

The deployed instances are compatible with OpenAI API format:

### Chat Completions
```
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 100,
  "temperature": 0.7
}
```

### Health Check
```
GET /health
```

## Scaling

### Horizontal Scaling
- Increase `NUM_INSTANCES` in config
- Ensure sufficient GPU memory
- Monitor performance vs. resource usage

### Vertical Scaling
- Increase `CONTEXT_SIZE` for longer conversations
- Adjust `GPU_LAYERS` for speed vs. memory trade-off
- Tune `THREADS_PER_INSTANCE` based on CPU cores

## License

This project follows the same license as llama.cpp.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request