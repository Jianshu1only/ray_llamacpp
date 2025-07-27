#!/usr/bin/env python3
"""
Flexible Ray-based deployment script for llama.cpp models
Supports command-line configuration and testing modes
"""

import ray
import subprocess
import time
import os
import logging
import requests
import json
import argparse
import sys
from typing import List, Dict, Optional
import psutil
import signal
import socket
import threading
import queue
from pathlib import Path
from flask import Flask, request, jsonify

# Configure logging
def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

def find_free_port(start_port: int) -> int:
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + 1000):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError("No free ports available")

def create_mock_server(port: int, instance_id: int) -> str:
    """Create a mock server script for testing without actual models"""
    script_path = f"/tmp/mock_llama_server_{instance_id}.py"
    
    script_content = f'''#!/usr/bin/env python3
import time
import argparse
from flask import Flask, request, jsonify
import threading
import random

app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({{"error": "No messages provided"}}), 400
        
        # Get the last message content
        prompt = messages[-1].get('content', '')
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.5))
        
        # Generate mock response
        mock_responses = [
            f"Mock response from instance {instance_id}: This is a simulated answer to your question about '{{prompt[:50]}}...'",
            f"Instance {instance_id} says: I'm a test model responding to your query.",
            f"Test response {instance_id}: Here's what I think about your prompt.",
            f"Simulated answer from server {instance_id}: This is for testing purposes."
        ]
        
        response_text = random.choice(mock_responses)
        
        return jsonify({{
            "id": f"chatcmpl-{{int(time.time())}}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "mock-qwen2.5-0.5b",
            "choices": [{{
                "index": 0,
                "message": {{
                    "role": "assistant",
                    "content": response_text
                }},
                "finish_reason": "stop"
            }}],
            "usage": {{
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }}
        }})
    
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({{"status": "healthy", "instance_id": {instance_id}, "mode": "mock"}})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    
    print(f"Starting mock server on port {{args.port}}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path

def create_llama_server_script(instance_id: int) -> str:
    """Create a real llama.cpp server wrapper script"""
    script_path = f"/tmp/llama_server_{instance_id}.py"
    
    script_content = f'''#!/usr/bin/env python3
import subprocess
import threading
import time
import argparse
from flask import Flask, request, jsonify
import queue
import os

app = Flask(__name__)
llama_process = None
request_queue = queue.Queue()
response_queue = queue.Queue()

def llama_worker(model_path, llama_path, context_size, threads, gpu_layers):
    global llama_process
    cmd = [
        llama_path,
        "-m", model_path,
        "-c", str(context_size),
        "-t", str(threads),
        "-ngl", str(gpu_layers),
        "--interactive",
        "-n", "512"
    ]
    
    try:
        llama_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        while True:
            try:
                prompt = request_queue.get(timeout=1)
                if prompt is None:
                    break
                    
                # Send prompt
                llama_process.stdin.write(f"{{prompt}}\\n")
                llama_process.stdin.flush()
                
                # Read response
                response_lines = []
                start_time = time.time()
                while time.time() - start_time < 30:  # 30 second timeout
                    if llama_process.stdout.readable():
                        line = llama_process.stdout.readline()
                        if line:
                            response_lines.append(line.strip())
                            if len(response_lines) >= 20 or ">" in line:
                                break
                    time.sleep(0.01)
                
                response_queue.put("\\n".join(response_lines))
                
            except queue.Empty:
                continue
            except Exception as e:
                response_queue.put(f"Error: {{e}}")
                
    except Exception as e:
        print(f"Failed to start llama process: {{e}}")

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.json
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({{"error": "No messages provided"}}), 400
        
        # Get the last message content
        prompt = messages[-1].get('content', '')
        
        # Send prompt to llama worker
        request_queue.put(prompt)
        
        # Wait for response
        try:
            response = response_queue.get(timeout=30)
        except queue.Empty:
            return jsonify({{"error": "Timeout waiting for response"}}), 500
        
        return jsonify({{
            "id": f"chatcmpl-{{int(time.time())}}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "qwen2.5-0.5b",
            "choices": [{{
                "index": 0,
                "message": {{
                    "role": "assistant",
                    "content": response
                }},
                "finish_reason": "stop"
            }}],
            "usage": {{
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split())
            }}
        }})
    
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

@app.route('/health', methods=['GET'])
def health():
    global llama_process
    is_alive = llama_process and llama_process.poll() is None
    return jsonify({{"status": "healthy" if is_alive else "unhealthy", "instance_id": {instance_id}}})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--gpu-layers", type=int, default=32)
    parser.add_argument("--llama-path", required=True)
    args = parser.parse_args()
    
    # Start llama worker thread
    worker_thread = threading.Thread(
        target=llama_worker,
        args=(args.model, args.llama_path, args.context_size, args.threads, args.gpu_layers)
    )
    worker_thread.daemon = True
    worker_thread.start()
    
    # Give it time to start
    time.sleep(2)
    
    # Start Flask server
    app.run(host="0.0.0.0", port=args.port, debug=False)
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path

@ray.remote(num_cpus=1, num_gpus=0.015625)
class FlexibleLlamaInstance:
    """Flexible Ray actor for llama instances with mock mode support"""
    
    def __init__(self, instance_id: int, port: int, mock_mode: bool = False, 
                 model_path: str = "", context_size: int = 2048, 
                 threads: int = 1, gpu_layers: int = 32, llama_path: str = ""):
        self.instance_id = instance_id
        self.port = port
        self.mock_mode = mock_mode
        self.model_path = model_path
        self.context_size = context_size
        self.threads = threads
        self.gpu_layers = gpu_layers
        self.llama_path = llama_path
        self.process = None
        self.is_running = False
        self.base_url = f"http://localhost:{port}"
        
    def start(self) -> Dict[str, any]:
        """Start the instance (mock or real)"""
        try:
            if self.mock_mode:
                # Create mock server
                script_path = create_mock_server(self.port, self.instance_id)
                cmd = ["python3", script_path, "--port", str(self.port)]
            else:
                # Validate real model requirements
                if not self.model_path or not os.path.exists(self.model_path):
                    return {"success": False, "error": f"Model file not found: {self.model_path}"}
                    
                if not os.path.exists(self.llama_path):
                    return {"success": False, "error": f"llama-cli executable not found: {self.llama_path}"}
                
                # Create real server
                script_path = create_llama_server_script(self.instance_id)
                cmd = [
                    "python3", script_path,
                    "--model", self.model_path,
                    "--port", str(self.port),
                    "--context-size", str(self.context_size),
                    "--threads", str(self.threads),
                    "--gpu-layers", str(self.gpu_layers),
                    "--llama-path", self.llama_path
                ]
            
            logger.info(f"Starting {'mock' if self.mock_mode else 'real'} instance {self.instance_id} on port {self.port}")
            
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for server to start
            time.sleep(2 if self.mock_mode else 5)
            
            # Check if process is still running
            if self.process.poll() is None:
                self.is_running = True
                return {
                    "success": True, 
                    "instance_id": self.instance_id, 
                    "port": self.port,
                    "url": self.base_url,
                    "mode": "mock" if self.mock_mode else "real"
                }
            else:
                stderr = self.process.stderr.read()
                return {"success": False, "error": f"Server failed to start: {stderr}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def stop(self) -> bool:
        """Stop the instance"""
        try:
            if self.process and self.is_running:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                
                self.is_running = False
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop instance {self.instance_id}: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 100) -> Dict[str, any]:
        """Generate text using this instance"""
        try:
            if not self.is_running:
                return {"error": "Instance not running"}
            
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, any]:
        """Check instance health"""
        try:
            if not self.is_running:
                return {"instance_id": self.instance_id, "status": "not_running"}
            
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                health_data["port"] = self.port
                return health_data
            else:
                return {"instance_id": self.instance_id, "status": "unhealthy", "port": self.port}
                
        except Exception as e:
            return {"instance_id": self.instance_id, "status": "error", "error": str(e)}

class FlexibleLlamaDeployment:
    """Flexible deployment manager"""
    
    def __init__(self, num_instances: int, base_port: int, mock_mode: bool = False, **kwargs):
        self.num_instances = num_instances
        self.base_port = base_port
        self.mock_mode = mock_mode
        self.kwargs = kwargs
        self.instances: List[ray.ObjectRef] = []
        self.is_deployed = False
        self.instance_ports = []
        
    def deploy(self) -> bool:
        """Deploy all instances"""
        try:
            logger.info(f"Deploying {self.num_instances} {'mock' if self.mock_mode else 'real'} instances...")
            
            # Find free ports
            used_ports = set()
            for i in range(self.num_instances):
                port = find_free_port(self.base_port + i)
                while port in used_ports:
                    port = find_free_port(port + 1)
                used_ports.add(port)
                self.instance_ports.append(port)
            
            # Create instances
            for i in range(self.num_instances):
                instance = FlexibleLlamaInstance.remote(
                    instance_id=i,
                    port=self.instance_ports[i],
                    mock_mode=self.mock_mode,
                    **self.kwargs
                )
                self.instances.append(instance)
            
            # Start all instances
            start_futures = [instance.start.remote() for instance in self.instances]
            results = ray.get(start_futures)
            
            successful_starts = sum(1 for r in results if r.get('success', False))
            logger.info(f"Successfully started {successful_starts}/{self.num_instances} instances")
            
            # Log results
            for i, result in enumerate(results):
                if result.get('success'):
                    logger.info(f"Instance {i}: {result['url']} ({result.get('mode', 'unknown')})")
                else:
                    logger.error(f"Instance {i} failed: {result.get('error', 'Unknown error')}")
            
            if successful_starts > 0:
                self.is_deployed = True
                return True
            else:
                logger.error("Failed to start any instances")
                return False
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown all instances"""
        try:
            if not self.is_deployed:
                return True
                
            logger.info("Shutting down all instances...")
            stop_futures = [instance.stop.remote() for instance in self.instances]
            results = ray.get(stop_futures)
            
            successful_stops = sum(results)
            logger.info(f"Successfully stopped {successful_stops}/{len(self.instances)} instances")
            
            self.instances.clear()
            self.instance_ports.clear()
            self.is_deployed = False
            return True
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            return False
    
    def health_check_all(self) -> List[Dict]:
        """Health check all instances"""
        if not self.is_deployed:
            return []
        
        health_futures = [instance.health_check.remote() for instance in self.instances]
        return ray.get(health_futures)
    
    def test_inference(self) -> List[Dict]:
        """Test inference on all instances"""
        if not self.is_deployed:
            return []
        
        test_prompt = "What is artificial intelligence?"
        test_futures = [instance.generate.remote(test_prompt, 50) for instance in self.instances]
        return ray.get(test_futures)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Flexible Ray-based llama.cpp deployment")
    
    # Basic configuration
    parser.add_argument("--num-instances", type=int, default=4, 
                       help="Number of instances to deploy (default: 4)")
    parser.add_argument("--base-port", type=int, default=8000, 
                       help="Base port for instances (default: 8000)")
    parser.add_argument("--mock", action="store_true", 
                       help="Use mock servers instead of real models")
    
    # Model configuration (for real mode)
    parser.add_argument("--model-path", type=str, default="", 
                       help="Path to GGUF model file")
    parser.add_argument("--llama-path", type=str, 
                       default="/mnt/weka/home/jianshu.she/jianshu/llama.cpp/build/bin/llama-cli",
                       help="Path to llama-cli executable")
    parser.add_argument("--context-size", type=int, default=2048, 
                       help="Context size (default: 2048)")
    parser.add_argument("--threads", type=int, default=1, 
                       help="Threads per instance (default: 1)")
    parser.add_argument("--gpu-layers", type=int, default=32, 
                       help="GPU layers (default: 32)")
    
    # Runtime options
    parser.add_argument("--test-only", action="store_true", 
                       help="Run tests and exit")
    parser.add_argument("--log-level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.log_level)
    
    # Validate arguments
    if not args.mock and not args.model_path:
        logger.error("--model-path is required when not using --mock mode")
        sys.exit(1)
    
    if not args.mock and not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.llama_path):
        logger.error(f"llama-cli executable not found: {args.llama_path}")
        sys.exit(1)
    
    # Initialize Ray
    logger.info("Initializing Ray...")
    ray.init(ignore_reinit_error=True)
    
    # Create deployment
    deployment_kwargs = {}
    if not args.mock:
        deployment_kwargs.update({
            "model_path": args.model_path,
            "llama_path": args.llama_path,
            "context_size": args.context_size,
            "threads": args.threads,
            "gpu_layers": args.gpu_layers
        })
    
    deployment = FlexibleLlamaDeployment(
        num_instances=args.num_instances,
        base_port=args.base_port,
        mock_mode=args.mock,
        **deployment_kwargs
    )
    
    try:
        # Deploy instances
        if not deployment.deploy():
            logger.error("Deployment failed")
            sys.exit(1)
        
        logger.info(f"Deployment successful! {args.num_instances} instances running.")
        
        # Health check
        logger.info("Performing health check...")
        health_results = deployment.health_check_all()
        healthy_count = len([h for h in health_results if h.get('status') == 'healthy'])
        logger.info(f"Health check: {healthy_count}/{len(health_results)} instances healthy")
        
        # Test inference
        logger.info("Testing inference...")
        start_time = time.time()
        inference_results = deployment.test_inference()
        elapsed = time.time() - start_time
        
        successful_inference = len([r for r in inference_results if 'choices' in r])
        logger.info(f"Inference test: {successful_inference}/{len(inference_results)} successful")
        logger.info(f"Total inference time: {elapsed:.2f}s")
        
        # Show sample responses
        for i, result in enumerate(inference_results[:3]):
            if 'choices' in result:
                content = result['choices'][0]['message']['content']
                logger.info(f"Sample response {i+1}: {content[:100]}...")
            else:
                logger.info(f"Sample error {i+1}: {result.get('error', 'Unknown error')}")
        
        if args.test_only:
            logger.info("Test completed, shutting down...")
        else:
            # Keep running
            logger.info("Deployment is running. Press Ctrl+C to shutdown.")
            logger.info(f"Instance URLs: http://localhost:{args.base_port} to http://localhost:{args.base_port + args.num_instances - 1}")
            
            def signal_handler(signum, frame):
                logger.info("Received shutdown signal...")
                deployment.shutdown()
                ray.shutdown()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            try:
                while True:
                    time.sleep(60)
                    health_results = deployment.health_check_all()
                    healthy_count = len([h for h in health_results if h.get('status') == 'healthy'])
                    logger.info(f"Health check: {healthy_count}/{len(health_results)} instances healthy")
            except KeyboardInterrupt:
                pass
        
    finally:
        deployment.shutdown()
        ray.shutdown()

if __name__ == "__main__":
    main()