#!/usr/bin/env python3
"""
Ray-based deployment script for 64 HTTP server instances of 0.5B Qwen2.5 models using llama.cpp
This version creates HTTP servers instead of direct CLI interactions for better scalability.
"""

import ray
import subprocess
import time
import os
import logging
import requests
import json
from typing import List, Dict, Optional
import psutil
import signal
import sys
from pathlib import Path
import threading
import socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
LLAMA_CPP_PATH = "/mnt/weka/home/jianshu.she/jianshu/llama.cpp/build/bin/llama-cli"
MODEL_PATH = ""  # Set this to your Qwen2.5 0.5B GGUF model path
NUM_INSTANCES = 64
BASE_PORT = 8000
CONTEXT_SIZE = 2048
THREADS_PER_INSTANCE = 1
GPU_LAYERS = 32  # Adjust based on your GPU memory

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

@ray.remote(num_cpus=1, num_gpus=0.015625)  # 1/64 of GPU per instance
class LlamaServerInstance:
    """Ray actor for a single llama.cpp server instance"""
    
    def __init__(self, instance_id: int, model_path: str, port: int, 
                 context_size: int = 2048, threads: int = 1, gpu_layers: int = 32):
        self.instance_id = instance_id
        self.model_path = model_path
        self.port = port
        self.context_size = context_size
        self.threads = threads
        self.gpu_layers = gpu_layers
        self.process = None
        self.is_running = False
        self.base_url = f"http://localhost:{port}"
        
    def start(self) -> Dict[str, any]:
        """Start the llama.cpp server instance"""
        try:
            if not os.path.exists(self.model_path):
                return {"success": False, "error": f"Model file not found: {self.model_path}"}
                
            if not os.path.exists(LLAMA_CPP_PATH):
                return {"success": False, "error": f"llama-cli executable not found: {LLAMA_CPP_PATH}"}
            
            # Create a simple HTTP server wrapper using llama-cli
            # Since the server build failed, we'll create a simple wrapper
            server_script = self._create_server_script()
            
            cmd = [
                "python3", server_script,
                "--model", self.model_path,
                "--port", str(self.port),
                "--context-size", str(self.context_size),
                "--threads", str(self.threads),
                "--gpu-layers", str(self.gpu_layers),
                "--llama-path", LLAMA_CPP_PATH
            ]
            
            logger.info(f"Starting server instance {self.instance_id} on port {self.port}")
            
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a bit for server to start
            time.sleep(3)
            
            # Check if process is still running
            if self.process.poll() is None:
                self.is_running = True
                logger.info(f"Server instance {self.instance_id} started on port {self.port}")
                return {
                    "success": True, 
                    "instance_id": self.instance_id, 
                    "port": self.port,
                    "url": self.base_url
                }
            else:
                stderr = self.process.stderr.read()
                return {"success": False, "error": f"Server failed to start: {stderr}"}
            
        except Exception as e:
            logger.error(f"Failed to start server instance {self.instance_id}: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_server_script(self) -> str:
        """Create a simple HTTP server wrapper script"""
        script_path = f"/tmp/llama_server_{self.instance_id}.py"
        
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
                        if len(response_lines) >= 50 or ">" in line:
                            break
                time.sleep(0.01)
            
            response_queue.put("\\n".join(response_lines))
            
        except queue.Empty:
            continue
        except Exception as e:
            response_queue.put(f"Error: {{e}}")

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
    return jsonify({{"status": "healthy" if is_alive else "unhealthy", "instance_id": {self.instance_id}}})

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
    
    # Start Flask server
    app.run(host="0.0.0.0", port=args.port, debug=False)
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        return script_path
    
    def stop(self) -> bool:
        """Stop the server instance"""
        try:
            if self.process and self.is_running:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                
                self.is_running = False
                logger.info(f"Server instance {self.instance_id} stopped")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop server instance {self.instance_id}: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 100) -> Dict[str, any]:
        """Generate text using this server instance via HTTP"""
        try:
            if not self.is_running:
                return {"error": "Server instance not running"}
            
            # Prepare chat completion request
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
        """Check the health of this server instance"""
        try:
            if not self.is_running:
                return {"instance_id": self.instance_id, "status": "not_running"}
            
            # HTTP health check
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    health_data["port"] = self.port
                    return health_data
                else:
                    return {"instance_id": self.instance_id, "status": "unhealthy", "port": self.port}
            except requests.exceptions.RequestException:
                return {"instance_id": self.instance_id, "status": "unreachable", "port": self.port}
                
        except Exception as e:
            return {"instance_id": self.instance_id, "status": "error", "error": str(e)}

class LlamaServerDeployment:
    """Manager for all llama server instances"""
    
    def __init__(self, model_path: str, num_instances: int = 64):
        self.model_path = model_path
        self.num_instances = num_instances
        self.instances: List[ray.ObjectRef] = []
        self.is_deployed = False
        self.instance_ports = []
        
    def deploy(self) -> bool:
        """Deploy all server instances"""
        try:
            logger.info(f"Deploying {self.num_instances} llama server instances...")
            
            # Find free ports for all instances
            used_ports = set()
            for i in range(self.num_instances):
                port = find_free_port(BASE_PORT + i)
                while port in used_ports:
                    port = find_free_port(port + 1)
                used_ports.add(port)
                self.instance_ports.append(port)
            
            # Create and start all instances
            for i in range(self.num_instances):
                instance = LlamaServerInstance.remote(
                    instance_id=i,
                    model_path=self.model_path,
                    port=self.instance_ports[i],
                    context_size=CONTEXT_SIZE,
                    threads=THREADS_PER_INSTANCE,
                    gpu_layers=GPU_LAYERS
                )
                self.instances.append(instance)
            
            # Start all instances in parallel
            start_futures = [instance.start.remote() for instance in self.instances]
            results = ray.get(start_futures)
            
            successful_starts = sum(1 for r in results if r.get('success', False))
            logger.info(f"Successfully started {successful_starts}/{self.num_instances} server instances")
            
            for i, result in enumerate(results):
                if result.get('success'):
                    logger.info(f"Instance {i}: {result['url']}")
                else:
                    logger.error(f"Instance {i} failed: {result.get('error', 'Unknown error')}")
            
            if successful_starts > 0:
                self.is_deployed = True
                return True
            else:
                logger.error("Failed to start any server instances")
                return False
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown all server instances"""
        try:
            if not self.is_deployed:
                return True
                
            logger.info("Shutting down all server instances...")
            
            # Stop all instances in parallel
            stop_futures = [instance.stop.remote() for instance in self.instances]
            results = ray.get(stop_futures)
            
            successful_stops = sum(results)
            logger.info(f"Successfully stopped {successful_stops}/{len(self.instances)} server instances")
            
            self.instances.clear()
            self.instance_ports.clear()
            self.is_deployed = False
            return True
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            return False
    
    def health_check_all(self) -> List[Dict]:
        """Check health of all server instances"""
        try:
            if not self.is_deployed:
                return []
                
            health_futures = [instance.health_check.remote() for instance in self.instances]
            health_results = ray.get(health_futures)
            return health_results
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return []
    
    def generate_parallel(self, prompts: List[str], max_tokens: int = 100) -> List[Dict]:
        """Generate responses for multiple prompts in parallel"""
        try:
            if not self.is_deployed or len(prompts) == 0:
                return []
            
            # Distribute prompts across available instances
            num_available = len(self.instances)
            results = []
            
            for i, prompt in enumerate(prompts):
                instance_idx = i % num_available
                future = self.instances[instance_idx].generate.remote(prompt, max_tokens)
                results.append(future)
            
            return ray.get(results)
            
        except Exception as e:
            logger.error(f"Parallel generation failed: {e}")
            return [{"error": str(e)} for _ in prompts]
    
    def get_instance_urls(self) -> List[str]:
        """Get URLs of all running instances"""
        return [f"http://localhost:{port}" for port in self.instance_ports]

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, cleaning up...")
    if 'deployment' in globals():
        deployment.shutdown()
    ray.shutdown()
    sys.exit(0)

def main():
    """Main function"""
    global deployment
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check if model path is provided
    if not MODEL_PATH:
        logger.error("Please set MODEL_PATH to your Qwen2.5 0.5B GGUF model file")
        logger.info("Example: MODEL_PATH = '/path/to/qwen2.5-0.5b-instruct-q4_0.gguf'")
        return
    
    try:
        # Initialize Ray
        logger.info("Initializing Ray...")
        ray.init(ignore_reinit_error=True)
        
        # Create deployment manager
        deployment = LlamaServerDeployment(MODEL_PATH, NUM_INSTANCES)
        
        # Deploy instances
        if not deployment.deploy():
            logger.error("Deployment failed")
            return
        
        logger.info("Deployment successful! All server instances are running.")
        
        # Print instance URLs
        urls = deployment.get_instance_urls()
        logger.info(f"Server instances available at:")
        for i, url in enumerate(urls[:5]):  # Show first 5 URLs
            logger.info(f"  Instance {i}: {url}")
        if len(urls) > 5:
            logger.info(f"  ... and {len(urls) - 5} more instances")
        
        # Example usage - health check
        logger.info("Performing health check...")
        health_results = deployment.health_check_all()
        healthy_instances = [h for h in health_results if h.get('status') == 'healthy']
        logger.info(f"Health check: {len(healthy_instances)}/{len(health_results)} instances healthy")
        
        # Example usage - parallel generation
        test_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about technology.",
            "How does machine learning work?"
        ]
        
        logger.info("Testing parallel generation...")
        start_time = time.time()
        responses = deployment.generate_parallel(test_prompts)
        elapsed = time.time() - start_time
        
        logger.info(f"Generated {len(responses)} responses in {elapsed:.2f} seconds")
        for i, response in enumerate(responses):
            if 'choices' in response:
                content = response['choices'][0]['message']['content']
                logger.info(f"Response {i+1}: {content[:100]}...")
            else:
                logger.info(f"Response {i+1}: {response}")
        
        # Keep running
        logger.info("Deployment is running. Press Ctrl+C to shutdown.")
        logger.info("You can now send requests to any of the server instances.")
        try:
            while True:
                time.sleep(60)  # Health check every minute
                health_results = deployment.health_check_all()
                healthy_count = len([h for h in health_results if h.get('status') == 'healthy'])
                logger.info(f"Health check: {healthy_count}/{len(health_results)} instances healthy")
        except KeyboardInterrupt:
            pass
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Cleanup
        if 'deployment' in locals():
            deployment.shutdown()
        ray.shutdown()

if __name__ == "__main__":
    main()