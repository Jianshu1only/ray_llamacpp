#!/usr/bin/env python3
"""
Ray-based deployment script for 64 instances of 0.5B Qwen2.5 models using llama.cpp
"""

import ray
import subprocess
import time
import os
import logging
import asyncio
from typing import List, Dict, Optional
import psutil
import signal
import sys
from pathlib import Path

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

@ray.remote(num_cpus=1, num_gpus=0.015625)  # 1/64 of GPU per instance
class LlamaInstance:
    """Ray actor for a single llama.cpp instance"""
    
    def __init__(self, instance_id: int, model_path: str, context_size: int = 2048, 
                 threads: int = 1, gpu_layers: int = 32):
        self.instance_id = instance_id
        self.model_path = model_path
        self.context_size = context_size
        self.threads = threads
        self.gpu_layers = gpu_layers
        self.process = None
        self.is_running = False
        
        # Set CUDA device for this instance
        os.environ['CUDA_VISIBLE_DEVICES'] = str(instance_id % torch.cuda.device_count() if 'torch' in sys.modules else '0')
        
    def start(self) -> bool:
        """Start the llama.cpp instance"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            if not os.path.exists(LLAMA_CPP_PATH):
                logger.error(f"llama-cli executable not found: {LLAMA_CPP_PATH}")
                return False
            
            # Build command arguments
            cmd = [
                LLAMA_CPP_PATH,
                "-m", self.model_path,
                "-c", str(self.context_size),
                "-t", str(self.threads),
                "-ngl", str(self.gpu_layers),
                "--interactive",
                "--color",
                "--temp", "0.7",
                "--repeat_penalty", "1.1",
                "-n", "512"  # max tokens
            ]
            
            logger.info(f"Starting instance {self.instance_id} with command: {' '.join(cmd)}")
            
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.is_running = True
            logger.info(f"Instance {self.instance_id} started with PID {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start instance {self.instance_id}: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the llama.cpp instance"""
        try:
            if self.process and self.is_running:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                
                self.is_running = False
                logger.info(f"Instance {self.instance_id} stopped")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to stop instance {self.instance_id}: {e}")
            return False
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text using this instance"""
        try:
            if not self.is_running or not self.process:
                return "Error: Instance not running"
            
            # Send prompt to the process
            self.process.stdin.write(f"{prompt}\n")
            self.process.stdin.flush()
            
            # Read response (simplified - in practice you'd want better parsing)
            output_lines = []
            start_time = time.time()
            timeout = 30  # 30 second timeout
            
            while time.time() - start_time < timeout:
                if self.process.stdout.readable():
                    line = self.process.stdout.readline()
                    if line:
                        output_lines.append(line.strip())
                        if len(output_lines) >= max_tokens or ">" in line:  # Simple end detection
                            break
                time.sleep(0.01)
            
            return "\n".join(output_lines)
            
        except Exception as e:
            logger.error(f"Generation failed for instance {self.instance_id}: {e}")
            return f"Error: {e}"
    
    def health_check(self) -> Dict[str, any]:
        """Check the health of this instance"""
        try:
            is_alive = self.process and self.process.poll() is None if self.is_running else False
            
            memory_info = None
            cpu_percent = None
            if self.process and is_alive:
                try:
                    process = psutil.Process(self.process.pid)
                    memory_info = process.memory_info()
                    cpu_percent = process.cpu_percent()
                except psutil.NoSuchProcess:
                    is_alive = False
            
            return {
                "instance_id": self.instance_id,
                "is_running": self.is_running,
                "is_alive": is_alive,
                "pid": self.process.pid if self.process else None,
                "memory_mb": memory_info.rss / 1024 / 1024 if memory_info else None,
                "cpu_percent": cpu_percent
            }
        except Exception as e:
            return {
                "instance_id": self.instance_id,
                "is_running": False,
                "is_alive": False,
                "error": str(e)
            }

class LlamaDeploymentManager:
    """Manager for all llama instances"""
    
    def __init__(self, model_path: str, num_instances: int = 64):
        self.model_path = model_path
        self.num_instances = num_instances
        self.instances: List[ray.ObjectRef] = []
        self.is_deployed = False
        
    def deploy(self) -> bool:
        """Deploy all instances"""
        try:
            logger.info(f"Deploying {self.num_instances} llama instances...")
            
            # Create and start all instances
            for i in range(self.num_instances):
                instance = LlamaInstance.remote(
                    instance_id=i,
                    model_path=self.model_path,
                    context_size=CONTEXT_SIZE,
                    threads=THREADS_PER_INSTANCE,
                    gpu_layers=GPU_LAYERS
                )
                self.instances.append(instance)
            
            # Start all instances in parallel
            start_futures = [instance.start.remote() for instance in self.instances]
            results = ray.get(start_futures)
            
            successful_starts = sum(results)
            logger.info(f"Successfully started {successful_starts}/{self.num_instances} instances")
            
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
            
            # Stop all instances in parallel
            stop_futures = [instance.stop.remote() for instance in self.instances]
            results = ray.get(stop_futures)
            
            successful_stops = sum(results)
            logger.info(f"Successfully stopped {successful_stops}/{len(self.instances)} instances")
            
            self.instances.clear()
            self.is_deployed = False
            return True
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            return False
    
    def health_check_all(self) -> List[Dict]:
        """Check health of all instances"""
        try:
            if not self.is_deployed:
                return []
                
            health_futures = [instance.health_check.remote() for instance in self.instances]
            health_results = ray.get(health_futures)
            return health_results
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return []
    
    def generate_parallel(self, prompts: List[str], max_tokens: int = 100) -> List[str]:
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
            return [f"Error: {e}" for _ in prompts]

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, cleaning up...")
    if 'manager' in globals():
        manager.shutdown()
    ray.shutdown()
    sys.exit(0)

def main():
    """Main function"""
    global manager
    
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
        manager = LlamaDeploymentManager(MODEL_PATH, NUM_INSTANCES)
        
        # Deploy instances
        if not manager.deploy():
            logger.error("Deployment failed")
            return
        
        logger.info("Deployment successful! All instances are running.")
        
        # Example usage - health check
        logger.info("Performing health check...")
        health_results = manager.health_check_all()
        healthy_instances = [h for h in health_results if h.get('is_alive', False)]
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
        responses = manager.generate_parallel(test_prompts)
        elapsed = time.time() - start_time
        
        logger.info(f"Generated {len(responses)} responses in {elapsed:.2f} seconds")
        for i, response in enumerate(responses):
            logger.info(f"Response {i+1}: {response[:100]}...")
        
        # Keep running
        logger.info("Deployment is running. Press Ctrl+C to shutdown.")
        try:
            while True:
                time.sleep(60)  # Health check every minute
                health_results = manager.health_check_all()
                healthy_count = len([h for h in health_results if h.get('is_alive', False)])
                logger.info(f"Health check: {healthy_count}/{len(health_results)} instances healthy")
        except KeyboardInterrupt:
            pass
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Cleanup
        if 'manager' in locals():
            manager.shutdown()
        ray.shutdown()

if __name__ == "__main__":
    main()