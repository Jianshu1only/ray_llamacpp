#!/usr/bin/env python3
"""
Example client to test the Ray-based llama.cpp deployment
"""

import requests
import json
import time
import concurrent.futures
from typing import List, Dict
import random

class LlamaClient:
    """Client for interacting with deployed llama instances"""
    
    def __init__(self, base_urls: List[str]):
        self.base_urls = base_urls
        self.current_index = 0
    
    def get_next_url(self) -> str:
        """Get next URL in round-robin fashion"""
        url = self.base_urls[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.base_urls)
        return url
    
    def chat_completion(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict:
        """Send a chat completion request"""
        url = self.get_next_url()
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{url}/v1/chat/completions",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "url": url}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}", "url": url}
                
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}
    
    def health_check(self, url: str) -> Dict:
        """Check health of a specific instance"""
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return {"success": True, "data": response.json(), "url": url}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}", "url": url}
        except Exception as e:
            return {"success": False, "error": str(e), "url": url}
    
    def health_check_all(self) -> List[Dict]:
        """Check health of all instances in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.health_check, url) for url in self.base_urls]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        return results
    
    def benchmark_parallel(self, prompts: List[str], max_workers: int = 10) -> Dict:
        """Benchmark parallel requests"""
        start_time = time.time()
        
        def send_request(prompt):
            return self.chat_completion(prompt)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(send_request, prompt) for prompt in prompts]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        successful_requests = [r for r in results if r.get('success')]
        failed_requests = [r for r in results if not r.get('success')]
        
        return {
            "total_requests": len(prompts),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "total_time": end_time - start_time,
            "requests_per_second": len(prompts) / (end_time - start_time),
            "results": results
        }

def generate_test_prompts(count: int) -> List[str]:
    """Generate test prompts for benchmarking"""
    base_prompts = [
        "What is artificial intelligence?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about technology.",
        "How does machine learning work?",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "What is the theory of relativity?",
        "How do computers process information?",
        "What is the importance of biodiversity?",
        "Explain the concept of blockchain technology."
    ]
    
    prompts = []
    for i in range(count):
        base_prompt = random.choice(base_prompts)
        prompts.append(f"{base_prompt} (Request #{i+1})")
    
    return prompts

def main():
    """Main function for testing"""
    # Configuration - adjust these based on your deployment
    base_port = 8000
    num_instances = 64
    
    # Generate URLs for all instances
    urls = [f"http://localhost:{base_port + i}" for i in range(num_instances)]
    
    print(f"Testing deployment with {num_instances} instances...")
    print(f"Instance URLs: {urls[0]} to {urls[-1]}")
    
    # Create client
    client = LlamaClient(urls)
    
    # Health check
    print("\\nPerforming health check...")
    health_results = client.health_check_all()
    healthy_instances = [r for r in health_results if r.get('success')]
    print(f"Health check: {len(healthy_instances)}/{len(health_results)} instances healthy")
    
    if len(healthy_instances) == 0:
        print("No healthy instances found. Make sure the deployment is running.")
        return
    
    # Update client to only use healthy instances
    healthy_urls = [r['url'] for r in healthy_instances]
    client = LlamaClient(healthy_urls)
    print(f"Using {len(healthy_urls)} healthy instances for testing")
    
    # Single request test
    print("\\nTesting single request...")
    test_prompt = "What is the capital of France?"
    result = client.chat_completion(test_prompt)
    
    if result['success']:
        response = result['data']['choices'][0]['message']['content']
        print(f"Single request successful!")
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response[:200]}...")
        print(f"Used instance: {result['url']}")
    else:
        print(f"Single request failed: {result['error']}")
    
    # Parallel benchmark test
    print("\\nRunning parallel benchmark...")
    test_prompts = generate_test_prompts(20)  # Test with 20 parallel requests
    
    benchmark_results = client.benchmark_parallel(test_prompts, max_workers=10)
    
    print(f"Benchmark Results:")
    print(f"  Total requests: {benchmark_results['total_requests']}")
    print(f"  Successful: {benchmark_results['successful_requests']}")
    print(f"  Failed: {benchmark_results['failed_requests']}")
    print(f"  Total time: {benchmark_results['total_time']:.2f}s")
    print(f"  Requests/second: {benchmark_results['requests_per_second']:.2f}")
    
    # Show some sample responses
    successful_results = [r for r in benchmark_results['results'] if r.get('success')]
    if successful_results:
        print("\\nSample responses:")
        for i, result in enumerate(successful_results[:3]):
            response = result['data']['choices'][0]['message']['content']
            print(f"  {i+1}. {response[:100]}...")
    
    # Show error details if any
    failed_results = [r for r in benchmark_results['results'] if not r.get('success')]
    if failed_results:
        print("\\nError details:")
        for result in failed_results[:3]:
            print(f"  Error from {result['url']}: {result['error']}")

if __name__ == "__main__":
    main()