#!/usr/bin/env python3
"""
Test client for the flexible llama deployment
"""

import requests
import json
import time
import concurrent.futures
import argparse
import sys

def test_single_instance(url: str, prompt: str = "What is AI?") -> dict:
    """Test a single instance"""
    try:
        # Health check
        health_response = requests.get(f"{url}/health", timeout=5)
        health_ok = health_response.status_code == 200
        
        # Chat completion
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        start_time = time.time()
        chat_response = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=30)
        response_time = time.time() - start_time
        
        chat_ok = chat_response.status_code == 200
        
        result = {
            "url": url,
            "health_ok": health_ok,
            "chat_ok": chat_ok,
            "response_time": response_time,
            "status_code": chat_response.status_code
        }
        
        if chat_ok:
            data = chat_response.json()
            if 'choices' in data and len(data['choices']) > 0:
                result["response"] = data['choices'][0]['message']['content']
            else:
                result["error"] = "No choices in response"
        else:
            result["error"] = chat_response.text
            
        return result
        
    except Exception as e:
        return {
            "url": url,
            "health_ok": False,
            "chat_ok": False,
            "error": str(e)
        }

def test_parallel_requests(urls: list, num_requests: int = 10) -> dict:
    """Test parallel requests across instances"""
    base_prompts = [
        "What is artificial intelligence?",
        "Explain machine learning.",
        "What is deep learning?",
        "How do neural networks work?",
        "What is natural language processing?"
    ]
    
    def send_request(i):
        url = urls[i % len(urls)]
        prompt = f"Request {i}: {base_prompts[i % len(base_prompts)]}"
        return test_single_instance(url, prompt)
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(20, num_requests)) as executor:
        futures = [executor.submit(send_request, i) for i in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_time
    
    successful = [r for r in results if r.get('chat_ok', False)]
    failed = [r for r in results if not r.get('chat_ok', False)]
    
    return {
        "total_requests": num_requests,
        "successful": len(successful),
        "failed": len(failed),
        "total_time": total_time,
        "requests_per_second": num_requests / total_time,
        "avg_response_time": sum(r.get('response_time', 0) for r in successful) / len(successful) if successful else 0,
        "results": results
    }

def main():
    parser = argparse.ArgumentParser(description="Test client for llama deployment")
    parser.add_argument("--base-port", type=int, default=8000, help="Base port")
    parser.add_argument("--num-instances", type=int, default=4, help="Number of instances to test")
    parser.add_argument("--parallel-requests", type=int, default=10, help="Number of parallel requests to send")
    parser.add_argument("--prompt", type=str, default="What is artificial intelligence?", help="Test prompt")
    
    args = parser.parse_args()
    
    # Generate URLs
    urls = [f"http://localhost:{args.base_port + i}" for i in range(args.num_instances)]
    
    print(f"Testing {args.num_instances} instances from port {args.base_port}")
    print(f"URLs: {urls[0]} to {urls[-1]}")
    print()
    
    # Test each instance individually
    print("=== Individual Instance Tests ===")
    for i, url in enumerate(urls):
        print(f"Testing instance {i}: {url}")
        result = test_single_instance(url, args.prompt)
        
        if result.get('chat_ok'):
            print(f"  ✓ Success - Response time: {result['response_time']:.2f}s")
            print(f"  Response: {result.get('response', '')[:100]}...")
        else:
            print(f"  ✗ Failed - {result.get('error', 'Unknown error')}")
        print()
    
    # Test parallel requests
    print("=== Parallel Request Test ===")
    print(f"Sending {args.parallel_requests} parallel requests...")
    
    parallel_results = test_parallel_requests(urls, args.parallel_requests)
    
    print(f"Results:")
    print(f"  Total requests: {parallel_results['total_requests']}")
    print(f"  Successful: {parallel_results['successful']}")
    print(f"  Failed: {parallel_results['failed']}")
    print(f"  Total time: {parallel_results['total_time']:.2f}s")
    print(f"  Requests/second: {parallel_results['requests_per_second']:.2f}")
    print(f"  Avg response time: {parallel_results['avg_response_time']:.2f}s")
    print()
    
    # Show sample responses
    successful_results = [r for r in parallel_results['results'] if r.get('chat_ok')]
    if successful_results:
        print("=== Sample Responses ===")
        for i, result in enumerate(successful_results[:3]):
            print(f"Sample {i+1} from {result['url']}:")
            print(f"  {result.get('response', '')[:150]}...")
            print()
    
    # Show errors if any
    failed_results = [r for r in parallel_results['results'] if not r.get('chat_ok')]
    if failed_results:
        print("=== Errors ===")
        for i, result in enumerate(failed_results[:5]):
            print(f"Error {i+1} from {result['url']}: {result.get('error', 'Unknown')}")
        print()
    
    # Summary
    success_rate = parallel_results['successful'] / parallel_results['total_requests'] * 100
    print(f"=== Summary ===")
    print(f"Success rate: {success_rate:.1f}%")
    if success_rate >= 95:
        print("✓ Deployment is working well!")
    elif success_rate >= 80:
        print("⚠ Deployment has some issues but mostly working")
    else:
        print("✗ Deployment has significant issues")

if __name__ == "__main__":
    main()