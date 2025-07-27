#!/usr/bin/env python3
"""
Demo script showing different deployment configurations
"""

import subprocess
import time
import sys
import os

def run_command(cmd, description, timeout=120):
    """Run a command with description and timeout"""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    """Run demo scenarios"""
    print("Ray Llama.cpp Deployment Demo")
    print("=============================")
    
    scenarios = [
        {
            "name": "Quick Mock Test (4 instances)",
            "cmd": ["python3", "ray_llama_flexible.py", "--mock", "--num-instances", "4", "--test-only"],
            "timeout": 60
        },
        {
            "name": "Scale Test (16 mock instances)",
            "cmd": ["python3", "ray_llama_flexible.py", "--mock", "--num-instances", "16", "--test-only"],
            "timeout": 90
        },
        {
            "name": "Performance Test (32 mock instances)",
            "cmd": ["python3", "ray_llama_flexible.py", "--mock", "--num-instances", "32", "--test-only", "--base-port", "9000"],
            "timeout": 120
        },
        {
            "name": "Debug Mode Test",
            "cmd": ["python3", "ray_llama_flexible.py", "--mock", "--num-instances", "4", "--test-only", "--log-level", "DEBUG"],
            "timeout": 60
        }
    ]
    
    success_count = 0
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🚀 Running Demo {i}/{len(scenarios)}: {scenario['name']}")
        
        if run_command(scenario["cmd"], scenario["name"], scenario["timeout"]):
            print("✅ SUCCESS")
            success_count += 1
        else:
            print("❌ FAILED")
        
        if i < len(scenarios):
            print("\nWaiting 5 seconds before next demo...")
            time.sleep(5)
    
    print(f"\n{'='*60}")
    print(f"DEMO SUMMARY")
    print(f"{'='*60}")
    print(f"Completed: {success_count}/{len(scenarios)} scenarios")
    
    if success_count == len(scenarios):
        print("🎉 All demos completed successfully!")
        print("\nYou can now:")
        print("1. Deploy with real models using --model-path")
        print("2. Scale up to 64 instances")
        print("3. Use the test client for validation")
        print("\nExample with real model:")
        print("python3 ray_llama_flexible.py --num-instances 8 --model-path /path/to/model.gguf")
    else:
        print("⚠️  Some demos failed. Please check the logs above.")
    
    print(f"\nFor detailed usage instructions, see: USAGE_GUIDE.md")

if __name__ == "__main__":
    main()