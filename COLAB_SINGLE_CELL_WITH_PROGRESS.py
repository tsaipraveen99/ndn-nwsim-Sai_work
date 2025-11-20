# ============================================================================
# NDN SIMULATION - COMPLETE SETUP AND RUN WITH REAL-TIME PROGRESS
# Copy this entire cell into Google Colab and run it
# ============================================================================

import os
import sys
import subprocess
import threading
import time
import json
from pathlib import Path
from datetime import datetime

print("="*70)
print("NDN SIMULATION - COMPLETE SETUP")
print("="*70)

# Step 1: Clone repository
print("\n[1/6] Cloning repository...")
result = subprocess.run(['git', 'clone', 'https://github.com/tsaipraveen99/ndn-nwsim-Sai_work.git'], 
                       cwd='/content', capture_output=True, text=True)
if result.returncode != 0 and 'already exists' not in result.stderr.lower():
    print(f"⚠️  {result.stderr}")
else:
    print("✅ Repository ready!")

# Step 2: Navigate to project
os.chdir('/content/ndn-nwsim-Sai_work')
subprocess.run(['git', 'pull', 'origin', 'main'], capture_output=True)
print(f"\n[2/6] Current directory: {os.getcwd()}")

# Step 3: Install dependencies
print("\n[3/6] Installing dependencies...")
subprocess.run(['pip', 'install', '-q', 
                'networkx', 'numpy', 'torch', 'scipy', 'matplotlib', 
                'pandas', 'scikit-learn', 'dill', 'bitarray', 
                'hdbscan', 'tensorflow', 'mmh3'], check=False)
print("✅ Dependencies installed!")

# Step 4: Verify GPU
print("\n[4/6] Checking GPU...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU - will use CPU")
except:
    print("⚠️  Could not check GPU")

# Step 5: Configure simulation
print("\n[5/6] Configuring simulation...")
# REDUCED NETWORK SIZE to prevent queue drain timeouts
os.environ['NDN_SIM_USE_DQN'] = '1'
os.environ['NDN_SIM_NODES'] = '20'              # Reduced from 30
os.environ['NDN_SIM_PRODUCERS'] = '3'           # Reduced from 5
os.environ['NDN_SIM_USERS'] = '30'              # Reduced from 50
os.environ['NDN_SIM_CONTENTS'] = '200'           # Reduced from 300
os.environ['NDN_SIM_ROUNDS'] = '30'             # Reduced from 50
os.environ['NDN_SIM_REQUESTS'] = '20'           # Reduced from 30
os.environ['NDN_SIM_WARMUP_ROUNDS'] = '3'       # Reduced from 5
os.environ['NDN_SIM_CACHE_CAPACITY'] = '500'
os.environ['NDN_SIM_ZIPF_PARAM'] = '1.2'
os.environ['NDN_SIM_TOPOLOGY'] = 'watts_strogatz'
os.environ['NDN_SIM_QUIET'] = '0'                # Show progress!
os.environ['NDN_SIM_SKIP_DELAYS'] = '1'
# INCREASE WORKER TIMEOUT to handle slow DQN operations
os.environ['NDN_SIM_WORKER_TIMEOUT'] = '30.0'   # Increased from 10.0

print("Configuration set!")
print("💡 Network size reduced to prevent timeouts")
print("💡 Worker timeout increased to 30s for DQN operations")

# Step 6: Progress monitor function
def monitor_progress(stop_event):
    """Monitor benchmark progress in background"""
    checkpoint_file = Path('benchmark_checkpoints/benchmark_checkpoint.json')
    results_file = Path('benchmark_checkpoints/benchmark_results.json')
    
    last_algorithm = None
    last_completed = 0
    
    while not stop_event.is_set():
        time.sleep(5)  # Check every 5 seconds
        
        # Check if completed
        if results_file.exists():
            print("\n" + "="*70)
            print("✅ BENCHMARK COMPLETED!")
            print("="*70)
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                print(f"\n{'Policy':<20} {'Hit Rate':<15} {'Std Dev':<15} {'Runs':<10}")
                print("-"*70)
                for name, result in results.items():
                    hit_rate = result.get('hit_rate', 0)
                    std = result.get('hit_rate_std', 0)
                    runs = result.get('num_runs', 0)
                    print(f"{name:<20} {hit_rate:6.2f}%      {std:6.2f}%      {runs:<10}")
            except Exception as e:
                print(f"Error reading results: {e}")
            stop_event.set()
            break
        
        # Check progress
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                algorithm = checkpoint.get('algorithm', 'unknown')
                completed = checkpoint.get('completed_runs', 0)
                total = checkpoint.get('total_runs', 0)
                
                # Only print if something changed
                if algorithm != last_algorithm or completed != last_completed:
                    progress_pct = (completed * 100 // total) if total > 0 else 0
                    print(f"\n⏳ Progress: {algorithm.upper()} - {completed}/{total} runs ({progress_pct}%)", flush=True)
                    last_algorithm = algorithm
                    last_completed = completed
            except:
                pass

# Start progress monitor
print("\n[6/6] Starting benchmark with progress monitoring...")
print("="*70)
print("BENCHMARK RUNNING")
print("="*70)
print("💡 Progress updates will appear every 5 seconds")
print("💡 This may take 10-30 minutes depending on configuration")
print("="*70 + "\n")

stop_monitor = threading.Event()
monitor_thread = threading.Thread(target=monitor_progress, args=(stop_monitor,), daemon=True)
monitor_thread.start()

# Run benchmark with real-time output
try:
    process = subprocess.Popen(
        ['python', '-u', 'benchmark.py'],  # -u for unbuffered output
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end='', flush=True)
    
    return_code = process.wait()
    
    # Stop monitor
    stop_monitor.set()
    time.sleep(1)  # Give monitor time to finish
    
    if return_code == 0:
        print("\n" + "="*70)
        print("✅ BENCHMARK COMPLETED SUCCESSFULLY!")
        print("="*70)
    else:
        print(f"\n⚠️  Benchmark exited with code: {return_code}")
        
except KeyboardInterrupt:
    print("\n⚠️  Benchmark interrupted by user")
    stop_monitor.set()
    if 'process' in locals():
        process.terminate()

# Final results check
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
results_file = Path('benchmark_checkpoints/benchmark_results.json')
if results_file.exists():
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"\n{'Policy':<20} {'Hit Rate':<15} {'Std Dev':<15} {'Runs':<10}")
        print("-"*70)
        for name, result in results.items():
            hit_rate = result.get('hit_rate', 0)
            std = result.get('hit_rate_std', 0)
            runs = result.get('num_runs', 0)
            print(f"{name:<20} {hit_rate:6.2f}%      {std:6.2f}%      {runs:<10}")
    except Exception as e:
        print(f"Error reading results: {e}")
else:
    print("Results file not found. Check output above for errors.")

print("\n" + "="*70)
print("COMPLETE!")
print("="*70)
print("\n💡 TIP: To check progress while running, use a separate cell:")
print("   !python check_colab_progress.py")
print("="*70)

