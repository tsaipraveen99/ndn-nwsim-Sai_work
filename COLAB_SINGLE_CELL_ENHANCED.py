# ============================================================================
# NDN SIMULATION - COMPLETE SETUP AND RUN (ENHANCED SINGLE CELL)
# ============================================================================
# Copy and paste this entire cell into Google Colab and run it
# Features: Progress monitoring, real-time output, error handling
# 
# ⚠️ IMPORTANT: To prevent Colab from disconnecting, run the keep-alive script
#    in a SEPARATE cell. See COLAB_KEEP_ALIVE_GUIDE.md for details.
#    Quick fix: Run this JavaScript in another cell:
#    function ClickConnect(){document.querySelector("colab-toolbar-button#connect").click()}
#    setInterval(ClickConnect,60000)
# ============================================================================

import os
import sys
import subprocess
import time
import json
import threading
from pathlib import Path
from threading import Thread, Event

print("="*70)
print("NDN SIMULATION - COMPLETE SETUP")
print("="*70)

# Step 1: Clone repository if not already cloned
repo_dir = Path('/content/ndn-nwsim-Sai_work')
if not repo_dir.exists():
    print("\n[1/6] Cloning repository from GitHub...")
    subprocess.run(['git', 'clone', 'https://github.com/tsaipraveen99/ndn-nwsim-Sai_work.git'], 
                  cwd='/content', check=True)
    print("✅ Repository cloned!")
else:
    print("\n[1/6] Repository already exists")

# Step 1.5: ALWAYS pull latest changes from GitHub (even if repo exists)
print("\n[1.5/6] Pulling latest changes from GitHub...")
os.chdir(str(repo_dir))  # Navigate to repo directory first

try:
    # First, fetch latest changes
    fetch_result = subprocess.run(['git', 'fetch', 'origin', 'main'], 
                                 capture_output=True, 
                                 text=True,
                                 check=False)
    
    # Then pull with rebase to avoid merge conflicts
    pull_result = subprocess.run(['git', 'pull', '--rebase', 'origin', 'main'], 
                                capture_output=True, 
                                text=True,
                                check=False)
    
    if pull_result.returncode == 0:
        if 'Already up to date' in pull_result.stdout or 'up to date' in pull_result.stdout.lower():
            print("✅ Repository is up to date with latest changes")
        else:
            print("✅ Repository updated with latest changes!")
            # Show what changed
            if pull_result.stdout.strip():
                output_lines = pull_result.stdout.strip().split('\n')
                for line in output_lines[:3]:  # Show first 3 lines
                    if line.strip() and not line.startswith('Updating'):
                        print(f"   {line[:80]}")
    else:
        # If rebase fails, try regular pull
        print("⚠️  Rebase pull failed, trying regular pull...")
        pull_result2 = subprocess.run(['git', 'pull', 'origin', 'main'], 
                                    capture_output=True, 
                                    text=True,
                                    check=False)
        if pull_result2.returncode == 0:
            print("✅ Repository updated with latest changes (regular pull)")
        else:
            # If that fails, reset and pull
            print("⚠️  Regular pull failed, resetting and pulling...")
            subprocess.run(['git', 'reset', '--hard', 'origin/main'], 
                         capture_output=True, 
                         check=False)
            final_pull = subprocess.run(['git', 'pull', 'origin', 'main'], 
                                       capture_output=True, 
                                       text=True,
                                       check=False)
            if final_pull.returncode == 0:
                print("✅ Repository reset and updated with latest changes")
            else:
                print(f"⚠️  Could not pull latest changes: {final_pull.stderr[:200]}")
                print("   Continuing with existing code...")
                
except Exception as e:
    print(f"⚠️  Error during git pull: {e}")
    print("   Attempting to continue with existing code...")
    # Try one more time with simple pull
    try:
        simple_pull = subprocess.run(['git', 'pull'], 
                                    capture_output=True, 
                                    text=True,
                                    check=False,
                                    timeout=10)
        if simple_pull.returncode == 0:
            print("✅ Successfully pulled latest changes (simple pull)")
    except:
        print("   Using existing code - you may not have the latest version")

# Step 2: Navigate to project directory
os.chdir(str(repo_dir))
print(f"\n[2/6] Current directory: {os.getcwd()}")

# Step 3: Install dependencies
print("\n[3/6] Installing dependencies...")
subprocess.run(['pip', 'install', '-q', 
                'networkx', 'numpy', 'torch', 'scipy', 'matplotlib', 
                'pandas', 'scikit-learn', 'dill', 'bitarray', 
                'hdbscan', 'tensorflow', 'mmh3'], check=False)
print("✅ Dependencies installed!")

# Step 4: Verify GPU (optional but recommended)
print("\n[4/6] Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected - simulation will run on CPU (slower)")
except:
    print("⚠️  Could not check GPU status")

# Step 5: Configure simulation
print("\n[5/6] Configuring simulation parameters...")
# REDUCED NETWORK SIZE to prevent queue drain timeouts
os.environ['NDN_SIM_USE_DQN'] = '1'              # Enable DQN
os.environ['NDN_SIM_NODES'] = '20'              # Core routers (reduced from 30)
os.environ['NDN_SIM_PRODUCERS'] = '3'           # Producers (reduced from 5)
os.environ['NDN_SIM_USERS'] = '30'              # Users (reduced from 50)
os.environ['NDN_SIM_CONTENTS'] = '200'          # Content catalog (reduced from 300)
os.environ['NDN_SIM_ROUNDS'] = '30'            # Simulation rounds (reduced from 50)
os.environ['NDN_SIM_REQUESTS'] = '20'          # Requests per round (reduced from 30)
os.environ['NDN_SIM_WARMUP_ROUNDS'] = '3'       # Warm-up rounds (reduced from 5)
os.environ['NDN_SIM_CACHE_CAPACITY'] = '500'    # Cache capacity
os.environ['NDN_SIM_ZIPF_PARAM'] = '1.2'        # Zipf parameter
os.environ['NDN_SIM_TOPOLOGY'] = 'watts_strogatz' # Network topology
os.environ['NDN_SIM_QUIET'] = '0'               # Show progress (changed from 1)
os.environ['NDN_SIM_SKIP_DELAYS'] = '1'         # Skip sleep delays
# INCREASE WORKER TIMEOUT to handle slow DQN operations
os.environ['NDN_SIM_WORKER_TIMEOUT'] = '30.0'   # Increased from default 10.0
# ENABLE BACKPRESSURE to prevent queue flooding
os.environ['NDN_SIM_MAX_QUEUE_SIZE'] = '5000'   # Backpressure threshold
# FIB UPDATE RATE LIMITING: Prevent FIB update queue flooding
os.environ['NDN_SIM_MAX_FIB_RATE'] = '30'       # Max 30 FIB updates per second (reduced from default 50)
# MAX_FIB_PROPAGATION: Limit neighbors per FIB update (0 = no limit, rely on rate limiting only)
# For Watts-Strogatz k=4, most routers have ~4 neighbors, so 5-8 is reasonable
os.environ['NDN_SIM_MAX_FIB_PROPAGATION'] = '8'  # Max 8 neighbors per FIB propagation (covers most routers)

print("Configuration:")
for key, value in sorted(os.environ.items()):
    if key.startswith('NDN_SIM_'):
        print(f"  {key} = {value}")

# Step 6: Verify benchmark.py exists
if not Path('benchmark.py').exists():
    print("\n❌ ERROR: benchmark.py not found!")
    print(f"Current directory: {os.getcwd()}")
    print("Files in directory:")
    for f in Path('.').glob('*.py'):
        print(f"  - {f}")
    raise FileNotFoundError("benchmark.py not found in project directory")

# Progress monitor function
def monitor_progress(stop_event, checkpoint_file, results_file):
    """Monitor benchmark progress in background"""
    last_algorithm = None
    last_completed = 0
    last_update = time.time()
    
    while not stop_event.is_set():
        time.sleep(10)  # Check every 10 seconds
        
        # Check if completed
        if results_file.exists():
            print("\n" + "="*70)
            print("✅ BENCHMARK COMPLETED!")
            print("="*70)
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
                    print(f"\n📊 Progress: {algorithm.upper()} - {completed}/{total} runs ({progress_pct}%)")
                    last_algorithm = algorithm
                    last_completed = completed
                    last_update = time.time()
                elif time.time() - last_update > 60:  # Print every minute even if no change
                    progress_pct = (completed * 100 // total) if total > 0 else 0
                    print(f"⏳ Still running: {algorithm.upper()} - {completed}/{total} runs ({progress_pct}%)")
                    last_update = time.time()
            except:
                pass

# Start progress monitor
checkpoint_file = Path('benchmark_checkpoints/benchmark_checkpoint.json')
results_file = Path('benchmark_checkpoints/benchmark_results.json')
stop_monitor = Event()
monitor_thread = Thread(target=monitor_progress, args=(stop_monitor, checkpoint_file, results_file), daemon=True)
monitor_thread.start()

# Step 7: Run benchmark with real-time output
print("\n" + "="*70)
print("STARTING BENCHMARK")
print("="*70)
print(f"Working directory: {os.getcwd()}")
print("💡 Progress updates will appear every 10 seconds")
print("💡 This may take 10-30 minutes depending on configuration")
print("="*70 + "\n")

# Run benchmark with unbuffered output and real-time streaming
try:
    process = subprocess.Popen(
        [sys.executable, '-u', 'benchmark.py'],  # -u for unbuffered output
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
    
    print("\n" + "="*70)
    if return_code == 0:
        print("✅ BENCHMARK COMPLETED SUCCESSFULLY!")
    else:
        print(f"⚠️  BENCHMARK EXITED WITH CODE: {return_code}")
    print("="*70)
    
except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    stop_monitor.set()
    if 'process' in locals():
        process.terminate()
except Exception as e:
    print(f"\n❌ Error running benchmark: {e}")
    import traceback
    traceback.print_exc()
    stop_monitor.set()

# Step 8: Display results if available
print("\n" + "="*70)
print("FINAL RESULTS CHECK")
print("="*70)

if results_file.exists():
    print("✅ Results file found!")
    try:
        import json
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"\n{'Policy':<20} {'Hit Rate':<15} {'Std Dev':<15} {'Runs':<10}")
        print("-"*70)
        for name, result in sorted(results.items()):
            hit_rate = result.get('hit_rate', 0)
            std = result.get('hit_rate_std', 0)
            runs = result.get('num_runs', 0)
            print(f"{name:<20} {hit_rate:6.2f}%      {std:6.2f}%      {runs:<10}")
    except Exception as e:
        print(f"❌ Error reading results: {e}")
elif checkpoint_file.exists():
    print("⏳ Benchmark incomplete - checkpoint found")
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        algorithm = checkpoint.get('algorithm', 'unknown')
        completed = checkpoint.get('completed_runs', 0)
        total = checkpoint.get('total_runs', 0)
        print(f"Progress: {algorithm} - {completed}/{total} runs")
        print("💡 Re-run this cell to continue, or check COLAB_RESUME_OR_CHECK.py")
    except:
        pass
else:
    print("⚠️  Results file not found. Check output above for errors.")

print("\n" + "="*70)
print("SETUP AND EXECUTION COMPLETE")
print("="*70)

