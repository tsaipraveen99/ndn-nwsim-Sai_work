# ============================================================================
# IMPROVED DIAGNOSTIC AND BENCHMARK RUN
# - Reduced verbosity to see actual progress
# - Better error detection
# - Progress monitoring
# ============================================================================

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from threading import Thread, Event

print("="*70)
print("DIAGNOSTIC CHECK")
print("="*70)

# Check directory
print(f"Current directory: {os.getcwd()}")

# Check if benchmark.py exists
benchmark_file = Path('benchmark.py')
if not benchmark_file.exists():
    print("❌ benchmark.py NOT FOUND!")
    print("Files in current directory:")
    for f in sorted(Path('.').glob('*.py'))[:10]:
        print(f"  - {f.name}")
    sys.exit(1)

print(f"✅ benchmark.py found ({benchmark_file.stat().st_size} bytes)")

# Check required files
required_files = ['main.py', 'router.py', 'utils.py', 'endpoints.py', 'dqn_agent.py']
missing = [f for f in required_files if not Path(f).exists()]
if missing:
    print(f"❌ Missing files: {missing}")
    sys.exit(1)
print("✅ All required files present")

# Check Python and imports
print(f"\nPython: {sys.version.split()[0]}")
try:
    import torch
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU - using CPU")
except:
    print("⚠️  Could not check GPU")

# Set configuration - REDUCE VERBOSITY
print("\n" + "="*70)
print("SETTING CONFIGURATION")
print("="*70)

os.environ['NDN_SIM_USE_DQN'] = '1'
os.environ['NDN_SIM_NODES'] = '30'
os.environ['NDN_SIM_PRODUCERS'] = '5'
os.environ['NDN_SIM_USERS'] = '50'
os.environ['NDN_SIM_CONTENTS'] = '300'
os.environ['NDN_SIM_ROUNDS'] = '50'
os.environ['NDN_SIM_REQUESTS'] = '30'
os.environ['NDN_SIM_WARMUP_ROUNDS'] = '5'
os.environ['NDN_SIM_CACHE_CAPACITY'] = '500'
os.environ['NDN_SIM_ZIPF_PARAM'] = '1.2'
os.environ['NDN_SIM_TOPOLOGY'] = 'watts_strogatz'
os.environ['NDN_SIM_QUIET'] = '1'  # REDUCE VERBOSITY - only show important messages
os.environ['NDN_SIM_SKIP_DELAYS'] = '1'

print("Configuration set (QUIET mode enabled to reduce output)")

# Progress monitor function
def monitor_progress(stop_event, checkpoint_file, results_file):
    """Monitor progress in background"""
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

# Run benchmark with filtered output
print("\n" + "="*70)
print("RUNNING BENCHMARK")
print("="*70)
print("💡 Verbose debug output is suppressed (QUIET=1)")
print("💡 Progress updates will appear every 10 seconds")
print("💡 This may take 10-30 minutes depending on configuration")
print("="*70 + "\n")

# Filter keywords to show only important messages
important_keywords = [
    'ERROR', 'Error', 'error', 'Exception', 'Traceback',
    'BENCHMARK', 'Benchmark', 'Starting', 'Completed',
    'Round', 'round', 'Algorithm', 'algorithm',
    'Results', 'results', 'Checkpoint', 'checkpoint',
    'TIMEOUT', 'Timeout', 'STUCK', 'Stuck',
    'Hit rate', 'hit rate', 'Cache', 'cache'
]

def should_print_line(line):
    """Filter to show only important lines"""
    line_lower = line.lower()
    
    # Always show errors and important messages
    for keyword in important_keywords:
        if keyword.lower() in line_lower:
            return True
    
    # Show lines that are not debug spam
    # Skip verbose router/worker messages
    skip_patterns = [
        'worker', 'router', 'handle_interest', 'dispatch_message',
        'processing interest', 'completed dispatch', 'about to call'
    ]
    
    for pattern in skip_patterns:
        if pattern in line_lower:
            return False
    
    # Show summary lines and important status
    if any(x in line_lower for x in ['summary', 'total', 'average', 'mean', 'std', '%']):
        return True
    
    # Show lines that are short (likely summaries, not verbose logs)
    if len(line.strip()) < 100:
        return True
    
    return False

process = subprocess.Popen(
    [sys.executable, '-u', 'benchmark.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True
)

line_count = 0
error_count = 0
last_output_time = time.time()
max_lines = 50000  # Increased limit
max_silence = 300  # 5 minutes without output = likely stuck

try:
    for line in process.stdout:
        line_count += 1
        current_time = time.time()
        
        # Check for errors
        if any(x in line.lower() for x in ['error', 'exception', 'traceback', 'failed']):
            error_count += 1
            print(line, end='', flush=True)
            last_output_time = current_time
        elif should_print_line(line):
            print(line, end='', flush=True)
            last_output_time = current_time
        else:
            # Suppress verbose output but update last_output_time
            last_output_time = current_time
        
        # Safety checks
        if line_count > max_lines:
            print(f"\n⚠️  Output truncated after {max_lines} lines (likely still running)")
            print("💡 Check checkpoint file for progress: benchmark_checkpoints/benchmark_checkpoint.json")
            break
        
        # Check if stuck (no output for too long)
        if current_time - last_output_time > max_silence:
            print(f"\n⚠️  No output for {max_silence}s - simulation may be stuck")
            print("💡 Check checkpoint file for progress")
            break
    
    return_code = process.wait()
    
    # Stop monitor
    stop_monitor.set()
    time.sleep(1)
    
    print("\n" + "="*70)
    if return_code == 0:
        print("✅ BENCHMARK COMPLETED!")
    else:
        print(f"❌ BENCHMARK EXITED WITH CODE: {return_code}")
    if error_count > 0:
        print(f"⚠️  Found {error_count} error messages in output")
    print("="*70)

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    process.terminate()
    stop_monitor.set()
except Exception as e:
    print(f"\n❌ Error running benchmark: {e}")
    import traceback
    traceback.print_exc()
    stop_monitor.set()

# Final results check
print("\n" + "="*70)
print("FINAL RESULTS CHECK")
print("="*70)

if results_file.exists():
    print("✅ Results file found!")
    try:
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
    except:
        pass
else:
    print("❌ No results or checkpoint found")
    print("   Check output above for error messages")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

