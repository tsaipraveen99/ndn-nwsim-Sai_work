#!/usr/bin/env python3
"""
Script to check and display router runtime metrics.
Useful for debugging performance issues and understanding system behavior.
"""

import json
import sys
from pathlib import Path

def print_metrics_summary(metrics):
    """Print a human-readable summary of metrics"""
    print("="*70)
    print("ROUTER RUNTIME METRICS SUMMARY")
    print("="*70)
    
    # Total messages processed
    total = metrics.get('total_messages_processed', 0)
    print(f"\n📊 Total Messages Processed: {total:,}")
    
    # Average processing times
    avg_times = metrics.get('average_processing_times', {})
    if avg_times:
        print("\n⏱️  Average Processing Times by Message Type:")
        print("-"*70)
        for msg_type, avg_time in sorted(avg_times.items()):
            print(f"  {msg_type:20s}: {avg_time*1000:8.2f} ms")
    
    # Timeout statistics
    timeout_count = metrics.get('timeout_count', {})
    timeout_rates = metrics.get('timeout_rate', {})
    if timeout_count:
        print("\n⚠️  Timeout Statistics:")
        print("-"*70)
        total_timeouts = sum(timeout_count.values())
        print(f"  Total Timeouts: {total_timeouts}")
        for msg_type in sorted(timeout_count.keys()):
            count = timeout_count[msg_type]
            rate = timeout_rates.get(msg_type, 0.0) * 100
            if count > 0:
                print(f"  {msg_type:20s}: {count:5d} timeouts ({rate:5.2f}% timeout rate)")
    
    # Queue size history
    queue_history = metrics.get('queue_size_history', [])
    if queue_history:
        print("\n📈 Queue Size History (last 10 samples):")
        print("-"*70)
        recent = queue_history[-10:]
        for sample in recent:
            size = sample.get('size', 'unknown')
            elapsed = sample.get('elapsed', 0)
            print(f"  Time: {elapsed:6.2f}s, Queue Size: {size}")
        
        # Calculate average queue size
        sizes = [s.get('size', 0) for s in queue_history if isinstance(s.get('size'), (int, float)) and s.get('size') >= 0]
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            max_size = max(sizes)
            print(f"\n  Average Queue Size: {avg_size:.1f}")
            print(f"  Maximum Queue Size: {max_size}")

def main():
    """Main function to retrieve and display metrics"""
    # Try to get metrics from a running simulation
    # This requires access to the RouterRuntime instance
    # For now, we'll show how to use it programmatically
    
    print("="*70)
    print("ROUTER METRICS CHECKER")
    print("="*70)
    print("\n💡 This script demonstrates how to access metrics.")
    print("💡 To use in your code:")
    print("\n   from router import RouterRuntime")
    print("   # Get runtime instance (from your network setup)")
    print("   metrics = runtime.get_metrics()")
    print("   print_metrics_summary(metrics)")
    print("\n" + "="*70)
    
    # Example: Show how to save metrics to file
    print("\n📝 Example: Saving metrics to file")
    print("-"*70)
    example_metrics = {
        'total_messages_processed': 0,
        'average_processing_times': {},
        'timeout_count': {},
        'timeout_rate': {},
        'queue_size_history': []
    }
    
    print("\nTo save metrics after simulation:")
    print("  metrics = runtime.get_metrics()")
    print("  with open('simulation_metrics.json', 'w') as f:")
    print("      json.dump(metrics, f, indent=2)")
    
    print("\n" + "="*70)
    print("For real-time metrics, add this to your simulation code:")
    print("="*70)
    print("""
# In your simulation loop or after completion:
from router import RouterRuntime

# Get the runtime (assuming you have access to it)
# This is typically available from create_network() return value
runtime = ...  # Your RouterRuntime instance

# Get metrics
metrics = runtime.get_metrics()

# Print summary
print_metrics_summary(metrics)

# Or save to file for analysis
with open('simulation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
""")

if __name__ == '__main__':
    main()

