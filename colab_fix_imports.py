#!/usr/bin/env python3
"""
Quick fix script for Colab - checks and fixes missing imports
Run this in Colab to diagnose and fix import issues
"""

import os
import sys

print("="*60)
print("COLAB FILE CHECKER & FIXER")
print("="*60)

# Required files
required_files = [
    'main.py',
    'router.py',
    'endpoints.py',
    'utils.py',
    'dqn_agent.py',
    'packet.py',
    'metrics.py',
    'benchmark.py',
    'statistical_analysis.py'
]

print("\n📋 Checking required files...")
print("-"*60)

missing = []
present = []

for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"  ✅ {file:<30} ({size:,} bytes)")
        present.append(file)
    else:
        print(f"  ❌ {file:<30} MISSING")
        missing.append(file)

print("\n" + "="*60)

if missing:
    print(f"⚠️  Missing {len(missing)} file(s):")
    for f in missing:
        print(f"   - {f}")
    
    print("\n📤 SOLUTION: Upload missing files")
    print("-"*60)
    print("Run this in a new cell:")
    print()
    print("from google.colab import files")
    print("files.upload()  # Select all missing files")
    print()
    print("Then re-run this script to verify.")
else:
    print("✅ All required files present!")
    
    # Check Python path
    print("\n🔍 Checking Python path...")
    current_dir = os.getcwd()
    print(f"   Current directory: {current_dir}")
    
    if current_dir not in sys.path:
        print(f"   ⚠️  Current directory not in sys.path")
        print(f"   ✅ Adding current directory to sys.path...")
        sys.path.insert(0, current_dir)
        print(f"   ✅ Fixed!")
    else:
        print(f"   ✅ Current directory in sys.path")
    
    # Try importing
    print("\n🧪 Testing imports...")
    print("-"*60)
    
    try:
        import main
        print("  ✅ main.py imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import main.py: {e}")
        print(f"   💡 Try: sys.path.insert(0, '.')")
    
    try:
        import router
        print("  ✅ router.py imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import router.py: {e}")
    
    try:
        import utils
        print("  ✅ utils.py imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import utils.py: {e}")
    
    try:
        from benchmark import run_benchmark
        print("  ✅ benchmark.py imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import benchmark.py: {e}")
    
    print("\n" + "="*60)
    print("✅ Ready to run benchmark!" if not missing else "⚠️  Upload missing files first")
    print("="*60)

