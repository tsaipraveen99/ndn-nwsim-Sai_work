#!/usr/bin/env python3
"""
Cloud GPU Setup Script for NDN Simulation
Automatically detects and configures GPU for optimal performance
"""

import os
import sys
import torch

def check_gpu_status():
    """Check available GPU devices"""
    print("=" * 70)
    print("GPU STATUS CHECK")
    print("=" * 70)
    
    status = {
        'cuda_available': False,
        'mps_available': False,
        'device': None,
        'device_name': None
    }
    
    # Check CUDA
    if torch.cuda.is_available():
        status['cuda_available'] = True
        status['device'] = torch.device('cuda')
        status['device_name'] = torch.cuda.get_device_name(0)
        print(f"✅ CUDA GPU Available: {status['device_name']}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CUDA GPU: Not available")
    
    # Check MPS (Metal for Mac)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        status['mps_available'] = True
        if not status['cuda_available']:  # Prefer CUDA over MPS
            status['device'] = torch.device('mps')
            status['device_name'] = "Apple Silicon GPU (MPS)"
        print(f"✅ MPS (Metal) GPU Available: Apple Silicon")
    else:
        print("❌ MPS (Metal) GPU: Not available")
    
    # Fallback to CPU
    if status['device'] is None:
        status['device'] = torch.device('cpu')
        status['device_name'] = "CPU"
        print("⚠️  No GPU available, using CPU")
    
    print(f"\n🎯 Active Device: {status['device']} ({status['device_name']})")
    
    # Test GPU computation
    print("\n" + "=" * 70)
    print("GPU COMPUTATION TEST")
    print("=" * 70)
    try:
        test_size = 2000
        if status['cuda_available']:
            test_tensor = torch.randn(test_size, test_size).to('cuda')
            result = torch.matmul(test_tensor, test_tensor)
            print(f"✅ CUDA GPU computation successful!")
            print(f"   Test: {test_size}x{test_size} matrix multiplication")
        elif status['mps_available']:
            test_tensor = torch.randn(test_size, test_size).to('mps')
            result = torch.matmul(test_tensor, test_tensor)
            print(f"✅ MPS GPU computation successful!")
            print(f"   Test: {test_size}x{test_size} matrix multiplication")
        else:
            test_tensor = torch.randn(test_size, test_size)
            result = torch.matmul(test_tensor, test_tensor)
            print(f"⚠️  CPU computation (no GPU acceleration)")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        status['device'] = torch.device('cpu')
        status['device_name'] = "CPU (fallback)"
    
    print("\n" + "=" * 70)
    return status

def check_dqn_configuration():
    """Check if DQN is enabled"""
    print("\n" + "=" * 70)
    print("DQN CONFIGURATION")
    print("=" * 70)
    
    use_dqn = os.environ.get("NDN_SIM_USE_DQN", "0")
    cache_policy = os.environ.get("NDN_SIM_CACHE_POLICY", "combined")
    
    print(f"NDN_SIM_USE_DQN: {use_dqn}")
    if use_dqn == "1":
        print("✅ DQN is ENABLED - GPU will be used for training")
    else:
        print("⚠️  DQN is DISABLED - GPU will NOT be used")
        print("💡 To enable GPU: export NDN_SIM_USE_DQN=1")
    
    print(f"Cache Policy: {cache_policy}")
    print("=" * 70)
    
    return use_dqn == "1"

def recommend_cloud_gpu():
    """Recommend cloud GPU options"""
    print("\n" + "=" * 70)
    print("CLOUD GPU RECOMMENDATIONS")
    print("=" * 70)
    
    gpu_status = check_gpu_status()
    dqn_enabled = check_dqn_configuration()
    
    if gpu_status['cuda_available']:
        print("\n✅ You have CUDA GPU - Best performance!")
        print("   Speedup: 3-5x faster than CPU")
        print("   No cloud GPU needed")
        return
    
    if gpu_status['mps_available']:
        print("\n✅ You have MPS GPU (Mac) - Good performance!")
        print("   Speedup: 2-3x faster than CPU")
        print("\n💡 For even faster results, consider cloud GPU:")
        print("\n   1. Google Colab (FREE):")
        print("      - Go to: https://colab.research.google.com")
        print("      - Upload your code")
        print("      - Enable GPU: Runtime → Change runtime type → GPU")
        print("      - Speedup: 3-5x (Tesla T4)")
        print("      - Cost: FREE")
        print("\n   2. AWS EC2 (PAID):")
        print("      - Instance: g4dn.xlarge (NVIDIA T4)")
        print("      - Speedup: 3-5x")
        print("      - Cost: ~$0.50/hour")
        print("\n   3. Google Cloud Platform (PAID):")
        print("      - Instance: n1-standard-4 with NVIDIA T4")
        print("      - Speedup: 3-5x")
        print("      - Cost: ~$0.35/hour")
    else:
        print("\n⚠️  No GPU available - Cloud GPU highly recommended!")
        print("\n   Best Options:")
        print("   1. Google Colab (FREE) - Best for testing")
        print("   2. AWS EC2 - Best for production")
        print("   3. Google Cloud Platform - Best value")
    
    if not dqn_enabled:
        print("\n⚠️  IMPORTANT: Enable DQN to use GPU:")
        print("   export NDN_SIM_USE_DQN=1")
    
    print("=" * 70)

if __name__ == "__main__":
    print("\n🚀 NDN Simulation - Cloud GPU Setup\n")
    
    gpu_status = check_gpu_status()
    dqn_enabled = check_dqn_configuration()
    
    if not dqn_enabled:
        print("\n⚠️  WARNING: DQN is disabled. GPU will not be used!")
        print("   Enable with: export NDN_SIM_USE_DQN=1\n")
    
    recommend_cloud_gpu()
    
    print("\n✅ Setup check complete!\n")

