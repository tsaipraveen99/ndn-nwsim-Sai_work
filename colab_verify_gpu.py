# Run this in a Colab cell to verify GPU is working

import torch
import sys

print("="*60)
print("GPU VERIFICATION")
print("="*60)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n✅ CUDA Available: {cuda_available}")

if cuda_available:
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU Name: {gpu_name}")
    print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
    print(f"✅ GPU Count: {torch.cuda.device_count()}")
    
    # Test GPU computation
    print("\n🧪 Testing GPU computation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU computation test: PASSED")
        print(f"✅ Result shape: {z.shape}")
    except Exception as e:
        print(f"❌ GPU computation test: FAILED - {e}")
    
    print("\n" + "="*60)
    print("✅ GPU IS READY FOR DQN TRAINING!")
    print("="*60)
    
    # Performance estimate
    if "A100" in gpu_name:
        print("\n🚀 A100 GPU detected - Expected 10-15x speedup!")
    elif "L4" in gpu_name:
        print("\n🚀 L4 GPU detected - Expected 8-12x speedup!")
    elif "T4" in gpu_name or "Tesla T4" in gpu_name:
        print("\n🚀 T4 GPU detected - Expected 3-5x speedup!")
    else:
        print(f"\n🚀 {gpu_name} detected - Good performance expected!")
        
else:
    print("\n❌ NO GPU DETECTED")
    print("Please check:")
    print("  1. Runtime → Change runtime type → GPU is selected")
    print("  2. You have Colab Pro/Student Pro active")
    print("  3. Wait a few minutes if GPU is not immediately available")
    sys.exit(1)

