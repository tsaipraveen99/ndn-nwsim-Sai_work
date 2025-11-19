# GPU Acceleration Guide for NDN Simulation

## Current GPU Status

**Your System**: MPS (Metal) GPU available ✅
- **Type**: Apple Silicon integrated GPU
- **Status**: Automatically used for DQN training
- **Speedup**: 2-3x faster than CPU for neural network operations

---

## What Can Be Accelerated with GPU?

### ✅ GPU-Accelerated Components:

1. **DQN Neural Network Training** (Primary Benefit)
   - Forward passes (action selection)
   - Backward passes (gradient computation)
   - Experience replay batch processing
   - **Impact**: 2-3x speedup for DQN training
   - **Time Saved**: ~5 minutes per simulation

2. **Semantic Encoder (CNN)** (If Enabled)
   - Convolutional operations
   - Embedding generation
   - **Impact**: 3-5x speedup for encoding
   - **Time Saved**: Minimal (encoding is fast)

3. **Neural Bloom Filter** (If Enabled)
   - Neural network inference
   - False positive prediction
   - **Impact**: 2-3x speedup
   - **Time Saved**: Minimal (optional feature)

### ❌ CPU-Bound Components (GPU Won't Help):

- Network simulation (threading-based)
- Packet routing (CPU-bound)
- Cache operations (CPU-bound)
- Metrics collection (CPU-bound)
- Graph operations (NetworkX)

**Key Insight**: GPU only helps when DQN is enabled. Without DQN, GPU provides no benefit.

---

## External GPU Options

### Option 1: External GPU (eGPU) for Mac

#### Requirements:
- Mac with Thunderbolt 3/4 port
- External GPU enclosure (e.g., Razer Core X, Sonnet eGFX)
- AMD GPU (NVIDIA not well supported on Mac)
- macOS 10.13.6+ with eGPU support

#### Setup:
1. Connect eGPU via Thunderbolt
2. Install AMD GPU drivers
3. PyTorch will automatically detect and use it

#### Limitations:
- **macOS**: Only AMD GPUs supported (not NVIDIA)
- **Performance**: May not be much faster than MPS (Metal is optimized)
- **Cost**: $200-500 for enclosure + GPU
- **Compatibility**: Not all Mac models support eGPU

#### Recommendation:
- **Not Recommended** for Mac: MPS GPU is already well-optimized
- **Better Alternative**: Use a Linux/Windows machine with NVIDIA GPU

---

### Option 2: NVIDIA CUDA GPU (Linux/Windows)

#### Requirements:
- Linux or Windows machine
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- PyTorch with CUDA support

#### Performance:
- **Speedup**: 3-5x faster than CPU for DQN training
- **Better than MPS**: CUDA typically faster than Metal
- **Time Saved**: ~7-10 minutes per DQN simulation

#### Setup:
```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Recommendation:
- **Best Option**: If you have access to a Linux/Windows machine with NVIDIA GPU
- **Cloud Options**: AWS EC2 (g4dn instances), Google Colab (free GPU), Azure

---

### Option 3: Cloud GPU Services

#### Google Colab (Free):
- **Free GPU**: Tesla T4 (15GB)
- **Time Limit**: ~12 hours per session
- **Setup**: Upload code, enable GPU runtime
- **Cost**: Free (with limitations)

#### AWS EC2:
- **Instance**: g4dn.xlarge (NVIDIA T4)
- **Cost**: ~$0.50/hour
- **Performance**: Excellent
- **Setup**: Launch instance, install dependencies

#### Google Cloud Platform:
- **Instance**: n1-standard-4 with NVIDIA T4
- **Cost**: ~$0.35/hour
- **Performance**: Excellent

#### Recommendation:
- **For Quick Tests**: Google Colab (free)
- **For Production**: AWS EC2 or GCP

---

## Performance Comparison

| GPU Type | DQN Training Speedup | Total Time (DQN) | Cost |
|----------|---------------------|------------------|------|
| **CPU Only** | 1x (baseline) | ~15-20 min | Free |
| **MPS (Current)** | 2-3x | ~10-15 min | Free |
| **eGPU (AMD)** | 2-3x | ~10-15 min | $200-500 |
| **CUDA (NVIDIA)** | 3-5x | ~8-12 min | Varies |
| **Cloud GPU** | 3-5x | ~8-12 min | $0-1/hour |

**Conclusion**: 
- Current MPS GPU is already good (2-3x speedup)
- External GPU for Mac: Not worth it (minimal improvement)
- CUDA GPU: Best option if available (3-5x speedup)
- Cloud GPU: Best for occasional use (free or cheap)

---

## How to Use External/Cloud GPU

### For CUDA GPU (Linux/Windows):

1. **Install PyTorch with CUDA**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

2. **Verify GPU**:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

3. **Run Simulation** (GPU will be used automatically):
```bash
export NDN_SIM_USE_DQN=1
python main.py
```

### For Google Colab:

1. **Upload Code**:
   - Upload `main.py`, `router.py`, `utils.py`, etc. to Colab
   - Or clone from GitHub

2. **Enable GPU**:
   - Runtime → Change runtime type → GPU

3. **Install Dependencies**:
```python
!pip install networkx numpy torch
```

4. **Run Simulation**:
```python
!python main.py
```

---

## Practical Recommendations

### Scenario 1: You Have Access to NVIDIA GPU (Linux/Windows)
**Recommendation**: ✅ **Use it!**
- **Speedup**: 3-5x for DQN training
- **Time Saved**: ~7-10 minutes per simulation
- **Setup**: Install PyTorch with CUDA, run simulation

### Scenario 2: You Only Have Mac
**Recommendation**: ✅ **Stick with MPS GPU**
- **Current**: Already using GPU (MPS)
- **External eGPU**: Not worth cost (minimal improvement)
- **Alternative**: Use cloud GPU for occasional runs

### Scenario 3: You Need Faster Results
**Recommendation**: ✅ **Use Cloud GPU (Google Colab)**
- **Free**: Google Colab provides free GPU
- **Fast**: 3-5x speedup
- **Easy**: Just upload code and run

### Scenario 4: You Run Many Simulations
**Recommendation**: ✅ **Consider Cloud GPU Service**
- **AWS EC2**: ~$0.50/hour, excellent performance
- **GCP**: ~$0.35/hour, excellent performance
- **Cost-Effective**: For many runs, cloud is cheaper than buying GPU

---

## Code Changes for External GPU

The code **already supports external GPUs automatically**! PyTorch will:
1. Detect CUDA GPU if available
2. Use it automatically for DQN training
3. Fall back to CPU if GPU unavailable

**No code changes needed** - just ensure PyTorch with CUDA is installed.

---

## Time Savings Analysis

### Current Setup (MPS GPU):
- **Without DQN**: ~5-10 minutes (GPU not used)
- **With DQN**: ~10-15 minutes (GPU used, 2-3x speedup)

### With CUDA GPU:
- **Without DQN**: ~5-10 minutes (GPU not used)
- **With DQN**: ~8-12 minutes (GPU used, 3-5x speedup)
- **Time Saved**: ~2-3 minutes per simulation

### For 10 Simulations:
- **MPS GPU**: ~100-150 minutes
- **CUDA GPU**: ~80-120 minutes
- **Total Savings**: ~20-30 minutes

---

## Quick Setup Guide

### Option A: Use Current MPS GPU (Recommended for Mac)
```bash
# Already configured - just run
export NDN_SIM_USE_DQN=1
python main.py
```
**No additional setup needed!**

### Option B: Use CUDA GPU (If Available)
```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"

# Run (GPU used automatically)
export NDN_SIM_USE_DQN=1
python main.py
```

### Option C: Use Google Colab (Free GPU)
1. Go to https://colab.research.google.com
2. Upload your code files
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run: `!python main.py`

---

## Summary

### ✅ Best Options:

1. **Current MPS GPU** (Mac): Already good, 2-3x speedup
2. **CUDA GPU** (Linux/Windows): Best performance, 3-5x speedup
3. **Google Colab** (Free): Best for testing, 3-5x speedup
4. **Cloud GPU** (AWS/GCP): Best for production, 3-5x speedup

### ❌ Not Recommended:

- **External eGPU for Mac**: Minimal improvement over MPS, expensive
- **Buying GPU just for this**: Cloud GPU is cheaper for occasional use

### 💡 Recommendation:

**For your Mac**: Stick with MPS GPU (already optimal)
**For faster results**: Use Google Colab (free) or cloud GPU service
**For many simulations**: Consider cloud GPU (AWS/GCP)

**The code already supports all GPU types automatically!**

