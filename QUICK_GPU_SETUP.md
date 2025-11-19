# Quick GPU Setup Guide

## Current Status ✅

**Your System**: MPS (Metal) GPU available and working
- PyTorch will automatically use it for DQN training
- No additional setup needed!

---

## External GPU Options

### Option 1: Use Current MPS GPU (Recommended) ✅
**Status**: Already configured and working
**Speedup**: 2-3x faster for DQN training
**Cost**: Free
**Action**: None needed - just run simulation with `NDN_SIM_USE_DQN=1`

### Option 2: Google Colab (Free GPU) 🆓
**Best For**: Quick tests, occasional runs
**GPU**: Tesla T4 (free)
**Speedup**: 3-5x faster
**Setup Time**: ~5 minutes

**Steps**:
1. Go to https://colab.research.google.com
2. Upload your code files
3. Enable GPU: Runtime → Change runtime type → GPU
4. Install dependencies: `!pip install networkx numpy torch`
5. Run: `!python main.py`

### Option 3: AWS EC2 GPU Instance 💰
**Best For**: Many simulations, production runs
**GPU**: NVIDIA T4
**Speedup**: 3-5x faster
**Cost**: ~$0.50/hour
**Setup Time**: ~15 minutes

**Steps**:
1. Launch EC2 g4dn.xlarge instance
2. Install dependencies
3. Upload code
4. Run simulation

### Option 4: External eGPU for Mac ❌
**Not Recommended**: Minimal improvement over MPS, expensive ($200-500)
**Speedup**: Similar to MPS (2-3x)
**Better Alternative**: Use cloud GPU

---

## Time Comparison

| Setup | DQN Simulation Time | Cost |
|-------|---------------------|------|
| CPU Only | ~15-20 min | Free |
| **MPS GPU (Current)** | **~10-15 min** | **Free** ✅ |
| CUDA GPU | ~8-12 min | Varies |
| Cloud GPU | ~8-12 min | $0-1/hour |

**Recommendation**: Your current MPS GPU is already optimal for Mac!

---

## Quick Commands

### Check GPU Status:
```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False); print('CUDA:', torch.cuda.is_available())"
```

### Run with Current GPU (Automatic):
```bash
export NDN_SIM_USE_DQN=1
python main.py
```

### Use Google Colab (Free):
1. Upload code to Colab
2. Enable GPU runtime
3. Run simulation

---

## Bottom Line

**For your Mac**: ✅ Current MPS GPU is already optimal
**For faster results**: Use Google Colab (free GPU)
**For many runs**: Consider cloud GPU service

**No external GPU needed for Mac - MPS is already good!**

