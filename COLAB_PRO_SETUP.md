# Google Colab Student Pro Setup Guide

## 🎯 What You Get with Colab Student Pro

**Cost**: Discounted student pricing (varies by region, typically $5-10/month)

**Benefits**:

- ✅ **V100 GPU** (16GB) - 5-8x faster than MPS
- ✅ **A100 GPU** (40GB) - Sometimes available, 10-15x faster
- ✅ **More compute units** - Longer sessions than free tier
- ✅ **Priority access** - Less waiting for GPU
- ✅ **Faster GPUs** - V100/A100 vs T4 (free tier)
- ✅ **Background execution** - Keep running when tab closed
- ✅ **Student discount** - Lower cost than regular Pro

**Speedup**: **5-8x faster** (V100) or **10-15x faster** (A100) than your current MPS GPU

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Sign Up for Colab Pro

1. Go to: https://colab.research.google.com/signup
2. Click "Upgrade to Colab Pro"
3. Pay $10/month (can cancel anytime)
4. Wait for activation (usually instant)

### Step 2: Create New Notebook

1. Go to: https://colab.research.google.com
2. Click "New notebook"
3. Name it: "NDN_Benchmark"

### Step 3: Enable GPU

1. Click: **Runtime** → **Change runtime type**
2. Set:
   - **Hardware accelerator**: GPU
   - **GPU type**: V100 (or A100 if available)
3. Click **Save**

### Step 4: Upload Your Code

**Option A: Upload Files Directly**

```python
# Run this in a cell to upload files
from google.colab import files
uploaded = files.upload()
```

**Option B: Clone from GitHub** (if you have a repo)

```python
# Run this in a cell
!git clone https://github.com/yourusername/ndn-nwsim-Sai_work.git
%cd ndn-nwsim-Sai_work
```

**Option C: Mount Google Drive** (best for large files)

```python
# Run this in a cell
from google.colab import drive
drive.mount('/content/drive')
# Then copy files from Drive to working directory
!cp -r /content/drive/MyDrive/ndn-nwsim-Sai_work/* .
```

### Step 5: Install Dependencies

```python
# Run this in a cell
!pip install networkx numpy torch scipy matplotlib pandas scikit-learn dill bitarray hdbscan tensorflow
```

### Step 6: Run Your Benchmark

```python
# Run this in a cell
import os
import sys

# Set environment variables
os.environ['NDN_SIM_NODES'] = '50'
os.environ['NDN_SIM_PRODUCERS'] = '10'
os.environ['NDN_SIM_CONTENTS'] = '200'
os.environ['NDN_SIM_USERS'] = '100'
os.environ['NDN_SIM_ROUNDS'] = '50'
os.environ['NDN_SIM_REQUESTS'] = '20'
os.environ['NDN_SIM_CACHE_CAPACITY'] = '1000'
os.environ['NDN_SIM_ZIPF_PARAM'] = '1.2'
os.environ['NDN_SIM_USE_DQN'] = '1'  # Enable DQN to use GPU

# Run benchmark
!python benchmark.py
```

---

## 📋 Complete Colab Notebook Template

Copy this into your Colab notebook:

```python
# Cell 1: Install dependencies
!pip install networkx numpy torch scipy matplotlib pandas scikit-learn dill bitarray hdbscan tensorflow

# Cell 2: Upload code files (or mount Drive)
from google.colab import files
# Upload: benchmark.py, main.py, utils.py, router.py, endpoints.py, dqn_agent.py, etc.
uploaded = files.upload()

# Cell 3: Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 4: Set configuration
import os
os.environ['NDN_SIM_NODES'] = '50'
os.environ['NDN_SIM_PRODUCERS'] = '10'
os.environ['NDN_SIM_CONTENTS'] = '200'
os.environ['NDN_SIM_USERS'] = '100'
os.environ['NDN_SIM_ROUNDS'] = '50'
os.environ['NDN_SIM_REQUESTS'] = '20'
os.environ['NDN_SIM_CACHE_CAPACITY'] = '1000'
os.environ['NDN_SIM_ZIPF_PARAM'] = '1.2'
os.environ['NDN_SIM_USE_DQN'] = '1'  # Enable DQN for GPU acceleration

# Cell 5: Run benchmark
!python benchmark.py

# Cell 6: Download results
from google.colab import files
files.download('benchmark_checkpoints/benchmark_results.json')
```

---

## 🎯 Optimized Colab Pro Configuration

For fastest results on Colab Pro:

```python
# Fast config for Colab Pro (target: 10-15 minutes total)
os.environ['NDN_SIM_NODES'] = '30'
os.environ['NDN_SIM_PRODUCERS'] = '6'
os.environ['NDN_SIM_CONTENTS'] = '150'
os.environ['NDN_SIM_USERS'] = '50'
os.environ['NDN_SIM_ROUNDS'] = '20'  # Reduced for speed
os.environ['NDN_SIM_REQUESTS'] = '15'
os.environ['NDN_SIM_CACHE_CAPACITY'] = '1000'
os.environ['NDN_SIM_ZIPF_PARAM'] = '1.2'
os.environ['NDN_SIM_USE_DQN'] = '1'  # GPU will accelerate this
```

---

## ⚡ Performance Expectations

**With Colab Pro V100 GPU**:

| Component           | Speedup         | Time Saved        |
| ------------------- | --------------- | ----------------- |
| DQN Training        | **5-8x**        | ~40-50 minutes    |
| Network Simulation  | 1x (CPU-bound)  | 0 minutes         |
| **Total Benchmark** | **3-4x faster** | **20-30 minutes** |

**Estimated Times**:

- **Current (MPS)**: 60-90 minutes
- **Colab Pro (V100)**: **20-30 minutes** ✅
- **With fast config**: **10-15 minutes** ✅

---

## 💡 Tips for Colab Pro

### 1. Keep Session Alive

```python
# Run this to prevent timeout
import time
while True:
    time.sleep(60)  # Keep alive
```

### 2. Monitor GPU Usage

```python
# Check GPU memory
!nvidia-smi
```

### 3. Save Progress

```python
# Save checkpoints to Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r benchmark_checkpoints /content/drive/MyDrive/
```

### 4. Download Results

```python
# Download results when done
from google.colab import files
files.download('benchmark_checkpoints/benchmark_results.json')
files.download('benchmark_run.log')
```

---

## 🚨 Important Notes

1. **Session Limits**: Colab Pro has longer sessions but still has limits
2. **GPU Availability**: V100 may not always be available (fallback to T4)
3. **Background Execution**: Pro allows background execution (keep running when tab closed)
4. **Data Persistence**: Files are deleted when session ends - save to Drive!

---

## 📊 Cost Comparison

| Option            | Cost     | Speedup | Time for Benchmark |
| ----------------- | -------- | ------- | ------------------ |
| **MPS (Current)** | Free     | 1x      | 60-90 min          |
| **Colab Free**    | Free     | 3-5x    | 20-30 min          |
| **Colab Pro**     | $10/mo   | 5-8x    | **15-20 min** ✅   |
| **Lambda Labs**   | $1.10/hr | 10-15x  | 5-10 min           |

**Colab Pro is best if**:

- You run benchmarks regularly (monthly cost < hourly cloud costs)
- You want convenience (no setup, just run)
- You need reliable access

---

## 🎯 Next Steps

1. **Sign up**: https://colab.research.google.com/signup
2. **Create notebook**: New notebook → Name it
3. **Enable GPU**: Runtime → Change runtime type → GPU (V100)
4. **Upload code**: Use one of the methods above
5. **Run benchmark**: Execute the cells

**Total setup time**: 5-10 minutes

---

## ✅ Verification

After setup, verify GPU is working:

```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A")
```

Should show: **V100** or **A100** with 16GB+ memory

---

**Ready to go!** Your benchmark should run **5-8x faster** on Colab Pro V100 GPU! 🚀
