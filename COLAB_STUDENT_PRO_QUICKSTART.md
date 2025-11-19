# Colab Student Pro Quick Start Guide

## ✅ You Have Colab Student Pro!

**Great choice!** You'll get V100 or A100 GPUs for your NDN benchmark.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Open Colab and Enable GPU

1. Go to: https://colab.research.google.com
2. Click **"New notebook"**
3. Name it: **"NDN_Benchmark"**
4. Click: **Runtime** → **Change runtime type**
5. Set:
   - **Hardware accelerator**: **GPU**
   - **GPU type**: **V100** or **A100** (if available)
6. Click **Save**

### Step 2: Install Dependencies

Create a new cell and run:

```python
!pip install networkx numpy torch scipy matplotlib pandas scikit-learn dill bitarray hdbscan tensorflow
```

### Step 3: Upload and Run

**Option A: Upload Files Directly** (Quick)
```python
from google.colab import files
uploaded = files.upload()  # Upload: benchmark.py, main.py, utils.py, router.py, endpoints.py, dqn_agent.py, etc.
```

**Option B: Mount Google Drive** (Recommended - files persist)
```python
from google.colab import drive
drive.mount('/content/drive')
# Copy your project folder to Drive first, then:
!cp -r /content/drive/MyDrive/ndn-nwsim-Sai_work/* .
```

Then run your benchmark:

```python
import os

# Set optimized configuration
os.environ['NDN_SIM_NODES'] = '50'
os.environ['NDN_SIM_PRODUCERS'] = '10'
os.environ['NDN_SIM_CONTENTS'] = '200'
os.environ['NDN_SIM_USERS'] = '100'
os.environ['NDN_SIM_ROUNDS'] = '50'
os.environ['NDN_SIM_REQUESTS'] = '20'
os.environ['NDN_SIM_CACHE_CAPACITY'] = '1000'
os.environ['NDN_SIM_ZIPF_PARAM'] = '1.2'
os.environ['NDN_SIM_USE_DQN'] = '1'  # Enable DQN to use GPU!

# Run benchmark
!python benchmark.py
```

---

## 📊 Verify GPU is Working

Run this to check:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("✅ GPU is ready!")
else:
    print("❌ No GPU - check Runtime → Change runtime type → GPU")
```

**Expected output**: V100 (16GB) or A100 (40GB)

---

## ⚡ Performance Expectations

**With Colab Student Pro**:

| GPU Type | Speedup vs MPS | Benchmark Time |
|----------|----------------|----------------|
| **V100** | 5-8x | 15-20 minutes |
| **A100** | 10-15x | 10-15 minutes |

**Your current MPS**: 60-90 minutes  
**With Student Pro**: **10-20 minutes** ✅

---

## 💡 Pro Tips

### 1. Keep Session Alive
```python
# Run in background to prevent timeout
import time
import threading

def keep_alive():
    while True:
        time.sleep(300)  # Every 5 minutes
        print("Still running...")

threading.Thread(target=keep_alive, daemon=True).start()
```

### 2. Monitor Progress
```python
# Check GPU usage
!nvidia-smi

# View benchmark log
!tail -50 benchmark_run.log
```

### 3. Save Results to Drive
```python
# Mount Drive first
from google.colab import drive
drive.mount('/content/drive')

# Save results
!cp -r benchmark_checkpoints /content/drive/MyDrive/ndn_results/
print("Results saved to Drive!")
```

### 4. Download Results
```python
from google.colab import files
files.download('benchmark_checkpoints/benchmark_results.json')
```

---

## 🎯 Complete Notebook Template

Copy this into your Colab notebook:

```python
# Cell 1: Install
!pip install networkx numpy torch scipy matplotlib pandas scikit-learn dill bitarray hdbscan tensorflow

# Cell 2: Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 3: Upload files
from google.colab import files
uploaded = files.upload()

# Cell 4: Configure
import os
os.environ['NDN_SIM_NODES'] = '50'
os.environ['NDN_SIM_PRODUCERS'] = '10'
os.environ['NDN_SIM_CONTENTS'] = '200'
os.environ['NDN_SIM_USERS'] = '100'
os.environ['NDN_SIM_ROUNDS'] = '50'
os.environ['NDN_SIM_REQUESTS'] = '20'
os.environ['NDN_SIM_CACHE_CAPACITY'] = '1000'
os.environ['NDN_SIM_ZIPF_PARAM'] = '1.2'
os.environ['NDN_SIM_USE_DQN'] = '1'

# Cell 5: Run
!python benchmark.py

# Cell 6: View results
import json
with open('benchmark_checkpoints/benchmark_results.json', 'r') as f:
    results = json.load(f)
for name, result in results.items():
    print(f"{name}: {result.get('hit_rate', 0):.2f}%")

# Cell 7: Download
from google.colab import files
files.download('benchmark_checkpoints/benchmark_results.json')
```

---

## 🚨 Important Notes

1. **GPU Availability**: V100 is usually available, A100 may require waiting
2. **Session Limits**: Student Pro has longer sessions but still has limits
3. **Save to Drive**: Files are deleted when session ends - save important results!
4. **Background Execution**: Student Pro allows background execution (runs when tab closed)

---

## ✅ You're Ready!

1. Open Colab: https://colab.research.google.com
2. Create new notebook
3. Enable GPU (V100 or A100)
4. Copy the template above
5. Run your benchmark!

**Expected time**: 10-20 minutes (vs 60-90 minutes on MPS) 🚀

---

## 🆘 Troubleshooting

**No GPU available?**
- Wait a few minutes and try again
- Check Runtime → Change runtime type → GPU is selected
- Student Pro should have priority access

**Session timeout?**
- Use background execution
- Save checkpoints to Drive
- Resume from checkpoint when restarting

**Slow upload?**
- Use Google Drive mount instead of direct upload
- Upload files to Drive first, then mount

---

**Happy benchmarking!** Your NDN simulation will run **5-15x faster** on Colab Student Pro! 🎉

