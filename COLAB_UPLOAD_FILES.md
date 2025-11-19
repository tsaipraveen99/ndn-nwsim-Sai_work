# Files to Upload to Google Colab

## ✅ Essential Files (Required)

Upload these files to run the benchmark:

### Core Simulation Files:
1. **`main.py`** - Network creation and simulation orchestration
2. **`router.py`** - NDN router logic (PIT, FIB, ContentStore, packet handling)
3. **`endpoints.py`** - User and Producer classes
4. **`utils.py`** - ContentStore, Bloom filters, DQN state representation
5. **`dqn_agent.py`** - DQN neural network and training
6. **`packet.py`** - Interest and Data packet classes
7. **`metrics.py`** - Metrics collection

### Benchmark Files:
8. **`benchmark.py`** - Main benchmark script
9. **`statistical_analysis.py`** - Statistical calculations (mean, std, CI, tests)

---

## 📋 Complete Upload List

### Quick Copy-Paste List:
```
main.py
router.py
endpoints.py
utils.py
dqn_agent.py
packet.py
metrics.py
benchmark.py
statistical_analysis.py
```

---

## 🚀 Upload Methods

### Method 1: Upload All at Once (Easiest)

```python
# Run this in a Colab cell
from google.colab import files

# Upload all files
uploaded = files.upload()

# Verify files uploaded
import os
print("Uploaded files:")
for filename in uploaded.keys():
    print(f"  ✅ {filename}")
```

### Method 2: Upload Specific Files

```python
from google.colab import files

# Upload files one by one or select multiple
files.upload()  # Select all 9 files from the dialog
```

### Method 3: Use Google Drive (Recommended for Large Projects)

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy from Drive (upload files to Drive first)
!cp -r /content/drive/MyDrive/ndn-nwsim-Sai_work/* .
```

---

## ✅ Verification After Upload

Run this to verify all files are present:

```python
import os

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

print("Checking required files...")
missing = []
for file in required_files:
    if os.path.exists(file):
        print(f"  ✅ {file}")
    else:
        print(f"  ❌ {file} - MISSING")
        missing.append(file)

if missing:
    print(f"\n⚠️  Missing {len(missing)} file(s). Please upload them.")
else:
    print("\n✅ All required files present! Ready to run benchmark.")
```

---

## 📝 Optional Files (Not Required, but Helpful)

These are optional - the benchmark will work without them:

- `ablation_study.py` - For ablation studies (optional)
- `sensitivity_analysis.py` - For sensitivity analysis (optional)
- `baselines.py` - For baseline comparisons (optional)
- `compare_results.py` - For result comparison (optional)
- `requirements.txt` - For reference (we'll pip install directly)

---

## 🎯 Quick Start After Upload

Once files are uploaded:

```python
# 1. Install dependencies
!pip install networkx numpy torch scipy matplotlib pandas scikit-learn dill bitarray hdbscan tensorflow

# 2. Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 3. Set configuration
import os
os.environ['NDN_SIM_NODES'] = '50'
os.environ['NDN_SIM_PRODUCERS'] = '10'
os.environ['NDN_SIM_CONTENTS'] = '200'
os.environ['NDN_SIM_USERS'] = '100'
os.environ['NDN_SIM_ROUNDS'] = '50'
os.environ['NDN_SIM_REQUESTS'] = '20'
os.environ['NDN_SIM_CACHE_CAPACITY'] = '1000'
os.environ['NDN_SIM_ZIPF_PARAM'] = '1.2'
os.environ['NDN_SIM_USE_DQN'] = '1'  # Enable DQN to use GPU!

# 4. Run benchmark
!python benchmark.py
```

---

## 💡 Tips

1. **Upload all 9 files at once** - Use the file upload dialog to select multiple files
2. **Check file sizes** - Should be relatively small (few KB to few MB each)
3. **Verify after upload** - Run the verification script above
4. **Save to Drive** - Consider mounting Drive for persistence

---

## ❓ Troubleshooting

**"Module not found" error?**
- Make sure all 9 essential files are uploaded
- Check file names match exactly (case-sensitive)

**"Import error"?**
- Verify files are in the same directory
- Run `!ls` to see uploaded files

**"File not found"?**
- Check you're in the right directory: `!pwd`
- List files: `!ls -la`

---

**That's it!** Upload these 9 files and you're ready to run your benchmark on Colab! 🚀

