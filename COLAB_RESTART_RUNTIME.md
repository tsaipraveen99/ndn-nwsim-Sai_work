# How to Restart Runtime in Google Colab

## 🔄 Method 1: Using Menu (Easiest)

### Steps:

1. Click **Runtime** in the top menu bar
2. Select **Restart runtime** (or **Restart session**)
3. Click **Yes** to confirm

**Keyboard Shortcut**: `Ctrl+M .` (Control + M, then period)

---

## 🔄 Method 2: Using Keyboard Shortcut

### Quick Restart:

- Press `Ctrl+M .` (Windows/Linux)
- Press `Cmd+M .` (Mac)

This opens the restart dialog - click **Yes** to confirm.

---

## 🔄 Method 3: Using Code Cell

### Option A: Restart and Clear Output

```python
# This will restart the runtime
import os
os._exit(0)  # Force exit (runtime will auto-restart)
```

### Option B: Restart via IPython

```python
# If using IPython/Jupyter
from IPython.display import Javascript
Javascript('IPython.notebook.kernel.restart()')
```

**Note**: These code methods may not work in all Colab versions. Use Method 1 (menu) for reliability.

---

## 🔄 Method 4: Factory Reset Runtime

If you need a complete reset:

1. Click **Runtime** → **Factory reset runtime**
2. This clears **everything**:
   - All variables
   - All installed packages
   - All uploaded files
   - All session state

**⚠️ Warning**: This deletes everything! Only use if you need a fresh start.

---

## 📋 When to Restart Runtime

### ✅ Good Reasons to Restart:

- After updating Python files (like `benchmark.py`)
- When you get import errors after file changes
- When variables are in inconsistent state
- When memory is full
- When packages aren't loading correctly
- Before starting a new benchmark run

### ❌ Don't Restart If:

- Benchmark is currently running (you'll lose progress)
- You have important variables in memory
- You're in the middle of a long computation

---

## 🔍 Check Runtime Status

### See Current Runtime Info:

```python
import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

# Check GPU (if using)
try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: Not available")
except:
    print("PyTorch not installed")
```

### Check Memory Usage:

```python
# Check RAM usage
import psutil
import os

process = psutil.Process(os.getpid())
ram_used = process.memory_info().rss / 1024 / 1024 / 1024  # GB
print(f"RAM used: {ram_used:.2f} GB")
```

---

## 🚀 Restart Workflow for Your Benchmark

### Before Restarting:

1. **Stop running benchmark** (if active)
2. **Save any important data**:
   ```python
   # Save results if needed
   import shutil
   shutil.copy('benchmark_checkpoints/benchmark_results.json',
               'results_backup.json')
   ```

### After Restarting:

1. **Re-upload files** (if needed):

   ```python
   from google.colab import files
   files.upload()  # Upload benchmark.py if updated
   ```

2. **Re-install dependencies** (if needed):

   ```python
   !pip install -q networkx numpy torch mmh3 dill
   ```

3. **Verify setup**:

   ```python
   # Check files exist
   import os
   required = ['benchmark.py', 'main.py', 'router.py', 'utils.py']
   for f in required:
       if os.path.exists(f):
           print(f"✅ {f}")
       else:
           print(f"❌ {f} - MISSING")
   ```

4. **Start benchmark**:
   ```python
   !python benchmark.py
   ```

---

## ⚡ Quick Restart Checklist

Before restarting:

- [ ] Stop any running processes
- [ ] Save important results/data
- [ ] Note which files you've uploaded
- [ ] Note which packages you've installed

After restarting:

- [ ] Re-upload updated files
- [ ] Re-install packages (if needed)
- [ ] Verify files exist
- [ ] Start your benchmark

---

## 🎯 Common Scenarios

### Scenario 1: Updated benchmark.py

```python
# 1. Stop benchmark (if running)
# 2. Upload new benchmark.py
from google.colab import files
files.upload()

# 3. Restart runtime: Runtime → Restart runtime
# 4. Re-run your benchmark
!python benchmark.py
```

### Scenario 2: Import Error After Update

```python
# 1. Restart runtime: Runtime → Restart runtime
# 2. Re-upload files
from google.colab import files
files.upload()

# 3. Verify imports work
import sys
sys.path.insert(0, '.')
from main import create_network, run_simulation
print("✅ Imports successful!")
```

### Scenario 3: Memory Full

```python
# 1. Restart runtime: Runtime → Restart runtime
# 2. This clears all variables and frees memory
# 3. Re-run your code
```

---

## 💡 Pro Tips

1. **Use checkpoints**: The benchmark saves checkpoints, so you can safely restart and resume
2. **Save to Drive**: Mount Google Drive to persist files across restarts
3. **Document packages**: Keep a list of packages you install
4. **Use requirements.txt**: Create a requirements file for easy re-installation

---

## 🆘 Troubleshooting

### Runtime Won't Restart?

- Try closing and reopening the Colab tab
- Use "Factory reset runtime" as last resort

### Lost Files After Restart?

- Files uploaded via `files.upload()` are lost on restart
- Use Google Drive to persist files:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  !cp benchmark.py /content/drive/MyDrive/
  ```

### Can't Reconnect to Runtime?

- Runtime may have timed out (idle for 90 minutes)
- Just re-run your cells to reconnect
- Or start a new runtime

---

**That's it!** Use **Runtime → Restart runtime** from the menu, or press `Ctrl+M .` for a quick restart! 🔄
