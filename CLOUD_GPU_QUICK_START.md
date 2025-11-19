# Cloud GPU Quick Start Guide

## ✅ Current Status

**Your Mac**: MPS GPU available and working
- **GPU Status**: ✅ Working
- **DQN Status**: ⚠️ Disabled (GPU not being used)
- **To Enable**: `export NDN_SIM_USE_DQN=1`

---

## 🚀 Quick Setup Options

### Option 1: Enable GPU on Your Mac (Fastest Setup)

```bash
# Enable DQN to use your MPS GPU
export NDN_SIM_USE_DQN=1

# Run simulation (GPU will be used automatically)
python main.py
```

**Speedup**: 2-3x faster than CPU
**Cost**: Free
**Time**: ~10-15 minutes per simulation

---

### Option 2: Google Colab (Free Cloud GPU) 🆓

**Best For**: Quick tests, occasional runs

#### Steps:

1. **Go to Google Colab**: https://colab.research.google.com

2. **Upload the notebook**: 
   - File → Upload → Select `colab_setup.ipynb`
   - Or create new notebook and copy code

3. **Enable GPU**:
   - Runtime → Change runtime type → GPU (Tesla T4)

4. **Upload your code files**:
   - Upload `main.py`, `router.py`, `utils.py`, etc.
   - Or clone from GitHub if your code is in a repo

5. **Run simulation**:
   ```python
   !python main.py
   ```

**Speedup**: 3-5x faster than CPU
**Cost**: FREE
**GPU**: Tesla T4 (15GB)
**Time Limit**: ~12 hours per session

---

### Option 3: AWS EC2 GPU Instance 💰

**Best For**: Many simulations, production runs

#### Steps:

1. **Launch EC2 Instance**:
   - Instance type: `g4dn.xlarge` (NVIDIA T4)
   - AMI: Amazon Linux 2 or Ubuntu 22.04
   - Storage: 20GB minimum

2. **Connect via SSH**:
   ```bash
   ssh -i your-key.pem ec2-user@your-instance-ip
   ```

3. **Run Setup Script**:
   ```bash
   chmod +x aws_gpu_setup.sh
   ./aws_gpu_setup.sh
   ```

4. **Upload Your Code**:
   ```bash
   scp -r *.py ec2-user@your-instance-ip:~/
   ```

5. **Run Simulation**:
   ```bash
   export NDN_SIM_USE_DQN=1
   python3.9 main.py
   ```

**Speedup**: 3-5x faster than CPU
**Cost**: ~$0.50/hour
**GPU**: NVIDIA T4 (16GB)
**Best For**: Long-running simulations

---

### Option 4: Google Cloud Platform 💰

**Best For**: Best value, flexible pricing

#### Steps:

1. **Create VM Instance**:
   - Machine type: `n1-standard-4`
   - Add GPU: NVIDIA T4
   - Image: Ubuntu 22.04

2. **Install Dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip3 install networkx numpy matplotlib scipy
   ```

3. **Run Simulation**:
   ```bash
   export NDN_SIM_USE_DQN=1
   python3 main.py
   ```

**Speedup**: 3-5x faster than CPU
**Cost**: ~$0.35/hour
**GPU**: NVIDIA T4

---

## 📊 Performance Comparison

| Option | Speedup | Cost | Setup Time | Best For |
|--------|---------|------|------------|----------|
| **Mac MPS GPU** | 2-3x | Free | 0 min | Daily use |
| **Google Colab** | 3-5x | Free | 5 min | Testing |
| **AWS EC2** | 3-5x | $0.50/hr | 15 min | Production |
| **Google Cloud** | 3-5x | $0.35/hr | 15 min | Best value |

---

## 🔧 Enable GPU on Your Mac (Right Now)

```bash
# Check GPU status
python3 cloud_gpu_setup.py

# Enable DQN
export NDN_SIM_USE_DQN=1

# Run with GPU
python main.py
```

**The GPU will be used automatically!**

---

## 📝 Important Notes

1. **GPU Only Helps with DQN**: 
   - Without `NDN_SIM_USE_DQN=1`, GPU is not used
   - GPU accelerates neural network training only

2. **Cloud GPU Setup**:
   - Google Colab: Easiest, free, best for testing
   - AWS/GCP: Best for production, paid but cheap

3. **Code Compatibility**:
   - Code automatically detects and uses GPU
   - No code changes needed for cloud GPU

4. **Time Savings**:
   - Mac MPS: ~5 minutes saved per simulation
   - Cloud GPU: ~7-10 minutes saved per simulation

---

## 🎯 Recommendation

**For Now**: Enable GPU on your Mac
```bash
export NDN_SIM_USE_DQN=1
python main.py
```

**For Faster Results**: Use Google Colab (free)
- Upload `colab_setup.ipynb`
- Enable GPU runtime
- Run simulation

**For Many Runs**: Use AWS EC2 or GCP
- Better value for long-running simulations
- More reliable than free services

---

## ✅ Quick Check

Run this to verify everything:
```bash
python3 cloud_gpu_setup.py
```

This will show:
- ✅ GPU status
- ✅ DQN configuration
- ✅ Cloud GPU recommendations

