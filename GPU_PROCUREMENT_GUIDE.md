# GPU Procurement Guide - Where to Get Faster GPUs

## 🎯 Your Current Setup

- **Current GPU**: MPS (Metal) - Apple Silicon integrated GPU
- **Speedup**: 2-3x vs CPU
- **Cost**: Free (already have it)

## 🚀 Options for Faster GPUs

### Option 1: Google Colab (FREE) ⭐ **RECOMMENDED**

**Best For**: Quick tests, occasional runs, no cost

**GPU Options**:

- **Free Tier**: Tesla T4 (15GB) - **FREE**
- **Colab Pro**: V100 (16GB) - $10/month
- **Colab Pro+**: A100 (40GB) - $50/month

**Speedup**:

- T4: **3-5x faster** than MPS
- V100: **5-8x faster** than MPS
- A100: **10-15x faster** than MPS

**How to Get**:

1. Go to: https://colab.research.google.com
2. Sign in with Google account
3. Upload your code
4. Runtime → Change runtime type → GPU
5. Run your benchmark

**Limitations**:

- Free tier: ~12 hours/day usage limit
- Sessions timeout after inactivity
- Need to re-upload code each time

**Setup Time**: 5 minutes

---

### Option 2: AWS EC2 GPU Instances (PAID)

**Best For**: Production runs, many simulations, reliable access

**GPU Options**:

- **g4dn.xlarge**: NVIDIA T4 (16GB) - **$0.526/hour**
- **g4dn.2xlarge**: NVIDIA T4 (16GB) - **$0.752/hour**
- **g5.xlarge**: NVIDIA A10G (24GB) - **$1.006/hour**
- **p3.2xlarge**: NVIDIA V100 (16GB) - **$3.06/hour**
- **p4d.24xlarge**: NVIDIA A100 (40GB) - **$32.77/hour**

**Speedup**:

- T4: **3-5x faster** than MPS
- A10G: **5-8x faster** than MPS
- V100: **5-8x faster** than MPS
- A100: **10-15x faster** than MPS

**How to Get**:

1. Sign up: https://aws.amazon.com/ec2/
2. Launch instance: EC2 Dashboard → Launch Instance
3. Select GPU instance type (g4dn.xlarge recommended)
4. Choose AMI: "Deep Learning AMI (Ubuntu)" or "Deep Learning Base AMI"
5. Configure security group (allow SSH)
6. Launch and connect via SSH

**Cost Estimate**:

- T4 (g4dn.xlarge): ~$0.50/hour = **$2.50 for 5-hour benchmark**
- A10G (g5.xlarge): ~$1/hour = **$5 for 5-hour benchmark**

**Setup Time**: 15-30 minutes

---

### Option 3: Google Cloud Platform (GCP) (PAID)

**Best For**: Similar to AWS, sometimes better pricing

**GPU Options**:

- **n1-standard-4 + T4**: **$0.35/hour** (cheaper than AWS)
- **n1-standard-8 + V100**: **$2.48/hour**
- **a2-highgpu-1g + A100**: **$3.67/hour**

**Speedup**: Same as AWS equivalents

**How to Get**:

1. Sign up: https://cloud.google.com/
2. Create project
3. Enable Compute Engine API
4. Create VM with GPU: Compute Engine → VM instances → Create
5. Select GPU type and attach

**Cost**: Similar or slightly cheaper than AWS

---

### Option 4: Azure GPU Instances (PAID)

**Best For**: If you have Azure credits or prefer Microsoft ecosystem

**GPU Options**:

- **NC6s_v3**: NVIDIA V100 - **$3.06/hour**
- **ND96asr_v4**: NVIDIA A100 - **$32.77/hour**

**Speedup**: Same as AWS/GCP equivalents

**How to Get**:

1. Sign up: https://azure.microsoft.com/
2. Create resource group
3. Create VM with GPU: Virtual machines → Create
4. Select GPU-optimized size

---

### Option 5: Lambda Labs (PAID) ⭐ **BEST VALUE**

**Best For**: Best price/performance, designed for ML workloads

**GPU Options**:

- **GPU Cloud**: RTX 6000 Ada (48GB) - **$0.50/hour**
- **GPU Cloud**: A100 (40GB) - **$1.10/hour**
- **GPU Cloud**: A6000 (48GB) - **$0.50/hour**

**Speedup**:

- RTX 6000 Ada: **8-12x faster** than MPS
- A100: **10-15x faster** than MPS

**How to Get**:

1. Sign up: https://lambdalabs.com/
2. Request access (usually approved quickly)
3. Launch instance from dashboard
4. SSH in and run your code

**Cost**: **Best value** - often cheaper than AWS/GCP
**Setup Time**: 10 minutes

---

### Option 6: RunPod (PAID)

**Best For**: Pay-per-use, no commitment

**GPU Options**:

- **RTX 3090**: **$0.29/hour**
- **RTX 4090**: **$0.49/hour**
- **A100**: **$1.79/hour**

**Speedup**:

- RTX 3090: **6-10x faster** than MPS
- RTX 4090: **8-12x faster** than MPS
- A100: **10-15x faster** than MPS

**How to Get**:

1. Sign up: https://www.runpod.io/
2. Create pod with GPU
3. Connect via Jupyter or SSH
4. Run your code

**Cost**: Very competitive pricing
**Setup Time**: 5 minutes

---

### Option 7: Vast.ai (PAID) ⭐ **CHEAPEST**

**Best For**: Absolute cheapest option, peer-to-peer GPU rental

**GPU Options**:

- **RTX 3090**: **$0.20-0.30/hour**
- **RTX 4090**: **$0.40-0.60/hour**
- **A100**: **$1.50-2.00/hour**

**Speedup**: Same as RunPod equivalents

**How to Get**:

1. Sign up: https://vast.ai/
2. Search for GPU instances
3. Rent by the hour
4. SSH in and run

**Cost**: **Cheapest option** - often 50% cheaper than others
**Setup Time**: 5 minutes
**Note**: Less reliable (peer-to-peer), but very cheap

---

## 📊 Comparison Table

| Provider         | GPU       | Speedup vs MPS | Cost/Hour  | Best For         |
| ---------------- | --------- | -------------- | ---------- | ---------------- |
| **Google Colab** | T4 (Free) | 3-5x           | **FREE**   | Quick tests      |
| **Lambda Labs**  | A100      | 10-15x         | $1.10      | Best value       |
| **RunPod**       | RTX 4090  | 8-12x          | $0.49      | Pay-per-use      |
| **Vast.ai**      | RTX 3090  | 6-10x          | $0.20-0.30 | **Cheapest**     |
| **AWS**          | T4        | 3-5x           | $0.53      | Enterprise       |
| **GCP**          | T4        | 3-5x           | $0.35      | Google ecosystem |

---

## 🎯 Recommendations

### For Quick Tests (FREE):

**Google Colab** - Just sign in and use T4 GPU for free

### For Best Value:

**Lambda Labs** or **RunPod** - Best price/performance ratio

### For Cheapest:

**Vast.ai** - Can get RTX 3090 for $0.20/hour

### For Reliability:

**AWS** or **GCP** - Most reliable, enterprise-grade

---

## 💡 Cost Estimates for Your Benchmark

**Current**: 60-90 minutes on MPS (free)

**With GPU**:

- **Colab T4 (Free)**: 20-30 minutes = **FREE**
- **Vast.ai RTX 3090**: 10-15 minutes = **$0.05-0.08** (3-5 cents!)
- **Lambda Labs A100**: 5-10 minutes = **$0.10-0.20** (10-20 cents)
- **AWS T4**: 20-30 minutes = **$0.20-0.30**

---

## 🚀 Quick Start: Google Colab (Recommended for First Try)

1. **Go to**: https://colab.research.google.com
2. **New Notebook**: File → New notebook
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
4. **Upload code**: Upload your benchmark files
5. **Run**: Execute cells to run benchmark

**Total time to first run**: 5 minutes
**Cost**: FREE

---

## 📝 Notes

- **GPU only helps with DQN training** - network simulation is CPU-bound
- **Current MPS GPU is already good** - 2-3x speedup is decent
- **For 5-minute benchmarks**: Consider reducing runs/rounds instead
- **Cloud GPUs**: Best for many runs or production workloads

---

**Recommendation**: Start with **Google Colab (free)** to test, then move to **Lambda Labs** or **RunPod** if you need more power.
