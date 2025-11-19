# Fix Missing Dependencies in Colab

## ❌ Error: `ModuleNotFoundError: No module named 'mmh3'`

## ✅ Quick Fix

Run this in a Colab cell to install all missing dependencies:

```python
# Install all required packages
!pip install networkx numpy torch scipy matplotlib pandas scikit-learn dill bitarray hdbscan tensorflow mmh3
```

## 📋 Complete Installation Command

Copy this into a Colab cell and run it:

```python
# Complete dependency installation
!pip install \
    networkx \
    numpy \
    torch \
    scipy \
    matplotlib \
    pandas \
    scikit-learn \
    dill \
    bitarray \
    hdbscan \
    tensorflow \
    mmh3
```

## ✅ Verify Installation

After installing, verify all packages are available:

```python
# Verify imports
try:
    import networkx as nx
    import numpy as np
    import torch
    import scipy
    import matplotlib
    import pandas as pd
    import sklearn
    import dill
    import bitarray
    import hdbscan
    import tensorflow as tf
    import mmh3
    print("✅ All dependencies installed successfully!")
except ImportError as e:
    print(f"❌ Missing: {e}")
```

## 🔍 What is mmh3?

`mmh3` is the MurmurHash3 library used for:
- Bloom filter hash functions
- Fast hashing in the NDN simulation
- Content name hashing

It's a critical dependency for the Bloom filter implementation in `utils.py`.

## 🚀 After Installing

Once `mmh3` is installed, try running your benchmark again:

```python
import os
os.environ['NDN_SIM_USE_DQN'] = '1'
# ... (set other configs)
!python benchmark.py
```

---

**Quick fix**: Just run `!pip install mmh3` in a new cell! 🚀

