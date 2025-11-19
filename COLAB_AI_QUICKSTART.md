# Quick Start: AI Collaboration in Colab

## 🚀 3-Step Setup

### Step 1: Upload Helper File

In Colab, run:
```python
from google.colab import files
files.upload()  # Upload colab_ai_helper.py
```

### Step 2: Initialize Helper

```python
from colab_ai_helper import AICollaborator
ai = AICollaborator()

# Check your setup
ai.log("AI collaboration initialized", 'INFO')
```

### Step 3: Use When You Get Errors

```python
try:
    from main import create_network
    ai.log("Imports successful", 'SUCCESS')
except Exception as e:
    ai.capture_error(e)
    ai.share_with_ai()  # This prints what to share with AI
```

---

## 📋 When You Get an Error

### Option A: Use Helper (Recommended)
```python
from colab_ai_helper import capture_error, share_with_ai

try:
    !python benchmark.py
except Exception as e:
    capture_error(e)
    share_with_ai()  # Copy the output and share with AI
```

### Option B: Manual Share
```python
# Just copy-paste the error to AI
!python benchmark.py 2>&1 | head -50
```

---

## 🔄 Workflow

1. **Run code** → Get error
2. **Capture error** → `ai.capture_error(e)`
3. **Share with AI** → `ai.share_with_ai()` or copy error
4. **Get fix from AI** → Apply in Colab
5. **Test again** → Repeat

---

## 💡 Pro Tip

Keep this in a Colab cell and run it whenever you need help:

```python
from colab_ai_helper import AICollaborator, share_with_ai

ai = AICollaborator()

# Check files
required = ['main.py', 'router.py', 'benchmark.py']
for f in required:
    ai.log_file_check(f)

# Check imports
try:
    from main import create_network
    ai.log("✅ All good!", 'SUCCESS')
except Exception as e:
    ai.capture_error(e)
    share_with_ai()  # Share this output with AI
```

---

**That's it!** Now you can easily share errors and get fixes from AI. 🎯

