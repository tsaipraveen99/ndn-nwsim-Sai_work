# How to Keep AI Assistant Connected to Colab Runtime

## 🎯 Goal

Create a workflow where the AI assistant can:
- Examine your Colab runtime state
- See errors and logs
- Make code changes
- Iterate quickly

---

## 🔄 Method 1: Google Drive Sync (Recommended)

### Setup: Sync Files Between Local and Colab

#### Step 1: Mount Google Drive in Colab
```python
from google.colab import drive
drive.mount('/content/drive')

# Create a project folder
!mkdir -p /content/drive/MyDrive/ndn-project
!cp *.py /content/drive/MyDrive/ndn-project/  # Copy files to Drive
```

#### Step 2: Create Sync Script in Colab
```python
# Save this as sync_from_drive.py in Colab
import shutil
import os

drive_path = '/content/drive/MyDrive/ndn-project'
local_path = '.'

files_to_sync = [
    'main.py', 'router.py', 'endpoints.py', 'utils.py',
    'dqn_agent.py', 'packet.py', 'metrics.py',
    'benchmark.py', 'statistical_analysis.py'
]

print("Syncing files from Drive...")
for f in files_to_sync:
    drive_file = os.path.join(drive_path, f)
    local_file = os.path.join(local_path, f)
    if os.path.exists(drive_file):
        shutil.copy(drive_file, local_file)
        print(f"  ✅ Synced {f}")
    else:
        print(f"  ⚠️  {f} not in Drive")

print("\n✅ Sync complete!")
```

#### Step 3: Workflow
1. **AI makes changes locally** → Files saved to your local machine
2. **You upload to Drive** → Copy files to `/content/drive/MyDrive/ndn-project/`
3. **Run sync in Colab** → `!python sync_from_drive.py`
4. **Test in Colab** → Run your benchmark
5. **Share errors/logs** → Copy-paste to AI
6. **Repeat**

---

## 🔄 Method 2: Shared Logging & Status File

### Create a Status File That AI Can Read

#### In Colab: Create Status Logger
```python
# status_logger.py - Run this in Colab
import json
import os
from datetime import datetime
from pathlib import Path

STATUS_FILE = 'ai_status.json'

def log_status(status_type, message, data=None):
    """Log status for AI to read"""
    status = {
        'timestamp': datetime.now().isoformat(),
        'type': status_type,  # 'error', 'progress', 'result', 'request'
        'message': message,
        'data': data or {}
    }
    
    # Append to status file
    statuses = []
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r') as f:
            statuses = json.load(f)
    
    statuses.append(status)
    
    # Keep last 100 entries
    if len(statuses) > 100:
        statuses = statuses[-100:]
    
    with open(STATUS_FILE, 'w') as f:
        json.dump(statuses, f, indent=2)
    
    print(f"[{status_type.upper()}] {message}")

# Example usage
log_status('progress', 'Benchmark started', {'run': 1, 'total': 10})
log_status('error', 'Import failed', {'error': 'ModuleNotFoundError: main'})
log_status('result', 'Hit rate calculated', {'hit_rate': 15.5})
```

#### In Your Local Project: Create Status Reader
```python
# read_colab_status.py - Run this locally
import json
import os

def read_colab_status(file_path='ai_status.json'):
    """Read status file from Colab (if synced via Drive)"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            statuses = json.load(f)
        
        print("Recent Colab Status:")
        print("="*60)
        for status in statuses[-10:]:  # Last 10 entries
            print(f"[{status['timestamp']}] {status['type'].upper()}: {status['message']}")
            if status['data']:
                print(f"   Data: {status['data']}")
        return statuses
    else:
        print("Status file not found. Sync from Drive first.")
        return []

if __name__ == '__main__':
    read_colab_status()
```

---

## 🔄 Method 3: Copy-Paste Workflow (Simplest)

### Quick Feedback Loop

#### Step 1: Share Colab Output with AI
When you get an error, copy-paste:
- The full error traceback
- Relevant log output
- Current file contents (if needed)

#### Step 2: AI Provides Fix
AI will give you:
- Updated code snippets
- File changes
- Instructions

#### Step 3: Apply in Colab
```python
# Option A: Edit file directly
with open('benchmark.py', 'r') as f:
    content = f.read()

# Make changes (AI will tell you what to change)
content = content.replace('old_code', 'new_code')

with open('benchmark.py', 'w') as f:
    f.write(content)

# Option B: Create new file
# AI will provide complete file content
```

#### Step 4: Test and Share Results
```python
# Run and capture output
!python benchmark.py 2>&1 | tee output.log

# Share output.log with AI
```

---

## 🔄 Method 4: Git-Based Workflow (Advanced)

### Use Git to Sync Code

#### Setup Git Repo
```python
# In Colab
!git clone https://github.com/yourusername/ndn-project.git
# Or use a private repo

# Work on branch
!git checkout -b colab-experiments
```

#### Workflow
1. **AI makes changes** → Commits to local repo
2. **You push to GitHub** → `git push origin colab-experiments`
3. **Pull in Colab** → `!git pull origin colab-experiments`
4. **Test in Colab** → Run benchmark
5. **Share results** → Copy logs/output to AI
6. **Repeat**

---

## 🎯 Recommended Setup: Hybrid Approach

### Best of All Worlds

#### 1. Create a Collaboration Notebook
```python
# colab_ai_helper.py - Helper functions for AI collaboration

import json
import os
from datetime import datetime
from pathlib import Path

class AICollaborator:
    def __init__(self, status_file='ai_status.json', log_file='ai_log.txt'):
        self.status_file = status_file
        self.log_file = log_file
        self.setup()
    
    def setup(self):
        """Initialize logging"""
        Path(self.log_file).touch()
    
    def log(self, message, level='INFO', data=None):
        """Log message for AI"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        # Also update status
        self.update_status(level.lower(), message, data)
        
        print(log_entry.strip())
    
    def update_status(self, status_type, message, data=None):
        """Update status JSON for easy reading"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'type': status_type,
            'message': message,
            'data': data or {}
        }
        
        statuses = []
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    statuses = json.load(f)
            except:
                pass
        
        statuses.append(status)
        if len(statuses) > 100:
            statuses = statuses[-100:]
        
        with open(self.status_file, 'w') as f:
            json.dump(statuses, f, indent=2)
    
    def capture_error(self, error, context=None):
        """Capture error for AI analysis"""
        self.log(f"ERROR: {str(error)}", 'ERROR', {
            'error_type': type(error).__name__,
            'context': context
        })
    
    def capture_output(self, output, label='output'):
        """Capture command output"""
        self.log(f"{label}: {output[:500]}", 'OUTPUT')  # First 500 chars
    
    def request_help(self, question, context=None):
        """Request help from AI"""
        self.log(f"QUESTION: {question}", 'REQUEST', context)
        print(f"\n📋 Question logged for AI: {question}")
        print("   Share ai_status.json or ai_log.txt with AI assistant")

# Usage
ai = AICollaborator()

# Log progress
ai.log("Benchmark started", 'PROGRESS', {'run': 1})

# Capture errors
try:
    from main import create_network
except Exception as e:
    ai.capture_error(e, context={'file': 'main.py', 'line': 16})

# Request help
ai.request_help("How to fix import error?", context={'error': 'ModuleNotFoundError'})
```

#### 2. Use in Your Benchmark
```python
# At the top of benchmark.py or in a Colab cell
from colab_ai_helper import AICollaborator
ai = AICollaborator()

# Wrap your code
try:
    from main import create_network, run_simulation
    ai.log("Imports successful", 'SUCCESS')
except Exception as e:
    ai.capture_error(e)
    ai.request_help("Fix import error")
    raise
```

#### 3. Share with AI
```python
# After running, share these files with AI:
# 1. ai_status.json - Structured status
# 2. ai_log.txt - Full log
# 3. Any error output

# View status
!cat ai_status.json

# View log
!tail -50 ai_log.txt
```

---

## 📋 Quick Start Checklist

### For Immediate AI Collaboration:

1. **Create helper in Colab**:
   ```python
   # Copy colab_ai_helper.py content above
   ```

2. **Use it in your code**:
   ```python
   from colab_ai_helper import AICollaborator
   ai = AICollaborator()
   ```

3. **When you get an error**:
   ```python
   try:
       # Your code
   except Exception as e:
       ai.capture_error(e)
       # Copy ai_status.json content and share with AI
   ```

4. **Share with AI**:
   - Copy `ai_status.json` content
   - Or copy error traceback
   - Paste in chat with AI

5. **Apply AI's fix**:
   - AI provides code changes
   - Apply in Colab
   - Test again

---

## 💡 Pro Tips

1. **Keep status file in Drive**: Sync `ai_status.json` to Drive for persistence
2. **Use version control**: Track changes with Git
3. **Log everything**: More context = better AI help
4. **Share structured data**: JSON is easier for AI to parse
5. **Iterate quickly**: Small changes, test, share results

---

## 🚀 Example Workflow

```python
# 1. Setup (run once)
from colab_ai_helper import AICollaborator
ai = AICollaborator()

# 2. Run benchmark
try:
    !python benchmark.py
    ai.log("Benchmark completed", 'SUCCESS')
except Exception as e:
    ai.capture_error(e)
    
# 3. Share status with AI
print("\n📋 Share this with AI:")
with open('ai_status.json', 'r') as f:
    print(f.read())

# 4. Get fix from AI, apply it, repeat
```

---

**This setup lets you iterate quickly with AI assistance!** The AI can see your errors, understand context, and provide targeted fixes. 🎯

