# How to Update Files in Google Colab

## 🔄 Method 1: Re-upload Single File (Easiest)

### Step 1: Download Updated File Locally
If you have the updated `benchmark.py` on your local machine, you're ready!

### Step 2: Upload to Colab

**Option A: Using Colab's File Upload**
```python
from google.colab import files

# Upload the updated file
uploaded = files.upload()

# Verify it was uploaded
import os
if os.path.exists('benchmark.py'):
    print("✅ benchmark.py uploaded successfully!")
    # Check file size to confirm
    size = os.path.getsize('benchmark.py')
    print(f"   File size: {size} bytes")
else:
    print("❌ Upload failed!")
```

**Option B: Direct Upload Button**
1. Click the **folder icon** (📁) in the left sidebar
2. Click the **upload icon** (⬆️) 
3. Select your updated `benchmark.py` file
4. Wait for upload to complete

---

## 🔄 Method 2: Edit File Directly in Colab

### Step 1: Open File in Colab
```python
# View current file
!cat benchmark.py | head -20
```

### Step 2: Create New File with Updated Content
```python
# Read your updated file content (if you have it)
# Or copy-paste the updated code into a new cell

# Write updated content
with open('benchmark.py', 'w') as f:
    f.write('''# Your updated benchmark.py content here
# ... (paste the entire updated file)
''')
```

**Note**: This is tedious for large files. Method 1 is better.

---

## 🔄 Method 3: Use Git (If You Have a Repo)

### If Your Code is in a Git Repository:
```python
# Clone or pull latest changes
!git clone https://github.com/yourusername/your-repo.git
# OR if already cloned:
!cd your-repo && git pull

# Copy updated file
!cp your-repo/benchmark.py benchmark.py
```

---

## 🔄 Method 4: Mount Google Drive (Best for Frequent Updates)

### Step 1: Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Copy File from Drive
```python
# If you saved benchmark.py to Drive
!cp /content/drive/MyDrive/benchmark.py benchmark.py
print("✅ File copied from Drive!")
```

### Step 3: Keep Drive Synced
- Save updated `benchmark.py` to your Google Drive
- Re-run the copy command in Colab to get latest version

---

## ✅ Recommended Workflow

### For One-Time Update:
1. **Use Method 1 (Upload)**: Quick and easy
   ```python
   from google.colab import files
   files.upload()  # Select benchmark.py
   ```

### For Frequent Updates:
1. **Use Method 4 (Drive)**: 
   - Save files to Google Drive
   - Mount Drive in Colab
   - Copy files as needed

---

## 🔍 Verify Update Worked

After uploading, verify the changes:

```python
import os

# Check file exists
if os.path.exists('benchmark.py'):
    print("✅ File exists")
    
    # Check file size (should match your updated file)
    size = os.path.getsize('benchmark.py')
    print(f"   Size: {size} bytes")
    
    # Check for warm-up code (verify update)
    with open('benchmark.py', 'r') as f:
        content = f.read()
        if 'warmup_cache' in content:
            print("✅ Warm-up code found - update successful!")
        else:
            print("⚠️  Warm-up code not found - may need to re-upload")
            
    # Show first few lines with warm-up
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'warmup' in line.lower() or 'WARM-UP' in line:
            print(f"\n   Line {i+1}: {line.strip()}")
            # Show context
            for j in range(max(0, i-2), min(len(lines), i+3)):
                if j != i:
                    print(f"   Line {j+1}: {lines[j].strip()}")
            break
else:
    print("❌ File not found!")
```

---

## 🚀 Quick Update Script

Copy this into a Colab cell:

```python
from google.colab import files
import os

print("="*60)
print("UPDATING benchmark.py")
print("="*60)

# Upload file
print("\n📤 Please select benchmark.py to upload...")
uploaded = files.upload()

# Verify
if 'benchmark.py' in uploaded:
    size = os.path.getsize('benchmark.py')
    print(f"\n✅ Upload successful!")
    print(f"   File size: {size:,} bytes")
    
    # Quick check for key features
    with open('benchmark.py', 'r') as f:
        content = f.read()
        checks = {
            'Warm-up phase': 'warmup_cache' in content,
            'Statistics reset': 'global_stats' in content,
            'Content alignment': 'align_user_distributions' in content
        }
        
        print("\n🔍 Feature check:")
        for feature, found in checks.items():
            status = "✅" if found else "❌"
            print(f"   {status} {feature}")
    
    print("\n✅ Ready to run benchmark!")
else:
    print("\n❌ Upload failed or file not selected")
```

---

## 📝 Step-by-Step for Your Current Situation

Since you just updated `benchmark.py` locally:

1. **In Colab, create a new cell and run:**
   ```python
   from google.colab import files
   files.upload()
   ```

2. **Select your updated `benchmark.py` file** from your local machine

3. **Verify it worked:**
   ```python
   !grep -n "warmup_cache" benchmark.py
   ```

4. **Restart your benchmark** (if it was running, stop it first)

---

## ⚠️ Important Notes

- **Stop running processes** before updating files
- **Backup current file** if needed:
  ```python
  !cp benchmark.py benchmark.py.backup
  ```
- **Restart runtime** if you get import errors after updating:
  - Runtime → Restart runtime (or Ctrl+M .)

---

## 🎯 That's It!

After uploading, your Colab environment will have the updated `benchmark.py` with the warm-up phase. You can then restart your benchmark and it will use the new evaluation methodology!

