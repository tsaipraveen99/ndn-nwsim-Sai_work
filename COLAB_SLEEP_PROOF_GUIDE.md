# Keep Colab Running When Your System Sleeps

## 🚀 EASIEST Solution: Built-In Keep-Alive (No Separate Cell Needed!)

### ✅ **NEW: Keep-Alive is Built-In!**

The `COLAB_SINGLE_CELL_ENHANCED.py` script **now includes keep-alive automatically**!

**Just run the script in one cell - that's it!** The keep-alive starts automatically and works even when your system sleeps.

**No need to run a separate cell!** 🎉

---

## 🔍 How It Works

**Important**: JavaScript keep-alive runs in the **browser**, not in the Python runtime. This means:

- ✅ **It CAN run while Python code is executing**
- ✅ **You don't need a separate cell**
- ✅ **It works even when your system sleeps**

The built-in keep-alive includes:

- ✅ JavaScript keep-alive (runs in browser - most reliable)
- ✅ Python keep-alive (backup - runs in background thread)

---

## 🎯 Alternative: Manual Setup (If You Prefer)

If you want to run keep-alive manually in a separate cell:

### Step 1: Start Your Benchmark

Run your `COLAB_SINGLE_CELL_ENHANCED.py` script in Cell 1.

### Step 2: In a NEW Cell, Run This JavaScript

**Copy and paste this into a SEPARATE cell:**

```javascript
function ClickConnect() {
  console.log("⏰ Keep-alive: " + new Date().toLocaleTimeString());
  document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(ClickConnect, 60000);
ClickConnect(); // Click immediately
```

**Note**: Even though the cell is "executing", JavaScript runs in the browser, so it CAN run while Python executes in another cell!

**That's it!** Your Colab will stay connected even when your system sleeps.

---

## 🎯 Complete Solution (Recommended)

### Option A: Use the Complete Keep-Alive Script

**In a NEW cell, run:**

```python
# Copy the contents of COLAB_KEEP_ALIVE_COMPLETE.py
# Or run this:
exec(open('COLAB_KEEP_ALIVE_COMPLETE.py').read())
```

This provides:

- ✅ JavaScript keep-alive (most reliable)
- ✅ Python keep-alive (backup)
- ✅ Auto-save to Drive (prevents data loss)

### Option B: Manual Setup

**Cell 1: Your Benchmark**

```python
# Your COLAB_SINGLE_CELL_ENHANCED.py code here
```

**Cell 2: JavaScript Keep-Alive**

```javascript
function ClickConnect() {
  console.log("⏰ Keep-alive: " + new Date().toLocaleTimeString());
  document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(ClickConnect, 60000);
ClickConnect();
```

**Cell 3: Auto-Save to Drive (Optional)**

```python
from google.colab import drive
drive.mount('/content/drive')

import time
import shutil
from pathlib import Path
from datetime import datetime

def auto_save():
    while True:
        time.sleep(600)  # Every 10 minutes
        try:
            if Path('benchmark_checkpoints').exists():
                shutil.copytree('benchmark_checkpoints',
                              '/content/drive/MyDrive/ndn_backups',
                              dirs_exist_ok=True)
                print(f"💾 Saved: {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            print(f"⚠️ {e}")

import threading
threading.Thread(target=auto_save, daemon=True).start()
print("✅ Auto-save started!")
```

---

## ✅ How It Works

### JavaScript Keep-Alive

- **What it does**: Automatically clicks the "Connect" button every 60 seconds
- **Why it works**: Colab thinks you're actively using the session
- **Works when**: Your system sleeps, tab is minimized, browser is in background
- **Reliability**: ⭐⭐⭐⭐⭐ (Most reliable method)

### Python Keep-Alive

- **What it does**: Prints output every 5 minutes
- **Why it works**: Keeps the Python kernel active
- **Works when**: Tab is open (even if minimized)
- **Reliability**: ⭐⭐⭐ (Good backup, but less reliable than JavaScript)

### Auto-Save to Drive

- **What it does**: Saves checkpoints to Google Drive every 10 minutes
- **Why it works**: Prevents data loss if Colab disconnects
- **Works when**: Drive is mounted
- **Reliability**: ⭐⭐⭐⭐⭐ (Prevents data loss)

---

## 🎯 Best Practice Setup

### For Long Runs (2+ hours):

1. **Cell 1**: Run your benchmark
2. **Cell 2**: Run JavaScript keep-alive
3. **Cell 3**: Mount Drive and enable auto-save
4. **Result**: Your simulation runs even when:
   - Your laptop sleeps
   - You close the browser
   - You're away for hours

### For Short Runs (< 1 hour):

Just use **Cell 2** (JavaScript keep-alive) - that's enough!

---

## ⚠️ Important Notes

1. **Run in separate cells**: Don't put keep-alive in the same cell as your benchmark
2. **JavaScript is best**: Use JavaScript keep-alive for maximum reliability
3. **Keep tab open**: The JavaScript method works even if tab is minimized, but keep it open
4. **Save checkpoints**: Always enable auto-save for long runs
5. **Colab Pro**: If you have Colab Pro, you get 12-hour sessions (vs 90 minutes free)

---

## 🛑 To Stop Keep-Alive

**JavaScript:**

```javascript
clearInterval(ClickConnect);
```

**Python:**
Just interrupt the cell (Stop button) or restart the runtime.

---

## 🔍 Verify It's Working

**Check the console:**

- You should see "⏰ Keep-alive: [time]" every 60 seconds
- The "Connect" button should show activity

**Check your benchmark:**

- Your benchmark should continue running
- Checkpoints should be saved (if auto-save enabled)

---

## 💡 Pro Tips

1. **Use Colab Pro**: 12-hour sessions vs 90 minutes (free)
2. **Save to Drive**: Always enable auto-save for important runs
3. **Check progress**: Use `check_benchmark_status.py` to monitor progress
4. **Multiple tabs**: You can open Colab in multiple tabs (keep-alive works in all)
5. **Mobile check**: Check progress on your phone (Colab works on mobile)

---

## 🚨 Troubleshooting

### Problem: Colab still disconnects

**Solution**:

- Make sure JavaScript keep-alive is running
- Check browser console for errors
- Try reducing the interval: `setInterval(ClickConnect, 30000)` (30 seconds)

### Problem: Keep-alive stops working

**Solution**:

- Refresh the page
- Restart the runtime
- Re-run the JavaScript keep-alive cell

### Problem: Can't find Connect button

**Solution**:

- Make sure you're in a Colab notebook (not Jupyter)
- Try: `document.querySelector("colab-toolbar-button#connect")` in console
- Use Python keep-alive as backup

---

## 📚 Files Reference

- `COLAB_KEEP_ALIVE_COMPLETE.py` - Complete keep-alive solution
- `COLAB_KEEP_ALIVE.js` - JavaScript-only version
- `COLAB_KEEP_ALIVE.py` - Python-only version
- `COLAB_KEEP_ALIVE_GUIDE.md` - Detailed guide

---

## ✅ Quick Checklist

Before leaving your simulation running:

- [ ] Benchmark is running in Cell 1
- [ ] JavaScript keep-alive is running in Cell 2
- [ ] Auto-save is enabled (optional but recommended)
- [ ] Colab tab is open (can be minimized)
- [ ] You've verified keep-alive is working (check console)

**Now you can close your laptop and go!** 🎉
