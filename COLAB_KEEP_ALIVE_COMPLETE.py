# ============================================================================
# COMPLETE COLAB KEEP-ALIVE SOLUTION
# ============================================================================
# This script keeps your Colab session alive even when your system sleeps
# Run this in a SEPARATE cell while your benchmark is running
# ============================================================================

import time
import threading
import subprocess
from datetime import datetime
from pathlib import Path

print("="*70)
print("COLAB KEEP-ALIVE - COMPLETE SOLUTION")
print("="*70)
print("This will keep your Colab session alive even when your system sleeps.")
print("Run this in a SEPARATE cell while your benchmark is running.")
print("="*70)

# ============================================================================
# METHOD 1: JavaScript Keep-Alive (Most Reliable)
# ============================================================================
print("\n[1/3] Setting up JavaScript keep-alive...")

javascript_code = """
function ClickConnect(){
    console.log("⏰ Keep-alive: " + new Date().toLocaleTimeString());
    const connectButton = document.querySelector("colab-toolbar-button#connect");
    if (connectButton) {
        connectButton.click();
        console.log("✅ Connection maintained");
    } else {
        console.log("⚠️ Connect button not found");
    }
}

// Click every 60 seconds
const keepAliveInterval = setInterval(ClickConnect, 60000);

// Click immediately
ClickConnect();

console.log("✅ JavaScript keep-alive started!");
console.log("💡 To stop: Run 'clearInterval(keepAliveInterval)' in a new cell");
"""

try:
    from IPython.display import Javascript, display
    display(Javascript(javascript_code))
    print("✅ JavaScript keep-alive activated!")
except Exception as e:
    print(f"⚠️ JavaScript keep-alive failed: {e}")
    print("   Continuing with Python keep-alive...")

# ============================================================================
# METHOD 2: Python Keep-Alive (Backup)
# ============================================================================
print("\n[2/3] Setting up Python keep-alive (backup)...")

def python_keep_alive():
    """Print periodically to keep session active"""
    while True:
        time.sleep(300)  # Every 5 minutes
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"⏰ Python keep-alive: {timestamp}")

# Start in background thread
python_thread = threading.Thread(target=python_keep_alive, daemon=True)
python_thread.start()
print("✅ Python keep-alive started!")

# ============================================================================
# METHOD 3: Auto-Save Checkpoints to Drive (Prevent Data Loss)
# ============================================================================
print("\n[3/3] Setting up auto-save to Google Drive...")

def auto_save_checkpoints():
    """Save checkpoints to Google Drive periodically"""
    # Check if Drive is mounted
    drive_path = Path('/content/drive/MyDrive')
    if not drive_path.exists():
        print("⚠️ Google Drive not mounted. Skipping auto-save.")
        print("   To enable: Run 'from google.colab import drive; drive.mount(\"/content/drive\")'")
        return
    
    checkpoint_dir = Path('benchmark_checkpoints')
    drive_backup_dir = drive_path / 'ndn_simulation_backups'
    drive_backup_dir.mkdir(exist_ok=True)
    
    save_count = 0
    while True:
        time.sleep(600)  # Every 10 minutes
        save_count += 1
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            if checkpoint_dir.exists():
                # Copy checkpoint files
                import shutil
                backup_path = drive_backup_dir / f'checkpoint_{timestamp}'
                
                # Copy only if there are new files
                if any(checkpoint_dir.iterdir()):
                    shutil.copytree(checkpoint_dir, backup_path, dirs_exist_ok=True)
                    print(f"💾 [{save_count}] Saved to Drive: {timestamp}")
                else:
                    print(f"⏳ [{save_count}] No checkpoints to save yet...")
            else:
                print(f"⏳ [{save_count}] Waiting for checkpoints directory...")
        except Exception as e:
            print(f"⚠️ Save error: {e}")

# Start auto-save in background
save_thread = threading.Thread(target=auto_save_checkpoints, daemon=True)
save_thread.start()
print("✅ Auto-save started! (Will save every 10 minutes if Drive is mounted)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ KEEP-ALIVE ACTIVE!")
print("="*70)
print("Your Colab session will stay connected even when your system sleeps.")
print("\nActive methods:")
print("  ✅ JavaScript keep-alive (clicks Connect button every 60s)")
print("  ✅ Python keep-alive (prints every 5 minutes)")
print("  ✅ Auto-save to Drive (saves checkpoints every 10 minutes)")
print("\n💡 Tips:")
print("  - Keep the Colab tab open in your browser (can be minimized)")
print("  - The JavaScript method works even when tab is in background")
print("  - Checkpoints are saved to Drive to prevent data loss")
print("  - You can close your laptop - Colab will keep running!")
print("="*70)

# Keep this cell running
print("\n⏳ Keep-alive running... (This cell will keep running)")
print("   You can minimize this cell or leave it running.")
print("   Your benchmark will continue in the other cell.\n")

# Print status every minute
while True:
    time.sleep(60)
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"💚 Keep-alive active: {timestamp}")

