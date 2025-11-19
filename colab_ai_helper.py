#!/usr/bin/env python3
"""
AI Collaboration Helper for Google Colab
Helps keep AI assistant connected to your Colab runtime
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class AICollaborator:
    """Helper class for AI collaboration in Colab"""
    
    def __init__(self, status_file='ai_status.json', log_file='ai_log.txt'):
        self.status_file = status_file
        self.log_file = log_file
        self.setup()
    
    def setup(self):
        """Initialize logging files"""
        # Create log file if it doesn't exist
        Path(self.log_file).touch()
        
        # Initialize status file if it doesn't exist
        if not os.path.exists(self.status_file):
            with open(self.status_file, 'w') as f:
                json.dump([], f)
    
    def log(self, message: str, level: str = 'INFO', data: Optional[Dict] = None):
        """Log message for AI to read"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
        
        # Update status JSON
        self.update_status(level.lower(), message, data)
        
        # Print to console
        print(f"[{level}] {message}")
    
    def update_status(self, status_type: str, message: str, data: Optional[Dict] = None):
        """Update status JSON file for easy AI reading"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'type': status_type,
            'message': message,
            'data': data or {}
        }
        
        # Load existing statuses
        statuses = []
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    statuses = json.load(f)
            except:
                pass
        
        # Append new status
        statuses.append(status)
        
        # Keep last 100 entries
        if len(statuses) > 100:
            statuses = statuses[-100:]
        
        # Save
        with open(self.status_file, 'w') as f:
            json.dump(statuses, f, indent=2)
    
    def capture_error(self, error: Exception, context: Optional[Dict] = None):
        """Capture error with full traceback for AI analysis"""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.log(f"ERROR: {error_info['error_type']}: {error_info['error_message']}", 
                'ERROR', error_info)
        
        return error_info
    
    def capture_output(self, output: str, label: str = 'output'):
        """Capture command output for AI"""
        # Truncate if too long
        truncated = output[:1000] if len(output) > 1000 else output
        self.log(f"{label}: {truncated}", 'OUTPUT')
    
    def request_help(self, question: str, context: Optional[Dict] = None):
        """Request help from AI assistant"""
        self.log(f"QUESTION: {question}", 'REQUEST', context)
        print(f"\n📋 Question logged for AI: {question}")
        print("   Share ai_status.json content with AI assistant")
    
    def log_file_check(self, filename: str):
        """Check if file exists and log status"""
        exists = os.path.exists(filename)
        if exists:
            size = os.path.getsize(filename)
            self.log(f"File check: {filename} exists ({size:,} bytes)", 'INFO')
        else:
            self.log(f"File check: {filename} MISSING", 'WARNING')
        return exists
    
    def log_import_check(self, module_name: str):
        """Check if module can be imported"""
        try:
            __import__(module_name)
            self.log(f"Import check: {module_name} ✅", 'SUCCESS')
            return True
        except ImportError as e:
            self.log(f"Import check: {module_name} ❌ - {str(e)}", 'ERROR')
            self.capture_error(e, {'module': module_name})
            return False
    
    def get_status_summary(self) -> str:
        """Get summary of recent status for sharing with AI"""
        if not os.path.exists(self.status_file):
            return "No status file found"
        
        with open(self.status_file, 'r') as f:
            statuses = json.load(f)
        
        if not statuses:
            return "No status entries"
        
        # Get last 10 entries
        recent = statuses[-10:]
        
        summary = "Recent Colab Status:\n"
        summary += "="*60 + "\n"
        
        for status in recent:
            summary += f"[{status['timestamp']}] {status['type'].upper()}: {status['message']}\n"
            if status.get('data'):
                summary += f"   Data: {json.dumps(status['data'], indent=2)}\n"
        
        return summary
    
    def share_with_ai(self):
        """Print status summary for sharing with AI"""
        print("\n" + "="*60)
        print("SHARE THIS WITH AI ASSISTANT:")
        print("="*60)
        print(self.get_status_summary())
        print("\nOr share the full file:")
        print(f"  !cat {self.status_file}")
        print("="*60)


# Convenience functions
_ai_instance = None

def get_ai():
    """Get or create AI collaborator instance"""
    global _ai_instance
    if _ai_instance is None:
        _ai_instance = AICollaborator()
    return _ai_instance

def log(message: str, level: str = 'INFO', data: Optional[Dict] = None):
    """Quick log function"""
    get_ai().log(message, level, data)

def capture_error(error: Exception, context: Optional[Dict] = None):
    """Quick error capture"""
    return get_ai().capture_error(error, context)

def request_help(question: str, context: Optional[Dict] = None):
    """Quick help request"""
    get_ai().request_help(question, context)

def share_with_ai():
    """Quick share function"""
    get_ai().share_with_ai()


# Example usage
if __name__ == '__main__':
    ai = AICollaborator()
    
    # Check files
    print("Checking required files...")
    required = ['main.py', 'router.py', 'benchmark.py']
    for f in required:
        ai.log_file_check(f)
    
    # Check imports
    print("\nChecking imports...")
    ai.log_import_check('main')
    ai.log_import_check('router')
    
    # Share status
    ai.share_with_ai()

