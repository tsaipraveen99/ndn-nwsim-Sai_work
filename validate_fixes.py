#!/usr/bin/env python3
"""Validate that fixes are applied correctly"""

import os
import sys

def validate_config():
    """Check simulation parameters"""
    # Read from benchmark.py file to verify actual code changes
    import re
    benchmark_file = 'benchmark.py'
    zipf = 1.5
    contents = 150
    cache = 2000
    
    if os.path.exists(benchmark_file):
        with open(benchmark_file, 'r') as f:
            content = f.read()
            # Extract values from base_config
            zipf_match = re.search(r"'NDN_SIM_ZIPF_PARAM':\s*'([0-9.]+)'", content)
            contents_match = re.search(r"'NDN_SIM_CONTENTS':\s*'(\d+)'", content)
            cache_match = re.search(r"'NDN_SIM_CACHE_CAPACITY':\s*'(\d+)'", content)
            
            if zipf_match:
                zipf = float(zipf_match.group(1))
            if contents_match:
                contents = int(contents_match.group(1))
            if cache_match:
                cache = int(cache_match.group(1))
    
    # Also check environment variables as fallback
    zipf = float(os.getenv('NDN_SIM_ZIPF_PARAM', str(zipf)))
    contents = int(os.getenv('NDN_SIM_CONTENTS', str(contents)))
    cache = int(os.getenv('NDN_SIM_CACHE_CAPACITY', str(cache)))
    
    print("=== Configuration Validation ===")
    print(f"Zipf Parameter: {zipf} (should be 0.8)")
    print(f"Contents: {contents} (should be 1000)")
    print(f"Cache Capacity: {cache} (should be 5-10)")
    if contents > 0:
        ratio = (cache / contents) * 100
        print(f"Cache-to-Catalog Ratio: {ratio:.2f}% (should be 0.5-1%)")
    else:
        ratio = 0
        print("Cache-to-Catalog Ratio: N/A (contents is 0)")
    
    issues = []
    if zipf > 1.0:
        issues.append(f"❌ Zipf too high: {zipf} (should be 0.8)")
    elif zipf < 0.6:
        issues.append(f"⚠️  Zipf very low: {zipf} (typical range is 0.6-0.9)")
    else:
        print(f"✅ Zipf parameter is correct: {zipf}")
    
    if contents < 500:
        issues.append(f"⚠️  Contents too low: {contents} (should be 1000 for realistic scenario)")
    elif contents == 1000:
        print(f"✅ Contents is correct: {contents}")
    else:
        print(f"⚠️  Contents is {contents} (expected 1000)")
    
    if ratio > 2.0:
        issues.append(f"❌ Cache too large: {ratio:.1f}% (should be 0.5-1%)")
    elif ratio < 0.1:
        issues.append(f"⚠️  Cache very small: {ratio:.2f}% (may be too restrictive)")
    elif 0.5 <= ratio <= 1.0:
        print(f"✅ Cache-to-catalog ratio is correct: {ratio:.2f}%")
    else:
        print(f"⚠️  Cache-to-catalog ratio: {ratio:.2f}% (expected 0.5-1%)")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("\n✅ Configuration looks good!")
        return True

if __name__ == '__main__':
    success = validate_config()
    sys.exit(0 if success else 1)

