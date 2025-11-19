#!/usr/bin/env python3
"""
Diagnostic script to verify benchmark configurations are being applied correctly
"""
import os
import sys
import numpy as np
from collections import Counter

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import NDNDistribution
from main import create_network, align_user_distributions_with_producers

def verify_config():
    """Verify that benchmark configurations are being applied"""
    print("="*80)
    print("Configuration Verification")
    print("="*80)
    
    # Set test config
    test_config = {
        'NDN_SIM_NODES': '50',
        'NDN_SIM_PRODUCERS': '10',
        'NDN_SIM_CONTENTS': '200',
        'NDN_SIM_USERS': '100',
        'NDN_SIM_ROUNDS': '50',
        'NDN_SIM_REQUESTS': '20',
        'NDN_SIM_CACHE_CAPACITY': '1000',
        'NDN_SIM_ZIPF_PARAM': '1.2',
        'NDN_SIM_WARMUP_ROUNDS': '10',
        'NDN_SIM_CACHE_POLICY': 'combined',
        'NDN_SIM_QUIET': '1',
        'NDN_SIM_SKIP_DELAYS': '1',
        'NDN_SIM_USE_DQN': '0'
    }
    
    # Set environment variables
    for key, value in test_config.items():
        os.environ[key] = str(value)
    
    print("\n1. Environment Variables:")
    for key in ['NDN_SIM_CACHE_CAPACITY', 'NDN_SIM_ZIPF_PARAM', 'NDN_SIM_ROUNDS', 
                'NDN_SIM_REQUESTS', 'NDN_SIM_CONTENTS']:
        val = os.environ.get(key, 'NOT SET')
        print(f"   {key}: {val}")
    
    # Test cache capacity reading
    print("\n2. Cache Capacity Verification:")
    router_capacity = int(os.environ.get("NDN_SIM_CACHE_CAPACITY", "500"))
    print(f"   Expected: 1000")
    print(f"   Actual: {router_capacity}")
    print(f"   Status: {'✅ PASS' if router_capacity == 1000 else '❌ FAIL'}")
    
    # Test Zipf distribution
    print("\n3. Zipf Distribution Verification:")
    zipf_param = float(os.environ.get('NDN_SIM_ZIPF_PARAM', '1.2'))
    num_contents = int(os.environ.get('NDN_SIM_CONTENTS', '200'))
    
    dist = NDNDistribution(num_contents, zipf_param=zipf_param)
    print(f"   Zipf Parameter: {dist.zipf_param}")
    print(f"   Number of Contents: {dist.num_contents}")
    print(f"   Probabilities sum: {dist.probabilities.sum():.6f} (should be 1.0)")
    
    # Generate sample requests and check popularity skew
    print("\n4. Request Distribution Analysis:")
    num_samples = 10000
    requests = [dist.generate_content_name() for _ in range(num_samples)]
    request_counts = Counter(requests)
    
    # Sort by frequency
    sorted_contents = sorted(request_counts.items(), key=lambda x: x[1], reverse=True)
    top_20_percent = int(len(sorted_contents) * 0.2)
    top_20_requests = sum(count for _, count in sorted_contents[:top_20_percent])
    total_requests = sum(request_counts.values())
    top_20_percentage = (top_20_requests / total_requests) * 100
    
    print(f"   Total unique contents requested: {len(request_counts)}")
    print(f"   Top 20% contents received: {top_20_percentage:.1f}% of requests")
    print(f"   Expected (Zipf property): ~80%")
    print(f"   Status: {'✅ PASS' if top_20_percentage > 60 else '⚠️  WEAK' if top_20_percentage > 40 else '❌ FAIL'}")
    
    # Show top 10 most requested contents
    print("\n   Top 10 Most Requested Contents:")
    for i, (content, count) in enumerate(sorted_contents[:10], 1):
        percentage = (count / total_requests) * 100
        print(f"   {i:2d}. {content}: {count} requests ({percentage:.2f}%)")
    
    # Test network creation
    print("\n5. Network Creation Test:")
    try:
        G, users, producers, runtime = create_network(
            num_nodes=int(test_config['NDN_SIM_NODES']),
            num_producers=int(test_config['NDN_SIM_PRODUCERS']),
            num_contents=int(test_config['NDN_SIM_CONTENTS']),
            num_users=int(test_config['NDN_SIM_USERS']),
            cache_policy=test_config['NDN_SIM_CACHE_POLICY'],
            logger=None
        )
        
        # Check router cache capacity
        first_router = None
        for node, data in G.nodes(data=True):
            if 'router' in data:
                first_router = data['router']
                break
        
        if first_router:
            router_capacity_actual = first_router.capacity
            print(f"   Router cache capacity: {router_capacity_actual}")
            print(f"   Expected: 1000")
            print(f"   Status: {'✅ PASS' if router_capacity_actual == 1000 else '❌ FAIL'}")
        
        # Check user distribution
        if users:
            user_dist = users[0].distribution
            print(f"   User distribution Zipf param: {user_dist.zipf_param}")
            print(f"   Expected: 1.2")
            print(f"   Status: {'✅ PASS' if abs(user_dist.zipf_param - 1.2) < 0.01 else '❌ FAIL'}")
        
        # Test alignment
        print("\n6. User-Producer Alignment Test:")
        align_user_distributions_with_producers(users, producers, logger=None)
        
        if users:
            user_dist_after = users[0].distribution
            print(f"   Zipf param after alignment: {user_dist_after.zipf_param}")
            print(f"   Status: {'✅ PASS' if abs(user_dist_after.zipf_param - 1.2) < 0.01 else '❌ FAIL'}")
            
            # Test that alignment preserved Zipf
            aligned_requests = [user_dist_after.generate_content_name() for _ in range(1000)]
            aligned_counts = Counter(aligned_requests)
            aligned_sorted = sorted(aligned_counts.items(), key=lambda x: x[1], reverse=True)
            aligned_top_20 = int(len(aligned_sorted) * 0.2)
            aligned_top_20_requests = sum(count for _, count in aligned_sorted[:aligned_top_20])
            aligned_total = sum(aligned_counts.values())
            aligned_top_20_pct = (aligned_top_20_requests / aligned_total) * 100
            
            print(f"   Top 20% after alignment: {aligned_top_20_pct:.1f}% of requests")
            print(f"   Status: {'✅ PASS' if aligned_top_20_pct > 60 else '⚠️  WEAK'}")
        
        runtime.shutdown()
        print("\n✅ Network creation test completed")
        
    except Exception as e:
        print(f"\n❌ Network creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Verification Complete")
    print("="*80)

if __name__ == '__main__':
    verify_config()

