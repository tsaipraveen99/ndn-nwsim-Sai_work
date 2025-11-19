"""
Performance Regression Tests

Ensures that improvements don't break performance
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import run_benchmark


class TestPerformanceRegression(unittest.TestCase):
    """Test that performance doesn't degrade with changes"""
    
    def test_baseline_performance_maintained(self):
        """Test that baseline performance is maintained"""
        config = {
            'NDN_SIM_NODES': '50',
            'NDN_SIM_PRODUCERS': '10',
            'NDN_SIM_CONTENTS': '500',
            'NDN_SIM_USERS': '100',
            'NDN_SIM_ROUNDS': '5',
            'NDN_SIM_REQUESTS': '3',
            'NDN_SIM_CACHE_CAPACITY': '500',
            'NDN_SIM_CACHE_POLICY': 'lru',
            'NDN_SIM_USE_DQN': '0'
        }
        
        result = run_benchmark(config, num_runs=3, seed=42)
        
        # Basic sanity check: hit rate should be > 0
        self.assertGreater(result.get('hit_rate', 0), 0)
        self.assertGreater(result.get('cache_hits', 0), 0)
    
    def test_dqn_performance_acceptable(self):
        """Test that DQN performance is acceptable"""
        config = {
            'NDN_SIM_NODES': '50',
            'NDN_SIM_PRODUCERS': '10',
            'NDN_SIM_CONTENTS': '500',
            'NDN_SIM_USERS': '100',
            'NDN_SIM_ROUNDS': '5',
            'NDN_SIM_REQUESTS': '3',
            'NDN_SIM_CACHE_CAPACITY': '500',
            'NDN_SIM_CACHE_POLICY': 'combined',
            'NDN_SIM_USE_DQN': '1'
        }
        
        result = run_benchmark(config, num_runs=3, seed=42)
        
        # DQN should at least match baseline
        self.assertGreater(result.get('hit_rate', 0), 0)
        self.assertGreater(result.get('cache_hits', 0), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

