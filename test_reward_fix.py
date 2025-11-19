#!/usr/bin/env python3
"""Test that immediate cache rewards are removed"""

import sys
sys.path.insert(0, '.')

from dqn_agent import DQNAgent

def test_reward_function():
    """Test that immediate cache reward returns 0.0"""
    agent = DQNAgent(state_dim=5, action_dim=2)
    
    # Test immediate cache decision reward
    reward = agent.calculate_reward(is_caching_decision=True)
    print(f"Immediate cache reward: {reward}")
    
    if reward != 0.0:
        print("❌ FAIL: Immediate cache reward should be 0.0")
        print(f"   Got: {reward}")
        return False
    else:
        print("✅ PASS: Immediate cache reward is 0.0")
    
    # Test that cache hit reward still works
    cache_hit_reward = agent.calculate_reward(is_cache_hit=True)
    print(f"Cache hit reward: {cache_hit_reward}")
    
    if cache_hit_reward <= 0:
        print("❌ FAIL: Cache hit reward should be positive")
        print(f"   Got: {cache_hit_reward}")
        return False
    else:
        print(f"✅ PASS: Cache hit reward is positive: {cache_hit_reward}")
    
    return True

if __name__ == '__main__':
    success = test_reward_function()
    sys.exit(0 if success else 1)

