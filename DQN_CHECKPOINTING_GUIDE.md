# DQN Checkpointing Guide - Research Best Practices

## Overview

This guide explains the checkpointing system implemented for DQN agents, following research best practices for reproducibility and experiment management.

## What Gets Saved

### 1. **Periodic Checkpoints** (`checkpoint_round_N.pth`)
- Saved every N rounds (default: every 10 rounds)
- Contains full model state:
  - Policy network weights
  - Target network weights
  - Optimizer state
  - Learning rate scheduler state
  - Training statistics (losses, rewards)
  - Learning curve data
  - Epsilon value
  - Step count
- **Metadata included**:
  - Round number
  - Timestamp
  - Hit rate at that round
  - Experiment configuration (num_rounds, num_requests, network_size, seed)

### 2. **Best Model Checkpoint** (`best_model.pth`)
- Saved automatically whenever hit rate improves
- Always contains the best-performing model
- Same structure as periodic checkpoints
- Marked with `'best': True` in metadata

### 3. **Final Checkpoint** (`final_model.pth`)
- Saved at the end of training (always)
- Final state regardless of performance
- Marked with `'final': True` in metadata

## Storage Structure

```
dqn_checkpoints/
└── YYYYMMDD_HHMMSS/          # Experiment ID (timestamp)
    ├── router_0/
    │   ├── checkpoint_round_10.pth
    │   ├── checkpoint_round_20.pth
    │   ├── checkpoint_round_30.pth
    │   ├── best_model.pth
    │   └── final_model.pth
    ├── router_1/
    │   └── ...
    └── router_N/
        └── ...
```

## Automatic Cleanup

- **Keeps last N checkpoints** (default: 5)
- Older periodic checkpoints are automatically deleted
- **Best model and final model are NEVER deleted**

## Configuration

### Environment Variables

```bash
# Checkpoint frequency (every N rounds)
export DQN_CHECKPOINT_FREQUENCY=10

# Number of checkpoints to keep
export DQN_KEEP_CHECKPOINTS=5
```

### Programmatic Configuration

```python
agent.set_checkpoint_config(
    checkpoint_dir="path/to/checkpoints",
    frequency=10,      # Save every 10 rounds
    keep_last=5        # Keep last 5 checkpoints
)
```

## Loading Checkpoints

### Load Any Checkpoint

```python
from dqn_agent import DQNAgent

# Create agent (same architecture)
agent = DQNAgent(state_dim=5, action_dim=2)

# Load checkpoint
metadata = agent.load_model("path/to/checkpoint.pth")

# Access metadata
print(f"Round: {metadata['round']}")
print(f"Hit rate: {metadata['hit_rate']}")
print(f"Timestamp: {metadata['timestamp']}")
```

### Resume Training

```python
# Load checkpoint
agent.load_model("checkpoint_round_50.pth")

# Continue training from round 51
# (The agent maintains all state: optimizer, scheduler, etc.)
```

## Research Best Practices Implemented

### ✅ 1. **Reproducibility**
- All checkpoints include experiment metadata (seed, hyperparameters, network config)
- Timestamps for experiment tracking
- Full training state saved (not just weights)

### ✅ 2. **Best Model Tracking**
- Automatically saves best-performing model
- Separate from periodic checkpoints
- Never deleted

### ✅ 3. **Storage Management**
- Automatic cleanup of old checkpoints
- Configurable retention policy
- Prevents disk space issues

### ✅ 4. **Resume Capability**
- Can resume from any checkpoint
- Maintains optimizer state (important for learning rate schedules)
- Maintains replay buffer state

### ✅ 5. **Experiment Organization**
- Timestamped experiment directories
- Per-router checkpoint directories
- Easy to compare experiments

## Usage Examples

### Example 1: Basic Usage (Automatic)

Just run your simulation with DQN enabled:

```bash
NDN_SIM_USE_DQN=1 python main.py
```

Checkpoints are automatically saved to `dqn_checkpoints/YYYYMMDD_HHMMSS/router_N/`

### Example 2: Custom Checkpoint Frequency

```bash
DQN_CHECKPOINT_FREQUENCY=5 DQN_KEEP_CHECKPOINTS=10 NDN_SIM_USE_DQN=1 python main.py
```

Saves every 5 rounds, keeps last 10 checkpoints.

### Example 3: Load Best Model for Evaluation

```python
from dqn_agent import DQNAgent

agent = DQNAgent(state_dim=5, action_dim=2)
metadata = agent.load_model("dqn_checkpoints/20250101_120000/router_0/best_model.pth")

print(f"Best hit rate: {metadata['hit_rate']}")
print(f"Achieved at round: {metadata['round']}")

# Use agent for inference
agent.policy_net.eval()  # Set to evaluation mode
```

### Example 4: Resume Training

```python
# Load checkpoint
agent = DQNAgent(state_dim=5, action_dim=2)
metadata = agent.load_model("checkpoint_round_50.pth")

# Continue training from round 51
# (simulation code handles this automatically)
```

## What's NOT Saved (By Design)

1. **Replay Buffer**: Not saved (too large, regenerated during training)
2. **Target Network**: Saved, but it's just a copy of policy network
3. **Old Configurations**: Only last N checkpoints kept (storage management)

## Recommendations for Research

### For Short Experiments (< 100 rounds)
- Use `DQN_CHECKPOINT_FREQUENCY=10`
- Keep all checkpoints: `DQN_KEEP_CHECKPOINTS=10`

### For Long Experiments (> 100 rounds)
- Use `DQN_CHECKPOINT_FREQUENCY=20` or `50`
- Keep fewer: `DQN_KEEP_CHECKPOINTS=5`

### For Production/Deployment
- Load `best_model.pth` (best performance)
- Or `final_model.pth` (most recent)

### For Analysis
- Load multiple checkpoints to analyze learning curves
- Compare checkpoints from different rounds
- Use metadata to understand training progression

## Troubleshooting

### Checkpoints Not Saving?

1. **Check if DQN is enabled**: `NDN_SIM_USE_DQN=1`
2. **Check directory permissions**: `dqn_checkpoints/` must be writable
3. **Check logs**: Look for checkpoint-related messages

### Out of Disk Space?

1. **Reduce checkpoint frequency**: `DQN_CHECKPOINT_FREQUENCY=20`
2. **Keep fewer checkpoints**: `DQN_KEEP_CHECKPOINTS=3`
3. **Manually delete old experiments**: Remove old `YYYYMMDD_HHMMSS/` directories

### Can't Load Checkpoint?

1. **Check architecture matches**: Same `state_dim` and `action_dim`
2. **Check file exists**: Verify path is correct
3. **Check PyTorch version**: Ensure compatible versions

## Summary

This checkpointing system follows research best practices:

- ✅ **Periodic checkpoints** for resume capability
- ✅ **Best model tracking** for deployment
- ✅ **Final checkpoint** for reproducibility
- ✅ **Metadata inclusion** for experiment tracking
- ✅ **Automatic cleanup** for storage management
- ✅ **Per-router organization** for multi-agent setups

All checkpoints are self-contained and include everything needed to resume training or evaluate the model.

