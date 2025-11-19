#!/usr/bin/env python3
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, defaultdict
import torch.nn.functional as F
from typing import List, Dict, Tuple

# Check if TensorBoard is available, but make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
    USE_TENSORBOARD = True
except ImportError:
    USE_TENSORBOARD = False
    print("TensorBoard not available, will not log training metrics")

def get_device():
    """Get appropriate device for training"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Use Metal Performance Shaders on Mac
    return torch.device('cpu')


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super(DQNNetwork, self).__init__()
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, error: float):
        priority = (abs(error) + 1e-5) ** self.alpha
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
        
    def sample(self, batch_size: int) -> Tuple:
        total = len(self.memory)
        if total < batch_size:
            # If not enough samples, return all with equal weights
            indices = list(range(total))
            samples = list(self.memory)
            weights = np.ones(total)
            return samples, indices, weights
            
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(total, batch_size, p=probs, replace=False)
        samples = [self.memory[idx] for idx in indices]
        # Importance sampling weights
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, indices, weights
        
    def update_priorities(self, indices: List[int], errors: np.ndarray):
        for idx, error in zip(indices, errors):
            if idx < len(self.priorities):  # Safeguard against out of bounds
                self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
            
    def __len__(self) -> int:
        return len(self.memory)

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_update_freq: int = 100,
        n_step: int = 20
    ):
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.step_count = 0
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize RL components
        self.memory = PrioritizedReplayBuffer(memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min')
        
        # Initialize TensorBoard if available
        if USE_TENSORBOARD:
            self.writer = SummaryWriter()
        else:
            self.writer = None
            
        self.training_stats = {'losses': [], 'rewards': [], 'cache_hits': 0, 'cache_misses': 0}
        self.best_reward = -float('inf')
        self.best_hit_rate = -float('inf')
        
        # Learning curve tracking: per-round metrics
        self.learning_curve: Dict[int, Dict] = {}  # {round: {hit_rate, loss, reward, epsilon, cache_decisions}}
        self.no_improvement_count = 0
        
        # Checkpoint management for research
        self.checkpoint_dir = None
        self.checkpoint_frequency = 10  # Save every N rounds
        self.keep_checkpoints = 5  # Keep last N checkpoints
        self.best_model_path = None
        self.last_checkpoint_round = 0
        
    def get_state_features(self, state: np.ndarray) -> np.ndarray:
        """Extract and normalize relevant features from the state"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        # Normalize using robust statistics
        state = (state - np.mean(state, axis=1, keepdims=True)) / (np.std(state, axis=1, keepdims=True) + 1e-8)
        return state.squeeze()
        
    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy with adaptive exploration"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            state = self.get_state_features(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # Handle batch norm during evaluation
            self.policy_net.eval()
            q_values = self.policy_net(state_tensor)
            self.policy_net.train()
            return q_values.argmax().item()
            
    def update_epsilon(self, reward: float):
        """Adaptively update epsilon based on performance"""
        if reward > self.best_reward:
            self.best_reward = reward
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
        # Slow down epsilon decay if performance is improving
        decay_rate = self.epsilon_decay if self.no_improvement_count < 5 else self.epsilon_decay * 0.95
        self.epsilon = max(self.epsilon_end, self.epsilon * decay_rate)
        
    def calculate_reward(self, is_cache_hit: bool = False, is_caching_decision: bool = False, 
                         content_size: int = 0, cluster_score: float = 0.0, was_cached: bool = False,
                         access_frequency: float = 0.0, latency_saved: float = 0.0, 
                         bandwidth_saved: float = 0.0) -> float:
        """
        Phase 6: Multi-objective reward function
        
        Calculate reward based on caching decision or cache hit.
        Optimizes for: hit rate, latency reduction, bandwidth savings.
        
        Args:
            is_cache_hit: True if this is a delayed reward for an actual cache hit
            is_caching_decision: True if this is an immediate reward for caching decision
            content_size: Size of the content (for size penalty)
            cluster_score: Popularity score of content's cluster
            was_cached: True if content was previously cached (for cache miss penalty)
            access_frequency: Normalized access frequency (0-1)
            latency_saved: Latency saved in seconds (for multi-objective optimization)
            bandwidth_saved: Bandwidth saved in bytes (for multi-objective optimization)
        
        Returns:
            Reward value (scaled appropriately)
        """
        if is_cache_hit:
            # Phase 6: Multi-objective reward
            # Delayed reward: Large positive reward for actual cache hit
            base_reward = 15.0  # Hit rate objective
            cluster_bonus = 1.0 * cluster_score
            frequency_bonus = 2.0 * access_frequency
            size_penalty = -0.1 * (content_size / 100)
            
            # Phase 6.1: Multi-objective components
            latency_reward = latency_saved * 0.1  # Reward for reducing latency (0.1 per second saved)
            bandwidth_reward = bandwidth_saved * 0.0001  # Reward for saving bandwidth (0.0001 per byte, scales to ~0.1 per KB)
            
            return base_reward + cluster_bonus + frequency_bonus + size_penalty + latency_reward + bandwidth_reward
        elif is_caching_decision:
            # NO IMMEDIATE REWARD - Only reward outcomes (cache hits)
            # This prevents the "bias loop" where agent caches everything for +0.3
            return 0.0
        elif was_cached:
            # Cache miss when we cached it: Medium negative
            # Penalizes caching content that doesn't get hit
            return -2.0  # Increased penalty from -1.0
        else:
            # Decision not to cache: Small negative (or zero if content is unpopular)
            # Encourages exploration but not too strongly
            # If content is popular, penalize not caching more
            if cluster_score > 0.5 or access_frequency > 0.3:
                return -0.5  # Penalize not caching popular content
            return -0.1  # Small penalty for not caching unpopular content
        
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer with priority"""
        with torch.no_grad():
            state = self.get_state_features(state)
            next_state = self.get_state_features(next_state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Handle batch norm during evaluation
            self.policy_net.eval()
            current_q = self.policy_net(state_tensor)[0][action]
            self.policy_net.train()
            
            # Use target network for next state Q-values
            self.target_net.eval()
            next_q = self.target_net(next_state_tensor).max(1)[0]
            self.target_net.train()
            
            expected_q = reward + (1 - done) * self.gamma * next_q
            error = abs(current_q - expected_q).item()
        
        # Add to replay buffer
        self.memory.push(state, action, reward, next_state, done, error)
        
    def replay(self) -> float:
        """Train the network using prioritized experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample batch with priorities
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values with N-Step Double DQN
        with torch.no_grad():
            if self.n_step > 1:
                # N-step approximation: scale rewards and use longer horizon
                effective_gamma = self.gamma ** self.n_step  # Discount over N steps
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                # Scale rewards to account for N-step horizon
                scaled_rewards = rewards.unsqueeze(1) * (1 + (self.n_step - 1) * 0.1)
                expected_q_values = scaled_rewards + (1 - dones.unsqueeze(1)) * effective_gamma * next_q_values
            else:
                # Standard 1-step
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                expected_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss with importance sampling weights
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(
            current_q_values,
            expected_q_values,
            reduction='none'
        )).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Update priorities in replay buffer
        errors = abs(current_q_values - expected_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, errors.squeeze())
        
        # Update epsilon and learning rate
        self.update_epsilon(rewards.mean().item())
        self.scheduler.step(loss)
        
        # Log metrics
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', loss.item(), self.step_count)
            self.writer.add_scalar('Reward/train', rewards.mean().item(), self.step_count)
            self.writer.add_scalar('Epsilon', self.epsilon, self.step_count)
        
        # Track statistics
        loss_value = loss.item()
        self.training_stats['losses'].append(loss_value)
        self.training_stats['rewards'].append(rewards.mean().item())
        return loss_value
        
    def get_statistics(self) -> Dict:
        """Return agent's performance statistics"""
        return {
            'avg_loss': np.mean(self.training_stats['losses']) if self.training_stats['losses'] else 0,
            'avg_reward': np.mean(self.training_stats['rewards']) if self.training_stats['rewards'] else 0,
            'cache_hit_rate': (
                self.training_stats['cache_hits'] /
                (self.training_stats['cache_hits'] + self.training_stats['cache_misses'] + 1e-8)
            ),
            'epsilon': self.epsilon,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def record_round_metrics(self, round_num: int, hit_rate: float, cache_decisions: int = 0):
        """
        Record metrics for a specific round (for learning curve analysis)
        
        Args:
            round_num: Round number
            hit_rate: Cache hit rate for this round
            cache_decisions: Number of cache decisions made this round
        """
        avg_loss = np.mean(self.training_stats['losses'][-100:]) if self.training_stats['losses'] else 0.0
        avg_reward = np.mean(self.training_stats['rewards'][-100:]) if self.training_stats['rewards'] else 0.0
        
        self.learning_curve[round_num] = {
            'hit_rate': hit_rate,
            'loss': avg_loss,
            'reward': avg_reward,
            'epsilon': self.epsilon,
            'cache_decisions': cache_decisions,
            'training_steps': self.step_count
        }
        
        # Update best hit rate for best model tracking
        if hit_rate > self.best_hit_rate:
            self.best_hit_rate = hit_rate
    
    def save_checkpoint_if_needed(self, round_num: int, hit_rate: float, experiment_metadata: Dict = None):
        """
        Save checkpoint if it's time (based on frequency) and manage checkpoint cleanup.
        Also saves best model if performance improved.
        
        Args:
            round_num: Current round number
            hit_rate: Current hit rate (for best model tracking)
            experiment_metadata: Optional metadata dict (hyperparams, seed, timestamp, etc.)
        """
        if self.checkpoint_dir is None:
            return
            
        import os
        import glob
        from datetime import datetime
        
        # Periodic checkpoint
        if round_num % self.checkpoint_frequency == 0 and round_num > self.last_checkpoint_round:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_round_{round_num}.pth')
            
            # Add timestamp to metadata
            metadata = experiment_metadata.copy() if experiment_metadata else {}
            metadata.update({
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'hit_rate': hit_rate,
                'epsilon': self.epsilon,
                'step_count': self.step_count
            })
            
            self.save_model(checkpoint_path, metadata=metadata)
            self.last_checkpoint_round = round_num
            
            # Clean up old checkpoints (keep only last N)
            self._cleanup_old_checkpoints()
        
        # Best model checkpoint (save whenever hit rate improves)
        if hit_rate > self.best_hit_rate:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            metadata = experiment_metadata.copy() if experiment_metadata else {}
            metadata.update({
                'round': round_num,
                'timestamp': datetime.now().isoformat(),
                'hit_rate': hit_rate,
                'best': True,
                'epsilon': self.epsilon,
                'step_count': self.step_count
            })
            self.save_model(best_path, metadata=metadata)
            self.best_model_path = best_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old periodic checkpoints, keeping only the last N"""
        if self.checkpoint_dir is None:
            return
            
        import os
        import glob
        import re
        
        # Find all periodic checkpoint files
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_round_*.pth')
        checkpoints = glob.glob(pattern)
        
        # Sort by round number (extract from filename)
        def get_round_num(path):
            match = re.search(r'round_(\d+)', path)
            return int(match.group(1)) if match else 0
        
        checkpoints.sort(key=get_round_num, reverse=True)
        
        # Keep only the last N checkpoints
        for old_checkpoint in checkpoints[self.keep_checkpoints:]:
            try:
                os.remove(old_checkpoint)
            except Exception as e:
                print(f"Warning: Could not remove old checkpoint {old_checkpoint}: {e}")
    
    def save_final_checkpoint(self, round_num: int, hit_rate: float, experiment_metadata: Dict = None):
        """
        Save final checkpoint at end of training (always called, regardless of frequency)
        
        Args:
            round_num: Final round number
            hit_rate: Final hit rate
            experiment_metadata: Optional metadata dict
        """
        if self.checkpoint_dir is None:
            return
            
        import os
        from datetime import datetime
        
        final_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        metadata = experiment_metadata.copy() if experiment_metadata else {}
        metadata.update({
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'hit_rate': hit_rate,
            'final': True,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        })
        self.save_model(final_path, metadata=metadata)
    
    def get_learning_curve(self) -> Dict:
        """Get learning curve data for analysis"""
        return self.learning_curve.copy()
    
    def export_learning_curve(self, filepath: str):
        """Export learning curve to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.learning_curve, f, indent=2)
        
    def set_checkpoint_config(self, checkpoint_dir: str, frequency: int = 10, keep_last: int = 5):
        """
        Configure checkpoint saving for research experiments
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            frequency: Save checkpoint every N rounds (default: 10)
            keep_last: Keep last N checkpoints, delete older ones (default: 5)
        """
        import os
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = frequency
        self.keep_checkpoints = keep_last
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_model(self, path: str, metadata: Dict = None):
        """
        Save model parameters and training state with optional metadata
        
        Args:
            path: Path to save checkpoint
            metadata: Optional dict with experiment metadata (hyperparams, seed, etc.)
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'training_stats': self.training_stats,
            'learning_curve': self.learning_curve,
            'best_reward': self.best_reward,
            'best_hit_rate': self.best_hit_rate,
        }
        
        # Add metadata if provided (for reproducibility)
        if metadata:
            checkpoint['metadata'] = metadata
            
        torch.save(checkpoint, path)
        
    def load_model(self, path: str):
        """Load model parameters and training state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.training_stats = checkpoint['training_stats']
        
        # Load additional state if available
        if 'learning_curve' in checkpoint:
            self.learning_curve = checkpoint['learning_curve']
        if 'best_reward' in checkpoint:
            self.best_reward = checkpoint['best_reward']
        if 'best_hit_rate' in checkpoint:
            self.best_hit_rate = checkpoint['best_hit_rate']
            
        return checkpoint.get('metadata', {})
        
    def close(self):
        """Clean up resources"""
        if self.writer is not None:
            self.writer.close()