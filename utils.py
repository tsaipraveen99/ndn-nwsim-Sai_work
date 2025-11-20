import threading
import numpy as np
import logging
import warnings
import mmh3
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Set, List, Optional, Deque
from datetime import datetime
import random
from collections import deque

warnings.filterwarnings("ignore")

logger = logging.getLogger('utils_logger')

# Simplified NDNDistribution for compatibility with both main.py and run.py
class NDNDistribution:
    """Generates content names following a Zipf-like distribution for NDN simulation"""
    
    def __init__(self, num_contents: int, zipf_param: float = 0.8):
        """
        Initialize content name distribution
        
        Args:
            num_contents: Total number of available contents
            zipf_param: Zipf distribution parameter (controls popularity skew)
        """
        self.num_contents = num_contents
        self.zipf_param = zipf_param
        self.content_list = self.generate_content_names()
        self.probabilities = self._zipf_distribution()
        
        logger.info(f"Initialized NDN distribution with {num_contents} contents")
        
    def generate_content_names(self) -> List[str]:
        """Generate hierarchical content names"""
        content_names = []
        organizations = ['ucla', 'mit', 'stanford', 'berkeley', 'oxford']
        departments = ['cs', 'ee', 'math', 'physics', 'biology']
        content_types = ['research', 'courses', 'projects', 'data', 'media']
        
        for i in range(self.num_contents):
            org = random.choice(organizations)
            dept = random.choice(departments)
            content_type = random.choice(content_types)
            name = f"/edu/{org}/{dept}/{content_type}/content_{i:03d}"
            content_names.append(name)
            
        return content_names
        
    def _zipf_distribution(self) -> np.ndarray:
        """Generate Zipf-like probability distribution for content popularity"""
        ranks = np.arange(1, self.num_contents + 1, dtype=float)
        probs = 1.0 / np.power(ranks, self.zipf_param)
        return probs / probs.sum()
        
    def generate_content_name(self) -> str:
        """Get a random content name based on popularity distribution"""
        return np.random.choice(self.content_list, p=self.probabilities)
        
    def get_popularity(self, content_name: str) -> float:
        """Get popularity score for a content name"""
        try:
            idx = self.content_list.index(content_name)
            return self.probabilities[idx]
        except ValueError:
            return 0.0


# For compatibility with run.py
class ZipfDistribution(NDNDistribution):
    def __init__(self, num_contents, a=0.8):
        super().__init__(num_contents, a)


class ClusterManager:
    def __init__(self, min_cluster_size: int = 2):
        self.min_cluster_size = min_cluster_size
        self.clusters: Dict[int, Set[str]] = {}
        self.content_to_cluster: Dict[str, int] = {}
        self.cluster_scores: Dict[int, float] = {}
        self.cluster_last_access: Dict[int, float] = {}
        self.lock = threading.Lock()
        
    def update_cluster_score(self, cluster_id: int, score_increment: float, current_time: float):
        with self.lock:
            if cluster_id not in self.cluster_scores:
                self.cluster_scores[cluster_id] = 0.0
                self.cluster_last_access[cluster_id] = current_time
                
            time_diff = current_time - self.cluster_last_access.get(cluster_id, 0)
            decay_factor = 0.95 ** max(0, time_diff)  # Ensure non-negative
            self.cluster_scores[cluster_id] = self.cluster_scores[cluster_id] * decay_factor + score_increment
            self.cluster_last_access[cluster_id] = current_time
            
    def get_cluster_score(self, cluster_id: int) -> float:
        with self.lock:
            return self.cluster_scores.get(cluster_id, 0.0)
            
    def get_least_popular_cluster(self) -> int:
        with self.lock:
            if not self.cluster_scores:
                return -1
            return min(self.cluster_scores.items(), key=lambda x: x[1])[0]


class BloomFilter:
    def __init__(self, size: int = 1000, hash_count: int = 4):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = np.zeros(size, dtype=bool)
        self.lock = threading.Lock()
        
    def add(self, item: str):
        with self.lock:
            for seed in range(self.hash_count):
                idx = mmh3.hash(str(item), seed) % self.size
                self.bit_array[idx] = True
                
    def __contains__(self, item: str) -> bool:
        with self.lock:
            return all(
                self.bit_array[mmh3.hash(str(item), seed) % self.size]
                for seed in range(self.hash_count)
            )
            
    def merge(self, other: 'BloomFilter'):
        with self.lock:
            if self.size != other.size:
                raise ValueError("Bloom filters must have the same size")
            self.bit_array |= other.bit_array
    
    def check(self, item: str) -> bool:
        """Check if an item might be in the Bloom filter"""
        return item in self


class NeuralBloomFilter(BloomFilter):
    """
    Task 2.3: Neural Bloom Filter - Enhanced Bloom filter with neural network
    to reduce false positive rates and improve cache state summarization
    
    Uses a small neural network to learn patterns in false positives and
    adjust the filter behavior accordingly.
    """
    def __init__(self, size: int = 2000, hash_count: int = 4, use_neural: bool = True):
        super().__init__(size, hash_count)
        self.use_neural = use_neural
        self.false_positive_history = []  # Track false positives for learning
        self.true_positive_count = 0
        self.false_positive_count = 0
        
        # Task 2.3: Neural network for false positive reduction
        if self.use_neural:
            try:
                import torch
                import torch.nn as nn
                self.torch_available = True
                self._init_neural_network()
            except ImportError:
                self.torch_available = False
                logger.warning("PyTorch not available for NeuralBloomFilter, using basic Bloom filter")
                self.use_neural = False
        else:
            self.torch_available = False
    
    def _init_neural_network(self):
        """Initialize neural network for false positive prediction"""
        try:
            import torch
            import torch.nn as nn
            
            # Small neural network to predict if a Bloom filter check is likely a false positive
            # Input: bit pattern around hash indices, output: probability of false positive
            self.neural_model = nn.Sequential(
                nn.Linear(self.hash_count * 2, 32),  # Input: hash indices and surrounding bits
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),  # Output: false positive probability
                nn.Sigmoid()
            )
            
            # Simple optimizer
            self.optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
            self.criterion = nn.BCELoss()
            
            # Training data buffer
            self.training_buffer = []
            self.buffer_size = 100
            
            logger.debug("Neural Bloom filter network initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize neural network for Bloom filter: {e}")
            self.use_neural = False
    
    def _get_bit_pattern(self, item: str) -> np.ndarray:
        """Extract bit pattern around hash indices for neural network input"""
        pattern = np.zeros(self.hash_count * 2, dtype=np.float32)
        for i in range(self.hash_count):
            idx = mmh3.hash(str(item), i) % self.size
            # Get the bit at index and a neighboring bit
            pattern[i * 2] = float(self.bit_array[idx])
            pattern[i * 2 + 1] = float(self.bit_array[(idx + 1) % self.size])
        return pattern
    
    def check(self, item: str, verify_callback=None) -> bool:
        """
        Enhanced check with neural network false positive reduction
        
        Args:
            item: Item to check
            verify_callback: Optional callback to verify if item actually exists
                            (for training the neural network)
        
        Returns:
            True if item might be in filter (with reduced false positive rate)
        """
        # First do basic Bloom filter check
        basic_result = super().check(item)
        
        if not basic_result:
            return False  # Definitely not in filter
        
        # If basic check passes, use neural network to estimate false positive probability
        if self.use_neural and self.torch_available:
            try:
                import torch
                
                # Get bit pattern
                pattern = self._get_bit_pattern(item)
                pattern_tensor = torch.FloatTensor(pattern).unsqueeze(0)
                
                # Get neural network prediction
                with torch.no_grad():
                    false_positive_prob = self.neural_model(pattern_tensor).item()
                
                # If neural network predicts high false positive probability, be more conservative
                # Adjust threshold based on false positive rate
                threshold = 0.5
                if self.false_positive_count + self.true_positive_count > 0:
                    current_fpr = self.false_positive_count / (self.false_positive_count + self.true_positive_count)
                    threshold = max(0.3, min(0.7, 0.5 + current_fpr))
                
                # If neural network suggests high false positive probability, return False
                if false_positive_prob > threshold:
                    return False
                
                # If we have a verification callback, use it to train the network
                if verify_callback is not None:
                    actually_exists = verify_callback(item)
                    if not actually_exists:
                        # False positive - learn from this
                        self.false_positive_count += 1
                        self._train_on_false_positive(pattern, True)
                    else:
                        # True positive
                        self.true_positive_count += 1
                        self._train_on_false_positive(pattern, False)
                
                return True
            except Exception as e:
                logger.debug(f"Neural Bloom filter check failed: {e}, falling back to basic")
                return basic_result
        
        return basic_result
    
    def _train_on_false_positive(self, pattern: np.ndarray, is_false_positive: bool):
        """Train neural network on false positive example"""
        if not self.use_neural or not self.torch_available:
            return
        
        try:
            import torch
            
            # Add to training buffer
            target = 1.0 if is_false_positive else 0.0
            self.training_buffer.append((pattern, target))
            
            # Keep buffer size manageable
            if len(self.training_buffer) > self.buffer_size:
                self.training_buffer.pop(0)
            
            # Train periodically (every 10 examples)
            if len(self.training_buffer) >= 10 and len(self.training_buffer) % 10 == 0:
                # Sample a batch
                batch_size = min(10, len(self.training_buffer))
                batch = self.training_buffer[-batch_size:]
                
                patterns = torch.FloatTensor([p[0] for p in batch])
                targets = torch.FloatTensor([p[1] for p in batch]).unsqueeze(1)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.neural_model(patterns)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                logger.debug(f"Neural Bloom filter trained, loss: {loss.item():.4f}")
        except Exception as e:
            logger.debug(f"Neural Bloom filter training failed: {e}")
    
    def get_false_positive_rate(self) -> float:
        """Get current estimated false positive rate"""
        total = self.false_positive_count + self.true_positive_count
        if total == 0:
            return 0.0
        return self.false_positive_count / total
    
    def reset_stats(self):
        """Reset false positive statistics"""
        self.false_positive_count = 0
        self.true_positive_count = 0
        self.training_buffer.clear()


class ContentStore:
    def __init__(self, total_capacity: int, router_id: int):
        self.store: Dict[str, Any] = {}
        self.size_map: Dict[str, int] = {}
        self.router_id = router_id
        self.remaining_capacity = total_capacity
        self.total_capacity = total_capacity
        self.store_order: Deque[str] = deque()
        self.insertions = 0
        
        # Simplified semantic clustering components
        self.cluster_manager = ClusterManager()
        self.content_embeddings: Dict[str, np.ndarray] = {}
        
        # Task 2.2: Enhanced semantic encoding with neural networks
        try:
            from semantic_encoder import get_semantic_encoder
            self.semantic_encoder = get_semantic_encoder(embedding_dim=64, use_cnn=True)
            logger.info(f"ContentStore {router_id}: Using CNN-based semantic encoder")
        except Exception as e:
            logger.warning(f"ContentStore {router_id}: Failed to load semantic encoder: {e}, using hash-based")
            self.semantic_encoder = None
        
        # Thread safety
        self.store_lock = threading.Lock()
        self.embedding_lock = threading.Lock()
        
        # Task 2.3: Use Neural Bloom Filter for better cache state summarization
        # Phase 3.2: Adaptive Bloom filter sizing based on cache capacity and false positive tolerance
        try:
            use_neural_bloom = os.environ.get("NDN_SIM_NEURAL_BLOOM", "0") == "1"
            
            # Phase 3.2: Calculate optimal Bloom filter size
            # Optimal size: m = -n * ln(p) / (ln(2)^2)
            # where n = expected cache size, p = desired false positive rate
            desired_fpr = float(os.environ.get("NDN_SIM_BLOOM_FPR", "0.01"))  # Default 1% FPR
            expected_cache_size = max(10, total_capacity)  # Use cache capacity as expected size
            
            # Calculate optimal size
            import math
            if expected_cache_size > 0 and desired_fpr > 0:
                optimal_size = int(-expected_cache_size * math.log(desired_fpr) / (math.log(2) ** 2))
                # Round to nearest 100 for efficiency
                optimal_size = ((optimal_size + 50) // 100) * 100
                # Clamp to reasonable range [500, 10000]
                optimal_size = max(500, min(10000, optimal_size))
                
                # Calculate optimal hash count: k = (m/n) * ln(2)
                optimal_hash_count = max(2, min(8, int((optimal_size / max(1, expected_cache_size)) * math.log(2))))
            else:
                # Fallback to defaults
                optimal_size = 2000
                optimal_hash_count = 4
            
            if use_neural_bloom:
                self.bloom_filter = NeuralBloomFilter(size=optimal_size, hash_count=optimal_hash_count, use_neural=True)
                logger.info(f"ContentStore {router_id}: Using Neural Bloom Filter (size={optimal_size}, hash_count={optimal_hash_count}, FPR={desired_fpr:.3f})")
            else:
                self.bloom_filter = BloomFilter(size=optimal_size, hash_count=optimal_hash_count)
                logger.debug(f"ContentStore {router_id}: Using Bloom Filter (size={optimal_size}, hash_count={optimal_hash_count}, FPR={desired_fpr:.3f})")
        except Exception as e:
            logger.warning(f"ContentStore {router_id}: Failed to initialize Bloom Filter with adaptive sizing: {e}, using defaults")
            self.bloom_filter = BloomFilter(size=2000, hash_count=4)
        
        self.neighbor_filters: Dict[int, BloomFilter] = {}
        
        # Task 2.4: Store references to router and graph for enhanced DQN state
        self.router_ref = None  # Will be set by router after initialization
        self.graph_ref = None  # Will be set by router after initialization
        
        # Access tracking
        self.access_count: Dict[str, int] = {}
        self.last_access_time: Dict[str, float] = {}
        
        # DQN tracking: track which contents were cached by DQN decisions
        self.dqn_cached_contents: Set[str] = set()
        self.dqn_decision_states: Dict[str, np.ndarray] = {}  # Track state when caching decision was made
        
        # DQN training frequency control
        self.dqn_training_step = 0
        # Allow training frequency to be configured via environment variable
        self.dqn_training_frequency = int(os.environ.get('DQN_TRAINING_FREQUENCY', '10'))  # Train every N steps
        
        # Bloom filter propagation control
        self.bloom_propagation_frequency = 10  # Propagate every N cache insertions
        self.last_bloom_propagation = 0
        
        # Cache embeddings for semantic similarity (feature 15)
        self.cached_embeddings: Dict[str, np.ndarray] = {}
        
        # For compatibility with run.py
        self.mode = "basic"  # Options: "basic", "dqn_cache"
        self.status = "router"
        self.dqn_agent = None
        self.replacement_policy = "lru"  # Options: fifo, lifo, lru, basic
        
        # Initialize DQN agent if mode is set to use it
        if self.mode == "dqn_cache":
            self.initialize_dqn_agent()
        
        logger.info(f"Initialized ContentStore for router {router_id} with capacity {total_capacity}")
    
    def debug_cache_contents(self):
        """Debug method to print cache contents"""
        with self.store_lock:
            print(f"ContentStore {self.router_id} contents:")
            print(f"  Capacity: {self.total_capacity}, Used: {self.total_capacity - self.remaining_capacity}")
            print(f"  Items cached: {len(self.store)}")
            for name, content in list(self.store.items())[:5]:  # Show first 5 items
                size = self.size_map.get(name, 0)
                accesses = self.access_count.get(name, 0)
                print(f"    - {name} (Size: {size}, Accesses: {accesses})")
            if len(self.store) > 5:
                print(f"    ... and {len(self.store) - 5} more items")

    def fix_caching_policies(self):
        """Fix and improve caching policies"""
        # First, make sure we have room for at least a few items
        with self.store_lock:
            # Check if we need to clear some space
            if self.remaining_capacity < (self.total_capacity * 0.2):
                # Remove least accessed items to free up 30% of capacity
                target_space = self.total_capacity * 0.3
                
                # Sort by access count (ascending)
                items_to_check = sorted(
                    self.store.keys(),
                    key=lambda x: self.access_count.get(x, 0)
                )
                
                space_freed = 0
                for name in items_to_check:
                    if space_freed >= target_space:
                        break
                    size = self.size_map.get(name, 0)
                    self.remove_content(name)
                    space_freed += size
                    
                print(f"ContentStore {self.router_id}: Cleared {space_freed} units to make room")
        
        # Make sure the mode is properly set
        if hasattr(self, 'mode') and self.mode != "dqn_cache":
            self.mode = "dqn_cache"
            # Initialize DQN agent if needed
            if hasattr(self, 'initialize_dqn_agent') and self.dqn_agent is None:
                try:
                    self.initialize_dqn_agent()
                except Exception as e:
                    print(f"Could not initialize DQN agent: {e}")
    
    def initialize_dqn_agent(self):
        """Initialize the DQN agent for caching decisions"""
        try:
            # Import here to avoid circular imports
            from dqn_agent import DQNAgent
            
            # Optimized state dimensions: 5 essential features (removed Feature 3: cache utilization - redundant)
            # State includes: content features, cache features, neighbor states (Bloom filters)
            state_dim = 5
            # Action dimensions: 0 = don't cache, 1 = cache
            action_dim = 2
            
            # Create the agent with consistent hyperparameters matching DQNAgent defaults
            # Allow batch size to be configured via environment variable for GPU optimization
            batch_size = int(os.environ.get('DQN_BATCH_SIZE', '64'))
            self.dqn_agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                memory_size=max(10000, self.total_capacity * 20),  # Increased for better stability
                batch_size=batch_size,  # Configurable via DQN_BATCH_SIZE env var
                gamma=0.99,  # Match default - future reward discount factor
                epsilon_start=1.0,
                epsilon_end=0.01,  # Match default
                epsilon_decay=0.995  # Match default
            )
            
            logger.info(f"DQN agent initialized for router {self.router_id} with state_dim={state_dim}")
            return True
        except Exception as e:
            logger.error(f"Error initializing DQN agent: {e}")
            # Fall back to basic caching if DQN fails
            self.mode = "basic"
            return False
    
    def set_mode(self, mode: str):
        """Set caching policy mode"""
        if mode == self.mode:
            return
            
        old_mode = self.mode
        self.mode = mode
        if mode == "dqn_cache" and self.dqn_agent is None:
            # Task 1.3: Initialize DQN agent when switching to DQN mode
            success = self.initialize_dqn_agent()
            if not success:
                logger.warning(f"ContentStore {self.router_id}: Failed to initialize DQN agent, falling back to basic mode")
                self.mode = "basic"
            else:
                logger.info(f"ContentStore {self.router_id}: Switched from {old_mode} to DQN caching mode")
        elif mode != "dqn_cache":
            # Fall back to requested replacement policy, keep existing choice
            self.mode = "basic"
            logger.debug(f"ContentStore {self.router_id}: Using basic caching mode ({mode})")

    def set_replacement_policy(self, policy: str):
        """Configure replacement policy used in basic mode caches."""
        normalized = policy.lower()
        # Task 2.1: Add "combined" and "lfu" to supported policies
        if normalized in {"fifo", "lifo", "lru", "lfu", "basic", "combined"}:
            self.replacement_policy = normalized
            logger.info(f"ContentStore {self.router_id}: Replacement policy set to {normalized}")
        else:
            logger.warning(f"ContentStore {self.router_id}: Unknown replacement policy {policy}, keeping {self.replacement_policy}")
    
    def generate_embedding(self, name: str) -> np.ndarray:
        """
        Task 2.2: Generate semantic embedding using CNN-based encoder or hash fallback
        """
        # Use semantic encoder if available
        if hasattr(self, 'semantic_encoder') and self.semantic_encoder is not None:
            try:
                return self.semantic_encoder.encode(name)
            except Exception as e:
                logger.warning(f"Semantic encoder failed for {name}: {e}, using hash fallback")
        
        # Fallback to hash-based embedding (compatible with existing code)
        parts = name.split('/')
        embedding = np.zeros(64, dtype=np.float32)  # Match semantic encoder dimension
        for i, part in enumerate(parts[:min(len(parts), 64)]):
            # Use hash to generate a pseudo-random embedding
            hash_val = mmh3.hash(part, i) % 1000
            embedding[i % 64] = hash_val / 1000.0
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
            
    def update_clusters(self, new_content: str, current_time: float):
        """Update content clusters - simplified version"""
        with self.embedding_lock:
            # Generate embedding for new content if needed
            if new_content not in self.content_embeddings:
                self.content_embeddings[new_content] = self.generate_embedding(new_content)
            
            # Simple clustering based on content name prefixes
            parts = new_content.split('/')
            if len(parts) >= 4:
                # Use domain prefix as cluster
                domain_prefix = '/'.join(parts[:4])
                cluster_id = abs(hash(domain_prefix) % 100)  # Use hash value as cluster ID
                
                # Update cluster membership
                if cluster_id not in self.cluster_manager.clusters:
                    self.cluster_manager.clusters[cluster_id] = set()
                    
                # Add to new cluster
                self.cluster_manager.clusters[cluster_id].add(new_content)
                self.cluster_manager.content_to_cluster[new_content] = cluster_id
                
                # Update cluster score
                access_score = self.access_count.get(new_content, 0)
                self.cluster_manager.update_cluster_score(cluster_id, access_score, current_time)
            
    def evict_from_least_popular_cluster(self, required_space: int) -> bool:
        """Evict content from least popular cluster"""
        least_popular_cluster = self.cluster_manager.get_least_popular_cluster()
        if least_popular_cluster == -1:
            # Fall back to LRU if no clusters exist
            return self.evict_lru(required_space)
            
        # Get content in the least popular cluster
        cluster_contents = self.cluster_manager.clusters.get(least_popular_cluster, set())
        if not cluster_contents:
            return self.evict_lru(required_space)
            
        # Sort by access count and time
        sorted_contents = sorted(
            [c for c in cluster_contents if c in self.store],
            key=lambda x: (self.access_count.get(x, 0), -self.last_access_time.get(x, 0))
        )
        
        space_freed = 0
        for content in sorted_contents:
            if content in self.store:
                space_freed += self.size_map.get(content, 0)
                self._remove_content_no_lock(content)
                
                if space_freed >= required_space:
                    return True
                    
        return space_freed >= required_space
    
    def combined_eviction_algorithm(self, required_space: int, weight: float = 0.5, current_time: float = None) -> bool:
        """
        Task 2.1: Algorithm 1 - Combined Eviction Algorithm (recency + frequency)
        From research report: combines recency_scores and frequency_scores
        
        Args:
            required_space: Space needed to free up
            weight: Weight for recency (0.0-1.0), frequency weight = 1-weight
            current_time: Current time for recency calculation (default: now)
        
        Returns:
            True if enough space was freed, False otherwise
        """
        if not self.store:
            return False
        
        if current_time is None:
            import time
            current_time = time.time()
        
        # Calculate recency_scores (inverse of time since last access)
        recency_scores = {}
        max_time_diff = 0.0
        for name in self.store:
            last_access = self.last_access_time.get(name, 0)
            time_diff = max(0.1, current_time - last_access)  # Avoid division by zero
            recency_scores[name] = 1.0 / time_diff  # Higher score = more recent
            max_time_diff = max(max_time_diff, time_diff)
        
        # Normalize recency scores to 0-1 range
        if max_time_diff > 0:
            for name in recency_scores:
                recency_scores[name] = recency_scores[name] / (1.0 / 0.1)  # Normalize
        
        # Calculate frequency_scores (based on access_count)
        frequency_scores = {}
        max_frequency = max(self.access_count.values()) if self.access_count else 1.0
        for name in self.store:
            freq = self.access_count.get(name, 0)
            frequency_scores[name] = freq / max(1.0, max_frequency)  # Normalize to 0-1
        
        # Combine scores: combined_scores = weight * recency + (1-weight) * frequency
        # Lower combined score = less important = candidate for eviction
        combined_scores = {}
        for name in self.store:
            recency = recency_scores.get(name, 0.0)
            frequency = frequency_scores.get(name, 0.0)
            combined_scores[name] = weight * recency + (1.0 - weight) * frequency
        
        # Sort by combined score (ascending) - lowest scores first (evict these)
        sorted_contents = sorted(
            self.store.keys(),
            key=lambda x: combined_scores.get(x, 0.0)
        )
        
        # Evict content with lowest combined scores until we have enough space
        space_freed = 0
        for content_name in sorted_contents:
            if space_freed >= required_space:
                break
            content_size = self.size_map.get(content_name, 0)
            space_freed += content_size
            self._remove_content_no_lock(content_name)
            logger.debug(
                f"ContentStore {self.router_id}: Evicted {content_name} "
                f"(score={combined_scores.get(content_name, 0):.3f}, "
                f"recency={recency_scores.get(content_name, 0):.3f}, "
                f"frequency={frequency_scores.get(content_name, 0):.3f})"
            )
        
        return space_freed >= required_space
    
    def evict_lru(self, required_space: int) -> bool:
        """Evict content using LRU policy"""
        if not self.store:
            return False
            
        # Sort content by last access time
        lru_contents = sorted(
            self.store.keys(),
            key=lambda x: self.last_access_time.get(x, 0)
        )
        
        space_freed = 0
        for content in lru_contents:
            space_freed += self.size_map.get(content, 0)
            self._remove_content_no_lock(content)
            
            if space_freed >= required_space:
                return True
                
        return space_freed >= required_space
    
    def evict_lfu(self, required_space: int) -> bool:
        """Evict content using LFU (Least Frequently Used) policy"""
        space_freed = 0
        if not self.store:
            return False
        
        # Sort by access count (ascending) - least frequent first
        lfu_contents = sorted(
            self.store.keys(),
            key=lambda x: self.access_count.get(x, 0)
        )
        
        for content_name in lfu_contents:
            if space_freed >= required_space:
                break
            space_freed += self.size_map.get(content_name, 0)
            self._remove_content_no_lock(content_name)
                
        return space_freed >= required_space
    
    def evict_fifo(self, required_space: int) -> bool:
        """Evict content using FIFO policy."""
        space_freed = 0
        while self.store_order and space_freed < required_space:
            oldest = self.store_order.popleft()
            if oldest in self.store:
                space_freed += self.size_map.get(oldest, 0)
                self._remove_content_no_lock(oldest)
        return space_freed >= required_space

    def evict_lifo(self, required_space: int) -> bool:
        """Evict content using LIFO policy."""
        space_freed = 0
        while self.store_order and space_freed < required_space:
            newest = self.store_order.pop()
            if newest in self.store:
                space_freed += self.size_map.get(newest, 0)
                self._remove_content_no_lock(newest)
        return space_freed >= required_space

    def evict_basic(self, required_space: int) -> bool:
        """Default heuristic eviction (cluster-based fallback)."""
        return self.evict_from_least_popular_cluster(required_space)
    
    def evict_by_policy(self, required_space: int, current_time: float = None) -> bool:
        """Dispatch eviction based on configured policy."""
        policy = self.replacement_policy
        if policy == "fifo":
            return self.evict_fifo(required_space)
        if policy == "lifo":
            return self.evict_lifo(required_space)
        if policy == "lru":
            return self.evict_lru(required_space)
        if policy == "lfu":
            return self.evict_lfu(required_space)
        if policy == "combined":
            # Task 2.1: Use combined eviction algorithm (Algorithm 1 from report)
            return self.combined_eviction_algorithm(required_space, weight=0.5, current_time=current_time)
        # "basic" or unknown - default to combined algorithm
        return self.combined_eviction_algorithm(required_space, weight=0.5, current_time=current_time)
    
    def get_state_for_dqn(self, content_name: str, content_size: int, router=None, G=None, current_time: float = None) -> np.ndarray:
        """
        Optimized DQN state space with only essential features (5 features)
        Removed redundant features: cluster score, node degree, semantic similarity, content popularity, cache utilization.
        
        Key feature: Feature 4 (neighbor has content via Bloom filters) - enables
        distributed coordination without central control.
        
        Args:
            content_name: Name of content
            content_size: Size of content
            router: Router reference (optional, uses self.router_ref if None)
            G: Network graph (optional, uses self.graph_ref if None)
            current_time: Current simulation time (optional)
        
        Returns:
            5-dimensional state vector:
            [0] Content already cached (binary)
            [1] Content size (normalized)
            [2] Remaining capacity (normalized)
            [3] Access frequency (normalized)
            [4] Neighbor has content via Bloom filters (KEY - enables coordination)
        """
        # Error handling: Validate inputs
        if router is None:
            router = self.router_ref
        if G is None:
            G = self.graph_ref
        if current_time is None and hasattr(self, 'router_ref') and self.router_ref:
            current_time = getattr(self.router_ref, 'router_time', 0.0)
        
        with self.store_lock:
            # Optimized state: 5 essential features (removed Feature 3: cache utilization - redundant with Feature 2)
            state = np.zeros(5, dtype=np.float32)
            
            # Feature 0: Content already cached (binary)
            state[0] = float(content_name in self.store)
            
            # Feature 1: Content size (normalized)
            state[1] = float(content_size) / max(1, self.total_capacity)
            
            # Feature 2: Remaining cache capacity (normalized)
            state[2] = self.remaining_capacity / max(1, self.total_capacity)
            
            # Feature 3: Access frequency (normalized)
            access_count = self.access_count.get(content_name, 0)
            total_accesses = sum(self.access_count.values()) if self.access_count else 1
            state[3] = access_count / max(1, total_accesses)
            
            # Feature 4: Neighbor has content (via Bloom filters) - KEY CONTRIBUTION
            # This enables distributed coordination without central control
            if router is not None and hasattr(router, 'neighbors'):
                try:
                    # Thread-safe access to neighbors
                    if hasattr(router, 'neighbor_lock'):
                        with router.neighbor_lock:
                            neighbors = list(router.neighbors) if hasattr(router, 'neighbors') else []
                    else:
                        neighbors = list(router.neighbors) if hasattr(router, 'neighbors') else []
                except (AttributeError, Exception) as e:
                    logger.debug(f"ContentStore {self.router_id}: Error accessing neighbors: {e}")
                    neighbors = []
                
                if neighbors:
                    # Phase 7.1: Support for ablation study - disable Bloom filter feature
                    disable_bloom = os.environ.get("NDN_SIM_DISABLE_BLOOM", "0") == "1"
                    
                    if disable_bloom:
                        # Ablation variant: DQN without Bloom filters (no neighbor awareness)
                        # Set Feature 4 to 0.0 (no neighbor information)
                        state[4] = 0.0
                    else:
                        # Phase 3.1: Adaptive neighbor selection (weighted by importance)
                        # Weight neighbors by: traffic volume, distance, hit rate
                        neighbor_weights = self._calculate_neighbor_importance(neighbors, router, G)
                        
                        # Weighted count of neighbors that might have content (via Bloom filters)
                        weighted_neighbor_has_content = 0.0
                        total_weight = 0.0
                        
                        for neighbor_id in neighbors:
                            neighbor_filter = self.neighbor_filters.get(neighbor_id)
                            if neighbor_filter is not None:
                                # Check if neighbor's Bloom filter indicates content might be cached
                                neighbor_has = 1.0 if neighbor_filter.check(content_name) else 0.0
                                weight = neighbor_weights.get(neighbor_id, 1.0)  # Default weight = 1.0
                                weighted_neighbor_has_content += neighbor_has * weight
                                total_weight += weight
                        
                        # Weighted fraction of neighbors that might have content
                        state[4] = weighted_neighbor_has_content / max(1.0, total_weight) if total_weight > 0 else 0.0
                else:
                    state[4] = 0.0
            else:
                state[4] = 0.0
            
            return state
    
    def _calculate_neighbor_importance(self, neighbors: List[int], router=None, G=None) -> Dict[int, float]:
        """
        Phase 3.1: Calculate importance weights for neighbors
        
        Weights neighbors by:
        - Traffic volume (number of messages exchanged)
        - Distance (hop count in network)
        - Hit rate (if available)
        
        Args:
            neighbors: List of neighbor IDs
            router: Router reference (optional)
            G: Network graph (optional)
        
        Returns:
            Dictionary mapping neighbor_id -> importance weight (0.0 to 1.0)
        """
        weights = {}
        
        if router is None:
            router = self.router_ref
        if G is None:
            G = self.graph_ref
        
        if router is None or G is None:
            # Default: equal weights
            return {neighbor_id: 1.0 for neighbor_id in neighbors}
        
        try:
            import networkx as nx
            
            for neighbor_id in neighbors:
                weight = 1.0  # Base weight
                
                # Factor 1: Distance (closer neighbors are more important)
                try:
                    if router.router_id in G and neighbor_id in G:
                        try:
                            path_length = nx.shortest_path_length(G, router.router_id, neighbor_id)
                            # Closer neighbors get higher weight: weight = 1.0 / (1 + distance)
                            distance_weight = 1.0 / (1.0 + path_length)
                            weight *= (0.5 + 0.5 * distance_weight)  # Scale to [0.5, 1.0]
                        except (nx.NetworkXNoPath, KeyError):
                            # No path found, use default weight
                            pass
                except Exception:
                    pass
                
                # Factor 2: Traffic volume (if tracked)
                # Check if we have statistics on messages to this neighbor
                if hasattr(router, 'stats') and hasattr(router.stats, 'stats'):
                    # Could track per-neighbor message counts here
                    # For now, use base weight
                    pass
                
                # Factor 3: Neighbor status (up/down)
                if hasattr(router, 'neighbor_status'):
                    neighbor_status = router.neighbor_status.get(neighbor_id, 'up')
                    if neighbor_status == 'down':
                        weight *= 0.1  # Down neighbors get very low weight
                
                weights[neighbor_id] = max(0.1, min(1.0, weight))  # Clamp to [0.1, 1.0]
        except Exception as e:
            logger.debug(f"ContentStore {self.router_id}: Error calculating neighbor importance: {e}")
            # Fallback: equal weights
            weights = {neighbor_id: 1.0 for neighbor_id in neighbors}
        
        return weights
            
    def store_content(self, name: str, content: Any, size: int, current_time: float, router=None, G=None) -> bool:
        """
        Store content in the cache using the selected caching policy
        
        Args:
            name: Content name
            content: Content object
            size: Content size
            current_time: Current time
            router: Router reference (for DQN state, optional)
            G: Network graph (for DQN state, optional)
        """
        # Task 2.4: Update router and graph references if provided
        if router is not None:
            self.router_ref = router
        if G is not None:
            self.graph_ref = G
        
        # Select caching policy based on mode
        if self.mode == "dqn_cache" and self.dqn_agent is not None:
            return self.store_content_with_dqn(name, content, size, current_time)
        else:
            # Default basic caching
            return self.store_content_basic(name, content, size, current_time)
    
    def store_content_basic(self, name: str, content: Any, size: int, current_time: float) -> bool:
        """Basic caching policy"""
        with self.store_lock:
            # Task 1.4: Add detailed logging for cache insertion debugging
            logger.debug(
                f"ContentStore {self.router_id}: store_content_basic called for {name} "
                f"(size={size}, remaining={self.remaining_capacity}, "
                f"total={self.total_capacity}, store_size={len(self.store)})"
            )
            
            # Check if content already exists
            if name in self.store:
                # Update access time
                self.last_access_time[name] = current_time
                self.access_count[name] = self.access_count.get(name, 0) + 1
                if self.replacement_policy in {"fifo", "lifo"}:
                    try:
                        self.store_order.remove(name)
                    except ValueError:
                        pass
                    self.store_order.append(name)
                logger.debug(f"ContentStore {self.router_id}: Content {name} already cached, updated access")
                return True
                
            # Check if content is too large
            if size > self.total_capacity:
                logger.warning(
                    f"ContentStore {self.router_id}: Content {name} size {size} "
                    f"exceeds total capacity {self.total_capacity}"
                )
                return False
                
            # Check if we need to evict
            eviction_attempts = 0
            while self.remaining_capacity < size:
                eviction_attempts += 1
                if eviction_attempts > 10:  # Prevent infinite loops
                    logger.error(
                        f"ContentStore {self.router_id}: Too many eviction attempts for {name}, aborting"
                    )
                    return False
                # Task 2.1: Pass current_time to eviction for combined algorithm
                eviction_result = self.evict_by_policy(size - self.remaining_capacity, current_time=current_time)
                if not eviction_result:
                    logger.warning(
                        f"ContentStore {self.router_id}: Failed to evict enough space for {name} "
                        f"(needed={size}, remaining={self.remaining_capacity}, "
                        f"store_size={len(self.store)})"
                    )
                    return False
                    
            # Store the content
            if hasattr(content, 'clone'):
                content = content.clone()
            self.store[name] = content
            self.size_map[name] = size
            self.remaining_capacity -= size
            self.last_access_time[name] = current_time
            self.access_count[name] = self.access_count.get(name, 0) + 1
            self.bloom_filter.add(name)
            self.insertions += 1
            if self.replacement_policy in {"fifo", "lifo"}:
                self.store_order.append(name)
            elif self.replacement_policy == "lru":
                try:
                    self.store_order.remove(name)
                except ValueError:
                    pass
                self.store_order.append(name)
            
            # Update clusters
            self.update_clusters(name, current_time)
            
            # Propagate Bloom filter to neighbors periodically
            self._maybe_propagate_bloom_filter()
            
            logger.debug(f"Stored content {name} of size {size}")
            return True
    
    def store_content_with_dqn(self, name: str, content: Any, size: int, current_time: float) -> bool:
        """Store content using DQN policy for caching decisions"""
        # Error handling: Validate DQN agent is available
        if self.dqn_agent is None:
            logger.warning(f"ContentStore {self.router_id}: DQN agent not initialized, falling back to basic caching")
            return self.store_content_basic(name, content, size, current_time)
        
        with self.store_lock:
            # Check if content already exists
            if name in self.store:
                # Update access time and count
                self.last_access_time[name] = current_time
                self.access_count[name] = self.access_count.get(name, 0) + 1
                return True
                
            # Check if content is too large
            if size > self.total_capacity:
                return False
                
            # Task 2.4: Get state for DQN with router and graph references, passing simulation time
            # Error handling: Validate state vector and handle errors gracefully
            try:
                state = self.get_state_for_dqn(name, size, router=self.router_ref, G=self.graph_ref, current_time=current_time)
                
                # Validate state vector
                if state is None or len(state) != 5:
                    logger.warning(f"ContentStore {self.router_id}: Invalid state vector (got {len(state) if state is not None else 0} features, expected 5), defaulting to cache")
                    action = 1
                else:
                    # Get action from DQN agent (0 = don't cache, 1 = cache)
                    if self.dqn_agent is not None:
                        action = self.dqn_agent.select_action(state)
                    else:
                        # If no agent, default to caching
                        action = 1
            except Exception as e:
                logger.warning(f"ContentStore {self.router_id}: Error getting DQN state/action: {e}, defaulting to cache")
                action = 1
            
            # Process the action
            if action == 1:  # Cache the content
                # Check if we need to evict
                while self.remaining_capacity < size:
                    if not self.evict_from_least_popular_cluster(size - self.remaining_capacity):
                        # If eviction failed, don't cache
                        if self.dqn_agent is not None:
                            # Learn from this experience - negative reward for failed caching
                            try:
                                reward = self.dqn_agent.calculate_reward(
                                    is_caching_decision=False, was_cached=False
                                )
                                next_state = self.get_state_for_dqn(name, size, router=self.router_ref, G=self.graph_ref, current_time=current_time)
                                if next_state is not None and len(next_state) == 6:
                                    self.dqn_agent.remember(state, action, reward, next_state, False)
                                    self._maybe_train_dqn()
                            except Exception as e:
                                logger.debug(f"ContentStore {self.router_id}: Error recording failed caching experience: {e}")
                        return False
                
                # Store the content
                if hasattr(content, 'clone'):
                    content = content.clone()
                self.store[name] = content
                self.size_map[name] = size
                self.remaining_capacity -= size
                self.last_access_time[name] = current_time
                self.access_count[name] = self.access_count.get(name, 0) + 1
                self.bloom_filter.add(name)
                self.insertions += 1
                try:
                    self.store_order.remove(name)
                except ValueError:
                    pass
                self.store_order.append(name)
                
                # Update clusters
                self.update_clusters(name, current_time)
                
                # Track that this content was cached by DQN decision
                self.dqn_cached_contents.add(name)
                
                # Store the original state when caching decision was made
                # This allows proper credit assignment when cache hit occurs
                if state is not None and len(state) == 5:
                    self.dqn_decision_states[name] = state.copy()
                
                # Propagate Bloom filter to neighbors periodically
                self._maybe_propagate_bloom_filter()
                
                if self.dqn_agent is not None:
                    # Store experience WITHOUT immediate reward - only record the action
                    # Reward will come later when cache hit occurs (via notify_cache_hit)
                    try:
                        # Get next state for experience storage
                        next_state = self.get_state_for_dqn(name, size, router=self.router_ref, G=self.graph_ref, current_time=current_time)
                        if next_state is not None and len(next_state) == 6:
                            # Store experience with zero reward - reward comes later on cache hit
                            reward = 0.0  # No immediate reward for caching decision
                            self.dqn_agent.remember(state, action, reward, next_state, False)
                            # Don't train immediately - wait for cache hit rewards
                    except Exception as e:
                        logger.debug(f"ContentStore {self.router_id}: Error recording caching experience: {e}")
                
                return True
            else:
                # Decision not to cache
                if self.dqn_agent is not None:
                    # Small negative reward for not caching
                    try:
                        reward = self.dqn_agent.calculate_reward(
                            is_caching_decision=False, was_cached=False
                        )
                        next_state = self.get_state_for_dqn(name, size, router=self.router_ref, G=self.graph_ref, current_time=current_time)
                        if next_state is not None and len(next_state) == 6:
                            self.dqn_agent.remember(state, action, reward, next_state, False)
                            self._maybe_train_dqn()
                    except Exception as e:
                        logger.debug(f"ContentStore {self.router_id}: Error recording non-caching experience: {e}")
                
                return False
    
    def _maybe_train_dqn(self):
        """
        Schedule DQN training asynchronously to avoid blocking message processing.
        Training happens in background thread pool.
        """
        if self.dqn_agent is None:
            return
        
        self.dqn_training_step += 1
        
        # Check if training should happen
        should_train = (
            self.dqn_training_step % self.dqn_training_frequency == 0 or 
            len(self.dqn_agent.memory) >= self.dqn_agent.batch_size
        )
        
        if should_train and len(self.dqn_agent.memory) >= self.dqn_agent.batch_size:
            # Get training manager singleton
            try:
                from router import DQNTrainingManager
                training_manager = DQNTrainingManager.get_instance()
                
                # Submit training to background thread (non-blocking)
                training_manager.submit_training(
                    training_fn=lambda: self.dqn_agent.replay(),
                    router_id=self.router_id
                )
            except Exception as e:
                # Fallback to synchronous training if manager not available
                logger.debug(f"ContentStore {self.router_id}: Training manager not available, using sync training: {e}")
                self.dqn_agent.replay()
    
    def notify_cache_hit(self, content_name: str, current_time: float):
        """
        Notify ContentStore that a cache hit occurred for delayed reward.
        This is called when content cached by DQN is actually used.
        
        Args:
            content_name: Name of the content that was hit
            current_time: Current simulation time
        """
        if self.dqn_agent is None:
            return
        
        with self.store_lock:
            # Check if this content was cached by DQN decision
            if content_name not in self.dqn_cached_contents:
                return  # Not cached by DQN, no delayed reward
            
            # Calculate delayed reward for cache hit
            cluster_id = self.cluster_manager.content_to_cluster.get(content_name, -1)
            cluster_score = self.cluster_manager.get_cluster_score(cluster_id) if cluster_id >= 0 else 0
            content_size = self.size_map.get(content_name, 0)
            
            # Large positive reward for actual cache hit
            access_count = self.access_count.get(content_name, 0)
            total_accesses = sum(self.access_count.values()) if self.access_count else 1
            access_frequency = access_count / max(1, total_accesses)
            
            # Phase 6.2: Calculate latency and bandwidth savings for multi-objective reward
            latency_saved = 0.0
            bandwidth_saved = 0.0
            
            try:
                from metrics import get_metrics_collector
                metrics_collector = get_metrics_collector()
                
                # Estimate latency saved: cache hit avoids network traversal
                # Average network latency for cache miss (estimate: 0.1-0.5 seconds)
                # Cache hit latency is near-zero, so saved = average_miss_latency
                latency_metrics = metrics_collector.get_latency_metrics()
                avg_latency = latency_metrics.get('mean', 0.2)  # Default 0.2s if no data
                latency_saved = avg_latency  # Cache hit saves this much latency
                
                # Bandwidth saved: content size (we didn't need to fetch from network)
                bandwidth_saved = float(content_size)  # Bytes saved
            except Exception as e:
                logger.debug(f"ContentStore {self.router_id}: Error calculating latency/bandwidth savings: {e}")
                # Use defaults if metrics not available
                latency_saved = 0.2  # Default 0.2s saved
                bandwidth_saved = float(content_size)
            
            reward = self.dqn_agent.calculate_reward(
                is_cache_hit=True,
                content_size=content_size,
                cluster_score=cluster_score,
                access_frequency=access_frequency,
                latency_saved=latency_saved,
                bandwidth_saved=bandwidth_saved
            )
            
            # Get ORIGINAL state when caching decision was made (for proper credit assignment)
            try:
                # Use original state from when caching decision was made
                original_state = self.dqn_decision_states.get(content_name)
                if original_state is None:
                    # Fallback to current state if original not found
                    original_state = self.get_state_for_dqn(content_name, content_size, router=self.router_ref, G=self.graph_ref, current_time=current_time)
                
                # Get current state for next_state
                next_state = self.get_state_for_dqn(content_name, content_size, router=self.router_ref, G=self.graph_ref, current_time=current_time)
                
                # Use original state for proper credit assignment
                state = original_state
                
                if state is not None and len(state) == 5:
                    # Create experience tuple with delayed reward
                    # Action was 1 (cache), and we got a hit
                    action = 1
                    done = False
                    
                    # Store experience with delayed reward
                    self.dqn_agent.remember(state, action, reward, next_state, done)
                    
                    # Train if needed
                    self._maybe_train_dqn()
                    
                    logger.debug(f"ContentStore {self.router_id}: Delayed reward {reward:.2f} for cache hit on {content_name}")
                else:
                    logger.debug(f"ContentStore {self.router_id}: Invalid state for cache hit reward, skipping")
            except Exception as e:
                logger.debug(f"ContentStore {self.router_id}: Error processing cache hit reward: {e}")
    
    def _maybe_propagate_bloom_filter(self):
        """
        Propagate Bloom filter to neighbors periodically.
        This enables neighbor awareness in DQN state space.
        """
        if self.router_ref is None:
            return
        
        # Check if it's time to propagate
        if (self.insertions - self.last_bloom_propagation) < self.bloom_propagation_frequency:
            return
        
        self.last_bloom_propagation = self.insertions
        self.propagate_bloom_filter_to_neighbors()
    
    def propagate_bloom_filter_to_neighbors(self):
        """
        Send local Bloom filter to all neighbors for cache state awareness.
        This enables DQN feature 4 (neighbor has content via Bloom filters).
        Tracks communication overhead for metrics collection.
        
        Phase 7.1: Respects NDN_SIM_DISABLE_BLOOM flag for ablation study.
        """
        # Phase 7.1: Support for ablation study - skip propagation if disabled
        disable_bloom = os.environ.get("NDN_SIM_DISABLE_BLOOM", "0") == "1"
        if disable_bloom:
            return  # Skip Bloom filter propagation for ablation study
        
        if self.router_ref is None:
            return
        
        # Fix: Thread-safe access to neighbors
        with self.router_ref.neighbor_lock:
            neighbors = list(self.router_ref.neighbors)  # Create a copy to avoid holding lock
        
        if not neighbors:
            return
        
        # Create a copy of the Bloom filter for sending
        # For basic Bloom filter, copy the bit array
        try:
            with self.bloom_filter.lock:
                if isinstance(self.bloom_filter, NeuralBloomFilter):
                    # For Neural Bloom Filter, create a basic Bloom filter copy
                    # (neural network state is not propagated)
                    filter_copy = BloomFilter(
                        size=self.bloom_filter.size,
                        hash_count=self.bloom_filter.hash_count
                    )
                    filter_copy.bit_array = self.bloom_filter.bit_array.copy()
                else:
                    # For basic Bloom filter, create a copy
                    filter_copy = BloomFilter(
                        size=self.bloom_filter.size,
                        hash_count=self.bloom_filter.hash_count
                    )
                    filter_copy.bit_array = self.bloom_filter.bit_array.copy()
            
            # Calculate Bloom filter size in bytes for overhead tracking
            bloom_filter_bytes = (self.bloom_filter.size + 7) // 8  # Convert bits to bytes
            
            # Send to all neighbors and track communication overhead
            for neighbor_id in neighbors:
                if self.router_ref and hasattr(self.router_ref, 'send_message'):
                    try:
                        self.router_ref.send_message(
                            neighbor_id,
                            'bloom_filter_update',
                            (self.router_id, filter_copy),
                            priority=2  # Lower priority than Interest/Data packets
                        )
                        
                        # Track communication overhead (Bloom filter propagation)
                        try:
                            from metrics import get_metrics_collector
                            metrics_collector = get_metrics_collector()
                            # Track Bloom filter overhead as Interest bytes (communication overhead)
                            metrics_collector.record_interest(
                                f"bloom_filter_{self.router_id}_{neighbor_id}",
                                f"bloom_filter_update",
                                self.router_id,
                                interest_size=bloom_filter_bytes
                            )
                        except Exception as e:
                            logger.debug(f"ContentStore {self.router_id}: Error tracking Bloom filter overhead: {e}")
                        
                        logger.debug(f"ContentStore {self.router_id}: Propagated Bloom filter to neighbor {neighbor_id}")
                    except Exception as e:
                        logger.debug(f"ContentStore {self.router_id}: Failed to send Bloom filter to {neighbor_id}: {e}")
        except Exception as e:
            logger.debug(f"ContentStore {self.router_id}: Error propagating Bloom filter: {e}")
    
    def receive_bloom_filter_update(self, neighbor_id: int, bloom_filter: BloomFilter):
        """
        Receive and store neighbor's Bloom filter update.
        This updates the neighbor_filters dictionary used in DQN state space.
        
        Phase 3.3: Initializes false positive tracking for this neighbor.
        
        Args:
            neighbor_id: ID of the neighbor router
            bloom_filter: Bloom filter from the neighbor
        """
        try:
            # Store the neighbor's Bloom filter
            # Create a new filter to avoid reference issues
            stored_filter = BloomFilter(
                size=bloom_filter.size,
                hash_count=bloom_filter.hash_count
            )
            with bloom_filter.lock:
                stored_filter.bit_array = bloom_filter.bit_array.copy()
            
            self.neighbor_filters[neighbor_id] = stored_filter
            
            # Phase 3.3: Initialize false positive tracking for this neighbor
            if not hasattr(self, 'neighbor_false_positives'):
                self.neighbor_false_positives: Dict[int, Dict[str, int]] = {}  # neighbor_id -> tracking data
            if neighbor_id not in self.neighbor_false_positives:
                self.neighbor_false_positives[neighbor_id] = {'total_checks': 0, 'false_positives': 0}
            
            logger.debug(f"ContentStore {self.router_id}: Received Bloom filter update from neighbor {neighbor_id}")
        except Exception as e:
            logger.debug(f"ContentStore {self.router_id}: Error receiving Bloom filter from {neighbor_id}: {e}")
    
    def _track_bloom_filter_false_positive(self, neighbor_id: int, content_name: str, actually_has: bool):
        """
        Phase 3.3: Track false positives for Bloom filter learning
        
        Args:
            neighbor_id: ID of neighbor
            content_name: Content name that was checked
            actually_has: True if neighbor actually has the content, False if false positive
        """
        if not hasattr(self, 'neighbor_false_positives'):
            self.neighbor_false_positives: Dict[int, Dict[str, int]] = {}
        
        if neighbor_id not in self.neighbor_false_positives:
            self.neighbor_false_positives[neighbor_id] = {'total_checks': 0, 'false_positives': 0}
        
        self.neighbor_false_positives[neighbor_id]['total_checks'] += 1
        if not actually_has:
            self.neighbor_false_positives[neighbor_id]['false_positives'] += 1
        
        # Calculate false positive rate for this neighbor
        fp_data = self.neighbor_false_positives[neighbor_id]
        if fp_data['total_checks'] > 0:
            fpr = fp_data['false_positives'] / fp_data['total_checks']
            # Adjust confidence in future checks based on FPR
            # Higher FPR = lower confidence
            if fpr > 0.1:  # If FPR > 10%, reduce weight
                logger.debug(f"ContentStore {self.router_id}: Neighbor {neighbor_id} has high FPR {fpr:.3f}, reducing confidence")
            
    def get_content(self, name: str, current_time: float) -> Optional[Any]:
        """Retrieve content from cache"""
        with self.store_lock:
            content = self.store.get(name)
            if content is not None:
                # Update access statistics
                self.last_access_time[name] = current_time
                self.access_count[name] = self.access_count.get(name, 0) + 1
                if self.replacement_policy == "lru":
                    try:
                        self.store_order.remove(name)
                    except ValueError:
                        pass
                    self.store_order.append(name)
                
                # Update cluster score
                cluster_id = self.cluster_manager.content_to_cluster.get(name)
                if cluster_id is not None:
                    self.cluster_manager.update_cluster_score(cluster_id, 1.0, current_time)
                    
                logger.debug(f"Cache hit for {name}")
                if hasattr(content, 'clone'):
                    content = content.clone()
            return content
            
    def _remove_content_no_lock(self, name: str):
        """Internal helper to remove content with lock already held."""
        if name in self.store:
            self.remaining_capacity += self.size_map.get(name, 0)
            
            del self.store[name]
            if name in self.size_map:
                del self.size_map[name]
            try:
                self.store_order.remove(name)
            except ValueError:
                pass
            # Remove from DQN tracking if it was cached by DQN
            self.dqn_cached_contents.discard(name)
            # Remove cached embedding if it exists
            self.cached_embeddings.pop(name, None)
            logger.debug(f"Removed content {name} from cache")

    def remove_content(self, name: str):
        """Remove content from cache"""
        with self.store_lock:
            self._remove_content_no_lock(name)
                
    def get_cluster_statistics(self) -> Dict:
        """Get statistics about content clusters"""
        with self.store_lock, self.embedding_lock:
            return {
                'num_clusters': len(self.cluster_manager.clusters),
                'cluster_sizes': {k: len(v) for k, v in self.cluster_manager.clusters.items()},
                'cluster_scores': dict(self.cluster_manager.cluster_scores),
                'capacity_used': self.total_capacity - self.remaining_capacity,
                'total_content': len(self.store)
            }


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)
    timestamp: datetime = field(default_factory=datetime.now, compare=True)


class PIT:
    def __init__(self, threshold: int = 500):
        self.entries: Dict[str, List[int]] = {}
        self.times: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.threshold = threshold
        
    def add_entry(self, name: str, incoming_face: int, router_time: float):
        with self.lock:
            if name not in self.entries:
                self.entries[name] = []
                self.times[name] = router_time
            if incoming_face not in self.entries[name]:
                self.entries[name].append(incoming_face)
                self.times[name] = router_time
                
    def remove_entry(self, name: str, incoming_face: int = None):
        with self.lock:
            if name in self.entries:
                if incoming_face is not None and incoming_face in self.entries[name]:
                    self.entries[name].remove(incoming_face)
                    if not self.entries[name]:
                        del self.entries[name]
                        if name in self.times:
                            del self.times[name]
                else:
                    del self.entries[name]
                    if name in self.times:
                        del self.times[name]
                    
    def get(self, name: str) -> List[int]:
        with self.lock:
            return self.entries.get(name, []).copy()
    
    def cleanup_expired(self, current_time: float, interest_lifetime: float = 4.0):
        """
        FIX #4: Remove expired PIT entries based on Interest lifetime (RFC 8569)
        
        Args:
            current_time: Current router time
            interest_lifetime: Maximum lifetime for Interest packets (default 4.0 seconds)
        """
        with self.lock:
            expired_names = []
            for name, entry_time in list(self.times.items()):
                if current_time - entry_time > interest_lifetime:
                    expired_names.append(name)
            
            for name in expired_names:
                if name in self.entries:
                    del self.entries[name]
                if name in self.times:
                    del self.times[name]
            
            if expired_names:
                logger.debug(f"PIT: Cleaned up {len(expired_names)} expired entries")
            
    def __contains__(self, name: str) -> bool:
        with self.lock:
            return name in self.entries
            # Fall back to basic caching if DQN fails
            
