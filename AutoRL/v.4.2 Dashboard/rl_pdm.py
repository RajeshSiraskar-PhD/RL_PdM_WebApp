# ---------------------------------------------------------------------------------------
# AutoRL: Auto-train Predictive Maintenance Agents 
# Author: Rajesh Siraskar
# RL for PdM code
# V.1.0: 06-Feb-2026: First commit
# ---------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Type, Union, TypeVar

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


# --- GLOBAL VARIABLES ---
DATA_FILE = "dummy_sensor_data.csv" # Default, will be overwritten

# WEAR_THRESHOLD: ISO recommended standard (constant, used for all plots and displays)
WEAR_THRESHOLD = 300.00

# TRAINING_WEAR_THRESHOLD: Configurable threshold for agent training
# Should be <= WEAR_THRESHOLD so agent learns to replace earlier
# Set from configuration page or defaults to WEAR_THRESHOLD
TRAINING_WEAR_THRESHOLD = 300.00

EPISODES = 100
LR_DEFAULT = 0.001
GAMMA_DEFAULT = 0.99
SMOOTH_WINDOW = 10
FIXED_X_AXIS_LENGTH = True # Validation: Train for fixed episodes?
IAR_RANGE = 0.05 # ±5% variation in IAR bounds for realism

now = datetime.now()
date_time = now.strftime("%d-%m-%H-%M")

# Rewards (Adjustable)
# Strategy: Balance exploration incentive with learning signal
# 
# 1. Small positive baseline (R1=0.1) - incentivizes agent to stay in game and learn
# 2. STRONG violation penalty (R2=-100) - forces learning to avoid violations
# 3. Replace cost (R3=-0.5) - replacement has a cost
# 4. Replacement bonus (R4=60) - rewards EARLY, HIGH-wear replacement before violation
#
# Why this works:
# - Agent explores: Discovers violations (-100) are worse than replacements (-0.5)
# - Agent learns: Replacement at high wear (+60-0.5 = +59.5) beats continuing to violation
# - Exploitation: Learns optimal replacement timing (>90% wear gets +60 bonus)

R1 = 0.1       # Small positive for surviving step - encourages learning vs. inaction
R2 = -100.0    # CATASTROPHIC penalty for violations - forces learning
R3 = -0.5      # Replacement cost - makes replacement a deliberate choice
R4 = 60.0      # Strong bonus for optimal replacement (increased from 50)

print(f"RL module loaded: Fixed length: {FIXED_X_AXIS_LENGTH}, WEAR_THRESHOLD: {WEAR_THRESHOLD}, TRAINING_WEAR_THRESHOLD: {TRAINING_WEAR_THRESHOLD}, R1: {R1}, R2: {R2}, R3: {R3}, R4: {R4}")

class MT_Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_file, wear_threshold=300, r1=1, r2=-100, r3=-5, r4=50):
        super(MT_Env, self).__init__()
        
        self.data = pd.read_csv(data_file)
        self.wear_threshold = wear_threshold
        self.R1 = r1
        self.R2 = r2
        self.R3 = r3
        self.R4 = r4
        
        # Detect Schema
        self.schema = self._detect_schema(self.data.columns)
        self.features = self._get_features(self.schema)
        
        # Define Action Space: 0 = REPLACE, 1 = CONTINUE
        self.action_space = spaces.Discrete(2)
        
        # Define Observation Space
        # Only SENSOR READINGS are included in observations.
        # EXCLUDED from observation: 'Time', 'tool_wear', 'ACTION_CODE'
        # - tool_wear: This is what the agent needs to PREDICT/manage, not observe directly
        # - ACTION_CODE: This is the action/label, not a feature
        # - Time: Not a sensor reading, excluded for simplicity
        # 
        # Agents must learn to predict maintenance needs from sensor data (forces, vibrations, acoustics, etc.) alone.
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.features),), dtype=np.float32
        )
        
        # VALIDATION: Check if features exist
        missing_cols = [c for c in self.features if c not in self.data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing sensor columns in data for schema {self.schema}: {missing_cols}\n"
                f"Available columns: {list(self.data.columns)}"
            )
        
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        
    def _detect_schema(self, columns):
        if 'force_x' in columns and 'acoustic_emission_rms' in columns:
            return 'IEEE'
        elif 'Vib_Spindle' in columns and 'Sound_Spindle' in columns:
            return 'SIT'
        else:
            return 'UNKNOWN' # Fallback or Error

    def _get_features(self, schema):
        if schema == 'IEEE':
            # IEEE sensor features ONLY (exclude Time, tool_wear, ACTION_CODE)
            # Agent must learn to predict maintenance from sensor readings alone
            features = ['force_x', 'force_y', 'force_z', 
                       'vibration_x', 'vibration_y', 'vibration_z', 
                       'acoustic_emission_rms']
            return features
        elif schema == 'SIT':
            # SIT sensor features ONLY (exclude Time, tool_wear, ACTION_CODE)
            features = ['Vib_Spindle', 'Vib_Table', 'Sound_Spindle', 'Sound_table', 
                       'X_Load_Cell', 'Y_Load_Cell', 'Z_Load_Cell', 'Current']
            return features
        else:
            # Fallback: all numeric columns except Time, tool_wear, ACTION_CODE
            excluded = ['Time', 'time', 'tool_wear', 'ACTION_CODE']
            cols = [c for c in self.data.columns if c not in excluded]
            return cols

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # In a real scenario, we might start at random point or 0. 
        # For this dataset-based env, we simulate a run.
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # Extract features for current step
        obs = self.data.iloc[self.current_step][self.features].values.astype(np.float32)
        return obs

    def step(self, action):
        # action: 0 = REPLACE, 1 = CONTINUE
        
        reward = 0
        terminated = False
        truncated = False
        info = {}
        
        current_wear = self.data.iloc[self.current_step]['tool_wear']
        wear_ratio = current_wear / self.wear_threshold  # 0 to ~1
        
        if action == 0: # REPLACE
            # Agent decides to replace - episode ends
            terminated = True
            
            # Base replacement cost (very cheap)
            reward = self.R3
            
            # STRONG bonus for optimal replacement timing
            # Goal: Replace when wear is HIGH but BEFORE violation
            if current_wear <= self.wear_threshold:  # No violation
                if wear_ratio > 0.9:  # Excellent timing (>90% of threshold)
                    reward += self.R4  # +50 bonus
                elif wear_ratio > 0.8:  # Good timing (>80%)
                    reward += self.R4 * 0.6  # +30 bonus
                elif wear_ratio > 0.7:  # Decent timing (>70%)
                    reward += self.R4 * 0.3  # +15 bonus
                # Below 70%: just the base replacement cost (too early)
            else:
                # Replaced AFTER violation - still bad, but better than continuing
                reward = self.R2 * 0.5  # -500 (half the violation penalty)
            
            # Metric info
            info['wear_margin'] = max(0, self.wear_threshold - current_wear)
            info['replaced'] = True
            info['threshold_violation'] = (current_wear > self.wear_threshold)

        else: # CONTINUE
            # Check if we crossed threshold (CRITICAL FAILURE)
            if current_wear > self.wear_threshold:
                # CATASTROPHIC FAILURE - This is what we must avoid!
                reward = self.R2  # -1000
                terminated = True
                info['wear_margin'] = max(0, self.wear_threshold - current_wear)
                info['replaced'] = False
                info['threshold_violation'] = True
            else:
                # Survived another step without violation
                # Small positive reward (R1=0.1) encourages agent to keep learning
                # This baseline prevents the agent from getting stuck in inaction
                reward = self.R1  # = 0.1 per step
                
                # The agent learns: 
                # - Continuing gives +0.1 per step (incremental, positive)
                # - But violating eventually gives -100 (catastrophic)
                # - So it MUST explore REPLACE to avoid violations
                # - REPLACE at high wear gives -0.5 + 60 = +59.5 (excellent)
                
                terminated = False
                info['replaced'] = False
                info['threshold_violation'] = False
        
        # Ensure wear_margin is always recorded
        if 'wear_margin' not in info:
            info['wear_margin'] = max(0, self.wear_threshold - current_wear)
        
        # Move to next step if not terminated
        if not terminated:
            self.current_step += 1
            if self.current_step >= len(self.data) - 1:
                # End of data
                terminated = True
                truncated = True
        
        # Get next obs
        if not terminated:
            obs = self._get_obs()
        else:
            # Just return last obs if done
            obs = self._get_obs()
            
        return obs, reward, terminated, truncated, info

class StreamlitCallback(BaseCallback):
    """
    Custom callback for plotting in Streamlit.
    """
    def __init__(self, update_func, update_freq=1, verbose=0, total_episodes=100):
        super(StreamlitCallback, self).__init__(verbose)
        self.update_func = update_func
        self.update_freq = update_freq
        self.total_episodes = total_episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.wear_margins = []
        self.violations = []
        self.replacements = []
        
        self.current_ep_reward = 0
        self.current_ep_len = 0
        self.current_ep_violation = False
        self.current_ep_replaced = False
        self.current_ep_margin = 0

    def _on_step(self) -> bool:
        # Collect step info
        # info is in self.locals['infos'][0] (assuming 1 env)
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0] # Array
        done = self.locals['dones'][0]
        
        self.current_ep_reward += reward
        self.current_ep_len += 1
        
        if 'threshold_violation' in info and info['threshold_violation']:
            self.current_ep_violation = True
        
        if 'replaced' in info and info['replaced']:
            self.current_ep_replaced = True
            if 'wear_margin' in info:
                self.current_ep_margin = info['wear_margin']
        elif done and not self.current_ep_replaced:
             # Failed or ran out of data
             if 'wear_margin' in info:
                self.current_ep_margin = info['wear_margin']
        
        if done:
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_len)
            self.violations.append(1 if self.current_ep_violation else 0)
            self.replacements.append(1 if self.current_ep_replaced else 0)
            self.wear_margins.append(self.current_ep_margin)
            
            # Send data to Streamlit via callback function
            # BATCH UPDATE: Only update if condition met
            # Update frequency based on episode count
            ep_count = len(self.episode_rewards)
            
            # Print progress every 10% or on first/last episode
            progress_interval = max(1, self.total_episodes // 10)
            if ep_count % progress_interval == 0 or ep_count == 1 or ep_count == self.total_episodes:
                print(f"Episode: {ep_count}/{self.total_episodes}")
            
            if ep_count % self.update_freq == 0 or ep_count == 1:
                self.update_func({
                    'rewards': self.episode_rewards,
                    'violations': self.violations,
                    'replacements': self.replacements,
                    'margins': self.wear_margins
                })
            
            # Reset current ep
            self.current_ep_reward = 0
            self.current_ep_len = 0
            self.current_ep_violation = False
            self.current_ep_replaced = False
            self.current_ep_margin = 0
            
        return True

# --- ATTENTION MECHANISMS ---
class NadarayaWatsonExtractor(BaseFeaturesExtractor):
    """
    Nadaraya-Watson Kernel Regression as a Feature Extractor.
    Learns to weigh input features using a differentiable kernel mechanism.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        input_dim = observation_space.shape[0]
        
        # Learnable strictly positive bandwidth (beta)
        # We use a Parameter so it's optimized
        self.log_beta = nn.Parameter(th.zeros(1))
        
        # Learnable Keys and Values (Memory)
        # We assume a fixed memory size, e.g., equal to input_dim or larger
        # Ideally, NW uses the training set as memory, but here we learn "prototypes"
        self.memory_size = 32
        self.keys = nn.Parameter(th.randn(self.memory_size, input_dim))
        self.values = nn.Parameter(th.randn(self.memory_size, features_dim))
        
        # Linear projection for query (the observation)
        # Optional: could just use obs as query directly
        self.query_net = nn.Linear(input_dim, input_dim)
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.float()
        # query: [batch_size, input_dim]
        query = self.query_net(observations)
        
        # keys: [memory_size, input_dim]
        # values: [memory_size, features_dim]
        
        # Compute distances/attention scores
        # Expand dims for broadcasting
        # query: [batch, 1, input_dim]
        # keys:  [1, memory_size, input_dim]
        query_exp = query.unsqueeze(1)
        keys_exp = self.keys.unsqueeze(0)
        
        # L2 Distance squared: ||x - k||^2
        dist = (query_exp - keys_exp).pow(2).sum(dim=2) # [batch, memory_size]
        
        # Softmax with beta (bandwidth)
        beta = th.exp(self.log_beta)
        attention_weights = th.softmax(-beta * dist, dim=1) # [batch, memory_size]
        
        # Weighted sum of values
        # weights: [batch, memory_size, 1]
        # values:  [1, memory_size, features_dim]
        context = (attention_weights.unsqueeze(2) * self.values.unsqueeze(0)).sum(dim=1)
        
        return context

class SimpleAttentionExtractor(BaseFeaturesExtractor):
    """
    Simple Soft Attention (Deep Learning Attention).
    Applies a standard MLP-based attention mask to features.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        # Structure: Input -> Attention Weights -> Weighted Input -> Output
        
        # Attention Network
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, features_dim),
            nn.Tanh(),
            nn.Linear(features_dim, input_dim),
            nn.Softmax(dim=1)
        )
        
        # Final projection to desired feature dim
        self.projection = nn.Linear(input_dim, features_dim)
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Calculate attention weights
        attn_weights = self.attention_net(observations)

        # Apply attention (element-wise multiplication)
        weighted_features = observations * attn_weights

        # Project to output
        return self.projection(weighted_features)

class TemporalAttentionExtractor(BaseFeaturesExtractor):
    """
    Temporal Attention with Positional Encoding.
    Captures progressive tool wear over time using temporal embeddings.
    Combines spatial features with temporal information.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        # Temporal encoding parameters
        self.max_timesteps = 1000  # Maximum expected timesteps in an episode
        self.temporal_dim = 32  # Dimension for temporal encoding
        
        # Sinusoidal positional encoding (fixed)
        # Create sinusoidal encoding matrix [max_timesteps, temporal_dim]
        position = th.arange(0, self.max_timesteps).unsqueeze(1).float()
        div_term = th.exp(th.arange(0, self.temporal_dim, 2).float() * (-np.log(10000.0) / self.temporal_dim))
        
        pe = th.zeros(self.max_timesteps, self.temporal_dim)
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        
        # Register as buffer (not trained, but moved to device)
        self.register_buffer('positional_encoding', pe)
        
        # Learnable temporal embedding (trained)
        self.temporal_embedding = nn.Parameter(th.randn(1, self.temporal_dim))
        
        # Attention network for spatial features
        self.spatial_attention = nn.Sequential(
            nn.Linear(input_dim, features_dim),
            nn.Tanh(),
            nn.Linear(features_dim, input_dim),
            nn.Softmax(dim=1)
        )
        
        # Combine spatial and temporal
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim + self.temporal_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim)
        )
        
        # Timestep tracker (incremented each forward pass within episode)
        self.register_buffer('current_timestep', th.tensor(0))
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        
        # Apply spatial attention
        spatial_weights = self.spatial_attention(observations)
        weighted_spatial = observations * spatial_weights
        
        # Get temporal encoding for current timestep
        # Use modulo to handle cases where timestep exceeds max
        timestep_idx = int(self.current_timestep.item()) % self.max_timesteps
        temporal_encoding = self.positional_encoding[timestep_idx].unsqueeze(0).expand(batch_size, -1)
        
        # Add learnable temporal component
        temporal_features = temporal_encoding + self.temporal_embedding.expand(batch_size, -1)
        
        # Fuse spatial and temporal
        combined = th.cat([weighted_spatial, temporal_features], dim=1)
        output = self.fusion_net(combined)
        
        # Increment timestep (will be reset at episode start in practice)
        self.current_timestep += 1
        
        return output

class MultiHeadAttentionExtractor(BaseFeaturesExtractor):
    """
    Multi-Head Attention Mechanism.
    Uses multiple parallel attention heads to capture diverse feature relationships.
    Each head learns different aspects of the input.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, num_heads: int = 8):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        self.num_heads = num_heads
        self.head_dim = features_dim // num_heads
        
        assert features_dim % num_heads == 0, "features_dim must be divisible by num_heads"
        
        # Query, Key, Value projections for all heads (parallel)
        self.query_proj = nn.Linear(input_dim, features_dim)
        self.key_proj = nn.Linear(input_dim, features_dim)
        self.value_proj = nn.Linear(input_dim, features_dim)
        
        # Output projection
        self.output_proj = nn.Linear(features_dim, features_dim)
        
        # Scaling factor for dot-product attention
        self.scale = self.head_dim ** -0.5
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        
        # Project to Q, K, V
        Q = self.query_proj(observations)  # [batch, features_dim]
        K = self.key_proj(observations)    # [batch, features_dim]
        V = self.value_proj(observations)  # [batch, features_dim]
        
        # Reshape to separate heads: [batch, num_heads, head_dim]
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention for each head
        # Q @ K^T: [batch, num_heads, head_dim] @ [batch, num_heads, head_dim]^T
        # For single observation (no sequence), we compute self-attention across heads
        attn_scores = th.einsum('bhd,bhd->bh', Q, K) * self.scale  # [batch, num_heads]
        attn_weights = th.softmax(attn_scores, dim=1).unsqueeze(2)  # [batch, num_heads, 1]
        
        # Apply attention to values
        attended_values = attn_weights * V  # [batch, num_heads, head_dim]
        
        # Concatenate heads
        concat_heads = attended_values.view(batch_size, self.num_heads * self.head_dim)  # [batch, features_dim]
        
        # Final projection
        output = self.output_proj(concat_heads)
        
        return output

class SelfAttentionExtractor(BaseFeaturesExtractor):
    """
    Self-Attention with Layer Normalization.
    Transformer-style self-attention mechanism with residual connections.
    Captures feature interactions using Q-K-V architecture.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        
        # Input projection to feature dimension
        self.input_proj = nn.Linear(input_dim, features_dim)
        
        # Self-attention components
        self.query = nn.Linear(features_dim, features_dim)
        self.key = nn.Linear(features_dim, features_dim)
        self.value = nn.Linear(features_dim, features_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(features_dim)
        self.norm2 = nn.LayerNorm(features_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(features_dim, features_dim * 2),
            nn.ReLU(),
            nn.Linear(features_dim * 2, features_dim)
        )
        
        # Scaling factor
        self.scale = features_dim ** -0.5
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Project input to feature dimension
        x = self.input_proj(observations)  # [batch, features_dim]
        
        # Self-attention block with residual connection
        residual = x
        
        Q = self.query(x)  # [batch, features_dim]
        K = self.key(x)    # [batch, features_dim]
        V = self.value(x)  # [batch, features_dim]
        
        # Scaled dot-product attention
        attn_scores = (Q * K).sum(dim=1, keepdim=True) * self.scale  # [batch, 1]
        attn_weights = th.softmax(attn_scores, dim=0)  # Normalize across batch
        
        # Apply attention
        attn_output = attn_weights * V  # [batch, features_dim]
        
        # Add residual and normalize
        x = self.norm1(residual + attn_output)
        
        # Feed-forward block with residual connection
        residual = x
        ffn_output = self.ffn(x)
        x = self.norm2(residual + ffn_output)
        
        return x

class HybridAttentionExtractor(BaseFeaturesExtractor):
    """
    Hybrid Attention: Combines Temporal Encoding + Multi-Head Attention.
    Most comprehensive attention mechanism for predictive maintenance.
    Captures both temporal progression and diverse feature relationships.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64, num_heads: int = 8):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]
        self.num_heads = num_heads
        self.head_dim = features_dim // num_heads
        
        assert features_dim % num_heads == 0, "features_dim must be divisible by num_heads"
        
        # Temporal encoding
        self.max_timesteps = 1000
        self.temporal_dim = 32
        
        # Sinusoidal positional encoding
        position = th.arange(0, self.max_timesteps).unsqueeze(1).float()
        div_term = th.exp(th.arange(0, self.temporal_dim, 2).float() * (-np.log(10000.0) / self.temporal_dim))
        
        pe = th.zeros(self.max_timesteps, self.temporal_dim)
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.register_buffer('positional_encoding', pe)
        
        # Learnable temporal embedding
        self.temporal_embedding = nn.Parameter(th.randn(1, self.temporal_dim))
        
        # Project input + temporal to feature dimension
        self.input_proj = nn.Linear(input_dim + self.temporal_dim, features_dim)
        
        # Multi-head attention components
        self.query_proj = nn.Linear(features_dim, features_dim)
        self.key_proj = nn.Linear(features_dim, features_dim)
        self.value_proj = nn.Linear(features_dim, features_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(features_dim)
        
        # Output projection
        self.output_proj = nn.Linear(features_dim, features_dim)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        
        # Timestep tracker
        self.register_buffer('current_timestep', th.tensor(0))
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        
        # Get temporal encoding
        timestep_idx = int(self.current_timestep.item()) % self.max_timesteps
        temporal_encoding = self.positional_encoding[timestep_idx].unsqueeze(0).expand(batch_size, -1)
        temporal_features = temporal_encoding + self.temporal_embedding.expand(batch_size, -1)
        
        # Combine spatial and temporal
        combined = th.cat([observations, temporal_features], dim=1)
        x = self.input_proj(combined)  # [batch, features_dim]
        
        # Multi-head attention
        residual = x
        
        Q = self.query_proj(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key_proj(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value_proj(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        attn_scores = th.einsum('bhd,bhd->bh', Q, K) * self.scale
        attn_weights = th.softmax(attn_scores, dim=1).unsqueeze(2)
        
        # Apply attention
        attended_values = attn_weights * V
        concat_heads = attended_values.view(batch_size, self.num_heads * self.head_dim)
        
        # Output projection with residual
        attn_output = self.output_proj(concat_heads)
        output = self.norm(residual + attn_output)
        
        # Increment timestep
        self.current_timestep += 1
        
        return output


class StopTrainingOnMaxEpisodes(BaseCallback):
    """
    Stops training when the maximum number of episodes is reached.
    """
    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self._n_episodes = 0

    def _on_step(self) -> bool:
        # Check if episode ended
        # For VecEnv, 'dones' is a list
        if self.locals['dones'][0]:
             self._n_episodes += 1
        
        if self._n_episodes >= self.max_episodes:
            return False # Stop training
            
        return True

def calculate_weighted_score(violations, wear_margin, replacements, avg_reward=0):
    """
    Calculate a normalized weighted score for comparing agents.
    
    Scoring approach:
    - Wear Margin: Highest weight, normalized 0-1 (lowest is best)
    - Rewards: 0.3 weightage, scaled 0-1
    - Violations + Replacements: Combined 0.3 total weight
    
    Returns a score between 0-1 where 1.0 is best performance.
    """
    # Wear Margin Score (highest weight)
    # Lower is better (closer to threshold is better)
    # Use reciprocal so that small margins score high
    # wear_margin_score = 1.0 / (1.0 + wear_margin)
    wear_margin_score = (max(0, 1 - (wear_margin / WEAR_THRESHOLD))) ** 3
    
    # Rewards Score (0.3 weightage)
    # Normalize rewards to 0-1 scale
    # Assuming typical reward range is roughly -10 to +200 (scaled based on environment)
    # Linear scaling: (reward - min) / (max - min)
    # Using a reasonable range: -50 to 200
    min_reward = -50
    max_reward = 200
    rewards_score = max(0.0, min(1.0, (avg_reward - min_reward) / (max_reward - min_reward)))
    
    # Violations + Replacements Score (combined 0.3 weight)
    # Since violations are generally 0 and replacements generally 1:
    # Violations: 1.0 if <= 0.05, otherwise penalized
    # Steeper drop-off
    violations_score = 1.0 if violations <= 0.05 else 1.0 / (1.0 + violations)**2
    
    # Replacements: 1.0 if <= 1.05 (around 1 per episode), otherwise penalized
    excess = max(0, replacements - 1.0)
    replacements_score = 1.0 / (1.0 + (excess**2))
    # replacements_score = 1.0 if replacements <= 1.05 else 1.0 / (1.0 + replacements)
    
    # Combine violations and replacements into a single component (0.3 total)
    violations_replacements_score = 0.5 * violations_score + 0.5 * replacements_score
    
    # Weighted combination: wear_margin (0.4), rewards (0.3), violations+replacements (0.3)
    weighted_score = (
        0.4 * wear_margin_score +
        0.3 * rewards_score +
        0.3 * violations_replacements_score
    )
    
    return weighted_score

def calculate_steady_state_metrics(margins):
    """
    Calculate steady-state metrics for wear margin using rolling window std deviation.
    
    T_ss: Episode number where steady state begins (when variation stabilizes and remains low)
    Sigma_ss: Standard deviation of margin in the steady-state region
    
    Algorithm:
    1. Calculate rolling window std deviation (window=15 steps)
    2. Find the point where std drops to its minimum and stays within ±5-10 units
    3. This detects the transition from high variation → stable/low variation
    
    Returns:
        tuple: (T_ss, Sigma_ss) where both are floats
               Returns (len(margins), 0.0) if no steady state detected
    """
    if not margins or len(margins) < 20:
        # Not enough data
        return float(len(margins)), 0.0
    
    margins_array = np.array(margins)
    n = len(margins_array)
    
    # Rolling window size (fixed at 15 for stability detection)
    window_size = 15
    
    # Calculate rolling standard deviation
    rolling_std = []
    for i in range(window_size, n):
        window_data = margins_array[i-window_size:i]
        rolling_std.append(np.std(window_data))
    
    if not rolling_std:
        return float(n), np.std(margins_array)
    
    rolling_std = np.array(rolling_std)
    
    # Find the minimum rolling std (the most stable point)
    min_std = np.min(rolling_std)
    
    # Define "stable region": within 10% of min_std or ±5 units (whichever is larger)
    # This allows for small fluctuations while still being in steady state
    stability_band = max(min_std * 0.1, 5.0)
    stable_threshold = min_std + stability_band
    
    # Find first index where rolling_std enters the stable band and stays there
    # Use a confirmation window to ensure it's truly stable
    confirmation_length = max(10, int(n * 0.1))  # Confirm stability over 10% of remaining data
    
    t_ss = None
    for i in range(len(rolling_std)):
        # Check if current point is in stable region
        if rolling_std[i] <= stable_threshold:
            # Confirm it stays stable for the confirmation window
            end_idx = min(i + confirmation_length, len(rolling_std))
            confirmation_window = rolling_std[i:end_idx]
            
            if len(confirmation_window) > 0 and np.all(confirmation_window <= stable_threshold):
                t_ss = i + window_size  # Convert back to episode index
                break
    
    # If no steady state detected, use the point of minimum std
    if t_ss is None:
        min_idx = np.argmin(rolling_std)
        t_ss = min_idx + window_size
    
    # Calculate Sigma_ss: std dev from T_ss onwards
    steady_state_margins = margins_array[t_ss:]
    if len(steady_state_margins) > 0:
        sigma_ss = np.std(steady_state_margins)
    else:
        sigma_ss = 0.0
    
    return float(t_ss), float(sigma_ss)

def train_single_model(data_file, algo_name, lr, gm, callback_func, attention_type=None, data_filename=None):
    """
    Trains a single agent and returns the result dictionary.
    attention_type: None, 'NW', 'DL', 'Temporal', 'MultiHead', 'SelfAttn', or 'Hybrid'
    data_filename: Optional filename to use for model naming (e.g., 'SIT_10' from 'SIT_10.csv')
    """
    # 4 Algos: PPO, A2C, DQN, REINFORCE
    algos = {
        'PPO': PPO,
        'A2C': A2C,
        'DQN': DQN,
        'REINFORCE': REINFORCE
    }
    
    AlgoClass = algos[algo_name]
    
    # Construct Combo Name
    att_suffix = ""
    if attention_type == 'NW':
        att_suffix = " (NW)"
    elif attention_type == 'DL':
        att_suffix = " (DL)"
    elif attention_type == 'Temporal':
        att_suffix = " (Temporal)"
    elif attention_type == 'MultiHead':
        att_suffix = " (MultiHead)"
    elif attention_type == 'SelfAttn':
        att_suffix = " (SelfAttn)"
    elif attention_type == 'Hybrid':
        att_suffix = " (Hybrid)"
        
    combo_name = f"{algo_name}{att_suffix} | LR={lr} | G={gm}"
    print(f"Training {combo_name}...")
    
    # Create Env with TRAINING_WEAR_THRESHOLD for reward mechanism
    # We wrap in DummyVecEnv for SB3
    env = DummyVecEnv([lambda: MT_Env(data_file, TRAINING_WEAR_THRESHOLD, R1, R2, R3, R4)])
    
    # Policy kwargs - map attention type to extractor class
    policy_kwargs = {}
    if attention_type == 'NW':
        policy_kwargs = dict(
            features_extractor_class=NadarayaWatsonExtractor,
            features_extractor_kwargs=dict(features_dim=64),
        )
    elif attention_type == 'DL':
        policy_kwargs = dict(
            features_extractor_class=SimpleAttentionExtractor,
            features_extractor_kwargs=dict(features_dim=64),
        )
    elif attention_type == 'Temporal':
        policy_kwargs = dict(
            features_extractor_class=TemporalAttentionExtractor,
            features_extractor_kwargs=dict(features_dim=64),
        )
    elif attention_type == 'MultiHead':
        policy_kwargs = dict(
            features_extractor_class=MultiHeadAttentionExtractor,
            features_extractor_kwargs=dict(features_dim=64, num_heads=8),
        )
    elif attention_type == 'SelfAttn':
        policy_kwargs = dict(
            features_extractor_class=SelfAttentionExtractor,
            features_extractor_kwargs=dict(features_dim=64),
        )
    elif attention_type == 'Hybrid':
        policy_kwargs = dict(
            features_extractor_class=HybridAttentionExtractor,
            features_extractor_kwargs=dict(features_dim=64, num_heads=8),
        )
    
    try:
        # Initialize Agent
        model = AlgoClass("MlpPolicy", env, learning_rate=lr, gamma=gm, verbose=0, policy_kwargs=policy_kwargs)
        
        # Create Callback
        # Calculate Update Frequency (approx 10 updates per run)
        update_freq = max(1, EPISODES // 10)
        
        # We need a wrapper callback to inject the combo_name
        def update_wrapper(metrics):
            if callback_func:
                callback_func(combo_name, metrics)
            
        cb_streamlit = StreamlitCallback(update_wrapper, update_freq=update_freq, total_episodes=EPISODES)
        
        # Combine Callbacks
        callbacks = [cb_streamlit]
        
        # Time Management
        data_len = len(pd.read_csv(data_file))
        
        if FIXED_X_AXIS_LENGTH:
            # Train for effectively infinite steps, but stop at X episodes
            total_timesteps = 10**6 
            cb_stop = StopTrainingOnMaxEpisodes(max_episodes=EPISODES)
            callbacks.append(cb_stop)
        else:
            # Old Logic: Estimate steps
            total_timesteps = EPISODES * data_len 
        
        # Train
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
        
        # Collect Final Metrics
        # Use the streamlit callback which stores history
        cb = cb_streamlit
        avg_margin = np.mean(cb.wear_margins) if cb.wear_margins else 0
        avg_reward = np.mean(cb.episode_rewards) if cb.episode_rewards else 0
        avg_violations = np.mean(cb.violations) if cb.violations else 0
        avg_replacements = np.mean(cb.replacements) if cb.replacements else 0
        
        # Calculate Weighted Score (including rewards)
        weighted_score = calculate_weighted_score(avg_violations, avg_margin, avg_replacements, avg_reward)
        
        # Calculate Steady-State Metrics
        t_ss, sigma_ss = calculate_steady_state_metrics(cb.wear_margins)
        
        # Save Model with naming convention: algo_datafilename_Episodes_LR_Gamma[_Attention]
        # Examples: DQN_SIT_10_200_01_90, PPO_IEEE_05_200_001_99_NW, A2C_SIT_10_200_001_99_DL
        # Use data_filename if provided, otherwise extract from path
        if data_filename:
            training_filename = data_filename
        else:
            from pathlib import Path
            file_path = data_file
            training_filename = Path(file_path).stem

        ep_str = f"{EPISODES}"  # Episodes as 3-digit (e.g., 100, 200)
        lr_str = f"{lr:.0e}" if lr < 1 else f"{lr:.2f}".replace(".", "")  # Scientific notation for small LR
        gm_str = f"{int(gm * 100):02d}"
        
        # Add Finite Horizon label
        horizon_label = "FH" if FIXED_X_AXIS_LENGTH else "NF"
        
        # Build filename with optional attention suffix
        att_suffix_file = ""
        if attention_type == 'NW':
            att_suffix_file = "_NW"
        elif attention_type == 'DL':
            att_suffix_file = "_DL"
        elif attention_type == 'Temporal':
            att_suffix_file = "_Temporal"
        elif attention_type == 'MultiHead':
            att_suffix_file = "_MultiHead"
        elif attention_type == 'SelfAttn':
            att_suffix_file = "_SelfAttn"
        elif attention_type == 'Hybrid':
            att_suffix_file = "_Hybrid"
        
        model_filename = f"{algo_name}_{training_filename}_{horizon_label}_{ep_str}_{lr_str}_{gm_str}{att_suffix_file}_{date_time}"
        model_path = os.path.join("models", model_filename)
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        return {
            'Agent': f"{algo_name}{att_suffix}",
            'LR': lr,
            'Gamma': gm,
            'Avg Wear Margin': avg_margin,
            'Avg Reward': avg_reward,
            'Avg Violations': avg_violations,
            'Avg Replacements': avg_replacements,
            'Weighted Score': weighted_score,
            'T_ss': t_ss,
            'Sigma_ss': sigma_ss,
            'full_metrics': {
                'rewards': cb.episode_rewards,
                'margins': cb.wear_margins,
                'violations': cb.violations,
                'replacements': cb.replacements
            },
            'model_path': model_path,
            'model_filename': model_filename
        }


        
    except Exception as e:
        import traceback
        traceback.print_exc() # PRINT TO CONSOLE
        err_msg = f"{e}\n{traceback.format_exc()}"
        print(f"Error training {combo_name}: {err_msg}")
        return {
            'Agent': algo_name,
            'LR': lr,
            'Gamma': gm,
            'error': str(e),
            'traceback': err_msg
        }

def AutoRL(data_file, hyperparams, callback_func):
    """
    Main training loop.
    hyperparams: dict containing lists of 'learning_rate' and 'gamma'
    callback_func: function to update UI
    """
    lrs = hyperparams.get('learning_rate', [LR_DEFAULT])
    gammas = hyperparams.get('gamma', [GAMMA_DEFAULT])
    
    results = [] # List of dicts
    
    # 4 Algos: PPO, A2C, DQN, REINFORCE
    algo_names = ['PPO', 'A2C', 'DQN', 'REINFORCE']
    
    for algo_name in algo_names:
        for lr in lrs:
            for gm in gammas:
                # Default attention is None
                res = train_single_model(data_file, algo_name, lr, gm, callback_func, attention_type=None)
                results.append(res)
                    
    return results

def compare_agents(results):
    # This might return data for the UI to plot
    # Or generate a figure.
    # User wants "Display a SUPERIMPOSED plot for all four plots"
    # We will return the data structure, and App.py will handle plotting with Plotly/Altair/Matplotlib.
    return pd.DataFrame(results)

def get_best_agents_for_comparison(training_results):
    """
    Identify best plain agent and best attention-enhanced agent for comparison.
    
    Returns:
        dict: {
            'best_plain': result_dict or None,
            'best_attention': result_dict or None,
            'show_comparison': bool  # True if both exist and attention is better
        }
    """
    if not training_results:
        return {'best_plain': None, 'best_attention': None, 'show_comparison': False}
    
    # Separate plain agents from attention-enhanced agents
    plain_agents = []
    attention_agents = []
    
    for res in training_results:
        if 'error' in res:
            continue
        
        agent_name = res.get('Agent', '')
        # Check if agent has attention suffix
        has_attention = any(suffix in agent_name for suffix in ['(NW)', '(DL)', '(Temporal)', '(MultiHead)', '(SelfAttn)', '(Hybrid)'])
        
        if has_attention:
            attention_agents.append(res)
        else:
            plain_agents.append(res)
    
    # Find best plain agent (highest weighted score)
    best_plain = None
    if plain_agents:
        best_plain = max(plain_agents, key=lambda x: x.get('Weighted Score', 0))
    
    # Find best attention agent (highest weighted score)
    best_attention = None
    if attention_agents:
        best_attention = max(attention_agents, key=lambda x: x.get('Weighted Score', 0))
    
    # Determine if we should show comparison
    show_comparison = False
    if best_plain and best_attention:
        # Show comparison if attention agent is better or comparable
        show_comparison = best_attention['Weighted Score'] >= best_plain['Weighted Score']
    
    return {
        'best_plain': best_plain,
        'best_attention': best_attention,
        'show_comparison': show_comparison
    }



def get_available_models():
    """
    Returns a list of available model files in the /models folder.
    Format: algo_LR_Gamma (e.g., A2C_001_99)
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    # Find all model files (SB3 saves as .zip)
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.zip'):
            models.append(file.replace('.zip', ''))
    
    return sorted(models)

# $$$ Evaluation Results with Ideal Action Region (IAR) Logic $$$
def adjusted_evaluate_model(model_path, data_file, wear_threshold=None):
    """
    Adjusted evaluation with Ideal Action Region (IAR) logic.
    
    Uses WEAR_THRESHOLD (ISO standard) for evaluation and display.
    IAR (Ideal Action Region) = [0.9 * WEAR_THRESHOLD, 1.05 * WEAR_THRESHOLD]
    
    Logic:
    1. Any REPLACEMENT before IAR is ignored (recorded as CONTINUE in display)
    2. Replacements within/beyond IAR are counted and displayed
    3. If replacement is made, no violation (replacement was suggested)
    4. If NO valid replacement in IAR found, force one at random point (MODEL_OVERRIDE = True)
    
    Returns:
    {
        'timesteps': [list of timesteps],
        'tool_wear': [list of tool wear values],
        'actions': [list of adjusted actions (0 or 1)],
        'wear_threshold': WEAR_THRESHOLD value (ISO standard),
        'total_replacements': number of valid replacements,
        'threshold_violations': number of violations (wear > WEAR_THRESHOLD with no replacement),
        'model_override': bool - True if forced replacement was added,
        'override_timestep': timestep where override occurred (None if no override)
    }
    """
    try:
        # Load data from file
        data = pd.read_csv(data_file)
        
        # Use WEAR_THRESHOLD (ISO standard) for evaluation
        eval_wear_threshold = WEAR_THRESHOLD
        
        # Determine algo from model filename
        model_name = os.path.basename(model_path)
        algo_name = model_name.split('_')[0]
        
        # Load model
        algos = {
            'PPO': PPO,
            'A2C': A2C,
            'DQN': DQN
        }
        
        AlgoClass = algos.get(algo_name, A2C)
        model = AlgoClass.load(model_path)
        
        # Create environment for feature extraction (use TRAINING_WEAR_THRESHOLD)
        env = MT_Env(data_file, TRAINING_WEAR_THRESHOLD)
        
        # Define IAR bounds with realistic random variation (based on ISO standard)
        IAR_lower = (1.0 - IAR_RANGE) * eval_wear_threshold + np.random.uniform(0.0, 1.0)
        IAR_upper = (1.0 + IAR_RANGE) * eval_wear_threshold + np.random.uniform(0.0, 1.0)
        
        # Track evaluation data
        timesteps = []
        tool_wear_values = []
        actions_taken = []
        total_replacements = 0
        threshold_violations = 0
        model_override = False
        override_timestep = None
        found_valid_replacement = False
        
        expected_shape = model.observation_space.shape[0]
        
        # Process all data points
        for timestep, idx in enumerate(range(len(data))):
            try:
                # Get observation
                obs = data.iloc[idx][env.features].values.astype(np.float32)
                
                if len(obs) != expected_shape:
                    raise ValueError(
                        f"Feature mismatch at row {idx}! Expected {expected_shape} features but got {len(obs)}.\n"
                        f"Features: {env.features}\n"
                        f"Data shape: {data.shape}"
                    )
                
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                current_wear = data.iloc[idx]['tool_wear']
                
                # ADJUSTMENT LOGIC: Filter early replacements
                adjusted_action = int(action)
                
                if action == 0:  # Model predicted REPLACE
                    if current_wear < IAR_lower:
                        # Before IAR - ignore replacement, treat as CONTINUE
                        adjusted_action = 1
                    else:
                        # Within or beyond IAR - valid replacement
                        found_valid_replacement = True
                        total_replacements += 1
                        adjusted_action = 0
                else:  # action == 1 (CONTINUE)
                    # Check for violations (wear > WEAR_THRESHOLD without replacement)
                    if current_wear > eval_wear_threshold and (total_replacements == 0):
                        threshold_violations += 1
                
                # Store adjusted data
                timesteps.append(timestep)
                tool_wear_values.append(float(current_wear))
                actions_taken.append(adjusted_action)
                
            except Exception as e:
                raise Exception(f"Error processing row {idx}: {str(e)}")
        
        # OVERRIDE LOGIC: If no valid replacement in IAR, force one at random timestep within band
        if not found_valid_replacement:
            # Find all timesteps where wear is within IAR band
            IAR_band_indices = []
            for idx, wear in enumerate(tool_wear_values):
                if IAR_lower <= wear <= IAR_upper:
                    IAR_band_indices.append(idx)
            
            # Randomly select one index from the IAR band, or use first IAR point if band is empty
            if IAR_band_indices:
                selected_idx = np.random.choice(IAR_band_indices)
            else:
                # Fallback: find first timestep where wear >= IAR lower bound
                selected_idx = None
                for idx, wear in enumerate(tool_wear_values):
                    if wear >= IAR_lower:
                        selected_idx = idx
                        break
            
            if selected_idx is not None:
                actions_taken[selected_idx] = 0  # Force REPLACE
                total_replacements = 1
                model_override = True
                override_timestep = timesteps[selected_idx]
        
        # Lambda (λ): Difference in timesteps between first replacement and first threshold crossing
        t_FR = next((i for i, a in enumerate(actions_taken) if a == 0), None)
        T_wt = next((i for i, w in enumerate(tool_wear_values) if w > eval_wear_threshold), None)
        # print(f"t_FR: {t_FR}, T_wt: {T_wt}")
        # T_wt = int((1.0-IAR_RANGE)*T_wt)    # Safe to change tool before 95 % of T_wt

        lambda_metric = None
        tool_usage_pct = None
        if t_FR is not None and T_wt is not None:
            lambda_metric = T_wt - t_FR
            try:
                tool_usage_pct = t_FR / T_wt if T_wt > 0 else None
            except Exception:
                tool_usage_pct = None

        return {
            'timesteps': timesteps,
            'tool_wear': tool_wear_values,
            'actions': actions_taken,
            'wear_threshold': eval_wear_threshold,
            'T_wt': T_wt,
            't_FR': t_FR,
            'lambda': lambda_metric,
            'tool_usage_pct': tool_usage_pct,
            'threshold_violations': threshold_violations,
            'model_override': model_override,
            'override_timestep': override_timestep,
            'IAR_lower': IAR_lower,
            'IAR_upper': IAR_upper
        }
    
    except Exception as e:
        raise Exception(f"Adjusted evaluation failed: {str(e)}")


def plot_sensor_data(data_file, wear_threshold=None):
    """
    Plots sensor data based on the detected schema (IEEE or SIT).
    
    For the tool_wear plot, displays as dual-axis chart:
    - Left axis (yellow): Tool Wear values (0-320 range)
    - Dotted red line: Wear Threshold (horizontal) - uses WEAR_THRESHOLD constant
    - Right axis (blue): ACTION_CODE (0-1 range)
    """
    try:
        # Use WEAR_THRESHOLD constant for display (ISO standard)
        display_threshold = WEAR_THRESHOLD
        
        # Load Data
        data = pd.read_csv(data_file)
        columns = data.columns
        
        # Detect Schema
        schema = 'UNKNOWN'
        if 'force_x' in columns and 'acoustic_emission_rms' in columns:
            schema = 'IEEE'
        elif 'Vib_Spindle' in columns and 'Sound_Spindle' in columns:
            schema = 'SIT'
            
        # Determine features to plot
        features_to_plot = []
        if schema == 'IEEE':
            # IEEE features
            features_to_plot = ['force_x', 'force_y', 'force_z', 'vibration_x', 'vibration_y', 'vibration_z', 'acoustic_emission_rms', 'tool_wear']
        elif schema == 'SIT':
            features_to_plot = ['Vib_Spindle', 'Vib_Table', 'Sound_Spindle', 'Sound_table', 'X_Load_Cell', 'Y_Load_Cell', 'Z_Load_Cell', 'Current', 'tool_wear']
        else:
            # Fallback
            features_to_plot = [c for c in columns if c != 'ACTION_CODE'][:8]
            
        # Filter available
        features_to_plot = [f for f in features_to_plot if f in columns]
        
        if not features_to_plot:
             return None

        # Find where tool_wear actually appears in the features list
        tool_wear_index = features_to_plot.index('tool_wear') if 'tool_wear' in features_to_plot else -1
        
        # Create specs for subplots - only the tool_wear subplot has secondary_y
        # specs should be a 2D list of dictionaries: [[{}, {}, {}], [{}, {}, {}], [{}, {}, {}]]
        specs = [[{"secondary_y": (i*3 + j == tool_wear_index)} for j in range(3)] for i in range(3)]
        
        # Create Subplots (3x3 Grid) with secondary_y support for tool_wear
        fig = make_subplots(
            rows=3, cols=3, 
            subplot_titles=features_to_plot, 
            vertical_spacing=0.05, 
            horizontal_spacing=0.02,
            specs=specs
        )
        
        for i, feature in enumerate(features_to_plot):
            row = (i // 3) + 1
            col = (i % 3) + 1
            if row > 3: break # Limit to 9 plots
            
            # Special handling for tool_wear: dual-axis with threshold line
            if feature == 'tool_wear':
                # Add tool wear as yellow line on primary y-axis
                fig.add_trace(
                    go.Scatter(y=data[feature], name=feature, mode='lines', 
                               line=dict(color='yellow')),
                    row=row, col=col, secondary_y=False
                )
                
                # Add wear threshold as dotted red horizontal line on primary y-axis
                fig.add_hline(y=display_threshold, line_dash="dot", line_color="red",
                              annotation_text=f"Wear Threshold ({display_threshold})", row=row, col=col, secondary_y=False)
                
                # Add ACTION_CODE as blue line on secondary y-axis if available
                if 'ACTION_CODE' in columns:
                    fig.add_trace(
                        go.Scatter(y=data['ACTION_CODE'], name='ACTION_CODE', mode='lines',
                                   line=dict(color='blue')),
                        row=row, col=col, secondary_y=True
                    )
                    # Update secondary y-axis title
                    fig.update_yaxes(title_text="ACTION", secondary_y=True, row=row, col=col)
                
                # Update primary y-axis title
                fig.update_yaxes(title_text="Tool Wear", secondary_y=False, row=row, col=col)
            else:
                # For light theme, we can use standard colors. Plotly cycles them automatically.
                fig.add_trace(
                    go.Scatter(y=data[feature], name=feature, mode='lines'),
                    row=row, col=col
                )
            
        fig.update_layout(
            height=800, 
            showlegend=False, 
            template="plotly_white", # Light background
            margin=dict(l=20, r=20, t=50, b=20) # Tight margins
        ) 
        return fig

    except Exception as e:
        print(f"Error plotting: {e}")
        return None


"""
REINFORCE (Monte Carlo Policy Gradient) Algorithm
Compatible with Stable Baselines3 API

This implementation follows the SB3 architecture to allow seamless integration
with existing PPO, A2C, and DQN algorithms in the AutoRL system.
"""

import numpy as np
import torch as th
from torch.nn import functional as F
from typing import Any, Dict, Optional, Type, Union, TypeVar
from gymnasium import spaces

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

SelfREINFORCE = TypeVar("SelfREINFORCE", bound="REINFORCE")

class REINFORCE(OnPolicyAlgorithm):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm
    
    Based on the policy gradient theorem with Monte Carlo returns.
    Uses complete episode returns (no bootstrapping).
    
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from
    :param learning_rate: The learning rate
    :param n_steps: The number of steps to run for each environment per update
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        (not used in pure REINFORCE, kept for compatibility)
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation (used if baseline is enabled)
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_baseline: Whether to use a value function as baseline (reduces variance)
    :param normalize_advantage: Whether to normalize the advantage
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """
    
    # Register policy aliases (required for SB3)
    policy_aliases: Dict[str, Type[ActorCriticPolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,  # For REINFORCE, we use 1.0 (pure Monte Carlo)
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_baseline: bool = True,
        normalize_advantage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,  # REINFORCE doesn't use SDE
            sde_sample_freq=-1,
            rollout_buffer_class=RolloutBuffer,
            rollout_buffer_kwargs=None,
            stats_window_size=100,
            tensorboard_log=None,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        
        self.use_baseline = use_baseline
        self.normalize_advantage = normalize_advantage

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        Implements the REINFORCE (Monte Carlo Policy Gradient) algorithm.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        # Collect metrics for logging
        entropy_losses = []
        pg_losses = []
        value_losses = []
        
        # Process all episodes in the buffer
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            
            # Convert to long for discrete actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()
            
            # Evaluate actions with current policy
            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions
            )
            values = values.flatten()
            
            # REINFORCE uses returns (G_t) instead of advantages
            # The rollout buffer computes returns for us
            returns = rollout_data.returns
            
            if self.use_baseline:
                # Advantage = Return - Baseline (value function)
                advantages = returns - values.detach()
            else:
                # Pure REINFORCE: use returns directly
                advantages = returns
            
            # Normalize advantage (optional, can help with stability)
            if self.normalize_advantage and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy gradient loss
            # L = -E[log π(a|s) * G_t] (negative because we want to maximize)
            policy_loss = -(log_prob * advantages).mean()
            
            # Value loss (only if using baseline)
            if self.use_baseline:
                value_loss = F.mse_loss(returns, values)
            else:
                value_loss = th.tensor(0.0).to(self.device)
            
            # Entropy loss (for exploration)
            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)
            
            # Total loss
            loss = policy_loss + self.ent_coef * entropy_loss
            if self.use_baseline:
                loss += self.vf_coef * value_loss
            
            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()
            
            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            
            # Collect metrics
            pg_losses.append(policy_loss.item())
            entropy_losses.append(entropy_loss.item())
            value_losses.append(value_loss.item())
        
        self._n_updates += 1
        
        # Log training metrics
        # Ensure values/returns are numpy arrays (handle both torch.Tensor and numpy.ndarray)
        values = self.rollout_buffer.values
        returns = self.rollout_buffer.returns
        if isinstance(values, th.Tensor):
            values_np = values.flatten().cpu().numpy()
        else:
            values_np = np.asarray(values).flatten()
        if isinstance(returns, th.Tensor):
            returns_np = returns.flatten().cpu().numpy()
        else:
            returns_np = np.asarray(returns).flatten()
        explained_var = explained_variance(values_np, returns_np)
        
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 1,
        tb_log_name: str = "REINFORCE",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """
        Learn the policy for a given number of timesteps.
        
        This is the main training loop that:
        1. Collects rollouts (episodes)
        2. Computes returns
        3. Updates the policy
        """
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )



# def evaluate_model(model_path, data_file, wear_threshold=None):
#     """
#     Evaluate a trained model on test data.
    
#     NOTE: This is a HISTORICAL REPLAY evaluation - we process all data points
#     regardless of whether the tool exceeds the threshold. This is different from
#     training episodes which terminate early.
    
#     Uses WEAR_THRESHOLD (ISO standard) for display and violation detection.
    
#     Returns:
#     {
#         'timesteps': [list of timesteps],
#         'tool_wear': [list of tool wear values],
#         'actions': [list of actions taken (0 or 1)],
#         'wear_threshold': WEAR_THRESHOLD value,
#         'total_replacements': number of replacements,
#         'threshold_violations': number of times WEAR_THRESHOLD was exceeded
#     }
#     """
#     try:
#         # Use WEAR_THRESHOLD (ISO standard) for evaluation
#         eval_wear_threshold = WEAR_THRESHOLD
        
#         # Determine algo from model filename
#         model_name = os.path.basename(model_path)
#         algo_name = model_name.split('_')[0]  # Extract algo (PPO, A2C, or DQN)
        
#         # Load model
#         algos = {
#             'PPO': PPO,
#             'A2C': A2C,
#             'DQN': DQN,
#             'REINFORCE': REINFORCE
#         }
        
#         AlgoClass = algos.get(algo_name, A2C)
#         model = AlgoClass.load(model_path)
        
#         # Load data directly for evaluation (not wrapped in environment)
#         data = pd.read_csv(data_file)
        
#         # Create environment just for feature extraction and observation building
#         # Use TRAINING_WEAR_THRESHOLD for feature extraction
#         env = MT_Env(data_file, TRAINING_WEAR_THRESHOLD)
        
#         # Validate observation shape
#         expected_shape = model.observation_space.shape[0]
        
#         # Track evaluation data
#         timesteps = []
#         tool_wear_values = []
#         actions_taken = []
#         total_replacements = 0
#         threshold_violations = 0
        
#         # Process all data points (historical replay - no early termination)
#         for timestep, idx in enumerate(range(len(data))):
#             try:
#                 # Get the observation for this row
#                 obs = data.iloc[idx][env.features].values.astype(np.float32)
                
#                 # Validate shape
#                 if len(obs) != expected_shape:
#                     raise ValueError(
#                         f"Feature mismatch at row {idx}! Expected {expected_shape} features but got {len(obs)}.\n"
#                         f"Features: {env.features}\n"
#                         f"Data shape: {data.shape}"
#                     )
                
#                 # Get action from model
#                 action, _ = model.predict(obs, deterministic=True)
                
#                 # Store data
#                 current_wear = data.iloc[idx]['tool_wear']
#                 timesteps.append(timestep)
#                 tool_wear_values.append(float(current_wear))
#                 actions_taken.append(int(action))
                
#                 # Track metrics
#                 if action == 0:  # REPLACE
#                     total_replacements += 1
                
#                 if (current_wear > eval_wear_threshold) and (total_replacements == 0):
#                     threshold_violations += 1
                    
#             except Exception as e:
#                 raise Exception(f"Error processing row {idx}: {str(e)}")
        
#         # Lambda (λ): Difference in timesteps between first replacement and first threshold crossing
#         t_FR = next((i for i, a in enumerate(actions_taken) if a == 0), None)
#         T_wt = next((i for i, w in enumerate(tool_wear_values) if w > eval_wear_threshold), None)
#         lambda_metric = None
#         tool_usage_pct = None
#         if t_FR is not None and T_wt is not None:
#             lambda_metric = T_wt - t_FR
#             try:
#                 tool_usage_pct = (t_FR / T_wt) if T_wt > 0 else None
#             except Exception:
#                 tool_usage_pct = None

#         return {
#             'timesteps': timesteps,
#             'tool_wear': tool_wear_values,
#             'actions': actions_taken,
#             'wear_threshold': eval_wear_threshold,
#             'T_wt': T_wt,
#             't_FR': t_FR,
#             'lambda': lambda_metric,
#             'tool_usage_pct': tool_usage_pct
#         }
    
#     except Exception as e:
#         # Re-raise with context
#         raise Exception(f"Evaluation failed: {str(e)}")
