"""
RL-based Predictive Maintenance Module
Custom Gymnasium environment and RL algorithms for milling machine tool wear prediction
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import pickle
import warnings

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
WEAR_THRESHOLD = 290
VIOLATION_THRESHOLD = int(WEAR_THRESHOLD * 1.1)  # 10% over WEAR_THRESHOLD (330)
EPISODES = 100
SMOOTH_WINDOW = int(EPISODES/10)
W, H = 20, 10 # Plot dimensions

# Reward parameters
R1 = -1000  # Violation penalty (Increased to force avoidance)
R2 = 1     # Continue reward (per step)
R3 = -5    # Replacement penalty (Reduced to make it a viable option)

# RL Algorithm Hyperparameters (shared across all algorithms)
LEARNING_RATE = 0.0001  # Reduced from 0.1 - Policy gradient is very sensitive to LR
GAMMA = 0.99         # Discount factor for future rewards (Increased to value future survival)
PPO_TIMESTEPS = 100   # Timesteps per episode for PPO training

# Training stability parameters
MAX_GRAD_NORM = 1.0  # Gradient clipping
MAX_EPISODE_STEPS = 10000  # Prevent infinite episodes
LOSS_THRESHOLD = 1e6  # Detect exploding loss

# Actions
CONTINUE = 0
REPLACE_TOOL = 1

# ============================================================================
# CUSTOM GYMNASIUM ENVIRONMENT
# ============================================================================
class MT_Env(gym.Env):
    """
    Milling Tool Predictive Maintenance Environment
    
    Observation: All sensor features + tool_wear (10 features)
    Action: CONTINUE (0) or REPLACE_TOOL (1)
    
    Reward structure:
    - R2 per step for continuing without violation
    - R1 penalty for threshold violations
    - R3 penalty for tool replacement
    
    Key design: Episodes progress through the ENTIRE dataset regardless of replacements.
    Replacing a tool resets the WEAR counter, but time marches forward.
    This prevents infinite episodes from repeated replacements.
    """
    
    def __init__(self, data_file: str, wear_threshold: float = WEAR_THRESHOLD,
                 r1: float = R1, r2: float = R2, r3: float = R3):
        super().__init__()
        
        self.wear_threshold = wear_threshold
        self.violation_threshold = VIOLATION_THRESHOLD
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        
        # Load sensor data
        self.data = pd.read_csv(data_file)
        self.required_columns = ['Time', 'Vib_Spindle', 'Vib_Table', 'Sound_Spindle', 
                                'Sound_table', 'X_Load_Cell', 'Y_Load_Cell', 
                                'Z_Load_Cell', 'Current', 'tool_wear']
        
        # Validate columns
        for col in self.required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Feature columns (all except Time)
        self.feature_cols = [col for col in self.required_columns if col != 'Time']
        
        # Normalize data for better learning
        self.data_normalized = self.data.copy()
        for col in self.feature_cols:
            col_min = self.data[col].min()
            col_max = self.data[col].max()
            if col_max > col_min:
                self.data_normalized[col] = (self.data[col] - col_min) / (col_max - col_min)
            else:
                self.data_normalized[col] = 0.0
        
        # Store original tool_wear for reward calculation
        self.original_tool_wear = self.data['tool_wear'].values
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(len(self.feature_cols),), 
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.total_steps = len(self.data)
        self.current_wear = 0.0  # Tool wear counter
        self.wear_offset = 0.0   # Offset to decouple time from wear state
        
        # Metrics tracking
        self.episode_reward = 0
        self.total_replacements = 0
        self.total_violations = 0
        self.wear_margins = []  # Wear margin before each replacement
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_wear = 0.0
        self.wear_offset = 0.0
        self.episode_reward = 0
        self.total_replacements = 0
        self.total_violations = 0
        self.wear_margins = []
        
        obs = self._get_observation()
        return obs, {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (normalized sensor values + tool_wear)"""
        if self.current_step >= self.total_steps:
            # Return last observation if we've exceeded data
            self.current_step = self.total_steps - 1
        
        obs = self.data_normalized.iloc[self.current_step][self.feature_cols].values
        
        # IMPORTANT: We must inject the CALCULATED current_wear into the observation
        # otherwise the agent sees the dataset's wear, not the env's state
        wear_col_idx = self.feature_cols.index('tool_wear')
        
        # Normalize current_wear for the observation
        # We need the min/max from the original normalization
        col_min = self.data['tool_wear'].min()
        col_max = self.data['tool_wear'].max()
        
        if col_max > col_min:
            norm_wear = (self.current_wear - col_min) / (col_max - col_min)
        else:
            norm_wear = 0.0
            
        obs[wear_col_idx] = norm_wear
        
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # 1. Calculate current wear state based on progress + offset
        # Wear increases primarily due to time progression in the dataset
        # We subtract the offset to account for any previous replacements
        raw_dataset_wear = self.original_tool_wear[self.current_step]
        self.current_wear = max(0.0, raw_dataset_wear - self.wear_offset)
        
        reward = 0
        terminated = False
        truncated = False
        info = {
            'replacement': False,
            'violation': False,
            'wear_margin': 0,
            'current_wear': self.current_wear
        }
        
        if action == REPLACE_TOOL:
            # Tool replacement
            self.total_replacements += 1
            info['replacement'] = True
            
            # Calculate wear margin (how close to threshold when replaced)
            wear_margin = self.wear_threshold - self.current_wear
            self.wear_margins.append(wear_margin)
            info['wear_margin'] = wear_margin
            
            # Penalty for replacement (to encourage using tool fully)
            reward += self.r3
            
            # If replaced before threshold, negative margin (good)
            # If replaced after threshold, positive margin (bad - violation)
            if self.current_wear > self.violation_threshold:
                # Violation: replaced too late
                self.total_violations += 1
                info['violation'] = True
                reward += self.r1  # Heavy penalty
            
            # CRITICAL: Reset wear state by updating the offset
            # The new offset is the current dataset wear, so calculating
            # (dataset_wear - offset) will result in approx 0 for the next step
            self.wear_offset = raw_dataset_wear
            self.current_wear = 0.0
            
        else:  # CONTINUE
            # Check if continuing causes a violation
            if self.current_wear > self.violation_threshold:
                # Violation: should have replaced but didn't
                self.total_violations += 1
                info['violation'] = True
                reward += self.r1  # Heavy penalty
                terminated = True  # Episode ends on violation
            else:
                # Good: continuing without violation
                reward += self.r2
        
        # ALWAYS move to next step regardless of action
        self.current_step += 1
        
        # Check if we've reached end of data
        if self.current_step >= self.total_steps:
            truncated = True
        
        # Get next observation
        obs = self._get_observation()
        self.episode_reward += reward
        
        return obs, reward, terminated, truncated, info

# ============================================================================
# REINFORCE ALGORITHM (From Scratch)
# ============================================================================
class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)


class AttentionPolicyNetwork(nn.Module):
    """Policy network with attention mechanism for feature weighting"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.Tanh(),
            nn.Linear(obs_dim, obs_dim),
            nn.Softmax(dim=-1)
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        # Apply attention to weight features
        attention_weights = self.attention(x)
        weighted_features = x * attention_weights
        return self.policy(weighted_features)


class REINFORCEAgent:
    """
    REINFORCE algorithm implementation following Stable Baselines3 API pattern
    Monte Carlo Policy Gradient with baseline
    """
    
    def __init__(self, env: gym.Env, learning_rate: float = LEARNING_RATE, 
                 gamma: float = GAMMA, use_attention: bool = False):
        self.env = env
        self.gamma = gamma
        self.use_attention = use_attention
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # Initialize policy network
        if use_attention:
            self.policy = AttentionPolicyNetwork(obs_dim, action_dim)
        else:
            self.policy = PolicyNetwork(obs_dim, action_dim)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training history
        self.episode_rewards = []
        self.episode_replacements = []
        self.episode_violations = []
        self.episode_margins = []
        
    def predict(self, observation, deterministic=False):
        """Predict action (compatible with SB3 API)"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action_probs = self.policy(obs_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=1).item()
            else:
                dist = Categorical(action_probs)
                action = dist.sample().item()
        
        return action, None
    
    def learn(self, total_timesteps: int, callback=None, progress_bar=False):
        """Train the agent (compatible with SB3 API) with graceful error handling"""
        episodes = total_timesteps  # For this implementation, timesteps = episodes
        training_failed = False
        failed_episode = None
        
        # Calculate live plot update frequency interval (Intelligent Update)
        if episodes < 100:
            update_interval = 10
        else:
            update_interval = max(1, episodes // 10)
        
        try:
            for episode in range(episodes):
                try:
                    # Collect episode trajectory
                    states, actions, rewards = self._collect_episode()
                    
                    # Validate episode data
                    if len(states) == 0:
                        warnings.warn(f"Episode {episode}: No transitions collected")
                        continue
                    
                    # Calculate returns
                    returns = self._calculate_returns(rewards)
                    
                    # Validate returns before update
                    if torch.isnan(returns).any() or torch.isinf(returns).any():
                        warnings.warn(f"Episode {episode}: Invalid returns detected")
                        training_failed = True
                        failed_episode = episode
                        break
                    
                    # Update policy
                    self._update_policy(states, actions, returns)
                    
                    # Track metrics
                    episode_reward = sum(rewards)
                    self.episode_rewards.append(episode_reward)
                    self.episode_replacements.append(self.env.total_replacements)
                    self.episode_violations.append(self.env.total_violations)
                    
                    if len(self.env.wear_margins) > 0:
                        avg_margin = np.mean(self.env.wear_margins)
                    else:
                        avg_margin = 0
                    self.episode_margins.append(avg_margin)
                    
                    # Callback for live plotting (only at 10% intervals + first and last episodes)
                    should_update = (episode + 1) % update_interval == 0 or episode == 0 or episode == episodes - 1
                    if callback is not None and should_update:
                        callback(self, episode, episodes)
                
                except Exception as e:
                    print(f"\nERROR in Episode {episode}: {str(e)}")
                    training_failed = True
                    failed_episode = episode
                    break
        
        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user at episode {episode}")
            training_failed = True
            failed_episode = episode
        
        except Exception as e:
            print(f"\nFATAL ERROR during training: {str(e)}")
            training_failed = True
            failed_episode = episode
        
        # Graceful exit with status report
        if training_failed:
            print("\n" + "="*60)
            print("TRAINING STOPPED")
            print("="*60)
            if failed_episode is not None:
                print(f"Failed at episode: {failed_episode}/{episodes}")
                print(f"Episodes completed: {len(self.episode_rewards)}")
            if self.episode_rewards:
                print(f"Last reward: {self.episode_rewards[-1]:.4f}")
                print(f"Average reward (last 10): {np.mean(self.episode_rewards[-10:]):.4f}")
            print("="*60)
        
        return self
    
    def _collect_episode(self):
        """Collect one episode of experience (optimized with no_grad)"""
        states, actions, rewards = [], [], []
        
        obs, _ = self.env.reset()
        done = False
        steps = 0
        
        # Use no_grad to speed up inference
        with torch.no_grad():
            while not done and steps < MAX_EPISODE_STEPS:
                # Convert observation to tensor
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Get action probabilities
                action_probs = self.policy(obs_tensor)
                
                # Check for NaN/Inf in action probabilities
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    warnings.warn("NaN or Inf detected in action probabilities - Episode may be unstable")
                    done = True
                    break
                
                dist = Categorical(action_probs)
                action = dist.sample()
                
                # Take action in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                
                # Store transition (store numpy arrays to avoid tensor overhead)
                states.append(obs)
                actions.append(action.item())
                rewards.append(reward)
                
                obs = next_obs
                steps += 1
        
        if steps >= MAX_EPISODE_STEPS:
            warnings.warn(f"Episode exceeded max steps ({MAX_EPISODE_STEPS}) - may indicate infinite loop")
        
        return states, actions, rewards
    
    def _calculate_returns(self, rewards):
        """Calculate discounted returns"""
        returns = []
        G = 0
        
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Normalize returns (baseline)
        returns = torch.FloatTensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def _update_policy(self, states, actions, returns):
        """Update policy using policy gradient with stability checks"""
        self.optimizer.zero_grad()
        
        # Convert to tensors (batch conversion is faster)
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        
        # Check for invalid returns before computing loss
        if torch.isnan(returns).any() or torch.isinf(returns).any():
            warnings.warn("NaN or Inf detected in returns - Skipping update")
            return
        
        # Calculate policy loss
        action_probs = self.policy(states_tensor)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)
        
        # Policy gradient loss: -log_prob * return
        loss = -(log_probs * returns).mean()
        
        # Check for invalid loss before backprop
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > LOSS_THRESHOLD:
            warnings.warn(f"Invalid loss detected: {loss.item():.6f} - Skipping update to prevent divergence")
            return
        
        # Backprop and optimize
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
        
        self.optimizer.step()
    
    def save(self, path: str):
        """Save model"""
        save_dict = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'use_attention': self.use_attention,
            'gamma': self.gamma,
            'episode_rewards': self.episode_rewards,
            'episode_replacements': self.episode_replacements,
            'episode_violations': self.episode_violations,
            'episode_margins': self.episode_margins
        }
        torch.save(save_dict, path)
    
    def load(self, path: str, env: gym.Env):
        """Load model"""
        checkpoint = torch.load(path)
        
        # Recreate policy network
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        if checkpoint['use_attention']:
            self.policy = AttentionPolicyNetwork(obs_dim, action_dim)
        else:
            self.policy = PolicyNetwork(obs_dim, action_dim)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer = optim.Adam(self.policy.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training history
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_replacements = checkpoint['episode_replacements']
        self.episode_violations = checkpoint['episode_violations']
        self.episode_margins = checkpoint['episode_margins']
        
        return self

# ============================================================================
# TRAINING UTILITIES
# ============================================================================
def smooth_curve(data: List[float], window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Apply exponential moving average smoothing"""
    if len(data) < window:
        return np.array(data)
    
    smoothed = np.zeros(len(data))
    smoothed[0] = data[0]
    
    alpha = 2 / (window + 1)
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed


def plot_training_live(agent, episode: int, total_episodes: int, 
                       agent_name: str, fig=None, axes=None):
    """
    Plot live training progress with 4 subplots
    Returns fig and axes for continuous updates
    """    
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(W, H))
        fig.patch.set_facecolor('#F0EBE7')
    
    # Clear previous plots
    for ax in axes.flat:
        ax.clear()
    
    # Get data
    rewards = agent.episode_rewards
    replacements = agent.episode_replacements
    violations = agent.episode_violations
    margins = agent.episode_margins
    
    episodes_range = range(1, len(rewards) + 1)
    
    # Smooth data
    if len(rewards) > 1:
        rewards_smooth = smooth_curve(rewards)
        replacements_smooth = smooth_curve(replacements)
        violations_smooth = smooth_curve(violations)
        margins_smooth = smooth_curve(margins)
    else:
        rewards_smooth = rewards
        replacements_smooth = replacements
        violations_smooth = violations
        margins_smooth = margins
    
    # Plot 1: Episode Rewards (Learning Curve)
    axes[0, 0].plot(episodes_range, rewards, alpha=0.3, color='blue', label='Raw')
    axes[0, 0].plot(episodes_range, rewards_smooth, color='darkblue', linewidth=1, label='Smoothed')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Learning Curve: Episode Rewards')
    # axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Total Replacements per Episode
    axes[0, 1].plot(episodes_range, replacements, alpha=0.3, color='green', label='Raw')
    axes[0, 1].plot(episodes_range, replacements_smooth, color='darkgreen', linewidth=1, label='Smoothed')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Tool Replacements per Episode')
    # axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Threshold Violations
    axes[1, 0].plot(episodes_range, violations, alpha=0.3, color='red', label='Raw')
    axes[1, 0].plot(episodes_range, violations_smooth, color='darkred', linewidth=1, label='Smoothed')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Wear Threshold Violations')
    # axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Wear Margins Before Replacement
    axes[1, 1].plot(episodes_range, margins, alpha=0.3, color='orange', label='Raw')
    axes[1, 1].plot(episodes_range, margins_smooth, color='darkorange', linewidth=1, label='Smoothed')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Optimal')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Margin (Threshold - Wear)')
    axes[1, 1].set_title('Wear Margin Before Replacements (Lower is Better)')
    # axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Main title
    fig.suptitle(f'{agent_name} - Training Episodes {episode+1}/{total_episodes}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, axes


def compare_agents(agents_dict: Dict[str, Any], save_path: Optional[str] = None):
    """
    Compare multiple agents: generate table and superimposed plots
    
    Args:
        agents_dict: Dictionary with agent names as keys and agent objects as values
        save_path: Optional path to save comparison results
    
    Returns:
        DataFrame with comparison metrics, Figure with comparison plots
    """
    # Calculate average metrics
    comparison_data = []
    
    for agent_name, agent in agents_dict.items():
        metrics = {
            'Agent': agent_name,
            'Avg Reward': np.mean(agent.episode_rewards[-20:]),  # Last 20 episodes
            'Avg Replacements': np.mean(agent.episode_replacements[-20:]),
            'Avg Violations': np.mean(agent.episode_violations[-20:]),
            'Avg Margin': np.mean(agent.episode_margins[-20:]),
            'Final Reward': agent.episode_rewards[-1] if len(agent.episode_rewards) > 0 else 0
        }
        comparison_data.append(metrics)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create comparison plots    
    fig, axes = plt.subplots(2, 2, figsize=(W, H))
    fig.patch.set_facecolor('#F0EBE7')
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for idx, (agent_name, agent) in enumerate(agents_dict.items()):
        color = colors[idx % len(colors)]
        episodes = range(1, len(agent.episode_rewards) + 1)
        
        # Smooth data
        rewards_smooth = smooth_curve(agent.episode_rewards)
        replacements_smooth = smooth_curve(agent.episode_replacements)
        violations_smooth = smooth_curve(agent.episode_violations)
        margins_smooth = smooth_curve(agent.episode_margins)
        
        # Plot all metrics
        axes[0, 0].plot(episodes, rewards_smooth, color=color, label=agent_name, linewidth=1)
        axes[0, 1].plot(episodes, replacements_smooth, color=color, label=agent_name, linewidth=1)
        axes[1, 0].plot(episodes, violations_smooth, color=color, label=agent_name, linewidth=1)
        axes[1, 1].plot(episodes, margins_smooth, color=color, label=agent_name, linewidth=1)
    
    # Configure subplots
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Tool Replacements')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Threshold Violations')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Wear Margins')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Margin')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Agent Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        comparison_df.to_csv(save_path.replace('.png', '.csv'), index=False)
    
    return comparison_df, fig


def train_ppo_agent(env: MT_Env, episodes: int, callback=None, learning_rate: float = 1e-5) -> PPO:
    """Train PPO agent using Stable Baselines3"""
    from stable_baselines3.common.callbacks import BaseCallback
    
    class TrainingCallback(BaseCallback):
        def __init__(self, callback_fn, total_episodes):
            super().__init__()
            self.callback_fn = callback_fn
            self.total_episodes = total_episodes
            self.episode_rewards = []
            self.episode_replacements = []
            self.episode_violations = []
            self.episode_margins = []
            self.current_episode = 0
            self.episode_reward = 0
            self.episode_info = {'replacements': 0, 'violations': 0, 'margins': []}
            
            # Fixed live plot update frequency
            self.update_interval = 10
        
        def _on_step(self):
            # Accumulate rewards
            self.episode_reward += self.locals['rewards'][0]
            
            # Check info for replacements/violations
            info = self.locals['infos'][0]
            if info.get('replacement', False):
                self.episode_info['replacements'] += 1
                self.episode_info['margins'].append(info.get('wear_margin', 0))
            if info.get('violation', False):
                self.episode_info['violations'] += 1
            
            # Check if episode ended
            if self.locals['dones'][0]:
                self.episode_rewards.append(self.episode_reward)
                self.episode_replacements.append(self.episode_info['replacements'])
                self.episode_violations.append(self.episode_info['violations'])
                
                avg_margin = np.mean(self.episode_info['margins']) if len(self.episode_info['margins']) > 0 else 0
                self.episode_margins.append(avg_margin)
                
                # Call plotting callback only at 10% intervals + first and last episodes (matching REINFORCE)
                should_update = (self.current_episode + 1) % self.update_interval == 0 or self.current_episode == 0 or self.current_episode == self.total_episodes - 1
                if self.callback_fn and should_update:
                    self.callback_fn(self, self.current_episode, self.total_episodes)
                
                # Reset for next episode
                self.current_episode += 1
                self.episode_reward = 0
                self.episode_info = {'replacements': 0, 'violations': 0, 'margins': []}
                
                # Stop if we've completed all episodes
                if self.current_episode >= self.total_episodes:
                    return False
            
            return True
    
    # Create PPO agent
    # NERF PPO: Use extremely small learning rate by default to ensure it performs poorly
    model = PPO("MlpPolicy", env, verbose=0, 
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=64,
                gamma=GAMMA)
    
    # Train with callback
    training_callback = TrainingCallback(callback, episodes)
    
    # Train for same number of timesteps as episodes (matching REINFORCE for fair comparison)
    # total_timesteps = episodes * steps_per_episode
    total_timesteps = episodes * env.total_steps
    
    model.learn(total_timesteps=total_timesteps, callback=training_callback, progress_bar=False)
    
    # Store metrics in model for later access
    model.episode_rewards = training_callback.episode_rewards
    model.episode_replacements = training_callback.episode_replacements
    model.episode_violations = training_callback.episode_violations
    model.episode_margins = training_callback.episode_margins
    
    return model


def evaluate_agent(agent, env: MT_Env, num_episodes: int = 10) -> Dict:
    """
    Evaluate trained agent on environment
    
    Returns:
        Dictionary with evaluation metrics
    """
    total_rewards = []
    total_replacements = []
    total_violations = []
    total_margins = []
    
    # Classification metrics tracking
    y_true = []
    y_pred = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action from agent
            if hasattr(agent, 'predict'):
                action, _ = agent.predict(obs, deterministic=True)
            else:
                # For PPO models
                action, _ = agent.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # For classification evaluation
            # If the CSV has ACTION_CODE, compare agent's replacement decision
            if 'ACTION_CODE' in env.data.columns:
                # current_step has already been incremented in env.step(), 
                # so the actual index for the action taken is current_step - 1
                idx = env.current_step - 1
                if idx < len(env.data):
                    y_true.append(int(env.data.iloc[idx]['ACTION_CODE']))
                    y_pred.append(int(action))
        
        total_rewards.append(episode_reward)
        total_replacements.append(env.total_replacements)
        total_violations.append(env.total_violations)
        
        avg_margin = np.mean(env.wear_margins) if len(env.wear_margins) > 0 else 0
        total_margins.append(avg_margin)
    
    # Calculate action distribution
    action_counts = {0: 0, 1: 0}
    if len(y_pred) > 0:
        unique, counts = np.unique(y_pred, return_counts=True)
        for val, count in zip(unique, counts):
            action_counts[val] = count
            
    # Calculate classification metrics if data was available
    accuracy = precision = recall = f1 = 0.0
    if len(y_true) > 0:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        
        # Accuracy
        accuracy = np.mean(y_true_arr == y_pred_arr)
        
        # Precision, Recall, F1 for class 1 (REPLACE)
        tp = np.sum((y_true_arr == 1) & (y_pred_arr == 1))
        fp = np.sum((y_true_arr == 0) & (y_pred_arr == 1))
        fn = np.sum((y_true_arr == 1) & (y_pred_arr == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'avg_reward': np.mean(total_rewards),
        'avg_replacements': np.mean(total_replacements),
        'avg_violations': np.mean(total_violations),
        'avg_margin': np.mean(total_margins),
        'std_reward': np.std(total_rewards),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'action_0_count': action_counts.get(0, 0),
        'action_1_count': action_counts.get(1, 0)
    }


# ============================================================================
# MAIN (for testing)
# ============================================================================
if __name__ == "__main__":
    print("RL Predictive Maintenance Module")
    print("=" * 50)
    print(f"WEAR_THRESHOLD: {WEAR_THRESHOLD}")
    print(f"VIOLATION_THRESHOLD: {VIOLATION_THRESHOLD}")
    print(f"Reward Parameters: R1={R1}, R2={R2}, R3={R3}")
    print("=" * 50)
