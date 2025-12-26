# ------------------------------------------------------------------------------------
# Reinforcement Learning for Predictive Maintenance
# Author: Rajesh Siraskar
# V.1.0: 22-Dec-2025
# - Training REINFORCE, PPO, and REINFORCE+Attention agents
# - Plotting training metrics and results
#- To-do: Evaluating trained models on test data
# ------------------------------------------------------------------------------------

import warnings
import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.kernel_ridge import KernelRidge

# ==========================================
# SUPPRESS WARNINGS
# ==========================================
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)
np.seterr(all="ignore")

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    """Configuration parameters for RL training and evaluation."""
    
    # Data and Model Paths
    SENSOR_DATA: str = 'x'
    DATA_FILE: str = f'data/{SENSOR_DATA}.csv'
    
    # Reward Configuration
    R1_CONTINUE: float = 0.5                    # Base reward for continuing (lowered from 1.0 to remove survival bias)
    R2_REPLACE: float = 5.0                     # Penalty weight for unused life
    R3_VIOLATION: float = 100.0                  # Large penalty for crossing threshold (increased from 50)
    R4_REPLACE_BONUS: float = 10.0              # Bonus for replacing (new)
    AMR: float = 5e-2                           # Advantage Mixing Ratio for Attention
    VIOLATION_MARGIN: float = 0.05               # Violation margin - set to 5% over threshold
    
    # Training Configuration
    WEAR_THRESHOLD: float = 290.0               # Wear threshold
    EPISODES: int = 200                          # Training episodes
    LEARNING_RATE: float = 1e-3                 # Learning rate for optimizers
    GAMMA: float = 0.99                         # Discount factor
    SMOOTH_WINDOW: int = 50                     # Window size for smoothing plots
    PLOT_SMOOTHING_FACTOR: int = int(EPISODES/10)             # Smoothing window for sensor data visualization
    TRAINING_PLOT_MA_WINDOW: int = int(EPISODES/10)           # Moving average window for training plot trends
    
    # Model Saving
    SAVE_MODEL: bool = True
    
    @classmethod
    def get_model_path(cls, agent_type: str, episodes: int = None, data_file: str = None) -> str:
        """Get path for a model file based on agent type, episode count, and data file name."""
        eps = episodes if episodes is not None else cls.EPISODES
        
        # Use provided data_file name or fallback to SENSOR_DATA
        if data_file:
            data_name = os.path.splitext(os.path.basename(data_file))[0]
        else:
            data_name = cls.SENSOR_DATA
            
        ext = 'zip' if agent_type.upper() == 'PPO' else 'h5'
        # Add date suffix (DDMM)
        date_str = pd.Timestamp.now().strftime('%d%m')
        return f'models/{data_name}_{agent_type}_Model_{eps}_{date_str}.{ext}'

    @classmethod
    def get_latest_model(cls, agent_type: str, data_file: str = None) -> str:
        """Find the latest trained model file for a specific agent type and optional data file in models/ folder."""
        models_dir = 'models'
        if not os.path.exists(models_dir):
            return ""
            
        files = [f for f in os.listdir(models_dir) if f.endswith(('.h5', '.zip'))]
        
        # If data_file is provided, filter by its base name
        filter_name = ""
        if data_file:
            filter_name = os.path.splitext(os.path.basename(data_file))[0]
        
        # Determine specific pattern based on agent type
        if agent_type.upper() == 'REINFORCE':
            pattern_files = [f for f in files if '_REINFORCE_' in f and '_AM_' not in f]
        elif agent_type.upper() in ['REINFORCE_AM', 'REINFORCE AM', 'AM']:
            pattern_files = [f for f in files if '_REINFORCE_AM_' in f]
        elif agent_type.upper() == 'PPO':
            pattern_files = [f for f in files if '_PPO_' in f]
        else:
            pattern_files = []
            
        # Further filter by data file name if provided
        if filter_name:
            pattern_files = [f for f in pattern_files if f.startswith(filter_name)]
            
        if not pattern_files:
            return ""
            
        # Sort by modification time
        full_paths = [os.path.join(models_dir, f) for f in pattern_files]
        latest_file = max(full_paths, key=os.path.getmtime)
        return latest_file

    # Keep these as placeholders or remove if not used elsewhere
    # (Note: app.py uses get_model_path instead of these static strings usually)
    REINFORCE_MODEL: str = 'models/PROC_9_REINFORCE_Model_200.h5'
    REINFORCE_AM_MODEL: str = 'models/PROC_9_REINFORCE_AM_Model_200.h5'
    PPO_MODEL: str = 'models/PROC_9_PPO_Model_200.zip'

# ==========================================
# ENVIRONMENTS
# ==========================================
class MT_Env(gym.Env):
    """
    Milling Tool Environment for RL training.
    Actions: CONTINUE (0), REPLACE_TOOL (1)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        data_file: str = Config.DATA_FILE,
        wear_threshold: float = Config.WEAR_THRESHOLD,
        r1: float = Config.R1_CONTINUE,
        r2: float = Config.R2_REPLACE,
        r3: float = Config.R3_VIOLATION,
        violation_margin: float = Config.VIOLATION_MARGIN
    ):
        super().__init__()
        self.data_file = data_file
        self.df = pd.read_csv(self.data_file)
        
        # Validate required columns
        required_cols = [
            "Time", "Vib_Spindle", "Vib_Table", "Sound_Spindle", "Sound_table",
            "X_Load_Cell", "Y_Load_Cell", "Z_Load_Cell", "Current", "tool_wear"
        ]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing required column in data file: {c}")
        
        self.tool_wear_series = pd.to_numeric(self.df["tool_wear"], errors="coerce").to_numpy()
        # Calculate Useful Life (UL)
        # UL = 0 at wear threshold, maximizing at start
        # Find the first index where wear exceeds threshold
        wear_exceeded = self.df[self.df["tool_wear"] >= wear_threshold]
        if not wear_exceeded.empty:
            threshold_idx = wear_exceeded.index[0]
        else:
            threshold_idx = len(self.df) - 1 # Fallback if never exceeds
            
        # Create UL column: counts down from threshold_idx
        self.df['UL'] = threshold_idx - self.df.index
        # Note: UL will be negative after threshold
        
        # Store Max UL (at index 0) for metric calculation
        self.max_ul = float(self.df.iloc[0]['UL'])
        
        # Feature compatibility with Attention Environment
        self.features = [
            'Vib_Spindle', 'Vib_Table', 'Sound_Spindle', 'Sound_table',
            'X_Load_Cell', 'Y_Load_Cell', 'Z_Load_Cell', 'Current'
        ]
        
        missing = [f for f in self.features if f not in self.df.columns]
        if missing:
            raise ValueError(f"Missing features in dataset: {missing}")
        
        # Compute per-feature normalization stats
        vals_df = self.df[self.features].astype(np.float32)
        self.feature_means = vals_df.mean(axis=0).to_numpy(dtype=np.float32)
        self.feature_stds = vals_df.std(axis=0).replace(0, 1.0).to_numpy(dtype=np.float32)
        self.attention_weights = np.ones(len(self.features), dtype=np.float32)
        
        # Configuration
        self.wear_threshold = float(wear_threshold)
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.r3 = float(r3)
        self.violation_margin = float(violation_margin)
        
        # Gym spaces
        obs_low = np.full((8,), -np.finfo(np.float32).max, dtype=np.float32)
        obs_high = np.full((8,), np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        # State tracking
        self.current_idx = 0
        self.done = False
        self._last_terminal_margin = np.nan
    
    def get_observation(self) -> np.ndarray:
        """Get normalized observation (sensor features) at current_idx."""
        idx = min(self.current_idx, len(self.df) - 1)
        row = self.df.iloc[idx]
        sensors = np.array([
            row["Vib_Spindle"], row["Vib_Table"], row["Sound_Spindle"], row["Sound_table"],
            row["X_Load_Cell"], row["Y_Load_Cell"], row["Z_Load_Cell"], row["Current"]
        ], dtype=np.float32)
        
        # Normalize sensors using precomputed mean and std
        norm_sensors = (sensors - self.feature_means) / (self.feature_stds + 1e-9)
        return norm_sensors.astype(np.float32)
    
    def _get_tool_wear(self, idx: int) -> float:
        """Get tool wear value at index."""
        return float(self.tool_wear_series[idx])
    
    def _compute_margin(self, idx: int) -> float:
        """Compute wear margin (threshold - wear)."""
        wear = self._get_tool_wear(idx)
        return float(self.wear_threshold - wear)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step of the environment."""
        if self.done:
            obs = self.get_observation()
            return obs, float(reward), True, False, info
        
        if action not in (0, 1):
            raise ValueError("Invalid action.")
        
        tool_wear = self._get_tool_wear(self.current_idx)
        reward = 0.0
        info = {'violation': False, 'replacement': False, 'margin': np.nan, 'UL': np.nan}
        
        # Define violation threshold: threshold + 5%
        violation_threshold = self.wear_threshold * (1 + self.violation_margin)
        
        if action == 0:  # CONTINUE
            if tool_wear > violation_threshold:
                reward = -self.r3
                self.done = True
                info['violation'] = True
                info['margin'] = self._compute_margin(self.current_idx)
                # Calculate Consumed UL (Productive Time): Max - Current Remaining
                current_ul_val = float(self.df.iloc[self.current_idx]['UL'])
                info['UL'] = self.max_ul - current_ul_val
                self._last_terminal_margin = info['margin']
                obs = self.get_observation()
                return obs, float(reward), True, False, info
            else:
                reward = self.r1
                if self.current_idx + 1 < len(self.df):
                    self.current_idx += 1
                    obs = self.get_observation()
                    return obs, float(reward), False, False, info
                else:
                    self.done = True
                    info['margin'] = self._compute_margin(self.current_idx)
                    # Calculate Consumed UL (Productive Time)
                    current_ul_val = float(self.df.iloc[self.current_idx]['UL'])
                    info['UL'] = self.max_ul - current_ul_val
                    self._last_terminal_margin = info['margin']
                    obs = self.get_observation()
                    return obs, float(reward), True, False, info
        
        else:  # REPLACE_TOOL (action == 1)
            info['replacement'] = True
            info['margin'] = self._compute_margin(self.current_idx)
            
            # Check if replacement is late (violation)
            if tool_wear > violation_threshold:
                reward = -self.r3  # Severe penalty for late replacement
                info['violation'] = True
            elif tool_wear > self.wear_threshold:
                # Late but within 5% buffer
                reward = -self.r3 * 0.2  # Moderate penalty
                info['violation'] = False
            else:
                # TIMELY replacement or EARLY replacement
                used_fraction = np.clip(tool_wear / self.wear_threshold, 0.0, 1.0)
                unused_fraction = 1.0 - used_fraction
                # Improved replacement reward: provide a positive bonus weighted by wear
                reward = Config.R4_REPLACE_BONUS * used_fraction - self.r2 * unused_fraction
                info['violation'] = False
                
            self.done = True
            self._last_terminal_margin = info['margin']
            # Calculate Consumed UL (Productive Time)
            current_ul_val = float(self.df.iloc[self.current_idx]['UL'])
            info['UL'] = self.max_ul - current_ul_val
            obs = self.get_observation()
            return obs, float(reward), True, False, info
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_idx = 0
        self.done = False
        self._last_terminal_margin = np.nan
        obs = self.get_observation()
        info = {'violation': False, 'replacement': False, 'margin': np.nan, 'UL': np.nan}
        return obs, info
    
    def render(self, mode="human"):
        """Render environment state."""
        if self.done:
            print(f"[MT-Env] done. idx={self.current_idx}, margin={self._last_terminal_margin}")
        else:
            wear_val = self._get_tool_wear(self.current_idx)
            print(f"[MT-Env] idx={self.current_idx}, tool_wear={wear_val:.3f}")
    
    def close(self):
        """Close environment."""
        pass


class AM_Env(MT_Env):
    """
    Attention-augmented variant of MT_Env.
    Uses kernel ridge regression to compute attention weights based on feature-to-wear relationships.
    """
    
    def __init__(
        self,
        data_file: str = Config.DATA_FILE,
        wear_threshold: float = Config.WEAR_THRESHOLD,
        r1: float = Config.R1_CONTINUE,
        r2: float = Config.R2_REPLACE,
        r3: float = Config.R3_VIOLATION,
        violation_margin: float = Config.VIOLATION_MARGIN,
        kr_alpha: float = 1.0
    ):
        super().__init__(data_file, wear_threshold, r1, r2, r3, violation_margin)
        
        # Prepare training data for KernelRidge
        vals_df = self.df[self.features].astype(np.float32)
        X = ((vals_df - self.feature_means) / (self.feature_stds + 1e-9)).to_numpy(dtype=np.float32)
        y = self.df['tool_wear'].to_numpy(dtype=np.float32)
        
        # Fit KernelRidge to estimate feature importances
        self._kr_model = KernelRidge(kernel='linear', alpha=kr_alpha)
        self._kr_model.fit(X, y)
        
        # Compute attention weights from primal coefficients
        dual = np.asarray(self._kr_model.dual_coef_).reshape(-1, 1)
        X_fit = np.asarray(self._kr_model.X_fit_)
        coef = (X_fit.T @ dual).ravel()
        
        attn = np.abs(coef)
        if attn.sum() == 0:
            attn = np.ones_like(attn)
        attn = attn / (attn.sum() + 1e-12)
        self.attention_weights = attn.astype(np.float32)
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32
        )
    
    def get_observation(self) -> np.ndarray:
        """Return attention-weighted, normalized feature vector."""
        idx = min(self.current_idx, len(self.df) - 1)
        vals = self.df.loc[idx, self.features].to_numpy(dtype=np.float32)
        norm = (vals - self.feature_means) / (self.feature_stds + 1e-9)
        weighted = norm * self.attention_weights
        return weighted.astype(np.float32)
    
    def recompute_attention(self, window: int = None, kr_alpha: float = None):
        """Recompute attention weights optionally using a sliding window."""
        if window is None:
            df_slice = self.df[self.features]
        else:
            start = max(0, self.current_idx - window + 1)
            df_slice = self.df.loc[start:self.current_idx, self.features]
            if len(df_slice) < 2:
                df_slice = self.df[self.features]
        
        X = ((df_slice - self.feature_means) / (self.feature_stds + 1e-9)).to_numpy(dtype=np.float32)
        y = self.df.loc[df_slice.index, 'tool_wear'].to_numpy(dtype=np.float32)
        
        model = KernelRidge(kernel='linear', alpha=(kr_alpha if kr_alpha is not None else self._kr_model.alpha))
        model.fit(X, y)
        
        dual = np.asarray(model.dual_coef_).reshape(-1, 1)
        X_fit = np.asarray(model.X_fit_)
        coef = (X_fit.T @ dual).ravel()
        
        attn = np.abs(coef)
        if attn.sum() == 0:
            attn = np.ones_like(attn)
        attn = attn / (attn.sum() + 1e-12)
        self.attention_weights = attn.astype(np.float32)
    
    def close(self):
        """Close environment."""
        pass

# ==========================================
# NEURAL NETWORK POLICY
# ==========================================
class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE agents."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)

# ==========================================
# REINFORCE AGENT
# ==========================================
class REINFORCE:
    """Custom REINFORCE agent implementation."""
    
    def __init__(
        self,
        policy: PolicyNetwork,
        env: gym.Env,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        model_file: str = None,
        agent_name: str = "REINFORCE",
        data_file_name: str = "Unknown"
    ):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.model_name = agent_name
        self.agent_name = agent_name
        self.model_file = model_file
        self.data_file_name = data_file_name
    
    def predict(self, obs: np.ndarray) -> int:
        """Select action based on observation."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            probs = self.policy(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            return action.item()
    
    def learn(self, total_episodes: int) -> Dict[str, List]:
        """Train the agent for given number of episodes."""
        print(f"--- Training {self.model_name} ---")
        
        all_rewards = []
        all_violations = []
        all_replacements = []
        all_margins = []
        all_uls = []
        
        # Try to import Streamlit for live visualization
        try:
            import streamlit as st
            is_streamlit = True
            # Create placeholder for live plots
            plot_placeholder = st.empty()
            progress_text = "Training the REINFORCE model ..."
            training_progress_bar = st.progress(0, text=progress_text)
        except (ImportError, RuntimeError):
            is_streamlit = False
            progress_text = None
            training_progress_bar = None
        
        for episode in range(total_episodes):
            log_probs = []
            rewards = []
            obs, _ = self.env.reset()
            done = False
            
            episode_info = {'violation': 0, 'replacement': 0, 'margin': np.nan, 'UL': np.nan}
            
            while not done:
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                probs = self.policy(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)
                
                obs, reward, terminated, truncated, info = self.env.step(action.item())
                rewards.append(reward)
                done = terminated or truncated
                
                if info.get('violation'):
                    episode_info['violation'] = 1
                if info.get('replacement'):
                    episode_info['replacement'] = 1
                if not np.isnan(info.get('margin')):
                    episode_info['margin'] = info.get('margin')
                if not np.isnan(info.get('UL')):
                    episode_info['UL'] = info.get('UL')
            
            # Collect metrics
            all_rewards.append(sum(rewards))
            all_violations.append(episode_info['violation'])
            all_replacements.append(episode_info['replacement'])
            all_margins.append(episode_info['margin'])
            all_uls.append(episode_info['UL'])
            
            # Calculate discounted returns
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # Update policy
            policy_loss = []
            for log_prob, G in zip(log_probs, returns):
                policy_loss.append(-log_prob * G)
            
            self.optimizer.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            self.optimizer.step()
            
            # Update progress and plots every 5 episodes or at the end
            if (episode + 1) % 5 == 0 or (episode + 1) == total_episodes:
                if is_streamlit:
                    # Update progress bar
                    progress_pct = (episode + 1) / total_episodes
                    progress_text = f"Episode {episode + 1}/{total_episodes}, Reward: {sum(rewards):.2f}"
                    training_progress_bar.progress(progress_pct, text=progress_text)
                    
                    # Update live plots
                    current_metrics = {
                        "rewards": all_rewards,
                        "violations": all_violations,
                        "replacements": all_replacements,
                        "margins": all_margins,
                        "uls": all_uls
                    }
                    fig = plot_training_live(
                        current_metrics,
                        episode=episode + 1,
                        total_episodes=total_episodes,
                        agent_name=self.agent_name,
                        data_file_name=self.data_file_name,
                        window=5
                    )
                    with plot_placeholder.container():
                        st.pyplot(fig, use_container_width=True)
                    
                    # Save the final plot if it's the last episode
                    if (episode + 1) == total_episodes and self.model_file:
                        try:
                            plot_dir = 'saved_plots'
                            os.makedirs(plot_dir, exist_ok=True)
                            # model_file is like 'models/Data_REINFORCE_Model_800_DDMM.h5'
                            # We want 'saved_plots/Data_REINFORCE_plot_800_DDMM.png'
                            base_name = os.path.basename(self.model_file)
                            plot_name = base_name.replace('Model', 'plot').replace('.h5', '.png')
                            plot_path = os.path.join(plot_dir, plot_name)
                            fig.savefig(plot_path)
                            print(f"Training plot saved to: {plot_path}")
                        except Exception as e:
                            print(f"Error saving training plot: {str(e)}")
                    
                    plt.close(fig)  # Free memory
                else:
                    if (episode + 1) % 50 == 0:
                        print(f"Episode {episode + 1}/{total_episodes}, Reward: {sum(rewards):.2f}")
        
        print("--- Training Complete ---")
        
        # Clear the progress bar after completion
        if is_streamlit:
            training_progress_bar.empty()
            st.success("🎉 Training complete!")
        
        # Save model if provided
        if self.model_file is not None:
            try:
                os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
                torch.save({
                    'policy_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'input_dim': self.policy.net[0].in_features,
                    'output_dim': self.policy.net[-2].out_features
                }, self.model_file)
                print(f"Model saved to: {self.model_file}")
                if is_streamlit:
                    st.info(f"📁 Model saved to: {self.model_file}")
            except Exception as e:
                print(f"Error saving model: {str(e)}")
                if is_streamlit:
                    st.error(f"Error saving model: {str(e)}")
        
        return {
            "rewards": all_rewards,
            "violations": all_violations,
            "replacements": all_replacements,
            "margins": all_margins,
            "uls": all_uls
        }

# ==========================================
# PPO TRAINING
# ==========================================
class MetricsCallback(BaseCallback):
    """Callback for collecting metrics during PPO training."""
    
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.rewards = []
        self.violations = []
        self.replacements = []
        self.margins = []
        self.uls = []
        self.episode_reward = 0.0
    
    def _on_step(self) -> bool:
        # Accumulate reward from current step
        self.episode_reward += self.locals.get('rewards', [0])[0]
        
        if self.locals['dones'][0]:
            info = self.locals['infos'][0]
            self.rewards.append(self.episode_reward)
            self.violations.append(1 if info.get('violation') else 0)
            self.replacements.append(1 if info.get('replacement') else 0)
            self.margins.append(info.get('margin', np.nan))
            self.uls.append(info.get('UL', np.nan))
            self.episode_reward = 0.0
        return True


def train_ppo(
    env: gym.Env,
    total_episodes: int,
    learning_rate: float = Config.LEARNING_RATE,
    gamma: float = Config.GAMMA,
    model_file: str = None,
    data_file_name: str = "Unknown"
) -> Dict[str, List]:
    """Train a PPO agent and collect metrics with live visualization."""
    print("--- Training PPO ---")
    callback = MetricsCallback()
    
    # Try to import Streamlit for live visualization
    try:
        import streamlit as st
        is_streamlit = True
        plot_placeholder = st.empty()
        progress_text = "Training the PPO model ..."
        training_progress_bar = st.progress(0, text=progress_text)
    except (ImportError, RuntimeError):
        is_streamlit = False
        progress_text = None
        training_progress_bar = None
    
    # Create model with PPO defaults
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=learning_rate, gamma=gamma, n_steps=256)
    
    obs, _ = env.reset()
    ep_count = 0
    total_timesteps = 0
    timesteps_per_update = 256  # PPO's n_steps
    
    while ep_count < total_episodes:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_timesteps += 1
        
        # Track metrics
        callback.episode_reward += reward
        if terminated or truncated:
            callback.rewards.append(callback.episode_reward)
            callback.violations.append(1 if info.get('violation') else 0)
            callback.replacements.append(1 if info.get('replacement') else 0)
            callback.margins.append(info.get('margin', np.nan))
            callback.uls.append(info.get('UL', np.nan))
            callback.episode_reward = 0.0
            ep_count += 1
            obs, _ = env.reset()
        
        # Train PPO when we've accumulated enough timesteps
        if total_timesteps % timesteps_per_update == 0 or ep_count >= total_episodes:
            model.learn(total_timesteps=timesteps_per_update)
            
            # Update progress and plots every 5 episodes or at the end
            if (ep_count) % 5 == 0 or (ep_count) >= total_episodes:
                if is_streamlit:
                    # Update progress bar
                    progress_pct = min(ep_count / total_episodes, 1.0)
                    current_reward = callback.rewards[-1] if callback.rewards else 0.0
                    progress_text = f"Episode {ep_count}/{total_episodes}, Reward: {current_reward:.2f}"
                    training_progress_bar.progress(progress_pct, text=progress_text)
                    
                    # Update live plots
                    current_metrics = {
                        "rewards": callback.rewards,
                        "violations": callback.violations,
                        "replacements": callback.replacements,
                        "margins": callback.margins,
                        "uls": callback.uls
                    }
                    fig = plot_training_live(
                        current_metrics,
                        episode=ep_count,
                        total_episodes=total_episodes,
                        agent_name="PPO",
                        data_file_name=data_file_name,
                        window=5
                    )
                    with plot_placeholder.container():
                        st.pyplot(fig, use_container_width=True)
                    
                    # Save the final plot if training is complete
                    if ep_count >= total_episodes and model_file:
                        try:
                            plot_dir = 'saved_plots'
                            os.makedirs(plot_dir, exist_ok=True)
                            # model_file is like 'models/Data_PPO_Model_800_DDMM.zip'
                            # We want 'saved_plots/Data_PPO_plot_800_DDMM.png'
                            base_name = os.path.basename(model_file)
                            plot_name = base_name.replace('Model', 'plot').replace('.zip', '.png')
                            plot_path = os.path.join(plot_dir, plot_name)
                            fig.savefig(plot_path)
                            print(f"Training plot saved to: {plot_path}")
                        except Exception as e:
                            print(f"Error saving training plot: {str(e)}")
                            
                    plt.close(fig)  # Free memory
                else:
                    if (ep_count) % 5 == 0 or ep_count >= total_episodes:
                        current_reward = callback.rewards[-1] if callback.rewards else 0.0
                        print(f"Episode {ep_count}/{total_episodes}, Reward: {current_reward:.2f}")
    
    print("--- Training Complete ---")
    
    # Clear the progress bar after completion
    if is_streamlit:
        training_progress_bar.empty()
        st.success("🎉 PPO Training complete!")
    
    if model_file is not None:
        try:
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            model.save(model_file)
            print(f"PPO model saved to: {model_file}")
            if is_streamlit:
                st.info(f"📁 PPO model saved to: {model_file}")
        except Exception as e:
            print(f"Error saving PPO model: {str(e)}")
            if is_streamlit:
                st.error(f"Error saving PPO model: {str(e)}")
    
    return {
        "rewards": callback.rewards,
        "violations": callback.violations,
        "replacements": callback.replacements,
        "margins": callback.margins,
        "uls": callback.uls
    }

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def smooth(data: List[float], window_size: int) -> np.ndarray:
    """Apply moving average smoothing."""
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().to_numpy()


def average_metrics(
    metrics_a: Dict[str, List],
    metrics_b: Dict[str, List],
    name_a: str = None,
    name_b: str = None,
    leave_first_N: int = 0
) -> Dict[str, Dict[str, float]]:
    """Compute averaged metrics for two algorithms."""
    
    N = max(0, int(leave_first_N))
    
    if name_a is None:
        import inspect
        calling_frame = inspect.currentframe().f_back
        name_a = next((n for n, v in calling_frame.f_locals.items() if v is metrics_a), "Algorithm_A")
    if name_b is None:
        import inspect
        calling_frame = inspect.currentframe().f_back
        name_b = next((n for n, v in calling_frame.f_locals.items() if v is metrics_b), "Algorithm_B")
    
    def safe_avg(seq):
        try:
            arr = np.asarray(seq, dtype=float)
        except Exception:
            return float("nan")
        sliced = arr[N:]
        if sliced.size == 0:
            return float("nan")
        return float(np.nanmean(sliced))
    
    metrics = {
        name_a: {
            'avg_reward': safe_avg(metrics_a.get('rewards', [])),
            'avg_violations': safe_avg(metrics_a.get('violations', [])),
            'avg_replacements': safe_avg(metrics_a.get('replacements', [])),
            'avg_margin': safe_avg(metrics_a.get('margins', []))
        },
        name_b: {
            'avg_reward': safe_avg(metrics_b.get('rewards', [])),
            'avg_violations': safe_avg(metrics_b.get('violations', [])),
            'avg_replacements': safe_avg(metrics_b.get('replacements', [])),
            'avg_margin': safe_avg(metrics_b.get('margins', []))
        }
    }
    
    # Print formatted table
    print(f'\nAverage Metrics Comparison {"(Steady state)" if N > 0 else ""}')
    print("-" * 64)
    print(f"{'Metric':<20} {name_a:>18} {name_b:>18}")
    print("-" * 64)
    
    metrics_names = {
        'avg_reward': 'Average Reward',
        'avg_violations': 'Violation Rate',
        'avg_replacements': 'Replacement Rate',
        'avg_margin': 'Average Margin'
    }
    
    for metric in metrics_names:
        a_val = metrics[name_a][metric]
        b_val = metrics[name_b][metric]
        a_str = f"{a_val:>18.3f}" if not np.isnan(a_val) else f"{'nan':>18}"
        b_str = f"{b_val:>18.3f}" if not np.isnan(b_val) else f"{'nan':>18}"
        print(f"{metrics_names[metric]:<20} {a_str} {b_str}")
    
    print("-" * 64)
    
    return metrics


def downsample_merged_file(data_file: str, records_per_procedure: int = 30, save_path: str = None) -> pd.DataFrame:
    """
    Downsample a merged CSV file preserving order, keeping 'records_per_procedure' for each 'PROC'.
    
    Args:
        data_file (str): Path to the input CSV file.
        records_per_procedure (int): Number of records to keep for each procedure (default 30).
        save_path (str, optional): Path to save the downsampled CSV.
        
    Returns:
        pd.DataFrame: Downsampled DataFrame.
    """
    try:
        df = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: File not found: {data_file}")
        return pd.DataFrame()
        
    if 'PROC' not in df.columns:
        print("Error: Input file must contain 'PROC' column.")
        return df
    
    # Get unique procedures while preserving order of appearance
    procedures = df['PROC'].unique()
    
    downsampled_dfs = []
    print(f"Downsampling {data_file}...")
    
    for proc in procedures:
        proc_df = df[df['PROC'] == proc]
        # Keep first N records
        subset = proc_df.iloc[:records_per_procedure]
        downsampled_dfs.append(subset)
        print(f"  Processed {proc}: kept {len(subset)}/{len(proc_df)} rows")
    
    if not downsampled_dfs:
        return pd.DataFrame()
        
    result_df = pd.concat(downsampled_dfs)
    
    print(f"Total rows: {len(result_df)}")
    
    if save_path:
        try:
            result_df.to_csv(save_path, index=False)
            print(f"Saved downsampled data to: {save_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
            
    return result_df

# ==========================================
# PLOTTING FUNCTIONS
# ==========================================
def plot_metrics(
    metrics_a: Dict[str, List],
    metrics_b: Dict[str, List],
    name_a: str = "Algorithm A",
    name_b: str = "Algorithm B",
    window: int = Config.SMOOTH_WINDOW,
    mode: str = "COMBINED"
) -> None:
    """Plot training metrics comparison."""
    
    W, H = 16, 6
    FONTSIZE_SUPER = 18
    FONTSIZE_TITLE = 12
    FONTSIZE_LABEL = 10
    FONTSIZE_TICK = 9
    BACKGROUND_COLOR = '#f8f8f8'
    
    def setup_subplot(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=FONTSIZE_TITLE)
        ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_facecolor(BACKGROUND_COLOR)
    
    if mode.upper() == "COMBINED":
        fig, axs = plt.subplots(2, 2, figsize=(W, H))
        fig.suptitle(f'{name_a} vs. {name_b} Performance Comparison', fontsize=FONTSIZE_SUPER)
        
        metrics_config = [
            {'data': 'rewards', 'title': 'Learning Curve (Smoothed Average Reward)', 'ylabel': 'Average Reward'},
            {'data': 'replacements', 'title': 'Replacements per episode (Smoothed)', 'ylabel': 'Replacement Rate'},
            {'data': 'violations', 'title': 'Threshold Violations per episode (Smoothed)', 'ylabel': 'Violation Rate'},
            {'data': 'margins', 'title': 'Wear Margin Before Replacement (Smoothed)', 'ylabel': 'Wear Margin'}
        ]
        
        for idx, config in enumerate(metrics_config):
            ax = axs[idx // 2, idx % 2]
            if config['data'] == 'margins':
                a_data = pd.Series(metrics_a[config['data']]).rolling(window, min_periods=1).mean()
                b_data = pd.Series(metrics_b[config['data']]).rolling(window, min_periods=1).mean()
            else:
                a_data = smooth(metrics_a[config['data']], window)
                b_data = smooth(metrics_b[config['data']], window)
            
            ax.plot(a_data, label=name_a, alpha=0.6)
            ax.plot(b_data, label=name_b, alpha=0.6)
            setup_subplot(ax, config['title'], 'Episode', config['ylabel'])
            ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_single_metrics(
    metrics: Dict[str, List],
    name: str = "Algorithm",
    window: int = Config.SMOOTH_WINDOW
) -> None:
    """Plot training metrics for a single algorithm."""
    
    W, H = 16, 6
    FONTSIZE_SUPER = 18
    FONTSIZE_TITLE = 12
    FONTSIZE_LABEL = 10
    FONTSIZE_TICK = 9
    BACKGROUND_COLOR = '#f8f8f8'
    
    def setup_subplot(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=FONTSIZE_TITLE)
        ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_facecolor(BACKGROUND_COLOR)
    
    fig, axs = plt.subplots(2, 2, figsize=(W, H))
    fig.suptitle(f'{name} Training Performance', fontsize=FONTSIZE_SUPER)
    
    metrics_config = [
        {'data': 'rewards', 'title': 'Learning Curve (Smoothed Average Reward)', 'ylabel': 'Average Reward', 'color': 'blue'},
        {'data': 'replacements', 'title': 'Replacements per episode (Smoothed)', 'ylabel': 'Replacement Rate', 'color': 'green'},
        {'data': 'violations', 'title': 'Threshold Violations per episode (Smoothed)', 'ylabel': 'Violation Rate', 'color': 'red'},
        {'data': 'margins', 'title': 'Wear Margin Before Replacement (Smoothed)', 'ylabel': 'Wear Margin', 'color': 'purple'}
    ]
    
    for idx, config in enumerate(metrics_config):
        ax = axs[idx // 2, idx % 2]
        if config['data'] == 'margins':
            data = pd.Series(metrics[config['data']]).rolling(window, min_periods=1).mean()
        else:
            data = smooth(metrics[config['data']], window)
        
        ax.plot(data, color=config['color'], alpha=0.6)
        setup_subplot(ax, config['title'], 'Episode', config['ylabel'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_training_live(
    metrics: Dict[str, List],
    episode: int = 0,
    total_episodes: int = 50,
    agent_name: str = "Agent",
    data_file_name: str = "Unknown",
    window: int = 10
):
    """
    Generic live plotting function for training visualization in Streamlit.
    Dynamically plots metrics as training progresses for any agent type.
    
    Args:
        metrics (Dict): Current metrics dictionary with keys: rewards, violations, replacements, margins
        episode (int): Current episode number
        total_episodes (int): Total episodes to train
        agent_name (str): Name of the agent being trained (e.g., 'REINFORCE', 'PPO', 'REINFORCE_AM')
        window (int): Smoothing window size
    
    Returns:
        matplotlib.figure.Figure: Figure object for Streamlit display
    """
    
    W, H = 14, 6 # 14, 8
    FONTSIZE_SUPER = 12 #16
    FONTSIZE_TITLE = 10 #11
    FONTSIZE_LABEL = 9 # 9
    FONTSIZE_TICK = 8  # 8
    BACKGROUND_COLOR = '#fafafa'
    LINE_WIDTH = 1.2
    
    def setup_subplot(ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=FONTSIZE_TITLE) # , fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL)
        ax.tick_params(labelsize=FONTSIZE_TICK)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.7)
        ax.set_facecolor(BACKGROUND_COLOR)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(W, H))
    fig.patch.set_facecolor('white')
    
    # Display names mapping for friendly titles
    agent_display_names = {
        'REINFORCE': 'REINFORCE',
        'PPO': 'PPO',
        'REINFORCE_AM': 'REINFORCE with Attention'
    }
    display_agent_name = agent_display_names.get(agent_name, agent_name)
    
    # Title with progress info
    progress_pct = (episode / max(total_episodes, 1)) * 100
    
    # 1. Left align the full title & 2. Make the main model bold and the rest small/plain
    fig.suptitle(
        f'{display_agent_name} Agent Training (Data: {data_file_name})',
        x=0.035, y=0.98,
        ha='left',
        fontsize=FONTSIZE_SUPER,
        fontweight='bold',
        color='#2C3E50'
    )
    
    fig.text(
        0.98, 0.96,
        f"Training progress - episode {episode}/{total_episodes} ({progress_pct:.1f}%)",
        ha='right',
        fontsize=FONTSIZE_TITLE,
        fontweight='normal',
        color='#2C3E50'
    )
    
    # Metrics configuration
    metrics_config = [
        {
            'data': 'rewards',
            'ax': axs[0, 0],
            'title': 'Cumulative Reward per Episode',
            'ylabel': 'Reward',
            'color': '#1f77b4',
            'smooth': True,
            'trend': True
        },
        {
            'data': 'violations',
            'ax': axs[0, 1],
            'title': 'Violations per Episode',
            'ylabel': 'Violation Count',
            'color': '#d62728',
            'smooth': False,
            'trend': False
        },
        {
            'data': 'uls', # Changed from 'replacements'
            'ax': axs[1, 0],
            'title': 'Productive Useful Life (UL) at Replacement', # Changed title
            'ylabel': 'Productive Life (steps)', # Changed label
            'color': '#2ca02c',
            'smooth': True, # Changed to True for better visualization
            'trend': True
        },
        # {
        #     'data': 'replacements',
        #     'ax': axs[1, 0],
        #     'title': 'Replacements per Episode',
        #     'ylabel': 'Replacement Count',
        #     'color': '#2ca02c',
        #     'smooth': False,
        #     'trend': False
        # },
        {
            'data': 'margins',
            'ax': axs[1, 1],
            'title': 'Wear Margin at Replacement',
            'ylabel': 'Margin Value',
            'color': '#9467bd',
            'smooth': True,
            'trend': True
        }
    ]
    
    # Plot each metric
    for config in metrics_config:
        ax = config['ax']
        data_key = config['data']
        data = metrics.get(data_key, [])
        
        if len(data) > 0:
            # Apply smoothing if requested and enough data points
            if config['smooth'] and len(data) > 1:
                smooth_data = smooth(data, min(window, len(data)))
            else:
                smooth_data = np.array(data)
            
            # Plot data
            episodes_range = np.arange(len(smooth_data))
            ax.plot(
                episodes_range,
                smooth_data,
                color=config['color'],
                linewidth=LINE_WIDTH,
                alpha=0.8,
                # marker='o',
                # markersize=4,
                # markerfacecolor=config['color'],
                # markeredgecolor='white',
                # markeredgewidth=0.5
            )
            
            # Add trend line (moving average) if enough data
            trend_window = max(int(total_episodes * 0.1), 5)
            if config['trend'] and len(data) > trend_window:
                ma_data = pd.Series(smooth_data).rolling(window=trend_window, min_periods=1).mean().to_numpy()
                ax.plot(
                    episodes_range,
                    ma_data,
                    color=config['color'],
                    linestyle='--',
                    alpha=0.3,
                    linewidth=2,
                    label=f'Trend (MA-{trend_window})'
                )
            
            # Set x-axis limit with some padding
            ax.set_xlim(-1, max(len(data), total_episodes) + 1)
        
        # Setup subplot styling
        setup_subplot(ax, config['title'], 'Episode', config['ylabel'])
        ax.set_axisbelow(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


# Keep old function name for backward compatibility
def plot_reinforce_training_live(
    metrics: Dict[str, List],
    episode: int = 0,
    total_episodes: int = 50,
    window: int = 10
):
    """
    Deprecated: Use plot_training_live() instead.
    Live plotting function for REINFORCE training visualization in Streamlit.
    """
    return plot_training_live(
        metrics=metrics,
        episode=episode,
        total_episodes=total_episodes,
        agent_name="REINFORCE",
        window=window
    )



# ==========================================
# SENSOR DATA VISUALIZATION
# ==========================================
def plot_sensor_data_with_wear(df, data_file_name, smoothing=None):
    """
    Creates a 3x3 multi-plot of sensor data with tool wear visualization.
    
    Args:
        df (pd.DataFrame): DataFrame containing sensor data with columns:
                          Time, Vib_Spindle, Vib_Table, Current, X_Load_Cell,
                          Y_Load_Cell, Z_Load_Cell, Sound_Spindle, Sound_table, tool_wear
        data_file_name (str): Name of the data file for the title
        smoothing (int, optional): Rolling window size for smoothing. If 0 or None, no smoothing applied.
    
    Returns:
        matplotlib.figure.Figure: The generated figure object
    """
    features_to_plot = {
        (0, 0): 'Vib_Spindle',
        (0, 1): 'Vib_Table',
        (0, 2): 'Current',
        (1, 0): 'X_Load_Cell',
        (1, 1): 'Y_Load_Cell',
        (1, 2): 'Z_Load_Cell',
        (2, 0): 'Sound_Spindle',
        (2, 1): 'Sound_table',
        (2, 2): 'tool_wear'  # Added tool wear as the last plot
    }

    # Set a pastel color palette using seaborn
    pastel_palette = sns.color_palette("pastel", 5)

    # Assign colors for each group of features
    color_group1_vib = pastel_palette[0]      
    color_group1_current = pastel_palette[1]   
    color_group2_load = pastel_palette[2]      
    color_group3_sound = pastel_palette[3]     

    # Create the 3x3 multi-plot figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Determine the main title
    main_title = f'Sensor Data File: {data_file_name} | Raw Features and Tool Wear' 
    if smoothing is not None and smoothing > 0:
        main_title = f'{main_title} (Smoothed with window={smoothing})'
    else:
        main_title = main_title + ' (No Smoothing)'
    fig.suptitle(main_title, fontsize=20, y=0.95)

    # Iterate through features and plot
    for (row, col), feature_name in features_to_plot.items():
        ax = axes[row, col]

        # Set light grey background for tool wear plot
        if feature_name == 'tool_wear':
            ax.set_facecolor('#f5f5f5')  # Light grey background
            # Add threshold line
            ax.axhline(y=Config.WEAR_THRESHOLD, color='red', linestyle='--', alpha=0.4, label='Wear Threshold')
            ax.legend(loc='upper left', fontsize=10)

        # Apply smoothing if specified, but not for tool_wear
        if smoothing is not None and smoothing > 0 and feature_name != 'tool_wear':
            data_to_plot = df[feature_name].rolling(window=smoothing, min_periods=1).mean()
            y_label_suffix = ' (Smoothed)'
        else:
            data_to_plot = df[feature_name]
            y_label_suffix = ''

        # Determine plot color
        if feature_name == 'tool_wear':
            plot_color = "#676778"  # Very dark grey for tool wear
        elif feature_name in ['Vib_Spindle', 'Vib_Table']:
            plot_color = color_group1_vib
        elif feature_name == 'Current':
            plot_color = color_group1_current
        elif feature_name in ['X_Load_Cell', 'Y_Load_Cell', 'Z_Load_Cell']:
            plot_color = color_group2_load
        elif feature_name in ['Sound_Spindle', 'Sound_table']:
            plot_color = color_group3_sound
        else:
            plot_color = 'gray'

        # Plot the data
        ax.plot(df['Time'], data_to_plot, color=plot_color, linewidth=3)
        ax.set_title(feature_name, fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(f'Value{y_label_suffix}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    return fig


# ==========================================
# MODEL EVALUATION FUNCTIONS
# ==========================================

def evaluate_single_model(
    model_path: str,
    test_file: str,
    model_type: str,
    wear_threshold: float = Config.WEAR_THRESHOLD
) -> Dict[str, float]:
    """
    Evaluate a single trained model on time-series test data.
    
    Args:
        model_path: Path to the saved model file (.h5 for PyTorch, .zip for PPO)
        test_file: Path to the test CSV file
        model_type: Type of model ('PPO', 'REINFORCE', or 'REINFORCE_AM')
        wear_threshold: Tool wear threshold for environment
    
    Returns:
        Dictionary with precision, recall, f1, and accuracy metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type} model: {os.path.basename(model_path)}")
    print(f"Test file: {os.path.basename(test_file)}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    print(f"{'='*60}")
    
    # Load test data
    test_df = pd.read_csv(test_file)
    
    # Validate required columns
    if 'ACTION_CODE' not in test_df.columns:
        raise ValueError("Test file must contain 'ACTION_CODE' column with human expert actions")
    
    # Create appropriate environment based on model type
    if model_type == 'REINFORCE_AM':
        env = AM_Env(data_file=test_file, wear_threshold=wear_threshold)
        print(f"Created AM_Env with observation space: {env.observation_space.shape}")
    else:
        env = MT_Env(data_file=test_file, wear_threshold=wear_threshold)
        print(f"Created MT_Env with observation space: {env.observation_space.shape}")
    
    # Load model
    if model_type == 'PPO':
        # Load Stable-Baselines3 PPO model
        print(f"Loading PPO model from: {model_path}")
        model = PPO.load(model_path)
        print(f"PPO model loaded successfully")
        
        def predict_action(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)
    
    else:  # REINFORCE or REINFORCE_AM
        # Load PyTorch model
        print(f"Loading PyTorch model from: {model_path}")
        checkpoint = torch.load(model_path)
        input_dim = checkpoint['input_dim']
        output_dim = checkpoint['output_dim']
        print(f"Model architecture: input_dim={input_dim}, output_dim={output_dim}")
        
        policy = PolicyNetwork(input_dim=input_dim, output_dim=output_dim)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        policy.eval()
        print(f"PyTorch model loaded successfully")
        
        def predict_action(obs):
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs).unsqueeze(0)
                probs = policy(state_tensor)
                action = torch.argmax(probs, dim=1)
                return int(action.item())
    
    # Run model on time-series data sequentially
    predictions = []
    ground_truth = test_df['ACTION_CODE'].values
    
    print(f"Processing {len(test_df)} timesteps...")
    
    
    # Reset environment
    env.reset()
    
    for idx in range(len(test_df)):
        # Get observation at this timestep
        env.current_idx = idx
        obs = env.get_observation()
        
        # Predict action
        action = predict_action(obs)
        predictions.append(action)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Debug: Show prediction distribution
    print(f'Ground truth: Classes: {np.unique(ground_truth)} | distribution: {np.bincount(ground_truth.astype(int))}')
    print(f'Prediction:   Classes: {np.unique(predictions)} | distribution: {np.bincount(predictions.astype(int))}')
    
    # Calculate metrics
    precision = precision_score(ground_truth, predictions, average='macro', zero_division=0)
    recall = recall_score(ground_truth, predictions, average='macro', zero_division=0)
    f1 = f1_score(ground_truth, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(ground_truth, predictions)
    
    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }
    
    # Print results
    print(f"\n{model_type} Evaluation Results:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"{'='*60}\n")
    
    return metrics


def evaluate_all_models(
    test_file: str,
    wear_threshold: float = Config.WEAR_THRESHOLD,
    ppo_model_file: str = None,
    reinforce_model_file: str = None,
    reinforce_am_model_file: str = None
) -> pd.DataFrame:
    """
    Evaluate all three trained models on time-series test data.
    
    Args:
        test_file: Path to the test CSV file
        wear_threshold: Tool wear threshold for environment
        ppo_model_file: Path to PPO model (optional, searches for latest if None)
        reinforce_model_file: Path to REINFORCE model (optional, searches for latest if None)
        reinforce_am_model_file: Path to REINFORCE with Attention model (optional, searches for latest if None)
    
    Returns:
        DataFrame with evaluation results for all models
    """
    
    # If paths not provided, try to find latest for this test file
    if ppo_model_file is None:
        ppo_model_file = Config.get_latest_model('PPO', test_file)
    if reinforce_model_file is None:
        reinforce_model_file = Config.get_latest_model('REINFORCE', test_file)
    if reinforce_am_model_file is None:
        reinforce_am_model_file = Config.get_latest_model('REINFORCE_AM', test_file)
        
    results = []
    
    # Evaluate PPO
    if ppo_model_file and os.path.exists(ppo_model_file):
        try:
            metrics = evaluate_single_model(
                model_path=ppo_model_file,
                test_file=test_file,
                model_type='PPO',
                wear_threshold=wear_threshold
            )
            results.append({
                'Model': 'PPO',
                'File': os.path.splitext(os.path.basename(ppo_model_file))[0],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'Accuracy': metrics['accuracy']
            })
        except Exception as e:
            print(f"Error evaluating PPO: {str(e)}")
            results.append({
                'Model': 'PPO',
                'Precision': 0.0,
                'Recall': 0.0,
                'F1': 0.0,
                'Accuracy': 0.0
            })
    else:
        print(f"PPO model not found: {ppo_model_file}")
    
    # Evaluate REINFORCE
    if os.path.exists(reinforce_model_file):
        try:
            metrics = evaluate_single_model(
                model_path=reinforce_model_file,
                test_file=test_file,
                model_type='REINFORCE',
                wear_threshold=wear_threshold
            )
            results.append({
                'Model': 'REINFORCE',
                'File': os.path.splitext(os.path.basename(reinforce_model_file))[0],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'Accuracy': metrics['accuracy']
            })
        except Exception as e:
            print(f"Error evaluating REINFORCE: {str(e)}")
            results.append({
                'Model': 'REINFORCE',
                'Precision': 0.0,
                'Recall': 0.0,
                'F1': 0.0,
                'Accuracy': 0.0
            })
    else:
        print(f"REINFORCE model not found: {reinforce_model_file}")
    
    # Evaluate REINFORCE with Attention
    if os.path.exists(reinforce_am_model_file):
        try:
            metrics = evaluate_single_model(
                model_path=reinforce_am_model_file,
                test_file=test_file,
                model_type='REINFORCE_AM',
                wear_threshold=wear_threshold
            )
            results.append({
                'Model': 'REINFORCE with Attention',
                'File': os.path.splitext(os.path.basename(reinforce_am_model_file))[0],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'Accuracy': metrics['accuracy']
            })
        except Exception as e:
            print(f"Error evaluating REINFORCE with Attention: {str(e)}")
            results.append({
                'Model': 'REINFORCE with Attention',
                'Precision': 0.0,
                'Recall': 0.0,
                'F1': 0.0,
                'Accuracy': 0.0
            })
    else:
        print(f"REINFORCE with Attention model not found: {reinforce_am_model_file}")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df
