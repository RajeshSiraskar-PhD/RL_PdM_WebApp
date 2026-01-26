
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- GLOBAL VARIABLES ---
DATA_FILE = "dummy_sensor_data.csv" # Default, will be overwritten
WEAR_THRESHOLD = 300
EPISODES = 100
LR_DEFAULT = 0.001
GAMMA_DEFAULT = 0.99
SMOOTH_WINDOW = 10
FIXED_X_AXIS_LENGTH = True # Validation: Train for fixed episodes?

# Rewards (Adjustable)
# Updated Strategy:
# 1. Violations (threshold breach): CRITICAL penalty - must be avoided at all costs
# 2. Replacements: Allowed and encouraged when necessary (cheaper than product quality loss)
# 3. Wear Margin: Optimize to be close to threshold but not negative
# 4. Survival: NO BASE REWARD - agents must optimize replacement timing, not episode length

R1 = 0.0       # NO survival reward (prevents 'always continue' policy)
R2 = -1000.0   # CRITICAL penalty for failure (crossing threshold) - very high penalty
R3 = -2.0      # Small penalty for replacement (acceptable cost)
# R4 will be dynamic based on wear margin (incentivize getting close to threshold)

class MT_Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, data_file, wear_threshold=300, r1=1, r2=-100, r3=-5):
        super(MT_Env, self).__init__()
        
        self.data = pd.read_csv(data_file)
        self.wear_threshold = wear_threshold
        self.R1 = r1
        self.R2 = r2
        self.R3 = r3
        
        # Detect Schema
        self.schema = self._detect_schema(self.data.columns)
        self.features = self._get_features(self.schema)
        
        # Define Action Space: 0 = REPLACE, 1 = CONTINUE
        self.action_space = spaces.Discrete(2)
        
        # Define Observation Space
        # We need to normalize or handle ranges, but for now specific Box
        # We exclude 'time', 'tool_wear', 'ACTION_CODE' from observation usually, 
        # but user listed them in features. I will assume 'tool_wear' is HIDDEN from agent 
        # normally in predictive maintenance, BUT the user prompt lists it as a feature.
        # "IEEE features: ... tool_wear ..."
        # If tool_wear is state, the problem is trivial (replace when wear > X).
        # However, I will follow the prompt and include what's requested.
        # If the user wants "predictive", usually we hide the actual wear. 
        # BUT: Prompt says "IEEE features: ... tool_wear ...". I will include it.
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.features),), dtype=np.float32
        )
        
        # VALIDATION: Check if features exist
        missing_cols = [c for c in self.features if c not in self.data.columns]
        if missing_cols:
            # Try to fix case sensitivity for 'time' / 'Time'
            if 'time' in missing_cols and 'Time' in self.data.columns:
                # Replace 'time' (lowercase) with 'Time' (uppercase)
                self.features = ['Time' if x == 'time' else x for x in self.features]
                missing_cols = [c for c in self.features if c not in self.data.columns]
            elif 'Time' in missing_cols and 'time' in self.data.columns:
                # Replace 'Time' (uppercase) with 'time' (lowercase)
                self.features = ['time' if x == 'Time' else x for x in self.features]
                missing_cols = [c for c in self.features if c not in self.data.columns]
            
            # Re-check
            if missing_cols:
                raise ValueError(f"Missing columns in data for schema {self.schema}: {missing_cols}")
        
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
            # IEEE features include: Time, force_x, force_y, force_z, vibration_x, vibration_y, vibration_z, 
            # acoustic_emission_rms, tool_wear, ACTION_CODE
            # We include all available features to match training data schema
            features = ['Time', 'force_x', 'force_y', 'force_z', 'vibration_x', 'vibration_y', 'vibration_z', 
                       'acoustic_emission_rms', 'tool_wear']
            # Check for ACTION_CODE and include if present (for backward compatibility)
            if 'ACTION_CODE' in self.data.columns:
                features.append('ACTION_CODE')
            return features
        elif schema == 'SIT':
            # SIT features include all these columns
            features = ['Time', 'Vib_Spindle', 'Vib_Table', 'Sound_Spindle', 'Sound_table', 'X_Load_Cell', 
                       'Y_Load_Cell', 'Z_Load_Cell', 'Current', 'tool_wear']
            return features
        else:
            # Fallback: all numeric columns except ACTION_CODE
            cols = [c for c in self.data.columns if c != 'ACTION_CODE']
            return cols

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # In a real scenario, we might start at random point or 0. 
        # For this dataset-based env, we simulate a run.
        # If the data contains multiple run-to-failures, we should select one.
        # SIMPLIFICATION: We assume the file is ONE run or we just iterate.
        # If we just iterate, 'reset' acts as 'start from 0'.
        
        # However, to simulate 'episodes', we might need to be careful.
        # If the data is just one long sequence, 'reset' should prob start at 0.
        self.current_step = 0 
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
        
        if action == 0: # REPLACE
            # Agent decides to replace.
            # Episode ends (Tool replaced).
            # Reward: Small penalty for replacement (it's acceptable)
            # But we can add a bonus for replacing at high wear (close to threshold)
            
            # Base replacement cost
            reward = self.R3
            
            # Bonus for replacing at high wear (encourages maximizing tool life)
            # If wear is high relative to threshold, bonus
            wear_ratio = current_wear / self.wear_threshold  # 0 to ~1
            if wear_ratio > 0.7:  # If wear > 70% of threshold
                bonus = (wear_ratio - 0.7) * 10  # Bonus increases as wear approaches threshold
                reward += bonus
            
            # Logic: We stop this episode because tool is changed.
            terminated = True
            
            # Metric info
            info['wear_margin'] = max(0, self.wear_threshold - current_wear)  # Ensure non-negative
            info['replaced'] = True
            info['threshold_violation'] = (current_wear > self.wear_threshold) # Replaced but was it too late?

        else: # CONTINUE
            # Check if we crossed threshold (CRITICAL FAILURE)
            if current_wear > self.wear_threshold:
                # FAILURE! - Use critical penalty
                reward = self.R2
                terminated = True # Machine broke
                info['wear_margin'] = max(0, self.wear_threshold - current_wear)  # Will be negative, track it
                info['replaced'] = False
                info['threshold_violation'] = True
            else:
                # Survived another step
                # Base reward for surviving
                reward = self.R1
                
                # Bonus for having high wear margin (close to threshold but not exceeded)
                # This encourages the agent to keep the tool running as long as possible
                wear_ratio = current_wear / self.wear_threshold  # 0 to 1
                if wear_ratio > 0.5:  # If wear is significant (> 50%)
                    # Give bonus for operating in high wear region (riskier but more efficient)
                    margin_bonus = (wear_ratio - 0.5) * 2  # Small bonus, encourages pushing wear higher
                    reward += margin_bonus
                
                terminated = False
                info['replaced'] = False
                info['threshold_violation'] = False
        
        # Ensure wear_margin is always recorded (non-negative for success cases)
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
    def __init__(self, update_func, update_freq=1, verbose=0):
        super(StreamlitCallback, self).__init__(verbose)
        self.update_func = update_func
        self.update_freq = update_freq
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

        # return True $ additoinal

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
        output = self.projection(weighted_features)
        
        return output

FIXED_X_AXIS_LENGTH = True

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

def calculate_weighted_score(violations, wear_margin, replacements):
    """
    Calculate a normalized weighted score for comparing agents.
    
    Scoring approach:
    - Violations: Lowest best, weight=0.5 (critical)
    - Wear Margin: Lowest best, weight=0.3 (optimize to be close to threshold)
    - Replacements: Lowest best, weight=0.2 (acceptable cost)
    
    Returns a score where 1.0 is the best possible for each component,
    and the weighted sum gives overall performance.
    """
    # All three should be minimized (lower is better)
    # Violations: Should be 0 ideally
    # Wear Margin: Should be close to 0 (close to threshold) ideally
    # Replacements: Should be low ideally
    
    # Normalize each metric
    # For violations: 1.0 if 0 violations, lower score if violations occur
    violations_score = 1.0 if violations == 0 else 1.0 / (1.0 + violations)
    
    # For wear_margin: Lower is better (closer to threshold is better)
    # Use reciprocal so that small margins score high
    # Add +1 to avoid division by zero and handle edge cases
    wear_margin_score = 1.0 / (1.0 + wear_margin)
    
    # For replacements: Prefer fewer replacements
    # 1.0 for 0 replacements, decreasing for higher counts
    replacements_score = 1.0 if replacements == 0 else 1.0 / (1.0 + replacements)
    
    # Weighted combination
    weighted_score = (
        0.5 * violations_score +
        0.3 * wear_margin_score +
        0.2 * replacements_score
    )
    
    return weighted_score

def train_single_model(data_file, algo_name, lr, gm, callback_func, attention_type=None):
    """
    Trains a single agent and returns the result dictionary.
    attention_type: None, 'NW', or 'DL'
    """
    # 3 Algos: PPO, A2C, DQN
    algos = {
        'PPO': PPO,
        'A2C': A2C,
        'DQN': DQN
    }
    
    AlgoClass = algos[algo_name]
    
    # Construct Combo Name
    att_suffix = ""
    if attention_type == 'NW':
        att_suffix = " (NW)"
    elif attention_type == 'DL':
        att_suffix = " (DL)"
        
    combo_name = f"{algo_name}{att_suffix} | LR={lr} | G={gm}"
    print(f"Training {combo_name}...")
    
    # Create Env
    # We wrap in DummyVecEnv for SB3
    env = DummyVecEnv([lambda: MT_Env(data_file, WEAR_THRESHOLD, R1, R2, R3)])
    
    # Policy kwargs
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
            
        cb_streamlit = StreamlitCallback(update_wrapper, update_freq=update_freq)
        
        # Combine Callbacks
        callbacks = [cb_streamlit]
        
        # Time Management
        data_len = len(pd.read_csv(data_file))
        
        if FIXED_X_AXIS_LENGTH:
            # Train for effectively infinite steps, but stop at X episodes
            total_timesteps = 10**9 
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
        
        # Calculate Weighted Score
        weighted_score = calculate_weighted_score(avg_violations, avg_margin, avg_replacements)
        
        # Save Model with naming convention: algo_Episodes_LR_Gamma[_Attention]
        # Examples: DQN_200_01_90, PPO_200_001_99_NW, A2C_200_001_99_DL
        ep_str = f"{EPISODES:03d}"  # Episodes as 3-digit (e.g., 100, 200)
        lr_str = f"{int(lr * 1000):03d}" if lr < 1 else f"{lr:.2f}".replace(".", "")
        gm_str = f"{int(gm * 100):02d}"
        
        # Build filename with optional attention suffix
        att_suffix_file = ""
        if attention_type == 'NW':
            att_suffix_file = "_NW"
        elif attention_type == 'DL':
            att_suffix_file = "_DL"
        
        model_filename = f"{algo_name}_{ep_str}_{lr_str}_{gm_str}{att_suffix_file}"
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
    
    # 3 Algos: PPO, A2C, DQN
    algo_names = ['PPO', 'A2C', 'DQN']
    
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


def evaluate_model(model_path, data_file, wear_threshold=300):
    """
    Evaluate a trained model on test data.
    
    NOTE: This is a HISTORICAL REPLAY evaluation - we process all data points
    regardless of whether the tool exceeds the threshold. This is different from
    training episodes which terminate early.
    
    Returns:
    {
        'timesteps': [list of timesteps],
        'tool_wear': [list of tool wear values],
        'actions': [list of actions taken (0 or 1)],
        'wear_threshold': wear_threshold value,
        'total_replacements': number of replacements,
        'threshold_violations': number of times threshold was exceeded
    }
    """
    try:
        # Determine algo from model filename
        model_name = os.path.basename(model_path)
        algo_name = model_name.split('_')[0]  # Extract algo (PPO, A2C, or DQN)
        
        # Load model
        algos = {
            'PPO': PPO,
            'A2C': A2C,
            'DQN': DQN
        }
        
        AlgoClass = algos.get(algo_name, A2C)
        model = AlgoClass.load(model_path)
        
        # Load data directly for evaluation (not wrapped in environment)
        data = pd.read_csv(data_file)
        
        # Create environment just for feature extraction and observation building
        env = MT_Env(data_file, wear_threshold)
        
        # Validate observation shape
        expected_shape = model.observation_space.shape[0]
        
        # Track evaluation data
        timesteps = []
        tool_wear_values = []
        actions_taken = []
        total_replacements = 0
        threshold_violations = 0
        
        # Process all data points (historical replay - no early termination)
        for timestep, idx in enumerate(range(len(data))):
            try:
                # Get the observation for this row
                obs = data.iloc[idx][env.features].values.astype(np.float32)
                
                # Validate shape
                if len(obs) != expected_shape:
                    raise ValueError(
                        f"Feature mismatch at row {idx}! Expected {expected_shape} features but got {len(obs)}.\n"
                        f"Features: {env.features}\n"
                        f"Data shape: {data.shape}"
                    )
                
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Store data
                current_wear = data.iloc[idx]['tool_wear']
                timesteps.append(timestep)
                tool_wear_values.append(float(current_wear))
                actions_taken.append(int(action))
                
                # Track metrics
                if action == 0:  # REPLACE
                    total_replacements += 1
                
                if current_wear > wear_threshold:
                    threshold_violations += 1
                    
            except Exception as e:
                raise Exception(f"Error processing row {idx}: {str(e)}")
        
        return {
            'timesteps': timesteps,
            'tool_wear': tool_wear_values,
            'actions': actions_taken,
            'wear_threshold': wear_threshold,
            'total_replacements': total_replacements,
            'threshold_violations': threshold_violations
        }
    
    except Exception as e:
        # Re-raise with context
        raise Exception(f"Evaluation failed: {str(e)}")


def plot_sensor_data(data_file):
    """
    Plots sensor data based on the detected schema (IEEE or SIT).
    """
    try:
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

        # Create Subplots (3x3 Grid)
        # Tight spacing: vertical_spacing=0.05 (default is usually 0.3/rows, let's make it small)
        fig = make_subplots(rows=3, cols=3, subplot_titles=features_to_plot, vertical_spacing=0.05, horizontal_spacing=0.02)
        
        for i, feature in enumerate(features_to_plot):
            row = (i // 3) + 1
            col = (i % 3) + 1
            if row > 3: break # Limit to 9 plots
            
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


