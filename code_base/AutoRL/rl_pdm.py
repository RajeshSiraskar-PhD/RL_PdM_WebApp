
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# --- GLOBAL VARIABLES ---
DATA_FILE = "dummy_sensor_data.csv" # Default, will be overwritten
WEAR_THRESHOLD = 300
EPISODES = 100
LR_DEFAULT = 0.001
GAMMA_DEFAULT = 0.99
SMOOTH_WINDOW = 10

# Rewards (Adjustable)
R1 = 1      # Reward for surviving a step
R2 = -100   # Penalty for failure (crossing threshold)
R3 = -5     # Penalty for replacement (or cost of replacement)
# The user wants to optimize for MAX tool usage WITHOUT crossing threshold.
# Long episodes = Good.
# Crossing threshold = Bad.
# Replacing too early = Bad (implied by max usage).

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
            # Try to fix strict case for 'time'
            if 'Time' in missing_cols and 'time' in self.data.columns:
                self.features = ['time' if x=='Time' else x for x in self.features]
                missing_cols.remove('Time')
            elif 'time' in missing_cols and 'Time' in self.data.columns:
                self.features = ['Time' if x=='time' else x for x in self.features]
                missing_cols.remove('time')
            
            # Re-check
            missing_cols = [c for c in self.features if c not in self.data.columns]
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
            # Time, force_x, force_y, force_z, vibration_x, vibration_y, vibration_z, acoustic_emission_rms, tool_wear, ACTION_CODE
            # Usually we don't feed ACTION_CODE as input feature?
            # Prompt says: "IEEE features: Time... ACTION_CODE"
            # I will exclude ACTION_CODE and Time from the *observation* passed to agent usually, 
            # but I will stick to physical readings + wear if requested.
            # I'll exclude 'ACTION_CODE' from observation as it's likely a label.
            # I'll exclude 'Time' as it's often not useful or handled separately, but prompt lists it.
            return ['time', 'force_x', 'force_y', 'force_z', 'vibration_x', 'vibration_y', 'vibration_z', 'acoustic_emission_rms', 'tool_wear']
        elif schema == 'SIT':
            return ['Time', 'Vib_Spindle', 'Vib_Table', 'Sound_Spindle', 'Sound_table', 'X_Load_Cell', 'Y_Load_Cell', 'Z_Load_Cell', 'Current', 'tool_wear']
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
            # Reward:
            # If we replaced too early (wear << threshold), maybe penalty?
            # User says: "Optimize for maximum tool-life usage * WITHOUT * crossing WEAR_THRESHOLD"
            # So ideally we replace exactly AT threshold.
            
            # Use R3 as replacement penalty/cost
            reward = self.R3 
            
            # Logic: We stop this episode because tool is changed.
            terminated = True
            
            # Metric info
            info['wear_margin'] = self.wear_threshold - current_wear
            info['replaced'] = True
            info['threshold_violation'] = (current_wear > self.wear_threshold) # Replaced but was it too late?

        else: # CONTINUE
            # Check if we crossed threshold
            if current_wear > self.wear_threshold:
                # FAILURE!
                reward = self.R2
                terminated = True # Machine broke
                info['wear_margin'] = self.wear_threshold - current_wear # Negative
                info['replaced'] = False
                info['threshold_violation'] = True
            else:
                # Survived another step
                reward = self.R1
                terminated = False
                info['replaced'] = False
                info['threshold_violation'] = False
        
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

def train_single_model(data_file, algo_name, lr, gm, callback_func):
    """
    Trains a single agent and returns the result dictionary.
    """
    # 3 Algos: PPO, A2C, DQN
    algos = {
        'PPO': PPO,
        'A2C': A2C,
        'DQN': DQN
    }
    
    AlgoClass = algos[algo_name]
    combo_name = f"{algo_name} | LR={lr} | G={gm}"
    print(f"Training {combo_name}...")
    
    # Create Env
    # We wrap in DummyVecEnv for SB3
    env = DummyVecEnv([lambda: MT_Env(data_file, WEAR_THRESHOLD, R1, R2, R3)])
    
    try:
        # Initialize Agent
        model = AlgoClass("MlpPolicy", env, learning_rate=lr, gamma=gm, verbose=0)
        
        # Create Callback
        # Calculate Update Frequency (approx 10 updates per run)
        update_freq = max(1, EPISODES // 10)
        
        # We need a wrapper callback to inject the combo_name
        def update_wrapper(metrics):
            if callback_func:
                callback_func(combo_name, metrics)
            
        cb = StreamlitCallback(update_wrapper, update_freq=update_freq)
        
        # Train
        # Hack: Check data length
        data_len = len(pd.read_csv(data_file))
        total_timesteps = EPISODES * data_len 
        
        model.learn(total_timesteps=total_timesteps, callback=cb)
        
        # Collect Final Metrics
        avg_margin = np.mean(cb.wear_margins) if cb.wear_margins else 0
        avg_reward = np.mean(cb.episode_rewards) if cb.episode_rewards else 0
        avg_violations = np.mean(cb.violations) if cb.violations else 0
        avg_replacements = np.mean(cb.replacements) if cb.replacements else 0
        
        return {
            'Agent': algo_name,
            'LR': lr,
            'Gamma': gm,
            'Avg Wear Margin': avg_margin,
            'Avg Reward': avg_reward,
            'Avg Violations': avg_violations,
            'Avg Replacements': avg_replacements,
            'full_metrics': {
                'rewards': cb.episode_rewards,
                'margins': cb.wear_margins,
                'violations': cb.violations,
                'replacements': cb.replacements
            }
        }
        
    except Exception as e:
        import traceback
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
                res = train_single_model(data_file, algo_name, lr, gm, callback_func)
                results.append(res)
                    
    return results

def compare_agents(results):
    # This might return data for the UI to plot
    # Or generate a figure.
    # User wants "Display a SUPERIMPOSED plot for all four plots"
    # We will return the data structure, and App.py will handle plotting with Plotly/Altair/Matplotlib.
    return pd.DataFrame(results)

