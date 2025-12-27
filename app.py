# ------------------------------------------------------------------------------------
# RL for Predictive Maintenance
# Web-app code
# Author: Rajesh Siraskar
# V.1.0: 27-Dec-2025 - Hidden State Design
# ------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from rl_pdm_module import (
    train_ppo, REINFORCE, PolicyNetwork, MT_Env, AM_Env,
    plot_metrics, Config, plot_training_live, average_metrics,
    plot_sensor_data_with_wear, evaluate_single_model, evaluate_all_models,
    train_HSD_model
)

# Set parameters for agents dynamically
episodes = Config.EPISODES
learning_rate = Config.LEARNING_RATE
gamma = Config.GAMMA

# Initial model paths
model_file = Config.get_model_path('REINFORCE')
ppo_model_file = Config.get_model_path('PPO')
reinforce_am_model_file = Config.get_model_path('REINFORCE_AM')

# --- Page Configuration ---
st.set_page_config(
    page_title="RL for Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Theming ---
# #d2e3fa, #87CDEE #cad3db
st.markdown("""
    <style>
        /* Main App Background (Right Column) - Very Light Grey */
        .stApp {
            background-color: #F0F2F6;
        }
        
        /* Sidebar Background (Left Column) - Pastel Blue */
        [data-testid="stSidebar"] {
            background-color: #a4d4eb;
            border-right: 1px solid #d1e7fbad;
        }

        /* Adjusting text colors */
        h1 {
            color: #2C3E50;
            font-size: 28px !important;
            margin-top: -2rem !important;
            margin-bottom: -0.5rem !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        
        hr {
            margin-top: 0.5rem !important;
            margin-bottom: 1rem !important;
        }
        h2 {
            color: #2C3E50;
            font-size: 22px !important;
        }
        h3 {
            color: #2C3E50;
            font-size: 18px !important;
        }
        
        /* Custom button styling */
        .stButton > button {
            width: 100%;
            background-color: #F0F2F6;
            border: 1px solid #B0C4DE;
        }
        
        .stButton > button:hover {
            background-color: #e1e6eb;
            border: 1px solid #A9C5E0;
        }
    </style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'idle' # options: idle, training_rf, training_ppo, training_rf_am, evaluation, compare
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        'REINFORCE': None,
        'PPO': None,
        'REINFORCE_AM': None,
        'HSD_Basic': None,
        'HSD_AM': None
    }
if 'training_log' not in st.session_state:
    st.session_state.training_log = {
        'REINFORCE': [],
        'PPO': [],
        'REINFORCE_AM': [],
        'HSD_Basic': [],
        'HSD_AM': []
    }
if 'averaged_metrics' not in st.session_state:
    st.session_state.averaged_metrics = {
        'REINFORCE': None,
        'PPO': None,
        'REINFORCE_AM': None,
        'HSD_Basic': None,
        'HSD_AM': None
    }
if 'sensor_data_plot' not in st.session_state:
    st.session_state.sensor_data_plot = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []

# --- Helper Functions for Dummy Logic ---

def train_reinforce_agent(data_file: str, data_file_name: str, episodes: int = 50, save_path: str = None):
    """
    Train REINFORCE agent and store metrics and averages.
    """
    try:
        # Create environment
        env = MT_Env(data_file=data_file)
        
        # Create policy network
        policy = PolicyNetwork(
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.n
        )
        
        # Create agent
        agent = REINFORCE(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            model_file=save_path if save_path else model_file,
            agent_name="REINFORCE",
            data_file_name=data_file_name
        )
        
        # Train
        metrics = agent.learn(total_episodes=episodes)
        
        # --- Train HSD Baseline (Basic) ---
        cml_save_path = save_path.replace('_REINFORCE_', '_HSD_Basic_') if save_path else None
        cml_model, cml_metrics = train_HSD_model(data_file=data_file, save_path=cml_save_path)
        
        # Store metrics and calculate averages
        st.session_state.metrics['REINFORCE'] = metrics
        st.session_state.averaged_metrics['REINFORCE'] = {
            'avg_reward': float(np.nanmean(metrics['rewards'])),
            'avg_violations': float(np.nanmean(metrics['violations'])),
            'avg_replacements': float(np.nanmean(metrics['replacements'])),
            'avg_margin': float(np.nanmean([m for m in metrics['margins'] if not np.isnan(m)]))
        }
        
        # Store CML Basic metrics if available
        if cml_metrics:
            st.session_state.metrics['HSD_Basic'] = cml_metrics
            st.session_state.averaged_metrics['HSD_Basic'] = {
                'avg_reward': 0, 'avg_violations': 0, 'avg_replacements': 0, # Not applicable for static CML
                'avg_margin': 0,
                'precision': cml_metrics['precision'],
                'recall': cml_metrics['recall'],
                'f1': cml_metrics['f1'],
                'accuracy': cml_metrics['accuracy']
            }

        st.session_state.training_log['REINFORCE'].append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'episodes': episodes,
            'data_file_name': data_file_name
        })

        st.toast(f"Agent trained. Model saved as: {model_file}", icon="💾", duration=3)
        return agent, metrics
    
    except Exception as e:
        st.error(f"Error training REINFORCE: {str(e)}")
        return None, None


def train_ppo_agent(data_file: str, data_file_name: str, episodes: int = 50, save_path: str = None):
    """
    Train PPO agent and store metrics and averages.
    """
    try:
        # Create environment
        env = MT_Env(data_file=data_file)
        
        # Train
        metrics = train_ppo(
            env=env,
            total_episodes=episodes,
            learning_rate=learning_rate,
            gamma=gamma,
            model_file=save_path if save_path else ppo_model_file,
            data_file_name=data_file_name
        )
        
        # --- Train HSD Baseline (Basic) ---
        cml_save_path = save_path.replace('_PPO_', '_HSD_Basic_') if save_path else None
        cml_model, cml_metrics = train_HSD_model(data_file=data_file, save_path=cml_save_path)

        # Store metrics and calculate averages
        st.session_state.metrics['PPO'] = metrics
        st.session_state.averaged_metrics['PPO'] = {
            'avg_reward': float(np.nanmean(metrics['rewards'])),
            'avg_violations': float(np.nanmean(metrics['violations'])),
            'avg_replacements': float(np.nanmean(metrics['replacements'])),
            'avg_margin': float(np.nanmean([m for m in metrics['margins'] if not np.isnan(m)]))
        }
        
        # Store CML Basic metrics if available
        if cml_metrics:
            st.session_state.metrics['HSD_Basic'] = cml_metrics
            st.session_state.averaged_metrics['HSD_Basic'] = {
                'avg_reward': 0, 'avg_violations': 0, 'avg_replacements': 0,
                'avg_margin': 0,
                'precision': cml_metrics['precision'],
                'recall': cml_metrics['recall'],
                'f1': cml_metrics['f1'],
                'accuracy': cml_metrics['accuracy']
            }

        st.session_state.training_log['PPO'].append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'episodes': episodes,
            'data_file_name': data_file_name
        })

        st.toast(f"PPO agent trained. Model saved as: {ppo_model_file}", icon="💾", duration=3)
        return metrics
    
    except Exception as e:
        st.error(f"Error training PPO: {str(e)}")
        return None


def train_reinforce_am_agent(data_file: str, data_file_name: str, episodes: int = 50, save_path: str = None):
    """
    Train REINFORCE with Attention agent and store metrics and averages.
    """
    try:
        # Create attention-augmented environment
        env = AM_Env(data_file=data_file)
        
        # Create policy network
        policy = PolicyNetwork(
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.n
        )
        
        # Create agent
        agent = REINFORCE(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            gamma=gamma,
            model_file=save_path if save_path else reinforce_am_model_file,
            agent_name="REINFORCE_AM",
            data_file_name=data_file_name
        )
        
        # Train
        metrics = agent.learn(total_episodes=episodes)
        
        # --- Train HSD Baseline (Basic) ---
        HSD_Basic_path = save_path.replace('_REINFORCE_AM_', '_HSD_Basic_') if save_path else None
        HSD_Basic_model, HSD_Basic_metrics = train_HSD_model(data_file=data_file, save_path=HSD_Basic_path)
        
        # --- Train HSD (Attention) ---
        HSD_AM_path = save_path.replace('_REINFORCE_AM_', '_HSD_AM_') if save_path else None
        HSD_AM_model, HSD_AM_metrics = train_HSD_model(
            data_file=data_file, 
            save_path=HSD_AM_path,
            attention_weights=env.attention_weights
        )
        
        # Store metrics and calculate averages
        st.session_state.metrics['REINFORCE_AM'] = metrics
        st.session_state.averaged_metrics['REINFORCE_AM'] = {
            'avg_reward': float(np.nanmean(metrics['rewards'])),
            'avg_violations': float(np.nanmean(metrics['violations'])),
            'avg_replacements': float(np.nanmean(metrics['replacements'])),
            'avg_margin': float(np.nanmean([m for m in metrics['margins'] if not np.isnan(m)]))
        }
        
        # Store CML metrics
        if HSD_Basic_metrics:
            st.session_state.metrics['HSD_Basic'] = HSD_Basic_metrics
            st.session_state.averaged_metrics['HSD_Basic'] = {
                'avg_reward': 0.0, 'avg_violations': 0.0, 'avg_replacements': 0.0, 'avg_margin': 0.0,
                'precision': HSD_Basic_metrics['precision'],
                'recall': HSD_Basic_metrics['recall'],
                'f1': HSD_Basic_metrics['f1'],
                'accuracy': HSD_Basic_metrics['accuracy']
            }
        
        if HSD_AM_metrics:
            st.session_state.metrics['HSD_AM'] = HSD_AM_metrics
            st.session_state.averaged_metrics['HSD_AM'] = {
                'avg_reward': 0.0, 'avg_violations': 0.0, 'avg_replacements': 0.0, 'avg_margin': 0.0,
                'precision': HSD_AM_metrics['precision'],
                'recall': HSD_AM_metrics['recall'],
                'f1': HSD_AM_metrics['f1'],
                'accuracy': HSD_AM_metrics['accuracy']
            }

        st.session_state.training_log['REINFORCE_AM'].append({
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'episodes': episodes,
            'data_file_name': data_file_name
        })

        st.toast(f"REINFORCE with Attention agent trained. Model saved as: {reinforce_am_model_file}", icon="💾", duration=3)
        return agent, metrics
    
    except Exception as e:
        st.error(f"Error training REINFORCE with Attention: {str(e)}")
        return None, None


def generate_eval_metrics():
    """Generates dummy evaluation metrics."""
    return {
        "Accuracy": f"{np.random.randint(85, 98)}%",
        "Precision": f"{np.random.randint(80, 95)}%",
        "Recall": f"{np.random.randint(80, 95)}%",
        "F1 Score": f"{np.random.uniform(0.8, 0.95):.2f}"
    }


# ==========================================
# HELPER FUNCTION FOR EVALUATION TABLE
# ==========================================
def display_evaluation_table():
    """Display evaluation metrics table for all models in session state."""
    
    if not st.session_state.evaluation_results:
        st.info("No evaluation results to display.")
        return
        
    # Prepare data for table
    models_data = []
    
    for row in st.session_state.evaluation_results:
        models_data.append({
            'Model': row.get('Model', 'Unknown'),
            'File': row.get('File', 'N/A'),
            'Precision': f"{row['Precision']:.4f}",
            'Recall': f"{row['Recall']:.4f}",
            'F1': f"{row['F1']:.4f}",
            'Accuracy': f"{row['Accuracy']:.4f}"
        })
    
    # Create HTML table with highlighting
    html_table = """
    <table style='width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;'>
    <thead>
    <tr style='border-bottom: 2px solid rgba(180, 180, 180, 0.6);'>
    <th style='text-align: left; padding: 12px 8px; font-weight: 600; font-size: 14px;'>Model</th>
    <th style='text-align: left; padding: 12px 8px; font-weight: 600; font-size: 14px;'>File</th>
    <th style='text-align: right; padding: 12px 8px; font-weight: 600; font-size: 14px;'>Precision</th>
    <th style='text-align: right; padding: 12px 8px; font-weight: 600; font-size: 14px;'>Recall</th>
    <th style='text-align: right; padding: 12px 8px; font-weight: 600; font-size: 14px;'>F1 Score</th>
    <th style='text-align: right; padding: 12px 8px; font-weight: 600; font-size: 14px;'>Accuracy</th>
    </tr>
    </thead>
    <tbody>
    """
    
    # Find best values for highlighting
    numeric_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': []
    }
    
    for m in st.session_state.evaluation_results:
        for metric_key in numeric_metrics.keys():
            val = m.get(metric_key.capitalize())
            if val is not None:
                numeric_metrics[metric_key].append(val)
    
    best_values = {}
    for metric_key, values in numeric_metrics.items():
        if values:
            best_values[metric_key] = max(values)
        else:
            best_values[metric_key] = None
    
    # Add rows
    for row_data in models_data:
        html_table += "<tr style='border-bottom: 1px solid rgba(220, 220, 220, 0.3);'>"
        html_table += f"<td style='text-align: left; padding: 10px 8px; font-size: 13px;'>{row_data['Model']}</td>"
        html_table += f"<td style='text-align: left; padding: 10px 8px; font-size: 13px;'>{row_data['File']}</td>"
        
        # Highlight best values
        for metric_name in ['Precision', 'Recall', 'F1', 'Accuracy']:
            value_str = row_data[metric_name]
            metric_key = metric_name.lower().replace(' score', '')
            
            style = 'text-align: right; padding: 10px 8px; font-size: 13px;'
            
            if value_str != 'N/A' and best_values.get(metric_key) is not None:
                value_float = float(value_str)
                if abs(value_float - best_values[metric_key]) < 1e-6:
                    style += ' background-color: #90EE90; font-weight: 600;'
            
            html_table += f"<td style='{style}'>{value_str}</td>"
        
        html_table += "</tr>"
    
    html_table += "</tbody></table>"
    
    st.markdown(html_table, unsafe_allow_html=True)


# ==========================================
# LEFT PANEL (SIDEBAR)
# ==========================================
with st.sidebar:
    # --- Section 1: Agent Training ---
    st.header('Agent Training')
    
    train_file = st.file_uploader("Select milling machine data:", type=['csv', 'txt', 'json'], key="train_uploader")
    
    if train_file:
        st.toast("Training file loaded.", icon="✅", duration=3)
        # Auto-generate sensor data visualization
        if st.session_state.uploaded_data is None or st.session_state.uploaded_data.name != train_file.name:
            try:
                if hasattr(train_file, 'seek'):
                    train_file.seek(0)
                df = pd.read_csv(train_file)
                st.session_state.uploaded_data = train_file
                st.session_state.sensor_data_plot = plot_sensor_data_with_wear(
                    df=df,
                    data_file_name=train_file.name,
                    smoothing=Config.PLOT_SMOOTHING_FACTOR
                )
                st.toast("Sensor data visualization generated!", icon="📊", duration=3)
            except Exception as e:
                st.warning(f"Could not generate sensor visualization: {str(e)}")
    
    # Hyperparameters
    st.subheader("Hyperparameters")
    
    episodes = st.number_input(
        "Episodes",
        min_value=1,
        value=Config.EPISODES,
        step=10
    )
    
    # Update model paths dynamically based on current episodes input and uploaded file
    if train_file:
        model_file = Config.get_model_path('REINFORCE', episodes, data_file=train_file.name)
        ppo_model_file = Config.get_model_path('PPO', episodes, data_file=train_file.name)
        reinforce_am_model_file = Config.get_model_path('REINFORCE_AM', episodes, data_file=train_file.name)
    else:
        model_file = Config.get_model_path('REINFORCE', episodes)
        ppo_model_file = Config.get_model_path('PPO', episodes)
        reinforce_am_model_file = Config.get_model_path('REINFORCE_AM', episodes)
    
    learning_rate = st.number_input(
        "Learning Rate (α)",
        min_value=1e-5,
        max_value=1.0,
        value=Config.LEARNING_RATE,
        format="%.5f"
    )
    
    gamma = st.number_input(
        "Discount Factor (γ)",
        min_value=0.0,
        max_value=0.9999,
        value=Config.GAMMA,
        format="%.4f"
    )
        
    # Button 1: Train PPO
    if st.button('Train PPO agent', use_container_width=True):
        st.session_state.view_mode = 'training_ppo'
    # Progress Bar container
    pbar_ppo = st.empty()

    # Button 2: Train REINFORCE
    if st.button('Train REINFORCE agent', use_container_width=True):
        st.session_state.view_mode = 'training_rf'   
    # Progress Bar container (will be used by logic below)
    pbar_rf = st.empty()

    # Button 3: Train REINFORCE with Attention
    if st.button('Train REINFORCE with Attention', use_container_width=True):
        st.session_state.view_mode = 'training_rf_am'
    # Progress Bar container
    pbar_rf_am = st.empty()
    
    st.markdown("---")
    
    # Count available metrics for comparison
    available_metrics = sum([
        st.session_state.metrics['REINFORCE'] is not None,
        st.session_state.metrics['PPO'] is not None,
        st.session_state.metrics['REINFORCE_AM'] is not None,
        st.session_state.metrics['HSD_Basic'] is not None,
        st.session_state.metrics['HSD_AM'] is not None
    ])
    
    # Button 4: Compare Agents (enabled only when at least 2 metrics available)
    compare_enabled = available_metrics >= 2
    if st.button('Compare Agents', use_container_width=True, disabled=not compare_enabled):
        st.session_state.view_mode = 'compare'
    
    if not compare_enabled and available_metrics > 0:
        st.caption(f"ℹ️ Train at least 2 agents to enable comparison ({available_metrics}/3)")
    
    st.markdown("---")
    
    # Button 5: View Training Logs
    if st.button('View Training Logs', use_container_width=True):
        st.session_state.view_mode = 'training_logs'

    # --- Section 2: Agent Evaluation ---
    st.header('Agent Evaluation')

    # File picker for evaluation dataset
    eval_file = st.file_uploader("Select evaluation dataset:", type=['csv', 'txt', 'json'], key="eval_uploader")

    if eval_file:
        st.toast("Evaluation file loaded!", icon="✅", duration=3)
    
    # Button: Evaluate All Saved Models
    if st.button('Evaluate Trained Models', use_container_width=True, disabled=eval_file is None):
        st.session_state.view_mode = 'evaluation_all'
    
    # NEW Button: Evaluate the most recentSaved Models
    if st.button('Evaluate Saved Models', use_container_width=True, disabled=eval_file is None):
        st.session_state.view_mode = 'evaluation_saved'
    
    if eval_file is None:
        st.caption("ℹ️ Select evaluation dataset to proceed")

# ==========================================
# RIGHT PANEL (MAIN AREA)
# ==========================================

st.markdown("<h1 style='color: #0066b2;'>Reinforcement Learning for Predictive Maintenance</h1>", unsafe_allow_html=True)
st.markdown("V.0.4 - Hidden State Design - 27-Dec-2025")
st.markdown("---")

# Container for the dynamic right column content
main_container = st.container()

with main_container:
    if st.session_state.view_mode == 'idle':
        if st.session_state.sensor_data_plot is not None:
            st.markdown("### Uploaded Sensor Data Visualization")
            st.pyplot(st.session_state.sensor_data_plot)
            st.markdown("---")
            st.info("<< Training file loaded. Select an agent and start training from the sidebar.")
        else:
            st.info("<< Please select a training file and start training an agent from the sidebar.")
            st.markdown("### System Status")
            st.write("Ready to process milling machine telemetry data.")

    # --- Training View ---
    elif st.session_state.view_mode == 'training_rf':
        # st.subheader("Training REINFORCE Agent")
        
        # Train with live visualization
        agent, metrics = train_reinforce_agent(
            data_file=train_file,
            data_file_name=train_file.name,
            episodes=episodes,
            save_path=model_file
        )
        
        st.toast("✅ REINFORCE Training Complete. Model saved.", icon="💾", duration=3)

    elif st.session_state.view_mode == 'training_ppo':
        # st.subheader("Training PPO Agent")
        
        # Train PPO agent
        metrics = train_ppo_agent(
            data_file=train_file,
            data_file_name=train_file.name,
            episodes=episodes,
            save_path=ppo_model_file
        )
        
        st.toast("✅ PPO Training Complete. Model saved.", icon="💾", duration=3)

    elif st.session_state.view_mode == 'training_rf_am':
        # st.subheader("Training REINFORCE with Attention Agent")
        
        # Train REINFORCE+AM agent
        agent, metrics = train_reinforce_am_agent(
            data_file=train_file,
            data_file_name=train_file.name,
            episodes=episodes,
            save_path=reinforce_am_model_file
        )
        
        st.toast("✅ REINFORCE+Attention Training Complete. Model saved.", icon="💾", duration=3)
    
    # --- Comparison View ---
    elif st.session_state.view_mode == 'compare':
        # st.subheader("Agent Performance Comparison")
        
        # Get available metrics (ordered: PPO, REINFORCE, REINFORCE_AM)
        available_agents = []
        for agent in ['PPO', 'REINFORCE', 'REINFORCE_AM', 'HSD_Basic', 'HSD_AM']:
            if st.session_state.metrics[agent] is not None:
                available_agents.append(agent)
        
        # Display names mapping $$$ ² ✲
        agent_display_names = {
            'REINFORCE': 'REINFORCE',
            'PPO': 'PPO',
            'REINFORCE_AM': 'REINFORCE with Attention',
            'HSD_Basic': 'REINFORCE*',
            'HSD_AM': 'REINFORCE* with Attention'
        }
        
        if len(available_agents) >= 2:
            # Display average metrics comparison with user-friendly names
            st.markdown("### Average Metrics Comparison")
            st.markdown("")
            
            # Metric display names
            metric_names = {
                'avg_reward': 'Average Reward',
                'avg_violations': 'Violation Rate',
                'avg_replacements': 'Replacement Rate',
                'avg_margin': 'Wear Margin',
                'precision': 'Precision (Val)',
                'recall': 'Recall (Val)',
                'f1': 'F1 Score (Val)',
                'accuracy': 'Accuracy (Val)'
            }
            
            # Build HTML table with fixed column widths
            num_cols = len(available_agents)
            col_width = 100  # Fixed width for each agent column in px
            metric_width = 100  # Fixed width for metric column
            
            # Create HTML table
            html_table = f"""
            <table style='width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;'>
            <thead>
            <tr style='border-bottom: 1px solid rgba(180, 180, 180, 0.6);'>
            <th style='text-align: left; width: {metric_width}px; padding: 8px 0; font-weight: 600;'>Metric</th>
            """
            
            for agent in available_agents:
                display_name = agent_display_names[agent]
                html_table += f"<th style='text-align: right; width: {col_width}px; padding: 8px 4px; font-weight: 600;'>{display_name}</th>"
            
            html_table += "</tr></thead><tbody>"
            
            # Add metric rows
            for metric_key, metric_display in metric_names.items():
                html_table += "<tr style='border-bottom: 1px solid rgba(220, 220, 220, 0.3);'>"
                html_table += f"<td style='text-align: left; padding: 6px 0;'>{metric_display}</td>"
                
                for i, agent in enumerate(available_agents):
                    value = st.session_state.averaged_metrics[agent].get(metric_key, np.nan)
                    if value is None or (isinstance(value, float) and np.isnan(value)) or (metric_key.startswith('avg_') and agent.startswith('CML') and value == 0):
                        value_str = "N/A"
                    else:
                        value_str = f"{value:.3f}"
                    
                    html_table += f"<td style='text-align: right; width: {col_width}px; padding: 6px 4px;'>{value_str}</td>"
                
                html_table += "</tr>"
            
            html_table += "</tbody></table>"
            
            st.markdown(html_table, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Display comparison plots
            # st.markdown("#### Training Progress Comparison")
            
            W, H = 16, 7
            FONTSIZE_SUPER = 14
            FONTSIZE_TITLE = 11
            FONTSIZE_LABEL = 10
            FONTSIZE_TICK = 9
            BACKGROUND_COLOR = '#fafafa'
            
            # Create comparison figure for all available agents
            fig = plt.figure(figsize=(W, H))
            
            metrics_config = [
                {'data': 'rewards', 'title': 'Learning Curve (Smoothed Average Reward)', 'ylabel': 'Average Reward'},
                {'data': 'violations', 'title': 'Threshold Violations (Smoothed)', 'ylabel': 'Violation Rate'},
                {'data': 'replacements', 'title': 'Replacements per Episode (Smoothed)', 'ylabel': 'Replacement Rate'},
                {'data': 'margins', 'title': 'Wear Margin at Replacement (Smoothed)', 'ylabel': 'Wear Margin'}
            ]
            
            fig, axs = plt.subplots(2, 2, figsize=(W, H))
            fig.suptitle('Agent Performance Comparison', fontsize=FONTSIZE_SUPER, color='#2C3E50')
            
            colors = {'REINFORCE': '#1f77b4', 'PPO': '#ff7f0e', 'REINFORCE_AM': '#2ca02c'}
            
            for idx, config in enumerate(metrics_config):
                ax = axs[idx // 2, idx % 2]
                data_key = config['data']
                
                for agent_name in available_agents:
                    metrics_dict = st.session_state.metrics[agent_name]
                    data = metrics_dict.get(data_key, [])
                    
                    if len(data) > 0:
                        if config['data'] == 'margins':
                            data = [m for m in data if not np.isnan(m)]
                        
                        if len(data) > 1:
                            smooth_data = pd.Series(data).rolling(window=10, min_periods=1).mean().to_numpy()
                        else:
                            smooth_data = np.array(data)
                        
                        ax.plot(smooth_data, label=agent_name, color=colors.get(agent_name, 'blue'), linewidth=2, alpha=0.5)
                
                ax.set_title(config['title'], fontsize=FONTSIZE_TITLE)
                ax.set_xlabel('Episode', fontsize=FONTSIZE_LABEL)
                ax.set_ylabel(config['ylabel'], fontsize=FONTSIZE_LABEL)
                ax.tick_params(labelsize=FONTSIZE_TICK)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_facecolor(BACKGROUND_COLOR)
                ax.legend(loc='best', fontsize=9)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            st.pyplot(fig)
            
            # Save artifacts
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("💾 Save Comparison", key="save_compare_main"):
                    try:
                        os.makedirs('saved_plots', exist_ok=True)
                        date_str = pd.Timestamp.now().strftime('%d%m')
                        
                        # 1. Save Plot
                        plot_filename = f"Comparison_plot_{date_str}.png"
                        plot_path = os.path.join('saved_plots', plot_filename)
                        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                        
                        # 2. Save Metrics Table
                        csv_filename = f"Comparison_metrics_{date_str}.csv"
                        csv_path = os.path.join('saved_plots', csv_filename)
                        
                        # Construct DataFrame for saving
                        csv_data = []
                        for agent in available_agents:
                            row = {'Agent': agent}
                            for metric_key, metric_display in metric_names.items():
                                val = st.session_state.averaged_metrics[agent].get(metric_key, np.nan)
                                row[metric_display] = val
                            csv_data.append(row)
                        
                        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
                        
                        st.success(f"✅ Saved:\n- Plot: {plot_filename}\n- Metrics: {csv_filename}")
                        
                    except Exception as e:
                        st.error(f"Error saving comparison artifacts: {str(e)}")
        else:
            st.warning("Not enough agents trained for comparison. Train at least 2 agents.")
    
    # --- Training Logs View ---
    elif st.session_state.view_mode == 'training_logs':
        st.subheader("Training Logs & History")
        
        # Display sensor data visualization as FIRST plot
        if st.session_state.sensor_data_plot is not None:
            with st.expander("📊 Input Sensor Data Visualization", expanded=True):
                st.pyplot(st.session_state.sensor_data_plot)
                st.markdown("---")
        
        agents_with_logs = [agent for agent in ['REINFORCE', 'PPO', 'REINFORCE_AM'] 
                            if st.session_state.training_log[agent]]
        
        # Display comparison plot if multiple agents are trained
        if len(agents_with_logs) >= 2:
            st.markdown("### 📊 Multi-Agent Comparison")
            
            W, H = 16, 7
            FONTSIZE_SUPER = 14
            FONTSIZE_TITLE = 11
            FONTSIZE_LABEL = 10
            FONTSIZE_TICK = 9
            BACKGROUND_COLOR = '#fafafa'
            
            metrics_config = [
                {'data': 'rewards', 'title': 'Learning Curve (Smoothed Average Reward)', 'ylabel': 'Average Reward'},
                {'data': 'violations', 'title': 'Threshold Violations (Smoothed)', 'ylabel': 'Violation Rate'},
                {'data': 'replacements', 'title': 'Replacements per Episode (Smoothed)', 'ylabel': 'Replacement Rate'},
                {'data': 'margins', 'title': 'Wear Margin at Replacement (Smoothed)', 'ylabel': 'Wear Margin'}
            ]
            
            fig, axs = plt.subplots(2, 2, figsize=(W, H))
            fig.suptitle('Multi-Agent Performance Comparison', fontsize=FONTSIZE_SUPER, color='#2C3E50')
            
            colors = {'REINFORCE': '#1f77b4', 'PPO': '#ff7f0e', 'REINFORCE_AM': '#2ca02c'}
            
            for idx, config in enumerate(metrics_config):
                ax = axs[idx // 2, idx % 2]
                data_key = config['data']
                
                for agent_name in agents_with_logs:
                    if st.session_state.metrics[agent_name] is not None:
                        metrics_dict = st.session_state.metrics[agent_name]
                        data = metrics_dict.get(data_key, [])
                        
                        if len(data) > 0:
                            if config['data'] == 'margins':
                                data = [m for m in data if not np.isnan(m)]
                            
                            if len(data) > 1:
                                smooth_data = pd.Series(data).rolling(window=10, min_periods=1).mean().to_numpy()
                            else:
                                smooth_data = np.array(data)
                            
                            ax.plot(smooth_data, label=agent_name, color=colors.get(agent_name, 'blue'), linewidth=2, alpha=0.7)
                
                ax.set_title(config['title'], fontsize=FONTSIZE_TITLE)
                ax.set_xlabel('Episode', fontsize=FONTSIZE_LABEL)
                ax.set_ylabel(config['ylabel'], fontsize=FONTSIZE_LABEL)
                ax.tick_params(labelsize=FONTSIZE_TICK)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_facecolor(BACKGROUND_COLOR)
                ax.legend(loc='best', fontsize=9)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            st.pyplot(fig)
            
            # Save button for comparison plot
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("💾 Save Comparison", key="save_comparison"):
                    date_str = pd.Timestamp.now().strftime('%d%m')
                    filename = f"Comparison_{date_str}.png"
                    os.makedirs('saved_plots', exist_ok=True)
                    filepath = os.path.join('saved_plots', filename)
                    fig.savefig(filepath, dpi=150, bbox_inches='tight')
                    st.success(f"✅ Comparison plot saved as: {filename}")
            
            st.markdown("---")
        
        if agents_with_logs:
            for agent_name in agents_with_logs:
                logs = st.session_state.training_log[agent_name]
                
                with st.expander(f"📊 {agent_name} Training Sessions ({len(logs)} session(s))"):
                    for i, log_entry in enumerate(logs):
                        st.write(f"**Session {i+1}** - {log_entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} ({log_entry['episodes']} episodes)")
                        
                        # Display metrics for this session
                        metrics_dict = log_entry['metrics']
                        
                        # Create visualization for this session
                        fig = plot_training_live(
                            metrics=metrics_dict,
                            episode=log_entry['episodes'],
                            total_episodes=log_entry['episodes'],
                            agent_name=agent_name,
                            data_file_name=log_entry.get('data_file_name', 'Unknown'),
                            window=10
                        )
                        st.pyplot(fig)
                        
                        # Save button
                        col1, col2, col3 = st.columns([1, 1, 2])
                        with col1:
                            if st.button(f"💾 Save Plot", key=f"save_{agent_name}_{i}"):
                                # Generate filename: <agent-name>_<ddmm>.png
                                date_str = log_entry['timestamp'].strftime('%d%m')
                                filename = f"{agent_name}_{date_str}.png"
                                
                                # Save figure
                                os.makedirs('saved_plots', exist_ok=True)
                                filepath = os.path.join('saved_plots', filename)
                                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                                st.success(f"✅ Plot saved as: {filename}")
                        
                        st.markdown("---")
        else:
            st.info("No training logs available yet. Train an agent to generate logs.")


    # --- Evaluation View: All Models (Matched to Dataset) ---
    elif st.session_state.view_mode == 'evaluation_all':
        # st.subheader("Evaluation Results: All Models (Dataset Matched)")
        
        if eval_file:
            st.toast(f"Evaluating on: {eval_file.name}", icon="🚀", duration=2)
            
            # Save uploaded file to a temporary location
            try:
                os.makedirs("data", exist_ok=True)
                temp_path = os.path.join("data", f"temp_{eval_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(eval_file.getbuffer())
                    
                with st.spinner("Running evaluation on best matched models..."):
                    # Dynamically find latest models before evaluation (matched to dataset name)
                    latest_ppo = Config.get_latest_model('PPO', temp_path)
                    latest_rf = Config.get_latest_model('REINFORCE', temp_path)
                    latest_rf_am = Config.get_latest_model('REINFORCE_AM', temp_path)
                    latest_HSD = Config.get_latest_model('HSD_Basic', temp_path)
                    latest_HSD_AM = Config.get_latest_model('HSD_AM', temp_path)
                    
                    import io
                    from contextlib import redirect_stdout
                    
                    f = io.StringIO()
                    with redirect_stdout(f):
                        results_df = evaluate_all_models(
                            test_file=temp_path,
                            wear_threshold=Config.WEAR_THRESHOLD,
                            ppo_model_file=latest_ppo if latest_ppo else None,
                            reinforce_model_file=latest_rf if latest_rf else None,
                            reinforce_am_model_file=latest_rf_am if latest_rf_am else None,
                            HSD_Basic_model_file=latest_HSD if latest_HSD else None,
                            HSD_AM_model_file=latest_HSD_AM if latest_HSD_AM else None
                        )
                    captured_output = f.getvalue()
                
                # Update session state with all metrics
                st.session_state.evaluation_results = results_df.to_dict('records')
                
                # Display metrics table
                st.markdown("### Model Comparison")
                display_evaluation_table()
                
                # Display detailed logs
                with st.expander("📄 View Detailed Evaluation Logs", expanded=False):
                    st.text(captured_output)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                st.success("✅ Models matching dataset evaluated successfully!")
                    
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please upload an evaluation dataset in the sidebar to proceed.")

    # --- NEW Evaluation View: N latest Models (Global) ---
    elif st.session_state.view_mode == 'evaluation_saved':
        # st.subheader("Evaluation Results: N latest Saved Models")
        
        if eval_file:
            st.toast(f"Evaluating on: {eval_file.name}", icon="🚀", duration=2)
            
            # Save uploaded file to a temporary location
            try:
                os.makedirs("data", exist_ok=True)
                temp_path = os.path.join("data", f"temp_{eval_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(eval_file.getbuffer())
                    
                with st.spinner("Finding and evaluating the N latest models..."):
                    # Find absolute latest models (ignore dataset name)
                    latest_ppo = Config.get_latest_model('PPO')
                    latest_rf = Config.get_latest_model('REINFORCE')
                    latest_rf_am = Config.get_latest_model('REINFORCE_AM')
                    latest_HSD = Config.get_latest_model('HSD_Basic')
                    latest_HSD_AM = Config.get_latest_model('HSD_AM')
                    
                    import io
                    from contextlib import redirect_stdout
                    
                    f = io.StringIO()
                    with redirect_stdout(f):
                        results_df = evaluate_all_models(
                            test_file=temp_path,
                            wear_threshold=Config.WEAR_THRESHOLD,
                            ppo_model_file=latest_ppo,
                            reinforce_model_file=latest_rf,
                            reinforce_am_model_file=latest_rf_am,
                            HSD_Basic_model_file=latest_HSD,
                            HSD_AM_model_file=latest_HSD_AM
                        )
                    captured_output = f.getvalue()
                
                # Update session state with all metrics
                st.session_state.evaluation_results = results_df.to_dict('records')
                
                # Display metrics table
                st.markdown("### Model Comparison (Global Latest)")
                display_evaluation_table()
                
                # Display detailed logs
                with st.expander("📄 View Detailed Evaluation Logs", expanded=False):
                    st.text(captured_output)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                st.success("✅ Most recent saved models evaluated successfully!")
                    
            except Exception as e:
                st.error(f"Error during global evaluation: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please upload an evaluation dataset in the sidebar to proceed.")
