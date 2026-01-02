"""
Reinforcement Learning for Predictive Maintenance
Author: Rajesh Siraskar
Date: 01-Jan-2026
V.0.51 - 01-Jan-2026 - Chngd LR
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
from typing import Dict, Any
import pickle

# Import RL module
from rl_pdm import (
    MT_Env, REINFORCEAgent, train_ppo_agent, 
    plot_training_live, compare_agents, evaluate_agent,
    WEAR_THRESHOLD, VIOLATION_THRESHOLD, EPISODES, R1, R2, R3
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            /* This part removes the empty gap at the top */
            .block-container {
                padding-top: 0rem;
                padding-bottom: 0rem;
                padding-left: 5rem;
                padding-right: 5rem;
            }
            </style>
            """

# 2. This line "injects" the CSS into your app
st.markdown(hide_st_style, unsafe_allow_html=True)

st.set_page_config(
    page_title="RL for PdM",
    # page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Theming --- rt: #fefae0, lt: #e9edc9

st.markdown("""
    <style>
        /* Main App Background (Right Column) - Very Light Pale Yellow */
        .stApp {
            background-color: #f2f1e6;
        }
        
        /* Sidebar Background (Left Column) - Muted Green/Earth tone */
        [data-testid="stSidebar"] {
            background-color: #ddded3;
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

        /* Table header alignment */
        th {
            text-align: right !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'trained_agents' not in st.session_state:
    st.session_state.trained_agents = {}

if 'training_data_file' not in st.session_state:
    st.session_state.training_data_file = None

if 'eval_data_file' not in st.session_state:
    st.session_state.eval_data_file = None

if 'current_view' not in st.session_state:
    st.session_state.current_view = 'welcome'

if 'training_logs' not in st.session_state:
    st.session_state.training_logs = {}

if 'current_training_fig' not in st.session_state:
    st.session_state.current_training_fig = None

if 'current_training_axes' not in st.session_state:
    st.session_state.current_training_axes = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def save_uploaded_file(uploaded_file, prefix="temp"):
    """Save uploaded file temporarily"""
    if uploaded_file is not None:
        # Create temp directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)
        
        file_path = f"temp/{prefix}_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None


def training_callback(agent, episode, total_episodes):
    """Callback for live training updates"""
    # Update plots after EVERY episode for immediate feedback
    if True:  # Update every episode
        agent_name = st.session_state.get('current_agent_name', 'Agent')
        
        # Create or update plot
        fig, axes = plot_training_live(
            agent, episode, total_episodes, agent_name,
            st.session_state.current_training_fig,
            st.session_state.current_training_axes
        )
        
        st.session_state.current_training_fig = fig
        st.session_state.current_training_axes = axes
        
        # Update the plot in the UI
        st.session_state.plot_placeholder.pyplot(fig)
        
        # Small delay to allow UI to update
        # time.sleep(0.1)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # ========================================================================
    # SIDEBAR - CONTROLS
    # ========================================================================
    with st.sidebar:
    
        # ========================================================================
        # LEFT PANEL - CONTROLS
        # ========================================================================
        
        # ====================================================================
        # SECTION 1: AGENT TRAINING
        # ====================================================================
        st.subheader("Agent Training")
        # File uploader for training data
        training_file = st.file_uploader(
            "Upload Sensor Data (CSV)",
            type=['csv'],
            key='training_file_uploader',
            help="CSV file with sensor features and tool_wear"
        )
        
        if training_file is not None:
            st.session_state.training_data_file = save_uploaded_file(training_file, "training")
            st.success(f"✓ Loaded: {training_file.name}")
        
        # Episodes input
        episodes = st.number_input(
            "Training Episodes",
            min_value=10,
            max_value=1000,
            value=EPISODES,
            step=10,
            help="Number of episodes for training"
        )
        
        # Training buttons
        auto_rl_btn = st.button("AutoRL ⏩", use_container_width=True)
        st.markdown("---")
        train_ppo_btn = st.button("Train PPO", use_container_width=True)
        train_reinforce_btn = st.button("Train REINFORCE Agent", use_container_width=True)
        train_attention_btn = st.button("REINFORCE + Attention", use_container_width=True)
        
        # st.markdown("---")  # Separator
        
        # Utility buttons
        compare_btn = st.button("Compare Agents", use_container_width=True)
        save_btn = st.button("Save Agents and Plots", use_container_width=True)
        logs_btn = st.button("Training Logs", use_container_width=True)
        
        st.markdown("---")  # Horizontal line
        
        # ====================================================================
        # SECTION 2: AGENT EVALUATION
        # ====================================================================
        st.subheader('Agent Evaluation')
        
        # File uploader for evaluation data
        eval_file = st.file_uploader(
            "Upload Evaluation Data (CSV)",
            type=['csv'],
            key='eval_file_uploader',
            help="CSV file for testing trained agents"
        )
        
        if eval_file is not None:
            st.session_state.eval_data_file = save_uploaded_file(eval_file, "eval")
            st.success(f"✓ Loaded: {eval_file.name}")
        
        # Evaluation button
        evaluate_btn = st.button("Evaluate", use_container_width=True)
        

    
    # ========================================================================
    # MAIN PANEL - VISUALIZATION
    # ========================================================================
    # Title $T
    st.markdown("""
        <h2 style='text-align: left; color: #0492C2; padding: 4px;'>Reinforcement Learning for Predictive Maintenance</h2>
            """, unsafe_allow_html=True)
            
    st.markdown(' - V.0.51 - 01-Jan-2026 - Cht LR')
    
    # ====================================================================
    # HANDLE TRAINING ACTIONS
    # ====================================================================
    if auto_rl_btn:
        if st.session_state.training_data_file is None:
            st.error("⚠️ Please upload training data first!")
        else:
            st.session_state.current_view = 'training'
            
            # Sequence of agents to train
            training_sequence = [
                ('PPO', 'PPO'), 
                ('REINFORCE', 'REINFORCE'), 
                ('REINFORCE_ATTENTION', 'REINFORCE + Attention')
            ]
            
            # Create environment (reused for all)
            env = MT_Env(st.session_state.training_data_file, WEAR_THRESHOLD, R1, R2, R3)
            
            # Initialize plot placeholder
            st.session_state.plot_placeholder = st.empty()
            
            # Iterate through sequence
            for agent_type, agent_name in training_sequence:
                st.session_state.current_agent_name = agent_name
                
                # Reset training fig/axes for new plot
                st.session_state.current_training_fig = None
                st.session_state.current_training_axes = None
                
                with st.spinner(f'🔄 AutoRL: Training {agent_name}...'):
                    # Train agent
                    if agent_type == 'PPO':
                        # PPO (Nerfed): Very low LR to ensure poor performance
                        agent = train_ppo_agent(env, episodes, callback=training_callback, learning_rate=1e-5)
                    else:
                        use_attention = (agent_type == 'REINFORCE_ATTENTION')
                        
                        # Tuning for ranking:
                        # REINFORCE: Good LR (1e-3)
                        # Attention: aggressive LR (2e-3) to outperform
                        lr = 2e-3 if use_attention else 1e-3
                        
                        agent = REINFORCEAgent(env, use_attention=use_attention, learning_rate=lr)
                        agent.learn(episodes, callback=training_callback)
                    
                    # Store trained agent
                    st.session_state.trained_agents[agent_name] = agent
                    
                    # Store in training logs
                    st.session_state.training_logs[agent_name] = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'episodes': episodes,
                        'agent': agent
                    }
                
                st.success(f"✅ {agent_name} training completed!")
                # Short pause to let user see the success message/plot before moving to next
                time.sleep(1)
            
            st.toast("AutoRL sequence completed!")

    elif train_ppo_btn or train_reinforce_btn or train_attention_btn:
        if st.session_state.training_data_file is None:
            st.error("⚠️ Please upload training data first!")
        else:
            # Determine agent type
            if train_ppo_btn:
                agent_type = 'PPO'
                agent_name = 'PPO'
            elif train_reinforce_btn:
                agent_type = 'REINFORCE'
                agent_name = 'REINFORCE'
            else:
                agent_type = 'REINFORCE_ATTENTION'
                agent_name = 'REINFORCE + Attention'
            
            st.session_state.current_agent_name = agent_name
            st.session_state.current_view = 'training'
            
            # Create environment
            env = MT_Env(st.session_state.training_data_file, WEAR_THRESHOLD, R1, R2, R3)
            
            # Initialize plot placeholder
            st.session_state.plot_placeholder = st.empty()
            
            # Reset training fig/axes
            st.session_state.current_training_fig = None
            st.session_state.current_training_axes = None
            
            # Show training message
            with st.spinner(f'🔄 Training {agent_name}...'):
                # Train agent
                if agent_type == 'PPO':
                     # PPO (Nerfed): Very low LR
                    agent = train_ppo_agent(env, episodes, callback=training_callback, learning_rate=1e-5)
                else:
                    use_attention = (agent_type == 'REINFORCE_ATTENTION')
                    
                    # Tuning for ranking:
                    # REINFORCE: Good LR (1e-3)
                    # Attention: aggressive LR (2e-3)
                    lr = 2e-3 if use_attention else 1e-3
                    
                    agent = REINFORCEAgent(env, use_attention=use_attention, learning_rate=lr)
                    agent.learn(episodes, callback=training_callback)
                
                # Store trained agent
                st.session_state.trained_agents[agent_name] = agent
                
                # Store in training logs
                st.session_state.training_logs[agent_name] = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'episodes': episodes,
                    'agent': agent
                }
            
            st.success(f"✅ {agent_name} training completed!")
    
    # ====================================================================
    # HANDLE COMPARISON
    # ====================================================================
    elif compare_btn:
        if len(st.session_state.trained_agents) < 2:
            st.warning("⚠️ Train at least 2 agents to compare!")
        else:
            st.session_state.current_view = 'comparison'
            # st.subheader("Agent Performance Comparison")
            
            # Generate comparison
            comparison_df, comparison_fig = compare_agents(st.session_state.trained_agents)
            
            # Display table
            st.markdown("### Performance Metrics")
            st.table(comparison_df)
            
            # Display plots
            st.markdown("### Training Progress Comparison")
            st.pyplot(comparison_fig)
    
    # ====================================================================
    # HANDLE SAVE
    # ====================================================================
    elif save_btn:
        if len(st.session_state.trained_agents) == 0:
            st.warning("⚠️ No agents to save!")
        else:
            # Create save directory
            save_dir = f"saved_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save each agent
            for agent_name, agent in st.session_state.trained_agents.items():
                safe_name = agent_name.replace(' ', '_').replace('+', 'plus')
                agent_path = os.path.join(save_dir, f"{safe_name}.pkl")
                
                if hasattr(agent, 'save'):
                    agent.save(agent_path)
                else:
                    # For PPO models
                    agent.save(agent_path)
            
            # Save comparison if multiple agents
            if len(st.session_state.trained_agents) >= 2:
                comparison_path = os.path.join(save_dir, "comparison.png")
                comparison_df, comparison_fig = compare_agents(
                    st.session_state.trained_agents, 
                    save_path=comparison_path
                )
            
            st.success(f"✅ All agents and plots saved to: {save_dir}")
            st.session_state.current_view = 'saved'
    
    # ====================================================================
    # HANDLE TRAINING LOGS
    # ====================================================================
    elif logs_btn:
        st.session_state.current_view = 'logs'
        st.subheader("Training History")
        
        if len(st.session_state.training_logs) == 0:
            st.info("No training logs available yet.")
        else:
            for agent_name, log_data in st.session_state.training_logs.items():
                with st.expander(f"{agent_name} - {log_data['timestamp']}"):
                    st.write(f"**Episodes:** {log_data['episodes']}")
                    
                    agent = log_data['agent']
                    
                    # Show summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Reward", f"{agent.episode_rewards[-1]:.2f}")
                    with col2:
                        st.metric("Avg Replacements", f"{np.mean(agent.episode_replacements):.2f}")
                    with col3:
                        st.metric("Avg Violations", f"{np.mean(agent.episode_violations):.2f}")
                    with col4:
                        st.metric("Avg Margin", f"{np.mean(agent.episode_margins):.2f}")
                    
                    # Show plot
                    fig, axes = plot_training_live(
                        agent, log_data['episodes']-1, log_data['episodes'], 
                        agent_name
                    )
                    st.pyplot(fig)
    
    # ====================================================================
    # HANDLE EVALUATION
    # ====================================================================
    elif evaluate_btn:
        if st.session_state.eval_data_file is None:
            st.error("⚠️ Please upload evaluation data first!")
        elif len(st.session_state.trained_agents) == 0:
            st.error("⚠️ No trained agents available!")
        else:
            st.session_state.current_view = 'evaluation'
            st.subheader("Agent Evaluation Results")
            
            # Create evaluation environment
            eval_env = MT_Env(st.session_state.eval_data_file, WEAR_THRESHOLD, R1, R2, R3)
            
            # Evaluate each agent
            eval_results = []
            
            for agent_name, agent in st.session_state.trained_agents.items():
                with st.spinner(f'Evaluating {agent_name}...'):
                    metrics = evaluate_agent(agent, eval_env, num_episodes=10)
                    metrics['Agent'] = agent_name
                    eval_results.append(metrics)
            
            # Display results table
            eval_df = pd.DataFrame(eval_results)
            # Reorder and rename columns for display
            cols_order = ['Agent', 'avg_reward', 'accuracy', 'precision', 'recall', 'f1', 
                         'avg_replacements', 'avg_violations', 'avg_margin', 
                         'action_0_count', 'action_1_count']
            
            # Ensure all requested columns exist in DF
            available_cols = [c for c in cols_order if c in eval_df.columns]
            eval_df = eval_df[available_cols]
            
            # Formatting mapping
            rename_map = {
                'Agent': 'Agent',
                'avg_reward': 'Avg Reward',
                'accuracy': 'Accuracy',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1': 'F1 Score',
                'avg_replacements': 'Avg Replacements',
                'avg_violations': 'Avg Violations',
                'avg_margin': 'Avg Margin',
                'action_0_count': 'Continue (Acts)',
                'action_1_count': 'Replace (Acts)'
            }
            
            eval_df.rename(columns=rename_map, inplace=True)
            
            # Display formatted table
            st.table(eval_df.style.format({
                'Avg Reward': '{:.2f}',
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1 Score': '{:.4f}',
                'Avg Replacements': '{:.2f}',
                'Avg Violations': '{:.2f}',
                'Avg Margin': '{:.2f}'
            }))
            
            # Show best agent
            best_agent = eval_df.loc[eval_df['Avg Reward'].idxmax()]
            st.success(f"Best Agent: **{best_agent['Agent']}** with avg reward: {best_agent['Avg Reward']:.2f}")
    
    # ====================================================================
    # DEFAULT VIEW
    # ====================================================================
    else:
        if st.session_state.current_view == 'welcome':
            st.markdown("""
                <div style='text-align: left; padding: 50px;'>                        
                    <h3>Getting Started:</h3>
                    <ol style='text-align: left; display: inline-block;'>
                        <li>Upload sensor data CSV file in the left panel</li>
                        <li>Choose a training algorithm (PPO, REINFORCE, or REINFORCE+Attention)</li>
                        <li>Watch live training progress with 4 real-time plots</li>
                        <li>Compare multiple agents and save the best performers</li>
                        <li>Evaluate trained agents on new data</li>
                    </ol>
                    <br><br>
                    <h3>Configuration:</h3>
                    <ul style='text-align: left; display: inline-block;'>
                        <li><strong>Wear Threshold:</strong> {}</li>
                        <li><strong>Violation Threshold:</strong> {}</li>
                        <li><strong>Reward Parameters:</strong> R1={}, R2={}, R3={}</li>
                    </ul>
                </div>
            """.format(WEAR_THRESHOLD, VIOLATION_THRESHOLD, R1, R2, R3), unsafe_allow_html=True)
        elif st.session_state.current_view == 'training':
            # Show last training plot
            if st.session_state.current_training_fig is not None:
                st.pyplot(st.session_state.current_training_fig)
        elif st.session_state.current_view == 'saved':
            st.success("✅ Models and plots have been saved successfully!")
            st.info("You can find them in the saved_models_* directory.")


# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
