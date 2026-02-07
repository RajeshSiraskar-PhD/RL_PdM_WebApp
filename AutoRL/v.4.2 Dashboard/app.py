# ---------------------------------------------------------------------------------------
# AutoRL: Auto-train Predictive Maintenance Agents 
# Author: Rajesh Siraskar
# Web-UX code
# V.1.0: 06-Feb-2026: First commit
# ---------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os
import rl_pdm # Import our backend
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="AutoRL - Predictive Maintenance", layout="wide")

# --- CUSTOM CSS (Dark Theme) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #1E1E2E;
        color: #F5E0B5;
    }
    
    /* Sidebar/Columns */
    section[data-testid="stSidebar"] {
        background-color: #2D2D44;
    }
    
    /* Texts title: #f5a55b labels #95b8d1 */
    h1 {
        color: #32a6a8 !important;
    }

    h2, h3, p, label {
        color: #2e9294 !important;
    }
    
    /* Inputs */
    .stTextInput>div>div>input {
        background-color: #2D2D44;
        color: #F5E0B5;
    }
    .stNumberInput>div>div>input {
        background-color: #2D2D44;
        color: #F5E0B5;
    }
    .stTextArea>div>div>textarea {
        background-color: #2D2D44;
        color: #F5E0B5;
    }
    
    /* Buttons #2196F3 hover: #1976D2 ##313e85 #1111d9*/
    .stButton>button {
        background-color: #0660ba; 
        color: #FFFFFF !important;
        border-radius: 5px;
        border: none;
        transition: background-color 0.3s ease;
        font-weight: bold !important;
        font-size: 14px !important;
    }
    
    .stButton>button p, .stButton>button span {
        color: #FFFFFF !important;
    }
    .stButton>button:hover {
        background-color: #1c04b8;
        color: #b0e7e8 !important;
    }
    
    /* Metrics/Plots Background */
    .plot-container {
        background-color: #2D2D44;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def save_uploaded_file(uploaded_file):
    try:
        with open("temp_sensor_data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        return "temp_sensor_data.csv"
    except Exception as e:
        return None

def smooth_data(data, window_size):
    if len(data) < window_size:
        return data
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().tolist()

def plot_4_panel(metrics, title, height=600, data_filename=None, t_ss=None, sigma_ss=None):
    """
    Helper to generate the 4-panel plot.
    
    metrics: dict with 'rewards', 'margins', 'violations', 'replacements'
    title: plot title
    height: plot height
    data_filename: optional filename to add to title
    t_ss: episode where steady state begins (for wear margin plot)
    sigma_ss: standard deviation in steady state (for wear margin plot)
    """
    # Smooth data
    # Window: 10 eps or 10% of len
    n_points = len(metrics['rewards'])
    window = max(10, int(n_points * 0.1))
    
    s_rewards = smooth_data(metrics['rewards'], window)
    s_margins = smooth_data(metrics['margins'], window)
    # Violations and Replacements are binary 0/1 usually, smoothing gives 'rate'
    s_violations = smooth_data(metrics['violations'], window)
    s_replacements = smooth_data(metrics['replacements'], window)
    
    # Create Grid Plot
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Avg Reward", "Wear Margin", "Violation Rate", "Replacement Rate"))
    
    # Add Traces (Legend removed)
    # Colors for each metric
    c1, c2, c3, c4 = '#636EFA', '#EF553B', '#00CC96', '#AB63FA'
    
    # Reward
    fig.add_trace(go.Scatter(y=metrics['rewards'], line=dict(color=c1, width=1), opacity=0.3, showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(y=s_rewards, line=dict(color=c1, width=3), name='Reward', showlegend=False), row=1, col=1)
    
    # Margin
    fig.add_trace(go.Scatter(y=metrics['margins'], line=dict(color=c2, width=1), opacity=0.3, showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(y=s_margins, line=dict(color=c2, width=3), name='Margin', showlegend=False), row=1, col=2)
    
    # Add steady-state visualization to Wear Margin plot (row=1, col=2)
    if t_ss is not None and t_ss > 0:
        t_ss_val = min(t_ss, len(metrics['margins']) - 1)
        
        # Add vertical line marking steady state start
        fig.add_vline(
            x=t_ss_val,
            line_dash="dash",
            line_color="#999",
            annotation_text=f"T_ss={int(t_ss_val)}",
            annotation_position="top",
            row=1,
            col=2
        )
        
        # Add shaded region for steady state
        fig.add_vrect(
            x0=t_ss_val,
            x1=len(metrics['margins']),
            fillcolor="#00CC96",
            opacity=0.15,
            layer="below",
            line_width=0,
            row=1,
            col=2
        )
        
        # Add band showing ±sigma_ss around mean in steady state
        if sigma_ss is not None and sigma_ss >= 0:
            steady_margins = metrics['margins'][int(t_ss_val):]
            if len(steady_margins) > 0:
                mean_ss = np.mean(steady_margins)
                upper_band = [mean_ss + sigma_ss] * len(steady_margins)
                lower_band = [mean_ss - sigma_ss] * len(steady_margins)
                
                # Create x values for steady state region
                ss_x = list(range(int(t_ss_val), len(metrics['margins'])))
                
                # Add upper and lower bands (as traces for legend)
                fig.add_trace(
                    go.Scatter(
                        x=ss_x,
                        y=upper_band,
                        line=dict(color='rgba(0,204,150,0)', width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1,
                    col=2
                )
                fig.add_trace(
                    go.Scatter(
                        x=ss_x,
                        y=lower_band,
                        fill='tonexty',
                        line=dict(color='rgba(0,204,150,0)', width=0),
                        fillcolor='rgba(0,204,150,0.2)',
                        name=f'±σ_ss ({sigma_ss:.2f})',
                        showlegend=True,
                        hoverinfo='skip'
                    ),
                    row=1,
                    col=2
                )
    
    # Violations
    fig.add_trace(go.Scatter(y=metrics['violations'], line=dict(color=c3, width=1), opacity=0.3, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(y=s_violations, line=dict(color=c3, width=3), name='Violations', showlegend=False), row=2, col=1)
    
    # Replacements
    fig.add_trace(go.Scatter(y=metrics['replacements'], line=dict(color=c4, width=1), opacity=0.3, showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(y=s_replacements, line=dict(color=c4, width=3), name='Replacements', showlegend=False), row=2, col=2)
    
    # Build final title with optional data filename
    final_title = title
    if data_filename:
        final_title = f"{title} | Data: {data_filename}"
    
    fig.update_layout(title_text=final_title, height=height, template="plotly_white")
    return fig

def plot_evaluation_results(eval_results, model_name):
    """
    Generate dual-axis plot for evaluation:
    - Left axis: Tool Wear (blue line)
    - Right axis: Actions (red spikes for replacements, blue for model overrides)
    - Dotted line: Wear Threshold
    - Shaded regions: IAR (Ideal Action Region) if present
    """
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    timesteps = eval_results['timesteps']
    tool_wear = eval_results['tool_wear']
    actions = eval_results['actions']
    wear_threshold = eval_results['wear_threshold']
    
    # Add tool wear as blue line on primary y-axis
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=tool_wear,
            name="Tool Wear",
            line=dict(color='#636EFA', width=3),
            mode='lines'
        ),
        secondary_y=False
    )
    
    # Add wear threshold as dotted line on primary y-axis
    fig.add_trace(
        go.Scatter(
            x=timesteps,
            y=[wear_threshold] * len(timesteps),
            name="Wear Threshold",
            line=dict(color='gray', width=2, dash='dot'),
            mode='lines',
            showlegend=True
        ),
        secondary_y=False
    )
    
    # Always attempt to plot IAR lines; warn if missing
    IAR_lower = eval_results.get('IAR_lower', None)
    IAR_upper = eval_results.get('IAR_upper', None)
    if IAR_lower is not None and IAR_upper is not None:
        # Add lower and upper bounds
        fig.add_hline(
            y=IAR_lower,
            line_dash="dash",
            line_color="lightgreen",
            opacity=0.5,
            annotation_text="IAR Lower",
            annotation_position="right",
            secondary_y=False
        )
        fig.add_hline(
            y=IAR_upper,
            line_dash="dash",
            line_color="lightgreen",
            opacity=0.5,
            annotation_text="IAR Upper",
            annotation_position="right",
            secondary_y=False
        )
    else:
        import warnings
        warnings.warn("IAR_lower and/or IAR_upper missing from eval_results. IAR lines will not be shown.")
    
    # Separate replacements into normal (red) and overridden (blue)
    model_override = eval_results.get('model_override', False)
    override_timestep = eval_results.get('override_timestep', None)
    
    # Normal replacements (red)
    replacement_timesteps = [t for t, a in zip(timesteps, actions) if a == 0]
    
    # If model override, separate the override point from normal replacements
    if model_override and override_timestep is not None:
        override_replacements = [t for t in replacement_timesteps if t == override_timestep]
        normal_replacements = [t for t in replacement_timesteps if t != override_timestep]
    else:
        normal_replacements = replacement_timesteps
        override_replacements = []
    
    # Add normal replacements as red markers (positioned at tool_wear - 5)
    if normal_replacements:
        normal_wear_values = [tool_wear[timesteps.index(t)] - 5 for t in normal_replacements]
        fig.add_trace(
            go.Scatter(
                x=normal_replacements,
                y=normal_wear_values,
                name="Tool Replacement",
                mode='markers',
                marker=dict(
                    size=12,
                    color='#EF553B',  # Red
                    symbol='diamond',
                    opacity=0.5
                ),
                showlegend=True
            ),
            secondary_y=False
        )
    
    # Add overridden replacements as blue markers (positioned at tool_wear - 5)
    if override_replacements:
        override_wear_values = [tool_wear[timesteps.index(t)] - 5 for t in override_replacements]
        fig.add_trace(
            go.Scatter(
                x=override_replacements,
                y=override_wear_values,
                name="Tool Replacement*",
                mode='markers',
                marker=dict(
                    size=12,
                    color="#EF553B",  # Use 00A3CC Blue-ish for star replacement
                    symbol='star',
                    opacity=0.9
                ),
                showlegend=True
            ),
            secondary_y=False
        )
    
    # Update layout
    fig.update_xaxes(title_text="Timestep")
    fig.update_yaxes(title_text="Tool Wear", secondary_y=False)
    fig.update_yaxes(title_text="Action", secondary_y=True, range=[0.0, 1.5], showticklabels=False, ticks="", showgrid=False)
    
    # Add faint dotted vertical lines at each replacement marker
    
    # Normal replacements (red) - add vertical lines
    for x in normal_replacements:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            line=dict(
                color="rgba(239,85,59,0.1)",
                width=0.3,
                # dash="dot"
            ),
            layer="below"
        )
    
    # Override replacements (blue/green) - add vertical lines
    for x in override_replacements:
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=x,
            x1=x,
            y0=0,
            y1=1,
            line=dict(
                color="rgba(0,204,150,0.2)",
                width=1,
                dash="dot"
            ),
            layer="below"
        )
    
    fig.update_layout(
        title=f"Model Evaluation: {model_name}",
        height=500,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

# --- LAYOUT --- # $$$
st.title(f'AutoRL: Auto-train Predictive Maintenance Agents') 
st.markdown(' - V.1.1: Add report')

col1, col2 = st.columns([1.7, 8.3])

# --- LEFT PANEL: AGENT TRAINING & EVALUATION ---
with col1:
    # Tabs for different operations
    left_tabs = st.tabs(["Configuration", "Training", "Evaluate"])
    
    # Configuration Tab
    with left_tabs[0]:
        st.subheader("Configuration")
        
        # MDP Type Selection
        st.markdown("**MDP Settings**")
        finite_horizon = st.checkbox(
            "Finite-Horizon MDPs", 
            value=True,
            help="Fixed-length episodes (stops after N episodes). Uncheck for variable-length episodes."
        )
        
        # Update global setting in rl_pdm
        rl_pdm.FIXED_X_AXIS_LENGTH = finite_horizon
        
        # Wear Threshold Setting
        wear_threshold = st.number_input(
            "Wear Threshold",
            min_value=1,
            value=285,
            step=1,
            help="Tool wear threshold for maintenance decisions"
        )
        
        # Store in session state for use during training and evaluation
        st.session_state.wear_threshold = wear_threshold
        
        st.markdown("---")
        
        # Algorithm Selection
        st.markdown("**Algorithm Selection**")
        # all_algorithms = ['PPO', 'A2C', 'DQN', 'REINFORCE']
        all_algorithms = ['PPO', 'A2C', 'REINFORCE']
        # all_algorithms = ['PPO']
        selected_algorithms = st.multiselect(
            "Training Algorithms",
            options=all_algorithms,
            default=all_algorithms,
            help="Select which algorithms to train during AutoRL"
        )
        
        # Store in session state for use during training
        st.session_state.selected_algorithms = selected_algorithms if selected_algorithms else all_algorithms
        
        # st.markdown("---")
        
        # Attention Mechanisms Selection
        st.markdown("**Attention Mechanisms**")
        st.caption("Select attention variants to apply after AutoRL training")
        
        attention_temporal = st.checkbox(
            "Temporal Encoding",
            value=True,
            help="Captures progressive tool wear over time using temporal embeddings"
        )
        
        attention_multihead = st.checkbox(
            "Multi-Head Attention",
            value=True,
            help="Uses 8 parallel attention heads to capture diverse feature relationships"
        )
        
        attention_selfattn = st.checkbox(
            "Self-Attention",
            value=True,
            help="Transformer-style self-attention with layer normalization"
        )
        
        attention_hybrid = st.checkbox(
            "Hybrid (Temporal + Multi-Head)",
            value=True,
            help="Combines temporal encoding with multi-head attention for best performance"
        )
        
        attention_nadaraya = st.checkbox(
            "Nadaraya-Watson",
            value=True,
            help="Non-parametric kernel-based attention for smooth local feature aggregation"
        )
        
        # Store attention options in session state
        st.session_state.attention_options = {
            'Temporal': attention_temporal,
            'MultiHead': attention_multihead,
            'SelfAttn': attention_selfattn,
            'Hybrid': attention_hybrid,
            'NadarayaWatson': attention_nadaraya
        }
        
        # Count selected attention mechanisms
        selected_attention = [k for k, v in st.session_state.attention_options.items() if v]
        
        # st.markdown("---")
        
        # Display current configuration
        st.markdown("**Current Settings:**")
        attention_summary = ', '.join(selected_attention) if selected_attention else 'None'
        st.info(f"MDP Type: {'Finite-Horizon' if finite_horizon else 'Infinite-Horizon'}\n\n"
                f"Wear Threshold: {wear_threshold}\n\n"
                f"Algorithms: {', '.join(st.session_state.selected_algorithms)}\n\n"
                f"Attention: {attention_summary}")
    
    # Training Tab
    with left_tabs[1]:
        st.subheader("Agent Training")
        
        # File Loader
        uploaded_file = st.file_uploader("Upload Sensor Data (CSV)", type="csv")
        
        # Params
        episodes = st.number_input("Episodes", min_value=1, value=100, step=20)
        
        # Hyperparams Input
        st.subheader("Hyperparameters")
        lr_input = st.text_input("Learning Rates (comma sep)", "0.0005")
        gamma_input = st.text_input("Gammas (comma sep)", "0.99")
        
        # Parse inputs
        try:
            lrs = [float(x.strip()) for x in lr_input.split(",")]
        except:
            lrs = [0.0005]
            
        try:
            gammas = [float(x.strip()) for x in gamma_input.split(",")]
        except:
            gammas = [0.99]
        
        # AutoRL Button
        start_training = st.button("AutoRL: Base agents")
        st.subheader("Attention Awareness")
        train_attention_only = st.button("Train Attention Models")
        apply_attention = st.button("AutoRL: Apply Attention")
        
        # st.markdown("---")
        st.subheader("Model comparison")
        compare_btn = st.button("Compare Trained Models")
    
    # Evaluate Tab
    with left_tabs[2]:
        # st.subheader("Results & Evaluation")
        # st.markdown("---")
        st.subheader("Evaluate Model")
        
        # Get available models
        available_models = rl_pdm.get_available_models()
        
        if available_models:
            selected_model = st.selectbox("Select Model", available_models)
            
            # File uploader for test data
            test_file = st.file_uploader("Upload Test Data (CSV)", type="csv", key="test_data")
            
            if test_file is not None:
                # Save test file
                test_file_path = save_uploaded_file(test_file)
                
                if st.button("Evaluate Model"):
                    st.info(f"Evaluating {selected_model}...")
                    
                    try:
                        # Run adjusted evaluation
                        model_path = os.path.join("models", selected_model)
                        wear_threshold_val = st.session_state.get('wear_threshold', 285)
                        eval_results = rl_pdm.adjusted_evaluate_model(model_path, test_file_path, wear_threshold=wear_threshold_val)
                        
                        # Store in session state for plotting
                        st.session_state.eval_results = eval_results
                        st.session_state.eval_model_name = selected_model
                        st.session_state.eval_file_name = test_file.name  # Store test file name
                        st.toast("✅ Evaluation complete!", icon="✅")
                        
                    except Exception as e:
                        st.error(f"Error evaluating model: {str(e)}")
        else:
            st.warning("No trained models available. Run 'AutoRL - Auto train' first to generate models.")
        
        
    
# --- RIGHT PANE: MONITORING ---
with col2:
    # Improved sticky tabs CSS (target Streamlit's tab bar more robustly)
    st.markdown("""
    <style>
    /* Make the tab bar sticky at the top of the right pane */
    div[data-testid="stTabs"] > div:first-child {
        position: sticky;
        top: 0;
        z-index: 100;
        background: #1E1E2E;
        border-bottom: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for results if not exists
    if 'training_results' not in st.session_state:
        st.session_state.training_results = []
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None

    right_tabs = st.tabs(["Training Panel", "Model Comparison Panel", "Evaluation Panel"])

    # TRAINING TAB
    with right_tabs[0]:
        # All right-panel content now lives INSIDE the tab!
        plot_placeholder = st.empty()
        logs_placeholder = st.empty()

        # Sensor plot logic (inside tab)
        # Check for uploaded file in UI (re-check state or handle it)
        if 'show_sensor_plot' not in st.session_state:
            st.session_state.show_sensor_plot = True
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None
        if 'sensor_fig' not in st.session_state:
            st.session_state.sensor_fig = None

        if 'uploaded_file' in locals() and uploaded_file is not None:
            # Check if it's a new file
            if st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.show_sensor_plot = True
                st.session_state.last_uploaded_file = uploaded_file.name
                # Extract base filename (without path and .csv extension)
                base_filename = uploaded_file.name.replace('.csv', '').replace('.CSV', '')
                st.session_state.data_filename = base_filename
                # Store the sensor figure for later display in Training Logs
                if 'sensor_fig' not in st.session_state:
                    st.session_state.sensor_fig = None

            path = save_uploaded_file(uploaded_file)
            if path and st.session_state.show_sensor_plot:
                wear_threshold = st.session_state.get('wear_threshold', 285)
                fig_sensor = rl_pdm.plot_sensor_data(path, wear_threshold=wear_threshold)
                if fig_sensor:
                    fig_sensor.update_layout(title_text=f"Sensor Data Overview: {uploaded_file.name}")
                    st.session_state.sensor_fig = fig_sensor  # Store for later
                    # Display sensor plot in collapsible section (collapsed by default)
                    with st.expander("📊 Sensor Data Overview", expanded=False):
                        st.plotly_chart(fig_sensor, use_container_width=True, key="sensor_plot_training")

        # st.subheader("Live Training Plots")
        
        if start_training:
            if uploaded_file is None:
                st.error("Please upload a CSV file first.")
            else:
                # Hide sensor plot when training starts
                st.session_state.show_sensor_plot = False
                
                # Save file
                data_path = save_uploaded_file(uploaded_file)
                
                # Prepare UI containers for live plots
                # We want 4 plots
                # We will use Plotly for nice updates
                
                def ui_callback(combo_name, metrics):
                    filename = st.session_state.get('data_filename', 'Unknown')
                    # Note: T_ss and sigma_ss will be added after training completes
                    fig = plot_4_panel(metrics, f"Training: {combo_name}", data_filename=filename)
                    plot_placeholder.plotly_chart(fig, use_container_width=True)

                # Run Training
                # Update Globals in rl_pdm (Hack, but effective)
                rl_pdm.EPISODES = episodes
                rl_pdm.WEAR_THRESHOLD = st.session_state.get('wear_threshold', 285)
                
                # Generate Task List
                task_list = []
                # Use selected algorithms from Configuration tab
                algo_names = st.session_state.get('selected_algorithms', ['PPO', 'A2C', 'DQN', 'REINFORCE'])
                for algo in algo_names:
                    for lr in lrs:
                        for gm in gammas:
                            task_list.append({'algo': algo, 'lr': lr, 'gamma': gm, 'status': 'Pending'})
                
                # Status Container (moved below plots)
                status_container = st.empty()
                
                def render_status():
                    content = "**Training Queue:**\n\n"
                    for i, t in enumerate(task_list):
                        icon = "⏳"
                        if t['status'] == 'Running': icon = "🔄"
                        elif t['status'] == 'Done': icon = "✅"
                        elif t['status'] == 'Error': icon = "❌"
                        
                        content += f"{icon} **{t['algo']}** (LR={t['lr']}, γ={t['gamma']})\n\n"
                    status_container.markdown(content)
                
                # Initialize status (will show below plots during training)
                results = []
                st.session_state.training_results = [] # Clear previous results
                
                # Loop through tasks
                for i, task in enumerate(task_list):
                    # Update Status
                    task_list[i]['status'] = 'Running'
                    render_status()
                    
                    # Train
                    data_filename = st.session_state.get('data_filename', None)
                    res = rl_pdm.train_single_model(data_path, task['algo'], task['lr'], task['gamma'], ui_callback, data_filename=data_filename)
                    results.append(res)
                    st.session_state.training_results.append(res) # Append incrementally
                    
                    # Update Status
                    if 'error' in res:
                         task_list[i]['status'] = 'Error'
                    else:
                         task_list[i]['status'] = 'Done'
                    render_status()
                
                st.toast("✅ All Training Complete!", icon="✅")
                
                # Check for errors
                valid_results = [r for r in st.session_state.training_results if 'error' not in r]
                errors = [r for r in st.session_state.training_results if 'error' in r]
                
                if errors:
                    st.warning(f"Encountered errors in {len(errors)} runs.")
                    for err in errors:
                        with st.expander(f"Error for {err['Agent']} (LR={err['LR']}, G={err['Gamma']})"):
                            st.error(err['error'])
                            st.code(err.get('traceback', ''))

        # --- PERSISTENT PLOTS (History) ---
        if st.session_state.training_results:
            st.subheader("Training History")
            # We can either show small plots for all, or just the last few.
            # Let's show all in an expander, or a grid.
        
        # Grid layout for plots
        cols = st.columns(2)
        for i, res in enumerate(st.session_state.training_results):
            if 'error' not in res:
                with cols[i % 2]:
                    title = f"{res['Agent']} (Score: {res['Weighted Score']:.3f} | R={res['Avg Reward']:.1f}, LR={res['LR']:.3f}, Gamma={res['Gamma']:.3f})"
                    with st.expander(title, expanded=False):
                        # Re-generate plot with steady state metrics
                        filename = st.session_state.get('data_filename', 'Unknown')
                        t_ss = res.get('T_ss', None)
                        sigma_ss = res.get('Sigma_ss', None)
                        fig = plot_4_panel(res['full_metrics'], title, height=400, data_filename=filename, t_ss=t_ss, sigma_ss=sigma_ss)
                        st.plotly_chart(fig, use_container_width=True, key=f"training_history_{i}_{res['Agent']}_{res['LR']}_{res['Gamma']}")

    # --- TRAIN ATTENTION MODELS ONLY (Independent of AutoRL) ---
    if 'train_attention_only' in locals() and train_attention_only:
        # Check if sensor data is available
        if not os.path.exists("temp_sensor_data.csv"):
            st.error("Please upload sensor data first in the Training tab.")
        else:
            # Check if any attention mechanisms are selected
            selected_attention = st.session_state.get('attention_options', {})
            enabled_attention = [k for k, v in selected_attention.items() if v]
            
            if not enabled_attention:
                st.warning("Please select at least one attention mechanism in the Configuration tab.")
            else:
                st.info(f"🎯 Training {len(enabled_attention)} Attention Mechanism(s) directly on sensor data")
                st.caption(f"Selected: {', '.join(enabled_attention)}")
                
                # Prepare UI Callback
                plot_placeholder = st.empty()
                def ui_callback_att_only(combo_name, metrics):
                    filename = st.session_state.get('data_filename', 'Unknown')
                    fig = plot_4_panel(metrics, f"Training: {combo_name}", data_filename=filename)
                    plot_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Set global parameters
                rl_pdm.EPISODES = episodes 
                rl_pdm.WEAR_THRESHOLD = st.session_state.get('wear_threshold', 285)
                
                # Build attention training tasks
                att_only_tasks = []
                algo_names = st.session_state.get('selected_algorithms', ['PPO', 'A2C', 'REINFORCE'])
                for algo in algo_names:
                    for lr in lrs:
                        for gm in gammas:
                            for att_type in enabled_attention:
                                att_only_tasks.append({
                                    'algo': algo,
                                    'lr': lr,
                                    'gamma': gm,
                                    'att': att_type,
                                    'status': 'Pending'
                                })
                
                status_container_att_only = st.empty()
                def render_att_only_status():
                    content = "**Training Queue:**\n\n"
                    for t in att_only_tasks:
                        icon = "⏳"
                        if t['status'] == 'Running': icon = "🔄"
                        elif t['status'] == 'Done': icon = "✅"
                        elif t['status'] == 'Error': icon = "❌"
                        
                        content += f"{icon} **{t['algo']} ({t['att']})** (LR={t['lr']}, γ={t['gamma']})\n\n"
                    status_container_att_only.markdown(content)
                
                render_att_only_status()
                
                data_path = "temp_sensor_data.csv"
                data_filename = st.session_state.get('data_filename', None)
                
                for i, t in enumerate(att_only_tasks):
                    att_only_tasks[i]['status'] = 'Running'
                    render_att_only_status()
                    
                    # Train with attention
                    res = rl_pdm.train_single_model(
                        data_path,
                        t['algo'],
                        t['lr'],
                        t['gamma'],
                        ui_callback_att_only,
                        attention_type=t['att'],
                        data_filename=data_filename
                    )
                    st.session_state.training_results.append(res)
                    
                    if 'error' in res:
                        att_only_tasks[i]['status'] = 'Error'
                        st.error(f"Error: {res['error']}")
                    else:
                        att_only_tasks[i]['status'] = 'Done'
                    
                    render_att_only_status()
                
                st.toast("✅ Attention Model Training Complete!", icon="✅")
                st.rerun()

    # --- ATTENTION STEP --- Training Handler (Outside start_training block to survive rerun)
    # --- ATTENTION STEP --- Training Handler (Outside start_training block to survive rerun)
    if 'apply_attention' in locals() and apply_attention:
        # Check if file uploaded
        if uploaded_file is None:
            st.error("Please upload a CSV file first in the Training tab.")
        else:
            # Check if any attention mechanisms are selected
            selected_attention = st.session_state.get('attention_options', {})
            enabled_attention = [k for k, v in selected_attention.items() if v]
            
            if not enabled_attention:
                st.warning("Please select at least one attention mechanism in the Configuration tab.")
            else:
                # --- PHASE 1: BASE AGENT TRAINING ---
                st.subheader("Phase 1: Training Base Agents")
                
                # specific containers for this run
                phase1_container = st.container()
                phase1_status = st.empty()
                
                # Save file
                data_path = save_uploaded_file(uploaded_file)
                base_filename = st.session_state.get('data_filename', 'Unknown')
                
                # Clear previous results for a fresh start
                st.session_state.training_results = []
                
                # Set Globals
                rl_pdm.EPISODES = episodes
                rl_pdm.WEAR_THRESHOLD = st.session_state.get('wear_threshold', 285)
                
                # Generate Base Tasks
                base_tasks = []
                algo_names = st.session_state.get('selected_algorithms', ['PPO', 'A2C', 'DQN', 'REINFORCE'])
                for algo in algo_names:
                    for lr in lrs:
                        for gm in gammas:
                            base_tasks.append({'algo': algo, 'lr': lr, 'gamma': gm, 'status': 'Pending'})
                
                # Define Callback
                def ui_callback_phase1(combo_name, metrics):
                    fig = plot_4_panel(metrics, f"Training: {combo_name}", data_filename=base_filename)
                    # We can use the main placeholder or a specific one
                    plot_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Render Status
                def render_phase1_status():
                    content = "**Phase 1 Queue:**\n\n"
                    for t in base_tasks:
                        icon = "⏳"
                        if t['status'] == 'Running': icon = "🔄"
                        elif t['status'] == 'Done': icon = "✅"
                        elif t['status'] == 'Error': icon = "❌"
                        content += f"{icon} **{t['algo']}** (LR={t['lr']}, γ={t['gamma']})\n\n"
                    phase1_status.markdown(content)
                
                # Run Phase 1
                for i, task in enumerate(base_tasks):
                    task['status'] = 'Running'
                    render_phase1_status()
                    
                    res = rl_pdm.train_single_model(
                        data_path, task['algo'], task['lr'], task['gamma'], 
                        ui_callback_phase1, data_filename=base_filename
                    )
                    st.session_state.training_results.append(res)
                    
                    if 'error' in res:
                        task['status'] = 'Error'
                    else:
                        task['status'] = 'Done'
                    render_phase1_status()
                    
                st.success("Phase 1 Complete!")
                
                # --- PHASE 2: SELECT BEST AGENT ---
                # Find Best Agent from results
                valid_results = [r for r in st.session_state.training_results if 'error' not in r]
                best_agent = None
                
                if not valid_results:
                    st.error("No valid base agents trained. Cannot proceed to attention.")
                else:
                    # Select best based on Weighted Score (or Avg Reward if desired)
                    # Using Weighted Score as it is the primary metric
                    best_agent = max(valid_results, key=lambda x: x.get('Weighted Score', -float('inf')))
                    
                    st.info(f"Phase 2: Best Agent Selected: **{best_agent['Agent']}** (Score: {best_agent['Weighted Score']:.3f})")
                    
                    # --- PHASE 3: APPLY ATTENTION ---
                    st.subheader(f"Phase 3: Applying Attention to {best_agent['Agent']}")
                    phase3_status = st.empty()
                    
                    st.caption(f"Selected Mechanisms: {', '.join(enabled_attention)}")
                    
                    # Define Attention Tasks
                    att_tasks = []
                    for att_type in enabled_attention:
                        att_tasks.append({
                            'algo': best_agent['Agent'], 
                            'lr': best_agent['LR'], 
                            'gamma': best_agent['Gamma'], 
                            'att': att_type, 
                            'status': 'Pending'
                        })
                    
                    def render_att_status():
                        content = "**Attention Queue:**\n\n"
                        for t in att_tasks:
                            icon = "⏳"
                            if t['status'] == 'Running': icon = "🔄"
                            elif t['status'] == 'Done': icon = "✅"
                            elif t['status'] == 'Error': icon = "❌"
                            content += f"{icon} **{t['algo']} ({t['att']})**\n\n"
                        phase3_status.markdown(content)
                        
                    # Run Phase 3
                    for i, t in enumerate(att_tasks):
                        t['status'] = 'Running'
                        render_att_status()
                        
                        # Clean Algo Name (remove any existing suffix if present, though base shouldn't have one)
                        algo_base = t['algo'].split(' ')[0]
                        
                        res = rl_pdm.train_single_model(
                            data_path, 
                            algo_base, 
                            t['lr'], 
                            t['gamma'], 
                            ui_callback_phase1, # Reuse callback
                            attention_type=t['att'],
                            data_filename=base_filename
                        )
                        st.session_state.training_results.append(res)
                        
                        if 'error' in res:
                            t['status'] = 'Error'
                            st.error(f"Error training {t['algo']} ({t['att']}): {res['error']}")
                        else:
                            t['status'] = 'Done'
                        render_att_status()
                    
                    st.toast("✅ Full AutoRL Pipeline Complete!", icon="🚀")
                    st.balloons()
                    # Optional: Rerun to refresh the Comparison Tab with all results
                    # time.sleep(1)
                    st.rerun()

    # COMPARISON TAB
    with right_tabs[1]:
        # Show Comparison
        if st.session_state.training_results:
            # st.markdown("---")
            # Toggle State for comparison
            if 'show_comparison' not in st.session_state:
                st.session_state.show_comparison = False
            
            if compare_btn:
                st.session_state.show_comparison = True # Activate
            
            if st.session_state.show_comparison:
                st.subheader("Training Logs & Comparison")
            
            # Show sensor plot in collapsed section at top
            if st.session_state.get('sensor_fig'):
                with st.expander("📊 Sensor Data Overview", expanded=False):
                    st.plotly_chart(st.session_state.sensor_fig, use_container_width=True, key="sensor_plot_comparison")
            
            valid_results = [r for r in st.session_state.training_results if 'error' not in r]
            
            if valid_results:
                df_res = pd.DataFrame(valid_results)
                
                # Display Table
                def highlight_custom(data):
                    is_max = data == data.max()
                    is_min = data == data.min()
                    styles = []
                    for v in data.index:
                        if data.name in ['Avg Reward', 'Weighted Score']:
                            styles.append('background-color: rgba(93, 172, 119, 0.8)' if is_max[v] else '') # 181, 221, 183 original RGB (76, 175, 80, 0.4)
                        elif data.name in ['Avg Violations', 'Avg Replacements', 'Avg Wear Margin', 'T_ss', 'Sigma_ss']:
                            styles.append('background-color: rgba(93, 172, 119, 0.8)' if is_min[v] else '')
                        else:
                            styles.append('')
                    return styles
                
                # Highlight Logic Wrapper
                st.dataframe(
                    df_res[['Agent', 'LR', 'Gamma', 'Avg Wear Margin', 'Avg Reward', 'Avg Violations', 'Avg Replacements', 'T_ss', 'Sigma_ss', 'Weighted Score']]
                    .style
                    .apply(highlight_custom, axis=0)
                    .format("{:.3f}", subset=['LR', 'Gamma', 'Avg Wear Margin', 'Avg Reward', 'Avg Violations', 'Avg Replacements', 'T_ss', 'Sigma_ss', 'Weighted Score'])
                    .set_properties(**{'text-align': 'right'})
                    .set_table_styles([
                        dict(selector="th", props=[("text-align", "right")])
                    ])
                )
                
                # --- SAVE RESULTS BUTTON ---
                if st.button("Save Results"):
                    excel_file = "Model_Training_Results.xlsx"
                    
                    # Prepare data for export
                    export_data = []
                    
                    # Determine next ID
                    start_id = 0
                    if os.path.exists(excel_file):
                        try:
                            existing_df = pd.read_excel(excel_file)
                            if 'ID' in existing_df.columns and not existing_df.empty:
                                start_id = existing_df['ID'].max() + 1
                        except Exception as e:
                            st.warning(f"Could not read existing file for ID increment: {e}")

                    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    data_filename = st.session_state.get('data_filename', 'Unknown')
                    
                    for i, res in enumerate(valid_results):
                        # Infer finite horizon
                        # We don't strictly track FH vs Infinite in 'res' right now other than context
                        # But typically AutoRL uses Finite Horizon (FH) if configured.
                        # We'll default to 'FH' for now as per AutoRL context, or check params if available.
                        # Assuming FH based on standard usage here.
                        horizon_type = "FH" 
                        
                        row = {
                            'ID': start_id + i,
                            'Date': current_date,
                            'Agent': res.get('Agent', 'Unknown'),
                            'Training data': data_filename,
                            'Episodes': rl_pdm.EPISODES, # Use global or retrieve if stored in res
                            'Finite Horizon': horizon_type, 
                            'LR': res.get('LR'),
                            'Gamma': res.get('Gamma'),
                            'Avg Wear Margin': res.get('Avg Wear Margin'),
                            'Avg Reward': res.get('Avg Reward'),
                            'Avg Violations': res.get('Avg Violations'),
                            'Avg Replacements': res.get('Avg Replacements'),
                            'T_ss': res.get('T_ss', 0),
                            'Sigma_ss': res.get('Sigma_ss', 0),
                            'Weighted Score': res.get('Weighted Score'),
                            'Model file name': res.get('model_filename', '')
                        }
                        export_data.append(row)
                    
                    if export_data:
                        new_df = pd.DataFrame(export_data)
                        
                        try:
                            if os.path.exists(excel_file):
                                # Append to existing
                                # openpyxl is needed for reading/writing xlsx
                                with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                                    # Load existing to find length? 
                                    # Pandas 'overlay' mode might strictly overlay. 
                                    # Safe bet: Read all, concat, write all. 'a' mode in pandas is tricky with overlay.
                                    # Let's simple-path it: Read, Concat, Write.
                                    pass # Logic handled below
                                
                                existing_df = pd.read_excel(excel_file)
                                final_df = pd.concat([existing_df, new_df], ignore_index=True)
                                final_df.to_excel(excel_file, index=False)
                            else:
                                new_df.to_excel(excel_file, index=False)
                            
                            st.success(f"✅ Results saved to **{excel_file}** ({len(export_data)} rows added).")
                        except Exception as e:
                            st.error(f"Error saving to Excel: {e}")
                    else:
                        st.warning("No data to save.")

                # --- BEST AGENT COMPARISON ---
                st.markdown("---")
                st.subheader("Best Agent Comparison")
                
                # Get best agents for comparison
                best_agents_info = rl_pdm.get_best_agents_for_comparison(valid_results)
                best_plain = best_agents_info['best_plain']
                best_attention = best_agents_info['best_attention']
                show_comparison = best_agents_info['show_comparison']
                
                if best_plain or best_attention:
                    # Determine what to display
                    if show_comparison and best_plain and best_attention:
                        # Show both superimposed
                        st.info(f"📊 Comparing: **{best_plain['Agent']}** vs **{best_attention['Agent']}**")
                        
                        # Create superimposed 4-panel plot
                        fig_best = make_subplots(
                            rows=2, cols=2, 
                            subplot_titles=("Avg Reward", "Wear Margin", "Violation Rate", "Replacement Rate")
                        )
                        
                        # Plain agent - solid lines, blue
                        plain_metrics = best_plain['full_metrics']
                        plain_color = '#636EFA'
                        
                        # Reward
                        fig_best.add_trace(go.Scatter(
                            y=smooth_data(plain_metrics['rewards'], max(10, int(len(plain_metrics['rewards']) * 0.1))),
                            name=f"{best_plain['Agent']} (Plain)",
                            line=dict(color=plain_color, width=3),
                            showlegend=True
                        ), row=1, col=1)
                        
                        # Margin
                        fig_best.add_trace(go.Scatter(
                            y=smooth_data(plain_metrics['margins'], max(10, int(len(plain_metrics['margins']) * 0.1))),
                            name=f"{best_plain['Agent']} (Plain)",
                            line=dict(color=plain_color, width=3),
                            showlegend=False
                        ), row=1, col=2)
                        
                        # Violations
                        fig_best.add_trace(go.Scatter(
                            y=smooth_data(plain_metrics['violations'], max(10, int(len(plain_metrics['violations']) * 0.1))),
                            name=f"{best_plain['Agent']} (Plain)",
                            line=dict(color=plain_color, width=3),
                            showlegend=False
                        ), row=2, col=1)
                        
                        # Replacements
                        fig_best.add_trace(go.Scatter(
                            y=smooth_data(plain_metrics['replacements'], max(10, int(len(plain_metrics['replacements']) * 0.1))),
                            name=f"{best_plain['Agent']} (Plain)",
                            line=dict(color=plain_color, width=3),
                            showlegend=False
                        ), row=2, col=2)
                        
                        # Attention agent - dashed lines, orange
                        att_metrics = best_attention['full_metrics']
                        att_color = '#FF6692'
                        
                        # Reward
                        fig_best.add_trace(go.Scatter(
                            y=smooth_data(att_metrics['rewards'], max(10, int(len(att_metrics['rewards']) * 0.1))),
                            name=f"{best_attention['Agent']} (Attention)",
                            line=dict(color=att_color, width=3, dash='dash'),
                            showlegend=True
                        ), row=1, col=1)
                        
                        # Margin
                        fig_best.add_trace(go.Scatter(
                            y=smooth_data(att_metrics['margins'], max(10, int(len(att_metrics['margins']) * 0.1))),
                            name=f"{best_attention['Agent']} (Attention)",
                            line=dict(color=att_color, width=3, dash='dash'),
                            showlegend=False
                        ), row=1, col=2)
                        
                        # Violations
                        fig_best.add_trace(go.Scatter(
                            y=smooth_data(att_metrics['violations'], max(10, int(len(att_metrics['violations']) * 0.1))),
                            name=f"{best_attention['Agent']} (Attention)",
                            line=dict(color=att_color, width=3, dash='dash'),
                            showlegend=False
                        ), row=2, col=1)
                        
                        # Replacements
                        fig_best.add_trace(go.Scatter(
                            y=smooth_data(att_metrics['replacements'], max(10, int(len(att_metrics['replacements']) * 0.1))),
                            name=f"{best_attention['Agent']} (Attention)",
                            line=dict(color=att_color, width=3, dash='dash'),
                            showlegend=False
                        ), row=2, col=2)
                        
                        # Update layout
                        fig_best.update_layout(
                            title_text=f"Best Plain vs Best Attention: {best_plain['Agent']} vs {best_attention['Agent']}",
                            height=600,
                            template="plotly_white",
                            showlegend=True,
                            legend=dict(x=0.5, y=-0.15, xanchor='center', orientation='h')
                        )
                        
                        st.plotly_chart(fig_best, use_container_width=True, key="best_agent_comparison")
                        
                        # Show comparison metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label="Best Plain",
                                value=best_plain['Agent'],
                                delta=f"Score: {best_plain['Weighted Score']:.3f}"
                            )
                        with col2:
                            delta = best_attention['Weighted Score'] - best_plain['Weighted Score']
                            st.metric(
                                label="Best Attention",
                                value=best_attention['Agent'],
                                delta=f"Score: {best_attention['Weighted Score']:.3f} ({delta:+.3f})"
                            )
                    
                    elif best_plain and not best_attention:
                        # Only plain agent exists
                        st.success(f"✅ Best Agent: **{best_plain['Agent']}** (No attention agents trained yet)")
                        
                        # Show single agent plot
                        plain_metrics = best_plain['full_metrics']
                        filename = st.session_state.get('data_filename', 'Unknown')
                        t_ss = best_plain.get('T_ss', None)
                        sigma_ss = best_plain.get('Sigma_ss', None)
                        fig_single = plot_4_panel(plain_metrics, f"Best Agent: {best_plain['Agent']}", 
                                                 data_filename=filename, t_ss=t_ss, sigma_ss=sigma_ss)
                        st.plotly_chart(fig_single, use_container_width=True, key="best_plain_single")
                    
                    elif best_attention and not show_comparison:
                        # Attention exists but plain is better - show only plain
                        st.success(f"✅ Best Agent: **{best_plain['Agent']}** (Plain agent outperforms all attention mechanisms)")
                        st.caption(f"Best attention was: {best_attention['Agent']} (Score: {best_attention['Weighted Score']:.3f})")
                        
                        # Show only plain agent plot
                        plain_metrics = best_plain['full_metrics']
                        filename = st.session_state.get('data_filename', 'Unknown')
                        t_ss = best_plain.get('T_ss', None)
                        sigma_ss = best_plain.get('Sigma_ss', None)
                        fig_single = plot_4_panel(plain_metrics, f"Best Agent: {best_plain['Agent']}", 
                                                 data_filename=filename, t_ss=t_ss, sigma_ss=sigma_ss)
                        st.plotly_chart(fig_single, use_container_width=True, key="best_plain_outperforms")
                
                # --- MULTI-SELECT COMPARISON (Original) ---
                st.markdown("---")
                st.subheader("Custom Agent Comparison")
                
                # Superimposed Plots
                # Create a unique ID for each run
                df_res['ID'] = df_res['Agent'] + " LR:" + df_res['LR'].astype(str) + " G:" + df_res['Gamma'].astype(str)
                
                selected_ids = st.multiselect("Select Agents to Compare", df_res['ID'].unique(), default=df_res['ID'].unique())
                
                if st.button("Update Plot"):
                     # Filter
                     subset = df_res[df_res['ID'].isin(selected_ids)]
                     
                     # Plot 4 Superimposed
                     fig_comp = make_subplots(rows=2, cols=2, subplot_titles=("Avg Reward", "Wear Margin", "Violation Rate", "Replacement Rate"))
                     
                     import plotly.colors as pc
                     
                     # Initialize color map if not exists
                     if 'agent_colors' not in st.session_state:
                         st.session_state.agent_colors = {}
                         
                     # Use Tableau 10 palette (Pastel-ish/Standard Tableau)
                     palette = pc.qualitative.T10
                     
                     for idx, row in subset.iterrows():
                         agent_id = row['ID']
                         
                         # Assign color if not assigned
                         if agent_id not in st.session_state.agent_colors:
                             # Pick next color cyclically
                             next_color_idx = len(st.session_state.agent_colors) % len(palette)
                             st.session_state.agent_colors[agent_id] = palette[next_color_idx]
                         
                         color = st.session_state.agent_colors[agent_id]
    
                         metrics = row['full_metrics']
                         
                         # Smooth window
                         w = max(10, int(len(metrics['rewards']) * 0.1))

                         # Reward
                         fig_comp.add_trace(go.Scatter(y=metrics['rewards'], line=dict(color=color, width=1), opacity=0.3, legendgroup=row['ID'], showlegend=False), row=1, col=1)
                         fig_comp.add_trace(go.Scatter(y=smooth_data(metrics['rewards'], w), line=dict(color=color, width=3), name=row['ID'], legendgroup=row['ID']), row=1, col=1)
                         
                         # Margin
                         fig_comp.add_trace(go.Scatter(y=metrics['margins'], line=dict(color=color, width=1), opacity=0.3, legendgroup=row['ID'], showlegend=False), row=1, col=2)
                         fig_comp.add_trace(go.Scatter(y=smooth_data(metrics['margins'], w), line=dict(color=color, width=3), name=row['ID'], legendgroup=row['ID'], showlegend=False), row=1, col=2)
                         
                         # Add steady-state shaded region
                         if 'T_ss' in row and row['T_ss'] is not None and row['T_ss'] > 0:
                             t_ss_val = row['T_ss']
                             # If T_ss is larger than length, clamp it
                             t_ss_val = min(t_ss_val, len(metrics['margins'])-1)
                             
                             fig_comp.add_vrect(
                                 x0=t_ss_val, 
                                 x1=len(metrics['margins']),
                                 fillcolor=color, 
                                 opacity=0.1,
                                 layer="below", 
                                 line_width=0,
                                 row=1, col=2
                             )
                         
                         # Violations
                         fig_comp.add_trace(go.Scatter(y=metrics['violations'], line=dict(color=color, width=1), opacity=0.3, legendgroup=row['ID'], showlegend=False), row=2, col=1)
                         fig_comp.add_trace(go.Scatter(y=smooth_data(metrics['violations'], w), line=dict(color=color, width=3), name=row['ID'], legendgroup=row['ID'], showlegend=False), row=2, col=1)
                         
                         # Replacements
                         fig_comp.add_trace(go.Scatter(y=metrics['replacements'], line=dict(color=color, width=1), opacity=0.3, legendgroup=row['ID'], showlegend=False), row=2, col=2)
                         fig_comp.add_trace(go.Scatter(y=smooth_data(metrics['replacements'], w), line=dict(color=color, width=3), name=row['ID'], legendgroup=row['ID'], showlegend=False), row=2, col=2)
                     
                     filename = st.session_state.get('data_filename', 'Unknown')
                     comp_title = f"Agent Comparison | Data: {filename}"
                     fig_comp.update_layout(height=700, template="plotly_white", title_text=comp_title)
                     st.plotly_chart(fig_comp, use_container_width=True, key="custom_agent_comparison")
            else:
                 st.info("No successful training runs to display.")

    # EVALUATION TAB
    with right_tabs[2]:
        # === EVALUATION RESULTS DISPLAY ===
        if st.session_state.eval_results is not None:
            # st.markdown("---")
            eval_results = st.session_state.eval_results
            eval_file_name = st.session_state.get('eval_file_name', 'Unknown')
            
            st.header(f"Evaluation Results   ⏵   {st.session_state.eval_model_name}")
            st.subheader(f'Test File: {eval_file_name}', divider=True)
            # st.subheader(f'Agent: {st.session_state.eval_model_name} | Test File: {eval_file_name}')
            # st.info(f"📁 Test Data File: **{eval_file_name}**")
            
            # Display model override warning if applicable
            # if eval_results.get('model_override', False):
                # st.warning(f"Replacement suggested at timestep {eval_results.get('override_timestep', 'N/A')}.")
            
            # Display IAR bounds if available
            # if 'IAR_lower' in eval_results and 'IAR_upper' in eval_results:
            #     # st.markdown(f"**Agent: {st.session_state.eval_model_name}**")
            #     col_IAR1, col_IAR2, col_IAR3, col_IAR4 = st.columns(4)
            #     with col_IAR1:
            #         st.metric("Wear Threshold", f"{eval_results['wear_threshold']:.2f}", border=True)
            #     with col_IAR2:
            #         st.metric("Ideal Replacement Range", f"± {100.0*rl_pdm.IAR_RANGE:.2f}%", border=True)
            #     with col_IAR3:
            #         st.metric("Test file", eval_file_name, border=True)
            #     with col_IAR4:   
            #         st.metric("Evaluation Steps", len(eval_results['timesteps']), border=True)
                   
            # Display metrics
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            with col_m1:
                st.metric("Tool Usage %", f"{100*eval_results['tool_usage_pct']:.1f}%" if eval_results.get('tool_usage_pct') is not None else 'N/A', help="t_FR / T_wt", border=True)
            with col_m2:
                st.metric("Lambda (λ)", eval_results.get('lambda', 'N/A'), help="T_wt - t_FR. Ideal is zero.", border=True)
            with col_m3:
                st.metric("Threshold Violations", eval_results['threshold_violations'], help="Number of timesteps where wear exceeded threshold.", border=True)
            with col_m4:
                st.metric("Wear Threshold Timestep (T_wt)", eval_results.get('T_wt', 'N/A'), help="Timestep where tool wear first crosses the threshold.", border=True)
            with col_m5:
                st.metric("First Replacement at (t_FR)", eval_results.get('t_FR', 'N/A'), help="Timestep of the first replacement action.", border=True)
                
            
            # Plot evaluation results
            fig_eval = plot_evaluation_results(eval_results, st.session_state.eval_model_name)
            st.plotly_chart(fig_eval, use_container_width=True, key="evaluation_results_plot")
            # Option to clear evaluation
            if st.button("Clear Evaluation Results"):
                st.session_state.eval_results = None
                st.rerun()
