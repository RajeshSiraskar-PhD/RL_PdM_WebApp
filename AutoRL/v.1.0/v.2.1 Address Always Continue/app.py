# ---------------------------------------------------------------------------------------
# AutoRL: Auto-train Predictive Maintenance Agents 
# 20-Jan: V.1.2: Fixed x-axis length
# 23-Jan: V.1.3: Reduced violations and add scores
# 23-Jan: V.2.0: Add evaluation
# 23-Jan: V.2.1: Add sensor plot on loading
# 
# ---------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import os
import rl_pdm # Import our backend
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    /* Texts title: #f5a55b*/
    h1 {
        color: #f5a55b !important;
    }

    h2, h3, p, label {
        color: #95b8d1 !important;
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
    
    /* Buttons #2196F3 hover: #1976D2 ##313e85*/
    .stButton>button {
        background-color: #0660ba; 
        color: white !important;
        border-radius: 5px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1111d9;
        color: white !important;
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

def plot_4_panel(metrics, title, height=600, data_filename=None):
    """
    Helper to generate the 4-panel plot.
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
    - Right axis: Actions (red spikes for replacements)
    - Dotted line: Wear Threshold
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
    
    # Add actions as red stem plot on secondary y-axis
    # Only show actions where action == 0 (replacement)
    replacement_timesteps = [t for t, a in zip(timesteps, actions) if a == 0]
    replacement_values = [1] * len(replacement_timesteps)
    
    fig.add_trace(
        go.Scatter(
            x=replacement_timesteps,
            y=replacement_values,
            name="Actions (Replacement)",
            mode='markers',
            marker=dict(
                size=12,
                color='#EF553B',
                symbol='diamond'
            ),
            showlegend=True
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_xaxes(title_text="Timestep")
    fig.update_yaxes(title_text="Tool Wear", secondary_y=False)
    fig.update_yaxes(title_text="Action (1=Replace)", secondary_y=True, range=[-0.5, 1.5])
    
    fig.update_layout(
        title=f"Model Evaluation: {model_name}",
        height=500,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

# --- LAYOUT --- # $$$
st.title("AutoRL: Auto-train Predictive Maintenance Agents") 
st.markdown(' - V.2.03: 3x3 grid plot')


col1, col2 = st.columns([1.7, 8.3])

# --- LEFT PANEL: AGENT TRAINING & EVALUATION ---
with col1:
    # Tabs for different operations
    left_tabs = st.tabs(["Training", "Evaluate"])
    
    # Training Tab
    with left_tabs[0]:
        st.subheader("Agent Training")
        
        # File Loader
        uploaded_file = st.file_uploader("Upload Sensor Data (CSV)", type="csv")
        
        # Params
        episodes = st.number_input("Episodes", min_value=1, value=100)
        
        # Hyperparams Input
        st.subheader("Hyperparameters")
        lr_input = st.text_input("Learning Rates (comma sep)", "0.001")
        gamma_input = st.text_input("Gammas (comma sep)", "0.99")
        
        # Parse inputs
        try:
            lrs = [float(x.strip()) for x in lr_input.split(",")]
        except:
            lrs = [0.001]
            
        try:
            gammas = [float(x.strip()) for x in gamma_input.split(",")]
        except:
            gammas = [0.99]
        
        # AutoRL Button
        start_training = st.button("AutoRL - Auto train")
        st.subheader("Attention mechanism: Nadaraya-Watson & Deep-Learning")
        apply_attention = st.button("Apply Attention")
        # Separator
        st.markdown("---")
        st.subheader("Results")
        compare_btn = st.button("Compare Agents")
    
    # Evaluate Tab
    with left_tabs[1]:
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
                        # Run evaluation
                        model_path = os.path.join("models", selected_model)
                        eval_results = rl_pdm.evaluate_model(model_path, test_file_path, wear_threshold=300)
                        
                        # Store in session state for plotting
                        st.session_state.eval_results = eval_results
                        st.session_state.eval_model_name = selected_model
                        st.session_state.eval_file_name = test_file.name  # Store test file name
                        st.success("Evaluation complete!")
                        
                    except Exception as e:
                        st.error(f"Error evaluating model: {str(e)}")
        else:
            st.warning("No trained models available. Run 'AutoRL - Auto train' first to generate models.")
    
# --- RIGHT PANE: MONITORING ---
with col2:
    plot_placeholder = st.empty()
    logs_placeholder = st.empty()
    
    # Check for uploaded file in UI (re-check state or handle it)
    # Since uploaded_file is defined in col1, we access it here.
    # Initialize session state for sensor plot visibility and file tracking
    if 'show_sensor_plot' not in st.session_state:
        st.session_state.show_sensor_plot = True
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None

    # Check for uploaded file in UI
    if 'uploaded_file' in locals() and uploaded_file is not None:
         # Check if it's a new file
         if st.session_state.last_uploaded_file != uploaded_file.name:
             st.session_state.show_sensor_plot = True
             st.session_state.last_uploaded_file = uploaded_file.name
             # Extract base filename (without path and .csv extension)
             base_filename = uploaded_file.name.replace('.csv', '').replace('.CSV', '')
             st.session_state.data_filename = base_filename
         
         path = save_uploaded_file(uploaded_file)
         if path and st.session_state.show_sensor_plot:
             fig_sensor = rl_pdm.plot_sensor_data(path)
             if fig_sensor:
                 fig_sensor.update_layout(title_text=f"Sensor Data Overview: {uploaded_file.name}")
                 st.plotly_chart(fig_sensor, use_container_width=True)

    # Initialize session state for results if not exists
    if 'training_results' not in st.session_state:
        st.session_state.training_results = []
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = None
    
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
                fig = plot_4_panel(metrics, f"Training: {combo_name}", data_filename=filename)
                plot_placeholder.plotly_chart(fig, use_container_width=True)


            
            # Run Training
            # Update Globals in rl_pdm (Hack, but effective)
            rl_pdm.EPISODES = episodes
            rl_pdm.WEAR_THRESHOLD = 300 # Fixed as per prompt or could be input
            
            # Generate Task List
            task_list = []
            algo_names = ['PPO', 'A2C', 'DQN']
            # algo_names = ['A2C', 'DQN']
            for algo in algo_names:
                for lr in lrs:
                    for gm in gammas:
                        task_list.append({'algo': algo, 'lr': lr, 'gamma': gm, 'status': 'Pending'})
            
            # Status Container
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
            
            render_status()
            
            results = []
            st.session_state.training_results = [] # Clear previous results
            
            # Loop through tasks
            for i, task in enumerate(task_list):
                # Update Status
                task_list[i]['status'] = 'Running'
                render_status()
                
                # Train
                res = rl_pdm.train_single_model(data_path, task['algo'], task['lr'], task['gamma'], ui_callback)
                results.append(res)
                st.session_state.training_results.append(res) # Append incrementally
                
                # Update Status
                if 'error' in res:
                     task_list[i]['status'] = 'Error'
                else:
                     task_list[i]['status'] = 'Done'
                render_status()
            
            st.success("All Training Complete!")
            

            # plot_placeholder.empty() # Keep last plot or clear? User said "Wipe out the plots, before starting the next..." implicit in loop.
            # "When the training is done... store... Wipe out... before starting next"
            # Since AutoRL loops internally, the callback updates nicely.
            # At end, we might want to show summary.

    # Show Comparison
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
                    title = f"{res['Agent']} (R={res['Avg Reward']:.1f}, LR={res['LR']:.3f}, Gamma={res['Gamma']:.3f})"
                    with st.expander(title, expanded=False):
                        # Re-generate plot (fast enough usually)
                        filename = st.session_state.get('data_filename', 'Unknown')
                        fig = plot_4_panel(res['full_metrics'], title, height=400, data_filename=filename)
                        st.plotly_chart(fig, use_container_width=True)

    # --- ATTENTION STEP --- Training Handler (Outside start_training block to survive rerun)
    if 'apply_attention' in locals() and apply_attention:
        # Check if results exist
        if 'training_results' not in st.session_state or not st.session_state.training_results:
             st.warning("Please run 'AutoRL - Auto train' first to generate base agents.")
        else:
            # Find Best Agent from Session State
            valid_results = [r for r in st.session_state.training_results if 'error' not in r]
            best_agent = None
            best_reward = -float('inf')
            
            for res in valid_results:
                if res['Avg Reward'] > best_reward:
                    best_reward = res['Avg Reward']
                    best_agent = res
            
            if best_agent:
                st.info(f"Applying Attention to Best Agent: **{best_agent['Agent']}**")
                
                # Prepare UI Callback
                def ui_callback_att(combo_name, metrics):
                    filename = st.session_state.get('data_filename', 'Unknown')
                    fig = plot_4_panel(metrics, f"Training: {combo_name}", data_filename=filename)
                    plot_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Data Path
                data_path = "temp_sensor_data.csv" # Standard path
                if not os.path.exists(data_path):
                    st.error("Data file not found. Please upload again.")
                else:
                    # Add Attention tasks
                    att_tasks = [
                        {'algo': best_agent['Agent'], 'lr': best_agent['LR'], 'gamma': best_agent['Gamma'], 'att': 'NW', 'status': 'Pending'},
                        {'algo': best_agent['Agent'], 'lr': best_agent['LR'], 'gamma': best_agent['Gamma'], 'att': 'DL', 'status': 'Pending'}
                    ]
                    
                    status_container_att = st.empty()
                    def render_att_status():
                        content = "**Attention Queue:**\n\n"
                        for t in att_tasks:
                            icon = "⏳"
                            if t['status'] == 'Running': icon = "🔄"
                            elif t['status'] == 'Done': icon = "✅"
                            elif t['status'] == 'Error': icon = "❌"
                            
                            content += f"{icon} **{t['algo']} ({t['att']})**\n\n"
                        status_container_att.markdown(content)
                    
                    render_att_status()

                    # Fix: params might need to be retrieved from Session or Globals
                    # We assume globals were set during AutoRL, but if rerun happens, they might reset?
                    # Best to ensure they are set.
                    rl_pdm.EPISODES = best_agent.get('EPISODES', 100) # Fallback
                    # Wait, 'res' doesn't store EPISODES. 
                    # We should probably use the UI input 'episodes' which is available in scope.
                    rl_pdm.EPISODES = episodes 
                    rl_pdm.WEAR_THRESHOLD = 300
                    
                    for i, t in enumerate(att_tasks):
                        att_tasks[i]['status'] = 'Running'
                        render_att_status()
                        
                        # Clean Algo Name
                        algo_base = t['algo'].split(' ')[0]
                        
                        res = rl_pdm.train_single_model(
                            data_path, 
                            algo_base, 
                            t['lr'], 
                            t['gamma'], 
                            ui_callback_att,
                            attention_type=t['att']
                        )
                        st.session_state.training_results.append(res)
                        
                        if 'error' in res:
                                att_tasks[i]['status'] = 'Error'
                                st.error(f"Error training {t['algo']} ({t['att']}): {res['error']}")
                                with st.expander("Traceback"):
                                    st.code(res.get('traceback', ''))
                        else:
                                att_tasks[i]['status'] = 'Done'
                        render_att_status()
                    
                    st.success("Attention Training Complete!")
                    st.rerun()



    # Show Comparison
    if st.session_state.training_results:
        st.markdown("---")
        # Toggle State for comparison
        if 'show_comparison' not in st.session_state:
            st.session_state.show_comparison = False
        
        if compare_btn:
            st.session_state.show_comparison = True # Activate
        
        if st.session_state.show_comparison:
            st.subheader("Training Logs & Comparison")
            
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
                        elif data.name in ['Avg Violations', 'Avg Replacements', 'Avg Wear Margin']:
                            styles.append('background-color: rgba(93, 172, 119, 0.8)' if is_min[v] else '')
                        else:
                            styles.append('')
                    return styles
                
                # Highlight Logic Wrapper
                st.dataframe(
                    df_res[['Agent', 'LR', 'Gamma', 'Avg Wear Margin', 'Avg Reward', 'Avg Violations', 'Avg Replacements', 'Weighted Score']]
                    .style
                    .apply(highlight_custom, axis=0)
                    .format("{:.3f}", subset=['LR', 'Gamma', 'Avg Wear Margin', 'Avg Reward', 'Avg Violations', 'Avg Replacements', 'Weighted Score'])
                    .set_properties(**{'text-align': 'right'})
                    .set_table_styles([
                        dict(selector="th", props=[("text-align", "right")])
                    ])
                )
                
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
                         
                         # Violations
                         fig_comp.add_trace(go.Scatter(y=metrics['violations'], line=dict(color=color, width=1), opacity=0.3, legendgroup=row['ID'], showlegend=False), row=2, col=1)
                         fig_comp.add_trace(go.Scatter(y=smooth_data(metrics['violations'], w), line=dict(color=color, width=3), name=row['ID'], legendgroup=row['ID'], showlegend=False), row=2, col=1)
                         
                         # Replacements
                         fig_comp.add_trace(go.Scatter(y=metrics['replacements'], line=dict(color=color, width=1), opacity=0.3, legendgroup=row['ID'], showlegend=False), row=2, col=2)
                         fig_comp.add_trace(go.Scatter(y=smooth_data(metrics['replacements'], w), line=dict(color=color, width=3), name=row['ID'], legendgroup=row['ID'], showlegend=False), row=2, col=2)
                     
                     filename = st.session_state.get('data_filename', 'Unknown')
                     comp_title = f"Agent Comparison | Data: {filename}"
                     fig_comp.update_layout(height=700, template="plotly_white", title_text=comp_title)
                     st.plotly_chart(fig_comp, use_container_width=True)
            else:
                 st.info("No successful training runs to display.")


    # === EVALUATION RESULTS DISPLAY ===
    if st.session_state.eval_results is not None:
        st.markdown("---")
        eval_results = st.session_state.eval_results
        eval_file_name = st.session_state.get('eval_file_name', 'Unknown')
        
        st.subheader(f"Evaluation Results: {st.session_state.eval_model_name}")
        st.info(f"📁 Test Data File: **{eval_file_name}**")
        
        # Display metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Replacements", eval_results['total_replacements'])
        with col_m2:
            st.metric("Threshold Violations", eval_results['threshold_violations'])
        with col_m3:
            st.metric("Evaluation Steps", len(eval_results['timesteps']))
        
        # Plot evaluation results
        fig_eval = plot_evaluation_results(eval_results, st.session_state.eval_model_name)
        st.plotly_chart(fig_eval, use_container_width=True)
        
        # Create results table
        st.subheader("Detailed Results Table")
        
        # Build dataframe from evaluation results
        results_df = pd.DataFrame({
            'Timestep': eval_results['timesteps'],
            'Tool Wear (W_T)': eval_results['tool_wear'],
            'Action': eval_results['actions'],
            'Action Type': ['REPLACE' if a == 0 else 'CONTINUE' for a in eval_results['actions']]
        })
        
        # Add column to show if threshold was exceeded at each timestep
        results_df['Exceed Threshold'] = results_df['Tool Wear (W_T)'] > eval_results['wear_threshold']
        
        # Display table with formatting
        def highlight_exceeds(x):
            return ['background-color: #ffcccc' if v else '' for v in x]
        
        def highlight_action(x):
            return ['background-color: #ff6b6b' if v == 'REPLACE' else '' for v in x]
        
        styled_df = (results_df
            .style
            .format({'Tool Wear (W_T)': '{:.2f}'})
            .apply(highlight_exceeds, subset=['Exceed Threshold'])
            .apply(highlight_action, subset=['Action Type']))
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Option to clear evaluation
        if st.button("Clear Evaluation Results"):
            st.session_state.eval_results = None
            st.rerun()
