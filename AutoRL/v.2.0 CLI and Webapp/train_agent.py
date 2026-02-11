# ---------------------------------------------------------------------------------------
# AutoRL: CLI Version - Train and Evaluate Agents
# Author: Rajesh Siraskar
# CLI for training and evaluation of RL agents for Predictive Maintenance
# V.1.0: 11-Feb-2026: First commit
# ---------------------------------------------------------------------------------------
print('\nAutoRL CLI')
print('- Loading libraries ...')
import argparse
import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import glob
try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    pass  # Will handle gracefully in plot creation function

# Import RL module
import rl_pdm

print('- Loaded.\n - Starting AutoRL CLI Pipeline ...\n')


def get_schema_files(schema: str, data_dir: str = "data") -> List[str]:
    """
    Get all CSV files for the specified schema (SIT or IEEE).
    
    Args:
        schema: 'SIT' or 'IEEE'
        data_dir: Path to data directory
    
    Returns:
        List of full file paths for the schema
    """
    schema_pattern = os.path.join(data_dir, f"{schema}_*.csv")
    files = glob.glob(schema_pattern)
    # Exclude test files and tiny files
    files = [f for f in files if not f.endswith(('_TEST.csv', '_tiny.csv'))]
    files.sort()
    return files


def train_agents(schema: str, algos: List[str], episodes: int, attention_mech: int) -> Dict:
    """
    Train agents on all data files of the specified schema.
    
    Args:
        schema: 'SIT' or 'IEEE'
        algos: List of algorithm names
        episodes: Number of training episodes
        attention_mech: 1 to use attention mechanism, 0 for none
    
    Returns:
        Dictionary mapping (algo, training_file) -> model_path
    """
    print(f"\n{'='*80}")
    print(f"TRAINING PHASE: Schema={schema}, Algos={algos}, Episodes={episodes}, Attention={attention_mech}")
    print(f"{'='*80}\n")
    
    # Set global episodes in rl_pdm
    rl_pdm.EPISODES = episodes
    
    # Get training files
    training_files = get_schema_files(schema)
    if not training_files:
        print(f"ERROR: No training files found for schema {schema}")
        sys.exit(1)
    
    print(f"Training files for {schema}: {[Path(f).stem for f in training_files]}\n")
    
    trained_models = {}
    
    # Mapping from short forms (used in CLI) to rl_pdm attention type names
    attention_short_to_full = {
        'NW': 'NW',           # NadarayaWatson
        'TP': 'Temporal',     # Temporal
        'MH': 'MultiHead',    # MultiHead
        'SA': 'SelfAttn'      # SelfAttn
    }
    
    # Define attention types to train when AM=1
    attention_types = []
    if attention_mech == 1:
        # Train with 4 attention mechanisms: NW, TP, MH, SA (short forms for display)
        attention_types = ['NW', 'TP', 'MH', 'SA']
    else:
        # Train without attention
        attention_types = [None]
    
    # Train combinations
    for training_file in training_files:
        training_filename = Path(training_file).stem
        
        for algo in algos:
            for att_short in attention_types:
                # Map short form to full rl_pdm name
                att_full = attention_short_to_full.get(att_short) if att_short else None
                
                # Format attention label for display
                att_label = f" ({att_short})" if att_short else ""
                print(f"\n>>> Training {algo} on {training_filename}{att_label}")
                
                try:
                    # Train with default hyperparameters
                    result = rl_pdm.train_single_model(
                        data_file=training_file,
                        algo_name=algo,
                        lr=rl_pdm.LR_DEFAULT,
                        gm=rl_pdm.GAMMA_DEFAULT,
                        callback_func=None,  # No callback for CLI
                        attention_type=att_full,  # Pass full name to rl_pdm
                        data_filename=training_filename
                    )
                    
                    if 'error' in result:
                        print(f"  ✗ Training failed: {result['error']}")
                    else:
                        model_path = result['model_path']
                        trained_models[(algo, training_filename, att_short)] = model_path
                        print(f"  ✓ Model saved: {model_path}")
                        print(f"    Weighted Score: {result['Weighted Score']:.4f}")
                        
                except Exception as e:
                    print(f"  ✗ Training error: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"Training complete. Total models trained: {len(trained_models)}")
    print(f"{'='*80}\n")
    
    return trained_models


def create_evaluation_plot(eval_result: Dict, model_filename: str, test_filename: str, results_dir: str = "results"):
    """
    Create and save an evaluation plot showing tool wear and replacements.
    
    Args:
        eval_result: Dictionary from adjusted_evaluate_model
        model_filename: Filename of the model
        test_filename: Filename of the test data
        results_dir: Directory to save plots
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        os.makedirs(results_dir, exist_ok=True)
        
        timesteps = eval_result['timesteps']
        tool_wear = eval_result['tool_wear']
        actions = eval_result['actions']
        wear_threshold = eval_result['wear_threshold']
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
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
        
        # Add IAR bounds if present
        IAR_lower = eval_result.get('IAR_lower', None)
        IAR_upper = eval_result.get('IAR_upper', None)
        if IAR_lower is not None and IAR_upper is not None:
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
        
        # Get replacement points
        model_override = eval_result.get('model_override', False)
        override_timestep = eval_result.get('override_timestep', None)
        
        # Find replacements
        replacement_timesteps = [t for t, a in zip(timesteps, actions) if a == 0]
        
        # Separate normal and override replacements
        if model_override and override_timestep is not None:
            override_replacements = [t for t in replacement_timesteps if t == override_timestep]
            normal_replacements = [t for t in replacement_timesteps if t != override_timestep]
        else:
            normal_replacements = replacement_timesteps
            override_replacements = []
        
        # Add normal replacements as red markers
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
                        color='#EF553B',
                        symbol='diamond',
                        opacity=0.7
                    ),
                    showlegend=True
                ),
                secondary_y=False
            )
        
        # Add overridden replacements if any
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
                        color="#EF3B59",
                        symbol='diamond',
                        opacity=0.7
                    ),
                    showlegend=True
                ),
                secondary_y=False
            )
        
        # Update layout
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Tool Wear", secondary_y=False)
        fig.update_yaxes(title_text="Action", secondary_y=True, range=[0.0, 1.5], showticklabels=False, ticks="", showgrid=False)
        
        fig.update_layout(
            title=f"Model Evaluation: {model_filename}_{test_filename}",
            height=500,
            template="plotly_white",
            hovermode='x unified'
        )
        
        # Save plot
        plot_filename = f"{model_filename}_{test_filename}_Eval.png"
        plot_filepath = os.path.join(results_dir, plot_filename)
        
        fig.write_image(plot_filepath, width=1200, height=600)
        
        return plot_filepath
        
    except Exception as e:
        print(f"  ⚠ Could not create plot: {str(e)}")
        return None


def evaluate_agents(schema: str, trained_models: Dict) -> pd.DataFrame:
    """
    Evaluate all trained models on all data files of the schema.
    
    Args:
        schema: 'SIT' or 'IEEE'
        trained_models: Dictionary mapping (algo, training_file, attention_type) -> model_path
    
    Returns:
        DataFrame with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"EVALUATION PHASE: Evaluating {len(trained_models)} model(s)")
    print(f"{'='*80}\n")
    
    # Get test files (same as training files for same schema)
    test_files = get_schema_files(schema)
    
    results = []
    
    for (algo, train_filename, att_short), model_path in trained_models.items():
        # Format model description with attention if present
        att_label = f" ({att_short})" if att_short else ""
        print(f"\nEvaluating {algo}{att_label} (trained on {train_filename}):")
        
        for test_file in test_files:
            test_filename = Path(test_file).stem
            
            try:
                # Evaluate model
                eval_result = rl_pdm.adjusted_evaluate_model(model_path, test_file)
                
                # Check for errors
                if eval_result.get('error', False):
                    print(f"  ⚠ {test_filename}: Feature mismatch error")
                    continue
                
                # Determine if self-eval
                self_eval = 'Y' if train_filename == test_filename else 'N'

                # Build result row
                # Extract model filename from path
                model_filename = Path(model_path).stem
                
                row = {
                    'Model': f"{algo}",
                    'Model File': model_filename,
                    'Training File': train_filename,
                    'Test File': test_filename,
                    'Self-eval': self_eval,
                    'Tool Usage %': eval_result.get('tool_usage_pct', 0.0) * 100 if eval_result.get('tool_usage_pct') else 0,
                    'Lambda': eval_result.get('lambda', 0),
                    'Threshold Violations': eval_result.get('threshold_violations', 0),
                    'T_wt': eval_result.get('T_wt'),
                    't_FR': eval_result.get('t_FR'),
                    'Model Override': 'Y' if eval_result.get('model_override', False) else 'N',
                    'Eval_Score': calculate_eval_score(eval_result)
                }
                
                results.append(row)
                print(f"  ✓ {test_filename}: Lambda={row['Lambda']}, Violations={row['Threshold Violations']}, Score={row['Eval_Score']:.4f}")
                
                # Create and save evaluation plot
                plot_path = create_evaluation_plot(eval_result, model_filename, test_filename)
                if plot_path:
                    print(f"    📊 Plot saved: {plot_path}")
                
            except Exception as e:
                print(f"  ✗ {test_filename}: Evaluation error - {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete. Total evaluations: {len(results_df)}")
    print(f"{'='*80}\n")
    
    return results_df


def calculate_eval_score(eval_result: Dict) -> float:
    """
    Calculate weighted evaluation score based on metrics.
    
    Weights:
    - Tool Usage: 50%
    - Lambda: 30%
    - Violations: 20%
    
    Args:
        eval_result: Dictionary from adjusted_evaluate_model
    
    Returns:
        Float score between 0-1 (higher is better)
    """
    tool_usage_pct = eval_result.get('tool_usage_pct', 0) or 0
    lambda_metric = eval_result.get('lambda', 0) or 0
    violations = eval_result.get('threshold_violations', 0)
    
    # Normalize tool usage (0-100% is ideal)
    tool_usage_score = min(1.0, max(0.0, tool_usage_pct))
    
    # Normalize lambda (lower is better, 0 is best)
    # Assume lambda range 0-100, scale inversely
    lambda_score = max(0.0, 1.0 - (abs(lambda_metric) / 100.0))
    
    # Normalize violations (lower is better, 0 is best)
    violation_score = max(0.0, 1.0 / (1.0 + violations))
    
    # Weighted combination
    eval_score = (
        0.50 * tool_usage_score +
        0.30 * lambda_score +
        0.20 * violation_score
    )
    
    return eval_score


def save_results(results_df: pd.DataFrame, schema: str, attention_mech: int, results_dir: str = "results") -> str:
    """
    Save evaluation results to CSV file.
    
    Args:
        results_df: DataFrame with results
        schema: 'SIT' or 'IEEE'
        attention_mech: Attention mechanism flag
        results_dir: Directory to save results
    
    Returns:
        Path to saved CSV file
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename with timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    att_label = "AM" if attention_mech else "NoAM"
    filename = f"Evaluation_Results_{schema}_{att_label}_{timestamp}.csv"
    filepath = os.path.join(results_dir, filename)
    
    # Save to CSV
    results_df.to_csv(filepath, index=False)
    print(f"\n✓ Results saved to: {filepath}")
    
    return filepath


def create_heatmaps(results_df: pd.DataFrame, schema: str, attention_mech: int, results_dir: str = "results"):
    """
    Create a single comprehensive heatmap showing evaluation scores for all agents.
    
    Args:
        results_df: DataFrame with evaluation results
        schema: 'SIT' or 'IEEE'
        attention_mech: Attention mechanism flag
        results_dir: Directory to save heatmap
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Pivot for heatmap: rows = ALL model files (all agents), cols = test files
    heatmap_data = results_df.pivot_table(
        index='Model File',
        columns='Test File',
        values='Eval_Score',
        aggfunc='first'
    )
    
    # Sort rows alphabetically for consistent ordering
    heatmap_data = heatmap_data.sort_index()
    
    # Create single comprehensive heatmap with larger figsize for all agents
    num_agents = len(heatmap_data)
    figsize_height = max(10, num_agents * 0.5)  # Scale height based on number of agents
    fig, ax = plt.subplots(figsize=(14, figsize_height))
    
    # Use color scheme: Green (1.0), Yellow (0.5), Red (0.0)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Evaluation Score'},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title(f'Evaluation Score Heatmap\nSchema: {schema}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Test File', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model File', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels (test files) to be horizontal for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
    # Rotate y-axis labels (model files) to be horizontal for readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    # Save single heatmap
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    att_label = "AM" if attention_mech else "NoAM"
    filename = f"Heatmap_AllAgents_{schema}_{att_label}_{timestamp}.png"
    filepath = os.path.join(results_dir, filename)
    
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Comprehensive heatmap saved: {filepath}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoRL: Train and Evaluate RL Agents for Predictive Maintenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_agent.py -S SIT -A PPO,A2C -E 200 -AM 0
  python train_agent.py -S IEEE -A PPO -E 300 -AM 1
  python train_agent.py  # Uses all defaults: SIT, PPO, 200 episodes, no attention
        """
    )
    
    parser.add_argument(
        '-S', '--schema',
        default='SIT',
        choices=['SIT', 'IEEE'],
        help="Data schema: 'SIT' or 'IEEE' (default: SIT)"
    )
    
    parser.add_argument(
        '-A', '--algos',
        default='PPO',
        help="Comma-separated list of algorithms: PPO,A2C,DQN,REINFORCE (default: PPO)"
    )
    
    parser.add_argument(
        '-E', '--episodes',
        type=int,
        default=200,
        help="Number of training episodes (default: 200)"
    )
    
    parser.add_argument(
        '-AM', '--attention-mechanism',
        type=int,
        default=0,
        choices=[0, 1],
        help="Apply attention mechanism: 1 (yes) or 0 (no) (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Parse algorithms
    algos = [algo.strip().upper() for algo in args.algos.split(',')]
    
    # Validate algorithms
    valid_algos = ['PPO', 'A2C', 'DQN', 'REINFORCE']
    for algo in algos:
        if algo not in valid_algos:
            print(f"ERROR: Algorithm '{algo}' not supported. Choose from: {', '.join(valid_algos)}")
            sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"AutoRL: RL Agent Training and Evaluation Pipeline")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Data Schema: {args.schema}")
    print(f"  Algorithms: {', '.join(algos)}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Attention Mechanism: {'ON' if args.attention_mechanism else 'OFF'}")
    print(f"{'='*80}\n")
    
    try:
        # Phase 1: Training
        trained_models = train_agents(
            schema=args.schema,
            algos=algos,
            episodes=args.episodes,
            attention_mech=args.attention_mechanism
        )
        
        if not trained_models:
            print("ERROR: No models were successfully trained.")
            sys.exit(1)
        
        # Phase 2: Evaluation
        results_df = evaluate_agents(schema=args.schema, trained_models=trained_models)
        
        if results_df.empty:
            print("ERROR: Evaluation produced no results.")
            sys.exit(1)
        
        # Phase 3: Save results
        results_file = save_results(results_df, args.schema, args.attention_mechanism)
        
        # Phase 4: Create heatmaps
        print(f"\nGenerating heatmaps...")
        create_heatmaps(results_df, args.schema, args.attention_mechanism)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Results Summary:")
        print(f"  Total Evaluations: {len(results_df)}")
        print(f"  Models Trained: {len(trained_models)}")
        print(f"  Test Files: {results_df['Test File'].nunique()}")
        print(f"  Average Eval Score: {results_df['Eval_Score'].mean():.4f}")
        print(f"\nResults saved in: results/")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
