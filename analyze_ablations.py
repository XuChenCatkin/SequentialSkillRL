#!/usr/bin/env python3
"""
Analyze and compare results from SequentialSkillRL ablation studies.

This script helps analyze training logs and checkpoints from ablation studies
to compare performance across different configurations.

Usage:
    python analyze_ablations.py [options]

Options:
    --logs_dir DIR          Directory containing log files (default: logs/)
    --runs_dir DIR          Directory containing run results (default: runs/)
    --output_dir DIR        Directory to save analysis (default: ablation_analysis/)
    --plot                  Generate comparison plots
    --wandb_project PROJECT W&B project name to fetch results from
"""

import argparse
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

def parse_log_file(log_path: str) -> Dict:
    """Parse a training log file to extract metrics."""
    metrics = {
        'steps': [],
        'return_ext': [],
        'int_dyn': [],
        'int_hdp': [],
        'int_trans': [],
        'int_rnd': [],
        'eval_return_mean': [],
        'eval_return_std': []
    }
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract metrics if present
                    if 'steps' in data:
                        metrics['steps'].append(data['steps'])
                    if 'return/mean_ext' in data:
                        metrics['return_ext'].append(data['return/mean_ext'])
                    if 'int/dyn_mean' in data:
                        metrics['int_dyn'].append(data['int/dyn_mean'])
                    if 'int/hdp_mean' in data:
                        metrics['int_hdp'].append(data['int/hdp_mean'])
                    if 'int/trans_mean' in data:
                        metrics['int_trans'].append(data['int/trans_mean'])
                    if 'int/rnd_mean' in data:
                        metrics['int_rnd'].append(data['int/rnd_mean'])
                    if 'eval/return_mean' in data:
                        metrics['eval_return_mean'].append(data['eval/return_mean'])
                    if 'eval/return_std' in data:
                        metrics['eval_return_std'].append(data['eval/return_std'])
                        
                except json.JSONDecodeError:
                    continue
                    
    except FileNotFoundError:
        print(f"⚠️  Log file not found: {log_path}")
    
    return metrics

def extract_run_info(filename: str) -> Optional[Dict]:
    """Extract run information from filename."""
    # Pattern: rl_MODE_TIMESTAMP.log
    pattern = r'rl_([^_]+)_(\d{8}_\d{6})\.log'
    match = re.match(pattern, filename)
    
    if match:
        return {
            'mode': match.group(1),
            'timestamp': match.group(2)
        }
    return None

def analyze_logs_directory(logs_dir: str) -> pd.DataFrame:
    """Analyze all log files in a directory."""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"❌ Logs directory not found: {logs_dir}")
        return pd.DataFrame()
    
    runs_data = []
    
    for log_file in logs_path.glob("rl_*.log"):
        run_info = extract_run_info(log_file.name)
        if not run_info:
            continue
        
        print(f"📊 Analyzing: {log_file.name}")
        metrics = parse_log_file(str(log_file))
        
        if not metrics['steps']:
            print(f"⚠️  No metrics found in {log_file.name}")
            continue
        
        # Calculate summary statistics
        final_ext_return = metrics['return_ext'][-1] if metrics['return_ext'] else 0
        mean_ext_return = np.mean(metrics['return_ext']) if metrics['return_ext'] else 0
        final_eval_return = metrics['eval_return_mean'][-1] if metrics['eval_return_mean'] else 0
        mean_eval_return = np.mean(metrics['eval_return_mean']) if metrics['eval_return_mean'] else 0
        
        # Calculate intrinsic reward contributions
        mean_int_dyn = np.mean(metrics['int_dyn']) if metrics['int_dyn'] else 0
        mean_int_hdp = np.mean(metrics['int_hdp']) if metrics['int_hdp'] else 0
        mean_int_trans = np.mean(metrics['int_trans']) if metrics['int_trans'] else 0
        mean_int_rnd = np.mean(metrics['int_rnd']) if metrics['int_rnd'] else 0
        
        total_steps = metrics['steps'][-1] if metrics['steps'] else 0
        
        runs_data.append({
            'mode': run_info['mode'],
            'timestamp': run_info['timestamp'],
            'log_file': log_file.name,
            'total_steps': total_steps,
            'final_ext_return': final_ext_return,
            'mean_ext_return': mean_ext_return,
            'final_eval_return': final_eval_return,
            'mean_eval_return': mean_eval_return,
            'mean_int_dyn': mean_int_dyn,
            'mean_int_hdp': mean_int_hdp,
            'mean_int_trans': mean_int_trans,
            'mean_int_rnd': mean_int_rnd,
            'total_intrinsic': mean_int_dyn + mean_int_hdp + mean_int_trans + mean_int_rnd
        })
    
    return pd.DataFrame(runs_data)

def generate_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Generate comparison plots for ablation study results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Final evaluation returns by mode
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='mode', y='final_eval_return')
    plt.title('Final Evaluation Returns by Ablation Mode')
    plt.xlabel('Ablation Mode')
    plt.ylabel('Final Evaluation Return')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'final_eval_returns_by_mode.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean training returns by mode
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='mode', y='mean_ext_return')
    plt.title('Mean Training Returns by Ablation Mode')
    plt.xlabel('Ablation Mode')
    plt.ylabel('Mean External Return')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'mean_training_returns_by_mode.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Intrinsic reward contributions
    intrinsic_cols = ['mean_int_dyn', 'mean_int_hdp', 'mean_int_trans', 'mean_int_rnd']
    intrinsic_data = df.groupby('mode')[intrinsic_cols].mean()
    
    plt.figure(figsize=(14, 8))
    intrinsic_data.plot(kind='bar', stacked=True)
    plt.title('Mean Intrinsic Reward Contributions by Mode')
    plt.xlabel('Ablation Mode')
    plt.ylabel('Mean Intrinsic Reward')
    plt.legend(['Dynamics', 'Skill Entropy', 'Transition Novelty', 'RND'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'intrinsic_rewards_by_mode.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance vs intrinsic reward total
    if len(df) > 1:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['total_intrinsic'], df['final_eval_return'], 
                   c=df['mode'].astype('category').cat.codes, cmap='tab10', s=100)
        plt.xlabel('Total Intrinsic Reward')
        plt.ylabel('Final Evaluation Return')
        plt.title('Performance vs Total Intrinsic Reward')
        
        # Add mode labels
        for i, row in df.iterrows():
            plt.annotate(row['mode'], (row['total_intrinsic'], row['final_eval_return']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_vs_intrinsic.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📈 Plots saved to: {output_path}")

def print_summary_table(df: pd.DataFrame):
    """Print a summary table of results."""
    if df.empty:
        print("❌ No data to summarize")
        return
    
    print(f"\n📊 ABLATION STUDY RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Group by mode and calculate statistics
    summary = df.groupby('mode').agg({
        'final_eval_return': ['count', 'mean', 'std'],
        'mean_ext_return': ['mean', 'std'],
        'total_intrinsic': ['mean', 'std'],
        'total_steps': 'mean'
    }).round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    print(summary)
    
    # Find best performing mode
    best_mode = df.loc[df['final_eval_return'].idxmax()]
    print(f"\n🏆 Best performing run:")
    print(f"   Mode: {best_mode['mode']}")
    print(f"   Final eval return: {best_mode['final_eval_return']:.3f}")
    print(f"   Log file: {best_mode['log_file']}")
    
    # Compare VAE+HMM vs VAE-only if both present
    if 'baseline' in df['mode'].values and 'no_hmm' in df['mode'].values:
        baseline_mean = df[df['mode'] == 'baseline']['final_eval_return'].mean()
        no_hmm_mean = df[df['mode'] == 'no_hmm']['final_eval_return'].mean()
        improvement = ((baseline_mean - no_hmm_mean) / abs(no_hmm_mean)) * 100
        
        print(f"\n🔍 VAE+HMM vs VAE-only comparison:")
        print(f"   VAE+HMM (baseline): {baseline_mean:.3f}")
        print(f"   VAE-only (no_hmm): {no_hmm_mean:.3f}")
        print(f"   Improvement: {improvement:+.1f}%")
    
    # Compare curiosity vs RND if both present
    if 'baseline' in df['mode'].values and 'rnd' in df['mode'].values:
        curiosity_mean = df[df['mode'] == 'baseline']['final_eval_return'].mean()
        rnd_mean = df[df['mode'] == 'rnd']['final_eval_return'].mean()
        improvement = ((curiosity_mean - rnd_mean) / abs(rnd_mean)) * 100
        
        print(f"\n🧠 Curiosity vs RND comparison:")
        print(f"   Curiosity (baseline): {curiosity_mean:.3f}")
        print(f"   RND: {rnd_mean:.3f}")
        print(f"   Improvement: {improvement:+.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Analyze SequentialSkillRL ablation study results")
    parser.add_argument("--logs_dir", default="logs", help="Directory containing log files")
    parser.add_argument("--runs_dir", default="runs", help="Directory containing run results")
    parser.add_argument("--output_dir", default="ablation_analysis", help="Output directory for analysis")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--wandb_project", help="W&B project name to fetch results from")
    
    args = parser.parse_args()
    
    print(f"🔍 Analyzing ablation study results...")
    print(f"📁 Logs directory: {args.logs_dir}")
    
    # Analyze log files
    df = analyze_logs_directory(args.logs_dir)
    
    if df.empty:
        print("❌ No ablation study results found")
        return
    
    print(f"✅ Found {len(df)} ablation runs across {df['mode'].nunique()} modes")
    
    # Print summary
    print_summary_table(df)
    
    # Save detailed results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path / 'ablation_results.csv', index=False)
    print(f"💾 Detailed results saved to: {output_path / 'ablation_results.csv'}")
    
    # Generate plots if requested
    if args.plot:
        try:
            generate_comparison_plots(df, args.output_dir)
        except ImportError:
            print("⚠️  Plotting libraries not available. Install matplotlib and seaborn to generate plots.")
        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")
    
    print(f"\n🎉 Analysis completed!")

if __name__ == "__main__":
    main()
