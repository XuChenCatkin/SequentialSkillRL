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
        'curiosity_efficiency': [],
        'eval_return_mean': [],
        'eval_return_std': [],
        'eval_success_rate': [],
        'eval_ep_len_mean': [],
        'eval_coverage_pos_mean': [],
        'eval_skill_boundary_mass_rate': [],
        'eval_skill_boundary_bool_rate': [],
        'eval_skill_entropy_mean': [],
        'eval_used_skills_mean': [],
        'eval_effective_K_mean': [],
        'vae_total_loss': [],
        'vae_raw_loss': [],
        'vae_mi_beta': [],
        'vae_tc_beta': [],
        'vae_dw_beta': []
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
                    if 'curiosity/efficiency' in data:
                        metrics['curiosity_efficiency'].append(data['curiosity/efficiency'])
                    if 'eval/return_mean' in data:
                        metrics['eval_return_mean'].append(data['eval/return_mean'])
                    if 'eval/return_std' in data:
                        metrics['eval_return_std'].append(data['eval/return_std'])
                    if 'eval/success_rate' in data:
                        metrics['eval_success_rate'].append(data['eval/success_rate'])
                    if 'eval/ep_len_mean' in data:
                        metrics['eval_ep_len_mean'].append(data['eval/ep_len_mean'])
                    if 'eval/coverage_pos_mean' in data:
                        metrics['eval_coverage_pos_mean'].append(data['eval/coverage_pos_mean'])
                    if 'eval/skill_boundary_mass_rate' in data:
                        metrics['eval_skill_boundary_mass_rate'].append(data['eval/skill_boundary_mass_rate'])
                    if 'eval/skill_boundary_bool_rate' in data:
                        metrics['eval_skill_boundary_bool_rate'].append(data['eval/skill_boundary_bool_rate'])
                    if 'eval/skill_entropy_mean' in data:
                        metrics['eval_skill_entropy_mean'].append(data['eval/skill_entropy_mean'])
                    if 'eval/used_skills_mean' in data:
                        metrics['eval_used_skills_mean'].append(data['eval/used_skills_mean'])
                    if 'eval/effective_K_mean' in data:
                        metrics['eval_effective_K_mean'].append(data['eval/effective_K_mean'])
                    if 'vae/total_loss' in data:
                        metrics['vae_total_loss'].append(data['vae/total_loss'])
                    if 'vae/raw_loss' in data:
                        metrics['vae_raw_loss'].append(data['vae/raw_loss'])
                    if 'vae/mi_beta' in data:
                        metrics['vae_mi_beta'].append(data['vae/mi_beta'])
                    if 'vae/tc_beta' in data:
                        metrics['vae_tc_beta'].append(data['vae/tc_beta'])
                    if 'vae/dw_beta' in data:
                        metrics['vae_dw_beta'].append(data['vae/dw_beta'])
                        
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
        final_success_rate = metrics['eval_success_rate'][-1] if metrics['eval_success_rate'] else 0
        mean_curiosity_efficiency = np.mean(metrics['curiosity_efficiency']) if metrics['curiosity_efficiency'] else 0
        
        # Calculate intrinsic reward contributions
        mean_int_dyn = np.mean(metrics['int_dyn']) if metrics['int_dyn'] else 0
        mean_int_hdp = np.mean(metrics['int_hdp']) if metrics['int_hdp'] else 0
        mean_int_trans = np.mean(metrics['int_trans']) if metrics['int_trans'] else 0
        mean_int_rnd = np.mean(metrics['int_rnd']) if metrics['int_rnd'] else 0
        
        # Calculate exploration and skill metrics
        mean_coverage = np.mean(metrics['eval_coverage_pos_mean']) if metrics['eval_coverage_pos_mean'] else 0
        mean_ep_length = np.mean(metrics['eval_ep_len_mean']) if metrics['eval_ep_len_mean'] else 0
        mean_skill_entropy = np.mean(metrics['eval_skill_entropy_mean']) if metrics['eval_skill_entropy_mean'] else 0
        mean_used_skills = np.mean(metrics['eval_used_skills_mean']) if metrics['eval_used_skills_mean'] else 0
        mean_effective_K = np.mean(metrics['eval_effective_K_mean']) if metrics['eval_effective_K_mean'] else 0
        mean_boundary_mass = np.mean(metrics['eval_skill_boundary_mass_rate']) if metrics['eval_skill_boundary_mass_rate'] else 0
        
        # Calculate VAE training metrics
        mean_vae_loss = np.mean(metrics['vae_total_loss']) if metrics['vae_total_loss'] else 0
        
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
            'final_success_rate': final_success_rate,
            'mean_curiosity_efficiency': mean_curiosity_efficiency,
            'mean_int_dyn': mean_int_dyn,
            'mean_int_hdp': mean_int_hdp,
            'mean_int_trans': mean_int_trans,
            'mean_int_rnd': mean_int_rnd,
            'total_intrinsic': mean_int_dyn + mean_int_hdp + mean_int_trans + mean_int_rnd,
            'mean_coverage': mean_coverage,
            'mean_ep_length': mean_ep_length,
            'mean_skill_entropy': mean_skill_entropy,
            'mean_used_skills': mean_used_skills,
            'mean_effective_K': mean_effective_K,
            'mean_boundary_mass': mean_boundary_mass,
            'mean_vae_loss': mean_vae_loss
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
    
    # 2. Success rate comparison
    if 'final_success_rate' in df.columns and df['final_success_rate'].sum() > 0:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='mode', y='final_success_rate')
        plt.title('Success Rate by Ablation Mode')
        plt.xlabel('Ablation Mode')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'success_rate_by_mode.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Exploration metrics (coverage and episode length)
    if 'mean_coverage' in df.columns and df['mean_coverage'].sum() > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.boxplot(data=df, x='mode', y='mean_coverage', ax=ax1)
        ax1.set_title('Spatial Coverage by Mode')
        ax1.set_xlabel('Ablation Mode')
        ax1.set_ylabel('Mean Coverage (unique positions)')
        ax1.tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=df, x='mode', y='mean_ep_length', ax=ax2)
        ax2.set_title('Episode Length by Mode')
        ax2.set_xlabel('Ablation Mode')
        ax2.set_ylabel('Mean Episode Length')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'exploration_metrics_by_mode.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Skill-based diagnostics (for HMM-enabled modes)
    skill_cols = ['mean_skill_entropy', 'mean_used_skills', 'mean_effective_K', 'mean_boundary_mass']
    skill_data = df[df['mode'] != 'no_hmm'][skill_cols] if 'no_hmm' in df['mode'].values else df[skill_cols]
    
    if not skill_data.empty and skill_data.sum().sum() > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        if 'mean_skill_entropy' in skill_data.columns:
            sns.boxplot(data=df[df['mode'] != 'no_hmm'] if 'no_hmm' in df['mode'].values else df, 
                       x='mode', y='mean_skill_entropy', ax=axes[0,0])
            axes[0,0].set_title('Skill Entropy by Mode')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        if 'mean_used_skills' in skill_data.columns:
            sns.boxplot(data=df[df['mode'] != 'no_hmm'] if 'no_hmm' in df['mode'].values else df, 
                       x='mode', y='mean_used_skills', ax=axes[0,1])
            axes[0,1].set_title('Used Skills by Mode')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        if 'mean_effective_K' in skill_data.columns:
            sns.boxplot(data=df[df['mode'] != 'no_hmm'] if 'no_hmm' in df['mode'].values else df, 
                       x='mode', y='mean_effective_K', ax=axes[1,0])
            axes[1,0].set_title('Effective K by Mode')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        if 'mean_boundary_mass' in skill_data.columns:
            sns.boxplot(data=df[df['mode'] != 'no_hmm'] if 'no_hmm' in df['mode'].values else df, 
                       x='mode', y='mean_boundary_mass', ax=axes[1,1])
            axes[1,1].set_title('Skill Boundary Mass by Mode')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'skill_diagnostics_by_mode.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Curiosity efficiency
    if 'mean_curiosity_efficiency' in df.columns and df['mean_curiosity_efficiency'].sum() > 0:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='mode', y='mean_curiosity_efficiency')
        plt.title('Curiosity Efficiency by Ablation Mode')
        plt.xlabel('Ablation Mode')
        plt.ylabel('Curiosity Efficiency (Ext/Int Ratio)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'curiosity_efficiency_by_mode.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. Mean training returns by mode
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='mode', y='mean_ext_return')
    plt.title('Mean Training Returns by Ablation Mode')
    plt.xlabel('Ablation Mode')
    plt.ylabel('Mean External Return')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'mean_training_returns_by_mode.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Intrinsic reward contributions
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
    
    # 8. Performance vs intrinsic reward total
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
    
    # 9. VAE training loss (if available)
    if 'mean_vae_loss' in df.columns and df['mean_vae_loss'].sum() > 0:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='mode', y='mean_vae_loss')
        plt.title('VAE Training Loss by Ablation Mode')
        plt.xlabel('Ablation Mode')
        plt.ylabel('Mean VAE Loss')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'vae_loss_by_mode.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📈 Plots saved to: {output_path}")

def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate a comprehensive summary report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'ablation_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 25 + "ABLATION STUDY SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Basic statistics
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Unique modes: {df['mode'].nunique()}\n")
        f.write(f"Modes tested: {', '.join(df['mode'].unique())}\n\n")
        
        # Performance Summary
        f.write("-" * 40 + " PERFORMANCE " + "-" * 29 + "\n\n")
        perf_summary = df.groupby('mode').agg({
            'final_eval_return': ['mean', 'std', 'max', 'min'],
            'mean_ext_return': ['mean', 'std']
        }).round(4)
        
        f.write("Final Evaluation Returns by Mode:\n")
        for mode in df['mode'].unique():
            mode_data = df[df['mode'] == mode]['final_eval_return']
            f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f} "
                   f"(min: {mode_data.min():6.4f}, max: {mode_data.max():6.4f})\n")
        f.write("\n")
        
        # Success rates (if available)
        if 'final_success_rate' in df.columns and df['final_success_rate'].sum() > 0:
            f.write("Success Rates by Mode:\n")
            for mode in df['mode'].unique():
                mode_data = df[df['mode'] == mode]['final_success_rate']
                f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
            f.write("\n")
        
        # Exploration metrics
        f.write("-" * 37 + " EXPLORATION " + "-" * 32 + "\n\n")
        if 'mean_coverage' in df.columns and df['mean_coverage'].sum() > 0:
            f.write("Spatial Coverage by Mode:\n")
            for mode in df['mode'].unique():
                mode_data = df[df['mode'] == mode]['mean_coverage']
                f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
            f.write("\n")
        
        if 'mean_ep_length' in df.columns and df['mean_ep_length'].sum() > 0:
            f.write("Episode Length by Mode:\n")
            for mode in df['mode'].unique():
                mode_data = df[df['mode'] == mode]['mean_ep_length']
                f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
            f.write("\n")
        
        # Skill-based diagnostics (for HMM modes)
        skill_modes = df[df['mode'] != 'no_hmm'] if 'no_hmm' in df['mode'].values else df
        if not skill_modes.empty and 'mean_skill_entropy' in df.columns:
            f.write("-" * 35 + " SKILL DIAGNOSTICS " + "-" * 27 + "\n\n")
            
            if 'mean_skill_entropy' in df.columns and df['mean_skill_entropy'].sum() > 0:
                f.write("Skill Entropy by Mode:\n")
                for mode in skill_modes['mode'].unique():
                    mode_data = skill_modes[skill_modes['mode'] == mode]['mean_skill_entropy']
                    if not mode_data.empty:
                        f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
                f.write("\n")
            
            if 'mean_used_skills' in df.columns and df['mean_used_skills'].sum() > 0:
                f.write("Used Skills by Mode:\n")
                for mode in skill_modes['mode'].unique():
                    mode_data = skill_modes[skill_modes['mode'] == mode]['mean_used_skills']
                    if not mode_data.empty:
                        f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
                f.write("\n")
            
            if 'mean_effective_K' in df.columns and df['mean_effective_K'].sum() > 0:
                f.write("Effective K by Mode:\n")
                for mode in skill_modes['mode'].unique():
                    mode_data = skill_modes[skill_modes['mode'] == mode]['mean_effective_K']
                    if not mode_data.empty:
                        f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
                f.write("\n")
            
            if 'mean_boundary_mass' in df.columns and df['mean_boundary_mass'].sum() > 0:
                f.write("Skill Boundary Mass by Mode:\n")
                for mode in skill_modes['mode'].unique():
                    mode_data = skill_modes[skill_modes['mode'] == mode]['mean_boundary_mass']
                    if not mode_data.empty:
                        f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
                f.write("\n")
        
        # Curiosity efficiency
        if 'mean_curiosity_efficiency' in df.columns and df['mean_curiosity_efficiency'].sum() > 0:
            f.write("-" * 33 + " CURIOSITY EFFICIENCY " + "-" * 26 + "\n\n")
            for mode in df['mode'].unique():
                mode_data = df[df['mode'] == mode]['mean_curiosity_efficiency']
                f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
            f.write("\n")
        
        # Intrinsic Rewards
        f.write("-" * 35 + " INTRINSIC REWARDS " + "-" * 27 + "\n\n")
        intrinsic_cols = ['mean_int_dyn', 'mean_int_hdp', 'mean_int_trans', 'mean_int_rnd']
        for col in intrinsic_cols:
            if col in df.columns and df[col].sum() > 0:
                reward_type = col.split('_')[-1].upper()
                f.write(f"{reward_type} Intrinsic Rewards by Mode:\n")
                for mode in df['mode'].unique():
                    mode_data = df[df['mode'] == mode][col]
                    f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
                f.write("\n")
        
        if 'total_intrinsic' in df.columns:
            f.write("Total Intrinsic Rewards by Mode:\n")
            for mode in df['mode'].unique():
                mode_data = df[df['mode'] == mode]['total_intrinsic']
                f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
            f.write("\n")
        
        # VAE Training Loss
        if 'mean_vae_loss' in df.columns and df['mean_vae_loss'].sum() > 0:
            f.write("-" * 37 + " VAE TRAINING " + "-" * 32 + "\n\n")
            f.write("VAE Training Loss by Mode:\n")
            for mode in df['mode'].unique():
                mode_data = df[df['mode'] == mode]['mean_vae_loss']
                f.write(f"  {mode:15}: {mode_data.mean():8.4f} ± {mode_data.std():6.4f}\n")
            f.write("\n")
        
        # Best performing mode
        f.write("-" * 35 + " RANKING SUMMARY " + "-" * 28 + "\n\n")
        best_mode = df.loc[df['final_eval_return'].idxmax(), 'mode']
        best_return = df['final_eval_return'].max()
        f.write(f"Best performing mode: {best_mode} (return: {best_return:.4f})\n")
        
        # Mode ranking by mean performance
        mode_performance = df.groupby('mode')['final_eval_return'].mean().sort_values(ascending=False)
        f.write("\nMode ranking by mean performance:\n")
        for rank, (mode, mean_return) in enumerate(mode_performance.items(), 1):
            f.write(f"  {rank}. {mode:15}: {mean_return:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"📊 Summary report saved to: {output_path / 'ablation_summary.txt'}")

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
    
    # Generate summary report
    generate_summary_report(df, args.output_dir)
    
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
