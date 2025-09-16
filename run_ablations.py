#!/usr/bin/env python3
"""
Automated ablation study runner for SequentialSkillRL.

This script runs multiple ablation studies in sequence to compare different
configurations of VAE+HMM+PPO training.

Usage:
    python run_ablations.py [options]

Options:
    --env ENV_NAME          Environment to use (default: MiniHack-Room-5x5-v0)
    --steps N               Total steps per run (default: 1000000)
    --seeds SEED1,SEED2,... Random seeds (default: 42,123,456)
    --wandb                 Enable W&B logging
    --no_upload             Disable HuggingFace uploads
    --modes MODE1,MODE2,... Specific modes to run (default: all)
    --dry_run               Print commands without running
    --resume_from PREFIX    Resume ablation studies from existing checkpoints
                            (looks for repos like: PREFIX-baseline, PREFIX-no_hmm, etc.)
    --resume_local_dir DIR  Resume from local checkpoint directory
    --checkpoint_every N    Steps between checkpoints (default: same as --steps)
"""

import argparse
import subprocess
import sys
import time
import os
import json
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path

# Default ablation modes
DEFAULT_MODES = [
    "baseline",              # VAE+HMM+PPO with full curiosity
    "no_hmm",               # VAE+PPO (no HMM)
    "rnd",                  # VAE+HMM+PPO with RND
    "no_intrinsic",         # VAE+HMM+PPO with no intrinsic rewards
    "curiosity_dyn_only",   # VAE+HMM+PPO with dynamics only
    "curiosity_skill_only", # VAE+HMM+PPO with skill entropy only
    "curiosity_trans_only", # VAE+HMM+PPO with transition novelty only
]

def save_progress(progress_file: str, results: List[Dict], current_run: int, total_runs: int):
    """Save current progress to a JSON file."""
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "current_run": current_run,
        "total_runs": total_runs,
        "results": results
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def load_progress(progress_file: str) -> Optional[Dict]:
    """Load progress from a JSON file."""
    if not os.path.exists(progress_file):
        return None
    
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load progress file: {e}")
        return None

def find_resume_repos(repo_prefix: str, modes: List[str]) -> Dict[str, str]:
    """Find existing HuggingFace repositories for resuming."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("⚠️  HuggingFace Hub not available for resume repo detection")
        return {}
    
    resume_repos = {}
    try:
        api = HfApi()
        # For each mode, check if a repo exists
        for mode in modes:
            repo_name = f"{repo_prefix}-{mode}"
            try:
                # Try to get repo info to see if it exists
                repo_info = api.repo_info(repo_name)
                if repo_info:
                    resume_repos[mode] = repo_name
                    print(f"✅ Found resume repo for {mode}: {repo_name}")
            except Exception:
                # Repo doesn't exist or isn't accessible
                print(f"ℹ️  No resume repo found for {mode}: {repo_name}")
                
    except Exception as e:
        print(f"⚠️  Failed to check for resume repos: {e}")
    
    return resume_repos

def find_local_checkpoints(checkpoint_dir: str, modes: List[str]) -> Dict[str, str]:
    """Find local checkpoint files for resuming."""
    resume_checkpoints = {}
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint directory doesn't exist: {checkpoint_dir}")
        return resume_checkpoints
    
    for mode in modes:
        # Look for checkpoint files matching the mode
        possible_patterns = [
            f"{mode}*.pth",
            f"*{mode}*.pth",
            f"checkpoint*{mode}*.pth",
            f"ppo_policy_{mode}.pth"
        ]
        
        for pattern in possible_patterns:
            matching_files = list(checkpoint_path.glob(pattern))
            if matching_files:
                # Use the most recent file
                latest_file = max(matching_files, key=os.path.getctime)
                resume_checkpoints[mode] = str(latest_file)
                print(f"✅ Found checkpoint for {mode}: {latest_file}")
                break
        
        if mode not in resume_checkpoints:
            print(f"ℹ️  No checkpoint found for {mode}")
    
    return resume_checkpoints

def run_ablation(
    mode: str, 
    env: str, 
    steps: int, 
    seed: int, 
    wandb: bool, 
    no_upload: bool, 
    resume_repo: Optional[str] = None,
    resume_local: Optional[str] = None,
    dry_run: bool = False
) -> bool:
    """Run a single ablation study."""
    cmd = [
        "python", "main.py", "rl", mode,
        "--env", env,
        "--steps", str(steps),
        "--seed", str(seed)
    ]
    
    if wandb:
        cmd.append("--wandb")
    if no_upload:
        cmd.append("--no_upload")
    
    # Add resume options if provided
    if resume_repo:
        cmd.extend(["--resume", resume_repo])
        print(f"\n🔄 Resuming ablation: {mode} (seed={seed}) from repo: {resume_repo}")
    elif resume_local:
        cmd.extend(["--resume_local", resume_local])
        print(f"\n� Resuming ablation: {mode} (seed={seed}) from local: {resume_local}")
    else:
        print(f"\n🚀 Starting fresh ablation: {mode} (seed={seed})")
    
    print(f"📝 Command: {' '.join(cmd)}")
    
    if dry_run:
        print("🔍 [DRY RUN] Would execute the above command")
        return True
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, cwd=".")
        duration = time.time() - start_time
        
        print(f"✅ Completed {mode} (seed={seed}) in {duration:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed {mode} (seed={seed}): {e}")
        return False
    except KeyboardInterrupt:
        print(f"🛑 Interrupted {mode} (seed={seed})")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run SequentialSkillRL ablation studies")
    parser.add_argument("--env", default="MiniHack-Room-5x5-v0", help="Environment name")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total steps per run")
    parser.add_argument("--seeds", default="42,123,456", help="Comma-separated random seeds")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--no_upload", action="store_true", help="Disable HuggingFace uploads")
    parser.add_argument("--modes", help="Comma-separated modes to run (default: all)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--resume_from", help="Resume from HuggingFace repo prefix (e.g., 'username/project')")
    parser.add_argument("--resume_local_dir", help="Resume from local checkpoint directory")
    parser.add_argument("--progress_file", default="ablation_progress.json", help="Progress tracking file")
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    # Parse modes
    if args.modes:
        modes = [m.strip() for m in args.modes.split(",")]
        # Validate modes
        invalid_modes = [m for m in modes if m not in DEFAULT_MODES]
        if invalid_modes:
            print(f"❌ Invalid modes: {invalid_modes}")
            print(f"   Valid modes: {', '.join(DEFAULT_MODES)}")
            sys.exit(1)
    else:
        modes = DEFAULT_MODES
    
    # Check for existing progress
    existing_progress = load_progress(args.progress_file)
    if existing_progress and not args.dry_run:
        print(f"📋 Found existing progress file: {args.progress_file}")
        print(f"   Last run: {existing_progress['current_run']}/{existing_progress['total_runs']}")
        print(f"   Timestamp: {existing_progress['timestamp']}")
        
        response = input(f"❓ Continue from existing progress? [y/N]: ")
        if response.lower() in ['y', 'yes']:
            # Resume from existing progress
            results = existing_progress['results']
            completed_runs = set((r['mode'], r['seed']) for r in results if r['success'])
            print(f"📊 Skipping {len(completed_runs)} already completed runs")
        else:
            existing_progress = None
            results = []
    else:
        results = []
    
    # Determine resume strategy
    resume_repos = {}
    resume_checkpoints = {}
    
    if args.resume_from:
        print(f"🔍 Checking for resume repositories with prefix: {args.resume_from}")
        resume_repos = find_resume_repos(args.resume_from, modes)
        
    if args.resume_local_dir:
        print(f"🔍 Checking for local checkpoints in: {args.resume_local_dir}")
        resume_checkpoints = find_local_checkpoints(args.resume_local_dir, modes)
    
    print(f"🔬 Starting ablation study batch")
    print(f"   Environment: {args.env}")
    print(f"   Steps per run: {args.steps:,}")
    print(f"   Seeds: {seeds}")
    print(f"   Modes: {modes}")
    print(f"   W&B logging: {args.wandb}")
    print(f"   HF upload: {not args.no_upload}")
    print(f"   Total runs: {len(modes) * len(seeds)}")
    print(f"   Progress file: {args.progress_file}")
    
    # Resume information
    if resume_repos:
        print(f"   Resume repos: {len(resume_repos)} modes")
        for mode, repo in resume_repos.items():
            print(f"     - {mode}: {repo}")
    
    if resume_checkpoints:
        print(f"   Resume checkpoints: {len(resume_checkpoints)} modes")
        for mode, checkpoint in resume_checkpoints.items():
            print(f"     - {mode}: {checkpoint}")
    
    if existing_progress:
        completed_count = sum(1 for r in existing_progress['results'] if r['success'])
        print(f"   Existing progress: {completed_count} completed runs")
    
    print(f"   Estimated time: {len(modes) * len(seeds) * args.steps / 1000000 * 30:.1f} hours (rough estimate)")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - Commands will be printed but not executed")
    
    if not args.dry_run:
        response = input(f"\n❓ Continue with ablation study? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("🛑 Aborted")
            sys.exit(0)
    
    # Run ablations
    total_runs = len(modes) * len(seeds)
    current_run = 0
    
    start_time = time.time()
    
    for mode in modes:
        for seed in seeds:
            current_run += 1
            
            # Skip if already completed (from existing progress)
            if existing_progress:
                completed_runs = set((r['mode'], r['seed']) for r in existing_progress['results'] if r['success'])
                if (mode, seed) in completed_runs:
                    print(f"⏭️  Skipping completed run: {mode} (seed={seed})")
                    results.append({'mode': mode, 'seed': seed, 'success': True})
                    continue
            
            print(f"\n{'='*80}")
            print(f"📊 Progress: {current_run}/{total_runs} | Mode: {mode} | Seed: {seed}")
            print(f"{'='*80}")
            
            # Determine resume strategy for this run
            resume_repo = resume_repos.get(mode)
            resume_local = resume_checkpoints.get(mode)
            
            success = run_ablation(
                mode=mode,
                env=args.env,
                steps=args.steps,
                seed=seed,
                wandb=args.wandb,
                no_upload=args.no_upload,
                resume_repo=resume_repo,
                resume_local=resume_local,
                dry_run=args.dry_run
            )
            
            results.append({
                'mode': mode,
                'seed': seed,
                'success': success,
                'timestamp': datetime.now().isoformat(),
                'resume_repo': resume_repo,
                'resume_local': resume_local
            })
            
            # Save progress after each run
            if not args.dry_run:
                save_progress(args.progress_file, results, current_run, total_runs)
            
            if not success and not args.dry_run:
                response = input(f"❓ Continue with remaining runs? [y/N]: ")
                if response.lower() not in ['y', 'yes']:
                    print("🛑 Aborted remaining runs")
                    break
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\n{'='*80}")
    print(f"📈 ABLATION STUDY SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print(f"\n❌ Failed runs:")
        for r in results:
            if not r['success']:
                print(f"   - {r['mode']} (seed={r['seed']})")
    
    if successful > 0:
        print(f"\n✅ Successful runs:")
        by_mode = {}
        resumed_count = 0
        for r in results:
            if r['success']:
                if r['mode'] not in by_mode:
                    by_mode[r['mode']] = []
                by_mode[r['mode']].append(r['seed'])
                
                # Count resumed runs
                if r.get('resume_repo') or r.get('resume_local'):
                    resumed_count += 1
        
        for mode, seeds_list in by_mode.items():
            print(f"   - {mode}: seeds {seeds_list}")
        
        if resumed_count > 0:
            print(f"\n🔄 Resumed runs: {resumed_count}/{successful}")
    
    # Clean up progress file on successful completion
    if not args.dry_run and failed == 0 and os.path.exists(args.progress_file):
        try:
            os.remove(args.progress_file)
            print(f"🧹 Cleaned up progress file: {args.progress_file}")
        except Exception:
            pass
    
    print(f"\n🎉 Ablation study batch completed!")
    
    if not args.dry_run and args.wandb:
        print(f"📊 Check W&B project 'SequentialSkillRL-Ablations' for results")
    
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
