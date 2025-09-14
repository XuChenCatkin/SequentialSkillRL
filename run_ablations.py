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
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from typing import List

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

def run_ablation(mode: str, env: str, steps: int, seed: int, wandb: bool, no_upload: bool, dry_run: bool = False) -> bool:
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
    
    print(f"\n🚀 Running ablation: {mode} (seed={seed})")
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
    
    print(f"🔬 Starting ablation study batch")
    print(f"   Environment: {args.env}")
    print(f"   Steps per run: {args.steps:,}")
    print(f"   Seeds: {seeds}")
    print(f"   Modes: {modes}")
    print(f"   W&B logging: {args.wandb}")
    print(f"   HF upload: {not args.no_upload}")
    print(f"   Total runs: {len(modes) * len(seeds)}")
    print(f"   Estimated time: {len(modes) * len(seeds) * args.steps / 1000000 * 30:.1f} hours (rough estimate)")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - Commands will be printed but not executed")
    
    if not args.dry_run:
        response = input(f"\n❓ Continue with ablation study? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("🛑 Aborted")
            sys.exit(0)
    
    # Run ablations
    results = []
    total_runs = len(modes) * len(seeds)
    current_run = 0
    
    start_time = time.time()
    
    for mode in modes:
        for seed in seeds:
            current_run += 1
            print(f"\n{'='*80}")
            print(f"📊 Progress: {current_run}/{total_runs} | Mode: {mode} | Seed: {seed}")
            print(f"{'='*80}")
            
            success = run_ablation(
                mode=mode,
                env=args.env,
                steps=args.steps,
                seed=seed,
                wandb=args.wandb,
                no_upload=args.no_upload,
                dry_run=args.dry_run
            )
            
            results.append({
                'mode': mode,
                'seed': seed,
                'success': success
            })
            
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
        for r in results:
            if r['success']:
                if r['mode'] not in by_mode:
                    by_mode[r['mode']] = []
                by_mode[r['mode']].append(r['seed'])
        
        for mode, seeds_list in by_mode.items():
            print(f"   - {mode}: seeds {seeds_list}")
    
    print(f"\n🎉 Ablation study batch completed!")
    
    if not args.dry_run and args.wandb:
        print(f"📊 Check W&B project 'SequentialSkillRL-Ablations' for results")
    
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
