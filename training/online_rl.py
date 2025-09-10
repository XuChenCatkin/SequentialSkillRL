"""
Online PPO training for NetHack using pre-trained VAE+HMM models from HuggingFace.

This module provides an online RL training pipeline that:
1. Loads pre-trained VAE and HMM models from HuggingFace
2. Sets up MiniHack environments for online training
3. Trains PPO agents with curiosity-driven intrinsic rewards
4. Monitors training progress with W&B and HuggingFace uploads
"""

import os
import sys
import time
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from tqdm import tqdm

import nle
import minihack

# Now try to import PPO components
from rl.ppo import (
    PPOConfig, CuriosityConfig, HMMOnlineConfig, RNDConfig, TrainConfig, PPOTrainer,
    set_seed
)

# Import model components
from src.model import MultiModalHackVAE, VAEConfig
from src.skill_space import StickyHDPHMMVI, StickyHDPHMMParams, NIWPrior

# Import training utilities
from training.training_utils import (
    load_model_from_huggingface, 
    load_hmm_from_huggingface,
    save_checkpoint,
    WANDB_AVAILABLE,
    HF_AVAILABLE
)

# Weights & Biases integration
if WANDB_AVAILABLE:
    import wandb

# HuggingFace integration
if HF_AVAILABLE:
    from huggingface_hub import HfApi, login, create_repo, upload_file

warnings.filterwarnings('ignore')


def train_online_ppo_with_pretrained_models(
    env_name: str = "MiniHack-Quest-Hard-v0",
    repo_id: str = "catid/SequentialSkillRL",
    vae_filename: str = "nethack-vae.pth",
    hmm_filename: str = "hmm_round3.pt",
    total_timesteps: int = 50000,
    learning_rate: float = 5e-4,
    batch_size: int = 32,
    n_epochs: int = 10,
    gamma: float = 0.99,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    use_curiosity: bool = True,
    curiosity_lr: float = 1e-4,
    curiosity_forward_coef: float = 0.2,
    curiosity_inverse_coef: float = 0.8,
    use_rnd: bool = False,
    rnd_lr: float = 1e-4,
    rnd_coef: float = 0.1,
    test_mode: bool = False,
    test_episodes: int = 10,
    project_name: str = "SequentialSkillRL",
    run_name: Optional[str] = None,
    save_freq: int = 1000,
    log_freq: int = 100,
    device: str = "auto",
    seed: Optional[int] = None,
    debug_mode: bool = False,
    upload_to_huggingface: bool = False,
    use_wandb: bool = True,
    upload_model_to_hf: bool = False,
    hf_model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train online PPO agent with pretrained VAE and HMM models.
    
    Args:
        env_name: MiniHack environment name
        repo_id: HuggingFace repository ID for pretrained models
        vae_filename: VAE model filename
        hmm_filename: HMM model filename
        total_timesteps: Total training timesteps
        learning_rate: PPO learning rate
        batch_size: Training batch size
        n_epochs: Number of PPO epochs per update
        gamma: Discount factor
        vf_coef: Value function coefficient
        ent_coef: Entropy coefficient
        max_grad_norm: Maximum gradient norm
        use_curiosity: Enable curiosity-driven exploration
        curiosity_lr: Curiosity module learning rate
        curiosity_forward_coef: Forward model coefficient
        curiosity_inverse_coef: Inverse model coefficient
        use_rnd: Enable Random Network Distillation
        rnd_lr: RND learning rate
        rnd_coef: RND coefficient
        test_mode: Run in test mode (no training)
        test_episodes: Number of test episodes
        project_name: W&B project name
        run_name: W&B run name
        save_freq: Model save frequency
        log_freq: Logging frequency
        device: Device to use ('auto', 'cuda', 'cpu')
        seed: Random seed
        debug_mode: Enable debug logging
        upload_to_huggingface: Upload checkpoints to HuggingFace
        use_wandb: Enable Weights & Biases logging
        upload_model_to_hf: Upload final model to HuggingFace
        hf_model_name: Custom HuggingFace model name
    
    Returns:
        Dictionary with training results and final metrics
    """
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(f"Starting Online PPO Training with Pretrained Models")
    logger.info(f"Environment: {env_name}")
    logger.info(f"Repository: {repo_id}")
    logger.info(f"VAE Model: {vae_filename}")
    logger.info(f"HMM Model: {hmm_filename}")
    logger.info(f"Total Timesteps: {total_timesteps:,}")
    logger.info(f"Test Mode: {test_mode}")
    logger.info("=" * 80)
    
    # Set random seed
    if seed is not None:
        set_seed(seed)
        logger.info(f"🌱 Random seed set to: {seed}")
    
    # Device setup
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"🔧 Using device: {device}")
    
    # Create run name if not provided
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = "test" if test_mode else "train"
        run_name = f"online_ppo_{env_name}_{mode_str}_{timestamp}"
    
    # Initialize W&B
    wandb_run = None
    if WANDB_AVAILABLE and use_wandb and not test_mode:
        try:
            wandb_run = wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "env_name": env_name,
                    "repo_id": repo_id,
                    "vae_filename": vae_filename,
                    "hmm_filename": hmm_filename,
                    "total_timesteps": total_timesteps,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "n_epochs": n_epochs,
                    "gamma": gamma,
                    "vf_coef": vf_coef,
                    "ent_coef": ent_coef,
                    "max_grad_norm": max_grad_norm,
                    "use_curiosity": use_curiosity,
                    "curiosity_lr": curiosity_lr,
                    "curiosity_forward_coef": curiosity_forward_coef,
                    "curiosity_inverse_coef": curiosity_inverse_coef,
                    "use_rnd": use_rnd,
                    "rnd_lr": rnd_lr,
                    "rnd_coef": rnd_coef,
                    "device": str(device),
                    "seed": seed
                }
            )
            logger.info(f"📊 W&B initialized with run name: {run_name}")
        except Exception as e:
            logger.warning(f"⚠️  W&B initialization failed: {e}")
            wandb_run = None
    
    try:
        # Load pretrained VAE
        logger.info("🔄 Loading pretrained VAE model...")
        vae_model, vae_config = load_model_from_huggingface(
            repo_id=repo_id,
            filename=vae_filename,
            device=device
        )
        logger.info(f"✅ VAE model loaded successfully")
        logger.info(f"   - Latent dimension: {vae_config.latent_dim}")
        logger.info(f"   - Architecture: {vae_config.architecture}")
        
        # Load pretrained HMM
        logger.info("🔄 Loading pretrained HMM model...")
        hmm_model = load_hmm_from_huggingface(
            repo_id=repo_id,
            filename=hmm_filename,
            device=device
        )
        logger.info(f"✅ HMM model loaded successfully")
        
        # Create environment
        logger.info(f"🎮 Creating environment: {env_name}")
        env = gym.make(env_name)
        
        # Log environment info
        logger.info(f"   - Observation space: {env.observation_space}")
        logger.info(f"   - Action space: {env.action_space}")
        
        # Configure training
        train_config = TrainConfig(
            total_timesteps=total_timesteps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            save_freq=save_freq,
            log_freq=log_freq
        )
        
        ppo_config = PPOConfig(
            lr=learning_rate,
            gamma=gamma,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm
        )
        
        # Configure curiosity if enabled
        curiosity_config = None
        if use_curiosity:
            curiosity_config = CuriosityConfig(
                lr=curiosity_lr,
                forward_coef=curiosity_forward_coef,
                inverse_coef=curiosity_inverse_coef
            )
            logger.info(f"🧠 Curiosity enabled: forward_coef={curiosity_forward_coef}, inverse_coef={curiosity_inverse_coef}")
        
        # Configure RND if enabled
        rnd_config = None
        if use_rnd:
            rnd_config = RNDConfig(
                lr=rnd_lr,
                coef=rnd_coef
            )
            logger.info(f"🔍 RND enabled: coef={rnd_coef}")
        
        # Configure HMM integration
        hmm_config = HMMOnlineConfig(
            use_hmm_reward=True,
            hmm_reward_scale=0.1,
            update_frequency=100
        )
        
        # Create PPO trainer
        logger.info("🚀 Initializing PPO trainer...")
        trainer = PPOTrainer(
            env=env,
            vae_model=vae_model,
            hmm_model=hmm_model,
            ppo_config=ppo_config,
            curiosity_config=curiosity_config,
            rnd_config=rnd_config,
            hmm_config=hmm_config,
            device=device
        )
        logger.info("✅ PPO trainer initialized successfully")
        
        # Test mode: run evaluation episodes
        if test_mode:
            logger.info(f"🧪 Running {test_episodes} test episodes...")
            test_results = []
            
            for episode in range(test_episodes):
                logger.info(f"Episode {episode + 1}/{test_episodes}")
                
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                
                while not done:
                    # Get action from trained policy
                    with torch.no_grad():
                        action, _, _, _ = trainer.policy.step(obs)
                    
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    episode_length += 1
                
                test_results.append({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'length': episode_length
                })
                
                logger.info(f"   Reward: {episode_reward}, Length: {episode_length}")
            
            # Calculate test statistics
            rewards = [r['reward'] for r in test_results]
            lengths = [r['length'] for r in test_results]
            
            test_stats = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'mean_length': np.mean(lengths),
                'std_length': np.std(lengths)
            }
            
            logger.info("📈 Test Results:")
            logger.info(f"   Mean Reward: {test_stats['mean_reward']:.2f} ± {test_stats['std_reward']:.2f}")
            logger.info(f"   Min/Max Reward: {test_stats['min_reward']:.2f} / {test_stats['max_reward']:.2f}")
            logger.info(f"   Mean Length: {test_stats['mean_length']:.1f} ± {test_stats['std_length']:.1f}")
            
            return {
                'test_results': test_results,
                'test_stats': test_stats,
                'run_name': run_name
            }
        
        # Training mode
        logger.info("🏋️  Starting PPO training...")
        start_time = time.time()
        
        training_results = trainer.train(
            config=train_config,
            wandb_run=wandb_run,
            run_name=run_name
        )
        
        training_time = time.time() - start_time
        logger.info(f"⏱️  Training completed in {training_time:.2f} seconds")
        
        # Save final checkpoint
        checkpoint_dir = Path("checkpoints") / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        final_checkpoint_path = save_checkpoint(
            trainer=trainer,
            checkpoint_dir=checkpoint_dir,
            step=total_timesteps,
            is_final=True
        )
        logger.info(f"💾 Final checkpoint saved: {final_checkpoint_path}")
        
        # Upload to HuggingFace if enabled
        if upload_to_huggingface and HF_AVAILABLE:
            try:
                logger.info("☁️  Uploading checkpoint to HuggingFace...")
                
                # Create repository if it doesn't exist
                api = HfApi()
                try:
                    create_repo(repo_id, private=False, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Repository creation warning: {e}")
                
                # Upload checkpoint
                upload_file(
                    path_or_fileobj=str(final_checkpoint_path),
                    path_in_repo=f"checkpoints/{run_name}_final.pth",
                    repo_id=repo_id,
                    commit_message=f"Upload final checkpoint for {run_name}"
                )
                logger.info("✅ Checkpoint uploaded to HuggingFace successfully")
                
            except Exception as e:
                logger.error(f"❌ HuggingFace upload failed: {e}")
        
        # Upload final model to HuggingFace Hub if enabled
        if upload_model_to_hf and HF_AVAILABLE:
            try:
                logger.info("☁️  Uploading final model to HuggingFace Hub...")
                
                hf_repo_name = hf_model_name or f"{project_name}-{run_name}"
                
                # Create model repository
                api = HfApi()
                try:
                    create_repo(hf_repo_name, private=False, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Model repository creation warning: {e}")
                
                # Upload final model
                upload_file(
                    path_or_fileobj=str(final_checkpoint_path),
                    path_in_repo="pytorch_model.bin",
                    repo_id=hf_repo_name,
                    commit_message=f"Upload final model for {run_name}"
                )
                
                logger.info(f"✅ Final model uploaded to HuggingFace: {hf_repo_name}")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to upload final model: {e}")
        
        # Close W&B run
        if use_wandb:
            wandb.finish()
        
        # Prepare final results
        results = {
            'training_results': training_results,
            'run_name': run_name,
            'training_time': training_time,
            'final_checkpoint': str(final_checkpoint_path),
            'config': {
                'env_name': env_name,
                'total_timesteps': total_timesteps,
                'learning_rate': learning_rate,
                'use_curiosity': use_curiosity,
                'use_rnd': use_rnd
            }
        }
        
        logger.info("🎉 Training completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if debug_mode:
            import traceback
            logger.error(traceback.format_exc())
        if use_wandb and wandb_run is not None:
            wandb_run.finish()
        raise
    
    finally:
        # Cleanup
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Online PPO Training with Pretrained Models")
    parser.add_argument("--env_name", type=str, default="MiniHack-Quest-Hard-v0",
                       help="MiniHack environment name")
    parser.add_argument("--repo_id", type=str, default="catid/SequentialSkillRL",
                       help="HuggingFace repository ID")
    parser.add_argument("--vae_filename", type=str, default="nethack-vae.pth",
                       help="VAE model filename")
    parser.add_argument("--hmm_filename", type=str, default="hmm_round3.pt",
                       help="HMM model filename")
    parser.add_argument("--total_timesteps", type=int, default=50000,
                       help="Total training timesteps")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--test_mode", action="store_true",
                       help="Run in test mode")
    parser.add_argument("--test_episodes", type=int, default=10,
                       help="Number of test episodes")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--upload_to_huggingface", action="store_true",
                       help="Upload checkpoints to HuggingFace")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Run training
    try:
        results = train_online_ppo_with_pretrained_models(
            env_name=args.env_name,
            repo_id=args.repo_id,
            vae_filename=args.vae_filename,
            hmm_filename=args.hmm_filename,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            test_mode=args.test_mode,
            test_episodes=args.test_episodes,
            debug_mode=args.debug,
            upload_to_huggingface=args.upload_to_huggingface,
            seed=args.seed
        )
        
        print("\n🎉 Training completed!")
        print(f"Run name: {results['run_name']}")
        if 'training_time' in results:
            print(f"Training time: {results['training_time']:.2f} seconds")
        if 'test_stats' in results:
            print(f"Test mean reward: {results['test_stats']['mean_reward']:.2f}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)
