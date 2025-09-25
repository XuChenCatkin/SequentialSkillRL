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
import traceback

import nle
import minihack

# Now try to import PPO components
from rl.ppo import (
    PPOConfig, CuriosityConfig, HMMOnlineConfig, VAEOnlineConfig, RNDConfig, TrainConfig, PPOTrainer,
    set_seed
)

# Import model components
from src.model import MultiModalHackVAE, VAEConfig
from src.skill_space import StickyHDPHMMVI, StickyHDPHMMParams, NIWPrior

# Import training utilities
from training.training_utils import (
    load_model_from_huggingface, 
    load_hmm_from_huggingface,
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
    vae_repo_id: str,
    hmm_repo_id: str,
    # Configuration objects - provide full control over training
    ppo_config: PPOConfig,
    curiosity_config: CuriosityConfig,
    hmm_config: Optional[HMMOnlineConfig],
    vae_config: VAEOnlineConfig,
    rnd_config: Optional[RNDConfig],
    train_config: TrainConfig,
    # PPO checkpoint loading for continuation
    ppo_repo_id: Optional[str] = None,  # HuggingFace repo with existing PPO checkpoint
    ppo_checkpoint_path: Optional[str] = None,  # Local path to PPO checkpoint
    resume_training: bool = False,  # Whether to continue from existing PPO checkpoint
    # Weights & Biases monitoring parameters
    use_wandb: bool = False,
    wandb_project: str = "SequentialSkillRL",
    wandb_entity: str = None,
    wandb_run_name: str = None,
    wandb_tags: List[str] = None,
    wandb_notes: str = None,
    # HuggingFace integration
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    hf_token: str | None = None,
    hf_private: bool = True,
    hf_upload_artifacts: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Train online PPO agent with pretrained VAE and HMM models.
    
    Args:
        vae_repo_id: HuggingFace repository ID for VAE model
        hmm_repo_id: HuggingFace repository ID for HMM model (optional)
            
        # Configuration objects (recommended approach)
        ppo_config: PPO training configuration.
        curiosity_config: Curiosity-driven exploration configuration.
        hmm_config: HMM online learning configuration.
        vae_config: VAE online learning configuration.
        rnd_config: Random Network Distillation configuration.
        train_config: General training configuration.
        
        # PPO checkpoint loading for continuation training
        ppo_repo_id: HuggingFace repository ID containing existing PPO checkpoint
        ppo_checkpoint_path: Local file path to existing PPO checkpoint (.pth file)
        resume_training: Whether to load and continue from existing PPO checkpoint
            - If True, requires either ppo_repo_id or ppo_checkpoint_path
            - If False, starts fresh PPO training (default behavior)
        
        # Monitoring and uploading
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_entity: W&B entity/team name
        wandb_run_name: Custom run name
        wandb_tags: List of tags for the run
        wandb_notes: Notes for the run
        push_to_hub: Upload checkpoints to HuggingFace
        hub_repo_id: Target HuggingFace repository ID
        hf_token: HuggingFace authentication token
        hf_private: Make repository private
        hf_upload_artifacts: Upload training artifacts
        logger: Optional logger instance
    
    Returns:
        Dictionary with training results and final metrics
    """
    if logger:
        logger.info("=" * 80)
        logger.info(f"Starting Online PPO Training with Pretrained Models")
        logger.info(f"Environment: {train_config.env_id}")
        logger.info(f"VAE Repository: {vae_repo_id}")
        logger.info(f"HMM Repository: {hmm_repo_id}")
        logger.info(f"Total Timesteps: {ppo_config.total_updates:,}")
        logger.info("=" * 80)
    
    # Device setup
    device = torch.device(train_config.device)
    if logger: logger.info(f"🔧 Using device: {device}")
    
    # Create run name if not provided
    if wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = "test" if test_mode else "train"
        wandb_run_name = f"online_ppo_{train_config.env_id}_{mode_str}_{timestamp}"

    # W&B will be initialized later after config objects are created
    wandb_run = None
    
    try:
        # Determine loading strategy based on resume mode
        if resume_training and ppo_repo_id:
            # Resume mode: Load all components (VAE, HMM, PPO) from the same unified repo
            unified_repo = ppo_repo_id
            if logger: logger.info(f"🔄 Resume mode: Loading all models from unified repo: {unified_repo}")
            
            # Load VAE from unified repo
            if logger: logger.info("🎨 Loading VAE from unified repo...")
            try:
                # Try loading from different possible VAE file names in the unified repo
                vae_model, config = None, None
                for vae_filename in ["vae_model.pth", "nethack-vae.pth", "vae.pth"]:
                    try:
                        vae_model, config = load_model_from_huggingface(
                            repo_name=unified_repo,
                            filename=vae_filename,
                            token=hf_token,
                            device=str(device),
                            fallback_to_checkpoint_config=True  # Enable fallback for unified repos
                        )
                        if logger: logger.info(f"✅ VAE loaded from {vae_filename}")
                        break
                    except Exception:
                        continue
                
                if vae_model is None:
                    raise ValueError(f"No VAE model found in unified repo {unified_repo}")
                    
            except Exception as e:
                if logger: logger.warning(f"⚠️  Failed to load VAE from unified repo: {e}")
                if logger: logger.info(f"🔄 Falling back to separate VAE repo: {vae_repo_id}")
                vae_model, config = load_model_from_huggingface(
                    repo_name=vae_repo_id,
                    token=hf_token,
                    device=str(device),
                    fallback_to_checkpoint_config=True  # Enable fallback for VAE repos
                )
            
            # Load HMM from unified repo (if HMM is being used)
            hmm_model, loaded_config, hmm_params, niw, metadata = None, None, None, None, None
            if train_config.use_hmm:  # Only load HMM if specified
                if logger: logger.info("🧠 Loading HMM from unified repo...")
                try:
                    # Try loading from different possible HMM file names in the unified repo
                    for hmm_filename in ["hmm_model.pt", "nethack-hmm.pt", "hmm.pt"]:
                        try:
                            hmm_model, loaded_config, hmm_params, niw, metadata = load_hmm_from_huggingface(
                                repo_name=unified_repo,
                                filename=hmm_filename,
                                round_num=None,
                                device=str(device)
                            )
                            if logger: logger.info(f"✅ HMM loaded from {hmm_filename}")
                            break
                        except Exception:
                            continue
                    
                    if hmm_model is None:
                        raise ValueError(f"No HMM model found in unified repo {unified_repo}")
                        
                except Exception as e:
                    if logger: logger.warning(f"⚠️  Failed to load HMM from unified repo: {e}")
                    if logger: logger.info(f"🔄 Falling back to separate HMM repo: {hmm_repo_id}")
                    hmm_model, loaded_config, hmm_params, niw, metadata = load_hmm_from_huggingface(
                        repo_name=hmm_repo_id,
                        round_num=None,
                        device=str(device)
                    )
        else:
            # Fresh training mode: Load VAE and HMM from separate repos (original behavior)
            if logger: logger.info("🆕 Fresh training mode: Loading models from separate repos")
            
            # Load VAE from dedicated VAE repo
            if logger: logger.info("🎨 Loading pretrained VAE model...")
            vae_model, config = load_model_from_huggingface(
                repo_name=vae_repo_id,
                token=hf_token,
                device=str(device),
                fallback_to_checkpoint_config=True  # Enable fallback for VAE repos
            )
            
            # Load HMM from dedicated HMM repo (if specified)
            hmm_model, loaded_config, hmm_params, niw, metadata = None, None, None, None, None
            if train_config.use_hmm:  # Only load HMM if specified
                if logger: logger.info("🧠 Loading pretrained HMM model...")
                hmm_model, loaded_config, hmm_params, niw, metadata = load_hmm_from_huggingface(
                    repo_name=hmm_repo_id,
                    round_num=None,  # None means latest
                    device=str(device)
                )
        
        # Log successful loading
        if logger: logger.info(f"✅ VAE model loaded successfully")
        if logger: logger.info(f"   - Latent dimension: {config.latent_dim}")
        if logger: logger.info(f"   - Skills: {config.skill_num}")
        
        if hmm_model is not None:
            if logger: logger.info(f"✅ HMM model loaded successfully")
            if metadata:
                if logger: logger.info(f"   🏷️  HMM Round: {metadata.get('round', 'unknown')}")
                if logger: logger.info(f"   📅 Created: {metadata.get('training_timestamp', 'unknown')}")
        else:
            if logger: logger.info("ℹ️  No HMM model loaded (HMM integration disabled)")
            
        # Create environment
        if logger: logger.info(f"🎮 Creating environment: {train_config.env_id}")
        env = gym.make(train_config.env_id)

        # Log environment info
        if logger: logger.info(f"   - Observation space: {env.observation_space}")
        if logger: logger.info(f"   - Action space: {env.action_space}")
        
        
        if hmm_model is not None:
            hmm_model.set_posterior_as_prior(hmm_config.temper_weight, skip_remainder_state=True)
            logger.info("🔄 HMM posterior set as prior for online updates with tempered weight {}".format(hmm_config.temper_weight))
            hmm_model.stream_rho_niw = hmm_config.rho_emission
            hmm_model.stream_rho_trans = hmm_config.rho_transition
        
        vae_config.training_config = config
        if logger: logger.info(f"🔄 Using provided VAE config")

        if hmm_model is None:
            vae_config.training_config.prior_mode = "standard"
            if logger: logger.info(f"   - VAE prior mode set to 'standard' for online training without HMM")
        else:
            vae_config.training_config.prior_mode = "hmm"
            if logger: logger.info(f"   - VAE prior mode set to 'hmm' for online training")

        # Initialize W&B now that all config objects are created
        if WANDB_AVAILABLE and use_wandb and not train_config.test_mode:
            try:
                # Create comprehensive config for W&B logging
                wandb_config = {
                    "env_name": train_config.env_id,
                    "vae_repo_id": vae_repo_id,
                    "hmm_repo_id": hmm_repo_id,
                    "device": str(device),
                    "train_seed": train_config.train_seed,
                    "eval_seed": train_config.eval_seed,
                    # PPO Configuration
                    "ppo": {
                        "num_envs": ppo_config.num_envs,
                        "rollout_len": ppo_config.rollout_len,
                        "total_updates": ppo_config.total_updates,
                        "minibatch_envs": ppo_config.minibatch_envs,
                        "epochs_per_update": ppo_config.epochs_per_update,
                        "gamma": ppo_config.gamma,
                        "gae_lambda": ppo_config.gae_lambda,
                        "clip_coef": ppo_config.clip_coef,
                        "ent_coef": ppo_config.ent_coef,
                        "vf_coef": ppo_config.vf_coef,
                        "max_grad_norm": ppo_config.max_grad_norm,
                        "learning_rate": ppo_config.learning_rate,
                        "policy_uses_skill": ppo_config.policy_uses_skill,
                        "deterministic_eval": ppo_config.deterministic_eval,
                        "rnn_hidden_size": ppo_config.rnn_hidden_size
                    },
                    # Curiosity Configuration
                    "curiosity": {
                        "use_dyn_kl": curiosity_config.use_dyn_kl,
                        "use_skill_entropy": curiosity_config.use_skill_entropy,
                        "use_skill_transition_novelty": curiosity_config.use_skill_transition_novelty,
                        "use_rnd": curiosity_config.use_rnd,
                        "eta0_dyn": curiosity_config.eta0_dyn,
                        "tau_dyn": curiosity_config.tau_dyn,
                        "eta0_hdp": curiosity_config.eta0_hdp,
                        "tau_hdp": curiosity_config.tau_hdp,
                        "eta0_rnd": curiosity_config.eta0_rnd,
                        "tau_rnd": curiosity_config.tau_rnd,
                        "use_skill_boundary_gate": curiosity_config.use_skill_boundary_gate,
                        "gate_delta_eps": curiosity_config.gate_delta_eps,
                        "eta0_stn": curiosity_config.eta0_stn,
                        "tau_stn": curiosity_config.tau_stn,
                        "ema_beta": curiosity_config.ema_beta,
                        "eps": curiosity_config.eps
                    },
                    # HMM Configuration
                    "hmm": {
                        "hmm_update_every": hmm_config.hmm_update_every,
                        "hmm_fit_window": hmm_config.hmm_fit_window,
                        "hmm_max_iters": hmm_config.hmm_max_iters,
                        "hmm_tol": hmm_config.hmm_tol,
                        "hmm_elbo_drop_tol": hmm_config.hmm_elbo_drop_tol,
                        "rho_emission": hmm_config.rho_emission,
                        "rho_transition": hmm_config.rho_transition,
                        "optimise_pi": hmm_config.optimise_pi
                    } if hmm_config else None,
                    # RND Configuration
                    "rnd": {
                        "proj_dim": rnd_config.proj_dim,
                        "hidden": rnd_config.hidden,
                        "lr": rnd_config.lr,
                        "update_per_rollout": rnd_config.update_per_rollout
                    } if rnd_config else None,
                    # Training Configuration
                    "training": {
                        "env_id": train_config.env_id,
                        "train_seed": train_config.train_seed,
                        "eval_seed": train_config.eval_seed,
                        "max_episode_steps_train": train_config.max_episode_steps_train,
                        "max_episode_steps_eval": train_config.max_episode_steps_eval,
                        "device": train_config.device,
                        "log_dir": train_config.log_dir,
                        "save_every": train_config.save_every,
                        "eval_every": train_config.eval_every,
                        "eval_episodes": train_config.eval_episodes
                    }
                }
                
                wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=wandb_config,
                    entity=wandb_entity,
                    tags=wandb_tags,
                    notes=wandb_notes
                )
                if logger: logger.info(f"📊 W&B initialized with run name: {wandb_run_name}")
            except Exception as e:
                if logger: logger.warning(f"⚠️  W&B initialization failed: {e}")
                wandb_run = None

        # Create PPO trainer
        if logger: logger.info("🚀 Initializing PPO trainer...")
        trainer = PPOTrainer(
            env_id=train_config.env_id,
            ppo_cfg=ppo_config,
            cur_cfg=curiosity_config,
            hmm_cfg=hmm_config,
            vae_cfg=vae_config,
            rnd_cfg=rnd_config,
            run_cfg=train_config,
            vae=vae_model,
            hmm=hmm_model,
            wandb_run=wandb_run
        )
        if logger: logger.info("✅ PPO trainer initialized successfully")
        
        # Load existing PPO checkpoint if resuming training
        if resume_training:
            if logger: logger.info("🔄 Loading existing PPO checkpoint for continuation...")
            
            # Validate checkpoint source
            if not ppo_repo_id and not ppo_checkpoint_path:
                raise ValueError("resume_training=True requires either ppo_repo_id or ppo_checkpoint_path")
            
            checkpoint_data = None
            
            # Load from HuggingFace repository
            if ppo_repo_id:
                try:
                    if logger: logger.info(f"📥 Loading PPO checkpoint from HuggingFace: {ppo_repo_id}")
                    from huggingface_hub import hf_hub_download
                    
                    # Download the PPO policy file
                    ppo_path = hf_hub_download(
                        repo_id=ppo_repo_id, 
                        filename="ppo_policy.pth",
                        token=hf_token
                    )
                    checkpoint_data = torch.load(ppo_path, map_location=device)
                    if logger: logger.info(f"✅ PPO checkpoint loaded from HuggingFace")
                    
                except Exception as e:
                    if logger: logger.error(f"❌ Failed to load PPO checkpoint from HuggingFace: {e}")
                    if ppo_checkpoint_path:
                        if logger: logger.info(f"🔄 Falling back to local checkpoint: {ppo_checkpoint_path}")
                    else:
                        raise RuntimeError(f"Failed to load PPO checkpoint from HuggingFace: {e}")
            
            # Load from local file (either primary or fallback)
            if not checkpoint_data and ppo_checkpoint_path:
                try:
                    if logger: logger.info(f"📥 Loading PPO checkpoint from local file: {ppo_checkpoint_path}")
                    checkpoint_data = torch.load(ppo_checkpoint_path, map_location=device)
                    if logger: logger.info(f"✅ PPO checkpoint loaded from local file")
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to load PPO checkpoint from local file: {e}")
            
            if not checkpoint_data:
                raise RuntimeError("No PPO checkpoint could be loaded")
            
            # Restore PPO trainer state
            try:
                # Load actor-critic model state
                if 'actor_critic' in checkpoint_data:
                    trainer.actor_critic.load_state_dict(checkpoint_data['actor_critic'])
                    if logger: logger.info("✅ Actor-critic model state restored")
                
                # Load optimizer state  
                if 'opt' in checkpoint_data:
                    trainer.opt.load_state_dict(checkpoint_data['opt'])
                    if logger: logger.info("✅ Optimizer state restored")
                
                # Load training step count if available
                if 'global_steps' in checkpoint_data:
                    trainer.global_steps = checkpoint_data['global_steps']
                    if logger: logger.info(f"✅ Training resumed from step {trainer.global_steps:,}")
                
                # Load training configuration for validation
                if 'config' in checkpoint_data:
                    loaded_config = checkpoint_data['config']
                    if logger: logger.info("ℹ️  Loaded training configuration from checkpoint")
                    
                    # Log key differences if any
                    if loaded_config.get('env_name') != train_config.env_id:
                        if logger: logger.warning(f"⚠️  Environment changed: {loaded_config.get('env_name')} → {train_config.env_id}")

                if logger: logger.info("🎯 PPO checkpoint restoration completed successfully")
                
            except Exception as e:
                if logger: logger.error(f"❌ Failed to restore PPO trainer state: {e}")
                raise RuntimeError(f"Failed to restore PPO trainer state: {e}")
        
        else:
            if logger: logger.info("🆕 Starting fresh PPO training (no checkpoint loading)")
        
        # Test mode: run evaluation episodes
        if train_config.test_mode:
            if logger: logger.info(f"🧪 Running {train_config.eval_episodes} test episodes...")
            test_results = []

            for episode in range(train_config.eval_episodes):
                if logger: logger.info(f"Episode {episode + 1}/{train_config.eval_episodes}")

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
                
                if logger: logger.info(f"   Reward: {episode_reward}, Length: {episode_length}")
            
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
            
            if logger: logger.info("📈 Test Results:")
            if logger: logger.info(f"   Mean Reward: {test_stats['mean_reward']:.2f} ± {test_stats['std_reward']:.2f}")
            if logger: logger.info(f"   Min/Max Reward: {test_stats['min_reward']:.2f} / {test_stats['max_reward']:.2f}")
            if logger: logger.info(f"   Mean Length: {test_stats['mean_length']:.1f} ± {test_stats['std_length']:.1f}")
            
            return {
                'test_results': test_results,
                'test_stats': test_stats,
                'run_name': wandb_run_name
            }
        
        # Training mode
        if logger: logger.info("🏋️  Starting PPO training...")
        start_time = time.time()

        trainer.train(logger)
        
        training_time = time.time() - start_time
        if logger: logger.info(f"⏱️  Training completed in {training_time:.2f} seconds")
        
        # Save final checkpoint
        checkpoint_dir = Path("checkpoints") / wandb_run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Use trainer's save method to create final checkpoint
        if logger: logger.info("💾 Saving final PPO checkpoint...")
        final_checkpoint_path = os.path.join(checkpoint_dir, "ppo_policy.pth")
        torch.save({
            "actor_critic": trainer.actor_critic.state_dict(),
            "opt": trainer.opt.state_dict(),
            "global_steps": trainer.global_steps,
            "config": {
                "ppo_config": ppo_config.__dict__,
                "curiosity_config": curiosity_config.__dict__,
                "hmm_config": hmm_config.__dict__ if hmm_config else None,
                "vae_config": vae_config.__dict__ if vae_config else None,
                "train_config": train_config.__dict__
            },
            "training_metadata": {
                "env_name": train_config.env_id,
                "total_timesteps": ppo_config.total_updates,
                "final_step": trainer.global_steps,
                "timestamp": datetime.now().isoformat()
            }
        }, final_checkpoint_path)
        if logger: logger.info(f"💾 Final checkpoint saved: {final_checkpoint_path}")
        
        use_curiosity = (curiosity_config.use_dyn_kl or 
                         curiosity_config.use_skill_entropy or 
                         curiosity_config.use_skill_transition_novelty)
        use_rnd = curiosity_config.use_rnd
            
        # Collect training results from log file or trainer state for artifact upload
        training_results = {
            'total_timesteps': ppo_config.total_updates,
            'training_time': training_time,
            'final_checkpoint_path': str(final_checkpoint_path),
            'global_steps': getattr(trainer, 'global_steps', ppo_config.total_updates),
            'config': {
                'env_name': train_config.env_id,
                'learning_rate': ppo_config.learning_rate,
                'n_epochs': ppo_config.epochs_per_update,
                'gamma': ppo_config.gamma,
                'vf_coef': ppo_config.vf_coef,
                'ent_coef': ppo_config.ent_coef,
                'max_grad_norm': ppo_config.max_grad_norm,
                'use_curiosity': use_curiosity,
                'use_rnd': use_rnd,
                'use_hmm': train_config.use_hmm
            }
        }
        
        # Try to read training logs if available
        if hasattr(trainer, '_log_file') and os.path.exists(trainer._log_file):
            try:
                training_logs = []
                with open(trainer._log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            training_logs.append(json.loads(line.strip()))
                training_results['training_logs'] = training_logs
                if logger: logger.info(f"📊 Collected {len(training_logs)} training log entries")
            except Exception as e:
                if logger: logger.warning(f"⚠️  Could not read training logs: {e}")
                training_results['training_logs'] = []
        
        # Upload to HuggingFace if enabled (upload all components to one repository)
        if push_to_hub and HF_AVAILABLE:
            try:
                if logger: logger.info("☁️  Uploading all model components to HuggingFace...")
                
                # Determine the target repository for all components
                target_repo = hub_repo_id or f"{wandb_project}-{wandb_run_name}"
                
                # Create repository if it doesn't exist
                api = HfApi()
                try:
                    create_repo(target_repo, private=hf_private, exist_ok=True, token=hf_token)
                    if logger: logger.info(f"📦 Created/verified repository: {target_repo}")
                except Exception as e:
                    if logger: logger.warning(f"Repository creation warning: {e}")
                
                # Upload PPO checkpoint (main training result)
                if logger: logger.info("📤 Uploading PPO policy checkpoint...")
                upload_file(
                    path_or_fileobj=str(final_checkpoint_path),
                    path_in_repo=f"ppo_policy.pth",
                    repo_id=target_repo,
                    token=hf_token,
                    commit_message=f"Upload PPO policy for {wandb_run_name}"
                )
                
                # Upload VAE model (copy from original repo to unified repo)
                if logger: logger.info("📤 Uploading VAE model...")
                try:
                    # Save VAE model locally first
                    vae_save_path = checkpoint_dir / "vae_model.pth"
                    torch.save({
                        'model_state_dict': vae_model.state_dict(),
                        'config': config.__dict__,
                        'model_class': vae_model.__class__.__name__
                    }, vae_save_path)
                    
                    upload_file(
                        path_or_fileobj=str(vae_save_path),
                        path_in_repo="vae_model.pth",
                        repo_id=target_repo,
                        token=hf_token,
                        commit_message=f"Upload VAE model for {wandb_run_name}"
                    )
                    if logger: logger.info("✅ VAE model uploaded successfully")
                except Exception as e:
                    if logger: logger.warning(f"⚠️  VAE upload failed: {e}")
                
                # Upload HMM model (copy from original repo to unified repo)
                if logger: logger.info("📤 Uploading HMM model...")
                try:
                    # Save HMM model locally first
                    hmm_save_path = checkpoint_dir / "hmm_model.pt"
                    hmm_save_data = {
                        'config': config,
                        'hmm_posterior_params': hmm_model.get_posterior_params(),
                        'hmm_params': hmm_params,
                        'niw_prior': hmm_model.get_niw_prior_params(),
                        'phi_prior': hmm_model.get_phi_prior_params(),
                        'rho_emission': hmm_model.get_rho_emission().cpu(),
                        'rho_transition': hmm_model.get_rho_transition().cpu(),
                        'round': None,
                        'training_timestamp': datetime.now().isoformat(),
                        'diagnostics': None
                    }
                    torch.save(hmm_save_data, hmm_save_path)
                    
                    upload_file(
                        path_or_fileobj=str(hmm_save_path),
                        path_in_repo="hmm_model.pt",
                        repo_id=target_repo,
                        token=hf_token,
                        commit_message=f"Upload HMM model for {wandb_run_name}"
                    )
                    if logger: logger.info("✅ HMM model uploaded successfully")
                except Exception as e:
                    if logger: logger.warning(f"⚠️  HMM upload failed: {e}")
                
                # Create and upload model card with information about all components
                if logger: logger.info("📝 Creating model card...")
                try:
                    model_card_content = f"""---
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
- nethack
- ppo
- vae
- hmm
- minihack
- sequential-skills

---

# {target_repo}

This repository contains a complete Sequential Skill RL model trained on NetHack/MiniHack environments.

## Model Components

### 1. PPO Policy (`ppo_policy.pth`)
- **Type**: Proximal Policy Optimization agent
- **Environment**: {train_config.env_id}
- **Training Steps**: {ppo_config.total_updates}
- **Features**: 
  - Curiosity-driven exploration: {use_curiosity}
  - Random Network Distillation: {use_rnd}

### 2. VAE Model (`vae_model.pth`)
- **Type**: Variational Autoencoder
- **Purpose**: Encodes NetHack observations into latent skill representations
- **Latent Dimension**: {getattr(config, 'latent_dim', 'Unknown')}
- **Architecture**: {getattr(config, 'architecture', 'Unknown')}

### 3. HMM Model (`hmm_model.pth`)
- **Type**: Hidden Markov Model (Sticky HDP-HMM)
- **Purpose**: Models sequential skill transitions and dynamics
- **Integration**: Used for intrinsic reward computation

## Usage

```python
import torch
from training.online_rl import train_online_ppo_with_pretrained_models

# Load the complete model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load individual components
ppo_checkpoint = torch.load('ppo_policy.pth', map_location=device)
vae_data = torch.load('vae_model.pth', map_location=device)
hmm_data = torch.load('hmm_model.pth', map_location=device)

# Use for inference or continued training
results = train_online_ppo_with_pretrained_models(
    env_name="{train_config.env_id}",
    vae_repo_id="{vae_repo_id}",
    hmm_repo_id="{hmm_repo_id}",
    test_mode=True
)
```

## Training Configuration

- **Environment**: {train_config.env_id}
- **Learning Rate**: {ppo_config.learning_rate}
- **Training Time**: {training_time:.2f} seconds
- **Device**: {device}
- **Train Seed**: {train_config.train_seed}
- **Eval Seed**: {train_config.eval_seed}

## Performance

Training completed successfully with the following configuration:
- Curiosity-driven exploration: {use_curiosity}
- Random Network Distillation: {use_rnd}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                    
                    model_card_path = checkpoint_dir / "README.md"
                    with open(model_card_path, 'w') as f:
                        f.write(model_card_content)
                    
                    upload_file(
                        path_or_fileobj=str(model_card_path),
                        path_in_repo="README.md",
                        repo_id=target_repo,
                        token=hf_token,
                        commit_message=f"Add model card for {wandb_run_name}"
                    )
                    if logger: logger.info("✅ Model card uploaded successfully")
                except Exception as e:
                    if logger: logger.warning(f"⚠️  Model card upload failed: {e}")
                
                # Upload training configuration
                if logger: logger.info("📄 Uploading training configuration...")
                try:
                    config_data = {
                        'training_config': {
                            'env_name': train_config.env_id,
                            'total_timesteps': ppo_config.total_updates,
                            'learning_rate': ppo_config.learning_rate,
                            'n_epochs': ppo_config.epochs_per_update,
                            'gamma': ppo_config.gamma,
                            'vf_coef': ppo_config.vf_coef,
                            'ent_coef': ppo_config.ent_coef,
                            'max_grad_norm': ppo_config.max_grad_norm,
                            'use_curiosity': use_curiosity,
                            'curiosity_dyn': curiosity_config.use_dyn_kl,
                            'curiosity_skill_entropy': curiosity_config.use_skill_entropy,
                            'curiosity_skill_transition_novelty': curiosity_config.use_skill_transition_novelty,
                            'curiosity_dyn_coef': curiosity_config.eta0_dyn,
                            'curiosity_hdp_coef': curiosity_config.eta0_hdp,
                            'curiosity_stn_coef': curiosity_config.eta0_stn,
                            'use_rnd': use_rnd,
                            'rnd_lr': rnd_config.lr if use_rnd and rnd_config else None,
                            'rnd_coef': curiosity_config.eta0_rnd if use_rnd else None,
                            'device': str(device),
                            'train_seed': train_config.train_seed,
                            'eval_seed': train_config.eval_seed,
                            'max_episode_steps_train': train_config.max_episode_steps_train,
                            'max_episode_steps_eval': train_config.max_episode_steps_eval,
                            'training_time': training_time
                        },
                        'model_sources': {
                            'vae_repo_id': vae_repo_id,
                            'hmm_repo_id': hmm_repo_id
                        },
                        'timestamp': datetime.now().isoformat(),
                        'run_name': wandb_run_name
                    }
                    
                    config_path = checkpoint_dir / "training_config.json"
                    with open(config_path, 'w') as f:
                        json.dump(config_data, f, indent=2)
                    
                    upload_file(
                        path_or_fileobj=str(config_path),
                        path_in_repo="training_config.json",
                        repo_id=target_repo,
                        token=hf_token,
                        commit_message=f"Add training configuration for {wandb_run_name}"
                    )
                    if logger: logger.info("✅ Training configuration uploaded successfully")
                except Exception as e:
                    if logger: logger.warning(f"⚠️  Configuration upload failed: {e}")
                
                if logger: logger.info(f"🎉 All model components uploaded to: https://huggingface.co/{target_repo}")
                
                # Upload training artifacts if requested
                if hf_upload_artifacts:
                    try:
                        if logger: logger.info("📈 Uploading training artifacts...")
                        from .training_utils import upload_training_artifacts_to_huggingface
                        
                        # Extract training metrics from logs for artifact upload
                        train_losses = []
                        eval_returns = []
                        steps = []
                        
                        if 'training_logs' in training_results:
                            for log_entry in training_results['training_logs']:
                                if 'steps' in log_entry:
                                    steps.append(log_entry['steps'])
                                if 'return/mean_ext' in log_entry:
                                    train_losses.append(-log_entry['return/mean_ext'])  # Convert reward to loss-like metric
                                if 'eval/return_mean' in log_entry:
                                    eval_returns.append(log_entry['eval/return_mean'])
                        
                        # Create artifact data structure similar to VAE training
                        artifact_config = {
                            'training_type': 'online_ppo',
                            'environment': train_config.env_id,
                            'total_timesteps': ppo_config.total_updates,
                            'training_time': training_time,
                            'device': str(device),
                            'ppo_config': {
                                'learning_rate': ppo_config.learning_rate,
                                'n_epochs': ppo_config.epochs_per_update,
                                'gamma': ppo_config.gamma,
                                'vf_coef': ppo_config.vf_coef,
                                'ent_coef': ppo_config.ent_coef,
                                'max_grad_norm': ppo_config.max_grad_norm
                            },
                            'exploration_config': {
                                'use_curiosity': use_curiosity,
                                'curiosity_dyn': curiosity_config.use_dyn_kl,
                                'curiosity_skill_entropy': curiosity_config.use_skill_entropy,
                                'curiosity_skill_transition_novelty': curiosity_config.use_skill_transition_novelty,
                                'curiosity_dyn_coef': curiosity_config.eta0_dyn,
                                'curiosity_hdp_coef': curiosity_config.eta0_hdp,
                                'curiosity_stn_coef': curiosity_config.eta0_stn,
                                'use_rnd': use_rnd,
                                'rnd_lr': rnd_config.lr if use_rnd and rnd_config else None,
                                'rnd_coef': curiosity_config.eta0_rnd if use_rnd else None
                            },
                            'model_sources': {
                                'vae_repo_id': vae_repo_id,
                                'hmm_repo_id': hmm_repo_id
                            }
                        }
                        
                        # Use eval returns as "test losses" and negative mean rewards as "train losses"
                        upload_training_artifacts_to_huggingface(
                            repo_name=target_repo,
                            train_losses=train_losses if train_losses else [0.0],  # Fallback if no data
                            test_losses=[-r for r in eval_returns] if eval_returns else [0.0],  # Convert rewards to loss-like
                            training_config=artifact_config,
                            token=hf_token,
                            plots_dir="training_artifacts"
                        )
                        
                        if logger: logger.info("✅ Training artifacts uploaded successfully")
                        
                    except Exception as e:
                        if logger: logger.warning(f"⚠️  Training artifacts upload failed: {e}")
                        if logger: logger.error(traceback.format_exc())
                
            except Exception as e:
                if logger: logger.error(f"❌ HuggingFace upload failed: {e}")
                if logger: logger.error(traceback.format_exc())
        
        # Close W&B run
        if use_wandb:
            wandb.finish()
        
        # Prepare final results
        results = {
            'training_results': training_results,
            'run_name': wandb_run_name,
            'training_time': training_time,
            'final_checkpoint': str(final_checkpoint_path),
            'config': {
                'env_name': train_config.env_id,
                'vae_repo_id': vae_repo_id,
                'hmm_repo_id': hmm_repo_id,
                'total_timesteps': ppo_config.total_updates,
                'learning_rate': ppo_config.learning_rate,
                'use_curiosity': use_curiosity,
                'use_rnd': use_rnd
            }
        }
        
        if logger: logger.info("🎉 Training completed successfully!")
        return results
        
    except Exception as e:
        if logger: logger.error(f"❌ Training failed: {e}")
        if logger: logger.error(traceback.format_exc())
        if use_wandb and wandb_run is not None:
            wandb_run.finish()
        raise
    
    finally:
        # Cleanup
        if 'env' in locals():
            env.close()