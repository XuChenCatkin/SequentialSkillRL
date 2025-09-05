"""
Complete VAE training pipeline for NetHack Learning Dataset
Supports both the simple NetHackVAE and the sophisticated MiniHackVAE from src/model.py
"""
import os
import numpy as np
import torch
import sys
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Callable
import warnings
import logging
from datetime import datetime
from torch.optim.lr_scheduler import OneCycleLR, ConstantLR
from src.skill_space import StickyHDPHMMVI, StickyHDPHMMParams, NIWPrior
import copy

warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Weights & Biases integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available. Install with: pip install wandb")

# HuggingFace integration imports
try:
    from huggingface_hub import HfApi, Repository, upload_file, create_repo, login
    # Try to import RepositoryNotFoundError from different locations
    try:
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        try:
            from huggingface_hub import RepositoryNotFoundError
        except ImportError:
            # Fallback for newer versions - use generic HTTP error
            from requests.exceptions import HTTPError as RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    print("⚠️  HuggingFace Hub not available. Install with: pip install huggingface_hub")
    HF_AVAILABLE = False

# Scikit-learn
from sklearn.cluster import KMeans

from src.model import MultiModalHackVAE, vae_loss, CHAR_DIM, VAEConfig
import torch.optim as optim
import random
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from src.data_collection import NetHackDataCollector, BLStatsAdapter
from training.training_utils import save_checkpoint, save_model_to_huggingface, load_model_from_huggingface, \
    upload_training_artifacts_to_huggingface, create_model_demo_notebook, load_model_from_local

# Import our utility functions
from utils.analysis import compute_hmm_diagnostics, visualize_hmm_after_estep


def ramp_weight(initial_weight: float, final_weight: float, shape: str, progress: float, rate: float = 10.0, centre: float = 0.5, f: Optional[Callable[[float, float, float], float]] = None) -> float:
    """
    Calculate ramped weight based on specified shape and progress
    
    Args:
        initial_weight: Starting weight
        final_weight: Final weight
        shape: Shape of the ramp ('linear', 'cubic', 'sigmoid', 'cosine', 'exponential')
        progress: Progress from 0.0 to 1.0
        rate: Rate of change (used for 'sigmoid' and 'exponential' shapes)
        centre: Centre point (used for 'sigmoid' shape)
        f: Custom function for 'custom' shape, should accept (initial_weight, final_weight, progress)

    Returns:
        Ramped weight value
    """
    if shape == 'linear':
        return initial_weight + (final_weight - initial_weight) * progress
    elif shape == 'cubic':
        return initial_weight + (final_weight - initial_weight) * (progress ** 3)
    elif shape == 'sigmoid':
        return initial_weight + (final_weight - initial_weight) * (1 / (1 + np.exp(-rate * (progress - centre))))
    elif shape == 'cosine':
        return initial_weight + (final_weight - initial_weight) * (0.5 * (1 - np.cos(np.pi * progress)))
    elif shape == 'exponential':
        return initial_weight + (final_weight - initial_weight) * (1 - np.exp(-rate * progress))
    elif shape == 'constant':
        assert initial_weight == final_weight, "For constant shape, initial and final weights must be equal."
        return initial_weight
    elif shape == 'custom':
        assert f is not None, "For custom shape, a function must be provided."
        return f(initial_weight, final_weight, progress)
    else:
        raise ValueError(f"Unknown shape: {shape}. Supported shapes: linear, cubic, sigmoid, cosine, exponential, constant, custom.")


def fit_sticky_hmm_one_pass(
    model, 
    dataset, 
    device, 
    hmm: StickyHDPHMMVI, 
    offline: bool = True,
    streaming_rho: float = 1.0, 
    max_iters: int = 10,
    elbo_drop_tol: float = 10.0,
    optimize_pi_every_n_steps: int = 5,
    pi_iters: int = 10,
    pi_lr: float = 0.001,
    max_batches: int | None = None, 
    batch_multiples: int = 1,
    logger=None, 
    use_wandb: bool = False):
    """
    Run one E-step pass: update sticky-HDP-HMM variational posteriors using encoder outputs.
    
    Args:
        model: VAE model for encoding
        dataset: Dataset to process
        device: Device to run on
        hmm: HMM model to update
        offline: Whether to use offline mode
        streaming_rho: Streaming parameter
        max_iters: Maximum iterations for HMM update
        elbo_drop_tol: ELBO drop tolerance
        optimize_pi_every_n_steps: How often to optimize pi
        pi_iters: Number of pi iterations
        pi_lr: Learning rate for pi optimization
        max_batches: Maximum number of batches to process
        batch_multiples: Number of batches to concatenate along time dimension (creates [B, batch_multiples*T, ...])
        logger: Logger instance
        use_wandb: Whether to use wandb logging
    """
    model.eval()
    n_batches = len(dataset) if max_batches is None else min(len(dataset), max_batches)
    # assert the dataset is in [B, T, ...] format
    B, T = dataset[0]['tty_chars'].shape[:2]
    
    # Adjust effective number of batches based on batch_multiples
    effective_batches = n_batches // batch_multiples
    if n_batches % batch_multiples != 0:
        if logger:
            logger.warning(f"Number of batches ({n_batches}) not divisible by batch_multiples ({batch_multiples}). "
                         f"Using {effective_batches} effective batches, ignoring {n_batches % batch_multiples} batches.")
    
    # Create progress bar for HMM E-step
    dataset_slice = dataset[:effective_batches * batch_multiples] if effective_batches * batch_multiples < len(dataset) else dataset
    
    with tqdm(range(effective_batches), desc="HMM E-step", unit="multi-batch") as pbar:
        for multi_batch_idx in pbar:
            # Collect multiple batches to concatenate
            multi_batch_data = []
            for sub_idx in range(batch_multiples):
                batch_idx = multi_batch_idx * batch_multiples + sub_idx
                if batch_idx >= len(dataset_slice):
                    break
                multi_batch_data.append(dataset_slice[batch_idx])
            
            if not multi_batch_data:
                continue
                
            # Process each batch and collect encoded outputs
            mu_list = []
            var_list = []
            F_list = []
            valid_list = []
            
            for batch in multi_batch_data:
                batch_dev = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        x = v.to(device, non_blocking=True)
                        if x.dim() >= 3 and k not in ('original_batch_shape',):
                            batch_dev[k] = x.view(B*T, *x.shape[2:])
                        else:
                            batch_dev[k] = x
                    else:
                        batch_dev[k] = v
                
                # forward encode only (keep no_grad for model inference)
                with torch.no_grad():
                    out = model(batch_dev)  # returns 'mu', 'logvar', optionally 'lowrank_factors'
                    mu    = out['mu'].detach()  # Detach from computation graph
                    logvar= out['logvar'].detach()  # Detach from computation graph
                    F     = out.get('lowrank_factors', None)
                    if F is not None:
                        F = F.detach()  # Detach from computation graph
                    valid = batch_dev['valid_screen'].view(B,T)
                    # reshape to [B,T,...]
                    mu_bt = mu.view(B,T,-1)
                    var_bt = logvar.exp().clamp_min(1e-6).view(B,T,-1)
                    F_bt  = None if F is None else F.view(B,T,F.size(-2),F.size(-1))
                
                # Collect outputs for concatenation
                mu_list.append(mu_bt)
                var_list.append(var_bt)
                if F_bt is not None:
                    F_list.append(F_bt)
                valid_list.append(valid)
            
            # Concatenate along time dimension to create [B, batch_multiples*T, ...]
            mu_combined = torch.cat(mu_list, dim=1)  # [B, batch_multiples*T, D]
            var_combined = torch.cat(var_list, dim=1)  # [B, batch_multiples*T, D]
            F_combined = torch.cat(F_list, dim=1) if F_list else None  # [B, batch_multiples*T, ...]
            valid_combined = torch.cat(valid_list, dim=1)  # [B, batch_multiples*T]

            # HMM update with combined batch
            hmm_out = hmm.update(mu_combined, var_combined, F_combined, mask=valid_combined, 
                               max_iters=(1 if multi_batch_idx < 0 else max_iters), elbo_drop_tol=elbo_drop_tol, rho=streaming_rho, 
                               optimize_pi=(multi_batch_idx > -1 and (multi_batch_idx + 1) % optimize_pi_every_n_steps == 0), 
                               pi_steps=pi_iters, pi_lr=pi_lr, offline=offline)

            # Extract ELBO from HMM update
            inner_elbo = hmm_out.get('inner_elbo', torch.tensor(float('nan')))
            elbo_history = hmm_out.get('elbo_history', torch.tensor([]))
            n_iterations = hmm_out.get('n_iterations', 0)
            
            # Log to wandb if enabled
            if use_wandb and torch.isfinite(inner_elbo):
                log_hmm_elbo_to_wandb(multi_batch_idx, inner_elbo.item(), elbo_history, n_iterations, use_wandb)
            
            # Compute diagnostics every few multi-batches to monitor progress
            if (multi_batch_idx + 1) % 1 == 0 or multi_batch_idx == 0:
                with torch.no_grad():
                    diag_results = hmm.diagnostics(
                        mu_t=mu_combined,
                        diag_var_t=var_combined,
                        F_t=F_combined,
                        mask=valid_combined
                    )
                    
                    top5_skills = diag_results['top5_idx'].tolist()
                    top5_probs = diag_results['top5_pi'].tolist()
                    logger.info(f"[E-step] Multi-batch {multi_batch_idx+1}/{effective_batches} (x{batch_multiples} batches) - HMM Diagnostics:")
                    logger.info(f"  - Avg log-likelihood per step: {diag_results['avg_loglik_per_step']:.4f}")
                    logger.info(f"  - Inner ELBO (final): {inner_elbo:.4f}")
                    if len(elbo_history) > 1:
                        elbo_improve = elbo_history[-1] - elbo_history[0]
                        logger.info(f"  - ELBO progression ({n_iterations} iters): {format_elbo_progression(elbo_history)} (Δ={elbo_improve.item():.4f})")
                    logger.info(f"  - Effective number of skills: {diag_results['effective_K']:.2f}")
                    logger.info(f"  - State entropy: {diag_results['state_entropy']:.3f}")
                    logger.info(f"  - Transition stickiness: {diag_results['stickiness_diag_mean']:.3f}")
                    logger.info(f"  - Top 5 skills: {top5_skills} (probs: {[f'{p:.3f}' for p in top5_probs]})")
                    logger.info(f"  - Combined time length: {mu_combined.shape[1]} (original T={T}, multiples={batch_multiples})")
            else:
                # Update progress bar with basic info including ELBO
                elbo_change = ""
                if len(elbo_history) > 1:
                    elbo_improve = elbo_history[-1] - elbo_history[0]
                    elbo_change = f" (Δ{elbo_improve.item():+.2f})"
                pbar.set_postfix({
                    'multi_batch': f"{multi_batch_idx+1}/{effective_batches}",
                    'time_len': f"{mu_combined.shape[1]}",
                    'rho': f"{streaming_rho:.3f}",
                    'elbo': f"{inner_elbo:.2f}{elbo_change}",
                    'iters': f"{n_iterations}"
                })
                
                if logger and (multi_batch_idx+1) % 50 == 0:
                    logger.info(f"[E-step] processed {multi_batch_idx+1}/{effective_batches} multi-batches")
    
    return hmm

def fit_sticky_hmm_with_batch_accumulation(
    model, 
    dataset, 
    device, 
    hmm: StickyHDPHMMVI, 
    max_batches: int | None = None,
    pi_iters: int = 10,
    pi_lr: float = 0.001,
    logger=None, 
    use_wandb: bool = False):
    """
    Run one E-step pass with batch accumulation: freeze HMM parameters and do a full pass
    over all batched data, accumulating sufficient statistics, then perform a single
    batch update of NIW and Dirichlet posteriors and optimize π once from aggregated r1.
    
    This is useful for getting a clean batch estimate without streaming updates.
    
    Args:
        model: VAE model for encoding
        dataset: Dataset to process
        device: Device to run on
        hmm: HMM model to update (will be frozen during accumulation)
        max_batches: Maximum number of batches to process (None = all batches)
        pi_iters: Number of π optimization iterations
        pi_lr: Learning rate for π optimization
        logger: Logger instance
        use_wandb: Whether to use wandb logging
    """
    model.eval()
    n_batches = len(dataset) if max_batches is None else min(len(dataset), max_batches)
    B, T = dataset[0]['tty_chars'].shape[:2]
    
    if logger:
        logger.info(f"🔄 Starting batch accumulation E-step with {n_batches} batches")
    
    # Get HMM dimensions
    Kp1 = hmm.niw.mu.shape[0]  # K+1 states
    D = hmm.niw.mu.shape[1]    # latent dimension
    
    # Initialize accumulators for sufficient statistics
    accumulated_counts = torch.zeros(Kp1, Kp1, device=device, dtype=hmm.niw.mu.dtype)
    accumulated_Nk = torch.zeros(Kp1, device=device, dtype=hmm.niw.mu.dtype)
    accumulated_M1 = torch.zeros(Kp1, D, device=device, dtype=hmm.niw.mu.dtype)
    accumulated_M2 = torch.zeros(Kp1, D, D, device=device, dtype=hmm.niw.mu.dtype)
    accumulated_r1_sum = torch.zeros(Kp1, device=device, dtype=hmm.niw.mu.dtype)
    total_sequences = 0
    
    # Freeze HMM parameters - get current state for consistent FB passes
    with torch.no_grad():
        # Get current parameter values (these will remain fixed during accumulation)
        current_u_beta = hmm.u_beta.clone()
        current_phi = hmm.dir.phi.clone()
        
        # Calculate derived parameters once
        current_pi_full = hmm._calc_Epi(current_u_beta)
        current_ElogA = hmm._calc_ElogA(current_phi)
        current_log_pi = torch.log(torch.clamp(current_pi_full, min=1e-30))
        
        # Get current NIW parameters for emission likelihoods
        current_mu = hmm.niw.mu.clone()
        current_kappa = hmm.niw.kappa.clone()
        current_Psi = hmm.niw.Psi.clone()
        current_nu = hmm.niw.nu.clone()
    
    # Process all batches with frozen parameters
    with tqdm(range(n_batches), desc="Batch accumulation E-step", unit="batch") as pbar:
        for batch_idx in pbar:
            batch = dataset[batch_idx]
            
            # Move batch to device and reshape for encoding
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    x = v.to(device, non_blocking=True)
                    if x.dim() >= 3 and k not in ('original_batch_shape',):
                        batch_dev[k] = x.view(B*T, *x.shape[2:])
                    else:
                        batch_dev[k] = x
                else:
                    batch_dev[k] = v
            
            # Encode with VAE (fixed parameters)
            with torch.no_grad():
                out = model(batch_dev)
                mu = out['mu'].detach()
                logvar = out['logvar'].detach()
                F = out.get('lowrank_factors', None)
                if F is not None:
                    F = F.detach()
                
                valid = batch_dev['valid_screen'].view(B, T)
                
                # Reshape to [B, T, ...] format
                mu_bt = mu.view(B, T, -1)
                var_bt = logvar.exp().clamp_min(1e-6).view(B, T, -1)
                F_bt = None if F is None else F.view(B, T, F.size(-2), F.size(-1))
            
            # Compute emission log-likelihoods with frozen NIW parameters
            with torch.no_grad():
                logB = hmm.expected_emission_loglik(
                    current_mu, current_kappa, current_Psi, current_nu,
                    mu_bt, var_bt, F_bt, valid
                )  # [B, T, Kp1]
            
            # Forward-backward for each sequence in batch and accumulate statistics
            for b in range(B):
                with torch.no_grad():
                    rhat, xihat, ll = hmm.forward_backward(current_log_pi, current_ElogA, logB[b])
                    
                    # Compute sufficient statistics for this sequence
                    Nk, M1, M2 = hmm._moments_from_encoder(
                        mu_bt[b],
                        rhat,
                        (var_bt[b] if var_bt is not None else None),
                        (F_bt[b] if F_bt is not None else None),
                        (valid[b] if valid is not None else None)
                    )
                    
                    # Accumulate statistics
                    accumulated_Nk += Nk
                    accumulated_M1 += M1
                    accumulated_M2 += M2
                    
                    # Handle transition counts
                    if valid is not None:
                        m = valid[b]
                        pair_m = (m[:-1] * m[1:]).view(-1, 1, 1)  # [T-1, 1, 1]
                        xihat = xihat * pair_m  # zero out transitions from/to masked frames
                        t0 = int(torch.nonzero(m, as_tuple=False)[0]) if m.any() else 0
                    else:
                        t0 = 0
                    
                    accumulated_counts += xihat.sum(dim=0)
                    accumulated_r1_sum += rhat[t0]
                    total_sequences += 1
            
            # Update progress bar
            pbar.set_postfix({
                'batch': f"{batch_idx+1}/{n_batches}",
                'sequences': f"{total_sequences}",
                'avg_Nk': f"{accumulated_Nk.mean():.2f}"
            })
    
    # Now perform batch update with accumulated statistics
    if logger:
        logger.info(f"📊 Performing batch update with accumulated statistics from {total_sequences} sequences")
    
    with torch.no_grad():
        # 1. Update NIW posteriors first (observation parameters)
        hmm._update_moments(accumulated_Nk, accumulated_M1, accumulated_M2, accumulated_counts)
        
        # Calculate new NIW posterior parameters
        mu_hat, kappa_hat, Psi_hat, nu_hat = hmm._calc_NIW_posterior(
            accumulated_Nk, accumulated_M1, accumulated_M2
        )
        
        hmm._update_NIW(mu_hat, kappa_hat, Psi_hat, nu_hat)
        
        # 2. Update φ (Dirichlet transition posteriors) with current π
        current_pi = hmm._calc_Epi(hmm.u_beta)
        updated_phi = hmm._calc_dir_posterior(accumulated_counts, current_pi)
        hmm._update_transitions(updated_phi)

        # 3. Optimize π from aggregated r1 (single optimization from all data)
        if total_sequences > 0:
            r1_mean = (accumulated_r1_sum / total_sequences).clamp_min(1e-12)
            r1_mean = r1_mean / r1_mean.sum()
            
            if logger:
                logger.info(f"🎯 Optimizing π from aggregated r1: {r1_mean.cpu().numpy()}")
            
            # Get current transition parameters for π optimization
            ElogA_for_pi = hmm._calc_ElogA(hmm.dir.phi)
            
            optimized_u_beta = hmm._optimize_u_beta(
                hmm.u_beta, r1_mean, ElogA_for_pi,
                hmm.p.alpha, hmm.p.kappa, hmm.p.gamma, hmm.p.K,
                steps=pi_iters, lr=pi_lr
            )
            
            # Update β parameters
            hmm._update_u_beta(optimized_u_beta)
    
    # Final diagnostics and logging
    with torch.no_grad():
        # Compute final ELBO/diagnostics with updated parameters
        final_pi = hmm._calc_Epi(hmm.u_beta)
        final_ElogA = hmm._calc_ElogA(hmm.dir.phi)

        # Get a sample batch for diagnostics
        sample_batch = dataset[0]
        sample_batch_dev = {}
        for k, v in sample_batch.items():
            if isinstance(v, torch.Tensor):
                x = v.to(device, non_blocking=True)
                if x.dim() >= 3 and k not in ('original_batch_shape',):
                    sample_batch_dev[k] = x.view(B*T, *x.shape[2:])
                else:
                    sample_batch_dev[k] = x
            else:
                sample_batch_dev[k] = v
        
        # Quick diagnostics on sample
        with torch.no_grad():
            out = model(sample_batch_dev)
            mu = out['mu'].view(B, T, -1)
            var = out['logvar'].exp().clamp_min(1e-6).view(B, T, -1)
            valid = sample_batch_dev['valid_screen'].view(B, T)
            
            diag_results = hmm.diagnostics(mu_t=mu, diag_var_t=var, F_t=None, mask=valid)
    
    if logger:
        logger.info(f"✅ Batch accumulation complete:")
        logger.info(f"  - Processed {total_sequences} sequences from {n_batches} batches")
        logger.info(f"  - Final π: {final_pi.cpu().numpy()}")
        logger.info(f"  - Effective number of skills: {diag_results['effective_K']:.2f}")
        logger.info(f"  - State entropy: {diag_results['state_entropy']:.3f}")
    
    # Log to wandb if enabled
    if use_wandb and torch.isfinite(torch.tensor(0.0)):  # Always log since no ELBO tracking
        if WANDB_AVAILABLE:
            wandb.log({
                "batch_accumulation/total_sequences": total_sequences,
                "batch_accumulation/total_batches": n_batches,
                "batch_accumulation/final_pi": final_pi.cpu().numpy().tolist(),
                "batch_accumulation/effective_K": diag_results['effective_K'],
                "batch_accumulation/state_entropy": diag_results['state_entropy']
            })

def fit_sticky_hmm_with_game_grouped_data(
    model, 
    grouped_data: Dict, 
    device, 
    hmm: StickyHDPHMMVI, 
    offline: bool = True,
    streaming_rho: float = 1.0, 
    max_iters: int = 10,
    elbo_drop_tol: float = 10.0,
    optimize_pi_every_n_steps: int = 5,
    pi_iters: int = 10,
    pi_lr: float = 0.001,
    max_games: int | None = None,
    logger=None, 
    use_wandb: bool = False):
    """
    Run one E-step pass using game-grouped data: update sticky-HDP-HMM variational posteriors 
    by processing one game at a time instead of batched data.
    
    Args:
        model: VAE model for encoding
        grouped_data: Dictionary with game_id -> {'sequence_data': {...}, 'total_length': int, 'is_complete': bool}
        device: Device to run on
        hmm: HMM model to update
        offline: Whether to use offline mode
        streaming_rho: Streaming parameter
        max_iters: Maximum iterations for HMM update
        elbo_drop_tol: ELBO drop tolerance
        optimize_pi_every_n_steps: How often to optimize pi
        pi_iters: Number of pi iterations
        pi_lr: Learning rate for pi optimization
        max_games: Maximum number of games to process (None = all games)
        logger: Logger instance
        use_wandb: Whether to use wandb logging
    """
    model.eval()
    
    # Get list of games, optionally limiting the number
    game_ids = list(grouped_data.keys())
    if max_games is not None:
        game_ids = game_ids[:max_games]
    
    n_games = len(game_ids)
    if logger:
        logger.info(f"Processing {n_games} games for E-step with game-grouped data")
    
    # Create progress bar for HMM E-step
    with tqdm(enumerate(game_ids), total=n_games, desc="HMM E-step (game-by-game)", unit="game") as pbar:
        for game_idx, game_id in pbar:
            game_info = grouped_data[game_id]
            sequence_data = game_info['sequence_data']
            total_length = game_info['total_length']
            
            # Skip games that are too short
            if total_length < 2:
                continue
            
            # Prepare the game data for encoding in a single loop
            # The sequence_data contains tensors of shape [T, ...] where T is the game length
            batch_flat = {}
            batch_dev = {}  
            
            for k, v in sequence_data.items():
                if isinstance(v, torch.Tensor):
                    # Move to device first
                    v_device = v.to(device, non_blocking=True)
                    
                    if v_device.dim() >= 2:
                        # Create [1, T, ...] version for batch_dev
                        v_batched = v_device.unsqueeze(0)  # [1, T, ...]
                        batch_dev[k] = v_batched
                        
                        # Determine final shape for batch_flat
                        if v_batched.dim() >= 3 and k not in ('valid_screen', 'gameids'):
                            # Flatten spatial dimensions: [1, T, ...] -> [T, ...]
                            T = v_batched.shape[1]
                            batch_flat[k] = v_batched.view(T, *v_batched.shape[2:])
                        elif k == 'valid_screen':
                            # Keep [1, T] shape for valid_screen
                            batch_flat[k] = v_batched
                        else:
                            batch_flat[k] = v_batched
                    else:
                        # 1D tensors - no unsqueeze needed
                        batch_dev[k] = v_device
                        batch_flat[k] = v_device
                else:
                    # Non-tensor values
                    batch_dev[k] = v
                    batch_flat[k] = v
            
            # Encode the game sequence
            with torch.no_grad():
                out = model(batch_flat)  # returns 'mu', 'logvar', optionally 'lowrank_factors'
                mu = out['mu'].detach()  # [T, D]
                logvar = out['logvar'].detach()  # [T, D]
                F = out.get('lowrank_factors', None)
                if F is not None:
                    F = F.detach()  # [T, ...]
                
                # Get valid mask
                if 'valid_screen' in batch_dev:
                    valid = batch_dev['valid_screen']  # [1, T]
                else:
                    # If no valid mask, assume all timesteps are valid
                    valid = torch.ones(1, total_length, dtype=torch.bool, device=device)
                
                # Ensure valid mask has correct shape and all True values for complete games
                if valid.dim() == 1:
                    valid = valid.unsqueeze(0)  # Make it [1, T]
                
                # For game-grouped data, typically all timesteps in a game are valid
                if valid.shape[1] != total_length:
                    # Adjust mask to match the sequence length
                    valid = torch.ones(1, total_length, dtype=torch.bool, device=device)
                
                # Reshape back to [1, T, D] format for HMM
                mu_bt = mu.unsqueeze(0)  # [1, T, D]
                var_bt = logvar.exp().clamp_min(1e-6).unsqueeze(0)  # [1, T, D]
                F_bt = None if F is None else F.unsqueeze(0)  # [1, T, ...]
            
            # Update HMM with this single game
            hmm_out = hmm.update(
                mu_bt, var_bt, F_bt, 
                mask=valid, 
                max_iters=1 if game_idx < 10 else max_iters, 
                elbo_drop_tol=elbo_drop_tol, 
                rho=streaming_rho, 
                optimize_pi=(game_idx > 9 and (game_idx + 1) % optimize_pi_every_n_steps == 0), 
                pi_steps=pi_iters, 
                pi_lr=pi_lr, 
                offline=offline
            )
            
            # Extract ELBO from HMM update
            inner_elbo = hmm_out.get('inner_elbo', torch.tensor(float('nan')))
            elbo_history = hmm_out.get('elbo_history', torch.tensor([]))
            n_iterations = hmm_out.get('n_iterations', 0)
            
            # Log to wandb if enabled
            if use_wandb and torch.isfinite(inner_elbo):
                log_hmm_elbo_to_wandb(game_idx, inner_elbo.item(), elbo_history, n_iterations, use_wandb)
            
            # Compute diagnostics for detailed logging
            if (game_idx + 1) % 1 == 0 or game_idx == 0:
                with torch.no_grad():
                    diag_results = hmm.diagnostics(
                        mu_t=mu_bt,
                        diag_var_t=var_bt,
                        F_t=F_bt,
                        mask=valid
                    )
                    
                    top5_skills = diag_results['top5_idx'].tolist()
                    top5_probs = diag_results['top5_pi'].tolist()
                    status = "complete" if game_info['is_complete'] else "incomplete"
                    
                    if logger:
                        logger.info(f"[E-step] Game {game_id} ({game_idx+1}/{n_games}) - HMM Diagnostics:")
                        logger.info(f"  - Game length: {total_length}, status: {status}")
                        logger.info(f"  - Avg log-likelihood per step: {diag_results['avg_loglik_per_step']:.4f}")
                        logger.info(f"  - Inner ELBO (final): {inner_elbo:.4f}")
                        if len(elbo_history) > 1:
                            elbo_improve = elbo_history[-1] - elbo_history[0]
                            logger.info(f"  - ELBO progression ({n_iterations} iters): {format_elbo_progression(elbo_history)} (Δ={elbo_improve.item():.4f})")
                        logger.info(f"  - Effective number of skills: {diag_results['effective_K']:.2f}")
                        logger.info(f"  - State entropy: {diag_results['state_entropy']:.3f}")
                        logger.info(f"  - Transition stickiness: {diag_results['stickiness_diag_mean']:.3f}")
                        logger.info(f"  - Top 5 skills: {top5_skills} (probs: {[f'{p:.3f}' for p in top5_probs]})")
            else:
                # Update progress bar with basic info including ELBO
                status = "complete" if game_info['is_complete'] else "incomplete"
                elbo_change = ""
                if len(elbo_history) > 1:
                    elbo_improve = elbo_history[-1] - elbo_history[0]
                    elbo_change = f" (Δ{elbo_improve.item():+.2f})"
                
                pbar.set_postfix({
                    'game_id': f"{game_id}",
                    'length': f"{total_length}",
                    'status': status,
                    'elbo': f"{inner_elbo:.2f}{elbo_change}",
                    'iters': f"{n_iterations}"
                })
                
                # Log summary for batches of games
                if (game_idx + 1) % 50 == 0:
                    if logger:
                        logger.info(f"[E-step] Processed {game_idx+1}/{n_games} games")
    
    if logger:
        logger.info(f"[E-step] Completed processing {n_games} games with game-grouped data")
    
    return hmm

def load_datasets(
    train_file: str, 
    test_file: str,                     
    dbfilename: str = 'ttyrecs.db',
    batch_size: int = 32, 
    sequence_size: int = 32, 
    training_batches: int = 100,
    testing_batches: int = 20,
    max_training_batches: int = 100,
    max_testing_batches: int = 20,
    training_game_ids: List[int] | None = None,
    testing_game_ids: List[int] | None = None,
    data_cache_dir: str = "data_cache",
    force_recollect: bool = False,
    logger: logging.Logger = None
) -> Tuple[List, List]:
    """
    Load training and testing datasets with caching
    
    Args:
        train_file: Path to the training samples
        test_file: Path to the testing samples
        dbfilename: Path to the NetHack Learning Dataset database file
        batch_size: Training batch size
        sequence_size: Sequence length for temporal data
        training_batches: Number of training batches to use
        testing_batches: Number of testing batches to use
        max_training_batches: Maximum training batches to collect
        max_testing_batches: Maximum testing batches to collect
        training_game_ids: Specific game IDs for training (optional)
        testing_game_ids: Specific game IDs for testing (optional)
        data_cache_dir: Directory to cache processed data
        force_recollect: Force data recollection even if cache exists
        logger: Logger instance
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create adapter and datasets with caching
    adapter = BLStatsAdapter()
    collector = NetHackDataCollector(dbfilename)
    
    # Create cache directory
    os.makedirs(data_cache_dir, exist_ok=True)
    
    # Cache file names based on dataset parameters
    train_cache_file = os.path.join(data_cache_dir, f"{train_file}_b{batch_size}_s{sequence_size}_m{max_training_batches}.pt")
    test_cache_file = os.path.join(data_cache_dir, f"{test_file}_b{batch_size}_s{sequence_size}_m{max_testing_batches}.pt")

    # Collect or load training data
    logger.info(f"📊 Preparing training data...")
    train_dataset = collector.collect_or_load_data(
        dataset_name=train_file,
        adapter=adapter,
        save_path=train_cache_file,
        max_batches=max_training_batches,
        batch_size=batch_size,
        seq_length=sequence_size,
        force_recollect=force_recollect,
        game_ids=training_game_ids
    )
    train_dataset = train_dataset[:training_batches] if len(train_dataset) > training_batches else train_dataset
    
    # Collect or load testing data
    logger.info(f"📊 Preparing testing data...")
    test_dataset = collector.collect_or_load_data(
        dataset_name=test_file,
        adapter=adapter,
        save_path=test_cache_file,
        max_batches=max_testing_batches,
        batch_size=batch_size,
        seq_length=sequence_size,
        force_recollect=force_recollect,
        game_ids=testing_game_ids
    )
    test_dataset = test_dataset[:testing_batches] if len(test_dataset) > testing_batches else test_dataset
    
    logger.info(f"✅ Datasets loaded: {len(train_dataset)} train batches, {len(test_dataset)} test batches")
    return train_dataset, test_dataset


def train_multimodalhack_vae(
    train_dataset: List,
    test_dataset: List,
    config: VAEConfig = None,
    epochs: int = 10, 
    max_learning_rate: float = 1e-3,
    device: str = None, 
    logger: logging.Logger = None,
    shuffle_batches: bool = True,
    shuffle_within_batch: bool = False,
    
    # Mixed precision parameters
    use_bf16: bool = False,
    
    # Custom KL beta function (optional override)
    custom_kl_beta_function: Optional[Callable[[float, float, float], float]] = None,
    
    # Learning rate scheduler parameters
    lr_scheduler: str = "onecycle",  # "onecycle" | "constant"
    
    # Model saving and checkpointing parameters
    save_path: str = "models/nethack-vae.pth",
    save_checkpoints: bool = False,
    checkpoint_dir: str = "checkpoints",
    save_every_n_epochs: int = 2,
    keep_last_n_checkpoints: int = 2,
    
    # HuggingFace integration parameters
    upload_to_hf: bool = False,
    hf_repo_name: str = None,
    hf_token: str = None,
    hf_private: bool = True,
    hf_upload_artifacts: bool = True,
    hf_upload_directly: bool = True,
    hf_upload_checkpoints: bool = False,
    hf_model_card_data: Dict = None,
    
    # Resume training parameters
    resume_checkpoint_path: str = None,
    
    # Weights & Biases monitoring parameters
    use_wandb: bool = True,
    wandb_project: str = "nethack-vae",
    wandb_entity: str = None,
    wandb_run_name: str = None,
    wandb_tags: List[str] = None,
    wandb_notes: str = None,
    log_every_n_steps: int = 10,
    log_model_architecture: bool = True,
    log_gradients: bool = False,
    
    # Early stopping parameters
    early_stopping: bool = True,
    early_stopping_patience: int = 3,
    early_stopping_min_delta: float = 0.01,
    
    # sticky-HDP-HMM parameters
    hmm: Optional[StickyHDPHMMVI] = None,
    use_hmm_prior: bool = False
    ) -> Tuple[MultiModalHackVAE, List[float], List[float]]:
    """
    Train MultiModalHackVAE on pre-loaded NetHack datasets with adaptive loss weighting

    Args:
        train_dataset: Pre-loaded training dataset (list of batches)
        test_dataset: Pre-loaded testing dataset (list of batches)
        config: VAEConfig object containing model configuration and training hyperparameters.
                If None, will create default config. For resuming from checkpoint, config from checkpoint takes precedence.
        epochs: Number of training epochs
        max_learning_rate: max learning rate for optimizer
        device: Device to use ('cuda' or 'cpu')
        logger: Logger instance
        shuffle_batches: Whether to shuffle training batches at the start of each epoch
        shuffle_within_batch: Whether to shuffle games within each batch (preserves temporal order within each game)
        use_bf16: Whether to use BF16 mixed precision training for memory efficiency
        custom_kl_beta_function: Optional custom function for KL beta ramping (overrides config beta curves)
        lr_scheduler: Learning rate scheduler type. Options: "onecycle" (default), "constant"
        save_path: Path to save the trained model
        save_checkpoints: Whether to save checkpoints during training
        checkpoint_dir: Directory to save checkpoints
        save_every_n_epochs: Save checkpoint every N epochs
        keep_last_n_checkpoints: Keep only last N checkpoints, delete older ones
        upload_to_hf: Whether to upload model to HuggingFace Hub
        hf_repo_name: HuggingFace repository name for uploading
        hf_token: HuggingFace authentication token
        hf_private: Whether to make the uploaded model private
        hf_upload_artifacts: Whether to upload artifacts (e.g. datasets)
        hf_upload_directly: Whether to upload model directly or via artifacts
        hf_upload_checkpoints: Whether to upload checkpoints to HuggingFace
        hf_model_card_data: Additional metadata for HuggingFace model card
        resume_checkpoint_path: Path to resume training from
        use_wandb: Whether to use Weights & Biases for monitoring
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity (team/user)
        wandb_run_name: Name for the Weights & Biases run
        wandb_tags: Tags for the Weights & Biases run
        wandb_notes: Notes for the Weights & Biases run
        log_every_n_steps: Log metrics every N steps
        log_model_architecture: Whether to log model architecture to Weights & Biases
        log_gradients: Whether to log gradients to Weights & Biases
        early_stopping: Whether to enable early stopping based on test loss
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped
        early_stopping_min_delta: Minimum relative change in test loss to qualify as an improvement
        hmm: Optional HMM model for skill-based priors
        use_hmm_prior: Whether to use HMM prior instead of standard normal prior

    Returns:
        Tuple of (trained_model, train_losses, test_losses)
    """
    if device is None:
        device = torch.device('cpu')  # Use CPU for debugging
    else:
        # Ensure device is a torch.device object, not a string
        device = torch.device(device)

    # Setup VAEConfig
    if config is None:
        config = VAEConfig()

    # Setup logging
    if logger is None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('training.log')
            ]
        )
        logger = logging.getLogger(__name__)
        
    batch_size, sequence_size = (train_dataset[0]['game_chars'].shape[0], train_dataset[0]['game_chars'].shape[1]) if len(train_dataset) > 0 else (None, None)

    # Initialize Weights & Biases if requested
    if use_wandb and WANDB_AVAILABLE:
        # Prepare configuration for wandb
        wandb_config = {
            "epochs": epochs,
            "max_learning_rate": max_learning_rate,
            "train_batches": len(train_dataset),
            "test_batches": len(test_dataset),
            "batch_size": batch_size,
            "sequence_size": sequence_size,
            "device": str(device),
            "use_bf16": use_bf16,
            "lr_scheduler": lr_scheduler,
            "shuffle_batches": shuffle_batches,
            "shuffle_within_batch": shuffle_within_batch,
            "vae_config": {
                "latent_dim": config.latent_dim,
                "encoder_dropout": config.encoder_dropout,
                "decoder_dropout": config.decoder_dropout,
                "initial_mi_beta": config.initial_mi_beta,
                "final_mi_beta": config.final_mi_beta,
                "mi_beta_shape": config.mi_beta_shape,
                "initial_tc_beta": config.initial_tc_beta,
                "final_tc_beta": config.final_tc_beta,
                "tc_beta_shape": config.tc_beta_shape,
                "initial_dw_beta": config.initial_dw_beta,
                "final_dw_beta": config.final_dw_beta,
                "dw_beta_shape": config.dw_beta_shape,
                "initial_prior_blend_alpha": config.initial_prior_blend_alpha,
                "final_prior_blend_alpha": config.final_prior_blend_alpha,
                "prior_blend_shape": config.prior_blend_shape,
                "warmup_epoch_ratio": config.warmup_epoch_ratio,
                "free_bits": config.free_bits,
                "focal_loss_alpha": config.focal_loss_alpha,
                "focal_loss_gamma": config.focal_loss_gamma
            },
            "adaptive_weighting": {
                "initial_mi_beta": config.initial_mi_beta,
                "final_mi_beta": config.final_mi_beta,
                "mi_beta_shape": config.mi_beta_shape,
                "initial_tc_beta": config.initial_tc_beta,
                "final_tc_beta": config.final_tc_beta,
                "tc_beta_shape": config.tc_beta_shape,
                "initial_dw_beta": config.initial_dw_beta,
                "final_dw_beta": config.final_dw_beta,
                "dw_beta_shape": config.dw_beta_shape,
                "initial_prior_blend_alpha": config.initial_prior_blend_alpha,
                "final_prior_blend_alpha": config.final_prior_blend_alpha,
                "prior_blend_shape": config.prior_blend_shape,
                "warmup_epoch_ratio": config.warmup_epoch_ratio
            },
            "regularization": {
                "encoder_dropout": config.encoder_dropout,
                "decoder_dropout": config.decoder_dropout,
                "free_bits": config.free_bits,
                "focal_loss_alpha": config.focal_loss_alpha,
                "focal_loss_gamma": config.focal_loss_gamma
            },
            "checkpointing": {
                "save_checkpoints": save_checkpoints,
                "save_every_n_epochs": save_every_n_epochs,
                "keep_last_n_checkpoints": keep_last_n_checkpoints
            },
            "early_stopping": {
                "enabled": early_stopping,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta
            }
        }
        
        # Initialize wandb run
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=wandb_config,
            tags=wandb_tags,
            notes=wandb_notes,
            resume="allow" if resume_checkpoint_path else False
        )
        
        logger.info("Weights & Biases initialized")
        
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("⚠️  wandb requested but not available. Install with: pip install wandb")

    logger.info(f"🔥Training MultiModalHackVAE with {len(train_dataset)} train batches, {len(test_dataset)} test batches")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Sequence size: {sequence_size}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Mixed Precision: BF16 = {use_bf16}")
    logger.info(f"   LR Scheduler: {lr_scheduler}")
    logger.info(f"   Shuffle batches: {shuffle_batches}")
    logger.info(f"   Shuffle within batch: {shuffle_within_batch}")
    logger.info(f"   VAE Configuration:")
    logger.info(f"     - Latent dimension: {config.latent_dim}")
    logger.info(f"     - Encoder dropout: {config.encoder_dropout}")
    logger.info(f"     - Decoder dropout: {config.decoder_dropout}")
    logger.info(f"   Adaptive Loss Weighting:")
    logger.info(f"     - MI beta: {config.initial_mi_beta:.3f} → {config.final_mi_beta:.3f}")
    logger.info(f"     - TC beta: {config.initial_tc_beta:.3f} → {config.final_tc_beta:.3f}")
    logger.info(f"     - DW beta: {config.initial_dw_beta:.3f} → {config.final_dw_beta:.3f}")
    logger.info(f"     - Blend alpha: {config.initial_prior_blend_alpha:.3f} → {config.final_prior_blend_alpha:.3f}")
    logger.info(f"     - Warmup epochs: {int(config.warmup_epoch_ratio * epochs)} out of {epochs} total epochs")
    logger.info(f"   Free bits: {config.free_bits}")
    logger.info(f"   Focal loss: alpha={config.focal_loss_alpha}, gamma={config.focal_loss_gamma}")
    logger.info(f"   Early Stopping:")
    logger.info(f"     - Enabled: {early_stopping}")
    if early_stopping:
        logger.info(f"     - Patience: {early_stopping_patience} epochs")
        logger.info(f"     - Min delta: {early_stopping_min_delta:.6f}")
    if hmm is not None:
        logger.info(f"   HMM Prior: {'Enabled' if use_hmm_prior else 'Available but disabled'}")

    def get_adaptive_weights(global_step: int, total_steps: int, f: Optional[Callable[[float, float, float], float]]) -> Tuple[float, float, float, float]:
        """Calculate adaptive weights based on current global training step"""
        # Calculate progress based on global step for smoother transitions
        progress = min(global_step / max(total_steps - 1, 1), 1.0)
        
        # Mutual Information beta: very small initially, then gradually increase
        mi_beta = ramp_weight(initial_weight=config.initial_mi_beta, 
            final_weight=config.final_mi_beta, 
            shape=config.mi_beta_shape, 
            progress=progress,
            f=f
        )
        
        # Total Correlation beta: very small initially, then gradually increase
        tc_beta = ramp_weight(initial_weight=config.initial_tc_beta, 
            final_weight=config.final_tc_beta, 
            shape=config.tc_beta_shape, 
            progress=progress,
            f=f
        )
        
        # Dimension-wise KL beta: very small initially, then gradually increase
        dw_beta = ramp_weight(initial_weight=config.initial_dw_beta, 
            final_weight=config.final_dw_beta, 
            shape=config.dw_beta_shape, 
            progress=progress,
            f=f
        )
        
        # Prior blend alpha: blend between standard and HMM priors
        blend_alpha = ramp_weight(initial_weight=config.initial_prior_blend_alpha, 
            final_weight=config.final_prior_blend_alpha, 
            shape=config.prior_blend_shape, 
            progress=progress,
            f=f
        )
        
        # Log the adaptive weights (only occasionally to avoid spam)
        if global_step % 100 == 0:
            logger.debug(f"Step {global_step}/{total_steps} - Adaptive weights: mi_beta={mi_beta:.3f}, tc_beta={tc_beta:.3f}, dw_beta={dw_beta:.3f}, blend_alpha={blend_alpha:.3f}")

        return mi_beta, tc_beta, dw_beta, blend_alpha

    # Resume from checkpoint if specified
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        logger.info(f"🔄 Resuming from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        logger.info(f"   Resuming from epoch {start_epoch}/{epochs}")
        logger.info(f"   Previous train loss: {checkpoint['final_train_loss']:.4f}")
        logger.info(f"   Previous test loss: {checkpoint['final_test_loss']:.4f}")
        
        # Create VAEConfig from checkpoint data or use provided config
        if 'config' in checkpoint:
            # Load config from checkpoint and update with any provided overrides
            checkpoint_config = checkpoint['config']
            if config is not None:
                # Override checkpoint config with provided config (provided config takes precedence)
                logger.info("   Overriding checkpoint config with provided config")
            else:
                config = checkpoint_config
                logger.info("   Using config from checkpoint")
        elif config is None:
            # Fallback: create default config with deprecated parameters
            config = VAEConfig()
            logger.warning("   No config found in checkpoint, using default config")
        model = MultiModalHackVAE(config=config, logger=logger)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Initialize optimizer and step-based scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_learning_rate, weight_decay=1e-4)
        total_train_steps = len(train_dataset) * epochs
        warmup_steps = int(config.warmup_epoch_ratio * total_train_steps) if config.warmup_epoch_ratio > 0 else 0

        if lr_scheduler == "onecycle":
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=max_learning_rate, 
                total_steps=total_train_steps, 
                pct_start=config.warmup_epoch_ratio,
                anneal_strategy='cos',
                div_factor=2.0,
                final_div_factor=5.0,
                cycle_momentum=False
            )
        elif lr_scheduler == "constant":
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=total_train_steps)
        else:
            raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}. Choose from 'onecycle' or 'constant'")
            
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) if 'scheduler_state_dict' in checkpoint else None
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) if 'optimizer_state_dict' in checkpoint else None
        
        # Initialize loss tracking from checkpoint
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])
    else:
        start_epoch = 0
        if config is None:
            # If no config provided and not resuming, create default config
            logger.info("   No config provided, using default VAEConfig")
            config = VAEConfig()
        model = MultiModalHackVAE(config=config, logger=logger)
        model = model.to(device)
        
        # Initialize optimizer and step-based scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_learning_rate, weight_decay=1e-4)
        total_train_steps = len(train_dataset) * epochs
        warmup_steps = int(config.warmup_epoch_ratio * total_train_steps) if config.warmup_epoch_ratio > 0 else 0
        
        if lr_scheduler == "onecycle":
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=max_learning_rate, 
                total_steps=total_train_steps, 
                pct_start=config.warmup_epoch_ratio,
                anneal_strategy='cos',
                div_factor=2.0,
                final_div_factor=5.0,
                cycle_momentum=False
            )
        elif lr_scheduler == "constant":
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=total_train_steps)
        else:
            raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}. Choose from 'onecycle' or 'constant'")
        
        # Initialize loss tracking
        train_losses = []
        test_losses = []

    # Initialize GradScaler for mixed precision training (for both new and resumed training)
    scaler = torch.amp.GradScaler('cuda') if use_bf16 and device.type == 'cuda' else None
    
    # Restore scaler state if resuming from checkpoint and scaler is available
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path) and scaler is not None:
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"   Restored GradScaler state from checkpoint")
    
    # Log BF16 status
    if use_bf16 and device.type == 'cuda':
        logger.info(f"✨ BF16 mixed precision training enabled with GradScaler")
    elif use_bf16 and device.type != 'cuda':
        logger.warning(f"⚠️  BF16 requested but device is {device.type}, using FP32 instead")
    else:
        logger.info(f"🔧 Using FP32 precision training")

    # Log model architecture to wandb if requested
    if use_wandb and WANDB_AVAILABLE and log_model_architecture:
        wandb.watch(model, log_freq=log_every_n_steps, log_graph=True, log="all" if log_gradients else None)

    # Calculate total training steps for step-based adaptive weights and learning rate
    total_train_steps = len(train_dataset) * epochs
    warmup_steps = int(config.warmup_epoch_ratio * total_train_steps) if config.warmup_epoch_ratio > 0 else 0
    
    logger.info(f"Model has latent dimension: {config.latent_dim}")
    logger.info(f"🎯 Starting training for {epochs} epochs (starting from epoch {start_epoch})...")
    logger.info(f"   Total training steps: {total_train_steps}")
    logger.info(f"   Warmup steps: {warmup_steps}")
    
    # Initialize global step counter
    global_step = start_epoch * len(train_dataset)
    
    # Initialize early stopping variables
    best_test_loss = float('inf')
    best_model_state = None
    early_stopping_counter = 0
    best_epoch = -1
    
    for epoch in range(start_epoch, epochs):
        logger.info(f"🎯 Epoch {epoch+1}/{epochs} - Starting epoch...")
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "progress/overall": global_step / total_train_steps,
                "progress/warmup": min(global_step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0,
                "progress/epoch": epoch / epochs,
                "progress/global_step": global_step
            })
        
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0
        
        # Shuffle training batches for this epoch (if enabled)
        if shuffle_batches:
            shuffled_train_dataset = train_dataset[:]  # Create a proper copy
            random.shuffle(shuffled_train_dataset)
            logger.debug(f"Shuffled {len(shuffled_train_dataset)} training batches for epoch {epoch+1}")
            shuffled_test_dataset = test_dataset[:]  # Create a proper copy
            random.shuffle(shuffled_test_dataset)
            logger.debug(f"Shuffled {len(shuffled_test_dataset)} testing batches for epoch {epoch+1}")
        else:
            shuffled_train_dataset = train_dataset
            shuffled_test_dataset = test_dataset
        
        with tqdm(shuffled_train_dataset, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                # Move batch to device
                batch_device = {}
                for key, value in batch.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        batch_device[key] = value.to(device)
                    else:
                        batch_device[key] = value
                
                # Optional: shuffle within batch (shuffle games but keep temporal order)
                if shuffle_within_batch:
                    # Shuffle across B dimension (games) while preserving T dimension (temporal order)
                    # Do this before reshaping to [B*T, ...]
                    if 'game_chars' in batch_device:
                        B, T = batch_device['game_chars'].shape[:2]
                        game_shuffle_indices = torch.randperm(B)
                        
                        for key, value in batch_device.items():
                            if value is not None and isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                                # Shuffle across B dimension (games)
                                batch_device[key] = value[game_shuffle_indices]
                
                # Now reshape tensors from [B, T, ...] to [B*T, ...]
                for key, value in batch_device.items():
                    if value is not None and isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                        B, T = value.shape[:2]
                        remaining_dims = value.shape[2:]
                        batch_device[key] = value.view(B * T, *remaining_dims)
                
                # Store original batch dimensions for dynamics processing
                if 'game_chars' in batch:
                    B, T = batch['game_chars'].shape[:2]
                    batch_device['original_batch_shape'] = (B, T)
                    batch_device['batch_size'] = B
                
                # Forward pass with mixed precision if enabled
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(use_bf16 and device.type == 'cuda')):
                    model_output = model(batch_device)
                    
                    # ==== (M‑step) HMM prior cache for this batch (no gradients) ====
                    if use_hmm_prior and (hmm is not None):
                        with torch.no_grad():
                            B,T = batch_device['original_batch_shape']
                            valid_bt = batch_device['valid_screen'].view(B,T)
                            mu_bt    = model_output['mu'].view(B,T,-1)
                            var_bt   = model_output['logvar'].exp().clamp_min(1e-6).view(B,T,-1)
                            F_btr    = model_output.get('lowrank_factors', None)
                            F_bt     = None if F_btr is None else F_btr.view(B,T,F_btr.size(-2),F_btr.size(-1))
                            # emission expected log-likelihood per (b,t,k)
                            logB = StickyHDPHMMVI.expected_emission_loglik(hmm.niw.mu, hmm.niw.kappa, hmm.niw.Psi, hmm.niw.nu, mu_bt, var_bt, F_bt, mask=valid_bt)  # [B,T,K]
                            pi_star  = hmm._Epi()                              # [K]
                            ElogA    = hmm._ElogA()                            # [K,K]
                            log_pi   = torch.log(torch.clamp(pi_star, min=1e-30))
                            
                            # Process each sequence in the batch individually
                            r_hat_list = []
                            for b in range(B):
                                r_hat_b, xi_hat_b, ll_b = hmm.forward_backward(log_pi, ElogA, logB[b])
                                r_hat_list.append(r_hat_b)
                            r_hat = torch.stack(r_hat_list, dim=0)  # [B,T,K]
                            
                            # prior Gaussians
                            mu_k, E_Lambda, ElogdetLambda = hmm.get_emission_expectations() # mu_k:[K,D], E_Lambda:[K,D,D]
                            # log|Σ_k| = - log|Λ_k|  (approx with E[Λ])
                            logdet_Sigma_k = -torch.logdet(E_Lambda + 1e-6*torch.eye(E_Lambda.size(-1), device=E_Lambda.device)).to(mu_bt.dtype)
                            # flatten responsibilities to align with valid_screen
                            r_hat_flat = r_hat[valid_bt]   # [valid_B, K]
                        # stash cache into batch so vae_loss can pick it up
                        batch_device['sticky_hmm'] = hmm
                        batch_device['hmm_cache'] = {
                            'mu_k': mu_k.to(model_output['mu'].dtype),
                            'E_Lambda': E_Lambda.to(model_output['mu'].dtype),
                            'logdet_Sigma_k': logdet_Sigma_k.to(model_output['mu'].dtype),
                            'r_hat_flat': r_hat_flat.to(model_output['mu'].dtype),
                        }

                    # Calculate adaptive weights for this step
                    mi_beta, tc_beta, dw_beta, blend_alpha = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
                    
                    # Optional head weight schedule: start at 0.5x → 1.0x over warm-up for pass/safety
                    head_prog = min(global_step / max(int(total_train_steps * config.warmup_epoch_ratio), 1), 1.0)
                    def _cosine(a, b, t):
                        import math
                        return a + 0.5*(b - a)*(1 - math.cos(math.pi * t))
                    pass_safety_scalar = _cosine(0.5, 1.0, head_prog)
                    weight_override = {
                        'passability': config.raw_modality_weights.get('passability', 1.0) * pass_safety_scalar,
                        'safety'     : config.raw_modality_weights.get('safety', 1.0) * pass_safety_scalar,
                    }
                    
                    # Calculate loss (vae_loss will handle dynamics internally)
                    train_loss_dict = vae_loss(
                        model_output=model_output,
                        batch=batch_device,
                        config=config,  # Use the VAEConfig object
                        mi_beta=mi_beta,
                        tc_beta=tc_beta,
                        dw_beta=dw_beta,
                        weight_override=weight_override,
                        prior_blend_alpha=blend_alpha
                    )

                    train_loss = train_loss_dict['total_loss']
                mu = model_output['mu'].detach()
                kl_diagnosis = train_loss_dict['kl_diagnosis']
                per_dim_kl = kl_diagnosis['dimension_wise_kl'].detach()
                dim_kl = kl_diagnosis['dimension_wise_kl_sum']
                mutual_info = kl_diagnosis['mutual_info']
                total_correlation = kl_diagnosis['total_correlation']
                eigvals = kl_diagnosis['eigenvalues'].detach()
                eigvals = eigvals.flip(0)  # Sort in descending order
                kl_eig = 0.5 * (eigvals - eigvals.log() - 1)
                var_explained = eigvals.cumsum(dim=0) / eigvals.sum(dim=0)
                median_idx = (var_explained >= 0.5).nonzero(as_tuple=True)[0][0]
                median_ratio = (median_idx + 1) / len(var_explained)
                ninety_percentile_idx = (var_explained >= 0.9).nonzero(as_tuple=True)[0][0]
                ninety_percentile_ratio = (ninety_percentile_idx + 1) / len(var_explained)

                # Backward pass with mixed precision scaling if enabled
                if scaler is not None:
                    scaler.scale(train_loss).backward()
                    # Allow grad clipping with AMP
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()  # Step-based learning rate scheduling
                
                # Update global step counter
                global_step += 1

                epoch_train_loss += train_loss.item()
                batch_count += 1

                # Log to wandb every N steps if enabled
                if use_wandb and WANDB_AVAILABLE and global_step % log_every_n_steps == 0:
                    # Helper function to safely convert tensors for wandb logging
                    def safe_tensor_for_wandb(tensor):
                        """Convert tensor to float32 for wandb compatibility"""
                        if isinstance(tensor, torch.Tensor):
                            return tensor.detach().float().cpu()
                        return tensor
                    
                    wandb_log_dict = {
                        # Training metrics
                        "train/loss": train_loss.item(),
                        "train/batch": batch_count,
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        
                        # Loss components (safely access with .get())
                        "train/raw_loss/total": train_loss_dict['total_raw_loss'].item(),
                        "train/raw_loss/occupancy": train_loss_dict['raw_losses'].get('occupy', torch.tensor(0.0)).item(),
                        "train/raw_loss/bag_loss": train_loss_dict['raw_losses'].get('bag', torch.tensor(0.0)).item(),
                        "train/raw_loss/hero_loc": train_loss_dict['raw_losses'].get('hero_loc', torch.tensor(0.0)).item(),
                        "train/raw_loss/blstats": train_loss_dict['raw_losses'].get('stats', torch.tensor(0.0)).item(),
                        "train/raw_loss/message": train_loss_dict['raw_losses'].get('msg', torch.tensor(0.0)).item(),
                        "train/raw_loss/ego_char": train_loss_dict['raw_losses'].get('ego_char', torch.tensor(0.0)).item(),
                        "train/raw_loss/ego_color": train_loss_dict['raw_losses'].get('ego_color', torch.tensor(0.0)).item(),
                        "train/raw_loss/ego_class": train_loss_dict['raw_losses'].get('ego_class', torch.tensor(0.0)).item(),
                        "train/raw_loss/passability": train_loss_dict['raw_losses'].get('passability', torch.tensor(0.0)).item(),
                        "train/raw_loss/reward": train_loss_dict['raw_losses'].get('reward', torch.tensor(0.0)).item(),
                        "train/raw_loss/done": train_loss_dict['raw_losses'].get('done', torch.tensor(0.0)).item(),
                        "train/raw_loss/value_k": train_loss_dict['raw_losses'].get('value_k', torch.tensor(0.0)).item(),
                        "train/raw_loss/safety": train_loss_dict['raw_losses'].get('safety', torch.tensor(0.0)).item(),
                        "train/raw_loss/goal": train_loss_dict['raw_losses'].get('goal', torch.tensor(0.0)).item(),
                        "train/raw_loss/forward_dynamics": train_loss_dict['raw_losses'].get('forward', torch.tensor(0.0)).item(),
                        "train/raw_loss/inverse_dynamics": train_loss_dict['raw_losses'].get('inverse', torch.tensor(0.0)).item(),

                        "train/kl_loss": train_loss_dict['kl_loss'].item(),
                        "train/kl_loss/dimension_wise": dim_kl,
                        "train/kl_loss/mutual_info": mutual_info,
                        "train/kl_loss/total_correlation": total_correlation,

                        # Adaptive weights
                        "adaptive_weights/mi_beta": mi_beta,
                        "adaptive_weights/tc_beta": tc_beta,
                        "adaptive_weights/dw_beta": dw_beta,
                        "adaptive_weights/blend_alpha": blend_alpha,
                        
                        # Model diagnostics
                        "model/mu_var": safe_tensor_for_wandb(mu.var(dim=0)),
                        "model/mu_var_max": mu.var(dim=0).max().item(),
                        "model/mu_var_min": mu.var(dim=0).min().item(),
                        "model/mu_var_exceed_0.1": mu.var(dim=0).gt(0.1).sum().item() / mu.var(dim=0).numel(),
                        "model/per_dim_kl": safe_tensor_for_wandb(per_dim_kl),
                        "model/per_dim_kl_max": per_dim_kl.max().item(),
                        "model/per_dim_kl_min": per_dim_kl.min().item(),
                        "model/var_explained_median": median_ratio,
                        "model/var_explained_90_percentile": ninety_percentile_ratio,
                        "model/eigenval_max": eigvals[0].item(),
                        "model/eigenval_min": eigvals[-1].item(),
                        "model/eigenval_ratio": (eigvals[0] / eigvals[-1]).item(),
                        "model/eigenval": safe_tensor_for_wandb(eigvals),
                        "model/eigenval_exceed_2": (eigvals > 2).sum().item() / eigvals.numel(),
                        "model/kl_eigenval": safe_tensor_for_wandb(kl_eig),
                        "model/kl_eigenval_max": kl_eig.max().item(),
                        "model/kl_eigenval_min": kl_eig.min().item(),
                        "model/kl_eigenval_exceed_0.2": (kl_eig > 0.2).sum().item() / kl_eig.numel(),
                        
                        # Metrics
                        **{f"train/{k}": v for k, v in train_loss_dict.get('metrics', {}).items()}
                    }
                    wandb.log(wandb_log_dict)

                # Update progress bar with summary metrics only
                pbar.set_postfix({
                    'loss': f"{train_loss.item():.2f}",
                    'total_raw': f"{train_loss_dict['total_raw_loss'].item():.2f}",
                    'kl': f"{train_loss_dict['kl_loss'].item():.2f}",
                })
        
        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Testing phase
        model.eval()
        epoch_test_loss = 0.0
        test_batch_count = 0
        
        with torch.no_grad():
            for batch in shuffled_test_dataset:
                # Move batch to device
                batch_device = {}
                for key, value in batch.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        batch_device[key] = value.to(device)
                    else:
                        batch_device[key] = value
                
                # Optional: shuffle within batch (shuffle games but keep temporal order)
                if shuffle_within_batch:
                    # Shuffle across B dimension (games) while preserving T dimension (temporal order)
                    # Do this before reshaping to [B*T, ...]
                    if 'game_chars' in batch_device:
                        B, T = batch_device['game_chars'].shape[:2]
                        game_shuffle_indices = torch.randperm(B)
                        
                        for key, value in batch_device.items():
                            if value is not None and isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                                # Shuffle across B dimension (games)
                                batch_device[key] = value[game_shuffle_indices]
                
                # Now reshape tensors from [B, T, ...] to [B*T, ...]
                for key, value in batch_device.items():
                    if value is not None and isinstance(value, torch.Tensor) and len(value.shape) >= 2:
                        B, T = value.shape[:2]
                        remaining_dims = value.shape[2:]
                        batch_device[key] = value.view(B * T, *remaining_dims)
                
                # Store original batch dimensions for dynamics processing
                if 'game_chars' in batch:
                    B, T = batch['game_chars'].shape[:2]
                    batch_device['original_batch_shape'] = (B, T)
                    batch_device['batch_size'] = B
                
                # Forward pass with mixed precision if enabled
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(use_bf16 and device.type == 'cuda')):
                    model_output = model(batch_device)
                    
                    # ==== (M‑step) HMM prior cache for this batch (no gradients) ====
                    if use_hmm_prior and (hmm is not None):
                        with torch.no_grad():
                            B,T = batch_device['original_batch_shape']
                            valid_bt = batch_device['valid_screen'].view(B,T)
                            mu_bt    = model_output['mu'].view(B,T,-1)
                            var_bt   = model_output['logvar'].exp().clamp_min(1e-6).view(B,T,-1)
                            F_btr    = model_output.get('lowrank_factors', None)
                            F_bt     = None if F_btr is None else F_btr.view(B,T,F_btr.size(-2),F_btr.size(-1))
                            # emission expected log-likelihood per (b,t,k)
                            logB = StickyHDPHMMVI.expected_emission_loglik(hmm.niw.mu, hmm.niw.kappa, hmm.niw.Psi, hmm.niw.nu, mu_bt, var_bt, F_bt, mask=valid_bt)  # [B,T,K]
                            pi_star  = hmm._Epi()                              # [K]
                            ElogA    = hmm._ElogA()                            # [K,K]
                            log_pi   = torch.log(torch.clamp(pi_star, min=1e-30))
                            
                            # Process each sequence in the batch individually
                            r_hat_list = []
                            for b in range(B):
                                r_hat_b, xi_hat_b, ll_b = hmm.forward_backward(log_pi, ElogA, logB[b])
                                r_hat_list.append(r_hat_b)
                            r_hat = torch.stack(r_hat_list, dim=0)  # [B,T,K]
                            
                            # prior Gaussians
                            mu_k, E_Lambda, ElogdetLambda = hmm.get_emission_expectations() # mu_k:[K,D], E_Lambda:[K,D,D]
                            # log|Σ_k| = - log|Λ_k|  (approx with E[Λ])
                            logdet_Sigma_k = -torch.logdet(E_Lambda + 1e-6*torch.eye(E_Lambda.size(-1), device=E_Lambda.device)).to(mu_bt.dtype)
                            # flatten responsibilities to align with valid_screen
                            r_hat_flat = r_hat[valid_bt]   # [valid_B, K]
                        # stash cache into batch so vae_loss can pick it up
                        batch_device['sticky_hmm'] = hmm
                        batch_device['hmm_cache'] = {
                            'mu_k': mu_k.to(model_output['mu'].dtype),
                            'E_Lambda': E_Lambda.to(model_output['mu'].dtype),
                            'logdet_Sigma_k': logdet_Sigma_k.to(model_output['mu'].dtype),
                            'r_hat_flat': r_hat_flat.to(model_output['mu'].dtype),
                        }
                        
                    # Calculate adaptive weights for this step (use current global step for consistency)
                    mi_beta, tc_beta, dw_beta, blend_alpha = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
                    
                    # Optional head weight schedule: start at 0.5x → 1.0x over warm-up for pass/safety
                    head_prog = min(global_step / max(int(total_train_steps * config.warmup_epoch_ratio), 1), 1.0)
                    def _cosine(a, b, t):
                        import math
                        return a + 0.5*(b - a)*(1 - math.cos(math.pi * t))
                    pass_safety_scalar = _cosine(0.5, 1.0, head_prog)
                    weight_override = {
                        'passability': config.raw_modality_weights.get('passability', 1.0) * pass_safety_scalar,
                        'safety'     : config.raw_modality_weights.get('safety', 1.0) * pass_safety_scalar,
                    }
                    
                    # Calculate loss
                    test_loss_dict = vae_loss(
                        model_output=model_output,
                        batch=batch_device,
                        config=config,
                        mi_beta=mi_beta,
                        tc_beta=tc_beta,
                        dw_beta=dw_beta,
                        weight_override=weight_override,
                        prior_blend_alpha=blend_alpha
                    )

                    test_loss = test_loss_dict['total_loss']
                epoch_test_loss += test_loss.item()
                test_batch_count += 1
                
                if use_wandb and WANDB_AVAILABLE and test_batch_count % log_every_n_steps == 0:
                    wandb_log_dict = {
                        "test/loss": test_loss.item(),
                        "test/raw_loss/total": test_loss_dict['total_raw_loss'].item(),
                        "test/raw_loss/occupancy": test_loss_dict['raw_losses'].get('occupy', torch.tensor(0.0)).item(),
                        "test/raw_loss/bag_loss": test_loss_dict['raw_losses'].get('bag', torch.tensor(0.0)).item(),
                        "test/raw_loss/hero_loc": test_loss_dict['raw_losses'].get('hero_loc', torch.tensor(0.0)).item(),
                        "test/raw_loss/blstats": test_loss_dict['raw_losses'].get('stats', torch.tensor(0.0)).item(),
                        "test/raw_loss/message": test_loss_dict['raw_losses'].get('msg', torch.tensor(0.0)).item(),
                        "test/raw_loss/ego_char": test_loss_dict['raw_losses'].get('ego_char', torch.tensor(0.0)).item(),
                        "test/raw_loss/ego_color": test_loss_dict['raw_losses'].get('ego_color', torch.tensor(0.0)).item(),
                        "test/raw_loss/ego_class": test_loss_dict['raw_losses'].get('ego_class', torch.tensor(0.0)).item(),
                        "test/raw_loss/passability": test_loss_dict['raw_losses'].get('passability', torch.tensor(0.0)).item(),
                        "test/raw_loss/reward": test_loss_dict['raw_losses'].get('reward', torch.tensor(0.0)).item(),
                        "test/raw_loss/done": test_loss_dict['raw_losses'].get('done', torch.tensor(0.0)).item(),
                        "test/raw_loss/value_k": test_loss_dict['raw_losses'].get('value_k', torch.tensor(0.0)).item(),
                        "test/raw_loss/safety": test_loss_dict['raw_losses'].get('safety', torch.tensor(0.0)).item(),
                        # Additional raw losses that were missing:
                        "test/raw_loss/goal": test_loss_dict['raw_losses'].get('goal', torch.tensor(0.0)).item(),
                        "test/raw_loss/forward_dynamics": test_loss_dict['raw_losses'].get('forward', torch.tensor(0.0)).item(),
                        "test/raw_loss/inverse_dynamics": test_loss_dict['raw_losses'].get('inverse', torch.tensor(0.0)).item(),

                        "test/kl_loss": test_loss_dict['kl_loss'].item(),
                        
                        # Metrics
                        **{f"test/{k}": v for k, v in test_loss_dict.get('metrics', {}).items()}
                    }
                    wandb.log(wandb_log_dict)
        
        avg_test_loss = epoch_test_loss / test_batch_count if test_batch_count > 0 else 0.0
        test_losses.append(avg_test_loss)
        
        # Early stopping logic
        if early_stopping:
            improvement = best_test_loss / avg_test_loss - 1
            if improvement > early_stopping_min_delta:
                # Improvement found
                best_test_loss = avg_test_loss
                best_epoch = epoch
                early_stopping_counter = 0
                # Save the best model state
                best_model_state = {
                    'model_state_dict': model.state_dict().copy(),
                    'optimizer_state_dict': optimizer.state_dict().copy(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'config': config,  # Save VAEConfig
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'test_loss': avg_test_loss,
                    'train_losses': train_losses.copy(),
                    'test_losses': test_losses.copy()
                }
                logger.info(f"💚 New best test loss: {best_test_loss:.4f} (epoch {epoch+1})")
            else:
                # No improvement
                early_stopping_counter += 1
                logger.info(f"⏰ No improvement in test loss for {early_stopping_counter}/{early_stopping_patience} epochs")
                
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"🛑 Early stopping triggered! Best test loss: {best_test_loss:.4f} at epoch {best_epoch+1}")
                    
                    # Restore best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state['model_state_dict'])
                        optimizer.load_state_dict(best_model_state['optimizer_state_dict'])
                        if scheduler and best_model_state['scheduler_state_dict'] is not None:
                            scheduler.load_state_dict(best_model_state['scheduler_state_dict'])
                        
                        # Update loss lists to reflect the best model
                        train_losses = best_model_state['train_losses']
                        test_losses = best_model_state['test_losses']
                        
                        logger.info(f"✅ Restored best model from epoch {best_epoch+1}")
                    
                    # Log early stopping to wandb
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "early_stopping/triggered": True,
                            "early_stopping/best_epoch": best_epoch + 1,
                            "early_stopping/best_test_loss": best_test_loss,
                            "early_stopping/stopped_at_epoch": epoch + 1,
                            "early_stopping/patience_used": early_stopping_counter
                        })
                    
                    break  # Exit training loop
        
        # Log epoch summary
        # Calculate current adaptive weights for display
        current_mi_beta, current_tc_beta, current_dw_beta, current_blend_alpha = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
        
        logger.info(f"\n=== Epoch {epoch+1}/{epochs} Summary ===")
        logger.info(f"Average Train Loss: {avg_train_loss:.3f} | Average Test Loss: {avg_test_loss:.3f}")
        if early_stopping:
            logger.info(f"Early Stopping: Best Test Loss: {best_test_loss:.4f} (epoch {best_epoch+1}) | Counter: {early_stopping_counter}/{early_stopping_patience}")
        logger.info(f"Current KL Betas: mi={current_mi_beta:.3f}, tc={current_tc_beta:.3f}, dw={current_dw_beta:.3f}, blend_alpha={current_blend_alpha:.3f}")
        logger.info(f"Global Step: {global_step}/{total_train_steps} ({100*global_step/total_train_steps:.1f}%)")
        
        # Show detailed modality breakdown for the last batch of training and testing
        logger.info(f"Final Training Batch Details:")
        raw_losses = train_loss_dict['raw_losses']
        raw_loss_str = " | ".join([f"{key}: {value.item():.3f}" for key, value in raw_losses.items()])
        logger.info(f"  Raw Losses: {raw_loss_str}")
        
        logger.info(f"Final Testing Batch Details:")
        raw_losses = test_loss_dict['raw_losses']
        raw_loss_str = " | ".join([f"{key}: {value.item():.3f}" for key, value in raw_losses.items()])
        logger.info(f"  Raw Losses: {raw_loss_str}")

        logger.info(f"Variance of model output (mu): {', '.join(f'{v:.4f}' for v in mu.var(dim=0).tolist())}")
        logger.info(f"Per-dim KL: {', '.join(f'{v:.4f}' for v in per_dim_kl.tolist())}")
        logger.info(f"Eigenvalues of latent space: {', '.join(f'{v:.4f}' for v in eigvals.tolist())}")
        logger.info(f"KL Eigenvalues: {', '.join(f'{v:.4f}' for v in kl_eig.tolist())}")
        logger.info(f"Variance explained by eigenvalues: {', '.join(f'{v:.4f}' for v in var_explained.tolist())}")

        logger.info("=" * 50)
        
        # Log epoch metrics to wandb
        if use_wandb and WANDB_AVAILABLE:
            epoch_log_dict = {
                # Epoch summaries
                "epoch/train_loss": avg_train_loss,
                "epoch/test_loss": avg_test_loss,
                "epoch/number": epoch + 1,
                
                # Final batch details for comparison
                "epoch/final_train_raw_total": train_loss_dict['total_raw_loss'].item(),
                "epoch/final_train_kl": train_loss_dict['kl_loss'].item(),
                
                "epoch/final_test_raw_total": test_loss_dict['total_raw_loss'].item(),
                "epoch/final_test_kl": test_loss_dict['kl_loss'].item()
            }
            
            # Add early stopping metrics
            if early_stopping:
                epoch_log_dict.update({
                    "early_stopping/best_test_loss": best_test_loss,
                    "early_stopping/best_epoch": best_epoch + 1,
                    "early_stopping/counter": early_stopping_counter,
                    "early_stopping/patience": early_stopping_patience,
                    "early_stopping/improvement": best_test_loss - avg_test_loss,
                    "early_stopping/is_best": avg_test_loss == best_test_loss
                })
            
            wandb.log(epoch_log_dict)
        
        
        # Save checkpoint if requested
        if save_checkpoints and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_losses=train_losses,
                test_losses=test_losses,
                config=config,  # Pass VAEConfig
                scheduler=scheduler,
                scaler=scaler,
                checkpoint_dir=checkpoint_dir,
                keep_last_n=keep_last_n_checkpoints,
                upload_to_hf=hf_upload_checkpoints and upload_to_hf,
                hf_repo_name=hf_repo_name,
                hf_token=hf_token
            )
            
            # Log checkpoint save event to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "checkpoint/saved": True,
                    "checkpoint/epoch": epoch + 1,
                    "checkpoint/path": checkpoint_path,
                    "checkpoint/train_loss": avg_train_loss,
                    "checkpoint/test_loss": avg_test_loss,
                })
    
    logger.info(f"\n✅ MultiModalVAE training completed!")
    
    # Handle early stopping results
    if early_stopping and best_model_state is not None:
        logger.info(f"  - Training stopped early at epoch {epoch+1}")
        logger.info(f"  - Best model from epoch {best_epoch+1}")
        logger.info(f"  - Best train loss: {best_model_state['train_loss']:.4f}")
        logger.info(f"  - Best test loss: {best_model_state['test_loss']:.4f}")
        
        # Ensure we're using the best model for final operations
        model.load_state_dict(best_model_state['model_state_dict'])
        final_train_loss = best_model_state['train_loss']
        final_test_loss = best_model_state['test_loss']
    else:
        logger.info(f"  - Completed all {epochs} epochs")
        logger.info(f"  - Final train loss: {train_losses[-1]:.4f}")
        logger.info(f"  - Final test loss: {test_losses[-1]:.4f}")
        final_train_loss = train_losses[-1]
        final_test_loss = test_losses[-1]
    
    # HuggingFace upload if requested
    if upload_to_hf and hf_repo_name and HF_AVAILABLE:
        logger.info(f"\n🤗 Uploading best model to HuggingFace Hub...")
        try:
            # Prepare training configuration for model card
            training_config = {
                "epochs": epochs,
                "batch_size": batch_size,
                "max_learning_rate": max_learning_rate,
                "sequence_size": sequence_size,
                "shuffle_batches": shuffle_batches,
                "shuffle_within_batch": shuffle_within_batch,
                "vae_config": {
                    "latent_dim": config.latent_dim,
                    "encoder_dropout": config.encoder_dropout,
                    "decoder_dropout": config.decoder_dropout,
                    "initial_mi_beta": config.initial_mi_beta,
                    "final_mi_beta": config.final_mi_beta,
                    "mi_beta_shape": config.mi_beta_shape,
                    "initial_tc_beta": config.initial_tc_beta,
                    "final_tc_beta": config.final_tc_beta,
                    "tc_beta_shape": config.tc_beta_shape,
                    "initial_dw_beta": config.initial_dw_beta,
                    "final_dw_beta": config.final_dw_beta,
                    "dw_beta_shape": config.dw_beta_shape,
                    "warmup_epoch_ratio": config.warmup_epoch_ratio,
                    "free_bits": config.free_bits,
                    "focal_loss_alpha": config.focal_loss_alpha,
                    "focal_loss_gamma": config.focal_loss_gamma
                },
                "adaptive_weighting": {
                    "initial_mi_beta": config.initial_mi_beta,
                    "final_mi_beta": config.final_mi_beta,
                    "mi_beta_shape": config.mi_beta_shape,
                    "initial_tc_beta": config.initial_tc_beta,
                    "final_tc_beta": config.final_tc_beta,
                    "tc_beta_shape": config.tc_beta_shape,
                    "initial_dw_beta": config.initial_dw_beta,
                    "final_dw_beta": config.final_dw_beta,
                    "dw_beta_shape": config.dw_beta_shape,
                    "warmup_epoch_ratio": config.warmup_epoch_ratio
                },
                "regularization": {
                    "encoder_dropout": config.encoder_dropout,
                    "decoder_dropout": config.decoder_dropout,
                    "free_bits": config.free_bits,
                    "focal_loss_alpha": config.focal_loss_alpha,
                    "focal_loss_gamma": config.focal_loss_gamma
                },
                "early_stopping": {
                    "enabled": early_stopping,
                    "patience": early_stopping_patience,
                    "min_delta": early_stopping_min_delta,
                    "triggered": early_stopping and best_model_state is not None,
                    "best_epoch": best_epoch + 1 if best_model_state is not None else None,
                }
            }
            
            # Merge with user-provided model card data
            model_card_data = hf_model_card_data or {}
            model_card_data.update({
                "training_config": training_config,
                "final_train_loss": final_train_loss,
                "final_test_loss": final_test_loss,
                "best_train_loss": min(train_losses),
                "best_test_loss": min(test_losses),
                "total_epochs": epochs
            })
            
            # Upload model (save locally first or upload directly)
            commit_msg = f"Upload MultiModalHackVAE"
            if early_stopping and best_model_state is not None:
                commit_msg += f" (early stop at epoch {epoch+1}, best epoch {best_epoch+1}, test_loss={final_test_loss:.4f})"
            else:
                commit_msg += f" (epochs={epochs}, final_loss={final_test_loss:.4f})"
                
            if hf_upload_directly:
                repo_url = save_model_to_huggingface(
                    model=model,
                    config=config,
                    repo_name=hf_repo_name,
                    token=hf_token,
                    private=hf_private,
                    commit_message=commit_msg,
                    model_card_data=model_card_data,
                    upload_directly=True
                )
            else:
                # Save model locally first
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,  # Save complete VAEConfig
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'final_train_loss': final_train_loss,
                    'final_test_loss': final_test_loss,
                    'training_timestamp': datetime.now().isoformat(),
                }, save_path)
                logger.info(f"💾 Model saved locally: {save_path}")
                
                repo_url = save_model_to_huggingface(
                    model=model,
                    config=config,
                    model_save_path=save_path,
                    repo_name=hf_repo_name,
                    token=hf_token,
                    private=hf_private,
                    commit_message=commit_msg,
                    model_card_data=model_card_data,
                    upload_directly=False
                )
            
            # Upload training artifacts if requested
            if hf_upload_artifacts:
                upload_training_artifacts_to_huggingface(
                    repo_name=hf_repo_name,
                    train_losses=train_losses,
                    test_losses=test_losses,
                    training_config=training_config,
                    token=hf_token
                )
                
                # Create and upload demo notebook
                create_model_demo_notebook(hf_repo_name, "demo_notebook.ipynb")
                
                from huggingface_hub import HfApi, login
                api = HfApi()
                if hf_token:
                    login(token=hf_token)
                    
                api.upload_file(
                    path_or_fileobj="demo_notebook.ipynb",
                    path_in_repo="demo_notebook.ipynb",
                    repo_id=hf_repo_name,
                    repo_type="model",
                    commit_message="Add demo notebook"
                )
                os.remove("demo_notebook.ipynb")
            
            logger.info(f"🎉 Model successfully shared at: {repo_url}")
            
            # Log HuggingFace upload success to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "huggingface/upload_success": True,
                    "huggingface/repo_url": repo_url,
                    "huggingface/artifacts_uploaded": hf_upload_artifacts,
                    "huggingface/final_train_loss": train_losses[-1],
                    "huggingface/final_test_loss": test_losses[-1],
                })
            
        except Exception as e:
            logger.error(f"❌ Failed to upload to HuggingFace: {e}")
            logger.info("   Model was still saved locally.")
            
            # Log HuggingFace upload failure to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "huggingface/upload_success": False,
                    "huggingface/error": str(e),
                })
    
    elif upload_to_hf and not HF_AVAILABLE:
        logger.warning("⚠️  HuggingFace Hub not available. Install with: pip install huggingface_hub")
    elif upload_to_hf and not hf_repo_name:
        logger.warning("⚠️  HuggingFace upload requested but no repo_name provided")
    elif not upload_to_hf:
        # Save model locally
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,  # Save complete VAEConfig
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': final_train_loss,
            'final_test_loss': final_test_loss,
            'training_timestamp': datetime.now().isoformat(),
        }, save_path)
        logger.info(f"💾 Model saved locally: {save_path}")
    
    # Final wandb logging and cleanup
    if use_wandb and WANDB_AVAILABLE:
        # Log final training summary
        final_log_dict = {
            "training/completed": True,
            "training/total_epochs": epochs,
            "training/best_train_loss": min(train_losses),
            "training/best_test_loss": min(test_losses),
            "training/final_train_loss": final_train_loss,
            "training/final_test_loss": final_test_loss,
        }
        
        # Add early stopping metrics to final summary
        if early_stopping:
            final_log_dict.update({
                "training/early_stopping_enabled": True,
                "training/early_stopping_triggered": best_model_state is not None,
                "training/early_stopping_patience": early_stopping_patience,
                "training/epochs_completed": epoch + 1,
            })
            if best_model_state is not None:
                final_log_dict.update({
                    "training/best_model_epoch": best_epoch + 1,
                    "training/stopped_early_at_epoch": epoch + 1,
                })
        else:
            final_log_dict["training/early_stopping_enabled"] = False
        
        wandb.log(final_log_dict)
        
        # Mark run as finished
        wandb.finish()
    
    return model, train_losses, test_losses

def train_vae_with_sticky_hmm_em(
    pretrained_ckpt_path: str = None,
    pretrained_hf_repo: str = None,
    train_dataset=None, 
    test_dataset=None,
    config: VAEConfig = None,
    batch_multiples: int = 1,
    # Sticky-HDP-HMM params
    alpha: float = 5.0, 
    kappa: float = 20.0, 
    gamma: float = 5.0,
    # NIW prior params
    niw_mu0: float = 0.0, 
    niw_kappa0: float = 0.1, 
    niw_Psi0: float = 1.0,
    niw_nu0: int = 96 + 2,
    init_niw_mu_with_kmean: bool = True,
    hmm_only: bool = False,
    vae_only_with_hmm: bool = False,
    pretrained_hmm_hf_repo: str = None,
    pretrained_hmm_round: int = None,
    em_rounds: int = 3, 
    m_epochs_per_round: int = 1,
    save_dir: str = "checkpoints_hmm",
    device: torch.device = torch.device('cuda'),
    use_bf16: bool = False,
    logger=None,
    offline: bool = True,
    streaming_rho: float = 1.0,
    max_iters: int = 10,
    elbo_drop_tol: float = 10.0,
    optimize_pi_every_n_steps: int = 5,
    pi_iters: int = 10,
    pi_lr: float = 0.001,
    # Game-grouped data options for E-step
    use_game_grouped_data: bool = False,
    game_grouped_data_path: str = None,
    max_games_per_estep: int = None,
    # Batch accumulation option for E-step
    use_batch_accumulation: bool = False,
    accumulation_max_batches: int = None,
    # Weights & Biases monitoring parameters
    use_wandb: bool = False,
    wandb_project: str = "nethack-hmm-em",
    wandb_entity: str = None,
    wandb_run_name: str = None,
    wandb_tags: List[str] = None,
    wandb_notes: str = None,
    # HuggingFace integration
    push_to_hub: bool = False,
    hub_repo_id_hmm: str | None = None,
    hub_repo_id_vae_hmm: str | None = None,
    hf_token: str | None = None,
    hf_private: bool = True,
    hf_upload_artifacts: bool = True,
    # Additional training arguments that might be needed for M-step
    **train_kwargs
):
    """
    Loads a pretrained VAE (plain prior), fits sticky-HDP-HMM in E-steps, and fine-tunes the VAE
    with the HMM prior in M-steps using train_multimodalhack_vae. Saves HMM and the VAE+HMM under separate profiles.
    
    This function implements an EM algorithm where:
    - E-step: Fits HMM posterior using current VAE representations 
    - M-step: Fine-tunes VAE with HMM prior using the full train_multimodalhack_vae training pipeline
    
    Args:
        pretrained_ckpt_path: Path to local checkpoint file (alternative to pretrained_hf_repo)
        pretrained_hf_repo: HuggingFace repo ID to load pretrained VAE from (e.g., "username/nethack-vae")
        train_dataset: Training dataset for HMM fitting and VAE fine-tuning
        test_dataset: Testing dataset for VAE fine-tuning
        config: VAEConfig object. If None, will be loaded from checkpoint/HF repo
        batch_multiples: Number of batches to combine for each training step
        alpha: DP concentration parameter for sticky-HDP-HMM
        kappa: Sticky parameter (self-transition bias)
        gamma: Top-level DP concentration parameter  
        niw_mu0: NIW prior mean parameter
        niw_kappa0: NIW prior concentration parameter
        niw_Psi0: NIW prior scale matrix parameter
        niw_nu0: NIW prior degrees of freedom
        init_niw_mu_with_kmean: If True, initializes NIW mean with k-means
        hmm_only: If True, only fits HMM without VAE fine-tuning
        vae_only_with_hmm: If True, only trains VAE with pre-trained HMM (no E-step HMM training)
        pretrained_hmm_hf_repo: HuggingFace repo ID to load pre-trained HMM from (required if vae_only_with_hmm=True)
        pretrained_hmm_round: Round number of pre-trained HMM to load (None for latest)
        em_rounds: Number of EM iterations
        m_epochs_per_round: Number of VAE training epochs per M-step
        save_dir: Local directory to save checkpoints
        device: Device for training
        use_bf16: Whether to use BF16 mixed precision
        logger: Logger instance
        offline: Whether to use offline mode (no streaming)
        streaming_rho: rho used for streaming on statistics
        max_iters: max iterations in hmm update
        elbo_drop_tol: tolerance for ELBO drop
        optimize_pi_every_n_steps: how often to optimize pi
        pi_iters: number of iterations for pi optimization
        pi_lr: learning rate for pi optimization
        use_game_grouped_data: If True, use game-id grouped data for E-step instead of batched data
        game_grouped_data_path: Path to game-grouped data file (if None, will group from train_dataset)
        max_games_per_estep: Maximum number of games to process per E-step (None = all games)
        use_batch_accumulation: If True, freeze HMM after initial fit and do a full pass accumulating counts, then batch update NIW/Dir and optimize π once
        accumulation_max_batches: Maximum number of batches to process during accumulation pass (None = all batches)
        use_wandb: Whether to use Weights & Biases for monitoring HMM training
        wandb_project: Weights & Biases project name for EM training
        wandb_entity: Weights & Biases entity (team/user)
        wandb_run_name: Name for the Weights & Biases run
        wandb_tags: Tags for the Weights & Biases run
        wandb_notes: Notes for the Weights & Biases run
        push_to_hub: Whether to push models to HuggingFace Hub
        hub_repo_id_hmm: HuggingFace repo ID for HMM models (e.g., "username/nethack-hmm")
        hub_repo_id_vae_hmm: HuggingFace repo ID for VAE+HMM models (e.g., "username/nethack-vae-hmm")
        hf_token: HuggingFace API token
        hf_private: Whether to create private repos on HuggingFace
        hf_upload_artifacts: Whether to upload training artifacts
        **train_kwargs: Additional arguments passed to train_multimodalhack_vae
        
    Returns:
        Tuple of (model, hmm, training_info)
    """
    from .training_utils import load_model_from_huggingface, load_model_from_local, save_model_to_huggingface
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize Weights & Biases if requested
    if use_wandb and WANDB_AVAILABLE:
        # Prepare configuration for wandb
        wandb_config = {
            "em_rounds": em_rounds,
            "m_epochs_per_round": m_epochs_per_round,
            "hmm_only": hmm_only,
            "vae_only_with_hmm": vae_only_with_hmm,
            "pretrained_hmm_hf_repo": pretrained_hmm_hf_repo,
            "pretrained_hmm_round": pretrained_hmm_round,
            "train_batches": len(train_dataset) if train_dataset else 0,
            "test_batches": len(test_dataset) if test_dataset else 0,
            "device": str(device),
            "use_bf16": use_bf16,
            "use_game_grouped_data": use_game_grouped_data,
            "max_games_per_estep": max_games_per_estep,
            "use_batch_accumulation": use_batch_accumulation,
            "accumulation_max_batches": accumulation_max_batches,
            "hmm_params": {
                "alpha": alpha,
                "kappa": kappa,
                "gamma": gamma,
                "K": config.skill_num if config else 32,
                "D": config.latent_dim if config else 96
            },
            "niw_params": {
                "mu0": niw_mu0,
                "kappa0": niw_kappa0,
                "Psi0": niw_Psi0,
                "nu0": niw_nu0
            }
        }
        
        # Initialize wandb run
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config=wandb_config,
            tags=wandb_tags,
            notes=wandb_notes
        )
        
        if logger: logger.info("Weights & Biases initialized for EM training")
        
    elif use_wandb and not WANDB_AVAILABLE:
        if logger: logger.warning("⚠️  wandb requested but not available. Install with: pip install wandb")
    
    # 1) Load the pretrained VAE from HuggingFace or local checkpoint
    if pretrained_hf_repo:
        if logger: logger.info(f"🤗 Loading pretrained VAE from HuggingFace: {pretrained_hf_repo}")
        model = load_model_from_huggingface(
            repo_name=pretrained_hf_repo,
            token=hf_token,
            device=device
        )
        # Extract config from loaded model
        if config is None and hasattr(model, 'config'):
            config = model.config
        elif config is None:
            if logger: logger.warning("⚠️  No config provided and none found in model. Using default VAEConfig.")
            config = VAEConfig()
    elif pretrained_ckpt_path:
        if logger: logger.info(f"📁 Loading pretrained VAE from local checkpoint: {pretrained_ckpt_path}")
        model = load_model_from_local(
            checkpoint_path=pretrained_ckpt_path,
            device=device
        )
        # Load config from checkpoint if not provided
        if config is None:
            try:
                ckpt = torch.load(pretrained_ckpt_path, map_location='cpu', weights_only=False)
                if 'config' in ckpt:
                    config = ckpt['config']
                    if logger: logger.info("✅ Loaded VAEConfig from checkpoint")
                else:
                    if logger: logger.warning("⚠️  No config in checkpoint. Using default VAEConfig.")
                    config = VAEConfig()
            except Exception as e:
                if logger: logger.warning(f"⚠️  Error loading config from checkpoint: {e}. Using default VAEConfig.")
                config = VAEConfig()
    else:
        raise ValueError("Either pretrained_ckpt_path or pretrained_hf_repo must be provided")

    model = model.to(device)
    if logger: logger.info("✅ Loaded pretrained VAE checkpoint (plain prior).")

    # Validate parameter combinations
    if hmm_only and vae_only_with_hmm:
        raise ValueError("Cannot set both hmm_only=True and vae_only_with_hmm=True")
    
    if vae_only_with_hmm and not pretrained_hmm_hf_repo:
        raise ValueError("pretrained_hmm_hf_repo must be provided when vae_only_with_hmm=True")

    # 2) Instantiate or Load HMM
    D = config.latent_dim
    K = config.skill_num  # include the remainder
    
    if vae_only_with_hmm:
        # Load pre-trained HMM from HuggingFace
        if logger: logger.info(f"� Loading pre-trained HMM from HuggingFace: {pretrained_hmm_hf_repo}")
        from .training_utils import load_hmm_from_huggingface
        
        hmm, loaded_config, hmm_params, niw, metadata = load_hmm_from_huggingface(
            repo_name=pretrained_hmm_hf_repo,
            round_num=pretrained_hmm_round,  # None means latest
            device=str(device)
        )
        
        if logger: logger.info(f"✅ Loaded pre-trained HMM: latent_dim={hmm.p.D}, skills={hmm.p.K+1}")
        if metadata:
            if logger: logger.info(f"   🏷️  HMM Round: {metadata.get('round', 'unknown')}")
            if logger: logger.info(f"   📅 Created: {metadata.get('created', 'unknown')}")
        
        # Verify HMM dimensions match VAE config
        if hmm.p.D != D:
            raise ValueError(f"HMM latent dimension ({hmm.p.D}) does not match VAE config ({D})")
        if hmm.p.K + 1 != K:
            if logger: logger.warning(f"⚠️  HMM skill count ({hmm.p.K + 1}) does not match VAE config ({K}). Continuing...")
    
    else:
        # Initialize new HMM
        if logger: logger.info(f"�🧠 Initializing new Sticky-HDP-HMM: latent_dim={D}, skills={K}")
        
        niw = NIWPrior(
            mu0=torch.full((D,), niw_mu0, device=device), 
            kappa0=niw_kappa0,
            Psi0=torch.eye(D, device=device) * niw_Psi0,
            nu0=niw_nu0
        )
        hmm_params = StickyHDPHMMParams(
            alpha=alpha, 
            kappa=kappa, 
            gamma=gamma,
            K=K-1, # exclude the remainder
            D=D,
            device=device
        )
        hmm = StickyHDPHMMVI(
            p=hmm_params,
            niw_prior=niw,
            rho_emission=1.0,
            rho_transition=1.0
        )

    def _kmeans_init_hmm(hmm, model, dataset, device, max_frames=20000):
        model.eval()
        X = []
        collected = 0
        B, T = dataset[0]['tty_chars'].shape[:2]
        for batch in dataset:
            with torch.no_grad():
                bdev = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        x = v.to(device, non_blocking=True)
                        if x.dim() >= 3 and k not in ('original_batch_shape',):
                            bdev[k] = x.view(B*T, *x.shape[2:])
                        else:
                            bdev[k] = x
                    else:
                        bdev[k] = v
                out = model(bdev)
                mu_bt = out['mu'].view(B, T, -1).detach().cpu()
                valid = bdev['valid_screen'].view(B, T).cpu().bool()
                X.append(mu_bt[valid])
                collected += int(valid.sum().item())
            if collected >= max_frames:
                break
        if not X:
            return
        X = torch.cat(X, dim=0).numpy()
        Kp1 = hmm.niw.mu.shape[0]
        kmeans = KMeans(n_clusters=Kp1, n_init=8, max_iter=200, random_state=0)
        labels = kmeans.fit_predict(X)
        centers = torch.from_numpy(kmeans.cluster_centers_).to(hmm.niw.mu.device, dtype=hmm.niw.mu.dtype)
        with torch.no_grad():
            hmm.niw.mu[:Kp1] = centers[:Kp1].to(hmm.niw.mu)
            hmm._cache_fresh = False

    if init_niw_mu_with_kmean and not vae_only_with_hmm:
        if logger: logger.info("🔍 Initializing NIW mean with k-means on VAE latents...")
        _kmeans_init_hmm(hmm, model, train_dataset, device, max_frames=100000)
    elif vae_only_with_hmm:
        if logger: logger.info("🔒 Skipping k-means initialization (using pre-trained HMM)")

    # Track training info for final summary
    training_info = {
        'hmm_only': hmm_only,
        'vae_only_with_hmm': vae_only_with_hmm,
        'em_rounds': em_rounds,
        'm_epochs_per_round': m_epochs_per_round,
        'hmm_params': {
            'alpha': alpha,
            'kappa': kappa, 
            'gamma': gamma,
            'K': K,
            'D': D
        },
        'niw_params': {
            'mu0': niw_mu0,
            'kappa0': niw_kappa0,
            'Psi0': niw_Psi0,
            'nu0': niw_nu0
        },
        'hmm_paths': [],
        'vae_hmm_paths': [],
        'hf_repos': {
            'hmm': hub_repo_id_hmm,
            'vae_hmm': hub_repo_id_vae_hmm
        }
    }

    # 3) EM Algorithm: Alternating E-steps (HMM fitting) and M-steps (VAE fine-tuning)
    for r in range(em_rounds):
        
        # Skip E-step if using pre-trained HMM (vae_only_with_hmm mode)
        if vae_only_with_hmm:
            if logger: logger.info(f"========== EM Round {r+1}/{em_rounds}: Skipping E-step (using pre-trained HMM) ==========")
        else:
            if logger: logger.info(f"========== EM Round {r+1}/{em_rounds}: E-step ==========")
            
            # Log EM round start to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "em/round": r + 1,
                    "em/phase": "E-step",
                    "em/progress": (r + 1) / em_rounds
                })

            hmm.streaming_reset()
            
            # E-step: Fit HMM posterior using current VAE representations
            if use_game_grouped_data:
                # Load or create game-grouped data
                collector = NetHackDataCollector('ttyrecs.db')
                if game_grouped_data_path and os.path.exists(game_grouped_data_path):
                    if logger: logger.info(f"📂 Loading game-grouped data from: {game_grouped_data_path}")
                    grouped_data = collector.load_grouped_sequences(game_grouped_data_path)
                else:
                    if logger: logger.info(f"🔄 Creating game-grouped data from train_dataset...")
                    grouped_data = collector.group_sequences_by_game(
                        collected_batches=train_dataset,
                        save_path=game_grouped_data_path  # Save for future use
                    )
                    if logger: logger.info(f"✅ Created game-grouped data with {len(grouped_data)} games")
                
                # Use game-grouped E-step
                fit_sticky_hmm_with_game_grouped_data(
                    model, grouped_data, device, hmm, 
                    offline=offline, streaming_rho=streaming_rho, 
                    max_iters=max_iters, elbo_drop_tol=elbo_drop_tol,
                    optimize_pi_every_n_steps=optimize_pi_every_n_steps,
                    pi_iters=pi_iters, pi_lr=pi_lr, 
                    max_games=max_games_per_estep,
                    logger=logger, use_wandb=use_wandb
                )
            else:
                # Use standard batched E-step
                fit_sticky_hmm_one_pass(model, train_dataset, device, hmm, 
                                        offline=offline, streaming_rho=streaming_rho, 
                                        max_iters=max_iters, elbo_drop_tol=elbo_drop_tol,
                                        optimize_pi_every_n_steps=optimize_pi_every_n_steps,
                                        pi_iters=pi_iters, pi_lr=pi_lr, batch_multiples=batch_multiples,
                                        logger=logger, use_wandb=use_wandb)
            
            # Optional: Additional batch accumulation pass after initial HMM fitting
            if use_batch_accumulation:
                if logger: logger.info(f"🔄 Running additional batch accumulation pass...")
                fit_sticky_hmm_with_batch_accumulation(
                    model, train_dataset, device, hmm,
                    max_batches=accumulation_max_batches,
                    pi_iters=pi_iters, pi_lr=pi_lr,
                    logger=logger, use_wandb=use_wandb
                )

            # Compute HMM diagnostics
            if logger: logger.info(f"🔍 Computing HMM diagnostics...")
            diagnostics = compute_hmm_diagnostics(model, train_dataset, device, hmm, 
                                                  logger=logger, max_batches=100, random_seed=50)
            if logger: 
                logger.info(f"📊 HMM Diagnostics for Round {r+1}:")
                logger.info(f"  - Avg log-likelihood per step: {diagnostics['avg_loglik_per_step']:.4f}")
                logger.info(f"  - State entropy: {diagnostics['state_entropy']:.4f}")
                logger.info(f"  - Effective number of skills: {diagnostics['effective_K']:.2f}")
                logger.info(f"  - Transition stickiness (diag): {diagnostics['stickiness_diag_mean']:.4f}")
                logger.info(f"  - Top 5 skill occupancies: {diagnostics['top5_pi'].tolist()}")
                logger.info(f"  - Top 5 skill indices: {diagnostics['top5_idx'].tolist()}")
                for k in torch.argsort(diagnostics['p_stay'], descending=True)[:5]:
                    logger.info(f"    - Skill {k}: emp={diagnostics['empirical_dwell_length_per_state'][k]:6.2f}  E[1/(1-p)]:{float(diagnostics['expected_dwell_length_per_state'][k]):6.2f}  p_stay={float(diagnostics['p_stay'][k]):.3f}")

            # Log HMM diagnostics to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    f"em_round_{r+1}/avg_loglik_per_step": diagnostics['avg_loglik_per_step'],
                    f"em_round_{r+1}/state_entropy": diagnostics['state_entropy'],
                    f"em_round_{r+1}/effective_K": diagnostics['effective_K'],
                    f"em_round_{r+1}/stickiness_diag_mean": diagnostics['stickiness_diag_mean'],
                    f"em_round_{r+1}/expected_dwell_length_per_state": diagnostics['expected_dwell_length_per_state'].cpu().numpy().tolist(),
                    f"em_round_{r+1}/empirical_dwell_length_per_state": diagnostics['empirical_dwell_length_per_state'].cpu().numpy().tolist(),
                    f"em_round_{r+1}/top5_pi": diagnostics['top5_pi'].tolist(),
                    f"em_round_{r+1}/top5_idx": diagnostics['top5_idx'].tolist(),
                })

            # Visualization artifacts
            if logger: logger.info(f"🖼️  Rendering HMM visualizations for round {r+1}")
            viz_paths = visualize_hmm_after_estep(
                model=model, dataset=train_dataset, device=device, hmm=hmm,
                save_dir="hmm_analysis", round_idx=r+1, logger=logger,
                max_diags_batches=50, max_raster_sequences=10, random_seed=100, batch_multiples=batch_multiples
            )
            training_info.setdefault('viz_paths', []).append(viz_paths)

            # Save HMM posterior locally
            hmm_path = os.path.join(save_dir, f"hmm_round{r+1}.pt")
            hmm_save_data = {
                'hmm_posterior_params': hmm.get_posterior_params(),
                'hmm_params': hmm_params,
                'niw_prior': niw,
                'rho_emission': hmm.get_rho_emission().cpu(),
                'rho_transition': hmm.get_rho_transition().cpu(),
                'round': r + 1,
                'config': config,
                'training_timestamp': datetime.now().isoformat(),
                'diagnostics': diagnostics,  # Include diagnostics in save
            }
            torch.save(hmm_save_data, hmm_path)
            training_info['hmm_paths'].append(hmm_path)
            if logger: logger.info(f"💾 Saved HMM posterior → {hmm_path}")
            
            # Push HMM to HuggingFace if requested
            if push_to_hub and hub_repo_id_hmm:
                try:
                    if logger: logger.info(f"🤗 Uploading HMM round {r+1} to HuggingFace: {hub_repo_id_hmm}")
                    
                    from .training_utils import save_hmm_to_huggingface
                    repo_url = save_hmm_to_huggingface(
                        hmm=hmm,
                        hmm_params=hmm_params,
                        niw_prior=niw,
                        config=config,
                        repo_name=hub_repo_id_hmm,
                        round_num=r + 1,
                        total_rounds=em_rounds,
                        token=hf_token,
                        private=hf_private,
                        commit_message=f"Upload Sticky-HDP-HMM round {r+1}/{em_rounds}",
                        base_vae_repo=pretrained_hf_repo if pretrained_hf_repo else "local_checkpoint"
                    )
                    if logger: logger.info(f"✅ HMM uploaded to: {repo_url}")
                    
                except Exception as e:
                    if logger: logger.error(f"❌ Failed to upload HMM to HuggingFace: {e}")

        # Skip M-step if hmm_only is True
        if hmm_only:
            if logger: logger.info(f"🔒 Skipping M-step (VAE fine-tuning) because hmm_only=True")
            continue

        if logger: logger.info(f"========== EM Round {r+1}/{em_rounds}: M-step ==========")
        
        # M-step: Fine-tune VAE with current HMM prior using train_multimodalhack_vae
        if logger: logger.info(f"🔧 Fine-tuning VAE with HMM prior for {m_epochs_per_round} epochs...")
        
        # Adjust VAE training parameters for this M-step round
        m_step_config = copy.deepcopy(config)
        
        # Define per-round parameter schedules
        if r == 0:  # First M-step (round 1)
            m_step_config.initial_prior_blend_alpha = 0.3
            m_step_config.final_prior_blend_alpha = 0.4
            m_step_config.prior_blend_shape = "cosine"
            m_step_config.initial_mi_beta = 0.2
            m_step_config.final_mi_beta = 0.2 + (0.6 - 0.2) * (1/3)  # 0.333
            m_step_config.initial_tc_beta = 0.0
            m_step_config.final_tc_beta = 0.0 + (0.2 - 0.0) * (1/3)  # 0.067
            m_step_config.initial_dw_beta = 0.5
            m_step_config.final_dw_beta = 0.5 + (1.0 - 0.5) * (1/3)  # 0.667
        elif r == 1:  # Second M-step (round 2)
            m_step_config.initial_prior_blend_alpha = 0.4
            m_step_config.final_prior_blend_alpha = 0.6
            m_step_config.prior_blend_shape = "cosine"
            m_step_config.initial_mi_beta = 0.2 + (0.6 - 0.2) * (1/3)  # 0.333
            m_step_config.final_mi_beta = 0.2 + (0.6 - 0.2) * (2/3)  # 0.467
            m_step_config.initial_tc_beta = 0.0 + (0.2 - 0.0) * (1/3)  # 0.067
            m_step_config.final_tc_beta = 0.0 + (0.2 - 0.0) * (2/3)  # 0.133
            m_step_config.initial_dw_beta = 0.5 + (1.0 - 0.5) * (1/3)  # 0.667
            m_step_config.final_dw_beta = 0.5 + (1.0 - 0.5) * (2/3)  # 0.833
        elif r == 2:  # Third M-step (round 3)
            m_step_config.initial_prior_blend_alpha = 0.6
            m_step_config.final_prior_blend_alpha = 0.8
            m_step_config.prior_blend_shape = "cosine"
            m_step_config.initial_mi_beta = 0.2 + (0.6 - 0.2) * (2/3)  # 0.467
            m_step_config.final_mi_beta = 0.6
            m_step_config.initial_tc_beta = 0.0 + (0.2 - 0.0) * (2/3)  # 0.133
            m_step_config.final_tc_beta = 0.2
            m_step_config.initial_dw_beta = 0.5 + (1.0 - 0.5) * (2/3)  # 0.833
            m_step_config.final_dw_beta = 1.0
        else:  # Additional rounds beyond 3
            m_step_config.initial_prior_blend_alpha = 0.8
            m_step_config.final_prior_blend_alpha = 0.8
            m_step_config.prior_blend_shape = "constant"
            m_step_config.initial_mi_beta = 0.6
            m_step_config.final_mi_beta = 0.6
            m_step_config.initial_tc_beta = 0.2
            m_step_config.final_tc_beta = 0.2
            m_step_config.initial_dw_beta = 1.0
            m_step_config.final_dw_beta = 1.0
        
        # Set the beta curve shapes (you can customize these)
        m_step_config.mi_beta_shape = "linear" 
        m_step_config.tc_beta_shape = "linear"
        m_step_config.dw_beta_shape = "linear"
        
        if logger: 
            logger.info(f"📊 M-step {r+1} parameter schedule:")
            logger.info(f"  - prior_blend_alpha: {m_step_config.initial_prior_blend_alpha:.3f} → {m_step_config.final_prior_blend_alpha:.3f} ({m_step_config.prior_blend_shape})")
            logger.info(f"  - mi_beta: {m_step_config.initial_mi_beta:.3f} → {m_step_config.final_mi_beta:.3f} ({m_step_config.mi_beta_shape})")
            logger.info(f"  - tc_beta: {m_step_config.initial_tc_beta:.3f} → {m_step_config.final_tc_beta:.3f} ({m_step_config.tc_beta_shape})")
            logger.info(f"  - dw_beta: {m_step_config.initial_dw_beta:.3f} → {m_step_config.final_dw_beta:.3f} ({m_step_config.dw_beta_shape})")
        
        # Log M-step start to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "em/round": r + 1,
                "em/phase": "M-step",
                f"em_round_{r+1}/m_step_config/initial_prior_blend_alpha": m_step_config.initial_prior_blend_alpha,
                f"em_round_{r+1}/m_step_config/final_prior_blend_alpha": m_step_config.final_prior_blend_alpha,
                f"em_round_{r+1}/m_step_config/initial_mi_beta": m_step_config.initial_mi_beta,
                f"em_round_{r+1}/m_step_config/final_mi_beta": m_step_config.final_mi_beta,
                f"em_round_{r+1}/m_step_config/initial_tc_beta": m_step_config.initial_tc_beta,
                f"em_round_{r+1}/m_step_config/final_tc_beta": m_step_config.final_tc_beta,
                f"em_round_{r+1}/m_step_config/initial_dw_beta": m_step_config.initial_dw_beta,
                f"em_round_{r+1}/m_step_config/final_dw_beta": m_step_config.final_dw_beta,
            })
        
        # Prepare training arguments for M-step
        m_step_kwargs = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'config': m_step_config,  # Use the adjusted config for this M-step
            'epochs': m_epochs_per_round,
            'device': device,
            'logger': logger,
            'use_bf16': use_bf16,
            'hmm': hmm,
            'use_hmm_prior': True,  # Always use HMM prior in M-step
            'save_path': os.path.join(save_dir, f"vae_hmm_round{r+1}_temp.pth"),
            'use_wandb': True,  # Disable wandb for M-step to avoid conflicts
            'wandb_project': f"nethack-vae-hmm-round{r+1}",
            'wandb_run_name': f"vae_hmm_em_round{r+1}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'log_every_n_steps': 2,
            'upload_to_hf': False,  # Don't upload intermediate M-step results
            'early_stopping': False,  # Disable early stopping for short M-step training
            **train_kwargs  # Include any additional training arguments
        }
        
        # Run M-step training
        model, train_losses, test_losses = train_multimodalhack_vae(**m_step_kwargs)
        
        # Clean up temporary file created by train_multimodalhack_vae
        temp_file = m_step_kwargs['save_path']
        if os.path.exists(temp_file):
            os.remove(temp_file)
            if logger: logger.debug(f"🗑️  Removed temporary file: {temp_file}")
        
        if logger: logger.info(f"✅ M-step completed: final_train_loss={train_losses[-1]:.4f}, final_test_loss={test_losses[-1]:.4f}")
        
        # Log M-step completion to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f"em_round_{r+1}/m_step_final_train_loss": train_losses[-1] if train_losses else float('nan'),
                f"em_round_{r+1}/m_step_final_test_loss": test_losses[-1] if test_losses else float('nan'),
                f"em_round_{r+1}/m_step_epochs": m_epochs_per_round
            })
        
        # Save the VAE+HMM locally
        vae_hmm_path = os.path.join(save_dir, f"vae_with_hmm_round{r+1}.pt")
        vae_hmm_save_data = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'hmm_posterior_params': hmm.get_posterior_params(),
            'hmm_params': hmm_params,
            'niw_prior': niw,
            'rho_emission': hmm.get_rho_emission().cpu(),
            'rho_transition': hmm.get_rho_transition().cpu(),
            'round': r + 1,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_test_loss': test_losses[-1] if test_losses else None,
            'training_timestamp': datetime.now().isoformat(),
            'diagnostics': None if vae_only_with_hmm else diagnostics,  # Include diagnostics in VAE+HMM save
        }
        torch.save(vae_hmm_save_data, vae_hmm_path)
        training_info['vae_hmm_paths'].append(vae_hmm_path)
        if logger: logger.info(f"💾 Saved VAE+HMM → {vae_hmm_path}")
        
        # Push VAE+HMM to HuggingFace if requested
        if push_to_hub and hub_repo_id_vae_hmm:
            try:
                if logger: logger.info(f"🤗 Uploading VAE+HMM round {r+1} to HuggingFace: {hub_repo_id_vae_hmm}")
                
                # Create model card content specific to VAE+HMM
                vae_hmm_model_card_data = {
                    "model_type": "MultiModalHackVAE_with_StickyHDPHMM",
                    "round": r + 1,
                    "total_rounds": em_rounds,
                    "hmm_parameters": training_info['hmm_params'],
                    "niw_parameters": training_info['niw_params'],
                    "latent_dim": D,
                    "num_skills": K,
                    "base_vae_repo": pretrained_hf_repo if pretrained_hf_repo else "local_checkpoint",
                    "final_train_loss": train_losses[-1] if train_losses else None,
                    "final_test_loss": test_losses[-1] if test_losses else None,
                    "m_step_epochs": m_epochs_per_round
                }
                
                # Upload VAE+HMM with additional files
                additional_files = {
                    vae_hmm_path: f"vae_with_hmm_round{r+1}.pt"
                }
                
                repo_url = save_model_to_huggingface(
                    model=model,
                    config=config,
                    repo_name=hub_repo_id_vae_hmm,
                    token=hf_token,
                    private=hf_private,
                    commit_message=f"Upload VAE+HMM round {r+1}/{em_rounds} (train_loss={train_losses[-1]:.4f})" if train_losses else f"Upload VAE+HMM round {r+1}/{em_rounds}",
                    model_card_data=vae_hmm_model_card_data,
                    upload_directly=True,
                    additional_files=additional_files
                )
                if logger: logger.info(f"✅ VAE+HMM uploaded to: {repo_url}")
                
                # Upload training artifacts if requested
                if hf_upload_artifacts:
                    from .training_utils import upload_training_artifacts_to_huggingface
                    upload_training_artifacts_to_huggingface(
                        repo_name=hub_repo_id_vae_hmm,
                        train_losses=train_losses,
                        test_losses=test_losses,
                        training_config=training_info,
                        token=hf_token
                    )
                
            except Exception as e:
                if logger: logger.error(f"❌ Failed to upload VAE+HMM to HuggingFace: {e}")

    # Final summary
    if logger: logger.info(f"\n🎉 {'HMM-only' if hmm_only else 'EM'} training completed! Summary:")
    if logger: logger.info(f"  - EM rounds: {em_rounds}")
    if not hmm_only:
        if logger: logger.info(f"  - M-step epochs per round: {m_epochs_per_round}")
    else:
        if logger: logger.info(f"  - M-step: Skipped (HMM-only mode)")
    if logger: logger.info(f"  - HMM checkpoints: {len(training_info['hmm_paths'])}")
    if not hmm_only:
        if logger: logger.info(f"  - VAE+HMM checkpoints: {len(training_info['vae_hmm_paths'])}")
    else:
        if logger: logger.info(f"  - VAE+HMM checkpoints: 0 (HMM-only mode)")
    if push_to_hub:
        if logger: logger.info(f"  - HuggingFace repos:")
        if hub_repo_id_hmm:
            if logger: logger.info(f"    * HMM: https://huggingface.co/{hub_repo_id_hmm}")
        if hub_repo_id_vae_hmm and not hmm_only:
            if logger: logger.info(f"    * VAE+HMM: https://huggingface.co/{hub_repo_id_vae_hmm}")

    # Final wandb logging and cleanup
    if use_wandb and WANDB_AVAILABLE:
        # Log final EM training summary
        final_log_dict = {
            "em_training/completed": True,
            "em_training/total_rounds": em_rounds,
            "em_training/hmm_only": hmm_only,
            "em_training/m_epochs_per_round": m_epochs_per_round,
            "em_training/hmm_checkpoints": len(training_info['hmm_paths']),
            "em_training/vae_hmm_checkpoints": len(training_info['vae_hmm_paths']),
        }
        
        wandb.log(final_log_dict)
        
        # Mark run as finished
        wandb.finish()

    return model, hmm, training_info

def format_elbo_progression(elbo_history):
    """Format ELBO progression for display with smart number of values shown."""
    if len(elbo_history) == 0:
        return "[]"
    elif len(elbo_history) <= 3:
        return f"[{', '.join(f'{x.item():.2f}' for x in elbo_history)}]"
    else:
        # Show first, last, and maybe one middle value
        first = elbo_history[0].item()
        last = elbo_history[-1].item()
        if len(elbo_history) == 4:
            mid = elbo_history[len(elbo_history)//2].item()
            return f"[{first:.2f}, {mid:.2f}, {last:.2f}]"
        else:
            return f"[{first:.2f}, ..., {last:.2f}]"


def log_hmm_elbo_to_wandb(batch_idx, inner_elbo, elbo_history, n_iterations, use_wandb):
    """Log HMM ELBO metrics to Weights & Biases."""
    if not use_wandb or not WANDB_AVAILABLE:
        return
    
    # Calculate ELBO improvement if we have history
    elbo_improvement = 0.0
    if len(elbo_history) > 1:
        elbo_improvement = (elbo_history[-1] - elbo_history[0]).item()
    
    # Calculate convergence rate (average improvement per iteration)
    convergence_rate = elbo_improvement / max(n_iterations - 1, 1) if n_iterations > 1 else 0.0
    
    # Log HMM metrics to wandb
    wandb_hmm_log = {
        "hmm/inner_elbo": inner_elbo,
        "hmm/elbo_improvement": elbo_improvement,
        "hmm/n_iterations": n_iterations,
        "hmm/convergence_rate": convergence_rate,
        "hmm/batch_idx": batch_idx,
    }
    
    # Add ELBO history if available
    if len(elbo_history) > 0:
        wandb_hmm_log.update({
            "hmm/elbo_first": elbo_history[0].item(),
            "hmm/elbo_final": elbo_history[-1].item(),
        })
        
        # Log full ELBO progression for detailed analysis (if not too many iterations)
        if len(elbo_history) <= 20:
            for i, elbo_val in enumerate(elbo_history):
                wandb_hmm_log[f"hmm/elbo_iter_{i:02d}"] = elbo_val.item()
    
    wandb.log(wandb_hmm_log)