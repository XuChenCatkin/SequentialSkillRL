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
from torch.optim.lr_scheduler import OneCycleLR
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

# Scikit-learn availability check
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

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
from utils.analysis import visualize_reconstructions, analyze_latent_space


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


@torch.no_grad()
def compute_hmm_diagnostics(model, dataset, device, hmm: StickyHDPHMMVI, max_batches: int = 5, logger=None):
    """
    Compute HMM diagnostics on a subset of the dataset
    """
    model.eval()
    n_batches = min(len(dataset), max_batches)
    
    all_mu = []
    all_var = []
    all_F = []
    all_mask = []
    
    # Collect VAE outputs for diagnostics
    with torch.no_grad():
        for bi, batch in enumerate(dataset):
            if bi >= n_batches: 
                break
                
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    x = v.to(device, non_blocking=True)
                    if x.dim() >= 3 and k not in ('original_batch_shape',):
                        B, T = x.shape[:2]
                        batch_dev[k] = x.view(B*T, *x.shape[2:])
                    else:
                        batch_dev[k] = x
                else:
                    batch_dev[k] = v
            
            # Forward encode only
            out = model(batch_dev)
            mu = out['mu']
            logvar = out['logvar']
            F = out.get('lowrank_factors', None)
            
            B, T = batch['original_batch_shape']
            valid = batch_dev['valid_screen'].view(B, T)
            
            # Reshape to [B,T,...]
            mu_bt = mu.view(B, T, -1)
            var_bt = logvar.exp().clamp_min(1e-6).view(B, T, -1)
            F_bt = None if F is None else F.view(B, T, F.size(-2), F.size(-1))
            
            all_mu.append(mu_bt)
            all_var.append(var_bt)
            if F_bt is not None:
                all_F.append(F_bt)
            all_mask.append(valid)
    
    # Concatenate all data
    mu_concat = torch.cat(all_mu, dim=0)  # [total_B, T, D]
    var_concat = torch.cat(all_var, dim=0)
    F_concat = torch.cat(all_F, dim=0) if all_F else None
    mask_concat = torch.cat(all_mask, dim=0)
    
    # Compute diagnostics using HMM's built-in function
    diagnostics = hmm.diagnostics(
        mu_t=mu_concat,
        diag_var_t=var_concat, 
        F_t=F_concat,
        mask=mask_concat
    )
    
    return diagnostics


def fit_sticky_hmm_one_pass(model, dataset, device, hmm: StickyHDPHMMVI, streaming_rho: float = 1.0, max_batches: int | None = None, logger=None):
    """
    Run one E-step pass: update sticky-HDP-HMM variational posteriors using encoder outputs.
    """
    model.eval()
    n_batches = len(dataset) if max_batches is None else min(len(dataset), max_batches)
    for bi, batch in enumerate(dataset):
        if bi >= n_batches: break
        batch_dev = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                x = v.to(device, non_blocking=True)
                if x.dim() >= 3 and k not in ('original_batch_shape',):
                    B,T = x.shape[:2]; batch_dev[k] = x.view(B*T, *x.shape[2:])
                else:
                    batch_dev[k] = x
            else:
                batch_dev[k] = v
        # forward encode only
        out = model(batch_dev)  # returns 'mu', 'logvar', optionally 'lowrank_factors'
        mu    = out['mu']
        logvar= out['logvar']
        F     = out.get('lowrank_factors', None)
        B, T = batch['original_batch_shape']
        valid = batch_dev['valid_screen'].view(B,T)
        # reshape to [B,T,...]
        mu_bt = mu.view(B,T,-1)
        var_bt = logvar.exp().clamp_min(1e-6).view(B,T,-1)
        F_bt  = None if F is None else F.view(B,T,F.size(-2),F.size(-1))
        hmm.update(mu_bt, var_bt, F_bt, mask=valid, rho=streaming_rho, optimize_pi=True)
        if logger and (bi+1) % 50 == 0:
            logger.info(f"[E-step] processed {bi+1}/{n_batches} batches")
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

    def get_adaptive_weights(global_step: int, total_steps: int, f: Optional[Callable[[float, float, float], float]]) -> Tuple[float, float, float]:
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
        
        # Log the adaptive weights (only occasionally to avoid spam)
        if global_step % 100 == 0:
            logger.debug(f"Step {global_step}/{total_steps} - Adaptive weights: mi_beta={mi_beta:.3f}, tc_beta={tc_beta:.3f}, dw_beta={dw_beta:.3f}")

        return mi_beta, tc_beta, dw_beta

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

                    # Calculate adaptive weights for this step
                    mi_beta, tc_beta, dw_beta = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
                    
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
                        weight_override=weight_override
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
                    
                    # Calculate adaptive weights for this step (use current global step for consistency)
                    mi_beta, tc_beta, dw_beta = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
                    
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
                        weight_override=weight_override
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
        current_mi_beta, current_tc_beta, current_dw_beta = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
        
        logger.info(f"\n=== Epoch {epoch+1}/{epochs} Summary ===")
        logger.info(f"Average Train Loss: {avg_train_loss:.3f} | Average Test Loss: {avg_test_loss:.3f}")
        if early_stopping:
            logger.info(f"Early Stopping: Best Test Loss: {best_test_loss:.4f} (epoch {best_epoch+1}) | Counter: {early_stopping_counter}/{early_stopping_patience}")
        logger.info(f"Current KL Betas: mi={current_mi_beta:.3f}, tc={current_tc_beta:.3f}, dw={current_dw_beta:.3f}")
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
    # Sticky-HDP-HMM params
    alpha: float = 5.0, 
    kappa: float = 20.0, 
    gamma: float = 5.0,
    # NIW prior params
    niw_mu0: float = 0.0, 
    niw_kappa0: float = 0.1, 
    niw_Psi0: float = 1.0,
    niw_nu0: int = 96 + 2,
    hmm_only: bool = False,
    em_rounds: int = 3, 
    m_epochs_per_round: int = 1,
    save_dir: str = "checkpoints_hmm",
    device: torch.device = torch.device('cuda'),
    use_bf16: bool = False,
    logger=None,
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
        alpha: DP concentration parameter for sticky-HDP-HMM
        kappa: Sticky parameter (self-transition bias)
        gamma: Top-level DP concentration parameter  
        niw_mu0: NIW prior mean parameter
        niw_kappa0: NIW prior concentration parameter
        niw_Psi0: NIW prior scale matrix parameter
        niw_nu0: NIW prior degrees of freedom
        hmm_only: If True, only fits HMM without VAE fine-tuning
        em_rounds: Number of EM iterations
        m_epochs_per_round: Number of VAE training epochs per M-step
        save_dir: Local directory to save checkpoints
        device: Device for training
        use_bf16: Whether to use BF16 mixed precision
        logger: Logger instance
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

    # 2) Instantiate HMM
    D = config.latent_dim
    K = config.skill_num
    if logger: logger.info(f"🧠 Initializing Sticky-HDP-HMM: latent_dim={D}, skills={K}")
    
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
        K=K,
        D=D,
        device=device
    )
    hmm = StickyHDPHMMVI(
        p=hmm_params,
        niw_prior=niw,
        rho_emission=1.0,
        rho_transition=1.0
    )

    # Track training info for final summary
    training_info = {
        'hmm_only': hmm_only,
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
        if logger: logger.info(f"========== EM Round {r+1}/{em_rounds}: E-step ==========")
        
        # E-step: Fit HMM posterior using current VAE representations
        fit_sticky_hmm_one_pass(model, train_dataset, device, hmm, streaming_rho=1.0, logger=logger)

        # Compute HMM diagnostics
        if logger: logger.info(f"🔍 Computing HMM diagnostics...")
        diagnostics = compute_hmm_diagnostics(model, train_dataset, device, hmm, logger=logger)
        if logger: 
            logger.info(f"📊 HMM Diagnostics for Round {r+1}:")
            logger.info(f"  - Avg log-likelihood per step: {diagnostics['avg_loglik_per_step']:.4f}")
            logger.info(f"  - State entropy: {diagnostics['state_entropy']:.4f}")
            logger.info(f"  - Effective number of skills: {diagnostics['effective_K']:.2f}")
            logger.info(f"  - Transition stickiness (diag): {diagnostics['stickiness_diag_mean']:.4f}")
            logger.info(f"  - Top 5 skill occupancies: {diagnostics['top5_pi'].tolist()}")
            logger.info(f"  - Top 5 skill indices: {diagnostics['top5_idx'].tolist()}")

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
        
        # Prepare training arguments for M-step
        m_step_kwargs = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
            'config': config,
            'epochs': m_epochs_per_round,
            'device': device,
            'logger': logger,
            'use_bf16': use_bf16,
            'hmm': hmm,
            'use_hmm_prior': True,  # Always use HMM prior in M-step
            'save_path': os.path.join(save_dir, f"vae_hmm_round{r+1}_temp.pth"),
            'use_wandb': False,  # Disable wandb for M-step to avoid conflicts
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
            'diagnostics': diagnostics,  # Include diagnostics in VAE+HMM save
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

    return model, hmm, training_info