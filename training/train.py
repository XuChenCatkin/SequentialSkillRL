"""
Complete VAE training pipeline for NetHack Learning Dataset
Supports both the simple NetHackVAE and the sophisticated MiniHackVAE from src/model.py
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Optional, Callable
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import re  # Add regex import for status line parsing
import json
import logging
from collections import Counter
from datetime import datetime
import tempfile  # Add tempfile import
from torch.optim.lr_scheduler import OneCycleLR
warnings.filterwarnings('ignore')

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
    from huggingface_hub.utils import RepositoryNotFoundError
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


def train_multimodalhack_vae(
    train_file: str, 
    test_file: str,                     
    dbfilename: str = 'ttyrecs.db',
    config: VAEConfig = None,
    epochs: int = 10, 
    batch_size: int = 32, 
    sequence_size: int = 32, 
    training_batches: int = 100,
    testing_batches: int = 20,
    max_training_batches: int = 100,
    max_testing_batches: int = 20,
    training_game_ids: List[int] | None = None,
    testing_game_ids: List[int] | None = None,
    max_learning_rate: float = 1e-3,
    device: str = None, 
    logger: logging.Logger = None,
    data_cache_dir: str = "data_cache",
    force_recollect: bool = False,
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
    early_stopping_min_delta: float = 0.01) -> Tuple[MultiModalHackVAE, List[float], List[float]]:
    """
    Train MultiModalHackVAE on NetHack Learning Dataset with adaptive loss weighting

    Args:
        train_file: Path to the training samples
        test_file: Path to the testing samples
        dbfilename: Path to the NetHack Learning Dataset database file
        config: VAEConfig object containing model configuration and training hyperparameters.
                If None, will create default config. For resuming from checkpoint, config from checkpoint takes precedence.
        epochs: Number of training epochs
        batch_size: Training batch size
        max_learning_rate: max learning rate for optimizer
        device: Device to use ('cuda' or 'cpu')
        data_cache_dir: Directory to cache processed data
        force_recollect: Force data recollection even if cache exists
        shuffle_batches: Whether to shuffle training batches at the start of each epoch
        shuffle_within_batch: Whether to shuffle samples within each batch (ignores temporal order)
        use_bf16: Whether to use BF16 mixed precision training for memory efficiency
        custom_kl_beta_function: Optional custom function for KL beta ramping (overrides config beta curves)
        free_bits: Target free bits for KL loss (0.0 disables)
        warmup_epoch_ratio: Ratio of epochs for warm-up phase
        focal_loss_alpha: Alpha parameter for focal loss (0.0 disables)
        focal_loss_gamma: Gamma parameter for focal loss (0.0 disables)
        dropout_rate: Dropout rate (0.0-1.0) for regularization. 0.0 disables dropout
        enable_dropout_on_latent: Whether to apply dropout to encoder fusion layers
        enable_dropout_on_decoder: Whether to apply dropout to decoder layers
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

    # Initialize Weights & Biases if requested
    if use_wandb and WANDB_AVAILABLE:
        # Prepare configuration for wandb
        wandb_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "sequence_size": sequence_size,
            "max_learning_rate": max_learning_rate,
            "training_batches": training_batches,
            "testing_batches": testing_batches,
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

    logger.info(f"🔥Training MultiModalHackVAE with {training_batches} train batches, {testing_batches} test batches")
    logger.info(f"   Epochs: {epochs}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Sequence size: {sequence_size}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Mixed Precision: BF16 = {use_bf16}")
    logger.info(f"   Data cache: {data_cache_dir}")
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
                # Merge provided config with checkpoint config (provided config takes precedence)
                logger.info("   Merging provided config with checkpoint config...")
                for field in vars(config):
                    if hasattr(checkpoint_config, field):
                        setattr(checkpoint_config, field, getattr(config, field))
                config = checkpoint_config
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
        # Initialize model with VAEConfig
        # dropout_rate: 0.0 = no dropout, 0.1-0.3 = mild regularization, 0.5+ = strong regularization
        # Dropout is applied to encoder fusion layers and decoder layers when enabled
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
                        value_device = value.to(device)
                        # Reshape tensors from [B, T, ...] to [B*T, ...]
                        # Multi-dimensional tensors (game_chars, game_colors, etc.)
                        B, T = value_device.shape[:2]
                        remaining_dims = value_device.shape[2:]
                        batch_device[key] = value_device.view(B * T, *remaining_dims)
                        
                    else:
                        batch_device[key] = value
                
                # Optional: shuffle within batch (ignore temporal order for VAE training)
                if shuffle_within_batch:
                    batch_size = batch_device['game_chars'].shape[0]  # B*T
                    shuffle_indices = torch.randperm(batch_size)
                    
                    for key, value in batch_device.items():
                        if value is not None and isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                            batch_device[key] = value[shuffle_indices]
                
                # Forward pass with mixed precision if enabled
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(use_bf16 and device.type == 'cuda')):
                    model_output = model(batch_device)
                    
                    # Calculate adaptive weights for this step
                    mi_beta, tc_beta, dw_beta = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
                    
                    # Calculate loss
                    train_loss_dict = vae_loss(
                        model_output=model_output,
                        batch=batch_device,
                        config=config,  # Use the VAEConfig object
                        mi_beta=mi_beta,
                        tc_beta=tc_beta,
                        dw_beta=dw_beta
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
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    train_loss.backward()
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
                        "train/raw_loss/value": train_loss_dict['raw_losses'].get('value', torch.tensor(0.0)).item(),
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
                        
                        # Dropout status
                        "dropout/rate": model.dropout_rate,
                        
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
                        "model/kl_eigenval_exceed_0.2": (kl_eig > 0.2).sum().item() / kl_eig.numel()
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
                        value_device = value.to(device)
                        # Reshape tensors from [B, T, ...] to [B*T, ...]
                        # Multi-dimensional tensors (game_chars, game_colors, etc.)
                        B, T = value_device.shape[:2]
                        remaining_dims = value_device.shape[2:]
                        batch_device[key] = value_device.view(B * T, *remaining_dims)
                    else:
                        batch_device[key] = value
                
                # Optional: shuffle within batch (ignore temporal order for VAE training)
                if shuffle_within_batch:
                    batch_size = batch_device['game_chars'].shape[0]  # B*T
                    shuffle_indices = torch.randperm(batch_size)
                    
                    for key, value in batch_device.items():
                        if value is not None and isinstance(value, torch.Tensor) and value.shape[0] == batch_size:
                            batch_device[key] = value[shuffle_indices]
                
                # Forward pass with mixed precision if enabled
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(use_bf16 and device.type == 'cuda')):
                    model_output = model(batch_device)
                    
                    # Calculate adaptive weights for this step (use current global step for consistency)
                    mi_beta, tc_beta, dw_beta = get_adaptive_weights(global_step, total_train_steps, custom_kl_beta_function)
                    
                    # Calculate loss
                    test_loss_dict = vae_loss(
                        model_output=model_output,
                        batch=batch_device,
                        config=config,
                        mi_beta=mi_beta,
                        tc_beta=tc_beta,
                        dw_beta=dw_beta
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
                        "test/raw_loss/value": test_loss_dict['raw_losses'].get('value', torch.tensor(0.0)).item(),
                        "test/raw_loss/safety": test_loss_dict['raw_losses'].get('safety', torch.tensor(0.0)).item(),
                        # Additional raw losses that were missing:
                        "test/raw_loss/goal": test_loss_dict['raw_losses'].get('goal', torch.tensor(0.0)).item(),
                        "test/raw_loss/forward_dynamics": test_loss_dict['raw_losses'].get('forward', torch.tensor(0.0)).item(),
                        "test/raw_loss/inverse_dynamics": test_loss_dict['raw_losses'].get('inverse', torch.tensor(0.0)).item(),

                        "test/kl_loss": test_loss_dict['kl_loss'].item(),
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
                    'model_config': {
                        'latent_dim': config.latent_dim,
                        'lowrank_dim': config.low_rank,
                    },
                    'training_timestamp': datetime.now().isoformat(),
                }, save_path)
                logger.info(f"💾 Model saved locally: {save_path}")
                
                repo_url = save_model_to_huggingface(
                    model=model,
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
            'model_config': {
                'latent_dim': config.latent_dim,
                'lowrank_dim': config.low_rank,
            },
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

def create_visualization_demo(
    repo_name: str,
    train_dataset: Optional[List[Dict]] = None,
    test_dataset: Optional[List[Dict]] = None,
    revision_name: Optional[str] = None,
    token: Optional[str] = None,
    device: str = "cpu",
    num_samples: int = 4,
    max_latent_samples: int = 100,
    save_dir: str = "vae_analysis",
    random_sampling: bool = True,
    random_seed: Optional[int] = None,
    # VAE sampling parameters
    use_mean: bool = True,
    include_logits: bool = False,
    # Map sampling parameters
    map_temperature: float = 1.0,
    map_occ_thresh: float = 0.5,
    rare_occ_thresh: float = 0.5,
    hero_presence_thresh: float = 0.5,
    map_deterministic: bool = True,
    glyph_top_k: int = 0,
    glyph_top_p: float = 1.0,
    color_top_k: int = 0,
    color_top_p: float = 1.0,
    # Message sampling parameters
    msg_temperature: float = 1.0,
    msg_top_k: int = 0,
    msg_top_p: float = 1.0,
    msg_deterministic: bool = True,
    allow_eos: bool = True,
    forbid_eos_at_start: bool = True,
    allow_pad: bool = False
) -> Dict:
    """
    Complete demo function that loads a model from HuggingFace and creates visualizations
    
    Args:
        repo_name: HuggingFace repository name
        train_dataset: Training dataset from NetHackDataCollector (optional)
        test_dataset: Test dataset from NetHackDataCollector (optional)
        token: HuggingFace token (optional)
        device: Device to run on
        num_samples: Number of reconstruction samples
        max_latent_samples: Maximum samples for latent analysis
        save_dir: Directory to save results
        random_sampling: Whether to use random sampling for reconstruction visualization
        random_seed: Random seed for reproducible sampling
        
        # VAE sampling parameters
        use_mean: If True, use mean of latent distribution; if False, sample from it
        include_logits: Whether to include raw logits in output
        
        # Map sampling parameters (legacy parameters map to new ones)
        map_occ_thresh: Threshold for occupancy prediction
        rare_occ_thresh: Threshold for rare occupancy prediction
        hero_presence_thresh: Threshold for hero presence prediction
        map_temperature: Temperature for map sampling (legacy: temperature)
        glyph_top_k: Top-k filtering for glyph sampling (legacy: top_k)
        glyph_top_p: Top-p filtering for glyph sampling (legacy: top_p)
        map_deterministic: If True, use deterministic sampling for map
        color_top_k: Top-k filtering for color sampling
        color_top_p: Top-p filtering for color sampling
        
        # Message sampling parameters
        msg_temperature: Temperature for message token sampling
        msg_top_k: Top-k filtering for message sampling
        msg_top_p: Top-p filtering for message sampling
        msg_deterministic: If True, use deterministic sampling for messages
        allow_eos: Whether to allow end-of-sequence tokens
        forbid_eos_at_start: Whether to forbid EOS tokens at start
        allow_pad: Whether to allow padding tokens
        
    Returns:
        Dictionary with analysis results
    """
    # Validate inputs
    if train_dataset is None and test_dataset is None:
        raise ValueError("At least one of train_dataset or test_dataset must be provided")
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 Starting VAE Analysis Demo")
    print(f"📦 Repository: {repo_name}")
    print(f"🎯 Device: {device}")
    print(f"📁 Save directory: {save_dir}")
    print(f"🎲 Random sampling: {random_sampling}")
    if random_seed is not None:
        print(f"🌱 Random seed: {random_seed}")
    
    # Load model from HuggingFace with local fallback
    print(f"\n1️⃣ Loading model from HuggingFace...")
    model = None
    
    try:
        model = load_model_from_huggingface(repo_name, token=token, device=device, revision_name=revision_name)
    except Exception as e:
        print(f"⚠️  Failed to load from HuggingFace: {e}")
        print(f"🔄 Attempting to load from local checkpoints...")
        
        # Try to find the latest local checkpoint
        checkpoint_dir = "checkpoints"
        local_checkpoint_path = None
        
        if os.path.exists(checkpoint_dir):
            # Find the latest checkpoint file
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoint_files:
                # Sort by modification time, latest first
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                local_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                print(f"📁 Found latest checkpoint: {local_checkpoint_path}")
            else:
                print(f"❌ No checkpoint files found in {checkpoint_dir}")
        
        # Also check for a saved model file
        if local_checkpoint_path is None:
            potential_paths = [
                "models/nethack-vae.pth",
                "nethack-vae.pth",
                "model.pth"
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    local_checkpoint_path = path
                    print(f"📁 Found saved model: {local_checkpoint_path}")
                    break
        
        if local_checkpoint_path is not None:
            try:
                model = load_model_from_local(local_checkpoint_path, device=device)
                print(f"✅ Successfully loaded model from local checkpoint")
            except Exception as local_e:
                print(f"❌ Failed to load from local checkpoint: {local_e}")
                raise RuntimeError(f"Failed to load model from both HuggingFace ({e}) and local checkpoint ({local_e})")
        else:
            print(f"❌ No local checkpoints found")
            raise RuntimeError(f"Failed to load model from HuggingFace ({e}) and no local checkpoints available")

    results = {'model': model, 'save_dir': save_dir}
    
    # Create TTY reconstructions for available datasets
    if train_dataset is not None:
        print(f"\n2️⃣ Creating TTY reconstruction visualizations for TRAINING dataset...")
        train_save_path = "train_recon_comparison.md"
        train_recon_results = visualize_reconstructions(
            model, train_dataset, device, 
            num_samples=num_samples, 
            out_dir=save_dir, 
            save_path=train_save_path,
            img_file_prefix="train_",
            random_sampling=random_sampling,
            dataset_name="Training",
            # VAE sampling parameters
            use_mean=use_mean,
            include_logits=include_logits,
            # Map sampling parameters (map legacy params)
            map_temperature=map_temperature,  # Legacy: temperature -> map_temperature
            map_occ_thresh=map_occ_thresh,
            rare_occ_thresh=rare_occ_thresh,
            hero_presence_thresh=hero_presence_thresh,
            map_deterministic=map_deterministic,
            glyph_top_k=glyph_top_k,  # Legacy: top_k -> glyph_top_k
            glyph_top_p=glyph_top_p,  # Legacy: top_p -> glyph_top_p
            color_top_k=color_top_k,
            color_top_p=color_top_p,
            # Message sampling parameters
            msg_temperature=msg_temperature,
            msg_top_k=msg_top_k,
            msg_top_p=msg_top_p,
            msg_deterministic=msg_deterministic,
            allow_eos=allow_eos,
            forbid_eos_at_start=forbid_eos_at_start,
            allow_pad=allow_pad
        )
        results['train_reconstruction_path'] = os.path.join(save_dir, train_save_path)
        results['train_reconstruction_results'] = train_recon_results
    
    if test_dataset is not None:
        print(f"\n2️⃣ Creating TTY reconstruction visualizations for TESTING dataset...")
        test_save_path = "test_recon_comparison.md"
        test_recon_results = visualize_reconstructions(
            model, test_dataset, device, 
            num_samples=num_samples, 
            out_dir=save_dir, 
            save_path=test_save_path,
            img_file_prefix="test_",
            random_sampling=random_sampling,
            dataset_name="Testing",
            # VAE sampling parameters
            use_mean=use_mean,
            include_logits=include_logits,
            # Map sampling parameters (map legacy params)
            map_temperature=map_temperature,  # Legacy: temperature -> map_temperature
            map_occ_thresh=map_occ_thresh,
            rare_occ_thresh=rare_occ_thresh,
            hero_presence_thresh=hero_presence_thresh,
            map_deterministic=map_deterministic,
            glyph_top_k=glyph_top_k,  # Legacy: top_k -> glyph_top_k
            glyph_top_p=glyph_top_p,  # Legacy: top_p -> glyph_top_p
            color_top_k=color_top_k,
            color_top_p=color_top_p,
            # Message sampling parameters
            msg_temperature=msg_temperature,
            msg_top_k=msg_top_k,
            msg_top_p=msg_top_p,
            msg_deterministic=msg_deterministic,
            allow_eos=allow_eos,
            forbid_eos_at_start=forbid_eos_at_start,
            allow_pad=allow_pad
        )
        results['test_reconstruction_path'] = os.path.join(save_dir, test_save_path)
        results['test_reconstruction_results'] = test_recon_results

    # Analyze latent space (use combined dataset or available one)
    print(f"\n3️⃣ Analyzing latent space...")
    
    # Combine datasets for latent analysis or use what's available
    analysis_datasets = []
    dataset_labels = []
    
    if train_dataset is not None:
        analysis_datasets.extend(train_dataset)
        dataset_labels.extend(['train'] * len(train_dataset))
    
    if test_dataset is not None:
        analysis_datasets.extend(test_dataset)
        dataset_labels.extend(['test'] * len(test_dataset))
    
    latent_path = os.path.join(save_dir, "latent_analysis.png")
    latent_analysis = analyze_latent_space(
        model, analysis_datasets, device, 
        save_path=latent_path, 
        max_samples=max_latent_samples,
        dataset_labels=dataset_labels
    )
    
    results['latent_analysis_path'] = latent_path
    results['latent_analysis'] = latent_analysis
    
    print(f"\n✅ Analysis complete! Results saved to: {save_dir}")
    if train_dataset is not None:
        print(f"📄 Training TTY reconstructions: {results['train_reconstruction_path']}")
    if test_dataset is not None:
        print(f"📄 Testing TTY reconstructions: {results['test_reconstruction_path']}")
    print(f"📊 Latent analysis plot: {latent_path}")
    
    return results

def analyze_glyph_char_color_pairs(
    dataset: List[Dict],
    top_k: int = 50,
    save_dir: str = "bin_count_analysis",
    save_plot: bool = True,
    show_ascii_chars: bool = True,
    save_complete_data: bool = True
) -> Dict:
    """
    Analyze the distribution of (glyph_char, glyph_color) pairs in the dataset.
    
    Args:
        dataset: List of data batches from NetHackDataCollector
        top_k: Number of top pairs to display
        save_dir: Directory to save analysis results
        save_plot: Whether to save the plot
        show_ascii_chars: Whether to show ASCII character representations
        save_complete_data: Whether to save complete count data to JSON
        
    Returns:
        Dictionary with analysis results
    """
    print(f"🔍 Starting glyph (char, color) pair analysis...")
    print(f"📊 Dataset size: {len(dataset)} batches")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Counter for (char, color) pairs
    pair_counter = Counter()
    total_cells = 0
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(dataset, desc="Processing batches")):
        game_chars = batch['game_chars']  # Shape: (num_games, num_time, 21, 79)
        game_colors = batch['game_colors']  # Shape: (num_games, num_time, 21, 79)
        
        # Flatten the spatial and temporal dimensions
        chars_flat = game_chars.flatten()  # All character codes
        colors_flat = game_colors.flatten()  # All color codes
        
        # Count pairs
        for char, color in zip(chars_flat.tolist(), colors_flat.tolist()):
            pair_counter[(char, color)] += 1
            total_cells += 1
    
    print(f"📈 Total cells analyzed: {total_cells:,}")
    print(f"🎨 Unique (char, color) pairs found: {len(pair_counter):,}")
    
    # Save complete count data to JSON if requested
    if save_complete_data:
        # Create readable format for pairs
        readable_pairs = {}
        for (char, color), count in pair_counter.items():
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            pair_key = f"({char},{color})"
            readable_pairs[pair_key] = {
                'char_code': char,
                'color_code': color,
                'ascii_char': ascii_repr,
                'count': count,
                'percentage': (count / total_cells) * 100
            }
        
        # Create readable format for characters
        char_counter = Counter()
        for (char, color), count in pair_counter.items():
            char_counter[char] += count
        
        readable_chars = {}
        for char, count in char_counter.items():
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            readable_chars[str(char)] = {
                'char_code': char,
                'ascii_char': ascii_repr,
                'total_count': count,
                'percentage': (count / total_cells) * 100
            }
        
        complete_data = {
            'total_cells': total_cells,
            'unique_pairs': len(pair_counter),
            'pair_counts': readable_pairs,
            'char_counts': readable_chars,
            'analysis_metadata': {
                'dataset_size': len(dataset),
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'glyph_char_color_pairs'
            }
        }
        
        # Save complete data
        complete_data_path = os.path.join(save_dir, "complete_bin_counts.json")
        with open(complete_data_path, 'w') as f:
            json.dump(complete_data, f, indent=2)
        print(f"💾 Complete count data saved to: {complete_data_path}")
    
    # Get top k pairs (excluding space character pairs)
    filtered_pairs = [(key, count) for key, count in pair_counter.items() if key[0] != 32]
    top_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Convert to readable format
    pair_data = []
    for (char, color), count in top_pairs:
        ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
        percentage = (count / total_cells) * 100
        pair_data.append({
            'char_code': char,
            'color_code': color,
            'ascii_char': ascii_repr,
            'count': count,
            'percentage': percentage
        })
    
    # Print top pairs
    print(f"\n🏆 Top {top_k} (char, color) pairs (excluding spaces):")
    print("-" * 80)
    if show_ascii_chars:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'ASCII':<8} {'Count':<12} {'Percentage':<10}")
    else:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'Count':<12} {'Percentage':<10}")
    print("-" * 80)
    
    for i, data in enumerate(pair_data):
        if show_ascii_chars:
            ascii_str = f"'{data['ascii_char']}'"
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{ascii_str:<8} {data['count']:<12,} {data['percentage']:<10.2f}%")
        else:
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{data['count']:<12,} {data['percentage']:<10.2f}%")
    
    # Create visualization
    if save_plot and len(top_pairs) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Bar chart of top pairs
        pair_labels = []
        counts = []
        colors_for_plot = []
        
        # Color mapping for NetHack colors (0-15)
        nethack_colors = [
            '#000000',  # 0: black
            '#800000',  # 1: red  
            '#008000',  # 2: green
            '#808000',  # 3: yellow
            '#000080',  # 4: blue
            '#800080',  # 5: magenta
            '#008080',  # 6: cyan
            '#C0C0C0',  # 7: white
            '#808080',  # 8: gray
            '#ff0000',  # 9: bright red
            '#00ff00',  # 10: bright green
            '#ffff00',  # 11: bright yellow
            '#0000ff',  # 12: bright blue
            '#ff00ff',  # 13: bright magenta
            '#00ffff',  # 14: bright cyan
            '#ffffff'   # 15: bright white
        ]
        
        for (char, color), count in top_pairs:
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            pair_labels.append(f"'{ascii_repr}' ({char}, {color})")
            counts.append(count)
            # Use actual NetHack color if available, otherwise use default
            if 0 <= color < len(nethack_colors):
                colors_for_plot.append(nethack_colors[color])
            else:
                colors_for_plot.append('#808080')  # Default gray
        
        bars = ax1.bar(range(len(counts)), counts, color=colors_for_plot, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('(Character, Color) Pairs')
        ax1.set_ylabel('Count (log scale)')
        ax1.set_yscale('log')
        ax1.set_title(f'Top {top_k} Most Frequent (Glyph Char, Glyph Color) Pairs (Excluding Spaces)')
        ax1.set_xticks(range(len(pair_labels)))
        ax1.set_xticklabels(pair_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Character distribution (top chars only, excluding space)
        char_counter = Counter()
        for (char, color), count in pair_counter.items():
            if char != 32:  # Exclude space character
                char_counter[char] += count
        
        top_chars = char_counter.most_common(20)  # Top 20 characters
        char_codes = [char for char, _ in top_chars]
        char_counts = [count for _, count in top_chars]
        char_labels = [f"'{chr(char)}' ({char})" if 32 <= char <= 126 else f"\\x{char:02x} ({char})" 
                      for char in char_codes]
        
        bars2 = ax2.bar(range(len(char_counts)), char_counts, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Character Codes')
        ax2.set_ylabel('Total Count (log scale)')
        ax2.set_yscale('log')
        ax2.set_title('Top 20 Most Frequent Characters (All Colors Combined, Excluding Spaces)')
        ax2.set_xticks(range(len(char_labels)))
        ax2.set_xticklabels(char_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars2, char_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(save_dir, f"glyph_char_color_analysis_top_{top_k}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()
    
    # Save detailed results to JSON
    results = {
        'total_cells': total_cells,
        'unique_pairs': len(pair_counter),
        'top_pairs': pair_data,
        'analysis_params': {
            'top_k': top_k,
            'dataset_size': len(dataset),
            'show_ascii_chars': show_ascii_chars
        }
    }
    
    if save_plot:
        results_path = os.path.join(save_dir, "glyph_analysis_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_path}")
    
    return results


def plot_glyph_char_color_pairs_from_saved(
    data_path: str,
    top_k: int = 50,
    save_dir: str = None,
    save_plot: bool = True,
    show_ascii_chars: bool = True,
    exclude_space: bool = True
) -> Dict:
    """
    Load saved bin count data and create visualizations.
    
    Args:
        data_path: Path to the saved complete_bin_counts.json file
        top_k: Number of top pairs to display
        save_dir: Directory to save plots (if None, uses directory of data_path)
        save_plot: Whether to save the plot
        show_ascii_chars: Whether to show ASCII character representations
        exclude_space: Whether to exclude space character (ASCII 32) from analysis
        
    Returns:
        Dictionary with analysis results
    """
    print(f"📥 Loading saved bin count data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Saved data file not found: {data_path}")
    
    # Load the complete data
    with open(data_path, 'r') as f:
        complete_data = json.load(f)
    
    total_cells = complete_data['total_cells']
    unique_pairs = complete_data['unique_pairs']
    
    print(f"📈 Loaded data: {total_cells:,} total cells, {unique_pairs:,} unique pairs")
    
    # Convert pair_counts back to Counter format
    pair_counter = Counter()
    for pair_key, pair_data in complete_data['pair_counts'].items():
        char = pair_data['char_code']
        color = pair_data['color_code']
        count = pair_data['count']
        pair_counter[(char, color)] = count
    
    # Set save directory
    if save_dir is None:
        save_dir = os.path.dirname(data_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter pairs if excluding space
    if exclude_space:
        filtered_pairs = [(key, count) for key, count in pair_counter.items() if key[0] != 32]
        print(f"🚫 Excluding space character pairs")
    else:
        filtered_pairs = list(pair_counter.items())
    
    # Get top k pairs
    top_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Convert to readable format
    pair_data = []
    for (char, color), count in top_pairs:
        ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
        percentage = (count / total_cells) * 100
        pair_data.append({
            'char_code': char,
            'color_code': color,
            'ascii_char': ascii_repr,
            'count': count,
            'percentage': percentage
        })
    
    # Print top pairs
    exclude_text = " (excluding spaces)" if exclude_space else ""
    print(f"\n🏆 Top {top_k} (char, color) pairs{exclude_text}:")
    print("-" * 80)
    if show_ascii_chars:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'ASCII':<8} {'Count':<12} {'Percentage':<10}")
    else:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'Count':<12} {'Percentage':<10}")
    print("-" * 80)
    
    for i, data in enumerate(pair_data):
        if show_ascii_chars:
            ascii_str = f"'{data['ascii_char']}'"
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{ascii_str:<8} {data['count']:<12,} {data['percentage']:<10.2f}%")
        else:
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{data['count']:<12,} {data['percentage']:<10.2f}%")
    
    # Create visualization
    if save_plot and len(top_pairs) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Bar chart of top pairs
        pair_labels = []
        counts = []
        colors_for_plot = []
        
        # Color mapping for NetHack colors (0-15)
        nethack_colors = [
            '#000000',  # 0: black
            '#800000',  # 1: red  
            '#008000',  # 2: green
            '#808000',  # 3: yellow
            '#000080',  # 4: blue
            '#800080',  # 5: magenta
            '#008080',  # 6: cyan
            '#C0C0C0',  # 7: white
            '#808080',  # 8: gray
            '#ff0000',  # 9: bright red
            '#00ff00',  # 10: bright green
            '#ffff00',  # 11: bright yellow
            '#0000ff',  # 12: bright blue
            '#ff00ff',  # 13: bright magenta
            '#00ffff',  # 14: bright cyan
            '#ffffff'   # 15: bright white
        ]
        
        for (char, color), count in top_pairs:
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            pair_labels.append(f"'{ascii_repr}' ({char}, {color})")
            counts.append(count)
            # Use actual NetHack color if available, otherwise use default
            if 0 <= color < len(nethack_colors):
                colors_for_plot.append(nethack_colors[color])
            else:
                colors_for_plot.append('#808080')  # Default gray
        
        bars = ax1.bar(range(len(counts)), counts, color=colors_for_plot, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('(Character, Color) Pairs')
        ax1.set_ylabel('Count (log scale)')
        ax1.set_yscale('log')
        title_suffix = " (Excluding Spaces)" if exclude_space else ""
        ax1.set_title(f'Top {top_k} Most Frequent (Glyph Char, Glyph Color) Pairs{title_suffix}')
        ax1.set_xticks(range(len(pair_labels)))
        ax1.set_xticklabels(pair_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Character distribution (top chars only, optionally excluding space)
        char_counter = Counter()
        for char_str, char_data in complete_data['char_counts'].items():
            char = char_data['char_code']
            count = char_data['total_count']
            if not exclude_space or char != 32:
                char_counter[char] = count
        
        top_chars = char_counter.most_common(20)  # Top 20 characters
        char_codes = [char for char, _ in top_chars]
        char_counts = [count for _, count in top_chars]
        char_labels = [f"'{chr(char)}' ({char})" if 32 <= char <= 126 else f"\\x{char:02x} ({char})" 
                      for char in char_codes]
        
        bars2 = ax2.bar(range(len(char_counts)), char_counts, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Character Codes')
        ax2.set_ylabel('Total Count (log scale)')
        ax2.set_yscale('log')
        char_title_suffix = " (Excluding Spaces)" if exclude_space else ""
        ax2.set_title(f'Top 20 Most Frequent Characters (All Colors Combined{char_title_suffix})')
        ax2.set_xticks(range(len(char_labels)))
        ax2.set_xticklabels(char_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars2, char_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_plot:
            plot_suffix = "_no_space" if exclude_space else ""
            plot_path = os.path.join(save_dir, f"glyph_char_color_analysis_top_{top_k}{plot_suffix}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()
    
    # Return results
    results = {
        'total_cells': total_cells,
        'unique_pairs': unique_pairs,
        'top_pairs': pair_data,
        'analysis_params': {
            'top_k': top_k,
            'exclude_space': exclude_space,
            'show_ascii_chars': show_ascii_chars,
            'data_source': data_path
        },
        'metadata': complete_data.get('analysis_metadata', {})
    }
    
    return results


if __name__ == "__main__":
    
    train_file = "nld-aa-training"
    test_file = "nld-aa-testing"
    data_cache_dir = "data_cache"
    batch_size = 32
    sequence_size = 32
    max_training_batches = 100
    max_testing_batches = 20
    train_cache_file = os.path.join(data_cache_dir, f"{train_file}_b{batch_size}_s{sequence_size}_m{max_training_batches}.pt")
    test_cache_file = os.path.join(data_cache_dir, f"{test_file}_b{batch_size}_s{sequence_size}_m{max_testing_batches}.pt")
    
    
    if len(sys.argv) > 1 and sys.argv[1] == "vae_analysis":
        # Demo mode: python train.py vae_analysis <repo_name> [revision_name]
        repo_name = sys.argv[2] if len(sys.argv) > 2 else "CatkinChen/nethack-vae"
        revision_name = sys.argv[3] if len(sys.argv) > 3 else None
        
        print(f"🚀 Running VAE Analysis Demo")
        print(f"📦 Repository: {repo_name}")
        
        # Create both training and test data
        print(f"📊 Preparing training and test data...")
        
        collector = NetHackDataCollector('ttyrecs.db')
        adapter = BLStatsAdapter()
        
        # Load training dataset
        print(f"📊 Loading training dataset...")
        train_dataset = collector.collect_or_load_data(
            dataset_name=train_file,
            adapter=adapter,
            save_path=train_cache_file,
            max_batches=max_training_batches,
            batch_size=batch_size,
            seq_length=sequence_size,
            force_recollect=False
        )
        
        # Load test dataset  
        print(f"📊 Loading test dataset...")
        test_dataset = collector.collect_or_load_data(
            dataset_name=test_file,
            adapter=adapter,
            save_path=test_cache_file,
            max_batches=max_testing_batches,
            batch_size=batch_size,
            seq_length=sequence_size,
            force_recollect=False
        )
        
        # Run the complete analysis on both datasets
        try:
            results = create_visualization_demo(
                repo_name=repo_name,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                revision_name=revision_name,
                device="cpu",  # Use CPU for demo
                num_samples=10,
                max_latent_samples=1000,  # More samples since we have both datasets
                save_dir="vae_analysis",
                random_sampling=True,  # Enable random sampling
                random_seed=50,  # For reproducible results
                use_mean=True,  # Use mean for latent space
                map_occ_thresh=0.5,
                rare_occ_thresh=0.5,
                hero_presence_thresh=0.2,
                map_deterministic=True  # Use deterministic sampling for maps
            )
            print(f"✅ Demo completed successfully!")
            print(f"📁 Results saved to: {results['save_dir']}")
            print(f"📊 Training dataset: {len(train_dataset)} batches")
            print(f"📊 Test dataset: {len(test_dataset)} batches")
            
            # Print detailed results
            if 'train_reconstruction_results' in results:
                print(f"🎨 Training reconstructions: {results['train_reconstruction_results']['num_samples']} samples")
            if 'test_reconstruction_results' in results:
                print(f"🎨 Test reconstructions: {results['test_reconstruction_results']['num_samples']} samples")
            if 'latent_analysis' in results:
                print(f"🧠 Latent analysis: {len(results['latent_analysis']['latent_vectors'])} total samples analyzed")
                
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"💡 Make sure the repository exists and is accessible")
            print(f"💡 You can create synthetic data for testing by setting repo_name to a local path")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "collect_data":
        # test collecting and saving data
        collector = NetHackDataCollector('ttyrecs.db')
        adapter = BLStatsAdapter()
        train_dataset = collector.collect_or_load_data(
            dataset_name=train_file,
            adapter=adapter,
            save_path=train_cache_file,
            max_batches=max_training_batches,
            batch_size=batch_size,
            seq_length=sequence_size,
            force_recollect=True
        )
        test_dataset = collector.collect_or_load_data(
            dataset_name=test_file,
            adapter=adapter,
            save_path=test_cache_file,
            max_batches=max_testing_batches,
            batch_size=batch_size,
            seq_length=sequence_size,
            force_recollect=True
        )
        
        print(f"✅ Data collection completed!")
        print(f"   📊 Train batches: {len(train_dataset)}")
        print(f"   📊 Test batches: {len(test_dataset)}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "bin_count_analysis":
        # Bin count analysis mode: python train.py bin_count_analysis [top_k] [dataset_type]
        top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        dataset_type = sys.argv[3] if len(sys.argv) > 3 else "both"  # "train", "test", or "both"
        
        print(f"🔍 Running Glyph (Char, Color) Bin Count Analysis")
        print(f"📊 Top K pairs to analyze: {top_k}")
        print(f"📁 Dataset type: {dataset_type}")
        
        # Prepare data collector
        collector = NetHackDataCollector('ttyrecs.db')
        adapter = BLStatsAdapter()
        
        datasets_to_analyze = []
        dataset_names = []
        
        if dataset_type in ["train", "both"]:
            print(f"📊 Loading training dataset...")
            train_dataset = collector.collect_or_load_data(
                dataset_name=train_file,
                adapter=adapter,
                save_path=train_cache_file,
                max_batches=max_training_batches,
                batch_size=batch_size,
                seq_length=sequence_size,
                force_recollect=False
            )
            datasets_to_analyze.append(train_dataset)
            dataset_names.append("train")
        
        if dataset_type in ["test", "both"]:
            print(f"📊 Loading test dataset...")
            test_dataset = collector.collect_or_load_data(
                dataset_name=test_file,
                adapter=adapter,
                save_path=test_cache_file,
                max_batches=max_testing_batches,
                batch_size=batch_size,
                seq_length=sequence_size,
                force_recollect=False
            )
            datasets_to_analyze.append(test_dataset)
            dataset_names.append("test")
        
        # Run analysis on each dataset
        for dataset, dataset_name in zip(datasets_to_analyze, dataset_names):
            print(f"\n🔬 Analyzing {dataset_name} dataset...")
            save_dir = f"bin_count_analysis/{dataset_name}"
            
            try:
                results = analyze_glyph_char_color_pairs(
                    dataset=dataset,
                    top_k=top_k,
                    save_dir=save_dir,
                    save_plot=True,
                    show_ascii_chars=True,
                    save_complete_data=True
                )
                
                print(f"✅ {dataset_name.capitalize()} analysis completed!")
                print(f"📁 Results saved to: {save_dir}")
                print(f"📊 Total cells: {results['total_cells']:,}")
                print(f"🎨 Unique pairs: {results['unique_pairs']:,}")
                
            except Exception as e:
                print(f"❌ {dataset_name.capitalize()} analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        # If analyzing both datasets, create a combined analysis
        if dataset_type == "both" and len(datasets_to_analyze) == 2:
            print(f"\n🔗 Creating combined analysis...")
            combined_dataset = datasets_to_analyze[0] + datasets_to_analyze[1]
            save_dir = "bin_count_analysis/combined"
            
            try:
                results = analyze_glyph_char_color_pairs(
                    dataset=combined_dataset,
                    top_k=top_k,
                    save_dir=save_dir,
                    save_plot=True,
                    show_ascii_chars=True,
                    save_complete_data=True
                )
                
                print(f"✅ Combined analysis completed!")
                print(f"📁 Results saved to: {save_dir}")
                print(f"📊 Total cells: {results['total_cells']:,}")
                print(f"🎨 Unique pairs: {results['unique_pairs']:,}")
                
            except Exception as e:
                print(f"❌ Combined analysis failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 Bin count analysis completed!")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "plot_bin_count":
        # Plot from saved data mode: python train.py plot_bin_count <data_path> [top_k] [exclude_space]
        if len(sys.argv) < 3:
            print("❌ Usage: python train.py plot_bin_count <data_path> [top_k] [exclude_space]")
            print("   Example: python train.py plot_bin_count bin_count_analysis/train/complete_bin_counts.json 30 true")
            sys.exit(1)
        
        data_path = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 50
        exclude_space = sys.argv[4].lower() in ['true', '1', 'yes'] if len(sys.argv) > 4 else True
        
        print(f"📊 Plotting bin count analysis from saved data")
        print(f"📁 Data path: {data_path}")
        print(f"📊 Top K pairs: {top_k}")
        print(f"🚫 Exclude spaces: {exclude_space}")
        
        try:
            results = plot_glyph_char_color_pairs_from_saved(
                data_path=data_path,
                top_k=top_k,
                save_plot=True,
                show_ascii_chars=True,
                exclude_space=exclude_space
            )
            
            print(f"✅ Plot generation completed!")
            print(f"📊 Total cells: {results['total_cells']:,}")
            print(f"🎨 Unique pairs: {results['unique_pairs']:,}")
            print(f"📈 Showing top {len(results['top_pairs'])} pairs")
            
        except Exception as e:
            print(f"❌ Plot generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        hf_model_card_data = {
            "author": "Xu Chen",
            "description": "Advanced NetHack VAE",
            "tags": ["nethack", "reinforcement-learning", "multimodal", "world-modeling", "vae"],
            "use_cases": [
                "Game state representation learning",
                "RL agent state abstraction",
                "NetHack gameplay analysis"
            ],
        }
        
        print(f"\n🧪 Starting train_multimodalhack_vae...")
        model, train_losses, test_losses = train_multimodalhack_vae(
            train_file=train_file,
            test_file=test_file,
            epochs=15,          
            batch_size=batch_size,
            sequence_size=sequence_size,    
            max_learning_rate=1e-3,
            training_batches=max_training_batches,
            testing_batches=max_testing_batches,
            max_training_batches=max_training_batches,
            max_testing_batches=max_testing_batches,
            save_path="models/nethack-vae.pth",
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_bf16=False,  # Enable BF16 mixed precision training
            data_cache_dir="data_cache",
            force_recollect=False,  # Use the data we just collected
            shuffle_batches=True,  # Shuffle training batches each epoch for better training
            shuffle_within_batch=True,  # Shuffle within each batch for more variety
            initial_mi_beta=0.0,
            final_mi_beta=0.0,
            mi_beta_shape='constant',
            initial_tc_beta=5.0,
            final_tc_beta=5.0,
            tc_beta_shape='constant',
            initial_dw_beta=0.02,
            final_dw_beta=1.0,
            dw_beta_shape='linear',
            custom_kl_beta_function = lambda init, end, progress: init + (end - init) * min(progress, 0.2) * 5.0, 
            warmup_epoch_ratio = 0.2,
            free_bits=0.15,
            focal_loss_alpha=0.75,
            focal_loss_gamma=2.0,

            # Dropout and regularization settings
            dropout_rate=0.1,  # Set to 0.1 for mild regularization
            enable_dropout_on_latent=True,
            enable_dropout_on_decoder=True,
            
            # Early stopping settings
            early_stopping = False,
            early_stopping_patience = 3,
            early_stopping_min_delta = 0.01,

            # Enable checkpointing
            save_checkpoints=True,
            checkpoint_dir="checkpoints",
            save_every_n_epochs=1,
            keep_last_n_checkpoints=2,
            
            # Wandb integration example
            use_wandb=True,
            wandb_project="nethack-vae",
            wandb_entity="xchen-catkin-ucl",  # Replace with your wandb username
            wandb_run_name=f"vae-test-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            wandb_tags=["nethack", "vae"],
            wandb_notes="Full VAE training run",
            log_every_n_steps=5,  # Log every 5 steps
            log_model_architecture=True,
            log_gradients=True,
            
            # HuggingFace integration example
            upload_to_hf=True, 
            hf_repo_name="CatkinChen/nethack-vae",
            hf_upload_directly=True,  # Upload directly without extra local save
            hf_upload_checkpoints=True,  # Also upload checkpoints
            hf_model_card_data=hf_model_card_data
        )

        print(f"\n🎉 Full VAE training run completed successfully!")
        print(f"   📈 Train losses: {train_losses}")
        print(f"   📈 Test losses: {test_losses}")
