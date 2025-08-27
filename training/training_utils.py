from typing import List, Dict, Optional
import os
import json
import torch
from datetime import datetime
from huggingface_hub import HfApi, login, hf_hub_download, HfFileSystem
from matplotlib import pyplot as plt
from src.model import MultiModalHackVAE, VAEConfig

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
    # Define a dummy exception for when HF is not available
    class RepositoryNotFoundError(Exception):
        pass

# Scikit-learn availability check
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

def save_checkpoint(
    model: MultiModalHackVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_losses: List[float],
    test_losses: List[float],
    config = None,  # VAEConfig object
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    scaler = None,  # Remove specific type annotation since GradScaler API changed
    checkpoint_dir: str = "checkpoints",
    keep_last_n: int = 3,
    upload_to_hf: bool = False,
    hf_repo_name: str = None,
    hf_token: str = None
) -> str:
    """
    Save training checkpoint with optional HuggingFace upload
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        train_losses: Training loss history
        test_losses: Test loss history
        scheduler: Learning rate scheduler to save (optional)
        scaler: GradScaler for mixed precision training (optional)
        checkpoint_dir: Directory to save checkpoints
        keep_last_n: Number of recent checkpoints to keep (older ones are deleted)
        upload_to_hf: Whether to upload checkpoint to HuggingFace
        hf_repo_name: HuggingFace repository name
        hf_token: HuggingFace token
        
    Returns:
        Path to saved checkpoint
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create checkpoint filename
    checkpoint_filename = f"checkpoint_epoch_{epoch+1:04d}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'config': config,
        'checkpoint_timestamp': datetime.now().isoformat(),
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
    }
    
    # Add scheduler state if available
    if scheduler is not None:
        checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add scaler state if available
    if scaler is not None:
        checkpoint_data['scaler_state_dict'] = scaler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint_data, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Upload to HuggingFace if requested
    if upload_to_hf and hf_repo_name and HF_AVAILABLE:
        try:
            if hf_token:
                login(token=hf_token)
            
            api = HfApi()
            
            # Upload checkpoint to checkpoints/ folder in repo
            remote_path = f"checkpoints/{checkpoint_filename}"
            api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=remote_path,
                repo_id=hf_repo_name,
                repo_type="model",
                commit_message=f"Add checkpoint for epoch {epoch+1}"
            )
            print(f"☁️  Checkpoint uploaded to HuggingFace: {remote_path}")
            
        except Exception as e:
            print(f"⚠️  Failed to upload checkpoint to HuggingFace: {e}")
    
    # Clean up old checkpoints (keep only last N)
    if keep_last_n > 0:
        cleanup_old_checkpoints(checkpoint_dir, keep_last_n)
    
    return checkpoint_path


def cleanup_old_checkpoints(checkpoint_dir: str, keep_last_n: int) -> None:
    """
    Remove old checkpoint files, keeping only the most recent N checkpoints
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
    try:
        # Get all checkpoint files
        checkpoint_files = []
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith("checkpoint_epoch_") and filename.endswith(".pth"):
                filepath = os.path.join(checkpoint_dir, filename)
                # Extract epoch number for sorting
                try:
                    epoch_str = filename.replace("checkpoint_epoch_", "").replace(".pth", "")
                    epoch_num = int(epoch_str)
                    checkpoint_files.append((epoch_num, filepath))
                except ValueError:
                    continue
        
        # Sort by epoch number (newest first)
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints
        if len(checkpoint_files) > keep_last_n:
            for epoch_num, filepath in checkpoint_files[keep_last_n:]:
                os.remove(filepath)
                print(f"🗑️  Removed old checkpoint: {os.path.basename(filepath)}")
                
    except Exception as e:
        print(f"⚠️  Error cleaning up checkpoints: {e}")



def save_model_to_huggingface(
    model: MultiModalHackVAE,
    config: VAEConfig,
    model_save_path: str = None,
    repo_name: str = None,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: str = "Upload MultiModalHackVAE model",
    model_card_data: Optional[Dict] = None,
    upload_directly: bool = False,
    additional_files: Optional[Dict[str, str]] = None
) -> str:
    """
    Save trained MultiModalHackVAE model to HuggingFace Hub
    
    Args:
        model: Trained MultiModalHackVAE model
        model_save_path: Local path where model is saved (optional if upload_directly=True)
        repo_name: Name for the HuggingFace repository (e.g., "username/nethack-vae")
        token: HuggingFace token (if None, will try to use cached token)
        private: Whether to create a private repository
        commit_message: Commit message for the upload
        model_card_data: Additional metadata for the model card
        upload_directly: If True, save model to temp file and upload directly without permanent local save
        additional_files: Dict of {local_path: remote_path} for additional files to upload
        
    Returns:
        Repository URL on HuggingFace Hub
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace Hub is required. Install with: pip install huggingface_hub")
    
    # Login if token is provided
    if token:
        login(token=token)
    
    api = HfApi()
    
    try:
        # Check if repository exists, create if not
        try:
            repo_info = api.repo_info(repo_id=repo_name, repo_type="model")
            print(f"📁 Repository {repo_name} already exists")
        except Exception as e:
            # Handle both RepositoryNotFoundError and other HTTP errors
            if "404" in str(e) or "not found" in str(e).lower() or isinstance(e, RepositoryNotFoundError):
                print(f"🆕 Creating new repository: {repo_name}")
                api.create_repo(repo_id=repo_name, private=private, repo_type="model")
            else:
                raise e
    
        # Handle direct upload vs local file upload
        temp_model_file = None
        if upload_directly:
            # Create temporary file for direct upload
            import tempfile
            temp_model_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
            model_save_path = temp_model_file.name
            
            # Save model state dict to temporary file
            print(f"💾 Creating temporary model file for direct upload...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'upload_timestamp': datetime.now().isoformat(),
            }, model_save_path)
            temp_model_file.close()
        elif model_save_path is None:
            raise ValueError("Either model_save_path must be provided or upload_directly must be True")
    
        # Prepare model metadata
        model_info = {
            "model_type": "MultiModalHackVAE",
            "framework": "PyTorch",
            "task": "representation-learning",
            "dataset": "NetHack Learning Dataset",
            "latent_dim": getattr(model, 'latent_dim', 'unknown'),
            "lowrank_dim": getattr(model, 'lowrank_dim', 'unknown'),
            "architecture": "Multi-modal Variational Autoencoder for NetHack game states"
        }
        
        if model_card_data:
            model_info.update(model_card_data)
        
        # Create model card
        model_card_content = f"""---
license: mit
language: en
tags:
- nethack
- reinforcement-learning
- variational-autoencoder
- representation-learning
- multimodal
- world-modeling
pipeline_tag: feature-extraction
---

# MultiModalHackVAE

A multi-modal Variational Autoencoder trained on NetHack game states for representation learning.

## Model Description

This model is a MultiModalHackVAE that learns compact representations of NetHack game states by processing:
- Game character grids (21x79)
- Color information
- Game statistics (blstats)
- Message text
- Bag of glyphs
- Hero information (role, race, gender, alignment)

## Model Details

- **Model Type**: Multi-modal Variational Autoencoder
- **Framework**: PyTorch
- **Dataset**: NetHack Learning Dataset
- **Latent Dimensions**: {model_info.get('latent_dim', 'unknown')}
- **Low-rank Dimensions**: {model_info.get('lowrank_dim', 'unknown')}

## Usage

```python
from train import load_model_from_huggingface
import torch

# Load the model
model = load_model_from_huggingface("{repo_name}")

# Example usage with synthetic data
batch_size = 1
game_chars = torch.randint(32, 127, (batch_size, 21, 79))
game_colors = torch.randint(0, 16, (batch_size, 21, 79))
blstats = torch.randn(batch_size, 27)
msg_tokens = torch.randint(0, 128, (batch_size, 256))
hero_info = torch.randint(0, 10, (batch_size, 4))

with torch.no_grad():
    output = model(
        glyph_chars=game_chars,
        glyph_colors=game_colors,
        blstats=blstats,
        msg_tokens=msg_tokens,
        hero_info=hero_info
    )
    latent_mean = output['mu']
    latent_logvar = output['logvar']
    lowrank_factors = output['lowrank_factors']
```

## Training

This model was trained using adaptive loss weighting with:
- Embedding warm-up for quick convergence
- Gradual raw reconstruction focus
- KL beta annealing for better latent structure

## Citation

If you use this model, please consider citing:

```bibtex
@misc{{nethack-vae,
  title={{MultiModalHackVAE: Multi-modal Variational Autoencoder for NetHack}},
  author={{Xu Chen}},
  year={{2025}},
  url={{https://huggingface.co/{repo_name}}}
}}
```
"""
        
        # Save model card
        model_card_path = "VAE_README.md"
        with open(model_card_path, "w") as f:
            f.write(model_card_content)
        
        # Save model config
        config_path = "VAE_config.json"
        with open(config_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Upload files
        print(f"📤 Uploading model to {repo_name}...")
        
        # Upload model file
        api.upload_file(
            path_or_fileobj=model_save_path,
            path_in_repo="pytorch_model.bin",
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message
        )
        
        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Add model card"
        )
        
        # Upload config
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Add model config"
        )
        
        # Upload additional files if provided
        if additional_files:
            for local_path, remote_path in additional_files.items():
                if os.path.exists(local_path):
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=remote_path,
                        repo_id=repo_name,
                        repo_type="model",
                        commit_message=f"Add {remote_path}"
                    )
        
        # Clean up temporary files
        os.remove(model_card_path)
        os.remove(config_path)
        
        # Clean up temporary model file if created
        if temp_model_file is not None:
            os.unlink(model_save_path)
            print(f"🗑️  Cleaned up temporary model file")
        
        repo_url = f"https://huggingface.co/{repo_name}"
        print(f"✅ Model successfully uploaded to: {repo_url}")
        return repo_url
        
    except Exception as e:
        print(f"❌ Error uploading to HuggingFace: {e}")
        # Clean up temporary file on error
        if temp_model_file is not None and os.path.exists(model_save_path):
            os.unlink(model_save_path)
        raise


def load_model_from_local(
    checkpoint_path: str,
    device: str = "cpu",
    **model_kwargs
) -> MultiModalHackVAE:
    """
    Load MultiModalHackVAE model from local checkpoint
    
    Args:
        checkpoint_path: Path to local checkpoint file
        device: Device to load the model on
        **model_kwargs: Additional arguments for model initialization (override config)
        
    Returns:
        Loaded MultiModalHackVAE model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    try:
        print(f"📥 Loading model from local checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Check if we have a modern VAEConfig in the checkpoint
        if 'config' in checkpoint and hasattr(checkpoint['config'], 'latent_dim'):
            # Modern checkpoint with VAEConfig
            config = checkpoint['config']
            print(f"🏗️  Using saved VAEConfig from checkpoint")
        else:
            # Create default VAEConfig
            config = VAEConfig()
            print(f"🏗️  No config in checkpoint. Created default VAEConfig")
        
        # Apply any overrides from model_kwargs
        if model_kwargs:
            print(f"🔧 Applying config overrides: {model_kwargs}")
            for key, value in model_kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    print(f"⚠️  Unknown config parameter ignored: {key}={value}")
        
        print(f"🏗️  Initializing model with VAEConfig (latent_dim={config.latent_dim})")
        model = MultiModalHackVAE(config=config)
        
        # Load state dict
        print(f"⚡ Loading model weights...")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state dict from checkpoint")
            if 'final_train_loss' in checkpoint:
                print(f"📊 Final training loss: {checkpoint['final_train_loss']:.4f}")
            if 'final_test_loss' in checkpoint:
                print(f"📊 Final test loss: {checkpoint['final_test_loss']:.4f}")
        else:
            # Fallback: try to load the checkpoint directly as state dict
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model state dict directly")
        
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully from local checkpoint")
        print(f"🎯 Model on device: {device}")
        print(f"🎯 Model in evaluation mode")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading from local checkpoint: {e}")
        raise


def load_model_from_huggingface(
    repo_name: str,
    revision_name: Optional[str] = None,
    token: Optional[str] = None,
    device: str = "cpu",
    **model_kwargs
) -> MultiModalHackVAE:
    """
    Load MultiModalHackVAE model from HuggingFace Hub
    
    Args:
        repo_name: HuggingFace repository name (e.g., "username/nethack-vae")
        revision_name: Specific revision to load (default is latest)
        token: HuggingFace token (if needed for private repos)
        device: Device to load the model on
        **model_kwargs: Additional arguments for model initialization (override config)
        
    Returns:
        Loaded MultiModalHackVAE model
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace Hub is required. Install with: pip install huggingface_hub")
    
    # Login if token is provided
    if token:
        login(token=token)
    
    api = HfApi()
    
    try:
        # Download config
        print(f"📥 Downloading model config from {repo_name}...")
        config_path = hf_hub_download(
            repo_id=repo_name,
            filename="config.json",
            repo_type="model",
            revision=revision_name
        )
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        print(f"📋 Model config loaded: {config}")
        
        # Download model file
        print(f"📥 Downloading model weights from {repo_name}...")
        model_path = hf_hub_download(
            repo_id=repo_name,
            filename="pytorch_model.bin",
            repo_type="model",
            revision=revision_name
        )
        
        # Load checkpoint to check format
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if we have a modern VAEConfig in the config or checkpoint
        if 'config' in checkpoint and hasattr(checkpoint['config'], 'latent_dim'):
            # Modern checkpoint with VAEConfig
            vae_config = checkpoint['config']
            print(f"🏗️  Using saved VAEConfig from checkpoint")
        else:
            # Create default VAEConfig
            vae_config = VAEConfig()
            print(f"🏗️  No config in checkpoint. Created default VAEConfig")
        
        # Apply any overrides from model_kwargs
        if model_kwargs:
            print(f"🔧 Applying config overrides: {model_kwargs}")
            for key, value in model_kwargs.items():
                if hasattr(vae_config, key):
                    setattr(vae_config, key, value)
                else:
                    print(f"⚠️  Unknown config parameter ignored: {key}={value}")
        
        print(f"🏗️  Initializing model with VAEConfig (latent_dim={vae_config.latent_dim})")
        model = MultiModalHackVAE(config=vae_config)
        
        # Load state dict
        print(f"⚡ Loading model weights...")
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state dict from checkpoint")
            if 'final_train_loss' in checkpoint:
                print(f"📊 Final training loss: {checkpoint['final_train_loss']:.4f}")
            if 'final_test_loss' in checkpoint:
                print(f"📊 Final test loss: {checkpoint['final_test_loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model state dict directly")
        
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully from HuggingFace: {repo_name}")
        print(f"🎯 Model on device: {device}")
        print(f"🎯 Model in evaluation mode")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading from HuggingFace: {e}")
        raise


def upload_training_artifacts_to_huggingface(
    repo_name: str,
    train_losses: List[float],
    test_losses: List[float],
    training_config: Dict,
    token: Optional[str] = None,
    plots_dir: str = "training_plots"
) -> None:
    """
    Upload training artifacts (losses, plots, config) to HuggingFace
    
    Args:
        repo_name: HuggingFace repository name
        train_losses: List of training losses per epoch
        test_losses: List of test losses per epoch
        training_config: Dictionary with training configuration
        token: HuggingFace token
        plots_dir: Directory name for plots in the repo
    """
    if not HF_AVAILABLE:
        print("⚠️  HuggingFace Hub not available, skipping artifact upload")
        return
    
    if token:
        login(token=token)
    
    api = HfApi()
    
    try:
        # Create training plots
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss improvement plot
        plt.subplot(1, 2, 2)
        if len(train_losses) > 1:
            train_improvement = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
            test_improvement = [(test_losses[0] - loss) / test_losses[0] * 100 for loss in test_losses]
            plt.plot(epochs, train_improvement, 'b-', label='Training Improvement (%)', linewidth=2)
            plt.plot(epochs, test_improvement, 'r-', label='Test Improvement (%)', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Improvement (%)')
            plt.title('Loss Improvement Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save training data
        training_data = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "config": training_config,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_test_loss": test_losses[-1] if test_losses else None,
            "total_epochs": len(train_losses),
            "best_train_loss": min(train_losses) if train_losses else None,
            "best_test_loss": min(test_losses) if test_losses else None
        }
        
        with open("training_data.json", "w") as f:
            json.dump(training_data, f, indent=2)
        
        # Upload files
        api.upload_file(
            path_or_fileobj="training_curves.png",
            path_in_repo=f"{plots_dir}/training_curves.png",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Add training curves"
        )
        
        api.upload_file(
            path_or_fileobj="training_data.json",
            path_in_repo="training_data.json",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Add training data"
        )
        
        # Clean up
        os.remove("training_curves.png")
        os.remove("training_data.json")
        
        print(f"✅ Training artifacts uploaded to {repo_name}")
        
    except Exception as e:
        print(f"❌ Error uploading training artifacts: {e}")


def create_model_demo_notebook(repo_name: str, save_path: str = "demo_notebook.ipynb") -> None:
    """
    Create a Jupyter notebook demonstrating model usage
    
    Args:
        repo_name: HuggingFace repository name
        save_path: Path to save the notebook
    """
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# MultiModalHackVAE Demo\n\n",
                    f"This notebook demonstrates how to use the MultiModalHackVAE model from {repo_name}.\n\n",
                    "## Installation\n\n",
                    "```bash\n",
                    "pip install torch transformers huggingface_hub\n",
                    "# For NetHack environment (optional):\n",
                    "pip install nle\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import torch\n",
                    "import numpy as np\n",
                    "from huggingface_hub import hf_hub_download\n",
                    "import json\n",
                    "\n",
                    "# Load model config\n",
                    f"config_path = hf_hub_download(repo_id='{repo_name}', filename='config.json')\n",
                    "with open(config_path, 'r') as f:\n",
                    "    config = json.load(f)\n",
                    "\n",
                    "print('Model Configuration:')\n",
                    "for key, value in config.items():\n",
                    "    print(f'  {key}: {value}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load the model (you'll need to import your model class)\n",
                    "# from your_package import MultiModalHackVAE\n",
                    "# model = load_model_from_huggingface('{repo_name}')\n",
                    "\n",
                    "# Example synthetic data\n",
                    "batch_size = 1\n",
                    "game_chars = torch.randint(32, 127, (batch_size, 21, 79))\n",
                    "game_colors = torch.randint(0, 16, (batch_size, 21, 79))\n",
                    "blstats = torch.randn(batch_size, 27)\n",
                    "msg_tokens = torch.randint(0, 128, (batch_size, 256))\n",
                    "hero_info = torch.randint(0, 10, (batch_size, 4))\n",
                    "\n",
                    "print('Synthetic data shapes:')\n",
                    "print(f'  game_chars: {game_chars.shape}')\n",
                    "print(f'  game_colors: {game_colors.shape}')\n",
                    "print(f'  blstats: {blstats.shape}')\n",
                    "print(f'  msg_tokens: {msg_tokens.shape}')\n",
                    "print(f'  hero_info: {hero_info.shape}')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Encode to latent space\n",
                    "# with torch.no_grad():\n",
                    "#     output = model(\n",
                    "#         glyph_chars=game_chars,\n",
                    "#         glyph_colors=game_colors,\n",
                    "#         blstats=blstats,\n",
                    "#         msg_tokens=msg_tokens,\n",
                    "#         hero_info=hero_info\n",
                    "#     )\n",
                    "#     \n",
                    "#     latent_mean = output['mu']\n",
                    "#     latent_logvar = output['logvar']\n",
                    "#     lowrank_factors = output['lowrank_factors']\n",
                    "#     \n",
                    "#     print(f'Latent representation shape: {latent_mean.shape}')\n",
                    "#     print(f'Latent mean: {latent_mean[0][:5].tolist()}')\n",
                    "\n",
                    "print('Model inference example (uncomment when model is available)')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8+"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open(save_path, "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"📓 Demo notebook created: {save_path}")

def save_hmm_to_huggingface(
    hmm,  # StickyHDPHMMVI
    hmm_params,  # StickyHDPHMMParams
    niw_prior,  # NIWPrior
    config: VAEConfig,
    repo_name: str,
    round_num: int,
    total_rounds: int,
    token: Optional[str] = None,
    private: bool = True,
    commit_message: str = None,
    base_vae_repo: str = None
) -> str:
    """
    Save Sticky-HDP-HMM model to HuggingFace Hub
    
    Args:
        hmm: Trained StickyHDPHMMVI model
        hmm_params: HMM parameters
        niw_prior: NIW prior
        config: VAEConfig from associated VAE
        repo_name: HuggingFace repository name
        round_num: Current EM round number
        total_rounds: Total number of EM rounds
        token: HuggingFace token
        private: Whether to create private repository
        commit_message: Commit message
        base_vae_repo: Base VAE repository name
        
    Returns:
        Repository URL on HuggingFace Hub
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace Hub is required. Install with: pip install huggingface_hub")
    
    # Login if token is provided
    if token:
        login(token=token)
    
    api = HfApi()
    
    try:
        # Check if repository exists, create if not
        try:
            repo_info = api.repo_info(repo_id=repo_name, repo_type="model")
            print(f"📁 Repository {repo_name} already exists")
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower() or isinstance(e, RepositoryNotFoundError):
                print(f"🆕 Creating new repository: {repo_name}")
                api.create_repo(repo_id=repo_name, private=private, repo_type="model")
            else:
                raise e
    
        # Prepare HMM data
        import tempfile
        temp_hmm_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        hmm_data = {
            'hmm_state_dict': hmm.state_dict(),
            'hmm_params': hmm_params,
            'niw_prior': niw_prior,
            'round': round_num,
            'total_rounds': total_rounds,
            'config': config,
            'training_timestamp': datetime.now().isoformat(),
        }
        torch.save(hmm_data, temp_hmm_file.name)
        temp_hmm_file.close()
        
        # Prepare model metadata
        model_info = {
            "model_type": "StickyHDPHMM",
            "framework": "PyTorch",
            "task": "latent-dynamics-modeling",
            "round": round_num,
            "total_rounds": total_rounds,
            "latent_dim": config.latent_dim,
            "num_skills": config.skill_num,
            "hmm_parameters": {
                "alpha": hmm_params.alpha,
                "kappa": hmm_params.kappa,
                "gamma": hmm_params.gamma,
                "K": hmm_params.K,
                "D": hmm_params.D
            },
            "base_vae_repo": base_vae_repo or "unknown"
        }
        
        # Create model card
        model_card_content = f"""---
license: mit
language: en
tags:
- nethack
- reinforcement-learning
- hmm
- sticky-hdp-hmm
- latent-dynamics
- variational-inference
pipeline_tag: other
---

# Sticky-HDP-HMM for NetHack (Round {round_num}/{total_rounds})

A Sticky Hierarchical Dirichlet Process Hidden Markov Model trained on NetHack latent representations for learning skills and temporal dynamics.

## Model Description

This is a Sticky-HDP-HMM model that learns discrete skills and temporal transitions in the latent space of a NetHack VAE. The model uses:
- Sticky self-transitions to encourage temporal persistence of skills
- Hierarchical Dirichlet Process for automatic skill discovery
- Normal-Inverse-Wishart priors for skill emission distributions

## Model Details

- **Model Type**: Sticky Hierarchical Dirichlet Process Hidden Markov Model
- **Framework**: PyTorch with Variational Inference
- **EM Round**: {round_num} of {total_rounds}
- **Latent Dimensions**: {model_info.get('latent_dim', 'unknown')}
- **Maximum Skills**: {model_info.get('num_skills', 'unknown')}
- **Base VAE**: {base_vae_repo or 'local checkpoint'}

## HMM Parameters

- **Alpha (DP concentration)**: {hmm_params.alpha}
- **Kappa (sticky parameter)**: {hmm_params.kappa}
- **Gamma (top-level DP)**: {hmm_params.gamma}

## Usage

```python
from train import load_hmm_from_huggingface
import torch

# Load the HMM
hmm, config = load_hmm_from_huggingface("{repo_name}")

# The HMM can be used with a VAE for skill-based generation
```

## Training

This HMM was trained using Expectation-Maximization on VAE latent representations:
- E-step: Variational inference for posterior skill assignments
- M-step: VAE fine-tuning with HMM skill prior

## Citation

If you use this model, please consider citing:

```bibtex
@misc{{nethack-hmm,
  title={{Sticky-HDP-HMM for NetHack Skill Learning}},
  author={{Xu Chen}},
  year={{2025}},
  url={{https://huggingface.co/{repo_name}}}
}}
```
"""
        
        # Save model card
        model_card_path = "HMM_README.md"
        with open(model_card_path, "w") as f:
            f.write(model_card_content)
        
        # Save model config
        config_path = "HMM_config.json"
        with open(config_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        # Upload files
        print(f"📤 Uploading HMM to {repo_name}...")
        
        # Upload HMM file
        api.upload_file(
            path_or_fileobj=temp_hmm_file.name,
            path_in_repo=f"hmm_round{round_num}.pt",
            repo_id=repo_name,
            repo_type="model",
            commit_message=commit_message or f"Upload Sticky-HDP-HMM round {round_num}/{total_rounds}"
        )
        
        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Add HMM model card"
        )
        
        # Upload config
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=repo_name,
            repo_type="model",
            commit_message="Add HMM config"
        )
        
        # Clean up temporary files
        os.unlink(temp_hmm_file.name)
        os.remove(model_card_path)
        os.remove(config_path)
        
        repo_url = f"https://huggingface.co/{repo_name}"
        print(f"✅ HMM successfully uploaded to: {repo_url}")
        return repo_url
        
    except Exception as e:
        print(f"❌ Error uploading HMM to HuggingFace: {e}")
        # Clean up temporary file on error
        if 'temp_hmm_file' in locals() and os.path.exists(temp_hmm_file.name):
            os.unlink(temp_hmm_file.name)
        raise


def load_hmm_from_huggingface(
    repo_name: str,
    round_num: int = None,
    revision_name: Optional[str] = None,
    token: Optional[str] = None,
    device: str = "cpu"
) -> tuple:
    """
    Load Sticky-HDP-HMM model from HuggingFace Hub
    
    Args:
        repo_name: HuggingFace repository name
        round_num: Specific round number to load (if None, loads latest)
        revision_name: Specific revision to load
        token: HuggingFace token
        device: Device to load the model on
        
    Returns:
        Tuple of (hmm, config, hmm_params, niw_prior, metadata)
    """
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace Hub is required. Install with: pip install huggingface_hub")
    
    # Login if token is provided
    if token:
        login(token=token)
    
    try:
        # Download config first
        print(f"📥 Downloading HMM config from {repo_name}...")
        config_path = hf_hub_download(
            repo_id=repo_name,
            filename="config.json",
            repo_type="model",
            revision=revision_name
        )
        
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        # Determine which round to load
        if round_num is None:
            round_num = config_data.get('round', 1)
        
        # Download HMM file
        hmm_filename = f"hmm_round{round_num}.pt"
        print(f"📥 Downloading HMM weights from {repo_name}: {hmm_filename}...")
        
        hmm_path = hf_hub_download(
            repo_id=repo_name,
            filename=hmm_filename,
            repo_type="model",
            revision=revision_name
        )
        
        # Load HMM data
        hmm_checkpoint = torch.load(hmm_path, map_location=device, weights_only=False)
        
        # Extract components
        config = hmm_checkpoint['config']
        hmm_params = hmm_checkpoint['hmm_params'] 
        niw_prior = hmm_checkpoint['niw_prior']
        
        # Recreate HMM
        from src.skill_space import StickyHDPHMMVI
        hmm = StickyHDPHMMVI(
            p=hmm_params,
            niw_prior=niw_prior
        )
        hmm.load_state_dict(hmm_checkpoint['hmm_state_dict'])
        hmm = hmm.to(device)
        
        print(f"✅ HMM loaded successfully from HuggingFace: {repo_name}")
        print(f"🎯 Round: {round_num}, Device: {device}")
        
        metadata = {
            'round': hmm_checkpoint.get('round', round_num),
            'total_rounds': hmm_checkpoint.get('total_rounds', 'unknown'),
            'training_timestamp': hmm_checkpoint.get('training_timestamp', 'unknown')
        }
        
        return hmm, config, hmm_params, niw_prior, metadata
        
    except Exception as e:
        print(f"❌ Error loading HMM from HuggingFace: {e}")
        raise

