# Online PPO Training for NetHack/MiniHack

This module provides a comprehensive online reinforcement learning training pipeline that combines pre-trained VAE and HMM models with PPO for NetHack/MiniHack environments.

## Features

- 🎯 **Pre-trained Model Integration**: Load VAE and HMM models from HuggingFace Hub
- 🎮 **Environment Flexibility**: Support for MiniHack environments with fallback to standard Gym environments
- 🧠 **Curiosity-Driven Learning**: Multiple intrinsic reward mechanisms (dynamics surprise, skill entropy, RND)
- 📊 **Comprehensive Monitoring**: Weights & Biases integration with detailed metrics logging
- 🤗 **HuggingFace Integration**: Automatic model and checkpoint uploading
- ⚙️ **Highly Configurable**: Customizable PPO, curiosity, and HMM parameters
- 🧪 **Test-Friendly**: Built-in test mode and fallback mechanisms

## Quick Start

### Basic Usage

```python
from training.online_rl import train_online_ppo_with_pretrained_models

# Train PPO agent with pre-trained VAE+HMM
results = train_online_ppo_with_pretrained_models(
    vae_repo_name="your-username/nethack-vae",
    hmm_repo_name="your-username/nethack-hmm",
    env_id="MiniHack-Room-5x5-v0",
    total_env_steps=100000,
    use_wandb=True,
    wandb_project="my-ppo-experiment"
)

print(f"Best return: {results['best_eval_return']}")
```

### Test Mode (Quick Validation)

```python
# Quick test with minimal steps
results = train_online_ppo_with_pretrained_models(
    vae_repo_name="your-username/nethack-vae", 
    hmm_repo_name="your-username/nethack-hmm",
    test_mode=True,
    test_steps=1000,
    use_wandb=False,
    upload_to_hf=False
)
```

### Custom Configuration

```python
from rl.ppo import PPOConfig, CuriosityConfig

ppo_config = PPOConfig(
    num_envs=16,
    rollout_len=256,
    learning_rate=1e-4
)

curiosity_config = CuriosityConfig(
    use_dyn_kl=True,
    use_skill_entropy=True,
    eta0_dyn=0.5
)

results = train_online_ppo_with_pretrained_models(
    vae_repo_name="your-username/nethack-vae",
    hmm_repo_name="your-username/nethack-hmm", 
    ppo_config=ppo_config,
    curiosity_config=curiosity_config
)
```

## Command Line Testing

```bash
# Basic integration test
python training/online_rl.py test

# Test with mock models
python training/online_rl.py mock_test

# Test environment creation
python training/online_rl.py env_test

# Show example usage
python training/online_rl.py example
```

## Key Components

### 1. Model Loading
- Loads pre-trained VAE models from HuggingFace Hub
- Loads pre-trained HMM models with proper device placement
- Validates model compatibility (latent dimensions, etc.)

### 2. Environment Handling
- Primary support for MiniHack environments
- Automatic fallback to standard Gym environments if MiniHack unavailable
- Robust error handling and environment validation

### 3. PPO Training
- Uses the PPOTrainer class from `rl/ppo.py`
- Supports curiosity-driven intrinsic rewards
- Online HMM updates during training
- Comprehensive evaluation metrics

### 4. Monitoring & Logging
- Weights & Biases integration for experiment tracking
- Detailed training metrics and diagnostics
- HuggingFace Hub uploads for model sharing
- Structured logging with configurable verbosity

## Configuration Options

### PPOConfig
- `num_envs`: Number of parallel environments
- `rollout_len`: Length of rollout sequences
- `learning_rate`: PPO learning rate
- `clip_coef`: PPO clipping coefficient
- `policy_uses_skill`: Whether to use HMM skills in policy

### CuriosityConfig  
- `use_dyn_kl`: Enable dynamics surprise reward
- `use_skill_entropy`: Enable skill entropy reward
- `use_rnd`: Enable Random Network Distillation
- `eta0_*`: Initial reward scaling factors
- `tau_*`: Reward annealing time constants

### HMMOnlineConfig
- `hmm_update_every`: Steps between HMM updates
- `hmm_fit_window`: Window size for HMM fitting
- `rho_emission`: Streaming update rate for emissions

## Error Handling & Fallbacks

The system is designed to be robust:

- **MiniHack Unavailable**: Falls back to CartPole-v1 or other Gym environments
- **PPO Components Missing**: Uses mock configurations for testing
- **HuggingFace Errors**: Gracefully handles network/auth issues
- **Device Mismatches**: Automatic device placement and validation

## Testing

Built-in test functions validate:
- ✅ Basic imports and configuration creation
- ✅ Mock VAE and HMM model compatibility
- ✅ Environment creation with fallbacks
- ✅ Integration between components

## Requirements

- PyTorch
- Gymnasium
- HuggingFace Hub
- Weights & Biases (optional)
- MiniHack/NLE (optional, with fallbacks)

## Next Steps

1. **Fix MiniHack Setup**: Properly configure MiniHack environment registration
2. **Complete PPO Implementation**: Ensure all PPO components are fully implemented
3. **Add More Environments**: Support additional NetHack environments
4. **Hyperparameter Tuning**: Optimize default configurations
5. **Distributed Training**: Support for multi-GPU training

## Architecture

```
Training Pipeline:
├── Model Loading (HuggingFace)
│   ├── VAE Model
│   └── HMM Model  
├── Environment Setup
│   ├── MiniHack (preferred)
│   └── Gym Fallback
├── PPO Training
│   ├── Policy Network
│   ├── Value Network
│   └── Curiosity Module
├── Monitoring
│   ├── Weights & Biases
│   └── HuggingFace Uploads
└── Evaluation & Checkpointing
```

This implementation provides a solid foundation for online PPO training with pre-trained VAE+HMM models, with robust error handling and extensive configurability.
