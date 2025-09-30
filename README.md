# Sequential Skill Preservation with Curiosity-driven Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements an online PPO training system that uses pre-trained VAE and HMM models to learn sequential skills in NetHack environments with curiosity-driven intrinsic rewards.

## Features

- 🧠 **Pre-trained VAE + HMM Models**: Load skill representations from HuggingFace
- 🎮 **MiniHack Integration**: 161+ NetHack-based RL environments  
- 🔍 **Curiosity-Driven Learning**: Multiple intrinsic motivation mechanisms
- 📊 **Experiment Tracking**: Weights & Biases integration
- 🤗 **Model Sharing**: Automatic HuggingFace model uploads
- 🚀 **Production Ready**: Clean, tested, and maintainable codebase

You can find the full report [here](thesis.pdf).

## Available Pre-trained Models

All the latest trained PPO models are available on Hugging Face under the namespace `CatkinChen/nethack-*`. These models include:

🔗 **Model Repository**: [https://huggingface.co/CatkinChen](https://huggingface.co/CatkinChen)

### Available Models
All models follow the naming convention `CatkinChen/nethack-*` and include:
- **PPO Policy Networks**: Trained on various MiniHack environments
- **VAE + HMM Models**: For skill representation and sequential learning
- **Complete Training Artifacts**: Including training curves, configurations, and logs

To use any of these models, simply reference them by their full repository ID (e.g., `CatkinChen/nethack-vae-hmm`, `CatkinChen/nethack-hmm`) in your training scripts.

## Requirements

### System Requirements
- **Operating System**: Ubuntu 20.04+ (tested), macOS 10.15+, or other Unix-like systems
- **Python**: 3.10 or higher (required)
- **Memory**: 40GB+ recommended for training
- **Storage**: 30GB+ free disk space for dependencies and model storage
- **GPU**: CUDA-compatible GPU recommended for training (optional for inference)

### Hardware Recommendations
- **Training**: NVIDIA GPU with 40GB+ VRAM for optimal performance
- **CPU**: Multi-core processor (4+ cores recommended)
- **Development**: Any modern machine with 8GB+ RAM for code development and testing

### Software Dependencies
- **Core**: Python 3.10+, Poetry package manager
- **System Libraries**: CMake, Boost, SDL2, X11 development headers
- **Python Packages**: PyTorch 2.7+, Transformers, Gymnasium, W&B, HuggingFace Hub
- **Optional**: CUDA toolkit for GPU acceleration

### Supported Environments
- **Primary**: 161+ MiniHack NetHack-based environments
- **Tested On**: MiniHack-Room-5x5-v0, MiniHack-Quest-Hard-v0, MiniHack-KeyRoom-S15-v0
- **Platform**: Works on Linux, macOS (with some additional setup for system dependencies)

## Setup Instructions

### 1. Clone Repository and Submodules
```bash
git clone https://github.com/XuChenCatkin/SequentialSkillRL.git
cd SequentialSkillRL
git submodule update --init --recursive
```

### 2. Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y build-essential libboost-context-dev python3-dev libsdl2-dev libx11-dev cmake bison flex pkg-config
```

### 3. Install Poetry
```bash
sudo apt install -y pipx
pipx install poetry
pipx ensurepath
```
**⚠️ Important**: After installing Poetry, you MUST restart your terminal or source your shell profile:
```bash
source ~/.bashrc   # For bash users
# OR
source ~/.zshrc    # For zsh users
# OR simply close and reopen your terminal
```

Verify Poetry installation:
```bash
poetry --version
```

### 4. Build MiniHack Wheel (Required for Proper Environment Registration)
```bash
cd SequentialSkillRL/minihack
python setup.py bdist_wheel
cd ..
```

### 5. Install Dependencies
```bash
# Option 1: Use the provided installation script (recommended - installs everything)
./install_minihack.sh

# Option 2: Install all dependencies with Poetry (simple one-command approach)
poetry install

# Option 3: Manual step-by-step installation
# First build and install MiniHack wheel (bypasses Poetry hash issues)
cd minihack && python setup.py bdist_wheel && cd ..
pip install minihack/dist/minihack-1.0.2+95b11cc-py3-none-any.whl --force-reinstall
# Then install all other dependencies
poetry install

# Option 4: If Poetry fails due to lock file issues, update and install
poetry lock && poetry install
```

### 6. Verify Installation
```bash
# Test that MiniHack environments are properly registered
poetry run python -c "
import gymnasium as gym
import minihack
envs = [env for env in gym.envs.registry.keys() if 'MiniHack' in env]
print(f'✅ Found {len(envs)} MiniHack environments')
assert len(envs) > 0, 'MiniHack environments not found!'
print('✅ Installation successful!')
"
```

### 7. Environment Activation (Optional)
```bash
# Change to the project directory first
cd /workspace/SequentialSkillRL
# Get the environment path and activate it
source $(poetry env info --path)/bin/activate
```

### 8. Login to External Services
```bash
# Login to Weights & Biases for experiment tracking
wandb login

# Login to Hugging Face CLI for model uploads
hf auth login
```

## Troubleshooting

### Poetry Command Not Found
If you see "poetry: command not found" when running `./install_minihack.sh`:
1. Ensure Poetry is installed: `pipx install poetry`
2. Update your PATH: `pipx ensurepath`
3. **Restart your terminal** or run: `source ~/.bashrc`
4. Verify: `poetry --version`

### MiniHack Environments Not Found
If you see "0 MiniHack environments found", ensure you:
1. Built the MiniHack wheel: `cd minihack && python setup.py bdist_wheel`
2. Installed with pip: `pip install minihack/dist/minihack-1.0.2+95b11cc-py3-none-any.whl --force-reinstall`
3. Updated submodules: `git submodule update --init --recursive`

### Poetry Installation Issues
If Poetry fails to install dependencies:
```bash
# Option 1: Simple Poetry install (works in most cases)
poetry install

# Option 2: Use the fixed installation script (recommended)
./install_minihack.sh

# Option 3: Manual step-by-step (for MiniHack issues)
cd minihack && python setup.py bdist_wheel && cd ..
pip install minihack/dist/minihack-1.0.2+95b11cc-py3-none-any.whl --force-reinstall
poetry install

# Option 4: Update Poetry lock file if there are hash mismatches
poetry lock && poetry install

# Option 5: Clear Poetry cache and reinstall
poetry cache clear --all .
poetry install
```

### CMake Issues
If you encounter cmake-related errors during NLE compilation:
```bash
pip install --upgrade cmake
```

## Quick Start

### Training an Agent
```python
from training.online_rl import train_online_ppo_with_pretrained_models

# Train PPO with pre-trained VAE and HMM models from HuggingFace
# All models are available at https://huggingface.co/CatkinChen with naming convention 'nethack-*'
results = train_online_ppo_with_pretrained_models(
    vae_repo_id="CatkinChen/nethack-vae-hmm",
    hmm_repo_id="CatkinChen/nethack-hmm", 
    env_name="MiniHack-Room-5x5-v0",
    total_timesteps=50000,
    use_wandb=True,
    wandb_project="SequentialSkillRL",
    push_to_hub=True,  # Upload all components to unified repo
    hub_repo_id_vae_hmm="your-username/nethack-complete-model",
    device="cuda"
)
print(f"Training completed! Run: {results['run_name']}")
```

### Loading Pre-trained Models
```python
# Load the latest trained PPO models from CatkinChen's HuggingFace repository
# Browse all available models at: https://huggingface.co/CatkinChen
# All NetHack models follow the pattern: CatkinChen/nethack-*

# Example: Load VAE and HMM models
results = train_online_ppo_with_pretrained_models(
    vae_repo_id=None,
    hmm_repo_id=None,
    # ... other parameters
    ppo_repo_id=CatkinChen/nethack-ppo-ablation-baseline_full_curiosity,
    reset_global_steps=True,
    # ... other parameters
)
```

### Quick Test Mode
```python
# Quick test with minimal steps
results = train_online_ppo_with_pretrained_models(
    vae_repo_id=None,
    hmm_repo_id=None,
    ppo_repo_id=CatkinChen/nethack-ppo-ablation-baseline_full_curiosity,
    test_mode=True,
    test_episodes=10,
    use_wandb=False,
    push_to_hub=False
)
```

### Custom Configurations
```python
from rl.ppo import PPOConfig, CuriosityConfig

# Custom PPO configuration
ppo_config = PPOConfig(
    num_envs=16,
    rollout_len=256,
    learning_rate=1e-4,
    clip_coef=0.1
)

# Custom curiosity configuration
curiosity_config = CuriosityConfig(
    use_dyn_kl=True,
    use_skill_entropy=True,
    use_rnd=False,
    eta0_dyn=0.5,
    tau_dyn=1e6
)

results = train_online_ppo_with_pretrained_models(
    vae_repo_name="your-username/nethack-vae",
    hmm_repo_name="your-username/nethack-hmm",
    ppo_config=ppo_config,
    curiosity_config=curiosity_config,
    total_env_steps=1000000
)
```

### Command Line Usage
```bash
# Basic training with latest pre-trained models from CatkinChen's HuggingFace repository
# All models available at: https://huggingface.co/CatkinChen (pattern: nethack-*)
python main.py rl baseline full_curiosity \
  --env MiniHack-Room-5x5-v0 \
  --steps 50000
  --resume CatkinChen/nethack-ppo-ablation-baseline_full_curiosity
  --reset_step

# Training with different model and reward configurations
python main.py rl baseline curiosity_dyn_only \
  --env MiniHack-Quest-Hard-v0 \
  --steps 100000
  --resume CatkinChen/nethack-ppo-ablation-baseline_curiosity_dyn_only
  --reset_step

# Training without HMM (VAE only)
python main.py rl no_hmm curiosity_dyn_only \
  --env MiniHack-Room-Random-15x15-v0 \
  --steps 50000
  --resume CatkinChen/nethack-ppo-ablation-no_hmm_curiosity_dyn_only
  --reset_step

# Training with Random Network Distillation
python main.py rl baseline rnd \
  --env MiniHack-KeyRoom-S15-v0 \
  --steps 100000
  --resume CatkinChen/nethack-ppo-ablation-baseline_rnd
  --reset_step

# Training with no intrinsic rewards (extrinsic only)
python main.py rl baseline no_intrinsic \
  --env MiniHack-River-Narrow-v0 \
  --steps 50000
  --resume CatkinChen/nethack-ppo-ablation-no_hmm_no_intrinsic
  --reset_step

# Custom seed for reproducibility
python main.py rl baseline full_curiosity \
  --env MiniHack-Room-5x5-v0 \
  --steps 50000 \
  --seed 123
  --resume CatkinChen/nethack-ppo-ablation-baseline_full_curiosity
  --reset_step
```

## Project Structure

```
SequentialSkillRL/
├── src/                     # Core source code
│   ├── model.py            # VAE and HMM model definitions
│   ├── skill_space.py      # Skill space management
│   └── data_collection.py  # Data collection utilities
├── training/                # Training pipeline
│   ├── train.py            # Main training script
│   ├── online_rl.py        # Online PPO training system
│   ├── training_utils.py   # Training utilities
│   └── README_online_rl.md # Training documentation
├── rl/                      # Reinforcement learning components
│   └── ppo.py              # PPO implementation
├── utils/                   # Utility functions
│   ├── env_utils.py        # Environment utilities
│   ├── action_utils.py     # Action space utilities
│   ├── analysis.py         # Analysis and visualization
│   └── math_utils.py       # Mathematical utilities
├── nle/                     # NetHack Learning Environment (submodule)
├── minihack/               # MiniHack environments (submodule)
├── runs/                    # Training run outputs and logs
├── logs/                    # Training and experiment logs
├── wandb/                   # Weights & Biases experiment tracking
├── checkpoints_hmm/         # HMM model checkpoints
├── bin_count_analysis/      # Action frequency analysis
├── hmm_analysis/            # HMM analysis results
├── vae_analysis/           # VAE analysis results
├── vae_hmm_analysis/       # Combined VAE+HMM analysis
├── main.py                 # Main entry point
├── analyze_ablations.py    # Ablation study analysis
├── run_ablations.py        # Ablation study runner
├── run_experiments.sh      # Experiment automation script
├── install_minihack.sh     # Installation helper script
├── evaluation.ipynb        # Evaluation notebook
├── nld_tutorial.ipynb      # NLD tutorial notebook
├── trial.ipynb             # Trial experiments notebook
├── pyproject.toml          # Poetry project configuration
└── poetry.lock             # Poetry dependency lock file
```

## Key Components

### 1. VAE + HMM Models
- **VAE**: Encodes NetHack observations into latent skill representations
- **HMM**: Models sequential skill transitions and dynamics
- **Integration**: Combined for curiosity-driven exploration

### 2. Online PPO Training  
- **Environment**: MiniHack-based NetHack environments
- **Policy**: Uses skill-aware policy networks
- **Intrinsic Rewards**: Dynamic KL divergence, skill entropy, RND

### 3. HuggingFace Integration
- **Model Upload**: Automatically uploads trained models (PPO policy, VAE, HMM) to unified repositories
- **Training Artifacts**: Uploads training curves, logs, and configuration files
- **Model Cards**: Generates comprehensive model documentation
- **Separate Repositories**: Supports loading VAE and HMM from different repositories

#### Training Artifacts Include:
- **Training Curves**: Reward progression and performance metrics over time
- **Configuration Files**: Complete training hyperparameters and settings
- **Model Cards**: Detailed documentation with usage examples
- **Training Logs**: Step-by-step training metrics and evaluation results

#### Usage Example:
```bash
# Train and upload complete model with training artifacts
# Note: HuggingFace integration is built into the training pipeline
python main.py rl baseline full_curiosity \
  --env MiniHack-Room-5x5-v0 \
  --steps 100000

# Models are automatically loaded from CatkinChen's repositories
# Training results and checkpoints are saved locally in runs/ directory
```
- **Tracking**: Real-time metrics and model uploading

### 3. Experiment Management
- **W&B Integration**: Automatic experiment tracking
- **HuggingFace Hub**: Model versioning and sharing
- **Checkpointing**: Resume training from any point

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## Acknowledgments

- NetHack Learning Environment (NLE) team
- MiniHack team for the extensive environment suite
- HuggingFace for model hosting and sharing infrastructure