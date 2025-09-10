# Sequential Skill Preservation with Curiosity-driven Reinforcement Learning

This project implements an online PPO training system that uses pre-trained VAE and HMM models to learn sequential skills in NetHack environments with curiosity-driven intrinsic rewards.

## Features

- 🧠 **Pre-trained VAE + HMM Models**: Load skill representations from HuggingFace
- 🎮 **MiniHack Integration**: 161+ NetHack-based RL environments  
- 🔍 **Curiosity-Driven Learning**: Multiple intrinsic motivation mechanisms
- 📊 **Experiment Tracking**: Weights & Biases integration
- 🤗 **Model Sharing**: Automatic HuggingFace model uploads
- 🚀 **Production Ready**: Clean, tested, and maintainable codebase

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
Then relaunch the terminal.

### 4. Build MiniHack Wheel (Required for Proper Environment Registration)
```bash
cd SequentialSkillRL/minihack
python setup.py bdist_wheel
cd ..
```

### 5. Install Dependencies
```bash
# Install Poetry dependencies (this will automatically install NLE and MiniHack)
poetry install

# If poetry command is not found, use the full path:
/root/.local/bin/poetry install
```

### 6. Install Dependencies
```bash
# Install Poetry dependencies (this will automatically install NLE and MiniHack)
poetry install

# If poetry command is not found, use the full path:
/root/.local/bin/poetry install
```

### 7. Verify Installation
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

### 8. Environment Activation (Optional)
```bash
# Change to the project directory first
cd /workspace/SequentialSkillRL
# Get the environment path and activate it
source $(poetry env info --path)/bin/activate
```

### 9. Login to External Services
```bash
# Login to Weights & Biases for experiment tracking
wandb login

# Login to Hugging Face CLI for model uploads
huggingface-cli login
```

## Testing the Installation

Run the comprehensive test suite:
```bash
# Test MiniHack installation
poetry run python training/online_rl.py minihack_check

# Run all system tests  
poetry run python training/online_rl.py test
```

## Troubleshooting

### MiniHack Environments Not Found
If you see "0 MiniHack environments found", ensure you:
1. Built the MiniHack wheel: `cd minihack && python setup.py bdist_wheel`
2. Reinstalled dependencies: `poetry install --no-cache`
3. Updated submodules: `git submodule update --init --recursive`

### CMake Issues
If you encounter cmake-related errors during NLE compilation:
```bash
pip install --upgrade cmake
```

## Quick Start

### Training an Agent
```python
from training.online_rl import train_online_ppo_with_pretrained_models

# Train PPO with pre-trained VAE and HMM models
results = train_online_ppo_with_pretrained_models(
    vae_repo_name="your-username/nethack-vae",
    hmm_repo_name="your-username/nethack-hmm", 
    env_id="MiniHack-Room-5x5-v0",
    total_env_steps=100000,
    use_wandb=True,
    wandb_project="my-ppo-experiment",
    upload_to_hf=True,
    hf_repo_name="your-username/ppo-agent",
    device="cuda"
)
print(f"Training completed! Best return: {results['best_eval_return']}")
```

### Quick Test Mode
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
# Check MiniHack installation
poetry run python training/online_rl.py minihack_check

# Run system tests  
poetry run python training/online_rl.py test
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
│   └── training_utils.py   # Training utilities
├── utils/                   # Utility functions
│   ├── env_utils.py        # Environment utilities
│   ├── action_utils.py     # Action space utilities  
│   └── analysis.py         # Analysis and visualization
├── nle/                     # NetHack Learning Environment (submodule)
├── minihack/               # MiniHack environments (submodule)
├── models/                 # Pre-trained model storage
├── checkpoints/            # Training checkpoints
└── data_cache/            # Cached datasets
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
- **Tracking**: Real-time metrics and model uploading

### 3. Experiment Management
- **W&B Integration**: Automatic experiment tracking
- **HuggingFace Hub**: Model versioning and sharing
- **Checkpointing**: Resume training from any point

## Technical Notes

### MiniHack Installation Solution
This project uses a specialized installation approach for MiniHack to avoid circular import issues:

1. **Problem**: MiniHack's editable installation (`pip install -e .`) caused circular imports that prevented environment registration
2. **Solution**: Install MiniHack from a built wheel instead of editable mode
3. **Result**: All 161 MiniHack environments register properly without sys.path workarounds

For details, see [MINIHACK_SOLUTION.md](MINIHACK_SOLUTION.md).

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NetHack Learning Environment (NLE) team
- MiniHack team for the extensive environment suite
- HuggingFace for model hosting and sharing infrastructure