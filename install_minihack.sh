#!/bin/bash
# Quick installation script for MiniHack to avoid Poetry hash issues

set -e  # Exit on any error

echo "🔧 Installing MiniHack for SequentialSkillRL..."

# Ensure we're in the right directory
if [ ! -d "minihack" ]; then
    echo "❌ Error: minihack directory not found. Please run this script from the SequentialSkillRL root directory."
    exit 1
fi

# Clean and rebuild MiniHack wheel
echo "📦 Building MiniHack wheel..."
cd minihack
rm -rf dist/ build/ *.egg-info
python setup.py bdist_wheel
cd ..

# Install with pip (bypasses Poetry hash issues)
echo "⚙️ Installing MiniHack wheel..."
pip install minihack/dist/minihack-1.0.2+95b11cc-py3-none-any.whl --force-reinstall

# Install remaining dependencies
echo "📚 Installing remaining dependencies..."
poetry install --only-root

# Test installation
echo "🧪 Testing installation..."
python -c "
import gymnasium as gym
import minihack
envs = [env for env in gym.envs.registry.keys() if 'MiniHack' in env]
print(f'✅ Found {len(envs)} MiniHack environments')
assert len(envs) > 0, 'MiniHack environments not found!'
"

echo "🎉 Installation completed successfully!"
echo "You can now run:"
echo "  python training/online_rl.py --test_mode --test_episodes 5"
