# Sequential Skill Preservation with Curiosity-base Reinforcement Learning

## Setup Instructions

```bash
git clone https://github.com/XuChenCatkin/SequentialSkillRL.git
cd SequentialSkillRL
git submodule update --init --recursive
sudo apt-get update
sudo apt-get install -y build-essential libboost-context-dev python3-dev libsdl2-dev libx11-dev build-essential cmake bison flex pkg-config
sudo apt install -y pipx
pipx ensurepath
```
Then relaunch the terminal and run
```bash
pipx install poetry
poetry install
# If poetry command is not found, use the full path:
/root/.local/bin/poetry install
```

In terminal, run
```bash
# Get the environment path and activate it
source $(/root/.local/bin/poetry env info --path)/bin/activate
```
to activate the venv created by poetry.