#!/bin/bash

echo "Starting RL ablation experiments..."

python main.py rl baseline full_curiosity --env MiniHack-River-Narrow-v0 --steps 200000 --seed 51 --wandb --resume CatkinChen/nethack-ppo-ablation-baseline_full_curiosity --vae_revision v2.0 --hmm_revision v2.0 --reset_step
python main.py rl baseline curiosity_dyn_only --env MiniHack-River-Narrow-v0 --steps 200000 --seed 51 --wandb --resume CatkinChen/nethack-ppo-ablation-baseline_curiosity_dyn_only --vae_revision v2.0 --hmm_revision v2.0 --reset_step
python main.py rl baseline rnd --env MiniHack-River-Narrow-v0 --steps 200000 --seed 51 --wandb --resume CatkinChen/nethack-ppo-ablation-baseline_rnd --vae_revision v2.0 --hmm_revision v2.0 --reset_step
python main.py rl no_hmm curiosity_dyn_only --env MiniHack-River-Narrow-v0 --steps 200000 --seed 51 --wandb --resume CatkinChen/nethack-ppo-ablation-no_hmm_curiosity_dyn_only --vae_revision v2.0 --hmm_revision v2.0 --reset_step
python main.py rl no_hmm no_intrinsic --env MiniHack-River-Narrow-v0 --steps 200000 --seed 51 --wandb --resume CatkinChen/nethack-ppo-ablation-no_hmm_no_intrinsic --vae_revision v2.0 --hmm_revision v2.0 --reset_step

python main.py rl baseline full_curiosity --env MiniHack-KeyRoom-S15-v0 --steps 200000 --seed 51 --wandb --resume CatkinChen/nethack-ppo-ablation-baseline_full_curiosity --reset_step
python main.py rl baseline curiosity_dyn_only --env MiniHack-KeyRoom-S15-v0 --steps 200000 --seed 51 --wandb --resume CatkinChen/nethack-ppo-ablation-baseline_curiosity_dyn_only --reset_step
python main.py rl baseline rnd --env MiniHack-KeyRoom-S15-v0 --steps 200000 --seed 51 --wandb --resume CatkinChen/nethack-ppo-ablation-baseline_rnd --reset_step
python main.py rl no_hmm curiosity_dyn_only --env MiniHack-KeyRoom-S15-v0 --steps 200000 --seed 51 --wandb --resume CatkinChen/nethack-ppo-ablation-no_hmm_curiosity_dyn_only --reset_step
python main.py rl no_hmm no_intrinsic --env MiniHack-KeyRoom-S15-v0 --steps 200000 --seed 51 --wandb --resume CatkinChen/nethack-ppo-ablation-no_hmm_no_intrinsic --reset_step


echo "All experiments completed."