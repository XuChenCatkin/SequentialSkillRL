"""
Complete VAE+HMM training pipeline for NetHack Learning Dataset.

Usage:
    python main.py <command> [args...]

Commands:
    collect_data                    - Collect and cache NetHack data
    group_data_by_id               - Group collected data by game ID for game-based training
    train <mode> [options...]      - Train VAE and/or HMM models
        Modes:
            vae_only               - Train only VAE
            hmm_only [options...]  - Train only HMM with optional E-step modes
            vae_hmm [options...]   - Train VAE+HMM with optional E-step modes
            vae_only_with_hmm <hmm_repo> [round] - Train only VAE with pre-trained HMM from HuggingFace
        
        E-step Options (can be combined with hmm_only and vae_hmm):
            game_grouped           - Use game-grouped data for E-step (one game at a time)
            batch_accumulation     - Add batch accumulation pass after regular E-step
        
        Standard mode: HMM E-step uses batched data [B, T, D] from multiple games
        Game-grouped mode: HMM E-step processes one complete game at a time [1, game_length, D]
        Batch accumulation: Additional pass that freezes HMM, accumulates statistics, then batch updates
        VAE-only with HMM: Skip E-step HMM training, load pre-trained HMM, only train VAE with HMM prior
    
    rl [ablation_mode] [options]   - Train online PPO agent with pretrained VAE+HMM models
                                    
        Ablation Modes:
            baseline               - VAE+HMM+PPO with curiosity (default)
            no_hmm                 - VAE+PPO (no HMM integration)
            rnd                    - VAE+HMM+PPO with RND instead of curiosity
            no_intrinsic           - VAE+HMM+PPO with no intrinsic rewards
            curiosity_dyn_only     - VAE+HMM+PPO with only dynamics surprise
            curiosity_skill_only   - VAE+HMM+PPO with only skill entropy
            curiosity_trans_only   - VAE+HMM+PPO with only transition novelty
            
        Options:
            --env ENV_NAME         - Environment (default: MiniHack-Room-5x5-v0)
            --steps N              - Total training steps (default: 1M)
            --seed N               - Random seed (default: 42)
            --wandb                - Enable W&B logging
            --no_upload            - Disable HuggingFace uploads
            --resume REPO_ID       - Resume training from unified HuggingFace repo
                                     (loads VAE, HMM, and PPO from same repo)
            --resume_local PATH    - Resume training from local PPO checkpoint file
                                     (loads VAE/HMM from separate repos)
                                    
        Model Loading Patterns:
        - Fresh Training: VAE from vae_repo_id, HMM from hmm_repo_id (separate repos)
        - Resume Training: VAE, HMM, PPO from same unified repo (--resume)
        - Resume Local: PPO from local file, VAE/HMM from separate repos (--resume_local)
                                    
        Uses configuration objects for fine-grained control over:
        - PPO hyperparameters (learning rate, rollout length, etc.)
        - Curiosity-driven exploration (dynamics surprise, skill entropy, transition novelty)
        - HMM online learning (update frequency, fitting window)
        - Training setup (environment, device, logging, checkpointing)
    
    vae_analysis <repo_name>       - Run VAE analysis and visualization
    bin_count_analysis [top_k]     - Analyze glyph character/color distributions  
    hmm_analysis <repo_name>       - Run HMM analysis and visualization
    plot_bin_count <data_path>     - Plot from saved bin count data

Examples:
    python main.py collect_data
    python main.py group_data_by_id
    python main.py train hmm_only game_grouped
    python main.py train hmm_only batch_accumulation
    python main.py train hmm_only game_grouped batch_accumulation
    python main.py train vae_hmm game_grouped
    python main.py train vae_only_with_hmm CatkinChen/nethack-hmm
    python main.py train vae_only_with_hmm CatkinChen/nethack-hmm 2
    
    # RL Ablation Studies
    python main.py rl                                    # Default: VAE+HMM+PPO with full curiosity
    python main.py rl baseline --wandb                   # Same as default with W&B logging
    python main.py rl no_hmm --steps 2000000             # VAE+PPO (no HMM)
    python main.py rl rnd --env MiniHack-Quest-Medium-v0  # VAE+HMM+PPO with RND
    python main.py rl no_intrinsic --seed 123            # VAE+HMM+PPO with no intrinsic rewards
    python main.py rl curiosity_dyn_only                 # VAE+HMM+PPO with dynamics curiosity only
    python main.py rl curiosity_skill_only               # VAE+HMM+PPO with skill entropy only
    python main.py rl curiosity_trans_only               # VAE+HMM+PPO with transition novelty only
    
    # Continue training from existing checkpoints
    python main.py rl baseline --resume CatkinChen/nethack-ppo-unified     # Resume from unified repo (VAE+HMM+PPO)
    python main.py rl baseline --resume_local ./checkpoints/ppo_policy.pth  # Resume from local PPO, load VAE/HMM separately
"""
import logging
import os
import sys
import torch
from datetime import datetime
from src.data_collection import NetHackDataCollector, BLStatsAdapter
from training.train import train_multimodalhack_vae, VAEConfig, load_datasets, train_vae_with_sticky_hmm_em
from utils.analysis import create_visualization_demo, analyze_glyph_char_color_pairs, plot_glyph_char_color_pairs_from_saved
from training.online_rl import train_online_ppo_with_pretrained_models
from rl.ppo import PPOConfig, CuriosityConfig, HMMOnlineConfig, VAEOnlineConfig, RNDConfig, TrainConfig

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
                bag_presence_thresh=0.5,
                hero_presence_thresh=0.2,
                passability_thresh=0.5,
                safety_thresh=0.5,
                map_temperature=1.0,
                map_deterministic=False,  # Use deterministic sampling for maps
                glyph_top_k=5,
                glyph_top_p=0.9,
                color_top_k=5,
                color_top_p=0.9,
                class_top_k=5,
                class_top_p=0.9,
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
                print(f"🧠 Latent analysis: {len(results['latent_analysis']['mu'])} total samples analyzed")
                
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
        
    elif len(sys.argv) > 1 and sys.argv[1] == "group_data_by_id":
        collector = NetHackDataCollector('ttyrecs.db')
        adapter = BLStatsAdapter()
        train_dataset = collector.collect_or_load_data(
            dataset_name=train_file,
            adapter=adapter,
            save_path=train_cache_file,
            max_batches=max_training_batches,
            batch_size=batch_size,
            seq_length=sequence_size,
            force_recollect=False
        )
        train_group_cache_file = os.path.join(data_cache_dir, f"{train_file}_b{batch_size}_s{sequence_size}_m{max_training_batches}_group.pt")
        grouped_data = collector.group_sequences_by_game(train_dataset, save_path=train_group_cache_file)
        
        print(f"✅ Data grouping completed!")

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
    
    elif len(sys.argv) > 1 and sys.argv[1] == "hmm_analysis":
        # HMM analysis mode: python main.py hmm_analysis <repo_name> [round_num] [revision_name]
        if len(sys.argv) < 3:
            print("❌ Usage: python main.py hmm_analysis <repo_name> [round_num] [revision_name]")
            print("   Example: python main.py hmm_analysis CatkinChen/nethack-hmm 1")
            sys.exit(1)
        
        repo_name = sys.argv[2]
        round_num = int(sys.argv[3]) if len(sys.argv) > 3 else None
        revision_name = sys.argv[4] if len(sys.argv) > 4 else None
        
        print(f"🧠 Running HMM Analysis")
        print(f"📦 Repository: {repo_name}")
        print(f"🔄 Round: {round_num if round_num is not None else 'latest'}")
        print(f"📋 Revision: {revision_name or 'main'}")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")
        
        # Load datasets for analysis
        print(f"📊 Loading datasets...")
        collector = NetHackDataCollector('ttyrecs.db')
        adapter = BLStatsAdapter()
        
        train_dataset = collector.collect_or_load_data(
            dataset_name=train_file,
            adapter=adapter,
            save_path=train_cache_file,
            max_batches=max_training_batches,
            batch_size=batch_size,
            seq_length=sequence_size,
            force_recollect=False
        )
        
        # Load HMM from HuggingFace
        print(f"🔄 Loading HMM from HuggingFace...")
        try:
            # Import here to avoid circular imports
            from training.training_utils import load_hmm_from_huggingface
            
            hmm, config, hmm_params, niw_prior, metadata = load_hmm_from_huggingface(
                repo_name=repo_name,
                round_num=round_num,
                revision_name=revision_name,
                device=str(device)
            )
            
            print(f"✅ HMM loaded successfully!")
            print(f"   📊 States: {hmm.p.K}")
            print(f"   📐 Latent dim: {hmm.p.D}")
            if metadata:
                print(f"   🏷️  Round: {metadata.get('round', 'unknown')}")
                print(f"   📅 Created: {metadata.get('training_timestamp', 'unknown')}")
            
            # Load VAE model for encoding data
            print(f"🎨 Loading VAE model for data encoding...")
            
            # Try to load VAE from a related repo or use a default
            vae_repo = repo_name.replace('-hmm', '-vae')  # e.g., nethack-hmm -> nethack-vae
            try:
                from training.train import load_model_from_huggingface
                model, _ = load_model_from_huggingface(vae_repo, device=str(device))
                print(f"✅ VAE loaded from {vae_repo}")
            except Exception as e:
                print(f"⚠️  Could not load VAE from {vae_repo}: {e}")
                print(f"🔄 Trying fallback VAE repo...")
                model, _ = load_model_from_huggingface("CatkinChen/nethack-vae", device=str(device))
                print(f"✅ VAE loaded from fallback repo")
            
            # Set up analysis directory
            analysis_dir = "hmm_analysis"
            os.makedirs(analysis_dir, exist_ok=True)
            
            # Set up logging
            import logging
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            # Import visualization function here to avoid circular imports
            from utils.analysis import visualize_hmm_after_estep
            
            # Run HMM visualization
            print(f"📊 Running HMM visualization analysis...")
            analysis_round = round_num if round_num is not None else metadata.get('round', 1)
            
            results = visualize_hmm_after_estep(
                model=model,
                dataset=train_dataset,
                device=device,
                hmm=hmm,
                save_dir=analysis_dir,
                round_idx=analysis_round,
                logger=logger,
                max_diags_batches=50,
                max_raster_sequences=10,
                random_seed=100,
                batch_multiples=100
            )
            
            print(f"✅ HMM analysis completed!")
            print(f"📁 Results saved to: {analysis_dir}/round_{analysis_round:02d}/")
            print(f"📊 Generated plots:")
            for key, path in results.items():
                if path and os.path.exists(path):
                    print(f"   📈 {key}: {path}")
            
        except Exception as e:
            print(f"❌ HMM analysis failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"💡 Make sure the repository exists and contains HMM checkpoints")
            print(f"💡 Check the round number and revision name")
    
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
        
        # Load datasets first
        print("📊 Loading datasets...")
        train_dataset, test_dataset = load_datasets(
            train_file=train_file,
            test_file=test_file,
            dbfilename='ttyrecs.db',
            batch_size=batch_size,
            sequence_size=sequence_size,
            training_batches=max_training_batches,
            testing_batches=max_testing_batches,
            max_training_batches=max_training_batches,
            max_testing_batches=max_testing_batches,
            data_cache_dir=data_cache_dir,
            force_recollect=False
        )
        
        # Set up logging with file output
        os.makedirs("logs", exist_ok=True)  # Create logs directory
        log_filename = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),  # Save to file
                logging.StreamHandler()  # Also show in console
            ]
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        print(f"📝 Logging to file: {log_filename}")

        # Check for game-grouped data training mode and batch accumulation
        use_game_grouped = "game_grouped" in sys.argv[3:] if len(sys.argv) > 3 else False
        use_batch_accumulation = "batch_accumulation" in sys.argv[3:] if len(sys.argv) > 3 else False
        
        if sys.argv[2] == "vae_only":
            vae_config = VAEConfig(
                initial_mi_beta=1.0,
                final_mi_beta=1.0,
                mi_beta_shape='constant',
                initial_tc_beta=0.0,
                final_tc_beta=10.0,
                tc_beta_shape='custom',
                initial_dw_beta=0.2,
                final_dw_beta=1.0,
                dw_beta_shape='custom',
                warmup_epoch_ratio=0.2,
                free_bits=0.75,
                encoder_dropout=0.1,
                decoder_dropout=0.1,
                prior_mode="standard",
            )
            # Train with pre-loaded datasets
            model, train_losses, test_losses = train_multimodalhack_vae(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                config=vae_config,
                epochs=15,          
                max_learning_rate=1e-3,
                save_path="models/nethack-vae.pth",
                device='cuda' if torch.cuda.is_available() else 'cpu',
                logger=logger,
                use_bf16=False,  # Enable BF16 mixed precision training
                shuffle_batches=True,  # Shuffle training batches each epoch for better training
                shuffle_within_batch=True,  # Shuffle within each batch for more variety
                
                custom_kl_beta_function = lambda init, end, progress: init + (end - init) * min(progress, 0.2) * 5.0, 
                
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
            
        else:
            if sys.argv[2] == "hmm_only":
                hmm_only = True
                vae_only_with_hmm = False
            elif sys.argv[2] == "vae_hmm":
                hmm_only = False
                vae_only_with_hmm = False
            elif sys.argv[2] == "vae_only_with_hmm":
                # Parse HMM repo and optional round number
                if len(sys.argv) < 4:
                    print("❌ vae_only_with_hmm requires HMM repository argument.")
                    print("   Usage: python main.py train vae_only_with_hmm <hmm_repo> [round_number]")
                    print("   Example: python main.py train vae_only_with_hmm CatkinChen/nethack-hmm")
                    print("   Example: python main.py train vae_only_with_hmm CatkinChen/nethack-hmm 2")
                    sys.exit(1)
                hmm_only = False
                vae_only_with_hmm = True
                pretrained_hmm_repo = sys.argv[3]
                pretrained_hmm_round = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else None
            else:
                print("❌ Invalid argument. Use 'vae_only', 'hmm_only', 'vae_hmm', or 'vae_only_with_hmm'.")
                print("   💡 To use game-grouped data for E-step, add 'game_grouped' as an argument:")
                print("      python main.py train hmm_only game_grouped")
                print("      python main.py train vae_hmm game_grouped")
                print("   💡 To use batch accumulation, add 'batch_accumulation' as an argument:")
                print("      python main.py train hmm_only batch_accumulation")
                print("   💡 Options can be combined:")
                print("      python main.py train hmm_only game_grouped batch_accumulation")
                print("   💡 To train VAE with pre-trained HMM:")
                print("      python main.py train vae_only_with_hmm CatkinChen/nethack-hmm")
                print("      python main.py train vae_only_with_hmm CatkinChen/nethack-hmm 2")
                sys.exit(1)
            vae_config = VAEConfig(
                initial_mi_beta=1.0,
                final_mi_beta=1.0,
                mi_beta_shape='constant',
                initial_tc_beta=0.0,
                final_tc_beta=10.0,
                tc_beta_shape='custom',
                initial_dw_beta=0.2,
                final_dw_beta=1.0,
                dw_beta_shape='custom',
                warmup_epoch_ratio=0.2,
                free_bits=0.75,
                encoder_dropout=0.1,
                decoder_dropout=0.1,
                prior_mode="blend",
                initial_prior_blend_alpha=0.1,
                final_prior_blend_alpha=0.6,
                prior_blend_shape='cosine'
            )
            
            # HMM Training Schedule:
            # - EM Rounds 1-3: Only 1 HMM iteration per round (quick updates)
            # - EM Round 4 (last): Up to 10 HMM iterations with pi optimization every iteration
            # This allows fast initial learning followed by fine-grained optimization
            
            model, hmm, training_info = train_vae_with_sticky_hmm_em(
                # Load from HuggingFace
                pretrained_hf_repo="CatkinChen/nethack-vae",
                # Datasets
                train_dataset=train_dataset, 
                test_dataset=test_dataset,
                
                config=vae_config,  
                batch_multiples=100,
                init_niw_mu_with_kmean=True,
                # HMM parameters
                alpha=5.0,
                kappa=1.0,
                gamma=5.0,
                hmm_only=hmm_only,
                vae_only_with_hmm=vae_only_with_hmm if 'vae_only_with_hmm' in locals() else False,
                pretrained_hmm_hf_repo=pretrained_hmm_repo if 'pretrained_hmm_repo' in locals() else None,
                pretrained_hmm_round=pretrained_hmm_round if 'pretrained_hmm_round' in locals() else None,
                em_rounds=1 if hmm_only else 4,
                m_epochs_per_round=1,
                set_niw_mu0_with_global_mean=False,
                set_niw_Psi0_with_global_cov=False,
                niw_mu0 = 0.0,
                niw_kappa0 = 1.0, 
                niw_Psi0 = 30.0,
                niw_nu0 = vae_config.latent_dim + 10,
                streaming_rho_niw = 1.0,
                streaming_rho_trans = 1.0,
                max_iters = 10,  # Used only in last round; first 3 rounds use 1 iteration
                elbo_drop_tol = 0.01,  # 1% relative tolerance
                elbo_tol = 0.01,       # 1% relative tolerance
                optimize_pi_every_n_steps = 1,  # Optimize pi in every iteration of the last round
                pi_iters = 200,
                pi_lr = 1.0e-3,
                pi_early_stopping_patience = 5,  # Early stop if no improvement for 5 steps
                pi_early_stopping_min_delta = 1e-3,  # Minimum improvement threshold
                reset_to_prior = True,
                reset_streaming = True,
                reset_low_count_states = False,
                low_count_thresh = 0.01,  # States with <1% of total counts will be reset
                # Game-grouped data options
                use_game_grouped_data=use_game_grouped,
                game_grouped_data_path=os.path.join(data_cache_dir, f"{train_file}_b{batch_size}_s{sequence_size}_m{max_training_batches}_group.pt") if use_game_grouped else None,
                max_games_per_estep=None,  # Process all games

                # Batch accumulation options
                use_batch_accumulation=use_batch_accumulation,
                accumulation_max_batches=None,  # Process all batches

                # HuggingFace integration
                push_to_hub=True,
                hub_repo_id_hmm="CatkinChen/nethack-hmm",
                hub_repo_id_vae_hmm="CatkinChen/nethack-vae-hmm",
                
                # Training parameters
                use_bf16=False,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                logger=logger,
                custom_kl_beta_function = lambda init, end, progress: init + (end - init) * min(progress, 0.2) * 5.0, 
                # Additional training arguments passed to M-step
                max_learning_rate=1e-4,  # Lower learning rate for fine-tuning
                lr_scheduler="constant"   # Use constant LR scheduler for stability
            )
            
            if use_game_grouped:
                print(f"🎮 Using game-grouped data for E-step HMM training")
                print(f"   📂 Game data will be saved/loaded from: {os.path.join(data_cache_dir, f'{train_file}_b{batch_size}_s{sequence_size}_m{max_training_batches}_group.pt')}")
                print(f"   🎯 HMM will be trained on individual games instead of batched data")
            else:
                print(f"📦 Using standard batched data for E-step HMM training")
            
            if use_batch_accumulation:
                print(f"🔄 Using batch accumulation for clean batch updates")
                print(f"   ❄️  HMM parameters will be frozen during accumulation pass")
                print(f"   📊 Statistics will be accumulated then batch updated")
                print(f"   🎯 π will be optimized once from aggregated r1")
            
            print(f"✅ Training completed!")
            print(f"📊 HMM checkpoints: {len(training_info['hmm_paths'])}")
            print(f"📊 VAE+HMM checkpoints: {len(training_info['vae_hmm_paths'])}")
            
    elif len(sys.argv) > 1 and sys.argv[1] == "rl":
        print(f"🎮 Reinforcement Learning mode activated")
        
        # Parse ablation mode and options
        ablation_mode = "baseline"  # default
        env_name = "MiniHack-Room-Random-15x15-v0"  # default
        total_steps = 1_000_000  # default 1M steps
        seed = 42  # default
        use_wandb_flag = False
        disable_upload = False
        resume_repo_id = None  # HuggingFace repo for resuming
        resume_local_path = None  # Local checkpoint path for resuming
        
        # Parse command line arguments
        i = 2
        if len(sys.argv) > i and not sys.argv[i].startswith('--'):
            ablation_mode = sys.argv[i]
            i += 1
        
        while i < len(sys.argv):
            if sys.argv[i] == '--env' and i + 1 < len(sys.argv):
                env_name = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--steps' and i + 1 < len(sys.argv):
                total_steps = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--seed' and i + 1 < len(sys.argv):
                seed = int(sys.argv[i + 1])
                i += 2
            elif sys.argv[i] == '--wandb':
                use_wandb_flag = True
                i += 1
            elif sys.argv[i] == '--no_upload':
                disable_upload = True
                i += 1
            elif sys.argv[i] == '--resume' and i + 1 < len(sys.argv):
                resume_repo_id = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--resume_local' and i + 1 < len(sys.argv):
                resume_local_path = sys.argv[i + 1]
                i += 2
            else:
                print(f"⚠️  Unknown option: {sys.argv[i]}")
                i += 1
        
        # Validate ablation mode
        valid_modes = [
            "baseline", "no_hmm", "rnd", "no_intrinsic",
            "curiosity_dyn_only", "curiosity_skill_only", "curiosity_trans_only"
        ]
        if ablation_mode not in valid_modes:
            print(f"❌ Invalid ablation mode: {ablation_mode}")
            print(f"   Valid modes: {', '.join(valid_modes)}")
            sys.exit(1)
        
        print(f"🔬 Ablation Study Mode: {ablation_mode}")
        print(f"🎮 Environment: {env_name}")
        print(f"📊 Total Steps: {total_steps:,}")
        print(f"🌱 Seed: {seed}")
        print(f"📊 W&B Logging: {use_wandb_flag}")
        print(f"☁️  HF Upload: {not disable_upload}")
        
        # Set up logging with file output
        os.makedirs("logs", exist_ok=True)  # Create logs directory
        log_filename = f"logs/rl_{ablation_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),  # Save to file
                logging.StreamHandler()  # Also show in console
            ]
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        print(f"📝 Logging to file: {log_filename}")
        
        # Calculate PPO updates from total steps
        num_envs = 8
        rollout_len = 128
        ppo_updates = total_steps // (num_envs * rollout_len)
        
        # Base PPO Configuration (same for all ablations)
        ppo_config = PPOConfig(
            num_envs=num_envs,
            rollout_len=rollout_len,
            total_updates=ppo_updates,
            minibatch_size=512,
            epochs_per_update=4,
            gamma=0.999,
            gae_lambda=0.95,
            clip_coef=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            learning_rate=3e-4,
            vf_learning_rate=None,  # Use same as learning_rate
            policy_uses_skill=True,  # Will be overridden for no_hmm mode
            deterministic_eval=True
        )
        
        # Configure curiosity and RND based on ablation mode
        if ablation_mode == "baseline":
            curiosity_config = CuriosityConfig(
                use_dyn_kl=True,           # Dynamics surprise
                use_skill_entropy=True,    # Skill entropy with boundary gating
                use_skill_transition_novelty=True,  # Skill transition novelty
                use_rnd=False,
                # Standard annealing
                eta0_dyn=1.0, tau_dyn=3e6,
                eta0_hdp=1.0, tau_hdp=3e6,
                eta0_stn=1.0, tau_stn=3e6,
                # Skill boundary gating
                use_skill_boundary_gate=True,
                gate_delta_eps=1e-3,
                ema_beta=0.99, eps=1e-8
            )
            description = "VAE+HMM+PPO with full curiosity (dynamics + skill entropy + transition novelty)"
            
        elif ablation_mode == "no_hmm":
            ppo_config.policy_uses_skill = False  # No skill features for policy
            curiosity_config = CuriosityConfig(
                use_dyn_kl=True,           # Only dynamics surprise (no skill-based rewards)
                use_skill_entropy=False,   # No skill entropy without HMM
                use_skill_transition_novelty=False,  # No skill transitions without HMM
                use_rnd=False,
                eta0_dyn=1.0, tau_dyn=3e6,
                ema_beta=0.99, eps=1e-8
            )
            description = "VAE+PPO (no HMM) with dynamics curiosity only"
            
        elif ablation_mode == "rnd":
            curiosity_config = CuriosityConfig(
                use_dyn_kl=False,          # Disable curiosity components
                use_skill_entropy=False,
                use_skill_transition_novelty=False,
                use_rnd=True,              # Enable RND
                eta0_rnd=0.25, tau_rnd=3e6,
                ema_beta=0.99, eps=1e-8
            )
            description = "VAE+HMM+PPO with RND intrinsic motivation"
            
        elif ablation_mode == "no_intrinsic":
            curiosity_config = CuriosityConfig(
                use_dyn_kl=False,          # Disable all intrinsic rewards
                use_skill_entropy=False,
                use_skill_transition_novelty=False,
                use_rnd=False,
                ema_beta=0.99, eps=1e-8
            )
            description = "VAE+HMM+PPO with no intrinsic rewards (extrinsic only)"
            
        elif ablation_mode == "curiosity_dyn_only":
            curiosity_config = CuriosityConfig(
                use_dyn_kl=True,           # Only dynamics surprise
                use_skill_entropy=False,
                use_skill_transition_novelty=False,
                use_rnd=False,
                eta0_dyn=1.0, tau_dyn=3e6,
                ema_beta=0.99, eps=1e-8
            )
            description = "VAE+HMM+PPO with dynamics curiosity only"
            
        elif ablation_mode == "curiosity_skill_only":
            curiosity_config = CuriosityConfig(
                use_dyn_kl=False,
                use_skill_entropy=True,    # Only skill entropy
                use_skill_transition_novelty=False,
                use_rnd=False,
                eta0_hdp=1.0, tau_hdp=3e6,
                use_skill_boundary_gate=True,
                gate_delta_eps=1e-3,
                ema_beta=0.99, eps=1e-8
            )
            description = "VAE+HMM+PPO with skill entropy curiosity only"
            
        elif ablation_mode == "curiosity_trans_only":
            curiosity_config = CuriosityConfig(
                use_dyn_kl=False,
                use_skill_entropy=False,
                use_skill_transition_novelty=True,  # Only skill transition novelty
                use_rnd=False,
                eta0_stn=1.0, tau_stn=3e6,
                ema_beta=0.99, eps=1e-8
            )
            description = "VAE+HMM+PPO with skill transition novelty only"
        
        # HMM Online Learning Configuration (disabled for no_hmm mode)
        if ablation_mode == "no_hmm":
            # Use dummy HMM config (won't be used)
            hmm_config = HMMOnlineConfig(
                hmm_update_every=float('inf'),  # Never update
                hmm_fit_window=1,
                hmm_max_iters=1,
                hmm_tol=1e-2,
                hmm_elbo_drop_tol=1e-2,
                rho_emission=0.0,
                rho_transition=None,
                optimise_pi=False,
                pi_steps=200,
                pi_lr=0.05,
                pi_early_stopping_patience=10,
                pi_early_stopping_min_delta=1e-5
            )
            
            if resume_repo_id is None and resume_local_path is None:
                # Set VAE repo to VAE-only model
                vae_repo_id = "CatkinChen/nethack-vae"
            else:
                vae_repo_id = None  # Resume from existing PPO checkpoint which already has VAE
            hmm_repo_id = None  # Will use dummy HMM
        else:
            hmm_config = HMMOnlineConfig(
                hmm_update_every=5_120,   # Update HMM every ~5 rollouts
                hmm_update_growth=1.30,    # Growth factor for update interval
                hmm_update_every_cap=60_000, # Cap for update interval
                hmm_fit_window=400_000,    # Use 400k steps for HMM fitting
                hmm_max_iters=5,          # Up to 5 iterations per update
                hmm_tol=1e-2,
                hmm_elbo_drop_tol=1e-2,
                rho_emission=0.05,        # Streaming blend rate
                rho_transition=None,       # Use same as emission
                optimise_pi=True,
                pi_steps=10,               # π optimization steps
                pi_lr=5e-4,                # π optimization learning rate
                pi_early_stopping_patience=1,    # Early stopping patience
                pi_early_stopping_min_delta=1e-2 # Early stopping min delta
            )
            if resume_repo_id is None and resume_local_path is None:
                # Set VAE and HMM repos to pre-trained models
                vae_repo_id = "CatkinChen/nethack-vae-hmm"
                hmm_repo_id = "CatkinChen/nethack-hmm"
            else:
                vae_repo_id = None  # Resume from existing PPO checkpoint which already has VAE+HMM
                hmm_repo_id = None  # Resume from existing PPO checkpoint which already has VAE+HMM
        
        # VAE Online Configuration - synchronized with HMM updates
        vae_config = VAEOnlineConfig(
            vae_update_every=5_120,       # Match HMM update frequency  
            vae_update_growth=1.30,       # Same growth pattern as HMM
            vae_update_every_cap=60_000,  # Same cap as HMM
            vae_lr=1e-4                   # Learning rate for VAE updates
        )
        
        # RND Configuration (used only for RND ablation)
        rnd_config = RNDConfig(
            proj_dim=128,
            hidden=256,
            lr=1e-3,
            update_per_rollout=2
        )
        
        # Training Configuration
        run_name = f"ablation_{ablation_mode}_{env_name.replace('-', '_')}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        train_config = TrainConfig(
            env_id=env_name,
            seed=seed,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            log_dir=f"./runs/{run_name}",
            save_every=25_000,
            eval_every=25_000,
            eval_episodes=10
        )
        
        print(f"\n🔧 Ablation Configuration:")
        print(f"   Description: {description}")
        print(f"   Environment: {train_config.env_id}")
        print(f"   Device: {train_config.device}")
        print(f"   Total Steps: {total_steps:,}")
        print(f"   PPO Updates: {ppo_config.total_updates:,}")
        print(f"   Policy Uses Skills: {ppo_config.policy_uses_skill}")
        print(f"   VAE Repository: {vae_repo_id}")
        print(f"   HMM Repository: {hmm_repo_id if hmm_repo_id else 'None (no HMM)'}")
        print(f"\n🧠 Intrinsic Rewards:")
        print(f"   Dynamics KL: {curiosity_config.use_dyn_kl}")
        print(f"   Skill Entropy: {curiosity_config.use_skill_entropy}")
        print(f"   Skill Transition: {curiosity_config.use_skill_transition_novelty}")
        print(f"   RND: {curiosity_config.use_rnd}")
        if ablation_mode != "no_hmm":
            print(f"\n🔄 HMM Updates:")
            print(f"   Every: {hmm_config.hmm_update_every:,} steps")
            print(f"   Growth: {hmm_config.hmm_update_growth}")
            print(f"   Cap: {hmm_config.hmm_update_every_cap:,}")
        print(f"\n🎨 VAE Updates:")
        print(f"   Every: {vae_config.vae_update_every:,} steps")
        print(f"   Learning Rate: {vae_config.vae_lr}")
        
        # Train with the configured ablation
        try:
            results = train_online_ppo_with_pretrained_models(
                env_name=train_config.env_id,
                vae_repo_id=vae_repo_id,
                hmm_repo_id=hmm_repo_id,
                
                # Resume training from existing PPO checkpoint (new!)
                ppo_repo_id=resume_repo_id,
                ppo_checkpoint_path=resume_local_path,
                resume_training=resume_repo_id is not None or resume_local_path is not None,
                
                # Use config objects for full control
                ppo_config=ppo_config,
                curiosity_config=curiosity_config,
                hmm_config=hmm_config,
                vae_config=vae_config,
                rnd_config=rnd_config,
                train_config=train_config,
                
                # Monitoring and uploading
                use_wandb=use_wandb_flag,
                wandb_project="SequentialSkillRL-Ablations",
                wandb_run_name=run_name,
                wandb_tags=["ablation", ablation_mode, env_name.split('-')[1] if '-' in env_name else env_name],
                wandb_notes=f"Ablation study: {description}",
                
                push_to_hub=not disable_upload,
                hub_repo_id=f"CatkinChen/nethack-ppo-ablation-{ablation_mode}",
                hf_upload_artifacts=True,
                logger=logger,
            )
            
            print(f"\n🎉 Ablation study '{ablation_mode}' completed successfully!")
            print(f"📊 Run name: {run_name}")
            print(f"📁 Results: {results.get('final_checkpoint', 'N/A')}")
            
        except Exception as e:
            print(f"\n❌ Ablation study '{ablation_mode}' failed: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Ablation study failed", exc_info=True)
            sys.exit(1)