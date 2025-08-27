import os
import sys
import torch
from datetime import datetime
from src.data_collection import NetHackDataCollector, BLStatsAdapter
from training.train import train_multimodalhack_vae, VAEConfig, load_datasets
from utils.analysis import create_visualization_demo, analyze_glyph_char_color_pairs, plot_glyph_char_color_pairs_from_saved

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
            decoder_dropout=0.1
        )

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
        
        if sys.argv[2] == "vae_only":
            # Train with pre-loaded datasets
            model, train_losses, test_losses = train_multimodalhack_vae(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                config=vae_config,
                epochs=15,          
                max_learning_rate=1e-3,
                save_path="models/nethack-vae.pth",
                device='cuda' if torch.cuda.is_available() else 'cpu',
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