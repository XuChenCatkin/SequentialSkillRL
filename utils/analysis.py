import os
import numpy as np
import torch
from nle.nethack import tty_render
import matplotlib.pyplot as plt
import math
import datetime as dt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

# xterm-ish 16-color palette (0–7 normal, 8–15 bright)
ANSI_16_RGB = [
    (0,0,0), (205,0,0), (0,205,0), (205,205,0),
    (0,0,205), (205,0,205), (0,205,205), (229,229,229),
    (127,127,127), (255,0,0), (0,255,0), (255,255,0),
    (92,92,255), (255,0,255), (0,255,255), (255,255,255),
]

def _to_np(x):
    """Convert torch tensor or array-like to numpy array (CPU)."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def _color16(c):
    return ANSI_16_RGB[int(c) & 0xF]

def _render_map_image(chars, colors, font_path=None, font_size=16, bg=(0,0,0)):
    """
    chars, colors: [H,W] uint8 arrays (or torch tensors).
    Returns a PIL.Image.
    """
    chars = _to_np(chars)
    colors = _to_np(colors)
    H, W = chars.shape

    # Robust monospace choice
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()

    # Cell size from a representative glyph
    bbox = font.getbbox("M")  # (l, t, r, b)
    cell_w, cell_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if cell_w <= 0 or cell_h <= 0:  # rare fallback
        cell_w, cell_h = font_size, int(font_size * 1.8)

    img = Image.new("RGB", (W * cell_w, H * cell_h), bg)
    draw = ImageDraw.Draw(img)

    for i in range(H):
        for j in range(W):
            # Guard for non-printable ASCII -> space
            v = int(chars[i, j])
            ch = chr(v) if 32 <= v < 127 else " "
            draw.text((j * cell_w, i * cell_h), ch, font=font, fill=_color16(colors[i, j]))
    return img

def save_maps_and_markdown(
    originals,
    reconstructions,
    out_dir="vae_analysis",
    md_filename="recon_comparison.md",
    font_path=None,
    font_size=18,
    bg=(0,0,0),
    title="VAE Reconstruction Comparison",
):
    """
    originals, reconstructions: iterables of (chars, colors), each [H,W], uint8 or torch tensors.
    Creates PNGs in out_dir/images and a Markdown file with side-by-side comparisons.
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Write markdown
    md_path = out_dir / md_filename
    with md_path.open("w", encoding="utf-8") as md:
        md.write(f"# {title}\n\n")
        md.write(f"_Generated: {dt.datetime.now().isoformat(timespec='seconds')}_\n\n")

        for i, ((char_orig, color_orig), (char_recon, color_recon)) in enumerate(zip(originals, reconstructions)):
            # Render & save images
            img_o = _render_map_image(char_orig, color_orig, font_path=font_path, font_size=font_size, bg=bg)
            img_r = _render_map_image(char_recon, color_recon, font_path=font_path, font_size=font_size, bg=bg)

            fn_o = f"sample_{i:03d}_orig.png"
            fn_r = f"sample_{i:03d}_recon.png"
            img_o.save(img_dir / fn_o)
            img_r.save(img_dir / fn_r)

            # Markdown section with a 2-col table
            md.write(f"## Sample {i+1}\n\n")
            md.write("| Original | Reconstruction |\n")
            md.write("|---|---|\n")
            md.write(f"| ![orig {i}](images/{fn_o}) | ![recon {i}](images/{fn_r}) |\n\n")
            
            # Calculate and display accuracy for this sample
            char_matches = (char_orig == char_recon).float()
            color_matches = (color_orig == color_recon).float()
            char_accuracy = char_matches.mean().item()
            color_accuracy = color_matches.mean().item()

            md.write(f"\n Sample {i+1} Accuracy:")
            md.write(f"   Character accuracy: {char_accuracy:.3f} ({char_matches.sum().int()}/{char_matches.numel()} cells)\n")
            md.write(f"   Color accuracy: {color_accuracy:.3f} ({color_matches.sum().int()}/{color_matches.numel()} cells)\n")

            if i < len(reconstructions) - 1:
                md.write("\n" + "=" * 80 + "\n")
        
        md.write(f"\n📈 Overall Reconstruction Statistics:")
        char_accuracy = sum((orig[0] == recon[0]).float().mean().item() for orig, recon in zip(originals, reconstructions)) / len(originals)
        color_accuracy = sum((orig[1] == recon[1]).float().mean().item() for orig, recon in zip(originals, reconstructions)) / len(originals)

        md.write(f"   Average Character Reconstruction Accuracy: {char_accuracy:.3f}\n")
        md.write(f"   Average Color Reconstruction Accuracy: {color_accuracy:.3f}\n")

    print(f"Wrote Markdown: {md_path}")

def visualize_reconstructions(
    model, dataset, device, 
    num_samples=4, 
    out_dir="vae_analysis", save_path="recon_comparison.md",
    random_sampling=True, dataset_name="Dataset",
    # VAE sampling parameters
    use_mean=True,
    include_logits=False,
    # map sampling parameters
    map_temperature=1.0,
    map_occ_thresh=0.5,
    map_deterministic=True,
    glyph_top_k=0,
    glyph_top_p=1.0,
    color_top_k=0,
    color_top_p=1.0,
    # message sampling parameters
    msg_temperature=1.0,
    msg_top_k=0,
    msg_top_p=1.0,
    msg_deterministic=True,
    allow_eos=True,
    forbid_eos_at_start=True,
    allow_pad=False
):
    """
    Enhanced version of visualize_reconstructions with random sampling support
    
    Args:
        model: Trained MultiModalHackVAE model
        dataset: List of batches (from NetHackDataCollector)
        device: Device to run inference on
        num_samples: Number of samples to visualize
        random_sampling: Whether to randomly sample or use sequential sampling
        dataset_name: Name of the dataset for labeling
        
        # VAE sampling parameters
        use_mean: If True, use mean of latent distribution; if False, sample from it
        include_logits: Whether to include raw logits in output
        
        # Map sampling parameters
        map_temperature: Temperature for map sampling (higher = more random)
        map_occ_thresh: Threshold for occupancy prediction
        map_deterministic: If True, use deterministic (argmax) sampling for map
        glyph_top_k: Top-k filtering for glyph character sampling (0 = disabled)
        glyph_top_p: Top-p (nucleus) filtering for glyph character sampling
        color_top_k: Top-k filtering for color sampling (0 = disabled)
        color_top_p: Top-p (nucleus) filtering for color sampling
        
        # Message sampling parameters
        msg_temperature: Temperature for message token sampling
        msg_top_k: Top-k filtering for message sampling (0 = disabled)
        msg_top_p: Top-p (nucleus) filtering for message sampling
        msg_deterministic: If True, use deterministic (argmax) sampling for messages
        allow_eos: Whether to allow end-of-sequence tokens in messages
        forbid_eos_at_start: Whether to forbid EOS tokens at the start of messages
        allow_pad: Whether to allow padding tokens in messages
        
    Examples:
        # Basic usage (deterministic reconstruction)
        results = visualize_reconstructions(model, dataset, device)
        
        # More creative/random reconstruction
        results = visualize_reconstructions(
            model, dataset, device,
            use_mean=False,  # Sample from latent distribution
            map_temperature=1.5,  # Higher temperature for more variation
            map_deterministic=False,  # Non-deterministic map sampling
            glyph_top_k=10,  # Top-10 character sampling
            color_top_k=5,   # Top-5 color sampling
            msg_temperature=1.2,  # Slightly random message generation
            msg_deterministic=False
        )
        
        # High-quality reconstruction (conservative)
        results = visualize_reconstructions(
            model, dataset, device,
            use_mean=True,  # Use mean latent representation
            map_deterministic=True,  # Deterministic map sampling
            glyph_top_k=3,  # Conservative character sampling
            color_top_k=2,  # Conservative color sampling
            msg_deterministic=True  # Deterministic message generation
        )
    """
    
    model.eval()
    
    reconstructions = []
    originals = []
    output_lines = []
    output_lines.append(f"NetHack VAE Reconstructions - {dataset_name} Dataset")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    if random_sampling:
        output_lines.append(f"🎲 Using random sampling from {len(dataset)} batches")
    else:
        output_lines.append(f"📄 Using sequential sampling from {len(dataset)} batches")
    output_lines.append("")
    
    # Collect all valid samples from the dataset
    all_samples = []
    
    print(f"🔍 Collecting valid samples from {dataset_name.lower()} dataset...")
    for batch_idx, batch in enumerate(dataset):
        # Move batch to device and reshape like in training
        batch_device = {}
        for key, value in batch.items():
            if value is not None and isinstance(value, torch.Tensor):
                value_device = value.to(device)
                # Reshape tensors from [B, T, ...] to [B*T, ...]
                B, T = value_device.shape[:2]
                remaining_dims = value_device.shape[2:]
                batch_device[key] = value_device.view(B * T, *remaining_dims)
            else:
                batch_device[key] = value
        
        if 'valid_screen' in batch_device:
            valid_mask = batch_device['valid_screen'].cpu()  # [B*T]
            valid_indices = torch.where(valid_mask)[0]
        else:
            # If no valid_screen mask, assume all are valid
            valid_indices = torch.arange(batch_device['game_chars'].shape[0])
        
        # Store valid samples with their batch information
        for idx in valid_indices:
            sample_data = {}
            for key in ['game_chars', 'game_colors', 'blstats', 'message_chars', 'hero_info']:
                if key in batch_device and batch_device[key] is not None:
                    sample_data[key] = batch_device[key][idx]
            
            sample_data['batch_idx'] = batch_idx
            sample_data['sample_idx'] = idx.item()
            all_samples.append(sample_data)
    
    print(f"📊 Found {len(all_samples)} valid samples in {dataset_name.lower()} dataset")
    
    if len(all_samples) == 0:
        print(f"⚠️ No valid samples found in {dataset_name.lower()} dataset!")
        return {'num_samples': 0, 'originals': [], 'reconstructions': [], 'save_path': save_path}
    
    # Sample the requested number of samples
    if random_sampling:
        if len(all_samples) >= num_samples:
            selected_samples = random.sample(all_samples, num_samples)
        else:
            selected_samples = all_samples
            print(f"⚠️ Requested {num_samples} samples but only {len(all_samples)} available")
    else:
        selected_samples = all_samples[:num_samples]
    
    print(f"🎯 Processing {len(selected_samples)} samples for reconstruction...")
    
    with torch.no_grad():
        for i, sample in enumerate(selected_samples):
            print(f"  Processing sample {i+1}/{len(selected_samples)} (batch {sample['batch_idx']}, sample {sample['sample_idx']})")
            
            # Prepare input tensors (add batch dimension)
            model_inputs = {}
            for key in ['game_chars', 'game_colors', 'blstats', 'message_chars', 'hero_info']:
                if key in sample and sample[key] is not None:
                    model_inputs[key.replace('message_chars', 'msg_tokens')] = sample[key].unsqueeze(0)
            
            # Get model output with all sampling parameters
            model_output = model.sample(
                glyph_chars=model_inputs.get('game_chars'),
                glyph_colors=model_inputs.get('game_colors'),
                blstats=model_inputs.get('blstats'),
                msg_tokens=model_inputs.get('msg_tokens'),
                hero_info=model_inputs.get('hero_info'),
                use_mean=use_mean,
                include_logits=include_logits,
                # Map sampling parameters
                map_temperature=map_temperature,
                map_occ_thresh=map_occ_thresh,
                map_deterministic=map_deterministic,
                glyph_top_k=glyph_top_k,
                glyph_top_p=glyph_top_p,
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
            
            # Get reconstructed characters and colors
            char_recon = model_output['chars'][0].cpu()  # [H, W]
            color_recon = model_output['colors'][0].cpu()  # [H, W]
            
            # Get original data
            char_orig = sample['game_chars'].cpu()  # [H, W]
            color_orig = sample['game_colors'].cpu()  # [H, W]
            
            reconstructions.append((char_recon, color_recon))
            originals.append((char_orig, color_orig))
    
    save_maps_and_markdown(originals, reconstructions,
                            out_dir=out_dir,
                            md_filename=save_path,
                            font_path="DejaVuSansMono.ttf",  # optional
                            font_size=18)
    
    print(f"✅ {dataset_name} reconstruction visualization saved to {out_dir}/{save_path}")

    return {
        'num_samples': len(reconstructions),
        'originals': originals,
        'reconstructions': reconstructions,
        'save_path': save_path,
        'dataset_name': dataset_name,
        'random_sampling': random_sampling,
        'sampling_params': {
            'use_mean': use_mean,
            'include_logits': include_logits,
            'map_temperature': map_temperature,
            'map_occ_thresh': map_occ_thresh,
            'map_deterministic': map_deterministic,
            'glyph_top_k': glyph_top_k,
            'glyph_top_p': glyph_top_p,
            'color_top_k': color_top_k,
            'color_top_p': color_top_p,
            'msg_temperature': msg_temperature,
            'msg_top_k': msg_top_k,
            'msg_top_p': msg_top_p,
            'msg_deterministic': msg_deterministic,
            'allow_eos': allow_eos,
            'forbid_eos_at_start': forbid_eos_at_start,
            'allow_pad': allow_pad
        }
    }


def analyze_latent_space(
    model, dataset, device, 
    save_path="vae_analysis/latent_analysis.png", 
    max_samples=100, 
    dataset_labels=None
):
    """
    Enhanced version of analyze_latent_space with support for multiple datasets
    Automatically balances samples between training and testing datasets
    
    Args:
        model: Trained MultiModalHackVAE model
        dataset: List of batches from multiple datasets
        device: Device to run inference on
        save_path: Path to save the analysis plots
        max_samples: Maximum number of samples to analyze
        dataset_labels: List of labels for each batch indicating which dataset it comes from
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if dataset_labels is None:
        dataset_labels = ['unknown'] * len(dataset)
    
    # Separate datasets by type
    train_batches = []
    test_batches = []
    train_labels = []
    test_labels = []
    
    for batch, label in zip(dataset, dataset_labels):
        if label == 'train':
            train_batches.append(batch)
            train_labels.append(label)
        elif label == 'test':
            test_batches.append(batch)
            test_labels.append(label)
        else:
            # Unknown labels go to train by default
            train_batches.append(batch)
            train_labels.append('train')
    
    print(f"📊 Dataset composition: {len(train_batches)} train batches, {len(test_batches)} test batches")
    
    # Determine sample allocation
    if len(train_batches) > 0 and len(test_batches) > 0:
        # Both datasets available - split samples evenly
        train_samples_target = max_samples // 2
        test_samples_target = max_samples - train_samples_target
        print(f"🎯 Target samples: {train_samples_target} train, {test_samples_target} test")
    elif len(train_batches) > 0:
        # Only training data available
        train_samples_target = max_samples
        test_samples_target = 0
        print(f"🎯 Only training data available, using {train_samples_target} samples")
    elif len(test_batches) > 0:
        # Only testing data available  
        train_samples_target = 0
        test_samples_target = max_samples
        print(f"🎯 Only testing data available, using {test_samples_target} samples")
    else:
        raise ValueError("No valid datasets provided")
    
    # Shuffle datasets for random sampling
    if len(train_batches) > 0:
        train_indices = list(range(len(train_batches)))
        random.shuffle(train_indices)
        train_batches = [train_batches[i] for i in train_indices]
        train_labels = [train_labels[i] for i in train_indices]
    
    if len(test_batches) > 0:
        test_indices = list(range(len(test_batches)))
        random.shuffle(test_indices)
        test_batches = [test_batches[i] for i in test_indices]
        test_labels = [test_labels[i] for i in test_indices]
    
    latent_vectors = []
    batch_indices = []
    sample_info = []
    
    def extract_samples_from_batches(batches, labels, target_samples, dataset_type):
        """Extract samples from a list of batches"""
        local_latent_vectors = []
        local_sample_info = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (batch, label) in enumerate(zip(batches, labels)):
                if sample_count >= target_samples:
                    break
                    
                # Move batch to device and reshape like in training
                batch_device = {}
                for key, value in batch.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        value_device = value.to(device)
                        # Reshape tensors from [B, T, ...] to [B*T, ...]
                        B, T = value_device.shape[:2]
                        remaining_dims = value_device.shape[2:]
                        batch_device[key] = value_device.view(B * T, *remaining_dims)
                    else:
                        batch_device[key] = value
                        
                if 'valid_screen' in batch_device:
                    valid_screen = batch_device['valid_screen'].cpu()  # [B*T]
                else:
                    valid_screen = torch.ones(batch_device['game_chars'].shape[0], dtype=torch.bool)
                
                # Get model output (includes mu, logvar, lowrank_factors)
                samples_to_process = min(max(0, target_samples - sample_count), valid_screen.sum().item())
                if samples_to_process == 0:
                    continue
                    
                valid_indices = torch.where(valid_screen)[0]
                
                # Randomly shuffle valid indices for more diverse sampling
                perm = torch.randperm(len(valid_indices))
                valid_indices = valid_indices[perm][:samples_to_process]
                
                model_output = model(
                    glyph_chars=batch_device['game_chars'][valid_indices],
                    glyph_colors=batch_device['game_colors'][valid_indices],
                    blstats=batch_device['blstats'][valid_indices],
                    msg_tokens=batch_device['message_chars'][valid_indices],
                    hero_info=batch_device['hero_info'][valid_indices]
                )
                
                mu = model_output['mu']  # [samples_to_process, latent_dim]
                
                # Store latent representations
                local_latent_vectors.append(mu.cpu().numpy())
                
                # Store sample info with dataset labels
                batch_size = mu.shape[0]
                for i in range(batch_size):
                    local_sample_info.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'dataset': dataset_type,
                        'valid_screen': True,
                        'global_batch_idx': len(sample_info) + len(local_sample_info)
                    })
                
                sample_count += batch_size
                
                if sample_count % 50 == 0 or sample_count >= target_samples:
                    print(f"  📈 Extracted {sample_count}/{target_samples} {dataset_type} samples...")
        
        return local_latent_vectors, local_sample_info
    
    # Extract samples from training dataset
    if train_samples_target > 0:
        print(f"🔄 Extracting samples from training dataset...")
        train_latent_vectors, train_sample_info = extract_samples_from_batches(
            train_batches, train_labels, train_samples_target, 'train'
        )
        latent_vectors.extend(train_latent_vectors)
        sample_info.extend(train_sample_info)
        print(f"✅ Extracted {len(train_sample_info)} training samples")
    
    # Extract samples from testing dataset
    if test_samples_target > 0:
        print(f"🔄 Extracting samples from testing dataset...")
        test_latent_vectors, test_sample_info = extract_samples_from_batches(
            test_batches, test_labels, test_samples_target, 'test'
        )
        latent_vectors.extend(test_latent_vectors)
        sample_info.extend(test_sample_info)
        print(f"✅ Extracted {len(test_sample_info)} testing samples")
    
    # Combine all latent vectors
    if len(latent_vectors) == 0:
        raise ValueError("No samples could be extracted from the datasets")
    
    latent_vectors = np.vstack(latent_vectors)
    
    # Create batch indices for backward compatibility
    batch_indices = [info['global_batch_idx'] for info in sample_info]
    
    # Final shuffle for more uniform distribution in visualizations
    sample_indices = list(range(len(latent_vectors)))
    random.shuffle(sample_indices)
    latent_vectors = latent_vectors[sample_indices]
    sample_info = [sample_info[i] for i in sample_indices]
    batch_indices = [batch_indices[i] for i in sample_indices]
    
    # Create dataset color mapping
    unique_datasets = list(set([info['dataset'] for info in sample_info]))
    dataset_colors = {}
    colors = ['turquoise', 'lightcoral', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, dataset in enumerate(unique_datasets):
        dataset_colors[dataset] = colors[i % len(colors)]
    
    dataset_color_values = [dataset_colors[info['dataset']] for info in sample_info]
    
    print(f"📊 Enhanced Latent Space Analysis (Balanced Sampling):")
    print(f"  - Total samples analyzed: {len(latent_vectors)}")
    print(f"  - Datasets: {unique_datasets}")
    
    # Count samples per dataset after balancing
    actual_counts = {}
    for dataset in unique_datasets:
        count = sum(1 for info in sample_info if info['dataset'] == dataset)
        actual_counts[dataset] = count
        print(f"  - {dataset} samples: {count}")
    
    print(f"  - Latent dimensionality: {latent_vectors.shape[1]}")
    print(f"  - Latent mean: {np.mean(latent_vectors, axis=0)[:5]}...")
    print(f"  - Latent std: {np.std(latent_vectors, axis=0)[:5]}...")
    
    # Compute PCA and t-SNE first for use in multiple plots
    pca = PCA(n_components=min(10, latent_vectors.shape[1]))
    latent_pca = pca.fit_transform(latent_vectors)
    pca_explained_variance = pca.explained_variance_ratio_
        
    print("🔄 Computing t-SNE embedding (this may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)//4))
    latent_tsne = tsne.fit_transform(latent_vectors)
    
    # Compute statistics for plots
    latent_vars = np.var(latent_vectors, axis=0)
    latent_means = np.mean(latent_vectors, axis=0)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PCA of latent space colored by dataset (moved to first position)
    for dataset in unique_datasets:
        mask = [info['dataset'] == dataset for info in sample_info]
        if any(mask):
            dataset_pca = latent_pca[mask]
            axes[0, 0].scatter(dataset_pca[:, 0], dataset_pca[:, 1], 
                                c=dataset_colors[dataset], label=dataset, alpha=0.6, s=20)
    
    axes[0, 0].set_xlabel(f'PC1 ({pca_explained_variance[0]:.2%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({pca_explained_variance[1]:.2%} variance)')
    axes[0, 0].set_title('PCA of Latent Space (colored by dataset)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. t-SNE of latent space colored by dataset
    if latent_tsne is not None:
        for dataset in unique_datasets:
            mask = [info['dataset'] == dataset for info in sample_info]
            if any(mask):
                dataset_tsne = latent_tsne[mask]
                axes[0, 1].scatter(dataset_tsne[:, 0], dataset_tsne[:, 1], 
                                    c=dataset_colors[dataset], label=dataset, alpha=0.6, s=20)
        
        axes[0, 1].set_xlabel('t-SNE Dimension 1')
        axes[0, 1].set_ylabel('t-SNE Dimension 2')
        axes[0, 1].set_title('t-SNE of Latent Space (colored by dataset)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        # Fallback: show first two latent dimensions
        for dataset in unique_datasets:
            mask = [info['dataset'] == dataset for info in sample_info]
            if any(mask):
                dataset_latents = latent_vectors[mask]
                axes[0, 1].scatter(dataset_latents[:, 0], dataset_latents[:, 1], 
                                  c=dataset_colors[dataset], label=dataset, alpha=0.6, s=20)
        
        axes[0, 1].set_xlabel('Latent Dimension 0')
        axes[0, 1].set_ylabel('Latent Dimension 1')
        axes[0, 1].set_title('Raw Latent Space (t-SNE unavailable)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Variance per latent dimension
    axes[0, 2].bar(range(len(latent_vars)), latent_vars)
    axes[0, 2].set_xlabel('Latent Dimension')
    axes[0, 2].set_ylabel('Variance')
    axes[0, 2].set_title('Variance per Latent Dimension')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Mean per latent dimension
    axes[1, 0].bar(range(len(latent_means)), latent_means)
    axes[1, 0].set_xlabel('Latent Dimension')
    axes[1, 0].set_ylabel('Mean')
    axes[1, 0].set_title('Mean per Latent Dimension')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. PC1 distribution by dataset (instead of first latent dimension)
    dataset_pc1_values = []
    for dataset in unique_datasets:
        mask = [info['dataset'] == dataset for info in sample_info]
        if any(mask):
            dataset_pc1 = latent_pca[mask, 0]  # Get PC1 values for this dataset
            dataset_pc1_values.append(dataset_pc1)
        else:
            dataset_pc1_values.append(np.array([]))  # Empty array if no samples
    
    # Only plot non-empty datasets
    non_empty_datasets = [ds for ds, vals in zip(unique_datasets, dataset_pc1_values) if len(vals) > 0]
    non_empty_values = [vals for vals in dataset_pc1_values if len(vals) > 0]
    
    if non_empty_values:
        axes[1, 1].hist(non_empty_values, bins=30, alpha=0.7, label=non_empty_datasets, density=True)
    
    axes[1, 1].axvline(0, color='black', linestyle='--', alpha=0.7, label='Zero mean')
    axes[1, 1].set_xlabel(f'PC1 Value ({pca_explained_variance[0]:.2%} variance)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('PC1 Distribution by Dataset')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Correlation matrix of first 10 raw latent dimensions
    n_dims_to_show = min(10, latent_vectors.shape[1])
    corr_matrix = np.corrcoef(latent_vectors[:, :n_dims_to_show].T)
    im = axes[1, 2].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 2].set_xlabel('Latent Dimension')
    axes[1, 2].set_ylabel('Latent Dimension')
    axes[1, 2].set_title(f'Raw Latent Correlation Matrix (first {n_dims_to_show} dims)')
    
    # Add correlation colorbar
    cbar = plt.colorbar(im, ax=axes[1, 2])
    cbar.set_label('Correlation')
    
    pca_explained_variance = pca.explained_variance_ratio_
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Generate TTY visualization grid using PCA components
    print(f"\n🎮 Generating TTY visualization grid using PCA space...")
    # Use first 2 PCA components for the grid
    W = pca.components_[:2]  # [2, latent_dim] - first 2 principal components
    num_per_axis = 5
    points = torch.distributions.Normal(0,1).icdf(torch.linspace(0.01, 0.99, num_per_axis))
    XX, YY = torch.meshgrid(points, points, indexing='ij')
    XXYY = torch.stack((XX, YY)).reshape(2, -1).T  # [25, 2]
    
    # Transform from PCA space back to latent space
    pca_grid = XXYY.numpy()  # [25, 2] in PCA space
    # To go from PCA space to latent space: latent = pca_coords @ components + mean
    latent_grid = pca_grid @ W + latent_means[np.newaxis, :]  # [25, latent_dim]
    latent_grid = torch.tensor(latent_grid, dtype=torch.float32)
    
    with torch.no_grad():
        decode_output = model.sample(z=latent_grid.to(device))
        chars = decode_output['chars'].cpu()
        colors = decode_output['colors'].cpu()

    # Create TTY grid visualization
    from utils.analysis import _render_map_image
    tty_fig, tty_axes = plt.subplots(num_per_axis, num_per_axis, figsize=(30, 10))
    tty_fig.suptitle('Generated NetHack States from PCA Latent Space Grid (5x5)', fontsize=20, y=0.98)

    print(f"🎨 Rendering {num_per_axis * num_per_axis} NetHack states...")
    n = min(len(chars), num_per_axis * num_per_axis)
    imgs = [
        _render_map_image(chars[i], colors[i],
                          font_path="DejaVuSansMono.ttf", font_size=18, bg=(0,0,0))
        for i in range(n)
    ]
    tty_axes = tty_axes.ravel()
    for i, ax in enumerate(tty_axes):
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i], interpolation="nearest")
    plt.tight_layout()
    
    tty_save_path = save_path.replace('.png', '_tty_grid.png')
    plt.figure(tty_fig.number)
    plt.savefig(tty_save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"Enhanced latent space analysis saved to {save_path}")
    print(f"TTY grid visualization saved to {tty_save_path}")
    
    # Enhanced statistics with PCA focus
    print(f"\n📈 Enhanced Statistics (PCA-Focused Analysis):")
    print(f"  - Effective dimensionality (dims with var > 0.01): {np.sum(latent_vars > 0.01)}")
    print(f"  - High variance dimensions (var > 0.1): {np.sum(latent_vars > 0.1)}")
    print(f"  - Dimensions close to N(0,1): {np.sum((np.abs(latent_means) < 0.1) & (np.abs(latent_vars - 1.0) < 0.2))}")
    print(f"  - PCA components computed: {len(pca_explained_variance)}")
    print(f"  - PCA explained variance (first 5 components): {pca_explained_variance[:5]}")
    print(f"  - PCA cumulative variance (first 5): {np.cumsum(pca_explained_variance[:5])}")
    print(f"  - t-SNE computation: successful")
    
    print(f"\n🎯 Dataset Balance Analysis:")
    for dataset in unique_datasets:
        mask = [info['dataset'] == dataset for info in sample_info]
        dataset_latents = latent_vectors[mask]
        if len(dataset_latents) > 0:
            # Analyze in both raw latent space and PCA space
            dataset_pca = latent_pca[mask]
            dataset_mean_norm = np.linalg.norm(dataset_latents.mean(axis=0))
            dataset_std_norm = np.linalg.norm(dataset_latents.std(axis=0))
            dataset_pc1_mean = dataset_pca[:, 0].mean()
            dataset_pc1_std = dataset_pca[:, 0].std()
            print(f"  - {dataset} dataset:")
            print(f"    * Samples: {len(dataset_latents)}")
            print(f"    * Raw latent mean norm: {dataset_mean_norm:.3f}")
            print(f"    * Raw latent std norm: {dataset_std_norm:.3f}")
            print(f"    * PC1 mean: {dataset_pc1_mean:.3f}, PC1 std: {dataset_pc1_std:.3f}")
            print(f"    * Latent range: [{dataset_latents.min():.3f}, {dataset_latents.max():.3f}]")
            print(f"    * PC1 range: [{dataset_pca[:, 0].min():.3f}, {dataset_pca[:, 0].max():.3f}]")
    
    return {
        'latent_vectors': latent_vectors,
        'batch_indices': batch_indices, 
        'sample_info': sample_info,
        'latent_means': latent_means,
        'latent_vars': latent_vars,
        'pca_components': latent_pca,
        'pca_explained_variance': pca_explained_variance,
        'pca_model': pca,  # Include the fitted PCA model
        'tsne_components': latent_tsne,
        'dataset_labels': unique_datasets,
        'tty_grid_path': tty_save_path
    }
