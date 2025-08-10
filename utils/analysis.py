import os
import numpy as np
import torch
from nle.nethack import tty_render
import matplotlib.pyplot as plt
import math
import datetime as dt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

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
            md.write(f"## Sample {i}\n\n")
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

def visualize_reconstructions(model, test_dataset, device, num_samples=4, temperature=1.0, top_k=5, top_p=0.9, out_dir="vae_analysis", save_path="recon_comparison.md"):
    """
    Visualize VAE reconstructions for NetHack game states using tty_render
    
    Args:
        model: Trained MultiModalHackVAE model
        test_dataset: List of test batches (from NetHackDataCollector)
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
    """
    model.eval()
    
    reconstructions = []
    originals = []
    output_lines = []
    output_lines.append("NetHack VAE Reconstructions")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataset):  
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
            
            valid_screen = batch_device['valid_screen'].cpu()  # [B*T, H, W]
            
            # Get model output
            model_output = model(
                glyph_chars=batch_device['game_chars'][valid_screen][:max(0, num_samples - len(reconstructions))],
                glyph_colors=batch_device['game_colors'][valid_screen][:max(0, num_samples - len(reconstructions))],
                blstats=batch_device['blstats'][valid_screen][:max(0, num_samples - len(reconstructions))],
                msg_tokens=batch_device['message_chars'][valid_screen][:max(0, num_samples - len(reconstructions))],
                hero_info=batch_device['hero_info'][valid_screen][:max(0, num_samples - len(reconstructions))],
                training_mode=False,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # Get most likely character and color for each cell
            char_recon = model_output['generated_chars'][0].cpu()  # [H, W]
            color_recon = model_output['generated_colors'][0].cpu()  # [H, W]

            # Get original data (take first sample from batch)
            char_orig = batch_device['game_chars'][0].cpu()  # [H, W]
            color_orig = batch_device['game_colors'][0].cpu()  # [H, W]
            
            reconstructions.append((char_recon, color_recon))
            originals.append((char_orig, color_orig))
            
            if len(reconstructions) >= num_samples:
                break
    
    save_maps_and_markdown(originals, reconstructions,
                            out_dir=out_dir,
                            md_filename=save_path,
                            font_path="DejaVuSansMono.ttf",  # optional
                            font_size=18)

    return {
        'num_samples': len(reconstructions),
        'originals': originals,
        'reconstructions': reconstructions,
        'save_path': save_path
    }

def analyze_latent_space(model, test_dataset, device, save_path="vae_analysis/latent_analysis.png", max_samples=100):
    """
    Analyze the learned latent space of the VAE
    
    Args:
        model: Trained MultiModalHackVAE model
        test_dataset: List of test batches (from NetHackDataCollector)
        device: Device to run inference on
        save_path: Path to save the analysis plots
        max_samples: Maximum number of samples to analyze
    """
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    latent_vectors = []
    batch_indices = []
    sample_info = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataset):
            if sample_count >= max_samples:
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
                    
            valid_screen = batch_device['valid_screen'].cpu()  # [B*T, H, W]
            
            # Get model output (includes mu, logvar, lowrank_factors)
            model_output = model(
                glyph_chars=batch_device['game_chars'][valid_screen][:max(0, max_samples - sample_count)],
                glyph_colors=batch_device['game_colors'][valid_screen][:max(0, max_samples - sample_count)],
                blstats=batch_device['blstats'][valid_screen][:max(0, max_samples - sample_count)],
                msg_tokens=batch_device['message_chars'][valid_screen][:max(0, max_samples - sample_count)],
                hero_info=batch_device['hero_info'][valid_screen][:max(0, max_samples - sample_count)]
            )
            
            mu = model_output['mu']  # [B*T, latent_dim]
            
            # Store latent representations
            latent_vectors.append(mu.cpu().numpy())
            
            # Store batch information
            batch_size = mu.shape[0]
            batch_indices.extend([batch_idx] * batch_size)
            
            # Store some sample info if available
            for i in range(batch_size):
                sample_info.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'valid_screen': batch_device['valid_screen'][i].item() if 'valid_screen' in batch_device else True
                })
            
            sample_count += batch_size
    
    # Combine all latent vectors
    latent_vectors = np.vstack(latent_vectors)
    latent_vectors = latent_vectors[:max_samples]  # Trim to max_samples
    batch_indices = batch_indices[:max_samples]
    sample_info = sample_info[:max_samples]
    
    print(f"📊 Latent Space Analysis:")
    print(f"  - Total samples analyzed: {len(latent_vectors)}")
    print(f"  - Latent dimensionality: {latent_vectors.shape[1]}")
    print(f"  - Latent mean: {np.mean(latent_vectors, axis=0)[:5]}...")
    print(f"  - Latent std: {np.std(latent_vectors, axis=0)[:5]}...")
    print(f"  - Min latent values: {np.min(latent_vectors, axis=0)[:5]}...")
    print(f"  - Max latent values: {np.max(latent_vectors, axis=0)[:5]}...")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. First two latent dimensions colored by batch
    axes[0, 0].scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                      c=batch_indices, cmap='tab10', alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Latent Dimension 0')
    axes[0, 0].set_ylabel('Latent Dimension 1')
    axes[0, 0].set_title('Latent Space (Dims 0-1, colored by batch)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Latent dimension variances
    latent_vars = np.var(latent_vectors, axis=0)
    axes[0, 1].bar(range(len(latent_vars)), latent_vars)
    axes[0, 1].set_xlabel('Latent Dimension')
    axes[0, 1].set_ylabel('Variance')
    axes[0, 1].set_title('Variance per Latent Dimension')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Latent means
    latent_means = np.mean(latent_vectors, axis=0)
    axes[0, 2].bar(range(len(latent_means)), latent_means)
    axes[0, 2].set_xlabel('Latent Dimension')
    axes[0, 2].set_ylabel('Mean')
    axes[0, 2].set_title('Mean per Latent Dimension')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Distribution of first latent dimension
    axes[1, 0].hist(latent_vectors[:, 0], bins=50, alpha=0.7, density=True)
    axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='N(0,1) mean')
    axes[1, 0].set_xlabel('Latent Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Latent Dim 0')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Pairwise correlations (first 10 dimensions)
    n_dims_to_show = min(10, latent_vectors.shape[1])
    corr_matrix = np.corrcoef(latent_vectors[:, :n_dims_to_show].T)
    im = axes[1, 1].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('Latent Dimension')
    axes[1, 1].set_title(f'Correlation Matrix (first {n_dims_to_show} dims)')
    plt.colorbar(im, ax=axes[1, 1])
    
    # 6. Principal components analysis
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vectors)
    
    axes[1, 2].scatter(latent_pca[:, 0], latent_pca[:, 1], 
                      c=batch_indices, cmap='tab10', alpha=0.6, s=20)
    axes[1, 2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1, 2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1, 2].set_title('PCA of Latent Space')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # 7. Create standalone TTY visualization figure (15x15 grid = 225 samples)
    print(f"\n🎮 Generating TTY visualization grid...")
    W = torch.linalg.svd(torch.tensor(latent_vectors, dtype=torch.float32)).Vh[:2]
    num_per_axis = 5
    points = torch.distributions.Normal(0,1).icdf(torch.linspace(0.01, 0.99, num_per_axis))  # 5x5 grid = 25 samples
    XX, YY = torch.meshgrid(points, points, indexing='ij')
    XXYY = torch.stack((XX, YY)).reshape(2, -1).T
    latent_grid = XXYY @ W
    
    with torch.no_grad():
        decode_output = model.decode(latent_grid.to(device))
        chars = decode_output['generated_chars'].cpu()  # [225, 21, 79]
        colors = decode_output['generated_colors'].cpu()  # [225, 21, 79]
    
    # Create giant figure for TTY renders
    tty_fig, tty_axes = plt.subplots(num_per_axis, num_per_axis, figsize=(30, 10))
    tty_fig.suptitle('Generated NetHack States from Latent Space Grid (5x5)', fontsize=20, y=0.98)

    # Generate and display TTY renders in grid
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
    # Save the TTY visualization
    tty_save_path = save_path.replace('.png', '_tty_grid.png')
    plt.figure(tty_fig.number)  # Make sure we're working with the TTY figure
    plt.savefig(tty_save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"Latent space analysis saved to {save_path}")
    
    # Additional statistics
    print(f"\n📈 Additional Statistics:")
    print(f"  - Effective dimensionality (dims with var > 0.01): {np.sum(latent_vars > 0.01)}")
    print(f"  - High variance dimensions (var > 0.1): {np.sum(latent_vars > 0.1)}")
    print(f"  - Dimensions close to N(0,1): {np.sum((np.abs(latent_means) < 0.1) & (np.abs(latent_vars - 1.0) < 0.2))}")
    print(f"  - PCA explained variance (first 2 components): {pca.explained_variance_ratio_[:2].sum():.2%}")
    
    return {
        'latent_vectors': latent_vectors,
        'batch_indices': batch_indices, 
        'sample_info': sample_info,
        'latent_means': latent_means,
        'latent_vars': latent_vars,
        'pca_components': latent_pca,
        'pca_explained_variance': pca.explained_variance_ratio_
    }
