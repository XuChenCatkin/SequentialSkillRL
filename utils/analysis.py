import os
import numpy as np
import torch
from nle.nethack import tty_render
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm
from datetime import datetime
import json
import math

from src.model import bag_presence_to_glyph_sets, make_pair_bag, MapDecoder
from training.train import load_model_from_huggingface, load_model_from_local
from src.skill_space import StickyHDPHMMVI

# Import NetHackCategory from data_collection
try:
    from src.data_collection import NetHackCategory, crop_ego, categorize_glyph_tensor, compute_passability_and_safety
except ImportError:
    try:
        from .data_collection import NetHackCategory, crop_ego, categorize_glyph_tensor, compute_passability_and_safety
    except ImportError:
        raise ImportError("Could not import NetHackCategory from data_collection module.")

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
    img_file_prefix="",
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

            fn_o = f"{img_file_prefix}sample_{i:03d}_orig.png"
            fn_r = f"{img_file_prefix}sample_{i:03d}_recon.png"
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

def _render_ego_class_grid(ego_class: torch.Tensor, save_path: str = None) -> np.ndarray:
    """
    Render ego class predictions as a colored grid.
    
    Args:
        ego_class: [k, k] tensor with class indices
        save_path: Optional path to save the image
        
    Returns:
        Rendered image as numpy array
    """
    k = ego_class.shape[0]
    ego_class_np = ego_class.cpu().numpy()
    
    # Create a colormap for the classes
    num_classes = NetHackCategory.get_category_count()
    colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # Slightly wider to accommodate colorbar better
    
    # Create colored grid
    colored_grid = np.zeros((k, k, 3))
    for i in range(k):
        for j in range(k):
            class_idx = ego_class_np[i, j]
            if 0 <= class_idx < num_classes:
                colored_grid[i, j] = colors[class_idx][:3]
    
    im = ax.imshow(colored_grid, interpolation='nearest', vmin=0, vmax=num_classes-1)
    
    # Add text annotations with class numbers
    for i in range(k):
        for j in range(k):
            class_idx = ego_class_np[i, j]
            color = 'white' if np.mean(colored_grid[i, j]) < 0.5 else 'black'
            ax.text(j, i, str(class_idx), ha='center', va='center', 
                   color=color, fontsize=10, fontweight='bold')
    
    ax.set_title(f'Ego Class Predictions ({k}x{k})', fontsize=14)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.grid(True, alpha=0.3)
    
    # Add colorbar with fixed range to ensure consistency
    import matplotlib.colors as mcolors
    cmap = plt.cm.tab20
    norm = mcolors.Normalize(vmin=0, vmax=num_classes-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Class Categories', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return None
    
    # Convert to numpy array for return (only if not saving)
    fig.canvas.draw()
    # Use modern matplotlib method for getting canvas data
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    # Convert RGBA to RGB by dropping alpha channel
    img_array = img_array[:, :, :3]
    plt.close()
    return img_array

def _render_passability_safety_grid(
    passability_grid: torch.Tensor, 
    safety_grid: torch.Tensor, 
    save_path: str = None
) -> np.ndarray:
    """
    Render passability and safety as 3x3 grids with '@' in center.
    
    Args:
        passability_grid: [3, 3] tensor with passability probabilities
        safety_grid: [3, 3] tensor with safety probabilities  
        save_path: Optional path to save the image
        
    Returns:
        Rendered image as numpy array
    """
    pass_np = passability_grid.cpu().numpy()
    safe_np = safety_grid.cpu().numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Passability grid
    im1 = ax1.imshow(pass_np, cmap='summer', vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title('Passability')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:  # Center position
                ax1.text(j, i, '@', ha='center', va='center', 
                        color='black', fontsize=20, fontweight='bold')
            else:
                value = pass_np[i, j]
                color = 'white' if value < 0.5 else 'black'
                ax1.text(j, i, f'{value:.2f}', ha='center', va='center', 
                        color=color, fontsize=12, fontweight='bold')
    
    ax1.set_xticks(range(3))
    ax1.set_yticks(range(3))
    ax1.grid(True, alpha=0.3)
    
    # Safety grid
    im2 = ax2.imshow(safe_np, cmap='summer', vmin=0, vmax=1, interpolation='nearest')
    ax2.set_title('Safety')
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:  # Center position
                ax2.text(j, i, '@', ha='center', va='center', 
                        color='black', fontsize=20, fontweight='bold')
            else:
                value = safe_np[i, j]
                color = 'white' if value < 0.5 else 'black'
                ax2.text(j, i, f'{value:.2f}', ha='center', va='center', 
                        color=color, fontsize=12, fontweight='bold')
    
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.grid(True, alpha=0.3)
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, shrink=0.6)
    plt.colorbar(im2, ax=ax2, shrink=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Convert to numpy array for return
    fig.canvas.draw()
    # Use modern matplotlib method for getting canvas data
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    # Convert RGBA to RGB by dropping alpha channel
    img_array = img_array[:, :, :3]
    
    if not save_path:
        plt.close()
    
    return img_array

def _format_bag_reconstruction(bag_data: Tuple) -> str:
    """
    Format bag reconstruction data into readable text.
    
    Args:
        bag_data: Tuple of (bag_original, bag_reconstruction) where each is a set of (char_ascii, color_idx) tuples
        
    Returns:
        Formatted string representation
    """
    if not bag_data or len(bag_data) != 2:
        return "No bag data available"
    
    bag_original, bag_reconstruction = bag_data
    
    if not bag_original and not bag_reconstruction:
        return "Both original and reconstructed bags are empty"
    
    lines = [f"Bag Analysis:"]
    lines.append("=" * 40)
    
    # Format original bag
    lines.append(f"\nOriginal Bag ({len(bag_original)} items):")
    lines.append("-" * 30)
    if bag_original:
        for char_ascii, color_idx in sorted(bag_original):
            char_str = chr(char_ascii) if 32 <= char_ascii <= 126 else f"\\x{char_ascii:02x}"
            lines.append(f"  '{char_str}' (color {color_idx:2d})")
    else:
        lines.append("  (empty)")
    
    # Format reconstructed bag
    lines.append(f"\nReconstructed Bag ({len(bag_reconstruction)} items):")
    lines.append("-" * 30)
    if bag_reconstruction:
        for char_ascii, color_idx in sorted(bag_reconstruction):
            char_str = chr(char_ascii) if 32 <= char_ascii <= 126 else f"\\x{char_ascii:02x}"
            lines.append(f"  '{char_str}' (color {color_idx:2d})")
    else:
        lines.append("  (empty)")
    
    # Calculate accuracy metrics
    lines.append(f"\nAccuracy Metrics:")
    lines.append("-" * 30)
    
    # Items in both (correct predictions)
    correct_items = bag_original & bag_reconstruction
    lines.append(f"  Correctly predicted: {len(correct_items)} items")
    if correct_items:
        for char_ascii, color_idx in sorted(correct_items):
            char_str = chr(char_ascii) if 32 <= char_ascii <= 126 else f"\\x{char_ascii:02x}"
            lines.append(f"    '{char_str}' (color {color_idx:2d})")
    
    # Items only in original (missed items)
    missed_items = bag_original - bag_reconstruction
    lines.append(f"  Missed items: {len(missed_items)} items")
    if missed_items:
        for char_ascii, color_idx in sorted(missed_items):
            char_str = chr(char_ascii) if 32 <= char_ascii <= 126 else f"\\x{char_ascii:02x}"
            lines.append(f"    '{char_str}' (color {color_idx:2d})")
    
    # Items only in reconstruction (false positives)
    false_positive_items = bag_reconstruction - bag_original
    lines.append(f"  False positives: {len(false_positive_items)} items")
    if false_positive_items:
        for char_ascii, color_idx in sorted(false_positive_items):
            char_str = chr(char_ascii) if 32 <= char_ascii <= 126 else f"\\x{char_ascii:02x}"
            lines.append(f"    '{char_str}' (color {color_idx:2d})")
    
    # Calculate accuracy percentages
    total_unique_items = len(bag_original | bag_reconstruction)
    if total_unique_items > 0:
        precision = len(correct_items) / len(bag_reconstruction) if bag_reconstruction else 0.0
        recall = len(correct_items) / len(bag_original) if bag_original else 1.0 if not bag_reconstruction else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        lines.append(f"\nPerformance Summary:")
        lines.append("-" * 30)
        lines.append(f"  Precision: {precision:.3f} ({len(correct_items)}/{len(bag_reconstruction)})")
        lines.append(f"  Recall: {recall:.3f} ({len(correct_items)}/{len(bag_original)})")
        lines.append(f"  F1-Score: {f1_score:.3f}")
        lines.append(f"  Total unique items: {total_unique_items}")
    
    return "\n".join(lines)

def save_enhanced_reconstructions(
    originals, 
    reconstructions, 
    bag_data, 
    pass_data,
    safe_data,
    out_dir="vae_analysis",
    md_filename="enhanced_recon_comparison.md",
    img_file_prefix="",
    font_path=None,
    font_size=18,
    bg=(0,0,0),
    title="Enhanced VAE Reconstruction Comparison",
):
    """
    Save enhanced reconstructions including ego, bag, and passability/safety data.
    
    Args:
        originals: List of (char_orig, color_orig) pairs
        reconstructions: List of (char_recon, color_recon) pairs  
        ego_data: List of ego reconstruction data per sample
        bag_data: List of bag reconstruction data per sample
        pass_data: List of passability data per sample
        safe_data: List of safety data per sample
        ... (other args same as save_maps_and_markdown)
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Write enhanced markdown
    md_path = out_dir / md_filename
    with md_path.open("w", encoding="utf-8") as md:
        md.write(f"# {title}\n\n")
        md.write(f"_Generated: {dt.datetime.now().isoformat(timespec='seconds')}_\n\n")
        md.write("This analysis includes the following reconstructions:\n")
        md.write("- **Ego View**: Character, color, and class predictions in ego-centric window\n")
        md.write("- **Bag Elements**: High-probability glyph elements\n")
        md.write("- **Passability/Safety**: 3x3 grids around hero position\n\n")

        for i, (orig, recon) in enumerate(zip(originals, reconstructions)):
            char_orig, color_orig, class_orig = orig
            char_recon, color_recon, class_recon = recon

            # Main map reconstruction
            img_o = _render_map_image(char_orig, color_orig, font_path=font_path, font_size=font_size, bg=bg)
            img_r = _render_map_image(char_recon, color_recon, font_path=font_path, font_size=font_size, bg=bg)

            fn_o = f"{img_file_prefix}sample_{i:03d}_orig.png"
            fn_r = f"{img_file_prefix}sample_{i:03d}_recon.png"
            img_o.save(img_dir / fn_o)
            img_r.save(img_dir / fn_r)

            # Ego class grid
            fn_ego_class_orig = f"{img_file_prefix}sample_{i:03d}_ego_class_orig.png"
            _render_ego_class_grid(class_orig, save_path=str(img_dir / fn_ego_class_orig))

            # Ego class grid
            fn_ego_class_recon = f"{img_file_prefix}sample_{i:03d}_ego_class_recon.png"
            _render_ego_class_grid(class_recon, save_path=str(img_dir / fn_ego_class_recon))

            # Passability/Safety grids
            if i < len(pass_data) and pass_data[i] is not None and i < len(safe_data) and safe_data[i] is not None:
                pass_grid, pass_grid_recon = pass_data[i]
                safe_grid, safe_grid_recon = safe_data[i]
                fn_pass_safe_orig = f"{img_file_prefix}sample_{i:03d}_pass_safe_orig.png"
                fn_pass_safe_recon = f"{img_file_prefix}sample_{i:03d}_pass_safe_recon.png"
                _render_passability_safety_grid(pass_grid, safe_grid, save_path=str(img_dir / fn_pass_safe_orig))
                _render_passability_safety_grid(pass_grid_recon, safe_grid_recon, save_path=str(img_dir / fn_pass_safe_recon))

            # Markdown section
            md.write(f"## Sample {i+1}\n\n")
            
            # Main reconstruction table
            md.write("### Ego Map Reconstruction\n\n")
            md.write("| Original | Reconstruction |\n")
            md.write("|---|---|\n")
            md.write(f"| ![orig {i}](images/{fn_o}) | ![recon {i}](images/{fn_r}) |\n\n")
            
            # Calculate and display accuracy
            char_matches = (char_orig == char_recon).float()
            color_matches = (color_orig == color_recon).float()
            char_accuracy = char_matches.mean().item()
            color_accuracy = color_matches.mean().item()

            md.write(f"**Accuracy**: Character: {char_accuracy:.3f}, Color: {color_accuracy:.3f}\n\n")

            md.write("### Ego Class Reconstruction\n\n")
            md.write("| Original | Reconstruction |\n")
            md.write("|---|---|\n")
            md.write(f"| ![orig class {i}](images/{fn_ego_class_orig}) | ![recon class {i}](images/{fn_ego_class_recon}) |\n\n")
            
            class_matches = (class_orig == class_recon).float()
            class_accuracy = class_matches.mean().item()
            md.write(f"**Class Accuracy**: {class_accuracy:.3f}\n\n")

            # Bag reconstruction section
            if i < len(bag_data) and bag_data[i] and len(bag_data[i]) == 2:
                md.write("### Bag Reconstruction\n\n")
                bag_text = _format_bag_reconstruction(bag_data[i])
                md.write("```\n")
                md.write(bag_text)
                md.write("\n```\n\n")

            # Passability/Safety section
            if i < len(pass_data) and pass_data[i] is not None and i < len(safe_data) and safe_data[i] is not None:
                md.write("### Passability & Safety\n\n")
                md.write("| Original | Reconstruction |\n")
                md.write("|---|---|\n")
                md.write(f"| ![orig pass safe {i}](images/{fn_pass_safe_orig}) | ![recon pass safe {i}](images/{fn_pass_safe_recon}) |\n\n")

            if i < len(reconstructions) - 1:
                md.write("=" * 80 + "\n\n")
        
        # Overall statistics
        md.write("## Overall Statistics\n\n")
        char_accuracy = sum((orig[0] == recon[0]).float().mean().item() for orig, recon in zip(originals, reconstructions)) / len(originals)
        color_accuracy = sum((orig[1] == recon[1]).float().mean().item() for orig, recon in zip(originals, reconstructions)) / len(originals)

        md.write(f"- **Average Character Accuracy**: {char_accuracy:.3f}\n")
        md.write(f"- **Average Color Accuracy**: {color_accuracy:.3f}\n")
        md.write(f"- **Total Samples**: {len(reconstructions)}\n")

    print(f"Wrote Enhanced Markdown: {md_path}")

def visualize_reconstructions(
    model, dataset, device, 
    num_samples=4, 
    out_dir="vae_analysis", save_path="recon_comparison.md",
    img_file_prefix="",
    random_sampling=True, dataset_name="Dataset",
    # VAE sampling parameters
    use_mean=True,
    include_logits=False,
    # map sampling parameters
    map_temperature=1.0,
    map_occ_thresh=0.5,
    bag_presence_thresh=0.5,
    hero_presence_thresh=0.5,
    passability_thresh=0.5,
    safety_thresh=0.5,
    map_deterministic=True,
    glyph_top_k=0,
    glyph_top_p=1.0,
    color_top_k=0,
    color_top_p=1.0,
    class_top_k=0,
    class_top_p=1.0,
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
        bag_presence_thresh: Threshold for bag presence prediction
        hero_presence_thresh: Threshold for hero presence prediction
        passability_thresh: Threshold for passability prediction
        safety_thresh: Threshold for safety prediction
        map_deterministic: If True, use deterministic (argmax) sampling for map
        glyph_top_k: Top-k filtering for glyph character sampling (0 = disabled)
        glyph_top_p: Top-p (nucleus) filtering for glyph character sampling
        color_top_k: Top-k filtering for color sampling (0 = disabled)
        color_top_p: Top-p (nucleus) filtering for color sampling
        class_top_k: Top-k filtering for class sampling (0 = disabled)
        class_top_p: Top-p (nucleus) filtering for class sampling
        
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
    bag_data = []
    pass_data = []
    safe_data = []
    output_lines = []
    output_lines.append(f"Enhanced NetHack VAE Reconstructions - {dataset_name} Dataset")
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
    
    print(f"🎯 Processing {len(selected_samples)} samples for enhanced reconstruction...")
    
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
                bag_presence_thresh=bag_presence_thresh,
                hero_presence_thresh=hero_presence_thresh,
                passability_thresh=passability_thresh,
                safety_thresh=safety_thresh,
                map_deterministic=map_deterministic,
                glyph_top_k=glyph_top_k,
                glyph_top_p=glyph_top_p,
                color_top_k=color_top_k,
                color_top_p=color_top_p,
                class_top_k=class_top_k,
                class_top_p=class_top_p,
                # Message sampling parameters
                msg_temperature=msg_temperature,
                msg_top_k=msg_top_k,
                msg_top_p=msg_top_p,
                msg_deterministic=msg_deterministic,
                allow_eos=allow_eos,
                forbid_eos_at_start=forbid_eos_at_start,
                allow_pad=allow_pad
            )
            
            # === Extract traditional map reconstruction ===
            # The new model focuses on ego-centric view, not full map reconstruction
            # Use ego data as the main "reconstructed" view for comparison
            if 'ego_chars' in model_output and 'ego_colors' in model_output and 'ego_class' in model_output:
                char_recon = model_output['ego_chars'][0].cpu()  # [k, k]
                color_recon = model_output['ego_colors'][0].cpu()  # [k, k]
                class_recon = model_output['ego_class'][0].cpu()  # [k, k]
                hero_x_coord, hero_y_coord = int(model_inputs['blstats'][0, 0].item()), int(model_inputs['blstats'][0, 1].item())
                k = char_recon.shape[0]  # ego window size
                # Also extract original ego view for fair comparison
                # We need to crop the original map to the ego window size
                char_orig_ego, color_orig_ego = crop_ego(sample['game_chars'], sample['game_colors'], hero_y_coord, hero_x_coord, k)  # [k, k]
                char_orig_ego = char_orig_ego.cpu()
                color_orig_ego = color_orig_ego.cpu()
                class_orig_ego = categorize_glyph_tensor(char_orig_ego, color_orig_ego)
                class_orig_ego = class_orig_ego.cpu()

            reconstructions.append((char_recon, color_recon, class_recon))
            originals.append((char_orig_ego, color_orig_ego, class_orig_ego))
            
            # === Extract bag reconstruction data ===
            if 'bag_sets' in model_output:
                bag_recon = model_output['bag_sets'][0]
                bag_presence_orig = make_pair_bag(sample['game_chars'].unsqueeze(0), sample['game_colors'].unsqueeze(0))
                bag_orig = bag_presence_to_glyph_sets(bag_presence_orig)[0]
                bag_data.append((bag_orig, bag_recon))
            else:
                bag_data.append([])
            
            # === Extract passability and safety data ===
            if 'passability_grid' in model_output and 'safety_grid' in model_output:
                hero_y_coord, hero_x_coord = int(model_inputs['blstats'][0, 0].item()), int(model_inputs['blstats'][0, 1].item())
                pass_grid_recon = model_output['passability_grid'][0].cpu()  # [3, 3]
                safe_grid_recon = model_output['safety_grid'][0].cpu()       # [3, 3]
                pass_presence_orig, safe_presence_orig, hard_mask_orig, weight_orig = compute_passability_and_safety(
                    sample['game_chars'], sample['game_colors'],
                    hero_y_coord, hero_x_coord
                )
                pass_safe_dict = MapDecoder.format_passability_safety(pass_presence_orig.unsqueeze(0), safe_presence_orig.unsqueeze(0))
                pass_data.append((pass_safe_dict['passability_grid'].squeeze(0).cpu(), pass_grid_recon))
                safe_data.append((pass_safe_dict['safety_grid'].squeeze(0).cpu(), safe_grid_recon))
            else:
                pass_data.append(None)
                safe_data.append(None)

    # Use enhanced save function
    save_enhanced_reconstructions(
        originals, reconstructions, bag_data, pass_data, safe_data,
        out_dir=out_dir,
        md_filename=save_path,
        img_file_prefix=img_file_prefix,
        font_path="DejaVuSansMono.ttf",  # optional
        font_size=18
    )
    
    print(f"✅ {dataset_name} enhanced reconstruction visualization saved to {out_dir}/{save_path}")

    return {
        'num_samples': len(reconstructions),
        'originals': originals,
        'reconstructions': reconstructions,
        'bag_data': bag_data,
        'pass_data': pass_data,
        'safe_data': safe_data,
        'save_path': save_path,
        'dataset_name': dataset_name,
        'random_sampling': random_sampling,
        'sampling_params': {
            'use_mean': use_mean,
            'include_logits': include_logits,
            'map_temperature': map_temperature,
            'map_occ_thresh': map_occ_thresh,
            'hero_presence_thresh': hero_presence_thresh,
            'bag_presence_thresh': bag_presence_thresh,
            'passability_thresh': passability_thresh,
            'safety_thresh': safety_thresh,
            'map_deterministic': map_deterministic,
            'glyph_top_k': glyph_top_k,
            'glyph_top_p': glyph_top_p,
            'color_top_k': color_top_k,
            'color_top_p': color_top_p,
            'class_top_k': class_top_k,
            'class_top_p': class_top_p,
            'msg_temperature': msg_temperature,
            'msg_top_k': msg_top_k,
            'msg_top_p': msg_top_p,
            'msg_deterministic': msg_deterministic,
            'allow_eos': allow_eos,
            'forbid_eos_at_start': forbid_eos_at_start,
            'allow_pad': allow_pad
        }
    }

def _mean_ekl_diag_or_lowrank(mu_np, logvar_np, lowrank_np=None):
    """
    Average per-sample KL(q(z|x) || N(0,I)).
    Supports diag posterior (logvar) and optional low-rank factors F (B,D,R).
    Uses matrix determinant lemma: det(Λ + FF^T) = det(Λ) * det(I + F^T Λ^{-1} F).
    """
    B, D = mu_np.shape
    mu2_sum = (mu_np**2).sum(axis=1)                      # [B]
    if lowrank_np is None:
        # diag Σ = diag(exp(logvar))
        s2 = np.exp(logvar_np)                            # [B,D]
        tr_sum = s2.sum(axis=1)                           # [B]
        logdet = logvar_np.sum(axis=1)                    # [B] (since det(Λ)=exp(sum logvar))
        kl = 0.5 * (tr_sum + mu2_sum - D - logdet)
        return float(kl.mean())
    else:
        # Σ = Λ + F F^T, Λ=diag(s2)
        F = lowrank_np                                    # [B,D,R]
        s2 = np.exp(logvar_np)                            # [B,D]
        tr_sum = s2.sum(axis=1) + np.sum(F**2, axis=(1,2))
        # logdet(Σ) = logdet(Λ) + logdet(I + F^T Λ^{-1} F)
        logdet_L = logvar_np.sum(axis=1)                  # [B]
        # build R x R matrix per-sample: I + F^T Λ^{-1} F
        invL_F = F / np.maximum(s2[..., None], 1e-8)      # [B,D,R]
        Bt = np.matmul(F.transpose(0,2,1), invL_F)        # [B,R,R]
        # stable logdet via slogdet
        sign, logdet_B = np.linalg.slogdet(np.eye(Bt.shape[1])[None,...] + Bt)
        logdet_B = np.where(sign > 0, logdet_B, -1e9)     # guard
        logdet = logdet_L + logdet_B
        kl = 0.5 * (tr_sum + mu2_sum - D - logdet)
        return float(kl.mean())

def analyze_latent_space(
    model,
    dataset,
    device,
    save_path="vae_analysis/latent_analysis.png",
    max_samples=100,
    dataset_labels=None,
    tsne_on_pca=True,
    jitter_eps=1e-5,
):
    """
    Balanced latent-space analysis with MI/TC/DW that includes posterior noise.
    - Aggregated posterior covariance: Var_x[mu_x] + E_x[diag_var_x + F_x F_x^T]
    - Safer t-SNE, covariance shrinkage, and numerically stable logdets.
    """
    import os, random
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import torch

    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if dataset_labels is None:
        dataset_labels = ['train'] * len(dataset)  # default to train

    # --- split batches by label
    train_batches, test_batches = [], []
    for b, lbl in zip(dataset, dataset_labels):
        (train_batches if lbl == 'train' else test_batches).append(b)

    # --- target allocation
    if len(train_batches) and len(test_batches):
        n_train = max_samples // 2
        n_test  = max_samples - n_train
    elif len(train_batches):
        n_train, n_test = max_samples, 0
    elif len(test_batches):
        n_train, n_test = 0, max_samples
    else:
        raise ValueError("No valid datasets provided.")

    random.shuffle(train_batches)
    random.shuffle(test_batches)

    # containers
    mu_list, logvar_list, lowrank_list, ds_list = [], [], [], []

    def _pull_from_batches(batches, target, tag):
        nonlocal mu_list, logvar_list, lowrank_list, ds_list
        got = 0
        with torch.no_grad():
            for batch in batches:
                if got >= target:
                    break
                # Move to device & flatten [B,T,...] -> [B*T,...]
                flat = {}
                BT = None
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        v = v.to(device)
                        if v.ndim >= 2:
                            B, T = v.shape[:2]
                            v = v.view(B * T, *v.shape[2:])
                            BT = B * T if BT is None else BT
                        flat[k] = v
                valid = flat.get('valid_screen')
                if valid is None:
                    # assume everything valid
                    N = next(iter(flat.values())).shape[0]
                    valid = torch.ones(N, dtype=torch.bool, device=device)

                idx = torch.where(valid)[0]
                if idx.numel() == 0:
                    continue
                # shuffle valid positions, take up to what's left
                need = int(min(target - got, idx.numel()))
                if need <= 0:
                    break
                perm = torch.randperm(idx.numel(), device=device)[:need]
                idx = idx[perm]

                feed = {
                    'game_chars':    flat['game_chars'][idx],
                    'game_colors':   flat['game_colors'][idx],
                    'blstats':       flat['blstats'][idx],
                    'message_chars': flat['message_chars'][idx],
                    'hero_info':     flat['hero_info'][idx],
                }
                out = model(feed)
                mu      = out['mu']                        # [n,D]
                logvar  = out.get('logvar', None)          # [n,D]
                lowrank = out.get('lowrank_factors', None) # [n,D,R] or None

                mu_list.append(mu.detach().cpu().numpy())
                if logvar is not None:
                    logvar_list.append(logvar.detach().cpu().numpy())
                if lowrank is not None:
                    lowrank_list.append(lowrank.detach().cpu().numpy())
                ds_list.extend([tag] * mu.shape[0])
                got += mu.shape[0]
        return got

    got_train = _pull_from_batches(train_batches, n_train, 'train') if n_train else 0
    got_test  = _pull_from_batches(test_batches,  n_test,  'test')  if n_test  else 0
    total = got_train + got_test
    if total == 0:
        raise RuntimeError("No valid samples collected. Check valid_screen and inputs.")

    # --- stack
    mu_all = np.vstack(mu_list)                            # [N,D]
    D = mu_all.shape[1]
    logvar_all  = np.vstack(logvar_list) if logvar_list else None
    lowrank_all = np.vstack(lowrank_list) if lowrank_list else None  # shape: [N,D,R] collapsed ok

    # --- compute aggregated posterior covariance: Var[mu] + E[diag(var)] + E[FF^T]
    mu_mean = mu_all.mean(axis=0)                          # [D]
    mu_centered = mu_all - mu_mean
    cov_between = (mu_centered.T @ mu_centered) / (mu_all.shape[0] - 1 + 1e-6)  # Var_x[mu_x]  [D,D]

    if logvar_all is not None:
        E_diag = np.exp(logvar_all).mean(axis=0)           # [D]
    else:
        E_diag = np.zeros(D, dtype=np.float64)

    if lowrank_all is not None:
        # lowrank_all: [N, D, R] -> average FF^T over samples (R is small, so this is cheap for ~100 samples)
        N, D_, R = lowrank_all.shape
        assert D_ == D
        E_FFt = np.zeros((D, D), dtype=np.float64)
        for i in range(N):
            F = lowrank_all[i]                             # [D,R]
            E_FFt += F @ F.T
        E_FFt /= N
    else:
        E_FFt = np.zeros((D, D), dtype=np.float64)

    Sigma_agg = cov_between + np.diag(E_diag) + E_FFt      # [D,D]
    # numerical shrinkage
    Sigma_agg = 0.5 * (Sigma_agg + Sigma_agg.T)
    Sigma_agg += jitter_eps * np.eye(D)

    # --- per-dim stats for plots
    var_total = np.diag(Sigma_agg)                         # Var(z_i)
    mean_total = mu_mean                                   # E[z_i]

    # --- PCA (for plots & grid)
    pca = PCA(n_components=min(10, D), svd_solver='auto', random_state=42)
    latent_pca = pca.fit_transform(mu_all)                 # PCA of means for viz
    pca_expl = pca.explained_variance_ratio_

    # --- TSNE (on PCA-10 for stability / speed)
    n = mu_all.shape[0]
    if tsne_on_pca:
        tsne_input = latent_pca
    else:
        tsne_input = mu_all
    # clamp perplexity
    perpl = max(5, min(30, n - 1, n // 4 if n >= 24 else 5))
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perpl, init='pca', learning_rate='auto')
        latent_tsne = tsne.fit_transform(tsne_input)
    except Exception as e:
        print(f"[t-SNE warning] {e} — falling back to first two PCs.")
        latent_tsne = None

    # --- KL decomposition (Gaussian assumption): KL(q(z)||N(0,I)) = MI + TC + DW
    # Total KL with full Σ_agg and mean_total
    ekl = _mean_ekl_diag_or_lowrank(mu_all, logvar_all, lowrank_all)  # E_x KL(q(z|x)||N(0,I))

    # Dimension-wise KL: sum_i KL(q(z_i)||N(0,1)) with Var(z_i)=var_total_i, mean=mean_total_i
    dw_kl = 0.5 * np.sum(var_total + mean_total**2 - 1.0 - np.log(np.clip(var_total, jitter_eps, None)))

    # Total correlation: -0.5 * log det( R ),  R = D^{-1/2} Σ D^{-1/2}
    Dinv2 = np.diag(1.0 / np.sqrt(np.clip(var_total, jitter_eps, None)))
    R = Dinv2 @ Sigma_agg @ Dinv2
    R = 0.5 * (R + R.T) + jitter_eps * np.eye(D)
    _, logdetR = np.linalg.slogdet(R)
    tc = -0.5 * logdetR

    mi = float(ekl - tc - dw_kl)

    # --- Visualization
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    # dataset palette (avoid shadowing)
    palette = ['turquoise', 'lightcoral', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    ds_unique = sorted(set(ds_list))
    ds_to_color = {ds: palette[i % len(palette)] for i, ds in enumerate(ds_unique)}
    ds_mask = {ds: np.array([d == ds for d in ds_list]) for ds in ds_unique}

    # (0,0) PCA scatter
    for ds in ds_unique:
        m = ds_mask[ds]
        axes[0,0].scatter(latent_pca[m,0], latent_pca[m,1], s=20, alpha=0.6, c=ds_to_color[ds], label=ds)
    axes[0,0].set_title('PCA of Latent Means (by dataset)')
    axes[0,0].set_xlabel(f'PC1 ({pca_expl[0]:.1%})'); axes[0,0].set_ylabel(f'PC2 ({pca_expl[1]:.1%})'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    # (0,1) t-SNE
    if latent_tsne is not None:
        for ds in ds_unique:
            m = ds_mask[ds]
            axes[0,1].scatter(latent_tsne[m,0], latent_tsne[m,1], s=20, alpha=0.6, c=ds_to_color[ds], label=ds)
        axes[0,1].set_title(f't-SNE (perplexity={perpl})'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)
    else:
        for ds in ds_unique:
            m = ds_mask[ds]
            axes[0,1].scatter(latent_pca[m,0], latent_pca[m,1], s=20, alpha=0.6, c=ds_to_color[ds], label=ds)
        axes[0,1].set_title('Fallback: PC1 vs PC2'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    # (0,2) per-dim variance (from Σ_agg diagonal)
    axes[0,2].bar(range(D), var_total)
    axes[0,2].set_title('Variance per Latent (Σ_agg diag)'); axes[0,2].set_xlabel('dim'); axes[0,2].grid(alpha=0.3)

    # (1,0) per-dim mean
    axes[1,0].bar(range(D), mean_total)
    axes[1,0].set_title('Mean per Latent'); axes[1,0].set_xlabel('dim'); axes[1,0].grid(alpha=0.3)

    # (1,1) PC1 distribution by dataset
    for ds in ds_unique:
        m = ds_mask[ds]
        if m.any():
            axes[1,1].hist(latent_pca[m,0], bins=30, alpha=0.6, density=True, label=ds, color=ds_to_color[ds])
    axes[1,1].axvline(0, color='k', ls='--', alpha=0.7)
    axes[1,1].set_title('PC1 Distribution'); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

    # (1,2) correlation of first 10 raw dims using Σ_agg
    k = min(10, D)
    Sigma_k = Sigma_agg[:k,:k]
    d = np.sqrt(np.clip(np.diag(Sigma_k), jitter_eps, None))
    Rk = (Sigma_k / d[:,None]) / d[None,:]
    im = axes[1,2].imshow(Rk, vmin=-1, vmax=1, cmap='RdBu_r')
    axes[1,2].set_title('Correlation (first 10 dims)'); fig.colorbar(im, ax=axes[1,2], label='corr')

    # (2,0) per-dim KL
    per_dim_kl = 0.5 * (var_total + mean_total**2 - 1.0 - np.log(np.clip(var_total, jitter_eps, None)))
    axes[2,0].bar(range(D), per_dim_kl)
    axes[2,0].axhline(0.05, color='r', ls='--', alpha=0.7, label='0.05 nats'); axes[2,0].legend()
    axes[2,0].set_title('Per-dim KL'); axes[2,0].grid(alpha=0.3)

    # (2,1) eigen-spectrum
    evals = np.linalg.eigvalsh(Sigma_agg)
    evals = np.clip(evals, jitter_eps, None)
    axes[2,1].bar(range(D), np.sort(evals)[::-1])
    axes[2,1].set_yscale('log'); axes[2,1].set_title('Eigenvalues of Σ_agg (log)'); axes[2,1].grid(alpha=0.3)

    # (2,2) KL decomposition
    metrics = ['Mutual\nInformation', 'Total\nCorrelation', 'Dimension-wise\nKL', 'Total KL']
    vals = [mi, tc, dw_kl, ekl]
    bars = axes[2,2].bar(metrics, vals, color=['skyblue','lightcoral','lightgreen','gold'])
    axes[2,2].set_ylabel('nats'); axes[2,2].set_title('KL = MI + TC + DW')
    for b, v in zip(bars, vals):
        axes[2,2].text(b.get_x() + b.get_width()/2., v * 1.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.show()

    # --- PCA grid decode
    print("\n🎮 Generating TTY visualization grid using PCA space...")
    W = pca.components_[:2]                   # [2,D]
    num_per_axis = 5
    q = torch.distributions.Normal(0,1).icdf(torch.linspace(0.01, 0.99, num_per_axis))
    XX, YY = torch.meshgrid(q, q, indexing='ij')
    grid_2d = torch.stack((XX, YY)).reshape(2, -1).T.numpy()     # [25,2]
    latent_grid = grid_2d @ W + pca.mean_[None, :]               # [25,D]
    latent_grid = torch.tensor(latent_grid, dtype=torch.float32, device=device)

    with torch.no_grad():
        dec = model.sample(z=latent_grid)
        if 'ego_chars' in dec and 'ego_colors' in dec:
            ego_chars = dec['ego_chars'].detach().cpu()
            ego_colors = dec['ego_colors'].detach().cpu()
        else:
            print("⚠️ sample() lacks ego chars/colors; using placeholders.")
            ego_chars  = torch.full((latent_grid.shape[0], 8, 8), 32)
            ego_colors = torch.zeros_like(ego_chars)

    fig2, axs = plt.subplots(num_per_axis, num_per_axis, figsize=(20, 20))
    fig2.suptitle('Generated Ego Views from PCA Grid (5x5)', fontsize=16, y=0.98)
    import numpy as np
    imgs = [
        _render_map_image(ego_chars[i], ego_colors[i],
                          font_path="DejaVuSansMono.ttf", font_size=12, bg=(0,0,0))
        for i in range(min(len(ego_chars), num_per_axis**2))
    ]
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        ax.axis("off")
        if i < len(imgs):
            ax.imshow(imgs[i], interpolation="nearest")
    plt.tight_layout()
    tty_save_path = save_path.replace('.png', '_tty_grid.png')
    plt.figure(fig2.number); plt.savefig(tty_save_path, dpi=200, bbox_inches='tight'); plt.show()

    # --- reporting
    print(f"\n📊 Enhanced Latent Space Analysis")
    print(f"  - Samples: {total} ({got_train} train, {got_test} test), D={D}")
    print(f"  - KL total={ekl:.3f}, MI={mi:.3f}, TC={tc:.3f}, DW={dw_kl:.3f}")
    print(f"  - PCA first 5 explained: {pca_expl[:5]} (cum {np.cumsum(pca_expl[:5])})")

    return {
        'mu': mu_all,
        'Sigma_agg': Sigma_agg,
        'mean': mu_mean,
        'var_diag': var_total,
        'pca_components': latent_pca,
        'pca_model': pca,
        'tsne_components': latent_tsne,
        'metrics': {'kl_total': float(ekl), 'mi': float(mi), 'tc': float(tc), 'dw_kl': float(dw_kl)},
        'dataset_labels': ds_list,
        'plot_path': save_path,
        'tty_grid_path': tty_save_path,
    }



def create_visualization_demo(
    repo_name: str,
    train_dataset: Optional[List[Dict]] = None,
    test_dataset: Optional[List[Dict]] = None,
    revision_name: Optional[str] = None,
    token: Optional[str] = None,
    device: str = "cpu",
    num_samples: int = 4,
    max_latent_samples: int = 100,
    save_dir: str = "vae_analysis",
    random_sampling: bool = True,
    random_seed: Optional[int] = None,
    # VAE sampling parameters
    use_mean: bool = True,
    include_logits: bool = False,
    # Map sampling parameters
    map_temperature: float = 1.0,
    map_occ_thresh: float = 0.5,
    bag_presence_thresh: float = 0.5,
    hero_presence_thresh: float = 0.5,
    passability_thresh: float = 0.5,
    safety_thresh: float = 0.5,
    map_deterministic: bool = True,
    glyph_top_k: int = 0,
    glyph_top_p: float = 1.0,
    color_top_k: int = 0,
    color_top_p: float = 1.0,
    class_top_k: int = 0,
    class_top_p: float = 1.0,
    # Message sampling parameters
    msg_temperature: float = 1.0,
    msg_top_k: int = 0,
    msg_top_p: float = 1.0,
    msg_deterministic: bool = True,
    allow_eos: bool = True,
    forbid_eos_at_start: bool = True,
    allow_pad: bool = False
) -> Dict:
    """
    Complete demo function that loads a model from HuggingFace and creates visualizations
    
    Args:
        repo_name: HuggingFace repository name
        train_dataset: Training dataset from NetHackDataCollector (optional)
        test_dataset: Test dataset from NetHackDataCollector (optional)
        token: HuggingFace token (optional)
        device: Device to run on
        num_samples: Number of reconstruction samples
        max_latent_samples: Maximum samples for latent analysis
        save_dir: Directory to save results
        random_sampling: Whether to use random sampling for reconstruction visualization
        random_seed: Random seed for reproducible sampling
        
        # VAE sampling parameters
        use_mean: If True, use mean of latent distribution; if False, sample from it
        include_logits: Whether to include raw logits in output
        
        # Map sampling parameters (legacy parameters map to new ones)
        map_occ_thresh: Threshold for occupancy prediction
        rare_occ_thresh: Threshold for rare occupancy prediction
        hero_presence_thresh: Threshold for hero presence prediction
        map_temperature: Temperature for map sampling (legacy: temperature)
        glyph_top_k: Top-k filtering for glyph sampling (legacy: top_k)
        glyph_top_p: Top-p filtering for glyph sampling (legacy: top_p)
        map_deterministic: If True, use deterministic sampling for map
        color_top_k: Top-k filtering for color sampling
        color_top_p: Top-p filtering for color sampling
        
        # Message sampling parameters
        msg_temperature: Temperature for message token sampling
        msg_top_k: Top-k filtering for message sampling
        msg_top_p: Top-p filtering for message sampling
        msg_deterministic: If True, use deterministic sampling for messages
        allow_eos: Whether to allow end-of-sequence tokens
        forbid_eos_at_start: Whether to forbid EOS tokens at start
        allow_pad: Whether to allow padding tokens
        
    Returns:
        Dictionary with analysis results
    """
    # Validate inputs
    if train_dataset is None and test_dataset is None:
        raise ValueError("At least one of train_dataset or test_dataset must be provided")
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 Starting VAE Analysis Demo")
    print(f"📦 Repository: {repo_name}")
    print(f"🎯 Device: {device}")
    print(f"📁 Save directory: {save_dir}")
    print(f"🎲 Random sampling: {random_sampling}")
    if random_seed is not None:
        print(f"🌱 Random seed: {random_seed}")
    
    # Load model from HuggingFace with local fallback
    print(f"\n1️⃣ Loading model from HuggingFace...")
    model = None
    
    try:
        model = load_model_from_huggingface(repo_name, token=token, device=device, revision_name=revision_name)
    except Exception as e:
        print(f"⚠️  Failed to load from HuggingFace: {e}")
        print(f"🔄 Attempting to load from local checkpoints...")
        
        # Try to find the latest local checkpoint
        checkpoint_dir = "checkpoints"
        local_checkpoint_path = None
        
        if os.path.exists(checkpoint_dir):
            # Find the latest checkpoint file
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoint_files:
                # Sort by modification time, latest first
                checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                local_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                print(f"📁 Found latest checkpoint: {local_checkpoint_path}")
            else:
                print(f"❌ No checkpoint files found in {checkpoint_dir}")
        
        # Also check for a saved model file
        if local_checkpoint_path is None:
            potential_paths = [
                "models/nethack-vae.pth",
                "nethack-vae.pth",
                "model.pth"
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    local_checkpoint_path = path
                    print(f"📁 Found saved model: {local_checkpoint_path}")
                    break
        
        if local_checkpoint_path is not None:
            try:
                model = load_model_from_local(local_checkpoint_path, device=device)
                print(f"✅ Successfully loaded model from local checkpoint")
            except Exception as local_e:
                print(f"❌ Failed to load from local checkpoint: {local_e}")
                raise RuntimeError(f"Failed to load model from both HuggingFace ({e}) and local checkpoint ({local_e})")
        else:
            print(f"❌ No local checkpoints found")
            raise RuntimeError(f"Failed to load model from HuggingFace ({e}) and no local checkpoints available")

    results = {'model': model, 'save_dir': save_dir}
    
    # Create TTY reconstructions for available datasets
    if train_dataset is not None:
        print(f"\n2️⃣ Creating TTY reconstruction visualizations for TRAINING dataset...")
        train_save_path = "train_recon_comparison.md"
        train_recon_results = visualize_reconstructions(
            model, train_dataset, device, 
            num_samples=num_samples, 
            out_dir=save_dir, 
            save_path=train_save_path,
            img_file_prefix="train_",
            random_sampling=random_sampling,
            dataset_name="Training",
            # VAE sampling parameters
            use_mean=use_mean,
            include_logits=include_logits,
            # Map sampling parameters (map legacy params)
            map_temperature=map_temperature,  # Legacy: temperature -> map_temperature
            map_occ_thresh=map_occ_thresh,
            bag_presence_thresh=bag_presence_thresh,
            hero_presence_thresh=hero_presence_thresh,
            passability_thresh=passability_thresh,
            safety_thresh=safety_thresh,
            map_deterministic=map_deterministic,
            glyph_top_k=glyph_top_k,  # Legacy: top_k -> glyph_top_k
            glyph_top_p=glyph_top_p,  # Legacy: top_p -> glyph_top_p
            color_top_k=color_top_k,
            color_top_p=color_top_p,
            class_top_k=class_top_k,
            class_top_p=class_top_p,
            # Message sampling parameters
            msg_temperature=msg_temperature,
            msg_top_k=msg_top_k,
            msg_top_p=msg_top_p,
            msg_deterministic=msg_deterministic,
            allow_eos=allow_eos,
            forbid_eos_at_start=forbid_eos_at_start,
            allow_pad=allow_pad
        )
        results['train_reconstruction_path'] = os.path.join(save_dir, train_save_path)
        results['train_reconstruction_results'] = train_recon_results
    
    if test_dataset is not None:
        print(f"\n2️⃣ Creating TTY reconstruction visualizations for TESTING dataset...")
        test_save_path = "test_recon_comparison.md"
        test_recon_results = visualize_reconstructions(
            model, test_dataset, device, 
            num_samples=num_samples, 
            out_dir=save_dir, 
            save_path=test_save_path,
            img_file_prefix="test_",
            random_sampling=random_sampling,
            dataset_name="Testing",
            # VAE sampling parameters
            use_mean=use_mean,
            include_logits=include_logits,
            # Map sampling parameters (map legacy params)
            map_temperature=map_temperature,  # Legacy: temperature -> map_temperature
            map_occ_thresh=map_occ_thresh,
            bag_presence_thresh=bag_presence_thresh,
            hero_presence_thresh=hero_presence_thresh,
            passability_thresh=passability_thresh,
            safety_thresh=safety_thresh,
            map_deterministic=map_deterministic,
            glyph_top_k=glyph_top_k,  # Legacy: top_k -> glyph_top_k
            glyph_top_p=glyph_top_p,  # Legacy: top_p -> glyph_top_p
            color_top_k=color_top_k,
            color_top_p=color_top_p,
            class_top_k=class_top_k,
            class_top_p=class_top_p,
            # Message sampling parameters
            msg_temperature=msg_temperature,
            msg_top_k=msg_top_k,
            msg_top_p=msg_top_p,
            msg_deterministic=msg_deterministic,
            allow_eos=allow_eos,
            forbid_eos_at_start=forbid_eos_at_start,
            allow_pad=allow_pad
        )
        results['test_reconstruction_path'] = os.path.join(save_dir, test_save_path)
        results['test_reconstruction_results'] = test_recon_results

    # Analyze latent space (use combined dataset or available one)
    print(f"\n3️⃣ Analyzing latent space...")
    
    # Combine datasets for latent analysis or use what's available
    analysis_datasets = []
    dataset_labels = []
    
    if train_dataset is not None:
        analysis_datasets.extend(train_dataset)
        dataset_labels.extend(['train'] * len(train_dataset))
    
    if test_dataset is not None:
        analysis_datasets.extend(test_dataset)
        dataset_labels.extend(['test'] * len(test_dataset))
    
    latent_path = os.path.join(save_dir, "latent_analysis.png")
    latent_analysis = analyze_latent_space(
        model, analysis_datasets, device, 
        save_path=latent_path, 
        max_samples=max_latent_samples,
        dataset_labels=dataset_labels
    )
    
    results['latent_analysis_path'] = latent_path
    results['latent_analysis'] = latent_analysis
    
    print(f"\n✅ Analysis complete! Results saved to: {save_dir}")
    if train_dataset is not None:
        print(f"📄 Training TTY reconstructions: {results['train_reconstruction_path']}")
    if test_dataset is not None:
        print(f"📄 Testing TTY reconstructions: {results['test_reconstruction_path']}")
    print(f"📊 Latent analysis plot: {latent_path}")
    
    return results

def analyze_glyph_char_color_pairs(
    dataset: List[Dict],
    top_k: int = 50,
    save_dir: str = "bin_count_analysis",
    save_plot: bool = True,
    show_ascii_chars: bool = True,
    save_complete_data: bool = True
) -> Dict:
    """
    Analyze the distribution of (glyph_char, glyph_color) pairs in the dataset.
    
    Args:
        dataset: List of data batches from NetHackDataCollector
        top_k: Number of top pairs to display
        save_dir: Directory to save analysis results
        save_plot: Whether to save the plot
        show_ascii_chars: Whether to show ASCII character representations
        save_complete_data: Whether to save complete count data to JSON
        
    Returns:
        Dictionary with analysis results
    """
    print(f"🔍 Starting glyph (char, color) pair analysis...")
    print(f"📊 Dataset size: {len(dataset)} batches")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Counter for (char, color) pairs
    pair_counter = Counter()
    total_cells = 0
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(dataset, desc="Processing batches")):
        game_chars = batch['game_chars']  # Shape: (num_games, num_time, 21, 79)
        game_colors = batch['game_colors']  # Shape: (num_games, num_time, 21, 79)
        
        # Flatten the spatial and temporal dimensions
        chars_flat = game_chars.flatten()  # All character codes
        colors_flat = game_colors.flatten()  # All color codes
        
        # Count pairs
        for char, color in zip(chars_flat.tolist(), colors_flat.tolist()):
            pair_counter[(char, color)] += 1
            total_cells += 1
    
    print(f"📈 Total cells analyzed: {total_cells:,}")
    print(f"🎨 Unique (char, color) pairs found: {len(pair_counter):,}")
    
    # Save complete count data to JSON if requested
    if save_complete_data:
        # Create readable format for pairs
        readable_pairs = {}
        for (char, color), count in pair_counter.items():
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            pair_key = f"({char},{color})"
            readable_pairs[pair_key] = {
                'char_code': char,
                'color_code': color,
                'ascii_char': ascii_repr,
                'count': count,
                'percentage': (count / total_cells) * 100
            }
        
        # Create readable format for characters
        char_counter = Counter()
        for (char, color), count in pair_counter.items():
            char_counter[char] += count
        
        readable_chars = {}
        for char, count in char_counter.items():
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            readable_chars[str(char)] = {
                'char_code': char,
                'ascii_char': ascii_repr,
                'total_count': count,
                'percentage': (count / total_cells) * 100
            }
        
        complete_data = {
            'total_cells': total_cells,
            'unique_pairs': len(pair_counter),
            'pair_counts': readable_pairs,
            'char_counts': readable_chars,
            'analysis_metadata': {
                'dataset_size': len(dataset),
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'glyph_char_color_pairs'
            }
        }
        
        # Save complete data
        complete_data_path = os.path.join(save_dir, "complete_bin_counts.json")
        with open(complete_data_path, 'w') as f:
            json.dump(complete_data, f, indent=2)
        print(f"💾 Complete count data saved to: {complete_data_path}")
    
    # Get top k pairs (excluding space character pairs)
    filtered_pairs = [(key, count) for key, count in pair_counter.items() if key[0] != 32]
    top_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Convert to readable format
    pair_data = []
    for (char, color), count in top_pairs:
        ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
        percentage = (count / total_cells) * 100
        pair_data.append({
            'char_code': char,
            'color_code': color,
            'ascii_char': ascii_repr,
            'count': count,
            'percentage': percentage
        })
    
    # Print top pairs
    print(f"\n🏆 Top {top_k} (char, color) pairs (excluding spaces):")
    print("-" * 80)
    if show_ascii_chars:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'ASCII':<8} {'Count':<12} {'Percentage':<10}")
    else:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'Count':<12} {'Percentage':<10}")
    print("-" * 80)
    
    for i, data in enumerate(pair_data):
        if show_ascii_chars:
            ascii_str = f"'{data['ascii_char']}'"
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{ascii_str:<8} {data['count']:<12,} {data['percentage']:<10.2f}%")
        else:
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{data['count']:<12,} {data['percentage']:<10.2f}%")
    
    # Create visualization
    if save_plot and len(top_pairs) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Bar chart of top pairs
        pair_labels = []
        counts = []
        colors_for_plot = []
        
        # Color mapping for NetHack colors (0-15)
        nethack_colors = [
            '#000000',  # 0: black
            '#800000',  # 1: red  
            '#008000',  # 2: green
            '#808000',  # 3: yellow
            '#000080',  # 4: blue
            '#800080',  # 5: magenta
            '#008080',  # 6: cyan
            '#C0C0C0',  # 7: white
            '#808080',  # 8: gray
            '#ff0000',  # 9: bright red
            '#00ff00',  # 10: bright green
            '#ffff00',  # 11: bright yellow
            '#0000ff',  # 12: bright blue
            '#ff00ff',  # 13: bright magenta
            '#00ffff',  # 14: bright cyan
            '#ffffff'   # 15: bright white
        ]
        
        for (char, color), count in top_pairs:
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            pair_labels.append(f"'{ascii_repr}' ({char}, {color})")
            counts.append(count)
            # Use actual NetHack color if available, otherwise use default
            if 0 <= color < len(nethack_colors):
                colors_for_plot.append(nethack_colors[color])
            else:
                colors_for_plot.append('#808080')  # Default gray
        
        bars = ax1.bar(range(len(counts)), counts, color=colors_for_plot, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('(Character, Color) Pairs')
        ax1.set_ylabel('Count (log scale)')
        ax1.set_yscale('log')
        ax1.set_title(f'Top {top_k} Most Frequent (Glyph Char, Glyph Color) Pairs (Excluding Spaces)')
        ax1.set_xticks(range(len(pair_labels)))
        ax1.set_xticklabels(pair_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Character distribution (top chars only, excluding space)
        char_counter = Counter()
        for (char, color), count in pair_counter.items():
            if char != 32:  # Exclude space character
                char_counter[char] += count
        
        top_chars = char_counter.most_common(20)  # Top 20 characters
        char_codes = [char for char, _ in top_chars]
        char_counts = [count for _, count in top_chars]
        char_labels = [f"'{chr(char)}' ({char})" if 32 <= char <= 126 else f"\\x{char:02x} ({char})" 
                      for char in char_codes]
        
        bars2 = ax2.bar(range(len(char_counts)), char_counts, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Character Codes')
        ax2.set_ylabel('Total Count (log scale)')
        ax2.set_yscale('log')
        ax2.set_title('Top 20 Most Frequent Characters (All Colors Combined, Excluding Spaces)')
        ax2.set_xticks(range(len(char_labels)))
        ax2.set_xticklabels(char_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars2, char_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(save_dir, f"glyph_char_color_analysis_top_{top_k}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()
    
    # Save detailed results to JSON
    results = {
        'total_cells': total_cells,
        'unique_pairs': len(pair_counter),
        'top_pairs': pair_data,
        'analysis_params': {
            'top_k': top_k,
            'dataset_size': len(dataset),
            'show_ascii_chars': show_ascii_chars
        }
    }
    
    if save_plot:
        results_path = os.path.join(save_dir, "glyph_analysis_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_path}")
    
    return results


def plot_glyph_char_color_pairs_from_saved(
    data_path: str,
    top_k: int = 50,
    save_dir: str = None,
    save_plot: bool = True,
    show_ascii_chars: bool = True,
    exclude_space: bool = True
) -> Dict:
    """
    Load saved bin count data and create visualizations.
    
    Args:
        data_path: Path to the saved complete_bin_counts.json file
        top_k: Number of top pairs to display
        save_dir: Directory to save plots (if None, uses directory of data_path)
        save_plot: Whether to save the plot
        show_ascii_chars: Whether to show ASCII character representations
        exclude_space: Whether to exclude space character (ASCII 32) from analysis
        
    Returns:
        Dictionary with analysis results
    """
    print(f"📥 Loading saved bin count data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Saved data file not found: {data_path}")
    
    # Load the complete data
    with open(data_path, 'r') as f:
        complete_data = json.load(f)
    
    total_cells = complete_data['total_cells']
    unique_pairs = complete_data['unique_pairs']
    
    print(f"📈 Loaded data: {total_cells:,} total cells, {unique_pairs:,} unique pairs")
    
    # Convert pair_counts back to Counter format
    pair_counter = Counter()
    for pair_key, pair_data in complete_data['pair_counts'].items():
        char = pair_data['char_code']
        color = pair_data['color_code']
        count = pair_data['count']
        pair_counter[(char, color)] = count
    
    # Set save directory
    if save_dir is None:
        save_dir = os.path.dirname(data_path)
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter pairs if excluding space
    if exclude_space:
        filtered_pairs = [(key, count) for key, count in pair_counter.items() if key[0] != 32]
        print(f"🚫 Excluding space character pairs")
    else:
        filtered_pairs = list(pair_counter.items())
    
    # Get top k pairs
    top_pairs = sorted(filtered_pairs, key=lambda x: x[1], reverse=True)[:top_k]
    
    # Convert to readable format
    pair_data = []
    for (char, color), count in top_pairs:
        ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
        percentage = (count / total_cells) * 100
        pair_data.append({
            'char_code': char,
            'color_code': color,
            'ascii_char': ascii_repr,
            'count': count,
            'percentage': percentage
        })
    
    # Print top pairs
    exclude_text = " (excluding spaces)" if exclude_space else ""
    print(f"\n🏆 Top {top_k} (char, color) pairs{exclude_text}:")
    print("-" * 80)
    if show_ascii_chars:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'ASCII':<8} {'Count':<12} {'Percentage':<10}")
    else:
        print(f"{'Rank':<4} {'Char':<6} {'Color':<5} {'Count':<12} {'Percentage':<10}")
    print("-" * 80)
    
    for i, data in enumerate(pair_data):
        if show_ascii_chars:
            ascii_str = f"'{data['ascii_char']}'"
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{ascii_str:<8} {data['count']:<12,} {data['percentage']:<10.2f}%")
        else:
            print(f"{i+1:<4} {data['char_code']:<6} {data['color_code']:<5} "
                  f"{data['count']:<12,} {data['percentage']:<10.2f}%")
    
    # Create visualization
    if save_plot and len(top_pairs) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Bar chart of top pairs
        pair_labels = []
        counts = []
        colors_for_plot = []
        
        # Color mapping for NetHack colors (0-15)
        nethack_colors = [
            '#000000',  # 0: black
            '#800000',  # 1: red  
            '#008000',  # 2: green
            '#808000',  # 3: yellow
            '#000080',  # 4: blue
            '#800080',  # 5: magenta
            '#008080',  # 6: cyan
            '#C0C0C0',  # 7: white
            '#808080',  # 8: gray
            '#ff0000',  # 9: bright red
            '#00ff00',  # 10: bright green
            '#ffff00',  # 11: bright yellow
            '#0000ff',  # 12: bright blue
            '#ff00ff',  # 13: bright magenta
            '#00ffff',  # 14: bright cyan
            '#ffffff'   # 15: bright white
        ]
        
        for (char, color), count in top_pairs:
            ascii_repr = chr(char) if 32 <= char <= 126 else f"\\x{char:02x}"
            pair_labels.append(f"'{ascii_repr}' ({char}, {color})")
            counts.append(count)
            # Use actual NetHack color if available, otherwise use default
            if 0 <= color < len(nethack_colors):
                colors_for_plot.append(nethack_colors[color])
            else:
                colors_for_plot.append('#808080')  # Default gray
        
        bars = ax1.bar(range(len(counts)), counts, color=colors_for_plot, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('(Character, Color) Pairs')
        ax1.set_ylabel('Count (log scale)')
        ax1.set_yscale('log')
        title_suffix = " (Excluding Spaces)" if exclude_space else ""
        ax1.set_title(f'Top {top_k} Most Frequent (Glyph Char, Glyph Color) Pairs{title_suffix}')
        ax1.set_xticks(range(len(pair_labels)))
        ax1.set_xticklabels(pair_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Character distribution (top chars only, optionally excluding space)
        char_counter = Counter()
        for char_str, char_data in complete_data['char_counts'].items():
            char = char_data['char_code']
            count = char_data['total_count']
            if not exclude_space or char != 32:
                char_counter[char] = count
        
        top_chars = char_counter.most_common(20)  # Top 20 characters
        char_codes = [char for char, _ in top_chars]
        char_counts = [count for _, count in top_chars]
        char_labels = [f"'{chr(char)}' ({char})" if 32 <= char <= 126 else f"\\x{char:02x} ({char})" 
                      for char in char_codes]
        
        bars2 = ax2.bar(range(len(char_counts)), char_counts, color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Character Codes')
        ax2.set_ylabel('Total Count (log scale)')
        ax2.set_yscale('log')
        char_title_suffix = " (Excluding Spaces)" if exclude_space else ""
        ax2.set_title(f'Top 20 Most Frequent Characters (All Colors Combined{char_title_suffix})')
        ax2.set_xticks(range(len(char_labels)))
        ax2.set_xticklabels(char_labels, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars2, char_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_plot:
            plot_suffix = "_no_space" if exclude_space else ""
            plot_path = os.path.join(save_dir, f"glyph_char_color_analysis_top_{top_k}{plot_suffix}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_path}")
        
        plt.show()
    
    # Return results
    results = {
        'total_cells': total_cells,
        'unique_pairs': unique_pairs,
        'top_pairs': pair_data,
        'analysis_params': {
            'top_k': top_k,
            'exclude_space': exclude_space,
            'show_ascii_chars': show_ascii_chars,
            'data_source': data_path
        },
        'metadata': complete_data.get('analysis_metadata', {})
    }
    
    return results


@torch.no_grad()
def _collect_sample_rasters(model, dataset, device, hmm: StickyHDPHMMVI, max_sequences: int = 6, max_batches: int = 4):
    """
    Collect a few sequences' responsibility argmax per time step for raster plots.
    Returns: (rasters: List[np.ndarray], lengths: List[int])
    """
    rasters, lengths = [], []
    grabbed = 0
    for bi, batch in enumerate(dataset):
        if grabbed >= max_sequences or bi >= max_batches: break
        # Move to device and keep [B,T,...] shapes
        batch_dev = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_dev[k] = v.to(device, non_blocking=True)
            else:
                batch_dev[k] = v
        if 'game_chars' not in batch_dev: 
            continue
        B, T = batch_dev['game_chars'].shape[:2]
        with torch.no_grad():
            # Flatten to [B*T,...] for model forward, then reshape back
            flat = {}
            for k,v in batch_dev.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 3 and k not in ('original_batch_shape',):
                    flat[k] = v.view(B*T, *v.shape[2:])
                else:
                    flat[k] = v
            out = model(flat)
            mu_bt  = out['mu'].view(B, T, -1)
            var_bt = out['logvar'].exp().clamp_min(1e-6).view(B, T, -1)
            F_btr  = out.get('lowrank_factors', None)
            F_bt   = None if F_btr is None else F_btr.view(B, T, F_btr.size(-2), F_btr.size(-1))
            valid  = flat['valid_screen'].view(B, T).bool()

            logB  = hmm.expected_emission_loglik(mu_bt, var_bt, F_bt, mask=valid)  # [B,T,K]
            log_pi = torch.log(torch.clamp(hmm._Epi(), min=1e-30))
            ElogA = hmm._ElogA()
            
            # Process each sequence in the batch separately since forward_backward expects [T,K]
            for b in range(B):
                if grabbed >= max_sequences: break
                m = valid[b].cpu().numpy().astype(bool)
                if m.sum() == 0: 
                    continue
                
                # Extract single sequence and run forward_backward
                logB_single = logB[b]  # [T,K]
                r_hat_single, _, _ = hmm.forward_backward(log_pi, ElogA, logB_single)  # [T,K]
                
                # turn into argmax labels on valid frames only
                labels = r_hat_single.argmax(-1).cpu().numpy()  # [T]
                labels = labels[m]
                rasters.append(labels)
                lengths.append(len(labels))
                grabbed += 1
        if grabbed >= max_sequences:
            break
    return rasters, lengths

@torch.no_grad()
def compute_hmm_diagnostics(model, dataset, device, hmm: StickyHDPHMMVI, max_batches: int = 5, logger=None):
    """
    Compute HMM diagnostics on a subset of the dataset
    """
    model.eval()
    n_batches = min(len(dataset), max_batches)
    B, T = dataset[0]['tty_chars'].shape[:2]
    
    all_mu = []
    all_var = []
    all_F = []
    all_mask = []
    
    # Collect VAE outputs for diagnostics
    with torch.no_grad():
        for bi, batch in enumerate(dataset):
            if bi >= n_batches: 
                break
                
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    x = v.to(device, non_blocking=True)
                    if x.dim() >= 3 and k not in ('original_batch_shape',):
                        batch_dev[k] = x.view(B*T, *x.shape[2:])
                    else:
                        batch_dev[k] = x
                else:
                    batch_dev[k] = v
            
            # Forward encode only
            out = model(batch_dev)
            mu = out['mu']
            logvar = out['logvar']
            F = out.get('lowrank_factors', None)
            
            valid = batch_dev['valid_screen'].view(B, T)
            
            # Reshape to [B,T,...]
            mu_bt = mu.view(B, T, -1)
            var_bt = logvar.exp().clamp_min(1e-6).view(B, T, -1)
            F_bt = None if F is None else F.view(B, T, F.size(-2), F.size(-1))
            
            all_mu.append(mu_bt)
            all_var.append(var_bt)
            if F_bt is not None:
                all_F.append(F_bt)
            all_mask.append(valid)
    
    # Concatenate all data
    mu_concat = torch.cat(all_mu, dim=0)  # [total_B, T, D]
    var_concat = torch.cat(all_var, dim=0)
    F_concat = torch.cat(all_F, dim=0) if all_F else None
    mask_concat = torch.cat(all_mask, dim=0)
    
    # Compute diagnostics using HMM's built-in function
    diagnostics = hmm.diagnostics(
        mu_t=mu_concat,
        diag_var_t=var_concat, 
        F_t=F_concat,
        mask=mask_concat
    )
    
    return diagnostics

@torch.no_grad()
def visualize_hmm_after_estep(
    model, dataset, device, hmm: StickyHDPHMMVI, save_dir: str, round_idx: int,
    logger=None, max_diags_batches: int = 5, max_raster_sequences: int = 6
):
    """
    Produce a small set of plots summarizing the HMM after the E-step.
    Saves figures under {save_dir}/round_{round_idx:02d}/ and returns a dict of paths.
    """
    round_dir = os.path.join(save_dir, f"round_{round_idx:02d}")
    os.makedirs(round_dir, exist_ok=True)

    # 1) quick diagnostics and basic tensors
    diags = compute_hmm_diagnostics(model, dataset, device, hmm, max_batches=max_diags_batches, logger=logger)
    pi_hat = diags["occupancy_pi_hat"].cpu().numpy()
    A_bar  = torch.softmax(hmm._ElogA(), dim=1).cpu().numpy()  # proxy for E[A]
    mu_k, E_Lambda, _ = hmm.get_emission_expectations()        # [K,D], [K,D,D], [K]
    mu_k = mu_k.cpu().numpy()

    # 2) figure: occupancy bar
    f1 = plt.figure(figsize=(8, 3))
    xs = np.arange(len(pi_hat))
    plt.bar(xs, pi_hat)
    plt.xlabel("Skill k")
    plt.ylabel("Occupancy $\hat{\\pi}_k$")
    plt.title(f"Round {round_idx}: Skill occupancy (effK={diags['effective_K']:.2f})")
    plt.tight_layout()
    path_pi = os.path.join(round_dir, f"round{round_idx:02d}_pi_bar.png")
    f1.savefig(path_pi, dpi=160); plt.close(f1)

    # 3) figure: transition heatmap
    f2 = plt.figure(figsize=(6, 5))
    plt.imshow(A_bar, origin="lower", aspect="auto")
    plt.colorbar(label="Softmax(ElogA)")
    plt.xlabel("Next state")
    plt.ylabel("Current state")
    diag_mean = float(np.mean(np.diag(A_bar)))
    plt.title(f"Round {round_idx}: Transition matrix (diag mean={diag_mean:.3f})")
    plt.tight_layout()
    path_A = os.path.join(round_dir, f"round{round_idx:02d}_A_heatmap.png")
    f2.savefig(path_A, dpi=160); plt.close(f2)

    # 4) figure: PCA of mu_k (+ optional ellipses from E[Lambda]^{-1})
    try:
        from sklearn.decomposition import PCA
        comps = min(2, mu_k.shape[1]); pca = PCA(n_components=comps).fit(mu_k)
        xy = pca.transform(mu_k)
        f3 = plt.figure(figsize=(6, 5))
        plt.scatter(xy[:,0], xy[:,1], s=50*(pi_hat / (pi_hat.max()+1e-8) + 0.2))
        for i in range(mu_k.shape[0]):
            # small 1σ ellipse from covariance approx inv(E_Lambda)
            try:
                cov = np.linalg.inv(E_Lambda[i].cpu().numpy())
                cov = 0.5*(cov + cov.T)
                w, V = np.linalg.eigh(cov)
                w = np.clip(w, 1e-8, None)
                angle = math.degrees(math.atan2(V[1, -1], V[0, -1]))
                r1, r2 = np.sqrt(w[-1]), np.sqrt(w[-2]) if len(w) > 1 else np.sqrt(w[-1])
                from matplotlib.patches import Ellipse
                e = Ellipse(xy[i], 2*r1, 2*r2, angle=angle, fill=False, alpha=0.4)
                ax = plt.gca(); ax.add_patch(e)
            except Exception:
                pass
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.title(f"Round {round_idx}: PCA of emission means (size ∝ $\\hat\\pi_k$)")
        plt.tight_layout()
        path_pca = os.path.join(round_dir, f"round{round_idx:02d}_mu_pca.png")
        f3.savefig(path_pca, dpi=160); plt.close(f3)
    except Exception as e:
        path_pca = None

    # 5) figure: skill rasters on a few sequences
    rasters, lengths = _collect_sample_rasters(model, dataset, device, hmm, max_sequences=max_raster_sequences)
    if len(rasters) > 0:
        maxT = max(lengths)
        canvas = np.full((len(rasters), maxT), fill_value=-1, dtype=np.int32)
        for i, lab in enumerate(rasters):
            canvas[i, :len(lab)] = lab
        f4 = plt.figure(figsize=(10, 1.2 + 0.35*len(rasters)))
        plt.imshow(canvas, interpolation="nearest", aspect="auto", origin="lower", cmap="tab20")
        plt.colorbar(label="Skill id", fraction=0.025, pad=0.02)
        plt.xlabel("t (valid frames)"); plt.ylabel("sequence #")
        plt.title(f"Round {round_idx}: Skill raster (argmax responsibilities)")
        plt.tight_layout()
        path_raster = os.path.join(round_dir, f"round{round_idx:02d}_skill_raster.png")
        f4.savefig(path_raster, dpi=160); plt.close(f4)
    else:
        path_raster = None

    # Save a small JSON snapshot for quick diffing
    snap = {
        "round": round_idx,
        "avg_loglik_per_step": float(diags["avg_loglik_per_step"]),
        "state_entropy": float(diags["state_entropy"]),
        "effective_K": float(diags["effective_K"]),
        "stickiness_diag_mean": float(diags["stickiness_diag_mean"]),
        "top5_pi": [float(x) for x in diags["top5_pi"]],
        "top5_idx": [int(x) for x in diags["top5_idx"]],
    }
    json_path = os.path.join(round_dir, f"round{round_idx:02d}_diags.json")
    with open(json_path, "w") as f:
        json.dump(snap, f, indent=2)

    return {
        "dir": round_dir,
        "pi_bar": path_pi,
        "A_heatmap": path_A,
        "mu_pca": path_pca,
        "skill_raster": path_raster,
        "diags_json": json_path,
    }