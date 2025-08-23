import os
import numpy as np
import torch
from nle.nethack import tty_render
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import math
import datetime as dt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
from typing import List, Tuple, Dict

from src.model import bag_presence_to_glyph_sets, make_pair_bag, MapDecoder

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
                hero_y_coord, hero_x_coord = int(model_inputs['blstats'][0, 0].item()), int(model_inputs['blstats'][0, 1].item())
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
                
                # Create batch structure for model forward pass
                batch_for_model = {
                    'game_chars': batch_device['game_chars'][valid_indices],
                    'game_colors': batch_device['game_colors'][valid_indices],
                    'blstats': batch_device['blstats'][valid_indices],
                    'message_chars': batch_device['message_chars'][valid_indices],
                    'hero_info': batch_device['hero_info'][valid_indices]
                }
                
                model_output = model(batch_for_model)
                
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
        # Use ego data instead of full map chars/colors
        if 'ego_chars' in decode_output and 'ego_colors' in decode_output:
            chars = decode_output['ego_chars'].cpu()
            colors = decode_output['ego_colors'].cpu()
        else:
            print("⚠️ No ego_chars/ego_colors in decode output, creating dummy data")
            chars = torch.full((latent_grid.shape[0], 8, 8), 32)  # 8x8 grid of spaces
            colors = torch.zeros((latent_grid.shape[0], 8, 8))   # Black colors

    # Create TTY grid visualization
    tty_fig, tty_axes = plt.subplots(num_per_axis, num_per_axis, figsize=(20, 20))
    tty_fig.suptitle('Generated NetHack Ego-Centric Views from PCA Latent Space Grid (5x5)', fontsize=16, y=0.98)

    print(f"🎨 Rendering {num_per_axis * num_per_axis} NetHack ego-centric views...")
    n = min(len(chars), num_per_axis * num_per_axis)
    imgs = [
        _render_map_image(chars[i], colors[i],
                          font_path="DejaVuSansMono.ttf", font_size=12, bg=(0,0,0))
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
