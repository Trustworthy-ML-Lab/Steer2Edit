#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Helper: Path Builder
# ----------------------
def build_default_edits_path(model_name: str, rho_attn: float, rho_mlp: float, alpha: float, save_root: str):
    fname = f"rank1__ra{rho_attn}_rm{rho_mlp}_a{alpha}.pt"
    return os.path.join(save_root, model_name, fname)

# ----------------------
# Helper: Data Loader
# ----------------------
def load_model_data(model_short_name, rho_attn, rho_mlp, alpha, save_root):
    path = build_default_edits_path(model_short_name, rho_attn, rho_mlp, alpha, save_root)
    if not os.path.exists(path):
        print(f"❌ Error: File not found for {model_short_name} at {path}")
        return None
    
    print(f"Loading {model_short_name} from: {path}")
    data = torch.load(path, map_location="cpu")
    
    attn_dict = data.get("attn", {})
    mlp_dict = data.get("mlp", {})

    max_layer = 0
    if attn_dict: max_layer = max(max_layer, max(k[0] for k in attn_dict.keys()))
    if mlp_dict:  max_layer = max(max_layer, max(k[0] for k in mlp_dict.keys()))
    num_layers = max_layer + 1

    max_head_idx = max([k[1] for k in attn_dict.keys()] or [0])
    max_neuron_idx = max([k[1] for k in mlp_dict.keys()] or [0])
    
    attn_w = max_head_idx + 1
    mlp_w = max_neuron_idx + 1

    attn_raw = [[] for _ in range(num_layers)]
    mlp_raw = [[] for _ in range(num_layers)]

    for (layer, head), content in attn_dict.items():
        attn_raw[layer].append(content["lambda"])
        
    for (layer, neuron), content in mlp_dict.items():
        mlp_raw[layer].append(content["lambda"])

    return attn_raw, mlp_raw, num_layers, attn_w, mlp_w

# ----------------------
# Helper: Grid Processor (Pos Left / Neg Right) with Top K
# ----------------------
def process_grid(raw_data, rows, cols, top_k=None):
    # If top_k is provided, we constrain the grid width to top_k
    # Otherwise, we use the full capacity (cols)
    effective_width = cols
    if top_k is not None and top_k < cols:
        effective_width = top_k
        
    grid = np.zeros((rows, effective_width))
    
    for i in range(rows):
        row_vals = np.array(raw_data[i])
        if len(row_vals) == 0:
            continue
        
        # --- Top K Logic ---
        # If we need to filter, keep only the top_k largest magnitude edits
        if top_k is not None and len(row_vals) > top_k:
            # Sort by absolute magnitude descending
            idx = np.argsort(-np.abs(row_vals))
            # Keep top k
            row_vals = row_vals[idx[:top_k]]
            
        pos = row_vals[row_vals > 0]
        neg = row_vals[row_vals < 0]
        
        pos = np.sort(pos)[::-1] 
        neg = np.sort(neg)[::-1]
        
        # Fill Left (Positives)
        if len(pos) > 0:
            limit = min(len(pos), effective_width)
            grid[i, :limit] = pos[:limit]
        
        # Fill Right (Negatives)
        if len(neg) > 0:
            limit = min(len(neg), effective_width)
            grid[i, -limit:] = neg[:limit]
            
    return grid, effective_width

# ----------------------
# Helper: Layer Tick Labels
# ----------------------
def apply_layer_ticks(ax_left, ax_right, num_layers, step):
    if step <= 0:
        return
    tick_idx = np.arange(0, num_layers, step)
    if len(tick_idx) == 0:
        return
    tick_pos = tick_idx + 0.5
    tick_labels = [str(i) for i in tick_idx]
    for ax in (ax_left, ax_right):
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels)
        ax.tick_params(axis="y", labelsize=24)


def apply_cbar_ticksize(ax, size: int = 24):
    """Increase colorbar tick label size when a colorbar is present."""
    if not ax.collections:
        return
    cbar = ax.collections[0].colorbar
    if cbar is None:
        return
    cbar.ax.tick_params(labelsize=size)

# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Compare Steer2Edit Heatmaps (2 Models)")
    
    parser.add_argument("--main_title", type=str, default="Edit Component Distribution: Efficient Reasoning")
    parser.add_argument("--top_k", type=int, default=100, help="Plot only the top K neurons by magnitude (default: 5000)")
    parser.add_argument("--layer_tick_step", type=int, default=5, help="Layer tick step on the y-axis (default: 5)")

    parser.add_argument("--model1", type=str, default="qwen3-4b-thinking")
    parser.add_argument("--title1", type=str, default="Qwen3-4B-Thinking-2507")
    parser.add_argument("--ra1", type=float, default=-1.0)
    parser.add_argument("--rm1", type=float, default=0.8)
    parser.add_argument("--a1", type=float, default=0.05)
    
    parser.add_argument("--model2", type=str, default="nemotron-7b")
    parser.add_argument("--title2", type=str, default="OpenMath-Nemotron-7B")
    parser.add_argument("--ra2", type=float, default=0.3)
    parser.add_argument("--rm2", type=float, default=0.9)
    parser.add_argument("--a2", type=float, default=0.2)

    parser.add_argument("--save_root", type=str, default="edited_models")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--filename", type=str, default="edit_distribution_efficiency", help="Output filename prefix")
    args = parser.parse_args()

    # Load Data
    res1 = load_model_data(args.model1, args.ra1, args.rm1, args.a1, args.save_root)
    if not res1: return
    # Process grids (Attn usually small, so top_k not needed/applied; MLP needs it)
    attn_mat1, _ = process_grid(res1[0], res1[2], res1[3]) 
    mlp_mat1, width1 = process_grid(res1[1], res1[2], res1[4], top_k=args.top_k)
    
    res2 = load_model_data(args.model2, args.ra2, args.rm2, args.a2, args.save_root)
    if not res2: return
    attn_mat2, _ = process_grid(res2[0], res2[2], res2[3])
    mlp_mat2, width2 = process_grid(res2[1], res2[2], res2[4], top_k=args.top_k)

    # ----------------------
    # Plotting
    # ----------------------
    sns.set_context("talk", font_scale=1.45)
    
    fig, axes = plt.subplots(2, 2, figsize=(26, 12), 
                             gridspec_kw={'width_ratios': [1, 1.7], 
                                          'height_ratios': [1, 1], 
                                          'wspace': 0.12, 
                                          'hspace': 0.55}) 
    
    plt.subplots_adjust(top=0.82, bottom=0.1) 
    
    cmap = "vlag"
    
    # Subtitle for MLP
    mlp_subtitle = f"MLP Neurons (Top {args.top_k})"

    # --- ROW 1: Model 1 ---
    vmax_a1 = np.max(np.abs(attn_mat1)) if np.any(attn_mat1) else 1.0
    sns.heatmap(attn_mat1, ax=axes[0,0], cmap=cmap, center=0, vmin=-vmax_a1, vmax=vmax_a1,
                cbar=True, cbar_kws={'label': '', 'location': 'left', 'pad': 0.15},
                xticklabels=False, yticklabels=False, antialiased=False)
    apply_cbar_ticksize(axes[0,0])
    
    vmax_m1 = np.max(np.abs(mlp_mat1)) if np.any(mlp_mat1) else 1.0
    sns.heatmap(mlp_mat1, ax=axes[0,1], cmap=cmap, center=0, vmin=-vmax_m1, vmax=vmax_m1,
                cbar=True, cbar_kws={'label': '', 'location': 'right', 'pad': 0.02},
                xticklabels=False, yticklabels=False, antialiased=False)
    apply_cbar_ticksize(axes[0,1])

    axes[0,0].set_title("Attention Heads", fontsize=28, pad=12)
    axes[0,1].set_title(mlp_subtitle, fontsize=28, pad=12)
    apply_layer_ticks(axes[0,0], axes[0,1], res1[2], args.layer_tick_step)
    
    # --- ROW 2: Model 2 ---
    vmax_a2 = np.max(np.abs(attn_mat2)) if np.any(attn_mat2) else 1.0
    sns.heatmap(attn_mat2, ax=axes[1,0], cmap=cmap, center=0, vmin=-vmax_a2, vmax=vmax_a2,
                cbar=True, cbar_kws={'label': '', 'location': 'left', 'pad': 0.15},
                xticklabels=False, yticklabels=False, antialiased=False)
    apply_cbar_ticksize(axes[1,0])
    
    vmax_m2 = np.max(np.abs(mlp_mat2)) if np.any(mlp_mat2) else 1.0
    sns.heatmap(mlp_mat2, ax=axes[1,1], cmap=cmap, center=0, vmin=-vmax_m2, vmax=vmax_m2,
                cbar=True, cbar_kws={'label': '', 'location': 'right', 'pad': 0.02},
                xticklabels=False, yticklabels=False, antialiased=False)
    apply_cbar_ticksize(axes[1,1])

    axes[1,0].set_title("Attention Heads", fontsize=28, pad=12)
    axes[1,1].set_title(mlp_subtitle, fontsize=28, pad=12)
    apply_layer_ticks(axes[1,0], axes[1,1], res2[2], args.layer_tick_step)
    
    # Axis Labels
    axes[0,0].set_ylabel("Layer Index", fontsize=26)
    axes[1,0].set_ylabel("Layer Index", fontsize=26)
    axes[1,0].set_xlabel(r"$\leftarrow$ Pos Edits | Zeros | Neg Edits $\rightarrow$", fontsize=26, labelpad=10)
    axes[1,1].set_xlabel(r"$\leftarrow$ Pos Edits | Zeros | Neg Edits $\rightarrow$", fontsize=26, labelpad=10)

    # Titles & Layout
    plt.suptitle(args.main_title, fontsize=38, fontweight='bold', y=0.985)

    fig.text(0.5, 0.89, args.title1, ha='center', va='center', fontsize=32, fontweight='bold')
    fig.text(0.5, 0.46, args.title2, ha='center', va='center', fontsize=32, fontweight='bold')

    # Save
    out_png = os.path.join(args.output_dir, f"{args.filename}.png")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"✅ Saved PNG: {out_png}")

if __name__ == "__main__":
    main()
