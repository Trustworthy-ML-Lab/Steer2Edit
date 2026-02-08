#!/usr/bin/env python3
import os
import re
import json
import argparse
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# --- VISUAL STYLE CONFIGURATION ---
# Exact style settings from your Safety plot code
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 26

TRUTH_JSON = "1_run.json"
UTILITY_JSON = "10_runs.json"

# =============================================================================
#  METRIC READERS
# =============================================================================

def read_metric_pair(json_path: str, key_mean_list: List[str], key_std_list: List[str]) -> Tuple[Optional[float], float]:
    """Returns (mean, std)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agg = data.get("aggregate", {})
        
        val_mean = None
        for k in key_mean_list:
            if k in agg:
                val_mean = float(agg[k])
                break
        
        if val_mean is None:
            return None, 0.0

        val_std = 0.0
        for k in key_std_list:
            if k in agg:
                val_std = float(agg[k])
                break
        
        return val_mean, val_std
    except Exception:
        return None, 0.0

def read_accuracy_with_std(path):
    # For Utility (10_runs.json) - Standard deviation IS available
    return read_metric_pair(path, ["mean_accuracy", "accuracy", "acc"], ["std_accuracy", "std_acc"])

def read_accuracy_no_std(path):
    # For TruthfulQA (1_run.json) - No standard deviation available
    val, _ = read_metric_pair(path, ["mean_accuracy", "accuracy", "acc"], [])
    return val, 0.0

def avg_metrics(metrics_dict: Dict[str, Tuple[Optional[float], float]]) -> Tuple[Optional[float], float]:
    means = [m for m, s in metrics_dict.values() if m is not None]
    stds = [s for m, s in metrics_dict.values() if m is not None]
    
    if not means:
        return None, 0.0
    
    avg_mean = sum(means) / len(means)
    avg_std = sum(stds) / len(stds) 
    return avg_mean, avg_std

# =============================================================================
#  GRID & PATH HELPERS
# =============================================================================

RANK1_RX = re.compile(r"^rank1__ra(?P<ra>[-0-9.]+)_rm(?P<rm>[-0-9.]+)_a(?P<a>[-0-9.]+)\.pt$")

def _frange(start, end, step):
    out = []
    x = start
    while x <= end + 1e-9:
        out.append(round(x, 6))
        x += step
    return out

def _grid_set(ra_rng, rm_rng, a_rng):
    ras = _frange(*ra_rng)
    als = _frange(*a_rng)
    if rm_rng[0] == -1.0: rms = [-1.0]
    else: rms = _frange(*rm_rng)
    s = set()
    for ra in ras:
        for rm in rms:
            for a in als:
                s.add((round(ra,6), round(rm,6), round(a,6)))
    return s

# Specific grids for Truthfulness models
GEMMA_GRID = _grid_set((0.30, 0.50, 0.05), (-1.0, -1.0, 1.0), (0.75, 0.95, 0.05))
LLAMA3_GRID = _grid_set((0.10, 0.14, 0.01), (0.30, 0.50, 0.05), (0.30, 0.70, 0.10))

def allowed_grid_for_model(model):
    ml = model.lower()
    if "gemma" in ml: return GEMMA_GRID
    if "llama" in ml: return LLAMA3_GRID
    return set()

def parse_rank1_key(folder):
    m = RANK1_RX.match(folder)
    if not m: return None
    return (round(float(m.group("ra")), 6), round(float(m.group("rm")), 6), round(float(m.group("a")), 6))

def rank1_folder(key):
    return f"rank1__ra{key[0]}_rm{key[1]}_a{key[2]}.pt"

def strength_variants(s, decimals):
    fixed = f"{s:.{decimals}f}"
    minimal = fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    out = []
    for v in (fixed, minimal, f"{s:.1f}", f"{s:.2f}"):
        if v not in out: out.append(v)
    return out

def baseline_folder(base, dataset, model, strength, decimals):
    root = os.path.join(base, dataset)
    if not os.path.isdir(root): return None
    if abs(strength) < 1e-12:
        f = os.path.join(root, model)
        return f if os.path.isdir(f) else None
    for v in strength_variants(strength, decimals):
        f = os.path.join(root, f"{model}_steering_{v}_attnmlp")
        if os.path.isdir(f): return f
    return None

# =============================================================================
#  SCANNERS
# =============================================================================

def get_original_data(base, truth_ds, util_ds, model, decimals):
    # Truth (1_run.json)
    f_t = baseline_folder(base, truth_ds, model, 0.0, decimals)
    t_val, t_std = read_accuracy_no_std(os.path.join(f_t, TRUTH_JSON)) if f_t and os.path.isfile(os.path.join(f_t, TRUTH_JSON)) else (None, 0.0)

    # Utility (10_runs.json)
    u_vals = {}
    for ds in util_ds:
        f = baseline_folder(base, ds, model, 0.0, decimals)
        path = os.path.join(f, UTILITY_JSON) if f else None
        u_vals[ds] = read_accuracy_with_std(path) if path and os.path.isfile(path) else (None, 0.0)
    
    avg_u, std_u = avg_metrics(u_vals)
    return avg_u, std_u, t_val, t_std

def get_steering_data(base, truth_ds, util_ds, model, strengths, decimals):
    points = []
    # Pre-scan truth to find valid strengths
    valid_s = []
    for s in strengths:
        f = baseline_folder(base, truth_ds, model, s, decimals)
        if f and os.path.isfile(os.path.join(f, TRUTH_JSON)):
            valid_s.append(s)

    for s in valid_s:
        # Truth (No Std)
        f_t = baseline_folder(base, truth_ds, model, s, decimals)
        t_val, t_std = read_accuracy_no_std(os.path.join(f_t, TRUTH_JSON))
        
        # Utility (Has Std)
        u_vals = {}
        for ds in util_ds:
            f = baseline_folder(base, ds, model, s, decimals)
            path = os.path.join(f, UTILITY_JSON) if f else None
            u_vals[ds] = read_accuracy_with_std(path) if path and os.path.isfile(path) else (None, 0.0)
            
        avg_u, std_u = avg_metrics(u_vals)
        
        if avg_u is not None and t_val is not None:
            points.append({'u': avg_u, 'u_std': std_u, 't': t_val, 't_std': t_std, 'strength': s})
    return points

def get_rank1_data(base, truth_ds, util_ds, model, topk=10):
    grid_truth = {} 
    allowed = allowed_grid_for_model(model)
    
    # Scan TruthfulQA
    root = os.path.join(base, truth_ds, "edited_models", model)
    if os.path.isdir(root):
        for entry in os.listdir(root):
            key = parse_rank1_key(entry)
            if key and key in allowed:
                full = os.path.join(root, entry, TRUTH_JSON)
                if os.path.isfile(full):
                    val, _ = read_accuracy_no_std(full)
                    if val is not None:
                        grid_truth[key] = val

    # Top K by Truth
    valid_keys = [(k, v) for k, v in grid_truth.items()]
    valid_keys.sort(key=lambda x: x[1], reverse=True)
    top_keys = [x[0] for x in valid_keys[:topk]]
    
    points = []
    for key in top_keys:
        t_val = grid_truth[key]
        
        u_vals = {}
        folder_name = rank1_folder(key)
        for ds in util_ds:
            f = os.path.join(base, ds, "edited_models", model, folder_name, UTILITY_JSON)
            u_vals[ds] = read_accuracy_with_std(f) if os.path.isfile(f) else (None, 0.0)
        
        avg_u, std_u = avg_metrics(u_vals)
        if avg_u is not None:
            points.append({'u': avg_u, 'u_std': std_u, 't': t_val, 't_std': 0.0, 'key': key})
    return points

# =============================================================================
#  PLOTTING
# =============================================================================

def get_display_name(model_key: str) -> str:
    mk = model_key.lower()
    if "gemma" in mk:
        return "Gemma-2-2B-IT"
    if "llama" in mk:
        return "LLaMA-3-8B-Instruct"
    return model_key.capitalize()

def plot_final_tradeoff(plot_data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    models = list(plot_data.keys())
    
    # Figure size matches Safety plot logic
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 7), sharey=False)
    if len(models) == 1: axes = [axes]
    
    # --- STYLES ---
    style_orig = {"color": "#444444", "marker": "^", "s": 350, "label": "Original", "zorder": 10, "edgecolor": "white", "linewidth": 1.5}
    color_steer = "#95a5a6" # Greyish Blue
    style_steer_pt = {"color": color_steer, "marker": "o", "s": 80, "label": "Steering Vector", "zorder": 5, "alpha": 0.9}
    style_ours = {"color": "#d62728", "marker": "*", "s": 450, "label": "Steer2Edit (Ours)", "zorder": 15, "edgecolor": "white", "linewidth": 0.8}

    # Error bars for Steer2Edit and Steering (Thinner, lighter)
    err_kwargs = {
        'fmt': 'none', 
        'ecolor': '#d62728', 
        'elinewidth': 1.5,
        'alpha': 0.5,
        'capsize': 0, 
        'zorder': 14
    }
    
    # Steering Error Bars (Horizontal Utility)
    steer_err_kwargs = {
        'fmt': 'none', 
        'ecolor': color_steer, 
        'elinewidth': 1.5, 
        'alpha': 0.5, 
        'capsize': 0, 
        'zorder': 4.5 
    }
    
    # Specific error bar style for Original
    orig_err_kwargs = {
        'fmt': 'none',
        'ecolor': '#444444',
        'elinewidth': 1.5, 
        'alpha': 0.5,
        'capsize': 0, 
        'zorder': 14
    }

    for ax, model in zip(axes, models):
        d = plot_data[model]
        
        # 1. STEERING (Line + Points + Horizontal Error Bars)
        steer = d['steering']
        if steer:
            steer.sort(key=lambda x: x['u'], reverse=True)
            us = np.array([p['u'] for p in steer])
            ts = np.array([p['t'] for p in steer])
            u_err = np.array([p['u_std'] for p in steer])
            
            # Line (No vertical shade per instructions)
            ax.plot(us, ts, color=color_steer, linestyle='--', linewidth=3, alpha=0.6, zorder=4)
            
            # Error Bars (Horizontal only)
            ax.errorbar(us, ts, xerr=u_err, yerr=None, **steer_err_kwargs)
            
            # Points
            ax.scatter(us, ts, **style_steer_pt)

        # 2. ORIGINAL (Point + Horizontal Error Bar)
        orig = d['original']
        if orig and orig['u'] is not None:
            # Only X error bar
            ax.errorbar(orig['u'], orig['t'], xerr=orig['u_std'], yerr=None, **orig_err_kwargs)
            ax.scatter([orig['u']], [orig['t']], **style_orig)

        # 3. STEER2EDIT (Points + Horizontal Error Bars)
        ours = d['rank1']
        if ours:
            us = [p['u'] for p in ours]
            ts = [p['t'] for p in ours]
            u_err = [p['u_std'] for p in ours]
            
            # Only X error bar
            ax.errorbar(us, ts, xerr=u_err, yerr=None, **err_kwargs)
            ax.scatter(us, ts, **style_ours)

        # Titles & Labels (Large Fonts matching Safety plot)
        display_name = get_display_name(model)
        ax.set_title(display_name, fontweight='bold', pad=15)
        ax.set_xlabel("Average Utility (Accuracy)", fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if model == models[0]:
            ax.set_ylabel("Truthfulness (Accuracy)", fontweight='bold')

    # Main Title (Matching Safety plot)
    fig.suptitle("Truthfulness Promotion - Utility Trade-off", fontsize=30, fontweight='bold', y=0.98)

    sns.despine(trim=True)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#444444', markersize=15, label='Original'),
        Line2D([0], [0], color=color_steer, lw=3, linestyle='--', marker='o', markerfacecolor=color_steer, markersize=10, label='Steering Vector'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#d62728', markersize=20, label='Steer2Edit (Ours)')
    ]

    fig.legend(handles=legend_elements, 
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.0), 
               frameon=False)

    plt.tight_layout()
    # EXACT Layout adjustment from Safety plot to prevent overlap
    plt.subplots_adjust(top=0.83, bottom=0.24, wspace=0.2)
    
    out_file = os.path.join(out_dir, "tradeoff_final_truth.pdf")
    print(f"Saving to {out_file}...")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.savefig(out_file.replace(".pdf", ".png"), dpi=200, bbox_inches='tight')

# =============================================================================
#  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="evaluate_results")
    ap.add_argument("--models", default="gemma2-2b,llama3-8b")
    ap.add_argument("--truth_dataset", default="truthfulqa")
    ap.add_argument("--utility_datasets", default="commonsense,code-mmlu,gsm8k")
    ap.add_argument("--plot_dir", default="plots_tradeoff")
    args = ap.parse_args()
    
    models = [m.strip() for m in args.models.split(",")]
    truth_ds = args.truth_dataset
    u_ds = [d.strip() for d in args.utility_datasets.split(",")]
    
    # Strengths for steering 0.5 to 5.0
    strengths = [round(0.5 * i, 2) for i in range(1, 11)]
    
    data_map = {}
    
    for m in models:
        print(f"Processing {m}...")
        
        # Original
        ou, ou_s, ot, ot_s = get_original_data(args.base, truth_ds, u_ds, m, 1)
        orig_data = {'u': ou, 'u_std': ou_s, 't': ot, 't_std': ot_s}
        
        # Steering
        steer_data = get_steering_data(args.base, truth_ds, u_ds, m, strengths, 1)
        
        # Ours (Top 10 by truth)
        rank1_data = get_rank1_data(args.base, truth_ds, u_ds, m, topk=10)
        
        data_map[m] = {
            'original': orig_data,
            'steering': steer_data,
            'rank1': rank1_data
        }
        
    plot_final_tradeoff(data_map, args.plot_dir)

if __name__ == "__main__":
    main()