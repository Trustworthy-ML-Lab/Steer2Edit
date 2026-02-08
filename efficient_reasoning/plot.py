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
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 26

RUN_JSON = "3_runs.json"

# --- MODEL DISPLAY NAME MAPPING ---
# Maps the short folder name (arg) to the formal Figure Title
MODEL_DISPLAY_NAMES = {
    "qwen3-4b-thinking": "Qwen3-4B-Thinking-2507",
    "nemotron-7b": "OpenMath-Nemotron-7B"
}

# =============================================================================
#  METRIC READERS (Mean + Std)
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

def read_run_metrics(path: str):
    """Returns ((acc, acc_std), (len, len_std))"""
    if not os.path.isfile(path):
        return (None, 0.0), (None, 0.0)
    
    # Accuracy
    acc_stats = read_metric_pair(path, 
                                 ["mean_accuracy", "accuracy", "acc"], 
                                 ["std_accuracy", "std_acc"])
    
    # Length (Efficiency)
    len_stats = read_metric_pair(path, 
                                 ["mean_length", "avg_length", "mean_reasoning_length", "length"], 
                                 ["std_length", "std_reasoning_length", "std_len"])
    
    return acc_stats, len_stats

def avg_metrics_stats(metrics_dict: Dict[str, Tuple[Optional[float], float]]) -> Tuple[Optional[float], float]:
    """Averages means and stds across datasets."""
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
    if step == 0: return [round(start, 6)]
    out = []
    x = start
    while x <= end + 1e-9:
        out.append(round(x, 6))
        x += step
    return out

def _grid_set(ra_rng, rm_rng, a_rng):
    ras = _frange(*ra_rng)
    if rm_rng[0] == -1.0 and rm_rng[1] == -1.0: rms = [-1.0]
    else: rms = _frange(*rm_rng)
    als = _frange(*a_rng)
    s = set()
    for ra in ras:
        for rm in rms:
            for a in als:
                s.add((round(ra,6), round(rm,6), round(a,6)))
    return s

QWEN_GRID = _grid_set((-1.0, -1.0, 0.0), (0.65, 0.80, 0.05), (0.05, 0.20, 0.05)) # qwen3-4b-thinking
NEMO_GRID = _grid_set((0.2, 0.3, 0.05), (0.8, 0.9, 0.05), (0.1, 0.2, 0.05))      # nemotron-7b

def allowed_grid_for_model(model):
    ml = model.lower()
    if "qwen" in ml: return QWEN_GRID
    if "nemotron" in ml: return NEMO_GRID
    return set()

def strengths_for_model(model):
    ml = model.lower()
    if "qwen" in ml: 
        return [round(0.1 * i, 2) for i in range(1, 14)] # 1..13
    if "nemotron" in ml: 
        return [round(0.1 * i, 2) for i in range(1, 13)] # 1..12
    return [round(0.1 * i, 2) for i in range(1, 16)]

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

# --- FOLDER FINDER (Handles Exact & Lowercase) ---
def find_folder(root, name):
    if not os.path.isdir(root): return None
    # 1. Exact match
    cand = os.path.join(root, name)
    if os.path.isdir(cand): return cand
    # 2. Lowercase match
    cand_lower = os.path.join(root, name.lower())
    if os.path.isdir(cand_lower): return cand_lower
    return None

def baseline_folder(base, dataset, model, strength, decimals):
    root = os.path.join(base, dataset)
    
    if abs(strength) < 1e-12:
        return find_folder(root, model)
    
    for v in strength_variants(strength, decimals):
        # Try both exact and lowercased pattern
        # Pattern: {model}_steering_{v}_attnmlp
        name_exact = f"{model}_steering_{v}_attnmlp"
        path = os.path.join(root, name_exact)
        if os.path.isdir(path): return path
        
        name_lower = f"{model.lower()}_steering_{v}_attnmlp"
        path = os.path.join(root, name_lower)
        if os.path.isdir(path): return path
        
    return None

def rank1_root_folder(base, dataset, model):
    start = os.path.join(base, dataset, "edited_models")
    return find_folder(start, model)

# =============================================================================
#  SCANNERS
# =============================================================================

def get_original_data(base, datasets, model, decimals):
    acc_vals = {}
    len_vals = {}
    
    for ds in datasets:
        f = baseline_folder(base, ds, model, 0.0, decimals)
        path = os.path.join(f, RUN_JSON) if f else None
        
        if not path:
            print(f"[warn] Original data missing for {model} on {ds}")
        
        a_stat, l_stat = read_run_metrics(path) if path else ((None,0.0), (None,0.0))
        acc_vals[ds] = a_stat
        len_vals[ds] = l_stat
        
    avg_acc, std_acc = avg_metrics_stats(acc_vals)
    avg_len, std_len = avg_metrics_stats(len_vals)
    
    return avg_acc, std_acc, avg_len, std_len

def get_steering_data(base, datasets, model, strengths, decimals):
    points = []
    valid_s = []
    # Pre-scan
    for s in strengths:
        f = baseline_folder(base, datasets[0], model, s, decimals)
        if f: valid_s.append(s)
    
    if not valid_s:
        print(f"[warn] No steering folders found for {model}")

    for s in valid_s:
        acc_vals = {}
        len_vals = {}
        for ds in datasets:
            f = baseline_folder(base, ds, model, s, decimals)
            path = os.path.join(f, RUN_JSON) if f else None
            a_stat, l_stat = read_run_metrics(path) if path else ((None,0.0), (None,0.0))
            acc_vals[ds] = a_stat
            len_vals[ds] = l_stat
            
        avg_acc, std_acc = avg_metrics_stats(acc_vals)
        avg_len, std_len = avg_metrics_stats(len_vals)
        
        if avg_acc is not None and avg_len is not None:
            points.append({
                'acc': avg_acc, 'acc_std': std_acc,
                'len': avg_len, 'len_std': std_len,
                'strength': s
            })
    return points

def get_rank1_data(base, datasets, model, topk=10):
    allowed = allowed_grid_for_model(model)
    
    # 1. Scan one dataset
    root = rank1_root_folder(base, datasets[0], model)
    if not root:
        print(f"[warn] Rank1 folder not found for {model} in {datasets[0]}")
        return []
    
    candidates = []
    for entry in os.listdir(root):
        key = parse_rank1_key(entry)
        if key and key in allowed:
            candidates.append(key)
    
    if not candidates:
        print(f"[warn] No valid rank1 keys found in grid for {model}")

    # 2. Compute Avg Accuracy
    scored = []
    for key in candidates:
        folder_name = rank1_folder(key)
        acc_vals = {}
        for ds in datasets:
            ds_root = rank1_root_folder(base, ds, model)
            if not ds_root: continue
            
            path = os.path.join(ds_root, folder_name, RUN_JSON)
            a_stat, _ = read_run_metrics(path)
            acc_vals[ds] = a_stat
        
        avg_a, _ = avg_metrics_stats(acc_vals)
        if avg_a is not None:
            scored.append((key, avg_a))
            
    # 3. Top K
    scored.sort(key=lambda x: x[1], reverse=True)
    top_keys = [x[0] for x in scored[:topk]]
    
    # 4. Fetch Full Data
    points = []
    for key in top_keys:
        folder_name = rank1_folder(key)
        acc_vals = {}
        len_vals = {}
        for ds in datasets:
            ds_root = rank1_root_folder(base, ds, model)
            if not ds_root: continue
            
            path = os.path.join(ds_root, folder_name, RUN_JSON)
            a_stat, l_stat = read_run_metrics(path)
            acc_vals[ds] = a_stat
            len_vals[ds] = l_stat
            
        avg_acc, std_acc = avg_metrics_stats(acc_vals)
        avg_len, std_len = avg_metrics_stats(len_vals)
        
        if avg_acc is not None:
             points.append({
                'acc': avg_acc, 'acc_std': std_acc,
                'len': avg_len, 'len_std': std_len,
                'key': key
            })
    return points

# =============================================================================
#  PLOTTING
# =============================================================================

def get_display_name(model_key: str) -> str:
    """Uses the global mapping or falls back to capitalized key."""
    return MODEL_DISPLAY_NAMES.get(model_key, model_key.replace("-", " ").capitalize())

def plot_final_tradeoff(plot_data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    models = list(plot_data.keys())
    
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 7), sharey=False)
    if len(models) == 1: axes = [axes]
    
    # --- STYLES ---
    style_orig = {"color": "#444444", "marker": "^", "s": 350, "label": "Original", "zorder": 10, "edgecolor": "white", "linewidth": 1.5}
    color_steer = "#95a5a6" 
    style_steer_pt = {"color": color_steer, "marker": "o", "s": 80, "label": "Steering Vector", "zorder": 5, "alpha": 0.9}
    style_ours = {"color": "#d62728", "marker": "*", "s": 450, "label": "Steer2Edit (Ours)", "zorder": 15, "edgecolor": "white", "linewidth": 0.8}

    err_kwargs = {'fmt': 'none', 'ecolor': '#d62728', 'elinewidth': 1.5, 'alpha': 0.5, 'capsize': 0, 'zorder': 14}
    orig_err_kwargs = {'fmt': 'none', 'ecolor': '#444444', 'elinewidth': 1.5, 'alpha': 0.5, 'capsize': 0, 'zorder': 14}
    steer_err_kwargs = {'fmt': 'none', 'ecolor': color_steer, 'elinewidth': 1.5, 'alpha': 0.5, 'capsize': 0, 'zorder': 4.5}

    for ax, model in zip(axes, models):
        d = plot_data[model]
        
        # 1. STEERING
        steer = d['steering']
        if steer:
            # Sort by Strength to show trajectory
            steer.sort(key=lambda x: x['strength']) 
            
            accs = np.array([p['acc'] for p in steer])
            lens = np.array([p['len'] for p in steer])
            a_err = np.array([p['acc_std'] for p in steer])
            l_err = np.array([p['len_std'] for p in steer])
            
            ax.fill_between(accs, lens - l_err, lens + l_err, color=color_steer, alpha=0.2, zorder=3, linewidth=0)
            ax.plot(accs, lens, color=color_steer, linestyle='--', linewidth=3, alpha=0.6, zorder=4)
            ax.errorbar(accs, lens, xerr=a_err, yerr=l_err, **steer_err_kwargs)
            ax.scatter(accs, lens, **style_steer_pt)

        # 2. ORIGINAL
        orig = d['original']
        if orig and orig['acc'] is not None:
            ax.errorbar(orig['acc'], orig['len'], xerr=orig['acc_std'], yerr=orig['len_std'], **orig_err_kwargs)
            ax.scatter([orig['acc']], [orig['len']], **style_orig)

        # 3. STEER2EDIT
        ours = d['rank1']
        if ours:
            accs = [p['acc'] for p in ours]
            lens = [p['len'] for p in ours]
            a_err = [p['acc_std'] for p in ours]
            l_err = [p['len_std'] for p in ours]
            
            ax.errorbar(accs, lens, xerr=a_err, yerr=l_err, **err_kwargs)
            ax.scatter(accs, lens, **style_ours)

        # Titles & Labels
        display_name = get_display_name(model)
        ax.set_title(display_name, fontweight='bold', pad=15)
        ax.set_xlabel("Average Accuracy", fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Invert Y-axis for Efficiency
        ax.invert_yaxis()
        
        if model == models[0]:
            ax.set_ylabel("Reasoning Length ($\\downarrow$)", fontweight='bold')

    # Main Title
    fig.suptitle("Reasoning Efficiency - Accuracy Trade-off", fontsize=30, fontweight='bold', y=0.98)

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
    plt.subplots_adjust(top=0.83, bottom=0.24, wspace=0.2)
    
    out_file = os.path.join(out_dir, "tradeoff_final_efficiency.pdf")
    print(f"Saving to {out_file}...")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.savefig(out_file.replace(".pdf", ".png"), dpi=200, bbox_inches='tight')

# =============================================================================
#  MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="evaluate_results")
    ap.add_argument("--models", default="qwen3-4b-thinking,nemotron-7b") # Short names for disk
    ap.add_argument("--datasets", default="MATH-500,gpqa,code-mmlu,gsm8k")
    ap.add_argument("--baseline_decimals", type=int, default=1)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--plot_dir", default="plots_tradeoff")
    args = ap.parse_args()
    
    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    decimals = args.baseline_decimals
    
    data_map = {}
    
    for m in models:
        print(f"Processing {m}...")
        
        # Determine Strengths for this model
        strengths = strengths_for_model(m)
        
        # Original
        acc, acc_s, length, len_s = get_original_data(args.base, datasets, m, decimals)
        orig_data = {'acc': acc, 'acc_std': acc_s, 'len': length, 'len_std': len_s}
        
        # Steering
        steer_data = get_steering_data(args.base, datasets, m, strengths, decimals)
        
        # Ours
        rank1_data = get_rank1_data(args.base, datasets, m, topk=args.topk)
        
        data_map[m] = {
            'original': orig_data,
            'steering': steer_data,
            'rank1': rank1_data
        }
        
    plot_final_tradeoff(data_map, args.plot_dir)

if __name__ == "__main__":
    main()