#!/usr/bin/env python3
import os
import re
import json
import argparse
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# TruthfulQA scan outputs vs utility eval outputs
TRUTH_JSON_NAME = "1_run.json"
UTILITY_JSON_NAME = "10_runs.json"

RANK1_RX = re.compile(r"^rank1__ra(?P<ra>[-0-9.]+)_rm(?P<rm>[-0-9.]+)_a(?P<a>[-0-9.]+)\.pt$")

# ----------------- Metric reader (TruthfulQA) -----------------

def read_truth_metric(json_path: str) -> Optional[float]:
    """
    Read TruthfulQA metric from a run JSON.
    Prefer aggregate["accuracy"] if present (float).
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agg = data.get("aggregate", {}) if isinstance(data, dict) else {}

        v = agg.get("accuracy", None)
        if isinstance(v, (int, float)):
            return float(v)

        return None
    except Exception:
        return None

# ----------------- Grid utilities -----------------

def frange(start: float, end: float, step: float) -> List[float]:
    out = []
    x = start
    while x <= end + 1e-9:
        out.append(round(x, 10))
        x += step
    return out

def grid_set(
    ra_rng: Tuple[float, float, float],
    rm_rng: Tuple[float, float, float],
    a_rng: Tuple[float, float, float],
) -> set:
    ras = frange(*ra_rng)
    alphas = frange(*a_rng)

    # rm disabled convention: rm = -1.0 only
    if rm_rng[0] == -1.0:
        rms = [-1.0]
    else:
        rms = frange(*rm_rng)

    s = set()
    for ra in ras:
        for rm in rms:
            for a in alphas:
                s.add((round(ra, 6), round(rm, 6), round(a, 6)))
    return s

def parse_rank1_tag(tag: str) -> Optional[Tuple[float, float, float]]:
    m = RANK1_RX.match(tag)
    if not m:
        return None
    ra = round(float(m.group("ra")), 6)
    rm = round(float(m.group("rm")), 6)
    a  = round(float(m.group("a")),  6)
    return (ra, rm, a)

def allowed_grid_for_model(model: str) -> set:
    """
    Supported grids:

    gemma2-2b:
      RHO_ATTN_RANGE = 0.2..0.4 step 0.05
      RHO_MLP_RANGE  = -1.0 (disabled)
      ALPHA_RANGE    = 0.9..0.98 step 0.02

    llama3-8b:
      RHO_ATTN_RANGE = 0.06..0.14 step 0.02
      RHO_MLP_RANGE  = 0.3..0.5 step 0.05
      ALPHA_RANGE    = 0.3..0.7 step 0.1
    """

    # -------- gemma2-2b --------
    gemma_ra = (0.30, 0.50, 0.05)
    gemma_rm = (-1.0, -1.0, 1.0)   # disabled
    gemma_a  = (0.75, 0.95, 0.05)

    # -------- llama3-8b --------
    llama3_ra = (0.10, 0.14, 0.01)
    llama3_rm = (0.30, 0.50, 0.05)
    llama3_a  = (0.30, 0.70, 0.10)

    model_l = model.lower()

    if "gemma2-2b" in model_l or "gemma" in model_l:
        return grid_set(gemma_ra, gemma_rm, gemma_a)

    if "llama3-8b" in model_l or "llama-3-8b" in model_l or "llama3" in model_l:
        return grid_set(llama3_ra, llama3_rm, llama3_a)

    # default: be conservative and do NOT silently allow everything
    raise ValueError(f"No grid defined for model: {model}")

# ----------------- Steering strength formatting -----------------

def fmt_strength_variants(x: float) -> List[str]:
    fixed1 = f"{x:.1f}"
    minimal1 = fixed1.rstrip("0").rstrip(".")
    fixed2 = f"{x:.2f}"
    minimal2 = fixed2.rstrip("0").rstrip(".")
    out = []
    for s in (fixed1, minimal1, fixed2, minimal2):
        if s not in out:
            out.append(s)
    return out

# ----------------- Candidate -----------------

@dataclass(frozen=True)
class Candidate:
    model: str
    method: str   # "steering" | "steer2edit"
    tag: str      # steering: numeric str alpha, steer2edit: rank1__...pt
    truth_avg: float
    truth_by_ds: Dict[str, float]

# ----------------- Scanners -----------------

def scan_rank1_candidates_gridlocked(base_eval: str, truth_datasets: List[str], model: str) -> List[Candidate]:
    """
    Scan ONLY configs in the allowed grid for this model.
    Require that the config exists in ALL truth_datasets.
    """
    allowed = allowed_grid_for_model(model)

    per_ds: Dict[str, Dict[str, float]] = {}
    for ds in truth_datasets:
        root = os.path.join(base_eval, ds, "edited_models", model)
        d: Dict[str, float] = {}
        if os.path.isdir(root):
            for entry in os.listdir(root):
                full = os.path.join(root, entry)
                if not os.path.isdir(full):
                    continue
                triple = parse_rank1_tag(entry)
                if triple is None:
                    continue
                if triple not in allowed:
                    continue  # IGNORE outside-grid results

                jp = os.path.join(full, TRUTH_JSON_NAME)
                if not os.path.isfile(jp):
                    continue
                v = read_truth_metric(jp)
                if v is None:
                    continue
                d[entry] = v
        per_ds[ds] = d

    if any(len(per_ds[ds]) == 0 for ds in truth_datasets):
        return []

    common = set(per_ds[truth_datasets[0]].keys())
    for ds in truth_datasets[1:]:
        common &= set(per_ds[ds].keys())
    if not common:
        return []

    out: List[Candidate] = []
    for tag in sorted(common):
        by = {ds: per_ds[ds][tag] for ds in truth_datasets}
        avg = sum(by.values()) / len(by)
        out.append(Candidate(model=model, method="steer2edit", tag=tag, truth_avg=avg, truth_by_ds=by))
    return out

def scan_steering_candidates(base_eval: str, truth_datasets: List[str], model: str, strengths: List[float]) -> List[Candidate]:
    out: List[Candidate] = []
    for s in strengths:
        by: Dict[str, float] = {}
        ok = True
        for ds in truth_datasets:
            found = None
            for sv in fmt_strength_variants(s):
                folder = os.path.join(base_eval, ds, f"{model}_steering_{sv}_attnmlp")
                jp = os.path.join(folder, TRUTH_JSON_NAME)
                if os.path.isfile(jp):
                    found = jp
                    break
            if found is None:
                ok = False
                break
            v = read_truth_metric(found)
            if v is None:
                ok = False
                break
            by[ds] = v

        if not ok:
            continue
        avg = sum(by.values()) / len(by)
        out.append(Candidate(model=model, method="steering", tag=str(s), truth_avg=avg, truth_by_ds=by))
    return out

# ----------------- Utility run helpers -----------------

def expected_original_utility_json(base_eval: str, utility_ds: str, model: str) -> str:
    return os.path.join(base_eval, utility_ds, model, UTILITY_JSON_NAME)

def expected_utility_json(base_eval: str, utility_ds: str, cand: Candidate) -> str:
    if cand.method == "steering":
        alpha = cand.tag
        folder = os.path.join(base_eval, utility_ds, f"{cand.model}_steering_{alpha}_attnmlp")
        return os.path.join(folder, UTILITY_JSON_NAME)
    else:
        folder = os.path.join(base_eval, utility_ds, "edited_models", cand.model, cand.tag)
        return os.path.join(folder, UTILITY_JSON_NAME)

def run_eval_utility(model_arg: str, dataset: str, n_sample: int, tp: int, logs_dir: str, log_prefix: str, extra: List[str]) -> None:
    os.makedirs(logs_dir, exist_ok=True)
    log_prefix = log_prefix.replace("/", "_")
    log_path = os.path.join(logs_dir, f"{log_prefix}.log")
    cmd = [
        "python", "evaluate_utility.py",
        "--model", model_arg,
        "--dataset", dataset,
        "--n_sample", str(n_sample),
        "--tensor_parallel_size", str(tp),
        *extra,
    ]
    print(f"[RUN] {' '.join(cmd)} -> {log_path}")
    with open(log_path, "w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)

def ensure_original_utility(base_eval: str, model: str, utility_ds: str, n_sample: int, tp: int, logs_dir: str) -> None:
    out_json = expected_original_utility_json(base_eval, utility_ds, model)
    if os.path.isfile(out_json):
        print(f"[SKIP][original utility] exists: {out_json}")
        return
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    run_eval_utility(
        model_arg=model,
        dataset=utility_ds,
        n_sample=n_sample,
        tp=tp,
        logs_dir=logs_dir,
        log_prefix=f"{model}__{utility_ds}__original",
        extra=[],
    )

# ----------------- CLI -----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_eval", default="evaluate_results")
    ap.add_argument("--edited_models_root", default="edited_models")  # for steer2edit utility eval
    ap.add_argument("--models", default="gemma2-2b")
    ap.add_argument("--truth_datasets", default="truthfulqa")
    ap.add_argument("--utility_datasets", default="commonsense,code-mmlu,gsm8k")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--strengths", default="0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0")
    ap.add_argument("--n_sample", type=int, default=10)
    ap.add_argument("--tensor_parallel_size", type=int, default=4)
    ap.add_argument("--logs_dir", default="logs_tradeoff")
    ap.add_argument("--out_csv", default="topk_truthful_candidates.csv")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()

# ----------------- Main -----------------

def main():
    args = parse_args()
    base_eval = args.base_eval
    edited_models_root = args.edited_models_root
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    truth_datasets = [d.strip() for d in args.truth_datasets.split(",") if d.strip()]
    utility_datasets = [d.strip() for d in args.utility_datasets.split(",") if d.strip()]
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    topk = args.topk

    selected: List[Candidate] = []

    # -------- Select Top-K by truthfulness (grid-locked for rank1) --------
    for model in models:
        print(f"\n========== MODEL: {model} ==========")

        rank1 = scan_rank1_candidates_gridlocked(base_eval, truth_datasets, model)
        steer = scan_steering_candidates(base_eval, truth_datasets, model, strengths)

        rank1_sorted = sorted(rank1, key=lambda c: c.truth_avg, reverse=True)[:topk]
        steer_sorted = sorted(steer, key=lambda c: c.truth_avg, reverse=True)

        print(f"[select] steer2edit (grid-locked): found {len(rank1)} -> taking {len(rank1_sorted)}")
        for i, c in enumerate(rank1_sorted, 1):
            by = " ".join([f"{ds}={c.truth_by_ds[ds]:.3f}" for ds in truth_datasets])
            print(f"  #{i:02d} avg={c.truth_avg:.3f}  {by}  tag={c.tag}")

        print(f"[select] steering: found {len(steer)} -> taking {len(steer_sorted)}")
        for i, c in enumerate(steer_sorted, 1):
            by = " ".join([f"{ds}={c.truth_by_ds[ds]:.3f}" for ds in truth_datasets])
            print(f"  #{i:02d} avg={c.truth_avg:.3f}  {by}  alpha={c.tag}")


        selected.extend(rank1_sorted)
        selected.extend(steer_sorted)

    # -------- Save summary CSV --------
    try:
        import pandas as pd
        rows = []
        for c in selected:
            row = {"model": c.model, "method": c.method, "tag": c.tag, "truth_avg": c.truth_avg}
            for ds in truth_datasets:
                row[f"truth_{ds}"] = c.truth_by_ds.get(ds, None)
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"\n[write] selection summary -> {args.out_csv}")
    except Exception as e:
        print(f"\n[warn] could not write CSV: {e}")

    if args.dry_run:
        print("\n[dry_run] selection complete; not running utility evaluations.")
        return

    # -------- Ensure original utility exists --------
    for model in models:
        for uds in utility_datasets:
            ensure_original_utility(base_eval, model, uds, args.n_sample, args.tensor_parallel_size, args.logs_dir)

    # -------- Run utility evals for selected configs (skip if exists) --------
    skipped = 0
    ran = 0

    for c in selected:
        for uds in utility_datasets:
            out_json = expected_utility_json(base_eval, uds, c)
            if os.path.isfile(out_json):
                print(f"[SKIP][utility] exists: {out_json}")
                skipped += 1
                continue

            os.makedirs(os.path.dirname(out_json), exist_ok=True)

            if c.method == "steering":
                run_eval_utility(
                    model_arg=c.model,
                    dataset=uds,
                    n_sample=args.n_sample,
                    tp=args.tensor_parallel_size,
                    logs_dir=args.logs_dir,
                    log_prefix=f"{c.model}__{uds}__steering_alpha{c.tag}",
                    extra=["--do_steering", "--alpha", c.tag],
                )
            else:
                model_pt = os.path.join(edited_models_root, c.model, c.tag)  # FILE
                run_eval_utility(
                    model_arg=model_pt,
                    dataset=uds,
                    n_sample=args.n_sample,
                    tp=args.tensor_parallel_size,
                    logs_dir=args.logs_dir,
                    log_prefix=f"{c.model}__{uds}__steer2edit__{c.tag}",
                    extra=[],
                )
            ran += 1

    print(f"\nDone. Utility evals ran={ran}, skipped(existing)={skipped}")

if __name__ == "__main__":
    main()
