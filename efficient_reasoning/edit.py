#!/usr/bin/env python
import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import MODEL_MAP


# ----------------------
# Utils
# ----------------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2) + eps)


def load_directions(path: str):
    return torch.load(path, map_location="cpu")


def elastic_net_lambda(g: torch.Tensor, rho: float, alpha: float) -> torch.Tensor:
    """Elastic-Net closed-form λ_i = sign(g_i) * max(|g_i|-ρ·α, 0) / (ρ(1-α))"""
    if rho is None or rho <= 0:
        return torch.zeros_like(g)
    abs_g = g.abs()
    num = torch.clamp(abs_g - rho * alpha, min=0.0)
    den = rho * (1.0 - alpha)
    return torch.sign(g) * (num / den)


def build_default_edits_path(model_name: str, rho_attn: float, rho_mlp: float, alpha: float, save_root: str):
    fname = f"rank1__ra{rho_attn}_rm{rho_mlp}_a{alpha}.pt"
    return os.path.join(save_root, model_name, fname)


# ----------------------
# Compute + Save COMPACT (λ, k̂, v̂) factors
# ----------------------
def compute_edits(
    model,
    directions_path: str,
    rho_attn: float = -1.0,
    rho_mlp: float = -1.0,
    alpha: float = 0.95,
    save_path: str = "weight_edits.pt",
):
    """
    Compute compact rank-1 edits and store factors instead of dense ΔW.

    Saved dict structure:
      {
        "attn": {(layer, head): {"lambda": float, "k_hat": FloatTensor[head_dim]}},
        "mlp":  {(layer, neuron): {"lambda": float}},
        "meta": {
          "vhat_attn": {layer: FloatTensor[H]},   # per-layer steering vector (unit)
          "vhat_mlp":  {layer: FloatTensor[H]},
          "hidden_size": H, "num_heads": A, "head_dim": D, "num_neurons": I,
          "rho_attn": ..., "rho_mlp": ..., "alpha": ...
        }
      }
    """
    assert torch.cuda.is_available(), "CUDA is required."
    device = torch.device("cuda:0")
    model.to(device).eval()

    directions = load_directions(directions_path)
    layers = model.model.layers
    num_layers = len(layers)
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or (hidden_size // num_heads)
    num_neurons = model.config.intermediate_size

    attn_factors, mlp_factors = {}, {}
    vhat_attn_per_layer, vhat_mlp_per_layer = {}, {}

    print(f"Computing compact rank-1 factors via Elastic-Net (α={alpha})...")

    for i in tqdm(range(num_layers), desc="Layers", ncols=90):
        lyr = layers[i]
        attn = lyr.self_attn
        mlp = lyr.mlp

        attn_key = f"model.layers[{i}].self_attn.o_proj"
        mlp_key = f"model.layers[{i}].mlp.down_proj"

        # per-layer steering directions (unit)
        v_attn = directions["post"]["direction"][attn_key].to(device)
        v_mlp  = directions["post"]["direction"][mlp_key].to(device)
        vhat_attn = l2_normalize(v_attn)
        vhat_mlp  = l2_normalize(v_mlp)
        vhat_attn_per_layer[i] = vhat_attn.detach().cpu()
        vhat_mlp_per_layer[i]  = vhat_mlp.detach().cpu()

        # dataset means (pos+neg)
        x_mean_attn = 0.5 * (directions["pre"]["pos_mean"][attn_key] +
                       directions["pre"]["neg_mean"][attn_key]).to(device)  # [H]
        x_mean_mlp  = 0.5 * (directions["pre"]["pos_mean"][mlp_key] +
                       directions["pre"]["neg_mean"][mlp_key]).to(device)   # [I]

        # ----- Attention heads -----
        if rho_attn > 0:
            Wo = attn.o_proj.weight.data.to(device)  # [H, H]
            x_mean_heads = x_mean_attn.view(num_heads, head_dim)  # [A, D]
            for j in range(num_heads):
                c0, c1 = j * head_dim, (j + 1) * head_dim
                W_block = Wo[:, c0:c1]                         # [H, D]
                g = F.cosine_similarity(vhat_attn, W_block @ x_mean_heads[j], dim=0)  # scalar in [-1,1]

                lam = elastic_net_lambda(g.view(1), rho_attn, alpha).item()
                if lam == 0.0:
                    continue

                # k̂ in the head subspace
                k_vec = W_block.T @ vhat_attn                 # [D]
                k_norm = k_vec.norm(p=2).item()
                if k_norm == 0.0:
                    continue
                k_hat = (k_vec / k_norm).detach().cpu()       # [D], unit

                attn_factors[(i, j)] = {
                    "lambda": float(lam),
                    "k_hat": k_hat,                            # store on CPU
                }

        # ----- MLP neurons -----
        if rho_mlp > 0:
            Wd = mlp.down_proj.weight.data.to(device)          # [H, I]
            for idx in range(num_neurons):
                W_col = Wd[:, idx]                             # [H]
                g = F.cosine_similarity(vhat_mlp, W_col * x_mean_mlp[idx], dim=0)

                lam = elastic_net_lambda(g.view(1), rho_mlp, alpha).item()
                if lam == 0.0:
                    continue

                # k is scalar; normalization ⇒ sign(k)
                k_scalar = torch.dot(W_col, vhat_mlp).item()
                if k_scalar == 0.0:
                    continue
                lam_eff = float(lam * (1.0 if k_scalar > 0 else -1.0))  # fold sign into lambda

                mlp_factors[(i, idx)] = {
                    "lambda": lam_eff
                }

    edits = {
        "attn": attn_factors,
        "mlp": mlp_factors,
        "meta": {
            "vhat_attn": vhat_attn_per_layer,  # {layer:int -> 1D tensor[H]}
            "vhat_mlp":  vhat_mlp_per_layer,   # {layer:int -> 1D tensor[H]}
            "rho_attn": rho_attn,
            "rho_mlp": rho_mlp,
            "alpha": alpha,
            "format": {
                "attn": "ΔW_block = λ * vhat_attn[layer] @ k_hat^T",
                "mlp":  "Δw_col    = λ * vhat_mlp[layer]"
            }
        },
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(edits, save_path)
    print(f"\n✅ Saved compact edits: {len(attn_factors)} attn heads, {len(mlp_factors)} mlp neurons → {save_path}")


# ----------------------
# CLI
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3-4b-thinking")
    parser.add_argument("--directions_path", type=str, default=None)
    parser.add_argument("--rho_attn", type=float, default=-1.0)
    parser.add_argument("--rho_mlp", type=float, default=-1.0)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--save_root", type=str, default="edited_models")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required."
    assert args.alpha < 1.0, "alpha must be <1 for the Elastic-Net closed form."

    model_path = MODEL_MAP[args.model]
    directions_path = args.directions_path or os.path.join("directions", f"{args.model}_efficient_reasoning_dirs.pt")

    save_path = args.save_path or build_default_edits_path(
        model_name=args.model,
        rho_attn=args.rho_attn,
        rho_mlp=args.rho_mlp,
        alpha=args.alpha,
        save_root=args.save_root,
    )

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()
    compute_edits(model, directions_path, rho_attn=args.rho_attn, rho_mlp=args.rho_mlp, alpha=args.alpha, save_path=save_path)
