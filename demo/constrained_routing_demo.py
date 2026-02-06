#!/usr/bin/env python3
import os
import sys
from typing import Dict, List, Tuple

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from stabilized_mhc import stabilized_rational_chart  # type: ignore


def _read_primitives_yaml(path: str) -> Dict[str, List[str]]:
    """Minimal YAML reader for allowed/forbidden lists (no external deps)."""
    allowed: List[str] = []
    forbidden: List[str] = []
    section = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("allowed_primitives:"):
                section = "allowed"
                continue
            if line.startswith("forbidden_primitives:"):
                section = "forbidden"
                continue
            if line.startswith("- "):
                item = line[2:].strip()
                if section == "allowed":
                    allowed.append(item)
                elif section == "forbidden":
                    forbidden.append(item)
    return {"allowed": allowed, "forbidden": forbidden}


def _build_mapping(allowed: List[str], forbidden: List[str]) -> Tuple[List[str], Dict[str, int]]:
    all_prims = sorted(set(allowed + forbidden))
    mapping = {p: i for i, p in enumerate(all_prims)}
    return all_prims, mapping


def _safe_divide(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.where(den != 0, num / den, torch.zeros_like(num))


def build_masked_routing(primitives_path: str, seed: int = 1234, iters: int = 25) -> Dict[str, object]:
    data = _read_primitives_yaml(primitives_path)
    allowed = data["allowed"]
    forbidden = data["forbidden"]

    prims, mapping = _build_mapping(allowed, forbidden)
    n = len(prims)
    if n < 2:
        raise ValueError("Need at least 2 primitives to build routing matrix.")

    u_dim = (n - 1) ** 2
    torch.manual_seed(seed)
    u = torch.randn(1, u_dim, dtype=torch.float32)
    H = stabilized_rational_chart(u)

    # Pre-mask diagnostics
    row_sums_pre = H.sum(dim=-1)
    col_sums_pre = H.sum(dim=-2)
    row_err_pre = (row_sums_pre - 1.0).abs().max().item()
    col_err_pre = (col_sums_pre - 1.0).abs().max().item()

    # Build mask: forbidden columns -> 0
    forbidden_cols = [mapping[p] for p in forbidden if p in mapping]
    mask = torch.ones_like(H)
    if forbidden_cols:
        mask[:, :, forbidden_cols] = 0.0

    H_masked = H * mask

    # Renormalization logic:
    # Enforce row sums to 1.0 and set target column sums:
    # - forbidden columns: 0.0
    # - allowed columns: N / K (total mass preserved with K allowed columns)
    # This preserves non-negativity and keeps forbidden columns at hard zero.
    allowed_cols = [mapping[p] for p in allowed if p in mapping]
    if not allowed_cols:
        raise ValueError("No allowed primitives found; cannot renormalize.")
    k = len(allowed_cols)
    target_col = torch.zeros(1, 1, n, dtype=H.dtype)
    target_col[:, :, allowed_cols] = float(n) / float(k)

    for _ in range(iters):
        # Row normalization to 1.0
        row_sums = H_masked.sum(dim=-1, keepdim=True)
        H_masked = _safe_divide(H_masked, row_sums)
        H_masked = H_masked * mask

        # Column normalization to target_col
        col_sums = H_masked.sum(dim=-2, keepdim=True)
        scale = _safe_divide(target_col, col_sums)
        H_masked = H_masked * scale
        H_masked = H_masked * mask

        # Hard clamp to preserve non-negativity
        H_masked = torch.clamp(H_masked, min=0.0)

    # Post-mask diagnostics
    row_sums_post = H_masked.sum(dim=-1)
    col_sums_post = H_masked.sum(dim=-2)
    row_err_post = (row_sums_post - 1.0).abs().max().item()
    col_err_post_vs_one = (col_sums_post - 1.0).abs().max().item()
    col_err_post_vs_target = (col_sums_post - target_col.squeeze(0)).abs().max().item()

    max_forbidden = 0.0
    if forbidden_cols:
        max_forbidden = H_masked[:, :, forbidden_cols].max().item()

    return {
        "primitives": prims,
        "mapping": mapping,
        "allowed": allowed,
        "forbidden": forbidden,
        "forbidden_cols": forbidden_cols,
        "H_pre": H,
        "H_post": H_masked,
        "row_err_pre": row_err_pre,
        "col_err_pre": col_err_pre,
        "row_err_post": row_err_post,
        "col_err_post_vs_one": col_err_post_vs_one,
        "col_err_post_vs_target": col_err_post_vs_target,
        "max_forbidden": max_forbidden,
        "target_col": target_col.squeeze(0),
    }


def main() -> None:
    primitives_path = os.path.abspath(
        os.path.join(ROOT_DIR, "..", "synthetic-danger", "demo_outputs", "primitives.yaml")
    )
    result = build_masked_routing(primitives_path)

    print(f"Number of primitives: {len(result['primitives'])}")
    print("Forbidden primitives and column indices:")
    for p in result["forbidden"]:
        print(f"  - {p}: {result['mapping'][p]}")

    max_pre = 0.0
    if result["forbidden_cols"]:
        max_pre = result["H_pre"][:, :, result["forbidden_cols"]].max().item()
    print(f"max(H[:, forbidden_cols]) before mask: {max_pre}")
    print(f"max(H[:, forbidden_cols]) after mask: {result['max_forbidden']}")
    print(f"row sum error before: {result['row_err_pre']}")
    print(f"col sum error before: {result['col_err_pre']}")
    print(f"row sum error after: {result['row_err_post']}")
    print(f"col sum error after (vs 1.0): {result['col_err_post_vs_one']}")
    print(f"col sum error after (vs target): {result['col_err_post_vs_target']}")


if __name__ == "__main__":
    main()
