#!/usr/bin/env python3
import argparse
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
    """Minimal YAML reader for allowed/forbidden/all_primitives lists (no external deps)."""
    allowed: List[str] = []
    forbidden: List[str] = []
    all_primitives: List[str] = []
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
            if line.startswith("all_primitives:"):
                section = "all"
                continue
            if line.startswith("- "):
                item = line[2:].strip()
                if section == "allowed":
                    allowed.append(item)
                elif section == "forbidden":
                    forbidden.append(item)
                elif section == "all":
                    all_primitives.append(item)
    return {"allowed": allowed, "forbidden": forbidden, "all_primitives": all_primitives}


def _build_order(allowed: List[str], forbidden: List[str], all_primitives: List[str]) -> List[str]:
    if all_primitives:
        missing = [p for p in allowed + forbidden if p not in all_primitives]
        if missing:
            raise ValueError(f"all_primitives is missing entries: {missing}")
        return list(all_primitives)
    order: List[str] = []
    for p in allowed:
        if p not in order:
            order.append(p)
    for p in forbidden:
        if p not in order:
            order.append(p)
    return order


def _safe_divide(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.where(den != 0, num / den, torch.zeros_like(num))


def load_primitives(primitives_path: str) -> Dict[str, object]:
    data = _read_primitives_yaml(primitives_path)
    allowed = data["allowed"]
    forbidden = data["forbidden"]
    all_primitives = data["all_primitives"]
    order = _build_order(allowed, forbidden, all_primitives)
    mapping = {p: i for i, p in enumerate(order)}
    return {
        "order": order,
        "mapping": mapping,
        "allowed": allowed,
        "forbidden": forbidden,
    }


def build_masked_routing(primitives_path: str, seed: int = 1234, iters: int = 25) -> Dict[str, object]:
    prim_data = load_primitives(primitives_path)
    allowed = prim_data["allowed"]
    forbidden = prim_data["forbidden"]
    prims = prim_data["order"]
    mapping = prim_data["mapping"]

    n = len(prims)
    if n < 2:
        raise ValueError("Need at least 2 primitives to build routing matrix.")

    u_dim = (n - 1) ** 2
    torch.manual_seed(seed)
    u = torch.randn(1, u_dim, dtype=torch.float32)
    H = stabilized_rational_chart(u)
    assert isinstance(H, torch.Tensor)
    if H.dim() == 2:
        H = H.unsqueeze(0)
    assert H.dim() == 3 and H.shape[1] == n and H.shape[2] == n, (
        f"Expected H shape [B,{n},{n}], got {tuple(H.shape)}"
    )

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
    max_forbidden_pre = 0.0
    if forbidden_cols:
        max_forbidden_pre = H[:, :, forbidden_cols].max().item()

    # Explicit post-conditions
    if forbidden_cols:
        assert max_forbidden == 0.0
    assert H_masked.min().item() >= 0.0
    assert row_err_post <= 1e-6
    assert col_err_post_vs_target <= 1e-6

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
        "max_forbidden_pre": max_forbidden_pre,
        "target_col": target_col.squeeze(0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primitives", required=True, help="Path to primitives.yaml")
    args = ap.parse_args()

    primitives_path = os.path.abspath(args.primitives)
    result = build_masked_routing(primitives_path)

    print(f"primitives.yaml: {primitives_path}")
    print(f"H.shape before mask: {tuple(result['H_pre'].shape)}")
    print(f"H.shape after mask: {tuple(result['H_post'].shape)}")
    print("Primitive order:")
    for i, p in enumerate(result["primitives"]):
        print(f"  {i}: {p}")
    print(f"Number of primitives: {len(result['primitives'])}")
    print("Forbidden primitives and column indices:")
    for p in result["forbidden"]:
        print(f"  - {p}: {result['mapping'][p]}")

    print(f"max(H[:, forbidden_cols]) before mask: {result['max_forbidden_pre']}")
    print(f"max(H[:, forbidden_cols]) after mask: {result['max_forbidden']}")
    print(f"row sum error before: {result['row_err_pre']}")
    print(f"row sum error after: {result['row_err_post']}")
    print(f"col sum error after (vs target): {result['col_err_post_vs_target']}")


if __name__ == "__main__":
    main()
