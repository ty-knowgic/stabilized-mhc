import os
import sys

import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DEMO_DIR = os.path.join(ROOT_DIR, "demo")
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from constrained_routing_demo import build_masked_routing  # type: ignore


def test_forbidden_primitives_zero():
    primitives_path = os.path.abspath(
        os.path.join(ROOT_DIR, "..", "synthetic-danger", "demo_outputs", "primitives.yaml")
    )
    result = build_masked_routing(primitives_path, seed=1234, iters=25)
    H = result["H_post"]
    forbidden_cols = result["forbidden_cols"]
    target_col = result["target_col"]

    # Forbidden primitives are hard zero
    if forbidden_cols:
        max_forbidden = H[:, :, forbidden_cols].max().item()
        assert max_forbidden == 0.0

    # Non-negativity
    assert H.min().item() >= 0.0

    # Row sums are 1.0
    row_sums = H.sum(dim=-1)
    row_err = (row_sums - 1.0).abs().max().item()
    assert row_err <= 1e-6

    # Column sums match target (allowed=N/K, forbidden=0)
    col_sums = H.sum(dim=-2)
    col_err = (col_sums - target_col.squeeze(0)).abs().max().item()
    assert col_err <= 1e-6
