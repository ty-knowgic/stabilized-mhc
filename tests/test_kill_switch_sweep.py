import os
import sys
import tempfile
from typing import List, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DEMO_DIR = os.path.join(ROOT_DIR, "demo")
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from constrained_routing_demo import build_masked_routing  # type: ignore


def _write_primitives_yaml(
    allowed: List[str], forbidden: List[str], all_primitives: List[str]
) -> str:
    lines = []
    lines.append("all_primitives:")
    for p in all_primitives:
        lines.append(f"- {p}")
    lines.append("allowed_primitives:")
    for p in allowed:
        lines.append(f"- {p}")
    lines.append("forbidden_primitives:")
    for p in forbidden:
        lines.append(f"- {p}")
    content = "\n".join(lines) + "\n"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


def _run_case(n: int, forbidden_count: int, seed: int) -> None:
    primitives = [f"p{i}" for i in range(n)]
    forbidden = primitives[-forbidden_count:] if forbidden_count > 0 else []
    allowed = [p for p in primitives if p not in forbidden]
    yaml_path = _write_primitives_yaml(allowed, forbidden, allowed + forbidden)
    try:
        result = build_masked_routing(yaml_path, seed=seed, iters=25)
    finally:
        os.unlink(yaml_path)

    H = result["H_post"]
    forbidden_cols = result["forbidden_cols"]

    # Hard-zero guarantee
    if forbidden_cols:
        assert result["max_forbidden_pre"] > 0.0
        assert result["max_forbidden"] == 0.0

    # Non-negativity
    assert H.min().item() >= 0.0

    # Row-sum and target column-sum constraints
    assert result["row_err_post"] <= 1e-6
    assert result["col_err_post_vs_target"] <= 1e-6

    # Determinism (repeat with same seed)
    yaml_path = _write_primitives_yaml(allowed, forbidden, allowed + forbidden)
    try:
        result2 = build_masked_routing(yaml_path, seed=seed, iters=25)
    finally:
        os.unlink(yaml_path)
    assert abs(result["max_forbidden_pre"] - result2["max_forbidden_pre"]) <= 1e-12
    assert abs(result["row_err_post"] - result2["row_err_post"]) <= 1e-12
    assert abs(result["col_err_post_vs_target"] - result2["col_err_post_vs_target"]) <= 1e-12


def test_kill_switch_sweep():
    cases: List[Tuple[int, float]] = [
        (3, 0.0),
        (5, 0.2),
        (7, 0.5),
        (15, 0.2),
    ]
    seeds = [11, 23, 37]
    for n, ratio in cases:
        forbidden_count = int(n * ratio)
        forbidden_count = min(forbidden_count, n - 1)
        for seed in seeds:
            _run_case(n, forbidden_count, seed)


def test_kill_switch_edge_cases():
    # forbidden empty -> still valid
    _run_case(n=5, forbidden_count=0, seed=17)

    # allowed empty -> must raise
    primitives = [f"p{i}" for i in range(4)]
    allowed: List[str] = []
    forbidden = primitives[:]
    yaml_path = _write_primitives_yaml(allowed, forbidden, allowed + forbidden)
    try:
        raised = False
        try:
            build_masked_routing(yaml_path, seed=19, iters=25)
        except ValueError:
            raised = True
        assert raised
    finally:
        os.unlink(yaml_path)
