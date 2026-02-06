import glob
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DEMO_DIR = os.path.join(ROOT_DIR, "demo")
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from constrained_routing_demo import build_masked_routing  # type: ignore


def _resolve_primitives_path() -> str:
    env_path = os.environ.get("PRIMITIVES_PATH")
    if env_path:
        return os.path.abspath(env_path)
    fixture = os.path.join(ROOT_DIR, "demo", "primitives_fixture.yaml")
    if os.path.exists(fixture):
        return os.path.abspath(fixture)
    parent = os.path.abspath(os.path.join(ROOT_DIR, ".."))
    candidates = glob.glob(os.path.join(parent, "*", "demo_outputs", "primitives.yaml"))
    if not candidates:
        raise FileNotFoundError(
            "Could not find primitives.yaml. Set PRIMITIVES_PATH env var."
        )
    return os.path.abspath(sorted(candidates)[0])


def test_forbidden_primitives_zero():
    primitives_path = _resolve_primitives_path()
    result = build_masked_routing(primitives_path, seed=1234, iters=25)
    H = result["H_post"]
    forbidden_cols = result["forbidden_cols"]
    target_col = result["target_col"]

    # Forbidden primitives are hard zero
    if forbidden_cols:
        max_forbidden = H[:, :, forbidden_cols].max().item()
        assert max_forbidden == 0.0
        assert result["max_forbidden_pre"] > 0.0

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
