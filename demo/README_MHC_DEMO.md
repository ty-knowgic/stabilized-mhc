# Algebraic mHC Reachability Kill Switch Demo

## How to Run the Demo

```bash
python demo/constrained_routing_demo.py
```

## How to Run the Test

```bash
pytest tests/test_forbidden_primitives_zero.py
```

## What This Proves (Factually)

- Forbidden primitives are structurally unreachable (their routing columns are hard zero).
- The routing matrix remains non-negative and satisfies the documented normalization constraints.

## What This Does NOT Prove

- Task optimality
- Learning performance
- Full system safety
