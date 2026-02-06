# Algebraic mHC Reachability Kill Switch Demo

## How to Run the Demo

```bash
python demo/constrained_routing_demo.py --primitives /path/to/primitives.yaml
```

## How to Run the Test

```bash
pytest tests/test_forbidden_primitives_zero.py
```

## What This Proves (Factually)

- Forbidden primitives are structurally unreachable (their routing columns are hard zero).
- After masking, the routing matrix is enforced to be row-stochastic (rows sum to 1).
- Column sums match a target distribution: forbidden columns → 0, allowed columns → N/K.
- Therefore the post-mask matrix is not doubly-stochastic.

## What This Does NOT Prove

- Task optimality
- Learning performance
- Full system safety
